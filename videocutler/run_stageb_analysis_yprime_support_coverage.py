from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.algorithms.reservoir_v1 import (  # noqa: E402
    _compute_t_dis,
    _load_reservoir_checkpoint,
)
from videocutler.ext_stageb_ovvis.analysis.extra_attribution_probe import (  # noqa: E402
    ExtraAttributionProbeConfig,
    _apply_stage_candidate_overrides,
    _default_checkpoint_path,
    _load_stage_responsibility_candidate_overrides,
    _materialize_valid_samples,
    _prepare_probe_examples,
    _score_batches,
)
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _all_gt_split_label  # noqa: E402
from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import load_gt_sidecar_lookup  # noqa: E402
from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import (  # noqa: E402
    load_lvvis_base_and_novel_raw_ids,
)

Record = Dict[str, Any]


@dataclass(frozen=True)
class Config:
    run_root: Path
    runtime_output_root: Path
    dataset_name: str
    trajectory_source_branch: str
    stage: str
    device: str
    batch_size: int
    output_dir: Optional[Path]
    sidecar_root: Optional[Path]
    smoke: bool
    smoke_max_trajectories: int
    subset_fraction: Optional[float]
    show_progress: bool
    hub_raw_ids: Tuple[int, ...]
    min_class_count: int
    top_examples: int
    write_rows: bool


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected bool, got {value!r}")


def _parse_ints(value: str) -> Tuple[int, ...]:
    out: List[int] = []
    for p in str(value).replace(";", ",").split(","):
        p = p.strip()
        if p:
            out.append(int(p))
    return tuple(out)


def _load_json(path: Path) -> Optional[Record]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[Record]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        v = float(value)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _unique_ints(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
    elif isinstance(value, Mapping):
        parts = value.keys()
    elif isinstance(value, Iterable):
        parts = value
    else:
        parts = [value]
    out: List[int] = []
    seen = set()
    for x in parts:
        ix = _safe_int(x)
        if ix is None or ix in seen:
            continue
        seen.add(ix)
        out.append(int(ix))
    return out


def _default_output_dir(run_root: Path, dataset_name: str, stage: str) -> Path:
    return run_root / "analysis" / "yprime_support_coverage" / dataset_name / stage


def _sidecar_root(config: Config) -> Path:
    return Path(config.sidecar_root).expanduser().resolve() if config.sidecar_root is not None else Path(config.run_root).expanduser().resolve()


def _read_responsibility_rows(run_root: Path, stage: str) -> Tuple[Dict[str, Record], Dict[str, Any]]:
    path = run_root / "train" / stage / "responsibility_records.jsonl"
    by_tid: Dict[str, Record] = {}
    total = 0
    for row in _iter_jsonl(path):
        total += 1
        tid = str(row.get("trajectory_id", "")).strip()
        if tid:
            by_tid[tid] = row
    return by_tid, {"path": str(path), "exists": path.is_file(), "record_count": int(total), "by_tid_count": int(len(by_tid))}


def _r_value(resp_row: Mapping[str, Any], raw_id: int) -> Optional[float]:
    # Prefer final responsibility distribution if present. Fall back to common alternatives.
    for key in ("r_final", "responsibility_final", "responsibilities", "r", "R_final"):
        obj = resp_row.get(key)
        if isinstance(obj, Mapping):
            for rk in (str(raw_id), raw_id):
                if rk in obj:
                    return _safe_float(obj.get(rk), default=None)
    # Some rows may store flat class-to-mass pairs.
    vals = resp_row.get("candidate_masses") or resp_row.get("mass_by_raw_id")
    if isinstance(vals, Mapping):
        for rk in (str(raw_id), raw_id):
            if rk in vals:
                return _safe_float(vals.get(rk), default=None)
    return None


def _class_name_records(runtime_output_root: Path, dataset_name: str) -> Dict[int, str]:
    names: Dict[int, str] = {}
    split = "train" if dataset_name == "lvvis_train_base" else "val"
    roots = [
        runtime_output_root / "videocutler" / "datasets" / "LV-VIS" / "annotations",
        runtime_output_root / "datasets" / "LV-VIS" / "annotations",
        Path("/home/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations"),
        Path("/mnt/sda/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations"),
    ]
    names_to_try = [f"{split}_instances.json", f"instances_{split}.json", "train_instances.json", "val_instances.json"]
    for root in roots:
        for name in names_to_try:
            p = root / name
            obj = _load_json(p)
            if not isinstance(obj, Mapping):
                continue
            cats = obj.get("categories")
            if isinstance(cats, list):
                for cat in cats:
                    if not isinstance(cat, Mapping):
                        continue
                    rid = _safe_int(cat.get("id", cat.get("raw_id", cat.get("category_id"))))
                    cname = cat.get("name") or cat.get("class_name") or cat.get("category_name")
                    if rid is not None and cname is not None:
                        names.setdefault(int(rid), str(cname))
    return names


def _mean(xs: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else None


def _median(xs: Sequence[float]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return float(median(vals)) if vals else None


def _rate(n: int, d: int) -> Optional[float]:
    return float(n / d) if d else None


def _gini(xs: Sequence[float]) -> Optional[float]:
    vals = np.asarray([float(x) for x in xs if x is not None and math.isfinite(float(x)) and float(x) >= 0.0], dtype=np.float64)
    if vals.size == 0:
        return None
    if float(vals.sum()) <= 0.0:
        return 0.0
    vals = np.sort(vals)
    n = vals.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * vals) / (n * np.sum(vals))) - (n + 1) / n)


def _rank_desc(scores: np.ndarray, target_index: int, mask: Optional[np.ndarray] = None) -> Optional[int]:
    if target_index < 0 or target_index >= scores.shape[0]:
        return None
    s = np.asarray(scores, dtype=np.float64).copy()
    if mask is not None:
        s[~mask] = -np.inf
    if not math.isfinite(float(s[target_index])):
        return None
    order = np.argsort(-s, kind="stable")
    finite = np.isfinite(s[order])
    order = order[finite]
    pos = np.where(order == int(target_index))[0]
    if pos.size == 0:
        return None
    return int(pos[0]) + 1


def run_audit(config: Config) -> Dict[str, Any]:
    run_root = Path(config.run_root).expanduser().resolve()
    runtime_output_root = Path(config.runtime_output_root).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve() if config.output_dir else _default_output_dir(run_root, config.dataset_name, config.stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_config = ExtraAttributionProbeConfig(
        run_root=run_root,
        runtime_output_root=runtime_output_root,
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
        device=str(config.device),
        smoke=bool(config.smoke),
        smoke_max_trajectories=int(config.smoke_max_trajectories),
        subset_fraction=None if config.subset_fraction is None else float(config.subset_fraction),
        stage_scope=(str(config.stage),),
        batch_size=max(1, int(config.batch_size)),
        output_dir=output_dir,
        sidecar_root=_sidecar_root(config),
        show_progress=bool(config.show_progress),
    )
    materialized = _materialize_valid_samples(proxy_config)
    valid_samples = list(materialized.get("valid_samples", []))
    prepared = _prepare_probe_examples(
        valid_samples,
        output_root=runtime_output_root,
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
    )
    examples = list(prepared.get("examples", []))
    if not examples:
        raise RuntimeError("no examples materialized; cannot audit Yprime support coverage")

    stage_overrides, override_meta = _load_stage_responsibility_candidate_overrides(run_root=run_root, stage_id=str(config.stage))
    examples, scope_meta = _apply_stage_candidate_overrides(examples, stage_overrides, stage_id=str(config.stage))

    sidecar_lookup = load_gt_sidecar_lookup(_sidecar_root(config), dataset_name=str(config.dataset_name), trajectory_source_branch=str(config.trajectory_source_branch))
    base_vocab_ids, _novel_ids = load_lvvis_base_and_novel_raw_ids()
    base_vocab_set = {int(x) for x in base_vocab_ids}
    resp_by_tid, resp_meta = _read_responsibility_rows(run_root, str(config.stage))
    names = _class_name_records(runtime_output_root, str(config.dataset_name))

    text_vocab_ids, _text_records, text_vocab_matrix = load_text_vocab(runtime_output_root)
    vocab_ids = [int(x) for x in text_vocab_ids]
    raw_to_index = {int(r): i for i, r in enumerate(vocab_ids)}
    checkpoint_path = _default_checkpoint_path(run_root, str(config.stage))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found for stage={config.stage}: {checkpoint_path}")
    device = torch.device(str(config.device))
    projector, theta_t, unknown_prototype, _checkpoint_payload = _load_reservoir_checkpoint(checkpoint_path, device=device)
    projector.eval()
    temperature = _compute_t_dis(theta_t).detach()
    logits_vocab, _logits_unknown = _score_batches(
        examples=examples,
        projector=projector,
        text_vocab_matrix=np.asarray(text_vocab_matrix, dtype=np.float32),
        unknown_prototype=unknown_prototype,
        temperature=temperature,
        device=device,
        batch_size=max(1, int(config.batch_size)),
        show_progress=bool(config.show_progress),
        stage_id=str(config.stage),
    )

    # Per-row GT / yprime / split materialization.
    row_meta: List[Record] = []
    by_clip_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        tid = str(ex.get("trajectory_id", "")).strip()
        clip_id = _safe_int(ex.get("clip_id"), -1)
        sidecar = sidecar_lookup.get(tid, {}) if tid else {}
        gt = None
        if isinstance(sidecar, Mapping):
            # Keep permissive and aligned with prior audit scripts.
            for key in ("matched_gt_raw_id_canonical", "gt_raw_id_canonical", "matched_gt_raw_id", "gt_raw_id", "category_id"):
                gt = _safe_int(sidecar.get(key), None)
                if gt is not None:
                    break
            if gt is None:
                nested = sidecar.get("matched_gt") or sidecar.get("gt") or sidecar.get("match")
                if isinstance(nested, Mapping):
                    for key in ("raw_id", "category_id", "id", "gt_raw_id"):
                        gt = _safe_int(nested.get(key), None)
                        if gt is not None:
                            break
        observed = _unique_ints(ex.get("observed_raw_ids")) or _unique_ints(ex.get("candidate_ids_known"))
        split = None
        if gt is not None:
            try:
                split = _all_gt_split_label(
                    dataset_name=str(config.dataset_name),
                    gt_raw_id=int(gt),
                    observed_raw_ids=observed,
                    base_vocab_ids=base_vocab_set,
                )
            except Exception:
                split = None
        audit_usable = bool(gt is not None)
        if isinstance(sidecar, Mapping) and "audit_usable" in sidecar:
            audit_usable = bool(sidecar.get("audit_usable")) and gt is not None
        m = {
            "row_index": int(idx),
            "trajectory_id": tid,
            "clip_id": int(clip_id if clip_id is not None else -1),
            "video_id": _safe_int(ex.get("video_id"), None),
            "gt_raw_id": int(gt) if gt is not None else None,
            "gt_name": names.get(int(gt), str(gt)) if gt is not None else None,
            "split": split,
            "auditable_gt": bool(audit_usable),
            "observed_raw_ids": observed,
        }
        row_meta.append(m)
        if clip_id is not None:
            by_clip_indices[int(clip_id)].append(int(idx))

    hub_set = {int(x) for x in config.hub_raw_ids}
    yprime_pair_rows: List[Record] = []
    class_stats: Dict[int, Counter] = defaultdict(Counter)
    class_support_counts: Dict[int, List[int]] = defaultdict(list)
    example_rows: List[Record] = []
    failure_counts: Counter = Counter()
    clip_all_supported_flags: List[bool] = []
    clip_any_yprime_flags: List[bool] = []

    # Compute per (clip, y) support and scoring/assignment quality.
    for clip_id, idxs in sorted(by_clip_indices.items()):
        yprime: List[int] = []
        seen = set()
        for idx in idxs:
            for y in row_meta[idx].get("observed_raw_ids", []):
                if y not in seen:
                    seen.add(int(y)); yprime.append(int(y))
        if not yprime:
            continue
        clip_any_yprime_flags.append(True)
        all_supported = True
        auditable_idxs = [idx for idx in idxs if bool(row_meta[idx].get("auditable_gt")) and row_meta[idx].get("gt_raw_id") is not None]
        gt_set = {int(row_meta[idx]["gt_raw_id"]) for idx in auditable_idxs if row_meta[idx].get("gt_raw_id") is not None}
        for y in yprime:
            support_idxs = [idx for idx in auditable_idxs if int(row_meta[idx].get("gt_raw_id")) == int(y)]
            support_count = len(support_idxs)
            has_support = support_count > 0
            if not has_support:
                all_supported = False
            class_stats[int(y)]["clip_yprime_count"] += 1
            class_support_counts[int(y)].append(int(support_count))
            if has_support:
                class_stats[int(y)]["support_count_positive"] += 1
            # Score quality on true-support trajectories.
            y_idx = raw_to_index.get(int(y))
            best_support_score = None
            best_yprime_rank = None
            best_vocab_rank = None
            best_margin_vs_hub = None
            support_exists_but_person_higher = False
            support_exists_but_score_bad = False
            if has_support and y_idx is not None:
                best_support_score = -float("inf")
                best_yprime_rank = None
                best_vocab_rank = None
                best_margin_vs_hub = None
                yprime_indices = [raw_to_index[z] for z in yprime if z in raw_to_index]
                yprime_mask = np.zeros((len(vocab_ids),), dtype=bool)
                if yprime_indices:
                    yprime_mask[np.asarray(yprime_indices, dtype=np.int64)] = True
                for idx in support_idxs:
                    scores = np.asarray(logits_vocab[idx], dtype=np.float64)
                    y_score = float(scores[int(y_idx)])
                    best_support_score = max(float(best_support_score), y_score)
                    r1 = _rank_desc(scores, int(y_idx), yprime_mask if yprime_indices else None)
                    if r1 is not None:
                        best_yprime_rank = int(r1) if best_yprime_rank is None else min(int(best_yprime_rank), int(r1))
                    r2 = _rank_desc(scores, int(y_idx), None)
                    if r2 is not None:
                        best_vocab_rank = int(r2) if best_vocab_rank is None else min(int(best_vocab_rank), int(r2))
                    hub_scores = []
                    for h in hub_set:
                        hi = raw_to_index.get(int(h))
                        if hi is not None:
                            hub_scores.append(float(scores[int(hi)]))
                    if hub_scores:
                        margin = float(y_score - max(hub_scores))
                        best_margin_vs_hub = margin if best_margin_vs_hub is None else max(float(best_margin_vs_hub), margin)
                support_exists_but_person_higher = bool(best_margin_vs_hub is not None and float(best_margin_vs_hub) < 0.0)
                # Conservative score-bad proxy: y's best full-vocab rank is outside top-20.
                support_exists_but_score_bad = bool(best_vocab_rank is None or int(best_vocab_rank) > 20)
            # Responsibility / assignment proxy.
            total_mass_y = 0.0
            true_support_mass_y = 0.0
            best_mass = -1.0
            best_mass_tid = None
            best_mass_gt = None
            resp_available_for_y = False
            for idx in auditable_idxs:
                tid = str(row_meta[idx].get("trajectory_id", ""))
                rv = _r_value(resp_by_tid.get(tid, {}), int(y)) if tid else None
                if rv is None:
                    continue
                resp_available_for_y = True
                val = float(rv)
                total_mass_y += val
                if int(row_meta[idx].get("gt_raw_id")) == int(y):
                    true_support_mass_y += val
                if val > best_mass:
                    best_mass = val
                    best_mass_tid = tid
                    best_mass_gt = _safe_int(row_meta[idx].get("gt_raw_id"), None)
            true_support_mass_ratio = None
            if resp_available_for_y and total_mass_y > 0.0:
                true_support_mass_ratio = float(true_support_mass_y / total_mass_y)
            true_support_top1 = bool(resp_available_for_y and has_support and best_mass_gt == int(y))
            hub_hijack = bool(resp_available_for_y and best_mass_gt in hub_set and int(y) not in hub_set and not true_support_top1)
            if not has_support:
                bucket = "stageA_or_sidecar_missing_trajectory_support"
            elif support_exists_but_person_higher:
                bucket = "support_exists_person_higher"
            elif support_exists_but_score_bad:
                bucket = "support_exists_score_bad"
            elif resp_available_for_y and not true_support_top1:
                bucket = "support_exists_assignment_hijacked"
            else:
                bucket = "supported_and_scored_or_assigned"
            failure_counts[bucket] += 1
            row = {
                "clip_id": int(clip_id),
                "yprime_raw_id": int(y),
                "yprime_name": names.get(int(y), str(y)),
                "clip_trajectory_count": int(len(idxs)),
                "auditable_trajectory_count": int(len(auditable_idxs)),
                "auditable_gt_class_count": int(len(gt_set)),
                "support_count": int(support_count),
                "has_trajectory_support": bool(has_support),
                "best_support_score_to_y": best_support_score if best_support_score is not None and math.isfinite(float(best_support_score)) else None,
                "best_rank_of_y_among_yprime": best_yprime_rank,
                "best_rank_of_y_among_vocab": best_vocab_rank,
                "best_margin_vs_hub": best_margin_vs_hub,
                "support_exists_but_text_score_bad": bool(support_exists_but_score_bad),
                "support_exists_but_person_higher": bool(support_exists_but_person_higher),
                "responsibility_available_for_y": bool(resp_available_for_y),
                "responsibility_total_mass_y": float(total_mass_y) if resp_available_for_y else None,
                "responsibility_true_support_mass_y": float(true_support_mass_y) if resp_available_for_y else None,
                "responsibility_true_support_mass_ratio": true_support_mass_ratio,
                "responsibility_true_support_top1": bool(true_support_top1),
                "responsibility_best_mass_trajectory_id": best_mass_tid,
                "responsibility_best_mass_gt_raw_id": best_mass_gt,
                "responsibility_hub_hijack": bool(hub_hijack),
                "bucket": bucket,
            }
            yprime_pair_rows.append(row)
            if len(example_rows) < max(1, int(config.top_examples)) and bucket != "supported_and_scored_or_assigned":
                example_rows.append(row)
        clip_all_supported_flags.append(bool(all_supported))

    pair_count = len(yprime_pair_rows)
    support_rows = [r for r in yprime_pair_rows if bool(r.get("has_trajectory_support"))]
    resp_rows = [r for r in yprime_pair_rows if bool(r.get("responsibility_available_for_y"))]

    class_rows: List[Record] = []
    for y, counter in sorted(class_stats.items()):
        counts = class_support_counts.get(int(y), [])
        if len(counts) < int(config.min_class_count):
            continue
        denom = int(counter.get("clip_yprime_count", 0))
        class_rows.append({
            "raw_id": int(y),
            "name": names.get(int(y), str(y)),
            "clip_yprime_count": denom,
            "supported_pair_count": int(counter.get("support_count_positive", 0)),
            "support_rate": _rate(int(counter.get("support_count_positive", 0)), denom),
            "mean_support_count": _mean([float(x) for x in counts]),
            "median_support_count": _median([float(x) for x in counts]),
            "zero_support_count": int(sum(1 for x in counts if int(x) <= 0)),
        })

    class_csv = output_dir / "class_support_summary.csv"
    with class_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_id", "name", "clip_yprime_count", "supported_pair_count", "support_rate", "mean_support_count", "median_support_count", "zero_support_count"])
        writer.writeheader(); writer.writerows(class_rows)

    failure_csv = output_dir / "failure_bucket_summary.csv"
    with failure_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", "count", "rate"])
        writer.writeheader()
        for bucket, count in failure_counts.most_common():
            writer.writerow({"bucket": bucket, "count": int(count), "rate": _rate(int(count), pair_count)})

    if config.write_rows:
        rows_path = output_dir / "clip_yprime_support_rows.jsonl"
        with rows_path.open("w", encoding="utf-8") as f:
            for r in yprime_pair_rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")
    else:
        rows_path = None

    examples_path = output_dir / "support_examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for r in example_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n")

    support_counts = [int(r.get("support_count", 0)) for r in yprime_pair_rows]
    true_mass_ratios = [float(r["responsibility_true_support_mass_ratio"]) for r in resp_rows if r.get("responsibility_true_support_mass_ratio") is not None]
    summary: Record = {
        "status": "PASS",
        "definition": "For each clip v and each weak observed label y in Y'(v), verify whether at least one auditable trajectory in the same clip has matched GT raw id y; then audit scoring and responsibility credit if available.",
        "dataset_name": str(config.dataset_name),
        "stage": str(config.stage),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "materialization": {
            "valid_sample_count": int(len(valid_samples)),
            "example_count": int(len(examples)),
            "stage_candidate_override": dict(override_meta),
            "candidate_scope_meta": dict(scope_meta),
            "responsibility_records": dict(resp_meta),
        },
        "clip_count": int(len(by_clip_indices)),
        "clip_with_yprime_count": int(len(clip_all_supported_flags)),
        "clip_yprime_pair_count": int(pair_count),
        "auditable_trajectory_count": int(sum(1 for m in row_meta if bool(m.get("auditable_gt")))),
        "unique_yprime_class_count": int(len(class_stats)),
        "class_summary_min_class_count": int(config.min_class_count),
        "class_summary_class_count": int(len(class_rows)),
        "yprime_trajectory_support_rate": _rate(len(support_rows), pair_count),
        "yprime_no_trajectory_support_rate": _rate(pair_count - len(support_rows), pair_count),
        "clip_all_yprime_supported_rate": _rate(sum(1 for x in clip_all_supported_flags if bool(x)), len(clip_all_supported_flags)),
        "mean_support_count_per_yprime": _mean([float(x) for x in support_counts]),
        "median_support_count_per_yprime": _median([float(x) for x in support_counts]),
        "min_support_count_per_yprime": int(min(support_counts)) if support_counts else None,
        "support_exists_but_text_score_bad_rate": _rate(sum(1 for r in support_rows if bool(r.get("support_exists_but_text_score_bad"))), len(support_rows)),
        "support_exists_but_person_higher_rate": _rate(sum(1 for r in support_rows if bool(r.get("support_exists_but_person_higher"))), len(support_rows)),
        "supported_best_vocab_rank_mean": _mean([float(r["best_rank_of_y_among_vocab"]) for r in support_rows if r.get("best_rank_of_y_among_vocab") is not None]),
        "supported_best_vocab_rank_median": _median([float(r["best_rank_of_y_among_vocab"]) for r in support_rows if r.get("best_rank_of_y_among_vocab") is not None]),
        "supported_best_yprime_rank_mean": _mean([float(r["best_rank_of_y_among_yprime"]) for r in support_rows if r.get("best_rank_of_y_among_yprime") is not None]),
        "supported_best_margin_vs_hub_mean": _mean([float(r["best_margin_vs_hub"]) for r in support_rows if r.get("best_margin_vs_hub") is not None]),
        "responsibility_available_pair_rate": _rate(len(resp_rows), pair_count),
        "sinkhorn_yprime_true_support_mass_mean": _mean(true_mass_ratios),
        "sinkhorn_yprime_true_support_mass_median": _median(true_mass_ratios),
        "sinkhorn_yprime_true_support_top1_rate": _rate(sum(1 for r in resp_rows if bool(r.get("responsibility_true_support_top1"))), len(resp_rows)),
        "sinkhorn_yprime_hub_hijack_rate": _rate(sum(1 for r in resp_rows if bool(r.get("responsibility_hub_hijack"))), len(resp_rows)),
        "failure_bucket_counts": dict(failure_counts),
        "failure_bucket_rates": {k: _rate(int(v), pair_count) for k, v in failure_counts.items()},
        "outputs": {
            "summary_json": str(output_dir / "summary.json"),
            "class_support_summary_csv": str(class_csv),
            "failure_bucket_summary_csv": str(failure_csv),
            "support_examples_jsonl": str(examples_path),
            "clip_yprime_support_rows_jsonl": str(rows_path) if rows_path is not None else None,
        },
        "interpretation": {},
    }
    # Compact verdict.
    all_supported = summary.get("clip_all_yprime_supported_rate")
    pair_supported = summary.get("yprime_trajectory_support_rate")
    true_mass = summary.get("sinkhorn_yprime_true_support_mass_mean")
    if pair_supported is not None and pair_supported < 0.80:
        verdict = "yprime_support_assumption_weak_or_false"
    elif all_supported is not None and all_supported < 0.80:
        verdict = "many_clips_have_at_least_one_unsupported_yprime"
    elif true_mass is not None and true_mass < 0.50:
        verdict = "data_support_exists_but_assignment_credit_hijacked"
    else:
        verdict = "yprime_support_assumption_largely_supported"
    summary["interpretation"] = {
        "verdict": verdict,
        "primary_question": "Does every y in Y'(v) have at least one auditable trajectory with matched GT y in clip v?",
        "use": "If support rates are high but responsibility true-support mass is low, the failure is credit assignment hijack rather than missing data support.",
    }
    _dump_json(output_dir / "summary.json", summary)
    takeover = output_dir / "YPRIME_SUPPORT_COVERAGE_TAKEOVER.md"
    takeover.write_text(
        "# Y′ Support Coverage Audit\n\n"
        f"- status: {summary['status']}\n"
        f"- verdict: {verdict}\n"
        f"- dataset: {config.dataset_name}\n"
        f"- stage: {config.stage}\n"
        f"- clip_yprime_pair_count: {pair_count}\n"
        f"- yprime_trajectory_support_rate: {summary['yprime_trajectory_support_rate']}\n"
        f"- clip_all_yprime_supported_rate: {summary['clip_all_yprime_supported_rate']}\n"
        f"- support_exists_but_person_higher_rate: {summary['support_exists_but_person_higher_rate']}\n"
        f"- sinkhorn_yprime_true_support_mass_mean: {summary['sinkhorn_yprime_true_support_mass_mean']}\n"
        f"- sinkhorn_yprime_hub_hijack_rate: {summary['sinkhorn_yprime_hub_hijack_rate']}\n"
        "\n## Outputs\n"
        f"- summary: `{output_dir / 'summary.json'}`\n"
        f"- class summary: `{class_csv}`\n"
        f"- failure buckets: `{failure_csv}`\n"
        f"- examples: `{examples_path}`\n"
        + (f"- rows: `{rows_path}`\n" if rows_path is not None else "- rows: disabled\n"),
        encoding="utf-8",
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit whether every weak observed Y′ class has at least one auditable GT-supported trajectory in its clip.")
    p.add_argument("--run_root", type=Path, required=True)
    p.add_argument("--runtime_output_root", type=Path, default=Path("/mnt/sda/zyy/code/wsovvis"))
    p.add_argument("--dataset_name", type=str, default="lvvis_train_base")
    p.add_argument("--trajectory_source_branch", type=str, default="mainline")
    p.add_argument("--stage", type=str, default="softem_aug")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--sidecar_root", type=Path, default=None)
    p.add_argument("--smoke", type=_parse_bool, default=False)
    p.add_argument("--smoke_max_trajectories", type=int, default=512)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--show_progress", type=_parse_bool, default=True)
    p.add_argument("--hub_raw_ids", type=_parse_ints, default=(773,))
    p.add_argument("--min_class_count", type=int, default=3)
    p.add_argument("--top_examples", type=int, default=128)
    p.add_argument("--write_rows", type=_parse_bool, default=True)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = Config(
        run_root=args.run_root,
        runtime_output_root=args.runtime_output_root,
        dataset_name=args.dataset_name,
        trajectory_source_branch=args.trajectory_source_branch,
        stage=args.stage,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        sidecar_root=args.sidecar_root,
        smoke=bool(args.smoke),
        smoke_max_trajectories=int(args.smoke_max_trajectories),
        subset_fraction=args.subset_fraction,
        show_progress=bool(args.show_progress),
        hub_raw_ids=tuple(int(x) for x in args.hub_raw_ids),
        min_class_count=int(args.min_class_count),
        top_examples=int(args.top_examples),
        write_rows=bool(args.write_rows),
    )
    summary = run_audit(cfg)
    print(json.dumps({
        "status": summary.get("status"),
        "verdict": summary.get("interpretation", {}).get("verdict"),
        "clip_yprime_pair_count": summary.get("clip_yprime_pair_count"),
        "yprime_trajectory_support_rate": summary.get("yprime_trajectory_support_rate"),
        "clip_all_yprime_supported_rate": summary.get("clip_all_yprime_supported_rate"),
        "output_dir": summary.get("output_dir"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
