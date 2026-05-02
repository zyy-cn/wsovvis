#!/usr/bin/env python3
"""Read-only demand-floor replay for GT-fullY clean nohub.

This analysis does not train, does not modify checkpoints, and does not run mAP.
It replays a small class-level additive score bonus derived from observable
under-assignment statistics and evaluates whether it could improve GT attribution
rank/top1 without hurting the existing nohub behavior.

The bonus is post-hoc and diagnostic only:

    score'(i, c) = score(i, c) + bonus_c

where bonus_c is computed from under_assigned_class_table.csv. GT labels are used
only for post-hoc evaluation, never for bonus construction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _bootstrap_repo_root_for_direct_cli() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_bootstrap_repo_root_for_direct_cli()

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import load_text_vocab  # noqa: E402
from videocutler.ext_stageb_ovvis.algorithms.prealign import _prepare_examples as _prepare_prealign_examples  # noqa: E402
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import Phase1MaterializationConfig, materialize_phase1_training_samples  # noqa: E402
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig  # noqa: E402


REPO_ASSET_LINK_NAMES = (
    "exports", "exports_gt", "carrier_bank", "carrier_bank_gt", "frame_bank",
    "text_bank", "gt_sidecar_bank", "weak_labels", "weights", "dataset", "eval",
)

METRICS_DEFAULT = "support_mass_gap,support_mass_ratio,hybrid_under_assignment_score,low_mass_per_support"


# ----------------------------- basic helpers --------------------------------


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(str(x).strip()))
        except Exception:
            return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _truthy(x: Any) -> bool:
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return repr(v)
    return v


def _csv_read(path: Path, *, required: bool = True) -> List[Dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _csv_write(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        seen = set()
        fields2: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fields2.append(str(k))
        fields = fields2
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k, "")) for k in fields})


def _json_write(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_str_list(text: str) -> List[str]:
    return [str(x).strip() for x in str(text).split(",") if str(x).strip()]


def _safe_link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def _bootstrap_asset_links(target_root: Path, asset_root: Path) -> None:
    if not asset_root.is_dir():
        return
    target_root.mkdir(parents=True, exist_ok=True)
    for name in REPO_ASSET_LINK_NAMES:
        src = asset_root / name
        dst = target_root / name
        if src.exists() and not dst.exists() and not dst.is_symlink():
            try:
                _safe_link(src, dst)
            except Exception:
                pass


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ----------------------------- data loading ---------------------------------


def _maybe_list_of_ids(v: Any) -> Optional[List[int]]:
    if isinstance(v, list):
        vals: List[int] = []
        for item in v:
            val = item.get("raw_id", item.get("id", item.get("category_id"))) if isinstance(item, Mapping) else item
            ii = _as_int(val)
            if ii is None:
                return None
            vals.append(int(ii))
        return vals
    return None


def _extract_split_ids(obj: Any, split_name: str) -> List[int]:
    keys = {
        "base": ["base", "base_ids", "base_raw_ids", "base_category_ids", "base_classes", "official_base", "base_raw_id_list", "base_categories"],
        "novel": ["novel", "novel_ids", "novel_raw_ids", "novel_category_ids", "novel_classes", "official_novel", "novel_raw_id_list", "novel_categories"],
    }[split_name]
    found: List[int] = []

    def walk(x: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(x, Mapping):
            for k in keys:
                if k in x:
                    ids = _maybe_list_of_ids(x[k])
                    if ids is not None:
                        found = ids
                        return
            for k, v in x.items():
                if str(k).lower() == split_name:
                    ids = _maybe_list_of_ids(v)
                    if ids is not None:
                        found = ids
                        return
            for v in x.values():
                walk(v)
                if found:
                    return
        elif isinstance(x, list):
            for v in x:
                walk(v)
                if found:
                    return

    walk(obj)
    if not found:
        raise KeyError(f"could not extract {split_name} ids from split json")
    return sorted({int(x) for x in found})


def _load_base_ids(split_json: Path) -> Set[int]:
    with split_json.open("r", encoding="utf-8") as f:
        return set(_extract_split_ids(json.load(f), "base"))


def _load_clip_y_base(annotation_json: Path, base_ids: Set[int]) -> Dict[int, Set[int]]:
    with annotation_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[int, Set[int]] = {}
    for ann in obj.get("annotations", []):
        if not isinstance(ann, Mapping):
            continue
        clip = _as_int(ann.get("video_id", ann.get("clip_id", ann.get("image_id"))))
        cat = _as_int(ann.get("category_id", ann.get("raw_id", ann.get("raw_category_id"))))
        if clip is not None and cat is not None and int(cat) in base_ids:
            out.setdefault(int(clip), set()).add(int(cat))
    return out


def _extract_gt_raw_id(sample: Mapping[str, Any]) -> Optional[int]:
    candidate_roots: List[Any] = [sample]
    for k in ("trajectory_record", "carrier_record", "gt_record", "annotation"):
        if isinstance(sample.get(k), Mapping):
            candidate_roots.append(sample[k])
    keys = (
        "matched_gt_raw_id_canonical", "matched_gt_raw_id", "best_gt_raw_id",
        "gt_raw_id", "raw_id", "category_id", "raw_category_id", "class_raw_id",
    )
    for root in candidate_roots:
        if not isinstance(root, Mapping):
            continue
        for k in keys:
            ii = _as_int(root.get(k))
            if ii is not None:
                return int(ii)
    return None


def _load_identity_binding(path: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", errors="replace") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            tid = str(row.get("trajectory_id", row.get("gt_trajectory_id", row.get("join_key", ""))))
            rid = _extract_gt_raw_id(row)
            if tid and rid is not None:
                out[tid] = int(rid)
    return out


def _load_examples_with_gt(args: argparse.Namespace, output_root_for_assets: Path) -> Tuple[List[Dict[str, Any]], Dict[int, Set[int]], Set[int], Dict[str, Any]]:
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    _bootstrap_asset_links(repo_root, asset_root)
    _bootstrap_asset_links(output_root_for_assets, asset_root)
    base_ids = _load_base_ids(Path(args.split_json))
    clip_y_base = _load_clip_y_base(Path(args.annotation_json), base_ids)

    with _pushd(repo_root):
        materialized = materialize_phase1_training_samples(
            repo_root,
            Phase1MaterializationConfig(
                dataset_name=str(args.dataset_name),
                trajectory_source_branch="gt_upper_bound",
                smoke=bool(args.smoke),
                smoke_max_trajectories=int(args.smoke_max_trajectories),
                subset_fraction=args.subset_fraction,
                subset_seed=int(args.seed),
            ),
        )
    raw_samples = materialized.get("valid_samples") or materialized.get("samples") or []
    id_binding = _load_identity_binding(Path(args.asset_root) / "carrier_bank_gt" / str(args.dataset_name) / "gt_carrier_identity_binding.jsonl")
    samples: List[Dict[str, Any]] = []
    gt_by_tid: Dict[str, int] = {}
    counters = Counter()
    for s in raw_samples:
        if not bool(s.get("sample_valid", False)):
            counters["skip_invalid"] += 1
            continue
        clip = _as_int(s.get("clip_id"))
        if clip is None:
            counters["skip_no_clip"] += 1
            continue
        yb = sorted(clip_y_base.get(int(clip), set()))
        if not yb:
            counters["skip_no_y_base"] += 1
            continue
        row = dict(s)
        row["observed_raw_ids"] = yb
        tid = str(row.get("trajectory_id", ""))
        rid = _extract_gt_raw_id(row)
        if rid is None and tid in id_binding:
            rid = int(id_binding[tid])
        if rid is not None:
            gt_by_tid[tid] = int(rid)
        samples.append(row)

    prepared = _prepare_prealign_examples(
        samples,
        output_root=output_root_for_assets,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch="gt_upper_bound",
    )
    examples = list(prepared.get("examples", []))
    for ex in examples:
        tid = str(ex.get("trajectory_id", ""))
        ex["matched_gt_raw_id"] = gt_by_tid.get(tid)
    meta = {
        "materialized_stats": materialized.get("stats", {}),
        "sample_counters": dict(counters),
        "prepare_skipped": dict(prepared.get("skipped_reason_histogram", {})),
        "identity_binding_count": len(id_binding),
    }
    return examples, clip_y_base, base_ids, meta


def _normalize_np(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(arr))
    return arr if denom <= 1e-12 else (arr / denom).astype(np.float32, copy=False)


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[Projector, torch.Tensor, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    cfg_raw = ckpt.get("text_projector_config", {}) if isinstance(ckpt, Mapping) else {}
    cfg = ProjectorConfig(
        input_dim=int(cfg_raw.get("input_dim", 512)),
        hidden_dim=int(cfg_raw.get("hidden_dim", 1024)),
        output_dim=int(cfg_raw.get("output_dim", 768)),
        dropout=float(cfg_raw.get("dropout", 0.0)),
        use_layernorm=bool(cfg_raw.get("use_layernorm", True)),
    )
    projector = Projector(cfg).to(device)
    projector.load_state_dict(ckpt.get("text_projector_state_dict", ckpt.get("state_dict", {})), strict=False)
    projector.eval()
    theta_raw = float(ckpt.get("theta_T", 0.0))
    theta_t = torch.tensor(theta_raw, device=device, dtype=torch.float32)
    temperature = F.softplus(theta_t) + 1.0e-4
    return projector, temperature, dict(ckpt)


# -------------------------- class metadata/bonus ----------------------------


def _load_under_table(path: Path) -> Dict[int, Dict[str, Any]]:
    rows = _csv_read(path, required=True)
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        rid = _as_int(r.get("raw_id"))
        if rid is None:
            continue
        out[int(rid)] = dict(r)
    return out


def _metric_value(row: Mapping[str, Any], metric: str) -> float:
    if metric == "negative_mean_responsibility_per_support":
        return -_safe_float(row.get("mean_responsibility_per_support"))
    return _safe_float(row.get(metric))


def _rank_norm(vals_by_id: Mapping[int, float]) -> Dict[int, float]:
    items = sorted(vals_by_id.items(), key=lambda kv: kv[1], reverse=True)
    n = len(items)
    if n <= 1:
        return {k: 1.0 for k in vals_by_id}
    out: Dict[int, float] = {}
    for idx, (k, _v) in enumerate(items):
        out[k] = 1.0 - idx / (n - 1)
    return out


def _minmax_norm(vals_by_id: Mapping[int, float]) -> Dict[int, float]:
    if not vals_by_id:
        return {}
    vals = [v for v in vals_by_id.values() if math.isfinite(v)]
    if not vals:
        return {k: 0.0 for k in vals_by_id}
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return {k: 0.0 for k in vals_by_id}
    return {k: max(0.0, min(1.0, (v - lo) / (hi - lo))) for k, v in vals_by_id.items()}


def _zclip_norm(vals_by_id: Mapping[int, float]) -> Dict[int, float]:
    if not vals_by_id:
        return {}
    vals = [v for v in vals_by_id.values() if math.isfinite(v)]
    if not vals:
        return {k: 0.0 for k in vals_by_id}
    mean = statistics.fmean(vals)
    st = statistics.pstdev(vals)
    if st <= 1e-12:
        return {k: 0.0 for k in vals_by_id}
    return {k: max(0.0, min(1.0, ((v - mean) / st + 2.0) / 4.0)) for k, v in vals_by_id.items()}


def _excluded_high_support_ids(table: Mapping[int, Mapping[str, Any]], policy: str) -> Set[int]:
    if policy == "exclude_top_1pct_support":
        items = sorted(((rid, _safe_float(row.get("candidate_support"))) for rid, row in table.items()), key=lambda kv: kv[1], reverse=True)
        k = max(1, int(math.ceil(0.01 * len(items))))
        return {rid for rid, _ in items[:k]}
    if policy == "exclude_top_5_support":
        items = sorted(((rid, _safe_float(row.get("candidate_support"))) for rid, row in table.items()), key=lambda kv: kv[1], reverse=True)
        return {rid for rid, _ in items[:5]}
    return set()


def _build_bonus_map(
    *,
    table: Mapping[int, Mapping[str, Any]],
    metric: str,
    alpha: float,
    max_bonus: float,
    support_threshold: float,
    high_support_policy: str,
    normalization: str,
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    excluded = _excluded_high_support_ids(table, high_support_policy)
    raw_vals: Dict[int, float] = {}
    for rid, row in table.items():
        support = _safe_float(row.get("candidate_support"))
        if support < support_threshold:
            continue
        if rid in excluded:
            continue
        v = _metric_value(row, metric)
        if high_support_policy == "log_squash":
            v = math.log1p(max(0.0, v))
        if not math.isfinite(v):
            continue
        raw_vals[rid] = v
    if normalization == "minmax":
        norm = _minmax_norm(raw_vals)
    elif normalization == "zclip":
        norm = _zclip_norm(raw_vals)
    else:
        norm = _rank_norm(raw_vals)
    bonus = {rid: min(float(max_bonus), max(0.0, float(alpha) * float(norm.get(rid, 0.0)))) for rid in raw_vals}

    rows: List[Dict[str, Any]] = []
    rank_items = sorted(raw_vals.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (rid, raw_v) in enumerate(rank_items, start=1):
        row = table.get(rid, {})
        rows.append({
            "raw_id": rid,
            "class_name": row.get("class_name", ""),
            "bonus_rank": rank,
            "metric_raw_value": raw_v,
            "metric_norm_value": norm.get(rid, 0.0),
            "bonus": bonus.get(rid, 0.0),
            "candidate_support": row.get("candidate_support", ""),
            "responsibility_mass": row.get("responsibility_mass", ""),
            "top1_count": row.get("top1_count", ""),
            "gt_count": row.get("gt_count", ""),
            "delta_gt_top1_hit_rate": row.get("delta_gt_top1_hit_rate", ""),
            "delta_mean_normalized_gt_rank": row.get("delta_mean_normalized_gt_rank", ""),
            "is_nohub_degraded_either": row.get("is_nohub_degraded_either", ""),
            "certificate_family": row.get("certificate_family", ""),
            "certificate_type": row.get("certificate_type", ""),
            "resolved_round": row.get("resolved_round", ""),
            "base_group": row.get("base_group", ""),
            "person_conditioned": row.get("person_conditioned", ""),
        })
    return bonus, rows


def _class_meta(table: Mapping[int, Mapping[str, Any]], raw_id: int) -> Dict[str, Any]:
    r = table.get(int(raw_id), {})
    rr = str(r.get("resolved_round", ""))
    cert = str(r.get("certificate_type", ""))
    family = str(r.get("certificate_family", ""))
    if not family:
        family = "person_conditioned" if "person" in cert else ("anchor_conditioned" if "anchor" in cert or rr not in {"", "0", "unresolved"} else "unresolved")
    return {
        "class_name": r.get("class_name", ""),
        "certificate_family": family or "unresolved",
        "certificate_type": cert or "unresolved",
        "resolved_round": rr if rr != "" else "unresolved",
        "base_group": r.get("base_group", ""),
        "person_conditioned": str(r.get("person_conditioned", "")),
        "is_anchor_conditioned": int(_safe_float(r.get("is_anchor_conditioned")) > 0 or family == "anchor_conditioned"),
        "is_nohub_degraded_top1": int(_safe_float(r.get("is_nohub_degraded_top1")) > 0),
        "is_nohub_degraded_rank": int(_safe_float(r.get("is_nohub_degraded_rank")) > 0),
        "is_nohub_degraded_either": int(_safe_float(r.get("is_nohub_degraded_either")) > 0),
    }


# ----------------------------- score cache ----------------------------------


def _build_score_rows(
    *,
    checkpoint_path: Path,
    examples: Sequence[Mapping[str, Any]],
    clip_y_base: Mapping[int, Set[int]],
    text_vocab_ids: Sequence[int],
    text_vocab_matrix: np.ndarray,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_to_idx = {int(r): i for i, r in enumerate(text_vocab_ids)}
    projector, temperature, ckpt = _load_checkpoint(checkpoint_path, device)
    text_tensor = torch.from_numpy(np.asarray(text_vocab_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        text_proj = F.normalize(projector(text_tensor), p=2.0, dim=-1)
    groups: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for ex in examples:
        groups[int(ex["clip_id"])].append(ex)

    score_rows: List[Dict[str, Any]] = []
    skipped = Counter()
    with torch.no_grad():
        for clip, group in groups.items():
            candidates = sorted(int(x) for x in clip_y_base.get(int(clip), set()) if int(x) in raw_to_idx)
            if not candidates:
                skipped["no_candidate"] += len(group)
                continue
            cand_idx = torch.tensor([raw_to_idx[int(x)] for x in candidates], device=device, dtype=torch.long)
            cand_text = text_proj[cand_idx]
            Z = torch.stack([
                torch.from_numpy(_normalize_np(np.asarray(ex["carrier_vec"], dtype=np.float32))).to(device=device, dtype=torch.float32)
                for ex in group
            ], dim=0)
            Z = F.normalize(Z, p=2.0, dim=-1)
            scores = (torch.matmul(Z, cand_text.t()) / temperature).detach().cpu().numpy().astype(np.float32, copy=False)
            for qi, ex in enumerate(group):
                gt_raw = _as_int(ex.get("matched_gt_raw_id"))
                if gt_raw is None:
                    skipped["no_gt_raw_id"] += 1
                    continue
                if int(gt_raw) not in candidates:
                    skipped["gt_not_in_y_base"] += 1
                    continue
                score_rows.append({
                    "clip_id": int(clip),
                    "trajectory_id": str(ex.get("trajectory_id", "")),
                    "gt_raw_id": int(gt_raw),
                    "candidate_ids": candidates,
                    "scores": scores[qi].copy(),
                })
    meta = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_protocol": ckpt.get("protocol", ""),
        "temperature": float(temperature.detach().cpu().item()),
        "score_row_count": len(score_rows),
        "skipped": dict(skipped),
    }
    return score_rows, meta


class Stats:
    def __init__(self) -> None:
        self.n = 0
        self.sums: Dict[str, float] = defaultdict(float)

    def add(self, **kw: float) -> None:
        self.n += 1
        for k, v in kw.items():
            if v is not None and math.isfinite(float(v)):
                self.sums[k] += float(v)

    def row(self, prefix: Mapping[str, Any]) -> Dict[str, Any]:
        out = dict(prefix)
        out["gt_count"] = int(self.n)
        for k, v in self.sums.items():
            out[k] = float(v / max(self.n, 1))
        return out


def _eval_setting(score_rows: Sequence[Mapping[str, Any]], bonus_map: Mapping[int, float], table: Mapping[int, Mapping[str, Any]], setting: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    stats_overall = Stats()
    grouped: Dict[Tuple[str, str], Stats] = defaultdict(Stats)
    by_class: Dict[int, Stats] = defaultdict(Stats)

    for row in score_rows:
        candidates: List[int] = [int(x) for x in row["candidate_ids"]]
        gt_raw = int(row["gt_raw_id"])
        scores = np.asarray(row["scores"], dtype=np.float32).copy()
        if bonus_map:
            bonuses = np.asarray([float(bonus_map.get(int(c), 0.0)) for c in candidates], dtype=np.float32)
            scores = scores + bonuses
        order = np.argsort(-scores)
        gt_pos = candidates.index(gt_raw)
        rank = int(np.where(order == gt_pos)[0][0]) + 1
        denom = max(len(candidates) - 1, 1)
        norm_rank = float((rank - 1) / denom)
        top1 = 1.0 if rank == 1 else 0.0
        vals = {
            "mean_normalized_gt_rank": norm_rank,
            "gt_top1_hit_rate": top1,
            "candidate_size_mean": float(len(candidates)),
            "gt_rank_mean": float(rank),
            "applied_bonus_mean": float(bonus_map.get(gt_raw, 0.0)),
        }
        stats_overall.add(**vals)
        by_class[gt_raw].add(**vals)
        md = _class_meta(table, gt_raw)
        grouped[("certificate_family", str(md.get("certificate_family", "unresolved")))].add(**vals)
        grouped[("certificate_type", str(md.get("certificate_type", "unresolved")))].add(**vals)
        grouped[("resolved_round", str(md.get("resolved_round", "unresolved")))].add(**vals)
        if md.get("base_group") not in (None, ""):
            grouped[("base_observed_unobserved", str(md.get("base_group")))].add(**vals)
        if md.get("person_conditioned") not in (None, ""):
            grouped[("person_conditioned", str(md.get("person_conditioned")))].add(**vals)
        grouped[("nohub_degraded_either", str(md.get("is_nohub_degraded_either", 0)))].add(**vals)
        grouped[("anchor_conditioned", str(md.get("is_anchor_conditioned", 0)))].add(**vals)

    setting_id = str(setting.get("setting_id"))
    prefix = dict(setting)
    rows_group = [stats_overall.row({**prefix, "group_name": "overall", "group_value": "overall"})]
    for (gname, gval), st in sorted(grouped.items()):
        rows_group.append(st.row({**prefix, "group_name": gname, "group_value": gval}))
    rows_class: List[Dict[str, Any]] = []
    for rid, st in sorted(by_class.items()):
        md = _class_meta(table, rid)
        r = st.row({**prefix, "raw_id": rid, **md})
        rows_class.append(r)
    summary = rows_group[0] if rows_group else {**prefix, "gt_count": 0}
    summary["setting_id"] = setting_id
    return summary, rows_group, rows_class


# ------------------------------ reporting -----------------------------------


def _index_by(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], Mapping[str, Any]]:
    return {tuple(r.get(k, "") for k in keys): r for r in rows}


def _delta(a: Mapping[str, Any], b: Mapping[str, Any], key: str) -> float:
    return _safe_float(a.get(key)) - _safe_float(b.get(key))


def _select_recommended(summary_rows: Sequence[Mapping[str, Any]], group_rows: Sequence[Mapping[str, Any]], baseline_setting_id: str = "no_bonus") -> Dict[str, Any]:
    by_setting = {str(r.get("setting_id")): r for r in summary_rows}
    base = by_setting.get(baseline_setting_id)
    if not base:
        return {}
    group_idx = _index_by(group_rows, ["setting_id", "group_name", "group_value"])
    base_group = {k[1:]: v for k, v in group_idx.items() if k[0] == baseline_setting_id}

    best: Optional[Dict[str, Any]] = None
    for r in summary_rows:
        sid = str(r.get("setting_id"))
        if sid == baseline_setting_id:
            continue
        overall_top1_delta = _delta(r, base, "gt_top1_hit_rate")
        overall_rank_delta = _delta(r, base, "mean_normalized_gt_rank")
        rg = {k[1:]: v for k, v in group_idx.items() if k[0] == sid}
        def gd(gname: str, gval: str, metric: str) -> float:
            return _delta(rg.get((gname, gval), {}), base_group.get((gname, gval), {}), metric)
        anchor_top1_delta = gd("certificate_family", "anchor_conditioned", "gt_top1_hit_rate")
        anchor_rank_delta = gd("certificate_family", "anchor_conditioned", "mean_normalized_gt_rank")
        initial_top1_delta = gd("certificate_family", "initial_context_identifiable", "gt_top1_hit_rate")
        person_top1_delta = gd("certificate_family", "person_conditioned", "gt_top1_hit_rate")
        degraded_top1_delta = gd("nohub_degraded_either", "1", "gt_top1_hit_rate")
        degraded_rank_delta = gd("nohub_degraded_either", "1", "mean_normalized_gt_rank")
        # Conservative utility: reward anchor/degraded repair; penalize overall/person/initial damage.
        utility = (
            3.0 * anchor_top1_delta
            - 1.5 * max(0.0, anchor_rank_delta)
            + 2.0 * degraded_top1_delta
            - 1.0 * max(0.0, degraded_rank_delta)
            + 2.0 * min(0.0, overall_top1_delta + 0.001)
            - 1.5 * max(0.0, overall_rank_delta - 0.001)
            + 1.0 * min(0.0, initial_top1_delta + 0.001)
            + 1.0 * min(0.0, person_top1_delta + 0.001)
        )
        cand = dict(r)
        cand.update({
            "overall_delta_gt_top1_hit_rate": overall_top1_delta,
            "overall_delta_mean_normalized_gt_rank": overall_rank_delta,
            "anchor_delta_gt_top1_hit_rate": anchor_top1_delta,
            "anchor_delta_mean_normalized_gt_rank": anchor_rank_delta,
            "initial_delta_gt_top1_hit_rate": initial_top1_delta,
            "person_delta_gt_top1_hit_rate": person_top1_delta,
            "degraded_delta_gt_top1_hit_rate": degraded_top1_delta,
            "degraded_delta_mean_normalized_gt_rank": degraded_rank_delta,
            "recommendation_utility": utility,
        })
        if best is None or utility > _safe_float(best.get("recommendation_utility"), -1e9):
            best = cand
    return best or {}


def _make_top_delta(per_class_rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]], setting_id: str, k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    base_idx = {int(r.get("raw_id")): r for r in baseline_rows}
    deltas: List[Dict[str, Any]] = []
    for r in per_class_rows:
        if str(r.get("setting_id")) != str(setting_id):
            continue
        rid = int(r.get("raw_id"))
        b = base_idx.get(rid)
        if not b:
            continue
        rr = dict(r)
        rr["delta_gt_top1_hit_rate"] = _delta(r, b, "gt_top1_hit_rate")
        rr["delta_mean_normalized_gt_rank"] = _delta(r, b, "mean_normalized_gt_rank")
        rr["delta_gt_rank_mean"] = _delta(r, b, "gt_rank_mean")
        deltas.append(rr)
    improved = sorted(deltas, key=lambda x: (_safe_float(x.get("delta_gt_top1_hit_rate")), -_safe_float(x.get("delta_mean_normalized_gt_rank"))), reverse=True)[:k]
    degraded = sorted(deltas, key=lambda x: (_safe_float(x.get("delta_gt_top1_hit_rate")), -_safe_float(x.get("delta_mean_normalized_gt_rank"))))[:k]
    return improved, degraded


# ---------------------------------- main -------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only demand-floor replay for GT-fullY clean nohub.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--checkpoint", required=True, help="soft_e2e_nohub prealign_last.pth")
    p.add_argument("--under_assigned_csv", required=True, help="under_assigned_class_table.csv from prior audit")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--repo_root", default="/mnt/sda/zyy/code/wsovvis")
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--split_json", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--smoke_max_trajectories", type=int, default=128)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--metrics", default=METRICS_DEFAULT)
    p.add_argument("--alphas", default="0.005,0.01,0.02")
    p.add_argument("--max_bonuses", default="0.01,0.02,0.05")
    p.add_argument("--support_thresholds", default="20,50")
    p.add_argument("--high_support_policies", default="none,log_squash,exclude_top_1pct_support")
    p.add_argument("--normalization", choices=["rank", "minmax", "zclip"], default="rank")
    p.add_argument("--top_k", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    under_csv = Path(args.under_assigned_csv).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not under_csv.is_file():
        raise FileNotFoundError(under_csv)

    _bootstrap_asset_links(out, asset_root)
    examples, clip_y_base, base_ids, mat_meta = _load_examples_with_gt(args, out)
    text_vocab_ids, _records, text_vocab_matrix = load_text_vocab(out)
    device = torch.device(str(args.device))
    score_rows, score_meta = _build_score_rows(
        checkpoint_path=checkpoint,
        examples=examples,
        clip_y_base=clip_y_base,
        text_vocab_ids=text_vocab_ids,
        text_vocab_matrix=text_vocab_matrix,
        device=device,
    )
    table = _load_under_table(under_csv)

    metrics = _parse_str_list(args.metrics)
    alphas = _parse_float_list(args.alphas)
    max_bonuses = _parse_float_list(args.max_bonuses)
    support_thresholds = _parse_float_list(args.support_thresholds)
    high_support_policies = _parse_str_list(args.high_support_policies)

    all_summary: List[Dict[str, Any]] = []
    all_groups: List[Dict[str, Any]] = []
    all_per_class: List[Dict[str, Any]] = []
    all_bonus_rows: List[Dict[str, Any]] = []

    base_setting = {
        "setting_id": "no_bonus",
        "metric": "none",
        "alpha": 0.0,
        "max_bonus": 0.0,
        "support_threshold": 0.0,
        "high_support_policy": "none",
        "normalization": args.normalization,
    }
    summary, groups, per_cls = _eval_setting(score_rows, {}, table, base_setting)
    all_summary.append(summary)
    all_groups.extend(groups)
    all_per_class.extend(per_cls)

    for metric in metrics:
        for alpha in alphas:
            for max_bonus in max_bonuses:
                for support_threshold in support_thresholds:
                    for policy in high_support_policies:
                        setting_id = f"metric={metric}__alpha={alpha:g}__max={max_bonus:g}__support={support_threshold:g}__policy={policy}__norm={args.normalization}"
                        setting = {
                            "setting_id": setting_id,
                            "metric": metric,
                            "alpha": alpha,
                            "max_bonus": max_bonus,
                            "support_threshold": support_threshold,
                            "high_support_policy": policy,
                            "normalization": args.normalization,
                        }
                        bonus_map, bonus_rows = _build_bonus_map(
                            table=table,
                            metric=metric,
                            alpha=alpha,
                            max_bonus=max_bonus,
                            support_threshold=support_threshold,
                            high_support_policy=policy,
                            normalization=args.normalization,
                        )
                        for br in bonus_rows[: max(100, args.top_k)]:
                            all_bonus_rows.append({**setting, **br})
                        srow, grows, crows = _eval_setting(score_rows, bonus_map, table, setting)
                        srow["bonus_class_count"] = len([v for v in bonus_map.values() if v > 0])
                        srow["bonus_mean_nonzero"] = statistics.fmean([v for v in bonus_map.values() if v > 0]) if any(v > 0 for v in bonus_map.values()) else 0.0
                        srow["bonus_max"] = max(bonus_map.values()) if bonus_map else 0.0
                        all_summary.append(srow)
                        all_groups.extend(grows)
                        all_per_class.extend(crows)

    # Attach deltas vs no_bonus to summaries and groups.
    base_summary = all_summary[0]
    enriched_summary: List[Dict[str, Any]] = []
    for r in all_summary:
        rr = dict(r)
        rr["delta_gt_top1_hit_rate_vs_no_bonus"] = _delta(r, base_summary, "gt_top1_hit_rate")
        rr["delta_mean_normalized_gt_rank_vs_no_bonus"] = _delta(r, base_summary, "mean_normalized_gt_rank")
        rr["delta_gt_rank_mean_vs_no_bonus"] = _delta(r, base_summary, "gt_rank_mean")
        enriched_summary.append(rr)
    all_summary = enriched_summary

    base_group_idx = {tuple(r.get(k, "") for k in ["group_name", "group_value"]): r for r in all_groups if r.get("setting_id") == "no_bonus"}
    enriched_groups: List[Dict[str, Any]] = []
    for r in all_groups:
        b = base_group_idx.get((r.get("group_name"), r.get("group_value")), {})
        rr = dict(r)
        rr["delta_gt_top1_hit_rate_vs_no_bonus"] = _delta(r, b, "gt_top1_hit_rate")
        rr["delta_mean_normalized_gt_rank_vs_no_bonus"] = _delta(r, b, "mean_normalized_gt_rank")
        rr["delta_gt_rank_mean_vs_no_bonus"] = _delta(r, b, "gt_rank_mean")
        enriched_groups.append(rr)
    all_groups = enriched_groups

    base_per_cls = [r for r in all_per_class if r.get("setting_id") == "no_bonus"]
    base_per_idx = {int(r.get("raw_id")): r for r in base_per_cls}
    enriched_per_class: List[Dict[str, Any]] = []
    for r in all_per_class:
        b = base_per_idx.get(int(r.get("raw_id")))
        rr = dict(r)
        if b:
            rr["delta_gt_top1_hit_rate_vs_no_bonus"] = _delta(r, b, "gt_top1_hit_rate")
            rr["delta_mean_normalized_gt_rank_vs_no_bonus"] = _delta(r, b, "mean_normalized_gt_rank")
            rr["delta_gt_rank_mean_vs_no_bonus"] = _delta(r, b, "gt_rank_mean")
        enriched_per_class.append(rr)
    all_per_class = enriched_per_class

    recommended = _select_recommended(all_summary, all_groups, baseline_setting_id="no_bonus")
    rec_id = str(recommended.get("setting_id", ""))
    rec_per = [r for r in all_per_class if str(r.get("setting_id")) == rec_id]
    improved, degraded = _make_top_delta(all_per_class, base_per_cls, rec_id, args.top_k) if rec_id else ([], [])
    rec_anchor = [r for r in rec_per if int(_safe_float(r.get("is_anchor_conditioned"))) > 0 and int(_safe_float(r.get("is_nohub_degraded_either"))) > 0]
    rec_degraded = [r for r in rec_per if int(_safe_float(r.get("is_nohub_degraded_either"))) > 0]

    summary_fields = [
        "setting_id", "metric", "alpha", "max_bonus", "support_threshold", "high_support_policy", "normalization",
        "gt_count", "mean_normalized_gt_rank", "gt_top1_hit_rate", "candidate_size_mean", "gt_rank_mean",
        "applied_bonus_mean", "bonus_class_count", "bonus_mean_nonzero", "bonus_max",
        "delta_gt_top1_hit_rate_vs_no_bonus", "delta_mean_normalized_gt_rank_vs_no_bonus", "delta_gt_rank_mean_vs_no_bonus",
    ]
    group_fields = summary_fields[:7] + ["group_name", "group_value", "gt_count", "mean_normalized_gt_rank", "gt_top1_hit_rate", "candidate_size_mean", "gt_rank_mean", "applied_bonus_mean", "delta_gt_top1_hit_rate_vs_no_bonus", "delta_mean_normalized_gt_rank_vs_no_bonus", "delta_gt_rank_mean_vs_no_bonus"]
    class_fields = summary_fields[:7] + ["raw_id", "class_name", "certificate_family", "certificate_type", "resolved_round", "base_group", "person_conditioned", "is_anchor_conditioned", "is_nohub_degraded_top1", "is_nohub_degraded_rank", "is_nohub_degraded_either", "gt_count", "mean_normalized_gt_rank", "gt_top1_hit_rate", "candidate_size_mean", "gt_rank_mean", "applied_bonus_mean", "delta_gt_top1_hit_rate_vs_no_bonus", "delta_mean_normalized_gt_rank_vs_no_bonus", "delta_gt_rank_mean_vs_no_bonus"]

    _csv_write(out / "summary_by_setting.csv", all_summary, summary_fields)
    _csv_write(out / "summary_by_setting_group.csv", all_groups, group_fields)
    _csv_write(out / "per_class_replay_delta.csv", all_per_class, class_fields)
    _csv_write(out / "demand_floor_bonus_table.csv", all_bonus_rows)
    _csv_write(out / "top20_replay_improved_classes.csv", improved, class_fields + ["delta_gt_top1_hit_rate", "delta_mean_normalized_gt_rank", "delta_gt_rank_mean"])
    _csv_write(out / "top20_replay_degraded_classes.csv", degraded, class_fields + ["delta_gt_top1_hit_rate", "delta_mean_normalized_gt_rank", "delta_gt_rank_mean"])
    _csv_write(out / "anchor_degraded_replay_result.csv", rec_anchor, class_fields)
    _csv_write(out / "nohub_degraded_replay_result.csv", rec_degraded, class_fields)

    overall_delta = _safe_float(recommended.get("overall_delta_gt_top1_hit_rate"))
    overall_rank_delta = _safe_float(recommended.get("overall_delta_mean_normalized_gt_rank"))
    anchor_delta = _safe_float(recommended.get("anchor_delta_gt_top1_hit_rate"))
    status_interpretation = "DEMAND_FLOOR_REPLAY_INCONCLUSIVE"
    if recommended:
        if overall_delta >= -0.001 and overall_rank_delta <= 0.001 and (anchor_delta > 0 or _safe_float(recommended.get("anchor_delta_mean_normalized_gt_rank")) < 0):
            status_interpretation = "DEMAND_FLOOR_REPLAY_PROMISING_FOR_VERYWEAK_PILOT"
        elif overall_delta < -0.001 or overall_rank_delta > 0.001:
            status_interpretation = "DEMAND_FLOOR_REPLAY_RISKY_OVERALL_DAMAGE"
        else:
            status_interpretation = "DEMAND_FLOOR_REPLAY_SAFE_BUT_WEAK"

    payload = {
        "status": "PASS" if score_rows else "FAIL",
        "output_dir": str(out),
        "checkpoint": str(checkpoint),
        "under_assigned_csv": str(under_csv),
        "score_meta": score_meta,
        "materialization": mat_meta,
        "base_count": len(base_ids),
        "gt_example_count": len(examples),
        "evaluated_gt_rows": len(score_rows),
        "sweep": {
            "metrics": metrics,
            "alphas": alphas,
            "max_bonuses": max_bonuses,
            "support_thresholds": support_thresholds,
            "high_support_policies": high_support_policies,
            "normalization": args.normalization,
        },
        "setting_count_including_no_bonus": len(all_summary),
        "recommended_setting": recommended,
        "interpretation": status_interpretation,
        "outputs": {
            "summary_by_setting": str(out / "summary_by_setting.csv"),
            "summary_by_setting_group": str(out / "summary_by_setting_group.csv"),
            "per_class_replay_delta": str(out / "per_class_replay_delta.csv"),
            "demand_floor_bonus_table": str(out / "demand_floor_bonus_table.csv"),
            "top20_replay_improved_classes": str(out / "top20_replay_improved_classes.csv"),
            "top20_replay_degraded_classes": str(out / "top20_replay_degraded_classes.csv"),
            "anchor_degraded_replay_result": str(out / "anchor_degraded_replay_result.csv"),
            "nohub_degraded_replay_result": str(out / "nohub_degraded_replay_result.csv"),
        },
    }
    _json_write(out / "summary.json", payload)

    rec_lines = [
        "# Demand Floor Replay Recommended Setting",
        "",
        f"Interpretation: `{status_interpretation}`",
        "",
    ]
    if recommended:
        for k in [
            "setting_id", "metric", "alpha", "max_bonus", "support_threshold", "high_support_policy", "normalization",
            "overall_delta_gt_top1_hit_rate", "overall_delta_mean_normalized_gt_rank",
            "anchor_delta_gt_top1_hit_rate", "anchor_delta_mean_normalized_gt_rank",
            "initial_delta_gt_top1_hit_rate", "person_delta_gt_top1_hit_rate",
            "degraded_delta_gt_top1_hit_rate", "degraded_delta_mean_normalized_gt_rank", "recommendation_utility",
        ]:
            rec_lines.append(f"- {k}: `{recommended.get(k, '')}`")
    else:
        rec_lines.append("No recommended non-baseline setting could be selected.")
    (out / "recommended_setting.md").write_text("\n".join(rec_lines) + "\n", encoding="utf-8")

    takeover = f"""# Demand Floor Replay Takeover

Status: `{payload['status']}`

Output: `{out}`

## Scope

Read-only demand-floor replay for GT-fullY clean nohub. No training, no checkpoint modification, no mAP, no VideoCutLER/Y′/extra. GT is used only for replay evaluation after bonus construction.

## Inputs

- checkpoint: `{checkpoint}`
- under_assigned_csv: `{under_csv}`
- dataset_name: `{args.dataset_name}`

## Key findings

- evaluated_gt_rows: `{len(score_rows)}`
- setting_count_including_no_bonus: `{len(all_summary)}`
- interpretation: `{status_interpretation}`
- recommended_setting: `{recommended.get('setting_id', '') if recommended else ''}`

## Core outputs

- summary.json
- summary_by_setting.csv
- summary_by_setting_group.csv
- per_class_replay_delta.csv
- demand_floor_bonus_table.csv
- top20_replay_improved_classes.csv
- top20_replay_degraded_classes.csv
- anchor_degraded_replay_result.csv
- nohub_degraded_replay_result.csv
- recommended_setting.md

## Decision rule

Proceed to a veryweak training pilot only if the recommended setting improves anchor/degraded groups while keeping overall top1 and rank within the nohub tolerance. Stop if the best setting only shifts rank/top1 by negligible amounts or damages initial/person/overall groups.
"""
    (out / "DEMAND_FLOOR_REPLAY_TAKEOVER.md").write_text(takeover, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
