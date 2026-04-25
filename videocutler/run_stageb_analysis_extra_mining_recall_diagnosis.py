from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
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
    _make_iter_progress,
    _materialize_valid_samples,
    _prepare_probe_examples,
    _score_batches,
)
from videocutler.ext_stageb_ovvis.audit.g8_minimal_split_audit import (  # noqa: E402
    MinimalSplitAuditConfig,
    _build_rows_and_cache as _minimal_build_rows_and_cache,
    _canonical_sidecar_gt_raw_id,
    _materialize_shared_inputs as _minimal_materialize_shared_inputs,
    _score_batch as _minimal_score_batch,
    _split_order_for_dataset as _minimal_split_order_for_dataset,
    _summarize_minimal_rows as _minimal_summarize_minimal_rows,
)
from videocutler.ext_stageb_ovvis.audit.gt_attribution_rank_audit import _all_gt_split_label  # noqa: E402
from videocutler.ext_stageb_ovvis.audit.trajectory_gt_audit import load_gt_sidecar_lookup  # noqa: E402
from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import (  # noqa: E402
    load_lvvis_base_and_novel_raw_ids,
)

Record = Dict[str, Any]
DEFAULT_RANK_KS: Tuple[int, ...] = (1, 2, 3, 4, 5, 10, 20, 50, 100)
TRAIN_SPLIT_ORDER: Tuple[str, ...] = ("base_observed", "base_unobserved")
VAL_SPLIT_ORDER: Tuple[str, ...] = ("base", "novel")


@dataclass(frozen=True)
class DiagnosisConfig:
    run_root: Path
    runtime_output_root: Path
    dataset_name: str
    trajectory_source_branch: str
    stage: str
    device: str
    output_dir: Optional[Path]
    sidecar_root: Optional[Path]
    batch_size: int
    rank_ks: Tuple[int, ...]
    top_examples: int
    top_classes: int
    smoke: bool
    smoke_max_trajectories: int
    subset_fraction: Optional[float]
    show_progress: bool


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def _parse_rank_ks(value: str) -> Tuple[int, ...]:
    items: List[int] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if k <= 0:
            raise argparse.ArgumentTypeError("rank K values must be positive")
        items.append(k)
    return tuple(sorted(set(items))) or DEFAULT_RANK_KS


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text).rstrip() + "\n", encoding="utf-8")


def _load_json(path: Path) -> Optional[Record]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Record]:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        v = float(value)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return default


def _unique_ints(values: Any) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    if values is None:
        return out
    if not isinstance(values, (list, tuple)):
        return out
    for value in values:
        iv = _safe_int(value)
        if iv is None or iv in seen:
            continue
        seen.add(iv)
        out.append(int(iv))
    return out


def _rate_bools(values: Sequence[bool]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray([1.0 if bool(v) else 0.0 for v in values], dtype=np.float64)))


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), float(q)))


def _split_order(dataset_name: str) -> Tuple[str, ...]:
    return TRAIN_SPLIT_ORDER if str(dataset_name) == "lvvis_train_base" else VAL_SPLIT_ORDER


def _default_output_dir(run_root: Path, dataset_name: str, stage: str) -> Path:
    return run_root / "analysis" / "extra_mining_recall_diagnosis" / dataset_name / stage


def _sidecar_root(config: DiagnosisConfig) -> Path:
    return Path(config.sidecar_root).expanduser().resolve() if config.sidecar_root is not None else Path(config.run_root).expanduser().resolve()


def _read_responsibility_rows(run_root: Path, stage: str) -> Tuple[Dict[str, Record], Dict[str, Any]]:
    path = run_root / "train" / stage / "responsibility_records.jsonl"
    rows: Dict[str, Record] = {}
    total = 0
    for row in _iter_jsonl(path):
        total += 1
        tid = str(row.get("trajectory_id", "")).strip()
        if tid:
            rows[tid] = dict(row)
    return rows, {"path": str(path), "exists": path.is_file(), "record_count": int(total), "by_tid_count": int(len(rows))}


def _extract_gt_iou(sidecar: Mapping[str, Any]) -> Optional[float]:
    # Keep this read-only and permissive. Different sidecars have used slightly
    # different names across audits.
    direct_keys = (
        "matched_gt_iou",
        "matched_iou",
        "best_iou",
        "trajectory_gt_iou",
        "gt_iou",
        "iou",
        "max_iou",
        "matched_mask_iou",
    )
    for key in direct_keys:
        val = _safe_float(sidecar.get(key), default=None)
        if val is not None:
            return float(val)
    for nested_key in ("match", "matched_gt", "gt_match", "metrics", "iou_metrics"):
        nested = sidecar.get(nested_key)
        if isinstance(nested, Mapping):
            val = _extract_gt_iou(nested)
            if val is not None:
                return float(val)
    return None


def _iou_bucket(iou: Optional[float]) -> str:
    if iou is None:
        return "MISSING_IOU"
    if iou < 0.1:
        return "lt_0.1"
    if iou < 0.3:
        return "0.1_0.3"
    if iou < 0.5:
        return "0.3_0.5"
    return "ge_0.5"


def _rank_from_scores(scores: np.ndarray, target_idx: int, candidate_mask: np.ndarray) -> Optional[int]:
    if target_idx < 0 or target_idx >= scores.shape[0] or not bool(candidate_mask[target_idx]):
        return None
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[~candidate_mask] = -np.inf
    order = np.argsort(-masked, kind="stable")
    finite = np.isfinite(masked[order])
    order = order[finite]
    positions = np.where(order == int(target_idx))[0]
    if positions.size == 0:
        return None
    return int(positions[0]) + 1


def _top_raw_ids(scores: np.ndarray, raw_ids: Sequence[int], candidate_mask: np.ndarray, k: int) -> List[int]:
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[~candidate_mask] = -np.inf
    order = np.argsort(-masked, kind="stable")
    out: List[int] = []
    for idx in order.tolist():
        if not np.isfinite(masked[int(idx)]):
            continue
        out.append(int(raw_ids[int(idx)]))
        if len(out) >= int(k):
            break
    return out


def _best_index(scores: np.ndarray, mask: np.ndarray) -> Optional[int]:
    if not bool(np.any(mask)):
        return None
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[~mask] = -np.inf
    if not np.isfinite(masked).any():
        return None
    return int(np.argmax(masked))


def _r_final_value(r_final: Mapping[str, Any], raw_id: Optional[int]) -> float:
    if raw_id is None:
        return 0.0
    for key in (str(int(raw_id)), int(raw_id)):
        try:
            if key in r_final:  # type: ignore[operator]
                return float(r_final[key])  # type: ignore[index]
        except Exception:
            pass
    return 0.0


def _r_final_numeric_items(r_final: Mapping[str, Any]) -> List[Tuple[int, float]]:
    out: List[Tuple[int, float]] = []
    for key, value in dict(r_final).items():
        try:
            raw_id = int(key)
        except Exception:
            continue
        val = _safe_float(value, default=None)
        if val is None:
            continue
        out.append((int(raw_id), float(val)))
    return out


def _r_rank(r_final: Mapping[str, Any], raw_id: Optional[int]) -> Optional[int]:
    if raw_id is None:
        return None
    items = _r_final_numeric_items(r_final)
    if not any(rid == int(raw_id) for rid, _ in items):
        return None
    items_sorted = sorted(items, key=lambda kv: (-float(kv[1]), int(kv[0])))
    for pos, (rid, _value) in enumerate(items_sorted, start=1):
        if int(rid) == int(raw_id):
            return int(pos)
    return None


def _r_winner(r_final: Mapping[str, Any], known: Sequence[int], extra: Sequence[int], gt_raw_id: Optional[int]) -> Dict[str, Any]:
    items = _r_final_numeric_items(r_final)
    known_set = {int(x) for x in known}
    extra_set = {int(x) for x in extra}
    if not items:
        unknown_val = _safe_float(dict(r_final).get("unknown"), default=0.0) or 0.0
        return {"r_winner_raw_id": None, "r_winner_domain": "unknown", "r_winner_value": float(unknown_val), "r_winner_is_gt": False}
    best_raw, best_val = sorted(items, key=lambda kv: (-float(kv[1]), int(kv[0])))[0]
    domain = "Yprime" if best_raw in known_set else ("extra" if best_raw in extra_set else "other_nonYprime")
    return {
        "r_winner_raw_id": int(best_raw),
        "r_winner_domain": domain,
        "r_winner_value": float(best_val),
        "r_winner_is_gt": bool(gt_raw_id is not None and int(best_raw) == int(gt_raw_id)),
    }


def _load_formal_summary(run_root: Path, dataset_name: str, stage: str) -> Dict[str, Any]:
    path = run_root / "audit" / "minimal_split" / dataset_name / f"{stage}_minimal_split_summary.json"
    return _load_json(path) or {"path": str(path), "exists": False}


def _load_existing_probe_summary(run_root: Path, dataset_name: str, stage: str) -> Dict[str, Any]:
    path = run_root / "analysis" / "extra_attribution_probe" / dataset_name / stage / "summary.json"
    return _load_json(path) or {"path": str(path), "exists": False}


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gt_in_extra = [bool(r.get("gt_in_extra")) for r in rows]
    final_top1 = [bool(r.get("final_top1_is_gt")) for r in rows]
    r_gt_winner = [bool(r.get("r_final_gt_winner")) for r in rows]
    return {
        "count": int(len(rows)),
        "gt_in_extra_rate": _rate_bools(gt_in_extra),
        "final_top1_rate": _rate_bools(final_top1),
        "r_final_gt_winner_rate": _rate_bools(r_gt_winner),
        "mean_gt_mining_rank": _mean([float(r["gt_mining_rank"]) for r in rows if r.get("gt_mining_rank") is not None]),
        "median_gt_mining_rank": _median([float(r["gt_mining_rank"]) for r in rows if r.get("gt_mining_rank") is not None]),
        "mean_final_gt_rank": _mean([float(r["final_gt_rank"]) for r in rows if r.get("final_gt_rank") is not None]),
        "mean_final_gt_normalized_rank": _mean([float(r["final_gt_normalized_rank"]) for r in rows if r.get("final_gt_normalized_rank") is not None]),
        "mean_R_final_gt": _mean([float(r.get("R_final_gt", 0.0)) for r in rows]),
        "mean_margin_gt_vs_Yprime": _mean([float(r["margin_gt_vs_Yprime"]) for r in rows if r.get("margin_gt_vs_Yprime") is not None]),
        "mean_margin_gt_vs_wrong_extra": _mean([float(r["margin_gt_vs_wrong_extra"]) for r in rows if r.get("margin_gt_vs_wrong_extra") is not None]),
        "mean_margin_gt_vs_other_nonYprime": _mean([float(r["margin_gt_vs_other_nonYprime"]) for r in rows if r.get("margin_gt_vs_other_nonYprime") is not None]),
        "winner_domain_histogram": dict(Counter(str(r.get("final_winner_domain", "")) for r in rows)),
        "r_winner_domain_histogram": dict(Counter(str(r.get("r_winner_domain", "")) for r in rows)),
        "failure_bucket_histogram": dict(Counter(str(r.get("failure_bucket", "")) for r in rows)),
    }


def _rank_bucket(rank: Optional[int]) -> str:
    if rank is None:
        return "missing_or_not_ranked"
    r = int(rank)
    if r <= 3:
        return "1_3"
    if r <= 5:
        return "4_5"
    if r <= 10:
        return "6_10"
    if r <= 20:
        return "11_20"
    if r <= 50:
        return "21_50"
    return "gt_50"


def _margin_bucket(rank: Optional[int], current_k: int) -> str:
    if rank is None:
        return "missing_or_not_ranked"
    r = int(rank)
    k = int(current_k)
    if r <= k:
        return "in_topK"
    if r <= k + 2:
        return "near_miss_Kplus1_Kplus2"
    if r <= 20:
        return "medium_miss_6_20"
    return "far_miss_gt20"


def _class_label(raw_id: int, records_by_raw: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    rec = dict(records_by_raw.get(int(raw_id), {}))
    name = rec.get("name") or rec.get("class_name") or rec.get("category_name") or rec.get("synset")
    return {"raw_id": int(raw_id), "name": str(name) if name is not None else None}


def _counter_payload(counter: Counter, *, records_by_raw: Mapping[int, Mapping[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_id, count in counter.most_common(max(1, int(top_n))):
        try:
            rid = int(raw_id)
        except Exception:
            continue
        item = _class_label(rid, records_by_raw)
        item["count"] = int(count)
        rows.append(item)
    return rows

def _counter_payload_with_rates(
    counter: Counter,
    *,
    records_by_raw: Mapping[int, Mapping[str, Any]],
    top_n: int,
    denominator: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows = _counter_payload(counter, records_by_raw=records_by_raw, top_n=top_n)
    denom = int(denominator or 0)
    for item in rows:
        if denom > 0:
            item["rate"] = float(int(item.get("count", 0)) / denom)
    return rows


def _counter_payload_with_affected_gt(
    counter: Counter,
    affected_gt_by_raw: Mapping[int, Counter],
    *,
    records_by_raw: Mapping[int, Mapping[str, Any]],
    top_n: int,
    affected_top_n: int = 8,
    denominator: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    denom = int(denominator or 0)
    for raw_id, count in counter.most_common(max(1, int(top_n))):
        try:
            rid = int(raw_id)
        except Exception:
            continue
        item = _class_label(rid, records_by_raw)
        item["count"] = int(count)
        if denom > 0:
            item["rate"] = float(int(count) / denom)
        affected_rows: List[Dict[str, Any]] = []
        for gt_raw, gt_count in affected_gt_by_raw.get(rid, Counter()).most_common(max(1, int(affected_top_n))):
            try:
                gid = int(gt_raw)
            except Exception:
                continue
            gt_item = _class_label(gid, records_by_raw)
            gt_item["count"] = int(gt_count)
            affected_rows.append(gt_item)
        item["affected_gt_ids_top"] = affected_rows
        rows.append(item)
    return rows


def _load_class_name_records(runtime_output_root: Path, dataset_name: str, existing_records: Mapping[int, Mapping[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Best-effort raw-id -> class-name mapping. Read-only; missing names are reported, not fatal."""
    merged: Dict[int, Dict[str, Any]] = {int(k): dict(v) for k, v in existing_records.items()}
    source_paths: List[str] = []
    category_count_by_source: Dict[str, int] = {}
    split = "train" if str(dataset_name) == "lvvis_train_base" else "val"
    root = Path(runtime_output_root)
    annotation_names = [f"{split}_instances.json", f"instances_{split}.json", "train_instances.json", "val_instances.json"]
    annotation_roots = [
        root / "videocutler" / "datasets" / "LV-VIS" / "annotations",
        root / "datasets" / "LV-VIS" / "annotations",
        root.parent / "wsovvis_asserts" / "dataset" / "LV-VIS" / "annotations",
        root.parent / "wsovvis_asserts" / "datasets" / "LV-VIS" / "annotations",
        Path("/home/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations"),
        Path("/mnt/sda/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations"),
    ]
    candidates: List[Path] = []
    for ar in annotation_roots:
        for name in annotation_names:
            candidates.append(ar / name)
    candidates.extend([root / "package" / "reference" / "lvvis_official_base_novel_split.json", root / "weak_labels" / "weak_labels_train.json"])

    def _maybe_add_category(cat: Mapping[str, Any], src: Path) -> bool:
        rid = _safe_int(cat.get("id", cat.get("raw_id", cat.get("category_id", cat.get("class_id")))))
        if rid is None:
            return False
        name = cat.get("name") or cat.get("class_name") or cat.get("category_name") or cat.get("synset")
        rec = dict(merged.get(int(rid), {}))
        if name is not None and not (rec.get("name") or rec.get("class_name") or rec.get("category_name")):
            rec["name"] = str(name)
        if name is not None:
            rec.setdefault("class_name", str(name))
        rec.setdefault("raw_id", int(rid))
        rec.setdefault("class_name_source", str(src))
        merged[int(rid)] = rec
        return True

    seen_paths: set[str] = set()
    for path in candidates:
        path = Path(path)
        key = str(path)
        if key in seen_paths or not path.is_file():
            continue
        seen_paths.add(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        added = 0
        if isinstance(payload, Mapping):
            cats = payload.get("categories")
            if isinstance(cats, list):
                for cat in cats:
                    if isinstance(cat, Mapping) and _maybe_add_category(cat, path):
                        added += 1
            for key2 in ("category_id_to_name", "raw_id_to_name", "class_id_to_name", "categories_by_id"):
                mapping = payload.get(key2)
                if isinstance(mapping, Mapping):
                    for rid_s, name in mapping.items():
                        rid = _safe_int(rid_s)
                        if rid is None:
                            continue
                        rec = dict(merged.get(int(rid), {}))
                        if name is not None:
                            rec.setdefault("name", str(name))
                            rec.setdefault("class_name", str(name))
                        rec.setdefault("raw_id", int(rid))
                        rec.setdefault("class_name_source", str(path))
                        merged[int(rid)] = rec
                        added += 1
        if added:
            source_paths.append(str(path))
            category_count_by_source[str(path)] = int(added)
    named_count = sum(1 for rec in merged.values() if rec.get("name") or rec.get("class_name") or rec.get("category_name"))
    meta = {"status": "PASS" if named_count > 0 else "MISSING_CLASS_NAME_SOURCE", "source_paths": source_paths, "category_count_by_source": category_count_by_source, "record_count": int(len(merged)), "named_record_count": int(named_count)}
    return merged, meta


def _transfer_payload(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    transfer_counter = Counter(str(r.get("r_to_logit_transition")) for r in rows)
    return {
        "target_count": int(len(rows)),
        "transition_counts": {k: int(v) for k, v in sorted(transfer_counter.items())},
        "transition_rates": {k: float(v / max(len(rows), 1)) for k, v in sorted(transfer_counter.items())},
        "by_transition": {key: _summarize_rows([r for r in rows if str(r.get("r_to_logit_transition")) == key]) for key in sorted(transfer_counter.keys())},
    }


def _formal_aligned_decomposition(rows: Sequence[Mapping[str, Any]], *, formal_summary: Mapping[str, Any], split: str = "base_unobserved") -> Dict[str, Any]:
    split_rows = [r for r in rows if str(r.get("split")) == str(split) and bool(r.get("formal_eligible"))]
    gt_in = [r for r in split_rows if bool(r.get("gt_in_extra"))]
    gt_not = [r for r in split_rows if not bool(r.get("gt_in_extra"))]
    formal_split_summary = (((formal_summary or {}).get("split_summaries") or {}).get(str(split)) or {}) if isinstance(formal_summary, Mapping) else {}
    existing_gt_count = _safe_int(formal_split_summary.get("gt_count"), default=None)
    existing_top1 = _safe_float(formal_split_summary.get("gt_top1_hit_rate"), default=None)
    existing_rank = _safe_float(formal_split_summary.get("mean_normalized_gt_rank"), default=None)
    own_summary = _summarize_rows(split_rows)
    return {
        "formal_split": str(split),
        "formal_gt_count": int(len(split_rows)),
        "formal_gt_top1_hit_rate": own_summary.get("final_top1_rate"),
        "formal_mean_normalized_gt_rank": own_summary.get("mean_final_gt_normalized_rank"),
        "all": own_summary,
        "gt_in_extra": _summarize_rows(gt_in),
        "gt_not_in_extra": _summarize_rows(gt_not),
        "conditional": {
            "P_formal_top1_given_gt_in_extra": _rate_bools([bool(r.get("final_top1_is_gt")) for r in gt_in]),
            "P_R_final_GT_winner_given_gt_in_extra": _rate_bools([bool(r.get("r_final_gt_winner")) for r in gt_in]),
            "P_Yprime_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "Yprime" for r in gt_in]),
            "P_wrong_extra_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "extra" and not bool(r.get("final_top1_is_gt")) for r in gt_in]),
            "P_other_nonYprime_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "other_nonYprime" for r in gt_in]),
            "P_unknown_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "unknown" for r in gt_in]),
        },
        "existing_minimal_split": {"gt_count": existing_gt_count, "gt_top1_hit_rate": existing_top1, "mean_normalized_gt_rank": existing_rank},
        "self_check": {
            "gt_count_matches_minimal_split": bool(existing_gt_count is None or int(existing_gt_count) == int(len(split_rows))),
            "gt_count_diff_vs_minimal_split": None if existing_gt_count is None else int(len(split_rows) - int(existing_gt_count)),
            "gt_top1_abs_diff_vs_minimal_split": None if existing_top1 is None or own_summary.get("final_top1_rate") is None else float(abs(float(own_summary["final_top1_rate"]) - float(existing_top1))),
            "mean_normalized_rank_abs_diff_vs_minimal_split": None if existing_rank is None or own_summary.get("mean_final_gt_normalized_rank") is None else float(abs(float(own_summary["mean_final_gt_normalized_rank"]) - float(existing_rank))),
        },
    }


def _known_bool_rate(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    vals: List[bool] = []
    for row in rows:
        if key in row and row.get(key) is not None:
            vals.append(bool(row.get(key)))
    return _rate_bools(vals)


def _formal_rows_basic_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    joined_rows = [r for r in rows if not bool(r.get("missing_diagnosis_record"))]
    gt_in_rows = [r for r in joined_rows if bool(r.get("gt_in_extra"))]
    gt_not_rows = [r for r in joined_rows if r.get("gt_in_extra") is False]
    return {
        "count": int(len(rows)),
        "diagnosis_joined_count": int(len(joined_rows)),
        "missing_diagnosis_record_count": int(len(rows) - len(joined_rows)),
        "missing_diagnosis_record_rate": float((len(rows) - len(joined_rows)) / max(len(rows), 1)),
        "gt_in_extra_known_count": int(len(gt_in_rows) + len(gt_not_rows)),
        "gt_in_extra_count": int(len(gt_in_rows)),
        "gt_not_in_extra_count": int(len(gt_not_rows)),
        "gt_in_extra_rate": _rate_bools([bool(r.get("gt_in_extra")) for r in joined_rows if r.get("gt_in_extra") is not None]),
        "final_top1_rate": _rate_bools([bool(r.get("final_top1_is_gt")) for r in rows]),
        "r_final_gt_winner_rate": _rate_bools([bool(r.get("r_final_gt_winner")) for r in joined_rows if r.get("r_final_gt_winner") is not None]),
        "mean_gt_mining_rank": _mean([float(r["gt_mining_rank"]) for r in joined_rows if r.get("gt_mining_rank") is not None]),
        "median_gt_mining_rank": _median([float(r["gt_mining_rank"]) for r in joined_rows if r.get("gt_mining_rank") is not None]),
        "mean_final_gt_rank": _mean([float(r["final_gt_rank"]) for r in rows if r.get("final_gt_rank") is not None]),
        "mean_final_gt_normalized_rank": _mean([float(r["final_gt_normalized_rank"]) for r in rows if r.get("final_gt_normalized_rank") is not None]),
        "mean_R_final_gt": _mean([float(r.get("R_final_gt", 0.0)) for r in joined_rows if r.get("R_final_gt") is not None]),
        "mean_margin_gt_vs_Yprime": _mean([float(r["margin_gt_vs_Yprime"]) for r in joined_rows if r.get("margin_gt_vs_Yprime") is not None]),
        "mean_margin_gt_vs_wrong_extra": _mean([float(r["margin_gt_vs_wrong_extra"]) for r in joined_rows if r.get("margin_gt_vs_wrong_extra") is not None]),
        "mean_margin_gt_vs_other_nonYprime": _mean([float(r["margin_gt_vs_other_nonYprime"]) for r in joined_rows if r.get("margin_gt_vs_other_nonYprime") is not None]),
        "winner_domain_histogram": dict(Counter(str(r.get("final_winner_domain", "missing_diagnosis_record")) for r in rows)),
        "r_winner_domain_histogram": dict(Counter(str(r.get("r_winner_domain", "missing_diagnosis_record")) for r in rows)),
        "failure_bucket_histogram": dict(Counter(str(r.get("failure_bucket", "missing_diagnosis_record")) for r in rows)),
    }


def _formal_failure_bucket(row: Mapping[str, Any]) -> str:
    if bool(row.get("missing_diagnosis_record")):
        return "missing_diagnosis_record"
    if not bool(row.get("gt_in_extra")):
        return "gt_not_in_extra_candidate"
    if bool(row.get("final_top1_is_gt")):
        return "success_final_gt_top1"
    domain = str(row.get("final_winner_domain", "unknown"))
    if domain == "Yprime":
        return "gt_in_extra_but_Yprime_wins"
    if domain == "extra":
        return "gt_in_extra_but_wrong_extra_wins"
    if domain == "other_nonYprime":
        return "gt_in_extra_but_other_nonYprime_wins"
    if domain == "unknown":
        return "gt_in_extra_but_unknown_wins"
    return "gt_in_extra_other_failure"


def _refresh_r_to_logit_transition(row: Mapping[str, Any]) -> str:
    if bool(row.get("missing_diagnosis_record")):
        return "missing_diagnosis_record"
    r_gt = bool(row.get("r_final_gt_winner", False))
    final_gt = bool(row.get("final_top1_is_gt", False))
    if r_gt and final_gt:
        return "R_GT_winner_to_final_GT_top1"
    if r_gt and not final_gt:
        return "R_GT_winner_to_final_nonGT"
    if (not r_gt) and final_gt:
        return "R_nonGT_to_final_GT_top1"
    return "R_nonGT_to_final_nonGT"


def _join_formal_authority_rows(
    *,
    formal_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    split: str,
) -> List[Dict[str, Any]]:
    diag_by_tid: Dict[str, Mapping[str, Any]] = {}
    for row in diagnostic_rows:
        tid = str(row.get("trajectory_id", "")).strip()
        if tid:
            diag_by_tid[tid] = row

    joined: List[Dict[str, Any]] = []
    for formal in formal_rows:
        if str(formal.get("split")) != str(split):
            continue
        if not bool(formal.get("gt_available_for_audit")):
            continue
        if formal.get("normalized_gt_rank") is None or formal.get("gt_top1_hit_rate") is None:
            continue
        tid = str(formal.get("trajectory_id", "")).strip()
        diag = dict(diag_by_tid.get(tid, {})) if tid else {}
        missing_diag = not bool(diag)
        row = dict(diag)
        gt_raw = _safe_int(formal.get("gt_raw_id_canonical", formal.get("gt_raw_id")))
        formal_top1 = bool(float(formal.get("gt_top1_hit_rate", 0.0)) >= 0.5)
        row.update({
            "trajectory_id": tid,
            "split": str(formal.get("split")),
            "gt_available_for_audit": True,
            "formal_eligible": True,
            "gt_raw_id": gt_raw,
            "gt_raw_id_canonical": gt_raw,
            "missing_diagnosis_record": bool(missing_diag),
            "formal_authority_joined_diagnosis_record": bool(not missing_diag),
            "formal_minimal_gt_top1_hit_rate": float(formal.get("gt_top1_hit_rate")),
            "formal_minimal_normalized_gt_rank": float(formal.get("normalized_gt_rank")),
            "final_top1_is_gt": formal_top1,
            "final_gt_normalized_rank": float(formal.get("normalized_gt_rank")),
        })
        # The formal authority only stores normalized rank, not the absolute rank. Preserve a diagnostic rank if available.
        if row.get("final_gt_rank") is None:
            row["final_gt_rank"] = None
        if missing_diag:
            row.update({
                "gt_in_extra": None,
                "gt_in_mined_extra": None,
                "final_winner_domain": "missing_diagnosis_record",
                "r_winner_domain": "missing_diagnosis_record",
                "R_final_gt": 0.0,
                "r_final_gt_winner": None,
            })
        else:
            row["R_final_gt"] = float(row.get("R_final_gt", 0.0) or 0.0)
        row["failure_bucket"] = _formal_failure_bucket(row)
        row["r_to_logit_transition"] = _refresh_r_to_logit_transition(row)
        joined.append(row)
    return joined


def _formal_aligned_decomposition_from_authority(
    *,
    formal_rows: Sequence[Mapping[str, Any]],
    existing_formal_summary: Mapping[str, Any],
    computed_minimal_summary: Mapping[str, Any],
    split: str = "base_unobserved",
    authority_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    gt_in = [r for r in formal_rows if bool(r.get("gt_in_extra")) and not bool(r.get("missing_diagnosis_record"))]
    gt_not = [r for r in formal_rows if r.get("gt_in_extra") is False and not bool(r.get("missing_diagnosis_record"))]
    missing = [r for r in formal_rows if bool(r.get("missing_diagnosis_record"))]
    formal_split_summary = (((existing_formal_summary or {}).get("split_summaries") or {}).get(str(split)) or {}) if isinstance(existing_formal_summary, Mapping) else {}
    computed_split_summary = (((computed_minimal_summary or {}).get("split_summaries") or {}).get(str(split)) or {}) if isinstance(computed_minimal_summary, Mapping) else {}
    existing_gt_count = _safe_int(formal_split_summary.get("gt_count"), default=_safe_int(computed_split_summary.get("gt_count"), default=None))
    existing_top1 = _safe_float(formal_split_summary.get("gt_top1_hit_rate"), default=_safe_float(computed_split_summary.get("gt_top1_hit_rate"), default=None))
    existing_rank = _safe_float(formal_split_summary.get("mean_normalized_gt_rank"), default=_safe_float(computed_split_summary.get("mean_normalized_gt_rank"), default=None))
    own_summary = _formal_rows_basic_summary(formal_rows)
    own_top1 = own_summary.get("final_top1_rate")
    own_rank = own_summary.get("mean_final_gt_normalized_rank")
    self_check = {
        "gt_count_matches_minimal_split": bool(existing_gt_count is None or int(existing_gt_count) == int(len(formal_rows))),
        "gt_count_diff_vs_minimal_split": None if existing_gt_count is None else int(len(formal_rows) - int(existing_gt_count)),
        "gt_top1_abs_diff_vs_minimal_split": None if existing_top1 is None or own_top1 is None else float(abs(float(own_top1) - float(existing_top1))),
        "mean_normalized_rank_abs_diff_vs_minimal_split": None if existing_rank is None or own_rank is None else float(abs(float(own_rank) - float(existing_rank))),
    }
    return {
        "formal_split": str(split),
        "formal_gt_count": int(len(formal_rows)),
        "formal_gt_top1_hit_rate": own_top1,
        "formal_mean_normalized_gt_rank": own_rank,
        "all": own_summary,
        "gt_in_extra": _formal_rows_basic_summary(gt_in),
        "gt_not_in_extra": _formal_rows_basic_summary(gt_not),
        "missing_diagnosis_record": _formal_rows_basic_summary(missing),
        "conditional": {
            "P_formal_top1_given_gt_in_extra": _rate_bools([bool(r.get("final_top1_is_gt")) for r in gt_in]),
            "P_R_final_GT_winner_given_gt_in_extra": _rate_bools([bool(r.get("r_final_gt_winner")) for r in gt_in if r.get("r_final_gt_winner") is not None]),
            "P_Yprime_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "Yprime" for r in gt_in]),
            "P_wrong_extra_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "extra" and not bool(r.get("final_top1_is_gt")) for r in gt_in]),
            "P_other_nonYprime_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "other_nonYprime" for r in gt_in]),
            "P_unknown_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "unknown" for r in gt_in]),
        },
        "existing_minimal_split": {"gt_count": existing_gt_count, "gt_top1_hit_rate": existing_top1, "mean_normalized_gt_rank": existing_rank},
        "computed_minimal_split": computed_split_summary,
        "self_check": self_check,
        "formal_authority_meta": dict(authority_meta or {}),
        "formal_authority_status": {
            "status": "PASS" if (
                bool(self_check.get("gt_count_matches_minimal_split"))
                and self_check.get("gt_top1_abs_diff_vs_minimal_split") is not None
                and abs(float(self_check.get("gt_top1_abs_diff_vs_minimal_split"))) <= 1e-12
                and self_check.get("mean_normalized_rank_abs_diff_vs_minimal_split") is not None
                and abs(float(self_check.get("mean_normalized_rank_abs_diff_vs_minimal_split"))) <= 1e-12
            ) else "SELF_CHECK_FAIL",
            "gt_count_ok": bool(self_check.get("gt_count_matches_minimal_split")),
            "top1_diff_ok": (
                self_check.get("gt_top1_abs_diff_vs_minimal_split") is not None
                and abs(float(self_check.get("gt_top1_abs_diff_vs_minimal_split"))) <= 1e-12
            ),
            "rank_diff_ok": (
                self_check.get("mean_normalized_rank_abs_diff_vs_minimal_split") is not None
                and abs(float(self_check.get("mean_normalized_rank_abs_diff_vs_minimal_split"))) <= 1e-12
            ),
            "tolerance": 1e-12,
        },
    }


def _formal_authority_failure_payload(*, error: Exception, split: str, existing_formal_summary: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    formal_split_summary = (((existing_formal_summary or {}).get("split_summaries") or {}).get(str(split)) or {}) if isinstance(existing_formal_summary, Mapping) else {}
    payload = {
        "status": "FAIL",
        "formal_split": str(split),
        "formal_gt_count": 0,
        "formal_gt_top1_hit_rate": None,
        "formal_mean_normalized_gt_rank": None,
        "existing_minimal_split": {
            "gt_count": _safe_int(formal_split_summary.get("gt_count"), default=None),
            "gt_top1_hit_rate": _safe_float(formal_split_summary.get("gt_top1_hit_rate"), default=None),
            "mean_normalized_gt_rank": _safe_float(formal_split_summary.get("mean_normalized_gt_rank"), default=None),
        },
        "self_check": {
            "gt_count_matches_minimal_split": False,
            "gt_count_diff_vs_minimal_split": None,
            "gt_top1_abs_diff_vs_minimal_split": None,
            "mean_normalized_rank_abs_diff_vs_minimal_split": None,
        },
        "formal_authority_meta": {
            "authority": "g8_minimal_split_private_row_builder",
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
            "join_policy": "HARD_FAIL_NO_FALLBACK_TO_PROBE_ROWS",
        },
        "formal_authority_status": {"status": "FAIL"},
    }
    return payload, []


def _build_formal_aligned_authority(
    *,
    config: DiagnosisConfig,
    diagnostic_rows: Sequence[Mapping[str, Any]],
    formal_summary: Mapping[str, Any],
    split: str = "base_unobserved",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        device = torch.device(str(config.device))
        ms_config = MinimalSplitAuditConfig(
            dataset_name=str(config.dataset_name),
            output_root=Path(config.run_root).expanduser().resolve(),
            stage=str(config.stage),
            device=device,
            trajectory_source_branch=str(config.trajectory_source_branch),
            all_gt_generate_sidecars_if_missing=False,
            heartbeat_every_rows=512,
            batch_size_rows=max(1, int(config.batch_size)),
            candidate_chunk_size=0,
        )
        prepared = dict(_minimal_materialize_shared_inputs(ms_config))
        rows, cache, vocab_index, candidate_tensor, temperature_tensor, metadata = _minimal_build_rows_and_cache(config=ms_config, prepared=prepared, stage=str(config.stage))
        split_order = _minimal_split_order_for_dataset(str(config.dataset_name))
        split_order_set = set(split_order)
        scored_rows = [
            row
            for row in rows
            if bool(row.get("gt_available_for_audit"))
            and row.get("split") in split_order_set
            and row.get("gt_raw_id_canonical") in vocab_index
            and str(row.get("trajectory_id")) in cache
        ]
        batch_size = max(1, int(config.batch_size))
        for start in range(0, len(scored_rows), batch_size):
            batch_rows = scored_rows[start:start + batch_size]
            normalized, top1 = _minimal_score_batch(
                batch_rows=batch_rows,
                cache=cache,
                candidate_tensor=candidate_tensor,
                temperature_tensor=temperature_tensor,
                vocab_index=vocab_index,
                device=device,
                candidate_chunk_size=0,
            )
            for row, norm_rank, top1_hit in zip(batch_rows, normalized.tolist(), top1.tolist()):
                row["normalized_gt_rank"] = float(norm_rank)
                row["gt_top1_hit_rate"] = float(top1_hit)
        computed_summary = _minimal_summarize_minimal_rows(rows, stage_id=str(config.stage), split_order=split_order)
        formal_joined_rows = _join_formal_authority_rows(formal_rows=scored_rows, diagnostic_rows=diagnostic_rows, split=str(split))
        authority_meta = {
            "authority": "g8_minimal_split_private_row_builder",
            "status": "PASS",
            "join_policy": "minimal_split_rows_are_universe; diagnosis/resp rows are optional joins; unmatched rows are retained; no fallback to probe rows",
            "split": str(split),
            "row_source_path": metadata.get("row_source_path"),
            "minimal_total_scored_rows": int(len(scored_rows)),
            "formal_joined_row_count": int(len(formal_joined_rows)),
            "missing_diagnosis_record_count": int(sum(1 for r in formal_joined_rows if bool(r.get("missing_diagnosis_record")))),
        }
        payload = _formal_aligned_decomposition_from_authority(
            formal_rows=formal_joined_rows,
            existing_formal_summary=formal_summary,
            computed_minimal_summary=computed_summary,
            split=str(split),
            authority_meta=authority_meta,
        )
        return payload, formal_joined_rows
    except Exception as exc:
        return _formal_authority_failure_payload(error=exc, split=str(split), existing_formal_summary=formal_summary)

def _snapshot_extra_ids(row: Mapping[str, Any], *, prefer_mined: bool = False) -> List[int]:
    keys = (
        ["candidate_ids_extra_mined", "candidate_ids_extra"] if prefer_mined else ["candidate_ids_extra", "candidate_ids_extra_mined"]
    )
    keys += ["selected_extra_raw_ids", "extra_ids", "chosen_extra_raw_ids", "extra_raw_ids", "chosen"]
    for key in keys:
        vals = _unique_ints(row.get(key))
        if vals:
            return vals
    return []


def _per_epoch_recall(
    *,
    run_root: Path,
    stage: str,
    target_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    snap_dir = run_root / "train" / stage / "extra_snapshots"
    paths = sorted(snap_dir.glob("epoch_*.jsonl"))
    by_tid_targets = {str(r.get("trajectory_id")): dict(r) for r in target_rows if str(r.get("trajectory_id", ""))}
    by_clip_targets: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        clip_id = _safe_int(row.get("clip_id"))
        if clip_id is not None:
            by_clip_targets[int(clip_id)].append(row)
    epoch_payloads: List[Dict[str, Any]] = []
    tid_hits_by_epoch: Dict[str, List[bool]] = {tid: [] for tid in by_tid_targets}
    for path in paths:
        rows_by_tid: Dict[str, List[int]] = {}
        rows_by_clip: Dict[int, List[int]] = {}
        row_count = 0
        for snap in _iter_jsonl(path):
            row_count += 1
            extras = _snapshot_extra_ids(snap)
            tid = str(snap.get("trajectory_id", "")).strip()
            if tid:
                rows_by_tid[tid] = extras
            clip_id = _safe_int(snap.get("clip_id"))
            if clip_id is not None and extras:
                rows_by_clip[int(clip_id)] = extras
        hits: List[bool] = []
        for tid, target in by_tid_targets.items():
            gt = _safe_int(target.get("gt_raw_id"))
            if gt is None:
                continue
            extras = rows_by_tid.get(tid)
            if extras is None:
                clip_id = _safe_int(target.get("clip_id"))
                extras = rows_by_clip.get(int(clip_id)) if clip_id is not None else None
            hit = bool(extras is not None and int(gt) in {int(x) for x in extras})
            hits.append(hit)
            tid_hits_by_epoch.setdefault(tid, []).append(hit)
        epoch_payloads.append({
            "snapshot_path": str(path),
            "snapshot_id": path.stem,
            "snapshot_row_count": int(row_count),
            "target_count": int(len(hits)),
            "gt_in_extra_candidate_rate": _rate_bools(hits),
        })
    if not paths:
        return {"status": "NO_EXTRA_SNAPSHOTS", "snapshot_dir": str(snap_dir), "epochs": []}
    entered = dropped = persisted = never_entered = final_only = 0
    for _tid, hits in tid_hits_by_epoch.items():
        if not hits:
            continue
        any_hit = any(bool(x) for x in hits)
        if any_hit:
            entered += 1
        if len(hits) >= 2 and any(bool(hits[i]) and not bool(hits[i + 1]) for i in range(len(hits) - 1)):
            dropped += 1
        if hits and all(bool(x) for x in hits):
            persisted += 1
        if not any_hit:
            never_entered += 1
        if hits and bool(hits[-1]) and sum(1 for x in hits if bool(x)) == 1:
            final_only += 1
    denom = max(len(tid_hits_by_epoch), 1)
    return {
        "status": "PASS",
        "snapshot_dir": str(snap_dir),
        "epoch_count": int(len(paths)),
        "epochs": epoch_payloads,
        "persistence": {
            "target_count": int(len(tid_hits_by_epoch)),
            "entered_count": int(entered),
            "entered_rate": float(entered / denom),
            "dropped_count": int(dropped),
            "dropped_rate": float(dropped / denom),
            "persisted_count": int(persisted),
            "persisted_rate": float(persisted / denom),
            "never_entered_count": int(never_entered),
            "never_entered_rate": float(never_entered / denom),
            "final_only_count": int(final_only),
            "final_only_rate": float(final_only / denom),
        },
    }


def run_diagnosis(config: DiagnosisConfig) -> Dict[str, Any]:
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
        raise RuntimeError("no valid examples after phase-1 materialization and carrier filtering")

    stage_overrides, override_meta = _load_stage_responsibility_candidate_overrides(run_root=run_root, stage_id=str(config.stage))
    effective_examples, scope_meta = _apply_stage_candidate_overrides(examples, stage_overrides, stage_id=str(config.stage))
    resp_by_tid, resp_meta = _read_responsibility_rows(run_root, str(config.stage))

    text_vocab_ids, text_records, text_vocab_matrix = load_text_vocab(runtime_output_root)
    vocab_ids = [int(x) for x in text_vocab_ids]
    raw_to_index = {int(raw_id): int(idx) for idx, raw_id in enumerate(vocab_ids)}
    records_by_raw: Dict[int, Mapping[str, Any]] = {}
    for rec in list(text_records):
        rid = _safe_int(rec.get("raw_id", rec.get("id", rec.get("category_id")))) if isinstance(rec, Mapping) else None
        if rid is not None:
            records_by_raw[int(rid)] = dict(rec)
    records_by_raw, class_name_meta = _load_class_name_records(runtime_output_root, str(config.dataset_name), records_by_raw)

    sidecar_lookup = load_gt_sidecar_lookup(
        _sidecar_root(config),
        dataset_name=str(config.dataset_name),
        trajectory_source_branch=str(config.trajectory_source_branch),
    )
    base_vocab_ids, _novel_ids = load_lvvis_base_and_novel_raw_ids()
    base_vocab_set = {int(x) for x in base_vocab_ids}

    checkpoint_path = _default_checkpoint_path(run_root, str(config.stage))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found for stage={config.stage}: {checkpoint_path}")
    device = torch.device(str(config.device))
    projector, theta_t, unknown_prototype, checkpoint_payload = _load_reservoir_checkpoint(checkpoint_path, device=device)
    projector.eval()
    temperature = _compute_t_dis(theta_t).detach()
    logits_vocab, logits_unknown = _score_batches(
        examples=effective_examples,
        projector=projector,
        text_vocab_matrix=np.asarray(text_vocab_matrix, dtype=np.float32),
        unknown_prototype=unknown_prototype,
        temperature=temperature,
        device=device,
        batch_size=max(1, int(config.batch_size)),
        show_progress=bool(config.show_progress),
        stage_id=str(config.stage),
    )

    # Clip-level mining scores: s(v,c)=max_q h(q,c) within each clip.
    indices_by_clip: Dict[int, List[int]] = defaultdict(list)
    yprime_by_clip: Dict[int, set[int]] = defaultdict(set)
    for idx, ex in enumerate(effective_examples):
        clip_id = int(ex.get("clip_id", -1))
        indices_by_clip[clip_id].append(int(idx))
        yprime_by_clip[clip_id].update(int(x) for x in _unique_ints(ex.get("candidate_ids_known")))

    clip_mining: Dict[int, Dict[str, Any]] = {}
    vocab_count = len(vocab_ids)
    for clip_id, row_indices in indices_by_clip.items():
        clip_logits = np.asarray(logits_vocab[np.asarray(row_indices, dtype=np.int64)], dtype=np.float64)
        max_scores = np.max(clip_logits, axis=0) if clip_logits.size else np.full((vocab_count,), -np.inf, dtype=np.float64)
        mask = np.ones((vocab_count,), dtype=bool)
        for raw_id in yprime_by_clip.get(int(clip_id), set()):
            idx = raw_to_index.get(int(raw_id))
            if idx is not None:
                mask[int(idx)] = False
        clip_mining[int(clip_id)] = {
            "scores": max_scores,
            "candidate_mask": mask,
            "top100_raw_ids": _top_raw_ids(max_scores, vocab_ids, mask, max(config.rank_ks)),
        }

    row_diagnostics: List[Record] = []
    selected_extra_counter: Counter = Counter()
    selected_extra_clip_counter: Counter = Counter()
    wrong_extra_winner_counter: Counter = Counter()
    suppressor_counter: Counter = Counter()
    suppressor_affected_gt_counter: Dict[int, Counter] = defaultdict(Counter)
    wrong_extra_winner_affected_gt_counter: Dict[int, Counter] = defaultdict(Counter)
    selected_when_gt_missing_counter: Counter = Counter()

    denom_rank = float(max(len(vocab_ids) - 1, 1))
    current_k_values: List[int] = []
    for row_index, ex in enumerate(effective_examples):
        tid = str(ex.get("trajectory_id", "")).strip()
        resp_row = dict(resp_by_tid.get(tid, {}))
        sidecar = dict(sidecar_lookup.get(tid, {})) if tid else {}
        gt_raw = _canonical_sidecar_gt_raw_id(sidecar) if sidecar else None
        gt_raw_id = int(gt_raw) if gt_raw is not None else None
        gt_idx = raw_to_index.get(int(gt_raw_id)) if gt_raw_id is not None else None
        gt_in_vocab = gt_idx is not None
        observed = _unique_ints(ex.get("observed_raw_ids"))
        known = _unique_ints(ex.get("candidate_ids_known"))
        extra = _unique_ints(ex.get("candidate_ids_extra"))
        mined_extra = _unique_ints(resp_row.get("candidate_ids_extra_mined")) or extra
        current_k_values.append(len(extra))
        for rid in extra:
            selected_extra_counter[int(rid)] += 1
        clip_id = int(ex.get("clip_id", -1))
        for rid in set(extra):
            selected_extra_clip_counter[int(rid)] += 1

        split = None
        if gt_raw_id is not None:
            split = _all_gt_split_label(
                dataset_name=str(config.dataset_name),
                gt_raw_id=int(gt_raw_id),
                observed_raw_ids=observed,
                base_vocab_ids=base_vocab_set,
            )
        known_set = {int(x) for x in known}
        extra_set = {int(x) for x in extra if int(x) not in known_set}
        gt_in_known = bool(gt_raw_id is not None and int(gt_raw_id) in known_set)
        gt_in_extra = bool(gt_raw_id is not None and int(gt_raw_id) in extra_set)
        gt_in_mined_extra = bool(gt_raw_id is not None and int(gt_raw_id) in {int(x) for x in mined_extra})

        logits = np.asarray(logits_vocab[row_index], dtype=np.float64)
        unknown_logit = float(logits_unknown[row_index]) if logits_unknown.size else float("-inf")
        sort_order = np.argsort(-logits, kind="stable")
        final_gt_rank = None
        final_gt_norm = None
        if gt_idx is not None:
            positions = np.where(sort_order == int(gt_idx))[0]
            if positions.size:
                final_gt_rank = int(positions[0]) + 1
                final_gt_norm = float((int(final_gt_rank) - 1) / denom_rank)
        best_vocab_idx = int(sort_order[0]) if len(sort_order) else -1
        best_vocab_raw = int(vocab_ids[best_vocab_idx]) if best_vocab_idx >= 0 else None
        best_vocab_score = float(logits[best_vocab_idx]) if best_vocab_idx >= 0 else float("-inf")
        final_winner_domain = "unknown"
        final_winner_raw_id = None
        if best_vocab_raw is not None and best_vocab_score >= unknown_logit:
            final_winner_raw_id = int(best_vocab_raw)
            if final_winner_raw_id in known_set:
                final_winner_domain = "Yprime"
            elif final_winner_raw_id in extra_set:
                final_winner_domain = "extra"
            else:
                final_winner_domain = "other_nonYprime"
        final_top1_is_gt = bool(gt_raw_id is not None and final_winner_raw_id == int(gt_raw_id))
        if final_winner_domain == "extra" and not final_top1_is_gt and final_winner_raw_id is not None:
            wrong_extra_winner_counter[int(final_winner_raw_id)] += 1
            if gt_raw_id is not None:
                wrong_extra_winner_affected_gt_counter[int(final_winner_raw_id)][int(gt_raw_id)] += 1
        if gt_in_extra and not final_top1_is_gt and final_winner_raw_id is not None:
            suppressor_counter[int(final_winner_raw_id)] += 1
            if gt_raw_id is not None:
                suppressor_affected_gt_counter[int(final_winner_raw_id)][int(gt_raw_id)] += 1
        if not gt_in_extra:
            for rid in extra:
                selected_when_gt_missing_counter[int(rid)] += 1

        masks = {}
        known_idx = [raw_to_index[x] for x in known_set if x in raw_to_index]
        extra_idx = [raw_to_index[x] for x in extra_set if x in raw_to_index]
        known_mask = np.zeros((vocab_count,), dtype=bool)
        extra_mask = np.zeros((vocab_count,), dtype=bool)
        if known_idx:
            known_mask[np.asarray(known_idx, dtype=np.int64)] = True
        if extra_idx:
            extra_mask[np.asarray(extra_idx, dtype=np.int64)] = True
        other_mask = (~known_mask) & (~extra_mask)
        masks["Yprime"] = known_mask
        masks["wrong_extra"] = extra_mask.copy()
        if gt_idx is not None and 0 <= int(gt_idx) < vocab_count:
            masks["wrong_extra"][int(gt_idx)] = False
        masks["other_nonYprime"] = other_mask
        if gt_idx is not None and 0 <= int(gt_idx) < vocab_count:
            masks["other_nonYprime"][int(gt_idx)] = False
        gt_score = float(logits[gt_idx]) if gt_idx is not None else None
        best_y_idx = _best_index(logits, masks["Yprime"])
        best_wrong_extra_idx = _best_index(logits, masks["wrong_extra"])
        best_other_idx = _best_index(logits, masks["other_nonYprime"])
        margin_gt_vs_yprime = float(gt_score - logits[best_y_idx]) if gt_score is not None and best_y_idx is not None else None
        margin_gt_vs_wrong_extra = float(gt_score - logits[best_wrong_extra_idx]) if gt_score is not None and best_wrong_extra_idx is not None else None
        margin_gt_vs_other = float(gt_score - logits[best_other_idx]) if gt_score is not None and best_other_idx is not None else None
        margin_gt_vs_unknown = float(gt_score - unknown_logit) if gt_score is not None else None

        mining = clip_mining.get(int(clip_id), {})
        mining_scores = np.asarray(mining.get("scores", np.zeros((vocab_count,), dtype=np.float64)), dtype=np.float64)
        mining_mask = np.asarray(mining.get("candidate_mask", np.ones((vocab_count,), dtype=bool)), dtype=bool)
        gt_mining_rank = _rank_from_scores(mining_scores, int(gt_idx), mining_mask) if gt_idx is not None else None
        current_k = max(1, len(extra) or len(mined_extra) or 1)
        kth_score = None
        margin_to_enter_topk = None
        if gt_idx is not None and mining_scores.size:
            masked_scores = mining_scores.copy()
            masked_scores[~mining_mask] = -np.inf
            finite_sorted = np.sort(masked_scores[np.isfinite(masked_scores)])[::-1]
            if finite_sorted.size >= current_k:
                kth_score = float(finite_sorted[current_k - 1])
                margin_to_enter_topk = float(mining_scores[int(gt_idx)] - kth_score)
        r_final = dict(resp_row.get("r_final", {})) if isinstance(resp_row.get("r_final"), Mapping) else {}
        r_gt = _r_final_value(r_final, gt_raw_id)
        r_rank_gt = _r_rank(r_final, gt_raw_id)
        r_win = _r_winner(r_final, known, extra, gt_raw_id)
        iou = _extract_gt_iou(sidecar)

        gt_available_for_audit = bool(sidecar.get("audit_usable", False)) and gt_raw_id is not None
        formal_eligible = bool(gt_available_for_audit and split in set(_split_order(str(config.dataset_name))) and gt_idx is not None)

        failure_bucket = "not_base_unobserved_or_no_gt"
        if split == "base_unobserved":
            if not gt_in_extra:
                failure_bucket = "gt_not_in_extra_candidate"
            elif final_top1_is_gt:
                failure_bucket = "success_final_gt_top1"
            elif final_winner_domain == "Yprime":
                failure_bucket = "gt_in_extra_but_Yprime_wins"
            elif final_winner_domain == "extra":
                failure_bucket = "gt_in_extra_but_wrong_extra_wins"
            elif final_winner_domain == "other_nonYprime":
                failure_bucket = "gt_in_extra_but_other_nonYprime_wins"
            elif final_winner_domain == "unknown":
                failure_bucket = "gt_in_extra_but_unknown_wins"
            else:
                failure_bucket = "gt_in_extra_other_failure"

        row_diagnostics.append({
            "trajectory_id": tid,
            "clip_id": int(clip_id),
            "video_id": _safe_int(ex.get("video_id"), -1),
            "split": split,
            "gt_available_for_audit": bool(gt_available_for_audit),
            "formal_eligible": bool(formal_eligible),
            "gt_raw_id": gt_raw_id,
            "gt_in_vocab": bool(gt_in_vocab),
            "gt_in_Yprime": bool(gt_in_known),
            "gt_in_extra": bool(gt_in_extra),
            "gt_in_mined_extra": bool(gt_in_mined_extra),
            "candidate_ids_known": known,
            "candidate_ids_extra": extra,
            "candidate_ids_extra_mined": mined_extra,
            "gt_mining_rank": gt_mining_rank,
            "gt_mining_rank_bucket": _rank_bucket(gt_mining_rank),
            "margin_bucket": _margin_bucket(gt_mining_rank, current_k),
            "gt_mining_score": float(mining_scores[int(gt_idx)]) if gt_idx is not None and mining_scores.size else None,
            "kth_selected_score": kth_score,
            "margin_to_enter_topK": margin_to_enter_topk,
            "final_winner_raw_id": final_winner_raw_id,
            "final_winner_domain": final_winner_domain,
            "final_top1_is_gt": bool(final_top1_is_gt),
            "final_gt_rank": final_gt_rank,
            "final_gt_normalized_rank": final_gt_norm,
            "final_gt_score": gt_score,
            "unknown_logit": float(unknown_logit),
            "margin_gt_vs_Yprime": margin_gt_vs_yprime,
            "margin_gt_vs_wrong_extra": margin_gt_vs_wrong_extra,
            "margin_gt_vs_other_nonYprime": margin_gt_vs_other,
            "margin_gt_vs_unknown": margin_gt_vs_unknown,
            "R_final_gt": float(r_gt),
            "R_final_gt_rank": r_rank_gt,
            "r_final_gt_winner": bool(r_win.get("r_winner_is_gt", False)),
            "r_winner_raw_id": r_win.get("r_winner_raw_id"),
            "r_winner_domain": r_win.get("r_winner_domain"),
            "r_winner_value": r_win.get("r_winner_value"),
            "r_to_logit_transition": (
                "R_GT_winner_to_final_GT_top1" if bool(r_win.get("r_winner_is_gt", False)) and final_top1_is_gt else
                "R_GT_winner_to_final_nonGT" if bool(r_win.get("r_winner_is_gt", False)) and not final_top1_is_gt else
                "R_nonGT_to_final_GT_top1" if (not bool(r_win.get("r_winner_is_gt", False))) and final_top1_is_gt else
                "R_nonGT_to_final_nonGT"
            ),
            "matched_gt_iou": iou,
            "iou_bucket": _iou_bucket(iou),
            "failure_bucket": failure_bucket,
        })

    target_rows = [r for r in row_diagnostics if r.get("split") == "base_unobserved"]
    gt_in_rows = [r for r in target_rows if bool(r.get("gt_in_extra"))]
    gt_not_rows = [r for r in target_rows if not bool(r.get("gt_in_extra"))]
    current_k = int(round(float(np.median(np.asarray(current_k_values, dtype=np.float64))))) if current_k_values else 0

    # Recall@K curve from final checkpoint mining scores.
    recall_curve = {
        "target_split": "base_unobserved",
        "target_count": int(len(target_rows)),
        "actual_active_gt_in_extra_rate": _rate_bools([bool(r.get("gt_in_extra")) for r in target_rows]),
        "actual_mined_gt_in_extra_rate": _rate_bools([bool(r.get("gt_in_mined_extra")) for r in target_rows]),
        "current_effective_k_median": int(current_k),
        "recomputed_final_checkpoint_recall_at_k": {
            str(k): _rate_bools([bool(r.get("gt_mining_rank") is not None and int(r["gt_mining_rank"]) <= int(k)) for r in target_rows])
            for k in config.rank_ks
        },
        "note": "recomputed_final_checkpoint_recall_at_k uses the final checkpoint logits and may differ from epoch-start runtime extra snapshots.",
    }

    rank_bucket_counts = Counter(str(r.get("gt_mining_rank_bucket")) for r in target_rows)
    rank_bucket_payload = {
        "target_count": int(len(target_rows)),
        "mean_rank": _mean([float(r["gt_mining_rank"]) for r in target_rows if r.get("gt_mining_rank") is not None]),
        "median_rank": _median([float(r["gt_mining_rank"]) for r in target_rows if r.get("gt_mining_rank") is not None]),
        "bucket_counts": {k: int(v) for k, v in sorted(rank_bucket_counts.items())},
        "bucket_rates": {k: float(v / max(len(target_rows), 1)) for k, v in sorted(rank_bucket_counts.items())},
    }

    margin_payload = {
        "target_count": int(len(target_rows)),
        "gt_not_in_extra_count": int(len(gt_not_rows)),
        "margin_to_enter_topK_mean_missing_only": _mean([float(r["margin_to_enter_topK"]) for r in gt_not_rows if r.get("margin_to_enter_topK") is not None]),
        "margin_to_enter_topK_p50_missing_only": _quantile([float(r["margin_to_enter_topK"]) for r in gt_not_rows if r.get("margin_to_enter_topK") is not None], 0.50),
        "margin_to_enter_topK_p90_missing_only": _quantile([float(r["margin_to_enter_topK"]) for r in gt_not_rows if r.get("margin_to_enter_topK") is not None], 0.90),
        "near_medium_far_counts": {k: int(v) for k, v in Counter(str(r.get("margin_bucket")) for r in gt_not_rows).items()},
    }

    # Clip oracle coverage.
    target_by_clip: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        target_by_clip[int(row.get("clip_id", -1))].append(row)
    oracle: Dict[str, Any] = {"target_count": int(len(target_rows)), "clip_count": int(len(target_by_clip)), "by_k": {}}
    for k in (1, 2, 3, 4, 5, 10):
        actual_hits = 0
        oracle_hits = 0
        clips_over_capacity = 0
        distinct_counts: List[int] = []
        for _clip, rows in target_by_clip.items():
            gt_counter = Counter(int(r["gt_raw_id"]) for r in rows if r.get("gt_raw_id") is not None)
            distinct_counts.append(len(gt_counter))
            if len(gt_counter) > int(k):
                clips_over_capacity += 1
            oracle_gt = {rid for rid, _count in gt_counter.most_common(int(k))}
            for r in rows:
                if r.get("gt_raw_id") is None:
                    continue
                if bool(r.get("gt_mining_rank") is not None and int(r["gt_mining_rank"]) <= int(k)):
                    actual_hits += 1
                if int(r["gt_raw_id"]) in oracle_gt:
                    oracle_hits += 1
        oracle["by_k"][str(k)] = {
            "actual_recomputed_recall": float(actual_hits / max(len(target_rows), 1)),
            "oracle_recall": float(oracle_hits / max(len(target_rows), 1)),
            "oracle_gap": float((oracle_hits - actual_hits) / max(len(target_rows), 1)),
            "clip_hidden_gt_class_count_gt_k": int(clips_over_capacity),
            "clip_hidden_gt_class_count_gt_k_rate": float(clips_over_capacity / max(len(target_by_clip), 1)),
        }
    oracle["distinct_hidden_gt_classes_per_clip_mean"] = _mean([float(x) for x in distinct_counts])
    oracle["distinct_hidden_gt_classes_per_clip_p95"] = _quantile([float(x) for x in distinct_counts], 0.95)

    per_epoch = _per_epoch_recall(run_root=run_root, stage=str(config.stage), target_rows=target_rows)

    recall_to_top1 = {
        "target_split": "base_unobserved",
        "all": _summarize_rows(target_rows),
        "gt_in_extra": _summarize_rows(gt_in_rows),
        "gt_not_in_extra": _summarize_rows(gt_not_rows),
        "conditional": {
            "P_formal_top1_given_gt_in_extra": _rate_bools([bool(r.get("final_top1_is_gt")) for r in gt_in_rows]),
            "P_R_final_GT_winner_given_gt_in_extra": _rate_bools([bool(r.get("r_final_gt_winner")) for r in gt_in_rows]),
            "P_Yprime_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "Yprime" for r in gt_in_rows]),
            "P_wrong_extra_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "extra" and not bool(r.get("final_top1_is_gt")) for r in gt_in_rows]),
            "P_other_nonYprime_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "other_nonYprime" for r in gt_in_rows]),
            "P_unknown_wins_given_gt_in_extra": _rate_bools([str(r.get("final_winner_domain")) == "unknown" for r in gt_in_rows]),
        },
    }

    transfer_payload = _transfer_payload(target_rows)

    # Existing summaries are included to support self-checks against known probe/audit outputs.
    formal_summary = _load_formal_summary(run_root, str(config.dataset_name), str(config.stage))
    existing_probe = _load_existing_probe_summary(run_root, str(config.dataset_name), str(config.stage))
    formal_aligned, formal_aligned_rows = _build_formal_aligned_authority(
        config=config,
        diagnostic_rows=row_diagnostics,
        formal_summary=formal_summary or {},
        split="base_unobserved",
    )
    formal_aligned_transfer = _transfer_payload(formal_aligned_rows) if formal_aligned_rows else {
        "target_count": 0,
        "transition_counts": {},
        "transition_rates": {},
        "by_transition": {},
        "status": "SKIPPED_NO_FORMAL_AUTHORITY_ROWS",
    }

    hub_payload = {
        "class_name_mapping": class_name_meta,
        "top_selected_extra_classes_by_row": _counter_payload_with_rates(selected_extra_counter, records_by_raw=records_by_raw, top_n=config.top_classes, denominator=len(row_diagnostics)),
        "top_selected_extra_classes_by_clip": _counter_payload_with_rates(selected_extra_clip_counter, records_by_raw=records_by_raw, top_n=config.top_classes, denominator=len(indices_by_clip)),
        "top_wrong_extra_winner_classes": _counter_payload_with_affected_gt(wrong_extra_winner_counter, wrong_extra_winner_affected_gt_counter, records_by_raw=records_by_raw, top_n=config.top_classes, denominator=len(target_rows)),
        "top_suppressor_classes_when_gt_in_extra_fails": _counter_payload_with_affected_gt(suppressor_counter, suppressor_affected_gt_counter, records_by_raw=records_by_raw, top_n=config.top_classes, denominator=len(gt_in_rows)),
        "top_selected_classes_when_gt_missing_from_extra": _counter_payload_with_rates(selected_when_gt_missing_counter, records_by_raw=records_by_raw, top_n=config.top_classes, denominator=len(gt_not_rows)),
    }
    hub_payload["top_selected"] = hub_payload["top_selected_extra_classes_by_row"]
    hub_payload["top_wrong_winners"] = hub_payload["top_wrong_extra_winner_classes"]
    hub_payload["top_suppressors"] = hub_payload["top_suppressor_classes_when_gt_in_extra_fails"]

    iou_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in target_rows:
        iou_groups[str(row.get("iou_bucket"))].append(row)
    iou_payload = {
        "status": "PASS" if any(str(k) != "MISSING_IOU" for k in iou_groups.keys()) else "MISSING_IOU_SIDECAR_OR_FIELDS",
        "buckets": {bucket: _summarize_rows(rows) for bucket, rows in sorted(iou_groups.items())},
    }

    missing_examples = sorted(
        [dict(r) for r in gt_not_rows],
        key=lambda r: (
            999999 if r.get("gt_mining_rank") is None else int(r.get("gt_mining_rank")),
            str(r.get("trajectory_id", "")),
        ),
    )[: max(1, int(config.top_examples))]

    summary = {
        "status": "PASS",
        "run_root": str(run_root),
        "runtime_output_root": str(runtime_output_root),
        "dataset_name": str(config.dataset_name),
        "stage": str(config.stage),
        "trajectory_source_branch": str(config.trajectory_source_branch),
        "checkpoint_path": str(checkpoint_path),
        "materialization_stats": dict(materialized.get("stats", {})),
        "candidate_authority": {"responsibility_overrides": dict(override_meta), "effective_scope": dict(scope_meta), "responsibility_records": dict(resp_meta)},
        "checkpoint_payload_meta": {
            "stage_id": checkpoint_payload.get("stage_id"),
            "epoch": checkpoint_payload.get("epoch"),
            "seed": checkpoint_payload.get("seed"),
            "pipeline": checkpoint_payload.get("pipeline"),
            "em_subiterations": checkpoint_payload.get("em_subiterations"),
            "extra_selection_mode": checkpoint_payload.get("extra_selection_mode"),
            "k_extra": checkpoint_payload.get("k_extra"),
            "extra_coverage_mode": checkpoint_payload.get("extra_coverage_mode"),
            "extra_coverage_scale": checkpoint_payload.get("extra_coverage_scale"),
        },
        "target_split": "base_unobserved",
        "target_count": int(len(target_rows)),
        "actual_gt_in_extra_candidate_rate": recall_curve["actual_active_gt_in_extra_rate"],
        "recomputed_final_checkpoint_recall_at_k": recall_curve["recomputed_final_checkpoint_recall_at_k"],
        "P_formal_top1_given_gt_in_extra": recall_to_top1["conditional"]["P_formal_top1_given_gt_in_extra"],
        "P_R_final_GT_winner_given_gt_in_extra": recall_to_top1["conditional"]["P_R_final_GT_winner_given_gt_in_extra"],
        "failure_bucket_histogram": recall_to_top1["all"]["failure_bucket_histogram"],
        "formal_aligned_summary": formal_aligned,
        "class_name_mapping": class_name_meta,
        "formal_summary_existing": formal_summary,
        "extra_probe_summary_existing": existing_probe,
        "self_check": {
            "gt_in_extra_rate_vs_existing_probe": {
                "diagnosis": recall_curve["actual_active_gt_in_extra_rate"],
                "existing_probe_base_unobserved": (((existing_probe.get("summary") or {}).get("by_split") or {}).get("base_unobserved") or {}).get("gt_in_extra_candidate_rate") if isinstance(existing_probe, Mapping) else None,
            }
        },
    }

    takeaways = []
    takeaways.append("# Extra Mining Recall-to-Top1 Diagnosis")
    takeaways.append("")
    takeaways.append(f"- run_root: `{run_root}`")
    takeaways.append(f"- stage: `{config.stage}`")
    takeaways.append(f"- target split: `base_unobserved`, count `{len(target_rows)}`")
    takeaways.append(f"- actual GT-in-extra rate: `{recall_curve['actual_active_gt_in_extra_rate']}`")
    takeaways.append(f"- P(final GT top1 | GT in extra): `{recall_to_top1['conditional']['P_formal_top1_given_gt_in_extra']}`")
    takeaways.append(f"- P(R_final GT winner | GT in extra): `{recall_to_top1['conditional']['P_R_final_GT_winner_given_gt_in_extra']}`")
    takeaways.append(f"- formal-aligned base_unobserved gt_count: `{formal_aligned.get('formal_gt_count')}`")
    takeaways.append(f"- formal-aligned authority status: `{formal_aligned.get('formal_authority_status', {}).get('status')}`")
    takeaways.append(f"- formal-aligned top1 self-check diff: `{formal_aligned.get('self_check', {}).get('gt_top1_abs_diff_vs_minimal_split')}`")
    takeaways.append("")
    takeaways.append("## Recall@K from final checkpoint mining scores")
    for k, v in recall_curve["recomputed_final_checkpoint_recall_at_k"].items():
        takeaways.append(f"- recall@{k}: `{v}`")
    takeaways.append("")
    takeaways.append("## Failure histogram")
    for key, val in recall_to_top1["all"]["failure_bucket_histogram"].items():
        takeaways.append(f"- {key}: `{val}`")
    takeaways.append("")
    takeaways.append("## R-to-logit transition")
    for key, val in transfer_payload["transition_counts"].items():
        takeaways.append(f"- {key}: `{val}`")

    files = {
        "summary": output_dir / "summary.json",
        "recall_at_k_curve": output_dir / "recall_at_k_curve.json",
        "gt_mining_rank_buckets": output_dir / "gt_mining_rank_buckets.json",
        "margin_to_enter_topk": output_dir / "margin_to_enter_topk.json",
        "clip_oracle_coverage": output_dir / "clip_oracle_coverage.json",
        "per_epoch_extra_recall": output_dir / "per_epoch_extra_recall.json",
        "recall_to_top1_decomposition": output_dir / "recall_to_top1_decomposition.json",
        "r_to_logit_transfer_gap": output_dir / "r_to_logit_transfer_gap.json",
        "wrong_extra_hub_report": output_dir / "wrong_extra_hub_report.json",
        "wrong_extra_hub_report_named": output_dir / "wrong_extra_hub_report_named.json",
        "class_id_name_map_used": output_dir / "class_id_name_map_used.json",
        "formal_aligned_summary": output_dir / "formal_aligned_summary.json",
        "formal_aligned_recall_to_top1_decomposition": output_dir / "formal_aligned_recall_to_top1_decomposition.json",
        "formal_aligned_r_to_logit_transfer_gap": output_dir / "formal_aligned_r_to_logit_transfer_gap.json",
        "formal_aligned_row_diagnostics": output_dir / "formal_aligned_row_diagnostics.jsonl",
        "top_selected_extra_classes_named": output_dir / "top_selected_extra_classes_named.json",
        "top_wrong_extra_winner_classes_named": output_dir / "top_wrong_extra_winner_classes_named.json",
        "top_gt_suppressor_classes_named": output_dir / "top_gt_suppressor_classes_named.json",
        "iou_bucket_report": output_dir / "iou_bucket_report.json",
        "missing_gt_examples": output_dir / "missing_gt_examples.jsonl",
        "row_diagnostics": output_dir / "row_diagnostics.jsonl",
        "diagnosis_takeaways": output_dir / "diagnosis_takeaways.md",
    }
    _write_json(files["summary"], summary)
    _write_json(files["recall_at_k_curve"], recall_curve)
    _write_json(files["gt_mining_rank_buckets"], rank_bucket_payload)
    _write_json(files["margin_to_enter_topk"], margin_payload)
    _write_json(files["clip_oracle_coverage"], oracle)
    _write_json(files["per_epoch_extra_recall"], per_epoch)
    _write_json(files["recall_to_top1_decomposition"], recall_to_top1)
    _write_json(files["r_to_logit_transfer_gap"], transfer_payload)
    _write_json(files["wrong_extra_hub_report"], hub_payload)
    _write_json(files["wrong_extra_hub_report_named"], hub_payload)
    _write_json(files["class_id_name_map_used"], {"meta": class_name_meta, "records_by_raw": {str(k): dict(v) for k, v in sorted(records_by_raw.items())}})
    _write_json(files["formal_aligned_summary"], formal_aligned)
    _write_json(files["formal_aligned_recall_to_top1_decomposition"], formal_aligned)
    _write_json(files["formal_aligned_r_to_logit_transfer_gap"], formal_aligned_transfer)
    _write_jsonl(files["formal_aligned_row_diagnostics"], formal_aligned_rows)
    _write_json(files["top_selected_extra_classes_named"], {"rows": hub_payload["top_selected_extra_classes_by_row"]})
    _write_json(files["top_wrong_extra_winner_classes_named"], {"rows": hub_payload["top_wrong_extra_winner_classes"]})
    _write_json(files["top_gt_suppressor_classes_named"], {"rows": hub_payload["top_suppressor_classes_when_gt_in_extra_fails"]})
    _write_json(files["iou_bucket_report"], iou_payload)
    _write_jsonl(files["missing_gt_examples"], missing_examples)
    _write_jsonl(files["row_diagnostics"], row_diagnostics)
    _write_text(files["diagnosis_takeaways"], "\n".join(takeaways))

    return {"status": "PASS", "output_dir": str(output_dir), "summary_path": str(files["summary"]), "files": {k: str(v) for k, v in files.items()}}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose why GT-in-extra recall does or does not transfer to trajectory-level top1")
    p.add_argument("--run_root", required=True)
    p.add_argument("--runtime_output_root", default=None)
    p.add_argument("--dataset_name", default="lvvis_train_base", choices=("lvvis_train_base", "lvvis_val"))
    p.add_argument("--trajectory_source_branch", default="mainline", choices=("mainline", "gt_upper_bound"))
    p.add_argument("--stage", default="softem_aug", choices=("softem_aug",))
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--rank_ks", type=_parse_rank_ks, default=DEFAULT_RANK_KS)
    p.add_argument("--top_examples", type=int, default=256)
    p.add_argument("--top_classes", type=int, default=50)
    p.add_argument("--smoke", type=_parse_bool, default=False)
    p.add_argument("--smoke_max_trajectories", type=int, default=128)
    p.add_argument("--subset_fraction", type=float, default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--sidecar_root", default=None)
    p.add_argument("--repo_root", default=None)
    p.add_argument("--show_progress", type=_parse_bool, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    runtime_output_root = Path(args.runtime_output_root).expanduser().resolve() if args.runtime_output_root else repo_root
    config = DiagnosisConfig(
        run_root=Path(args.run_root).expanduser().resolve(),
        runtime_output_root=runtime_output_root,
        dataset_name=str(args.dataset_name),
        trajectory_source_branch=str(args.trajectory_source_branch),
        stage=str(args.stage),
        device=str(args.device),
        output_dir=Path(args.output_dir).expanduser().resolve() if args.output_dir else None,
        sidecar_root=Path(args.sidecar_root).expanduser().resolve() if args.sidecar_root else None,
        batch_size=max(1, int(args.batch_size)),
        rank_ks=tuple(args.rank_ks),
        top_examples=max(1, int(args.top_examples)),
        top_classes=max(1, int(args.top_classes)),
        smoke=bool(args.smoke),
        smoke_max_trajectories=max(1, int(args.smoke_max_trajectories)),
        subset_fraction=None if args.subset_fraction is None else float(args.subset_fraction),
        show_progress=bool(args.show_progress),
    )
    result = run_diagnosis(config)
    print(result["summary_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
