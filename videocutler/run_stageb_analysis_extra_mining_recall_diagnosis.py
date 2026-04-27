from __future__ import annotations

import argparse
import csv
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
    emit_failure_taxonomy: bool
    emit_active_raw_conversion: bool
    emit_same_vs_other_hijack: bool
    emit_text_semantic_confusion: bool
    emit_hub_prior_beta_sweep: bool
    hub_raw_ids: Tuple[int, ...]
    hub_beta_values: Tuple[float, ...]
    text_neighbor_topk: int
    text_neighbor_sim_threshold: float
    emit_hub_formation_timeline: bool
    emit_gt_cooccurrence: bool
    emit_weak_label_cooccurrence: bool
    emit_fully_missed_class_report: bool
    emit_fully_missed_trajectory_weighted_report: bool
    emit_hub_collapse_rescue_audit: bool
    emit_annotation_non_gt_hub_rescue_audit: bool
    emit_full_class_cooccurrence: bool
    hub_collapse_risk_threshold: float
    hub_collapse_low_alone_threshold: float
    hub_collapse_top_examples: int
    strong_hub_cooccurrence_threshold: float
    weak_unobservable_present_threshold: float
    weak_unobservable_alone_threshold: float


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


def _parse_int_tuple(value: str) -> Tuple[int, ...]:
    items: List[int] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return tuple(sorted(set(items)))


def _parse_float_tuple(value: str) -> Tuple[float, ...]:
    items: List[float] = []
    for part in str(value).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(float(part))
    return tuple(items)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = [dict(r) for r in rows]
    if not rows_list:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows_list:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)


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


def _weak_vocab_raw_ids_from_examples(examples: Sequence[Mapping[str, Any]]) -> set[int]:
    return {
        int(raw_id)
        for ex in examples
        for raw_id in _unique_ints(ex.get("observed_raw_ids"))
    }


def _gt_raw_id_from_diag_row(row: Mapping[str, Any]) -> Optional[int]:
    for key in ("gt_raw_id", "matched_gt_raw_id_canonical", "gt_id", "gt_class_id"):
        val = _safe_int(row.get(key))
        if val is not None:
            return int(val)
    return None


def _reachable_unobserved_payload(rows: Sequence[Mapping[str, Any]], *, weak_vocab_raw_ids: set[int]) -> Dict[str, Any]:
    reachable = [r for r in rows if (_gt_raw_id_from_diag_row(r) in weak_vocab_raw_ids)]
    unreachable = [r for r in rows if (_gt_raw_id_from_diag_row(r) not in weak_vocab_raw_ids)]
    gt_in_reachable = [r for r in reachable if bool(r.get("gt_in_extra"))]
    gt_in_unreachable = [r for r in unreachable if bool(r.get("gt_in_extra"))]
    return {
        "status": "PASS",
        "scope_semantics": "base_unobserved_reachable keeps the original base_unobserved split and adds gt_raw_id in union(Yprime) from weak labels.",
        "weak_vocab_count": int(len(weak_vocab_raw_ids)),
        "overall": _summarize_rows(rows),
        "base_unobserved_reachable": {
            **_summarize_rows(reachable),
            "P_top1_given_gt_in_extra": _rate_bools([bool(r.get("final_top1_is_gt")) for r in gt_in_reachable]),
            "P_R_gt_winner_given_gt_in_extra": _rate_bools([bool(r.get("r_final_gt_winner")) for r in gt_in_reachable if r.get("r_final_gt_winner") is not None]),
            "gt_in_extra_count": int(len(gt_in_reachable)),
            "unique_gt_class_count": int(len({_gt_raw_id_from_diag_row(r) for r in reachable if _gt_raw_id_from_diag_row(r) is not None})),
        },
        "base_unobserved_unreachable_audit_only": {
            **_summarize_rows(unreachable),
            "P_top1_given_gt_in_extra": _rate_bools([bool(r.get("final_top1_is_gt")) for r in gt_in_unreachable]),
            "P_R_gt_winner_given_gt_in_extra": _rate_bools([bool(r.get("r_final_gt_winner")) for r in gt_in_unreachable if r.get("r_final_gt_winner") is not None]),
            "gt_in_extra_count": int(len(gt_in_unreachable)),
            "unique_gt_class_count": int(len({_gt_raw_id_from_diag_row(r) for r in unreachable if _gt_raw_id_from_diag_row(r) is not None})),
            "interpretation": "audit only; not the primary hidden recovery metric under observed-plus-reachable protocol",
        },
    }


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


def _active_extra_set(row: Mapping[str, Any]) -> set[int]:
    known = {int(x) for x in _unique_ints(row.get("candidate_ids_known"))}
    extra = {int(x) for x in _unique_ints(row.get("candidate_ids_extra"))}
    return {int(x) for x in extra if int(x) not in known}


def _active_raw_contains(row: Mapping[str, Any]) -> Optional[bool]:
    gt = _safe_int(row.get("gt_raw_id"))
    if gt is None:
        return None
    return bool(int(gt) in _active_extra_set(row))


def _rows_joined(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [r for r in rows if not bool(r.get("missing_diagnosis_record"))]



def _active_raw_conversion_payload(rows: Sequence[Mapping[str, Any]], *, hub_raw_ids: Sequence[int]) -> Dict[str, Any]:
    joined = _rows_joined(rows)
    active_rows = [r for r in joined if _active_raw_contains(r) is True]
    inactive_rows = [r for r in joined if _active_raw_contains(r) is False]
    hub_set = {int(x) for x in hub_raw_ids}

    def _domain_rate(domain: str, seq: Sequence[Mapping[str, Any]]) -> Optional[float]:
        return _rate_bools([str(r.get("final_winner_domain")) == str(domain) for r in seq])

    def _hub_selected_rate(seq: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not hub_set:
            return None
        return _rate_bools([bool(_active_extra_set(r) & hub_set) for r in seq])

    def _hub_top1_rate(seq: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not hub_set:
            return None
        return _rate_bools([_safe_int(r.get("final_winner_raw_id")) in hub_set for r in seq])

    def _hub_wrong_winner_rate(seq: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not hub_set:
            return None
        return _rate_bools([
            (_safe_int(r.get("final_winner_raw_id")) in hub_set) and (not bool(r.get("final_top1_is_gt")))
            for r in seq
        ])

    def _hub_suppressor_rate(seq: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not hub_set:
            return None
        return _rate_bools([_safe_int(r.get("top_suppressor_raw_id")) in hub_set for r in seq])

    def _payload(seq: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return {
            "count": int(len(seq)),
            "P_top1": _rate_bools([bool(r.get("final_top1_is_gt")) for r in seq]),
            "P_R_final_GT_winner": _rate_bools([bool(r.get("r_final_gt_winner")) for r in seq if r.get("r_final_gt_winner") is not None]),
            "P_Yprime_wins": _domain_rate("Yprime", seq),
            "P_wrong_extra_wins": _rate_bools([str(r.get("final_winner_domain")) == "extra" and not bool(r.get("final_top1_is_gt")) for r in seq]),
            "P_other_nonYprime_wins": _domain_rate("other_nonYprime", seq),
            "P_unknown_wins": _domain_rate("unknown", seq),
            "P_hub_selected": _hub_selected_rate(seq),
            "P_hub_top1": _hub_top1_rate(seq),
            "P_hub_wrong_winner": _hub_wrong_winner_rate(seq),
            "P_hub_suppressor": _hub_suppressor_rate(seq),
            "failure_bucket_histogram": dict(Counter(str(r.get("failure_bucket", "")) for r in seq)),
        }

    active_payload = _payload(active_rows)
    inactive_payload = _payload(inactive_rows)
    return {
        "status": "PASS",
        "definition": "active_raw_contains := gt_raw_id in (candidate_ids_extra - candidate_ids_known)",
        "hub_metric_definition": {
            "P_hub_selected": "hub raw id appears in active_extra := candidate_ids_extra - candidate_ids_known",
            "P_hub_top1": "final_winner_raw_id is one of hub_raw_ids",
            "P_hub_wrong_winner": "final_winner_raw_id is one of hub_raw_ids and final_top1_is_gt is false",
            "P_hub_suppressor": "top_suppressor_raw_id is one of hub_raw_ids",
            "deprecated": "P_hub_wins is intentionally not emitted because it conflated multiple meanings.",
        },
        "count": int(len(rows)),
        "total": int(len(rows)),
        "diagnosis_joined_count": int(len(joined)),
        "active_raw_contains_count": int(len(active_rows)),
        "active_raw_missing_count": int(len(inactive_rows)),
        "active_raw_membership_rate": float(len(active_rows) / max(len(joined), 1)),
        "old_gt_in_extra_true_count": int(sum(1 for r in joined if bool(r.get("gt_in_extra")))),
        "old_gt_in_extra_rate": _rate_bools([bool(r.get("gt_in_extra")) for r in joined if r.get("gt_in_extra") is not None]),
        "bool_false_but_raw_contains": int(sum(1 for r in joined if (not bool(r.get("gt_in_extra"))) and _active_raw_contains(r) is True)),
        "bool_true_but_raw_not_contains": int(sum(1 for r in joined if bool(r.get("gt_in_extra")) and _active_raw_contains(r) is False)),
        "active_raw": active_payload,
        "active_raw_missing": inactive_payload,
        "P_top1_given_active_raw": active_payload.get("P_top1"),
        "P_R_GT_winner_given_active_raw": active_payload.get("P_R_final_GT_winner"),
        "P_Yprime_wins_given_active_raw": active_payload.get("P_Yprime_wins"),
        "P_wrong_extra_wins_given_active_raw": active_payload.get("P_wrong_extra_wins"),
        "P_other_nonYprime_wins_given_active_raw": active_payload.get("P_other_nonYprime_wins"),
        "P_unknown_wins_given_active_raw": active_payload.get("P_unknown_wins"),
        "P_hub_selected_given_active_raw": active_payload.get("P_hub_selected"),
        "P_hub_top1_given_active_raw": active_payload.get("P_hub_top1"),
        "P_hub_wrong_winner_given_active_raw": active_payload.get("P_hub_wrong_winner"),
        "P_hub_suppressor_given_active_raw": active_payload.get("P_hub_suppressor"),
    }


def _same_vs_other_payload(rows: Sequence[Mapping[str, Any]], *, records_by_raw: Mapping[int, Mapping[str, Any]], top_n: int) -> Dict[str, Any]:
    joined = _rows_joined(rows)
    failures = [r for r in joined if not bool(r.get("final_top1_is_gt")) and _safe_int(r.get("top_suppressor_raw_id")) is not None]
    same = [r for r in failures if bool(r.get("same_trajectory_confusion"))]
    other = [r for r in failures if bool(r.get("other_trajectory_hijack"))]
    mixed = [r for r in failures if bool(r.get("mixed_confusion"))]
    overlap_count = sum(
        1 for r in failures
        if int(bool(r.get("same_trajectory_confusion"))) + int(bool(r.get("other_trajectory_hijack"))) + int(bool(r.get("mixed_confusion"))) > 1
    )
    assigned_count = len(same) + len(other) + len(mixed)
    unassigned_count = max(0, len(failures) - assigned_count)
    suppressor_counter = Counter(_safe_int(r.get("top_suppressor_raw_id")) for r in failures if _safe_int(r.get("top_suppressor_raw_id")) is not None)
    same_counter = Counter(_safe_int(r.get("top_suppressor_raw_id")) for r in same if _safe_int(r.get("top_suppressor_raw_id")) is not None)
    other_counter = Counter(_safe_int(r.get("top_suppressor_raw_id")) for r in other if _safe_int(r.get("top_suppressor_raw_id")) is not None)
    mixed_counter = Counter(_safe_int(r.get("top_suppressor_raw_id")) for r in mixed if _safe_int(r.get("top_suppressor_raw_id")) is not None)
    return {
        "status": "PASS" if overlap_count == 0 else "SELF_CHECK_FAIL",
        "definition": {
            "same_trajectory_confusion": "local_score(q,h) > local_score(q,g) and clip_argmax_traj(v,h) == q",
            "mixed_confusion": "local_score(q,h) > local_score(q,g) and clip_argmax_traj(v,h) != q",
            "other_trajectory_hijack": "clip_argmax_traj(v,h) != q and local_score(q,h) <= local_score(q,g)",
            "note": "same/mixed/other are mutually exclusive after the taxonomy fix.",
        },
        "target_count": int(len(joined)),
        "non_gt_winner_count": int(len(failures)),
        "exclusive_self_check": {
            "overlap_count": int(overlap_count),
            "assigned_count": int(assigned_count),
            "unassigned_count": int(unassigned_count),
            "assigned_rate": float(assigned_count / max(len(failures), 1)),
        },
        "same_trajectory_confusion_count": int(len(same)),
        "same_trajectory_confusion_rate": float(len(same) / max(len(failures), 1)),
        "other_trajectory_hijack_count": int(len(other)),
        "other_trajectory_hijack_rate": float(len(other) / max(len(failures), 1)),
        "mixed_confusion_count": int(len(mixed)),
        "mixed_confusion_rate": float(len(mixed) / max(len(failures), 1)),
        "top_suppressor_classes": _counter_payload_with_rates(suppressor_counter, records_by_raw=records_by_raw, top_n=top_n, denominator=len(failures)),
        "top_same_trajectory_classes": _counter_payload_with_rates(same_counter, records_by_raw=records_by_raw, top_n=top_n, denominator=len(same)),
        "top_other_hijack_classes": _counter_payload_with_rates(other_counter, records_by_raw=records_by_raw, top_n=top_n, denominator=len(other)),
        "top_mixed_confusion_classes": _counter_payload_with_rates(mixed_counter, records_by_raw=records_by_raw, top_n=top_n, denominator=len(mixed)),
    }

def _rank_bucket_payload_by_class(rows: Sequence[Mapping[str, Any]], *, records_by_raw: Mapping[int, Mapping[str, Any]], min_count: int = 1) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in _rows_joined(rows):
        gt = _safe_int(row.get("gt_raw_id"))
        if gt is not None:
            grouped[int(gt)].append(row)
    out: List[Dict[str, Any]] = []
    for gt, seq in sorted(grouped.items()):
        if len(seq) < int(min_count):
            continue
        buckets = Counter(str(r.get("margin_bucket", _margin_bucket(_safe_int(r.get("gt_mining_rank")), 3))) for r in seq)
        item = _class_label(gt, records_by_raw)
        ranks = [float(r["gt_mining_rank"]) for r in seq if r.get("gt_mining_rank") is not None]
        item.update({
            "gt_count": int(len(seq)),
            "in_topK_count": int(buckets.get("in_topK", 0)),
            "near_miss_Kplus1_Kplus2_count": int(buckets.get("near_miss_Kplus1_Kplus2", 0)),
            "medium_miss_6_20_count": int(buckets.get("medium_miss_6_20", 0)),
            "far_miss_gt20_count": int(buckets.get("far_miss_gt20", 0)),
            "missing_or_not_ranked_count": int(buckets.get("missing_or_not_ranked", 0)),
            "active_raw_membership_rate": _rate_bools([_active_raw_contains(r) is True for r in seq if _active_raw_contains(r) is not None]),
            "top1_rate": _rate_bools([bool(r.get("final_top1_is_gt")) for r in seq]),
            "mean_gt_mining_rank": _mean(ranks),
            "median_gt_mining_rank": _median(ranks),
            "rank_p90": _quantile(ranks, 0.9),
            "mean_R_final_gt": _mean([float(r.get("R_final_gt", 0.0)) for r in seq if r.get("R_final_gt") is not None]),
            "top_suppressor_raw_id": None,
            "top_suppressor_name": None,
            "blind_spot_type": None,
        })
        suppressors = Counter(_safe_int(r.get("top_suppressor_raw_id")) for r in seq if _safe_int(r.get("top_suppressor_raw_id")) is not None)
        if suppressors:
            sid, _count = suppressors.most_common(1)[0]
            item["top_suppressor_raw_id"] = int(sid)
            item["top_suppressor_name"] = _class_label(int(sid), records_by_raw).get("name")
        active_rate = item.get("active_raw_membership_rate")
        median_rank = item.get("median_gt_mining_rank")
        near_rate = float(item["near_miss_Kplus1_Kplus2_count"] / max(len(seq), 1))
        if active_rate is not None and float(active_rate) < 0.25 and (median_rank is None or float(median_rank) > 20):
            item["blind_spot_type"] = "fully_missed_blind_spot"
        elif item.get("top_suppressor_raw_id") == 773 and active_rate is not None and float(active_rate) < 0.50:
            item["blind_spot_type"] = "hub_suppressed_blind_spot"
        elif near_rate >= 0.25:
            item["blind_spot_type"] = "near_miss_capacity_limited"
        else:
            item["blind_spot_type"] = "mixed_or_uncategorized"
        out.append(item)
    out.sort(key=lambda r: (str(r.get("blind_spot_type")), -(int(r.get("gt_count", 0))), float(r.get("active_raw_membership_rate") or 0.0)))
    return out




def _class_cooccurrence_payload_from_units(
    *,
    units: Sequence[Mapping[str, Any]],
    target_raw_ids: Sequence[int],
    hub_raw_ids: Sequence[int],
    records_by_raw: Mapping[int, Mapping[str, Any]],
    top_n: int,
    source: str,
    unit_level: str,
) -> Dict[str, Any]:
    target_set = {int(x) for x in target_raw_ids}
    hub_set = {int(x) for x in hub_raw_ids}
    class_sets: List[set[int]] = []
    for unit in units:
        vals = {int(x) for x in _unique_ints(unit.get("class_ids"))}
        if vals:
            class_sets.append(vals)
    total = len(class_sets)
    per_class: List[Dict[str, Any]] = []
    for cid in sorted(target_set):
        present_units = [cs for cs in class_sets if int(cid) in cs]
        present = len(present_units)
        alone_count = 0
        with_other_count = 0
        with_any_hub_count = 0
        num_classes_values: List[float] = []
        hub_counter: Counter = Counter()
        co_counter: Counter = Counter()
        for cs in present_units:
            num_classes_values.append(float(len(cs)))
            others = set(cs) - {int(cid)}
            if not others:
                alone_count += 1
            else:
                with_other_count += 1
            hub_hits = others & hub_set
            if hub_hits:
                with_any_hub_count += 1
            for h in sorted(hub_hits):
                hub_counter[int(h)] += 1
            for other in sorted(others):
                co_counter[int(other)] += 1
        top_hubs: List[Dict[str, Any]] = []
        for hid, cnt in hub_counter.most_common(max(1, int(top_n))):
            hitem = _class_label(int(hid), records_by_raw)
            hitem["count"] = int(cnt)
            hitem["P_hub_given_class"] = float(cnt / max(present, 1)) if present else None
            hitem["P_class_given_hub"] = None
            hub_present = sum(1 for cs in class_sets if int(hid) in cs)
            if hub_present:
                hitem["P_class_given_hub"] = float(cnt / max(hub_present, 1))
            top_hubs.append(hitem)
        top_co: List[Dict[str, Any]] = []
        for oid, cnt in co_counter.most_common(max(1, int(top_n))):
            oitem = _class_label(int(oid), records_by_raw)
            oitem["count"] = int(cnt)
            oitem["P_other_given_class"] = float(cnt / max(present, 1)) if present else None
            other_present = sum(1 for cs in class_sets if int(oid) in cs)
            oitem["P_class_given_other"] = float(cnt / max(other_present, 1)) if other_present else None
            top_co.append(oitem)
        max_hub_raw_id = None
        max_hub_name = None
        max_p_hub_given_class = None
        if top_hubs:
            max_hub_raw_id = _safe_int(top_hubs[0].get("raw_id"))
            max_hub_name = top_hubs[0].get("name")
            max_p_hub_given_class = _safe_float(top_hubs[0].get("P_hub_given_class"))
        person_count = int(hub_counter.get(773, 0))
        item = _class_label(int(cid), records_by_raw)
        item.update({
            "present_count": int(present),
            "present_rate": float(present / max(total, 1)),
            "alone_count": int(alone_count),
            "alone_rate": float(alone_count / max(present, 1)) if present else None,
            "with_other_classes_count": int(with_other_count),
            "with_other_classes_rate": float(with_other_count / max(present, 1)) if present else None,
            "with_any_hub_count": int(with_any_hub_count),
            "with_any_hub_rate": float(with_any_hub_count / max(present, 1)) if present else None,
            "mean_num_classes_when_present": _mean(num_classes_values),
            "P_person_given_class": float(person_count / max(present, 1)) if present else None,
            "person_cooccurrence_count": int(person_count),
            "max_P_hub_given_class": max_p_hub_given_class,
            "max_cooccurring_hub_raw_id": max_hub_raw_id,
            "max_cooccurring_hub_name": max_hub_name,
            "top_cooccurring_hubs": top_hubs,
            "top_cooccurring_classes": top_co,
        })
        per_class.append(item)
    return {
        "status": "PASS" if total else "NO_UNITS",
        "source": str(source),
        "unit_level": str(unit_level),
        "definition": "Each unit contributes a set of raw class ids; target-class co-occurrence is measured against hub_raw_ids and all other classes within the same unit.",
        "target_raw_ids": [int(x) for x in sorted(target_set)],
        "hub_raw_ids": [int(x) for x in sorted(hub_set)],
        "unit_count": int(total),
        "per_class": per_class,
    }


def _cooccurrence_map(payload: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    if not isinstance(payload, Mapping):
        return out
    for item in payload.get("per_class", []) or []:
        if not isinstance(item, Mapping):
            continue
        rid = _safe_int(item.get("raw_id"))
        if rid is not None:
            out[int(rid)] = item
    return out


def _fully_missed_class_report_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    blind_spot_rows: Sequence[Mapping[str, Any]],
    gt_class_cooccurrence: Mapping[str, Any],
    weak_class_cooccurrence: Mapping[str, Any],
    hub_raw_ids: Sequence[int],
    records_by_raw: Mapping[int, Mapping[str, Any]],
    strong_hub_cooccurrence_threshold: float,
    weak_unobservable_present_threshold: float,
    weak_unobservable_alone_threshold: float,
) -> List[Dict[str, Any]]:
    joined = _rows_joined(rows)
    fully_ids = {
        int(r["raw_id"])
        for r in blind_spot_rows
        if str(r.get("blind_spot_type")) == "fully_missed_blind_spot" and _safe_int(r.get("raw_id")) is not None
    }
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in joined:
        gt = _safe_int(row.get("gt_raw_id"))
        if gt is not None and int(gt) in fully_ids:
            grouped[int(gt)].append(row)

    gt_co_map = _cooccurrence_map(gt_class_cooccurrence)
    weak_co_map = _cooccurrence_map(weak_class_cooccurrence)
    hub_set = {int(x) for x in hub_raw_ids}
    out: List[Dict[str, Any]] = []
    for gt in sorted(fully_ids):
        seq = grouped.get(int(gt), [])
        label = _class_label(int(gt), records_by_raw)
        gt_co = gt_co_map.get(int(gt), {})
        weak_co = weak_co_map.get(int(gt), {})
        ranks = [_safe_int(r.get("gt_mining_rank")) for r in seq]
        ranks_float = [float(r) for r in ranks if r is not None]
        buckets = Counter(_margin_bucket(r, 3) for r in ranks)
        suppressors = Counter(_safe_int(r.get("top_suppressor_raw_id")) for r in seq if _safe_int(r.get("top_suppressor_raw_id")) is not None)
        final_winners = Counter(_safe_int(r.get("final_winner_raw_id")) for r in seq if _safe_int(r.get("final_winner_raw_id")) is not None)
        top_suppressor_raw_id = None
        top_suppressor_name = None
        top_suppressor_count = 0
        if suppressors:
            top_suppressor_raw_id, top_suppressor_count = suppressors.most_common(1)[0]
            top_suppressor_name = _class_label(int(top_suppressor_raw_id), records_by_raw).get("name")
        top_winner_raw_id = None
        top_winner_name = None
        top_winner_count = 0
        if final_winners:
            top_winner_raw_id, top_winner_count = final_winners.most_common(1)[0]
            top_winner_name = _class_label(int(top_winner_raw_id), records_by_raw).get("name")

        gt_person = _safe_float(gt_co.get("P_person_given_class"))
        weak_person = _safe_float(weak_co.get("P_person_given_class"))
        gt_max_hub = _safe_float(gt_co.get("max_P_hub_given_class"))
        weak_max_hub = _safe_float(weak_co.get("max_P_hub_given_class"))
        gt_max_hub_id = _safe_int(gt_co.get("max_cooccurring_hub_raw_id"))
        weak_max_hub_id = _safe_int(weak_co.get("max_cooccurring_hub_raw_id"))
        max_hub = max([x for x in [gt_max_hub, weak_max_hub] if x is not None] or [0.0])
        max_person = max([x for x in [gt_person, weak_person] if x is not None] or [0.0])
        strong_level = "weak"
        if max_hub >= 0.75:
            strong_level = "very_strong"
        elif max_hub >= float(strong_hub_cooccurrence_threshold):
            strong_level = "strong"
        elif max_hub >= 0.25:
            strong_level = "moderate"

        rank_top3_rate = float(buckets.get("in_topK", 0) / max(len(seq), 1)) if seq else None
        rank_4_5_rate = float(buckets.get("near_miss_Kplus1_Kplus2", 0) / max(len(seq), 1)) if seq else None
        rank_6_20_rate = float(buckets.get("medium_miss_6_20", 0) / max(len(seq), 1)) if seq else None
        rank_gt20_rate = float(buckets.get("far_miss_gt20", 0) / max(len(seq), 1)) if seq else None
        rank_missing_rate = float(buckets.get("missing_or_not_ranked", 0) / max(len(seq), 1)) if seq else None
        weak_present_rate = _safe_float(weak_co.get("present_rate"))
        weak_alone_rate = _safe_float(weak_co.get("alone_rate"))

        subtype = "rank_far_representation_blind_spot"
        if max_person >= float(strong_hub_cooccurrence_threshold):
            subtype = "person_cooccurrence_blind_spot"
        elif max_hub >= float(strong_hub_cooccurrence_threshold):
            subtype = "nonperson_hub_cooccurrence_blind_spot"
        elif (weak_present_rate is not None and weak_present_rate <= float(weak_unobservable_present_threshold)) and (
            weak_alone_rate is None or weak_alone_rate <= float(weak_unobservable_alone_threshold)
        ):
            subtype = "weak_label_unobservable"
        elif ((rank_4_5_rate or 0.0) + (rank_6_20_rate or 0.0)) >= 0.50:
            subtype = "near_or_medium_miss_rescuable"
        elif ((rank_gt20_rate or 0.0) + (rank_missing_rate or 0.0)) >= 0.50:
            subtype = "rank_far_representation_blind_spot"
        else:
            subtype = "mixed_fully_missed"

        item: Dict[str, Any] = {
            "raw_id": int(gt),
            "name": label.get("name"),
            "gt_trajectory_count": int(len(seq)),
            "active_raw_count": int(sum(1 for r in seq if _active_raw_contains(r) is True)),
            "active_raw_rate": _rate_bools([_active_raw_contains(r) is True for r in seq if _active_raw_contains(r) is not None]),
            "top1_count": int(sum(1 for r in seq if bool(r.get("final_top1_is_gt")))),
            "top1_rate": _rate_bools([bool(r.get("final_top1_is_gt")) for r in seq]),
            "R_GT_winner_rate": _rate_bools([bool(r.get("r_final_gt_winner")) for r in seq if r.get("r_final_gt_winner") is not None]),
            "mean_gt_mining_rank": _mean(ranks_float),
            "median_gt_mining_rank": _median(ranks_float),
            "rank_p90": _quantile(ranks_float, 0.9),
            "rank_top3_count": int(buckets.get("in_topK", 0)),
            "rank_top3_rate": rank_top3_rate,
            "rank_4_5_count": int(buckets.get("near_miss_Kplus1_Kplus2", 0)),
            "rank_4_5_rate": rank_4_5_rate,
            "rank_6_20_count": int(buckets.get("medium_miss_6_20", 0)),
            "rank_6_20_rate": rank_6_20_rate,
            "rank_gt20_count": int(buckets.get("far_miss_gt20", 0)),
            "rank_gt20_rate": rank_gt20_rate,
            "rank_missing_count": int(buckets.get("missing_or_not_ranked", 0)),
            "rank_missing_rate": rank_missing_rate,
            "top_suppressor_raw_id": int(top_suppressor_raw_id) if top_suppressor_raw_id is not None else None,
            "top_suppressor_name": top_suppressor_name,
            "top_suppressor_count": int(top_suppressor_count),
            "top_suppressor_rate": float(top_suppressor_count / max(len(seq), 1)) if seq else None,
            "top_final_winner_raw_id": int(top_winner_raw_id) if top_winner_raw_id is not None else None,
            "top_final_winner_name": top_winner_name,
            "top_final_winner_count": int(top_winner_count),
            "top_final_winner_rate": float(top_winner_count / max(len(seq), 1)) if seq else None,
            "gt_present_count": int(gt_co.get("present_count", 0) or 0),
            "gt_present_rate": gt_co.get("present_rate"),
            "gt_alone_count": int(gt_co.get("alone_count", 0) or 0),
            "gt_alone_rate": gt_co.get("alone_rate"),
            "gt_with_any_hub_rate": gt_co.get("with_any_hub_rate"),
            "weak_present_count": int(weak_co.get("present_count", 0) or 0),
            "weak_present_rate": weak_co.get("present_rate"),
            "weak_alone_count": int(weak_co.get("alone_count", 0) or 0),
            "weak_alone_rate": weak_co.get("alone_rate"),
            "weak_with_any_hub_rate": weak_co.get("with_any_hub_rate"),
            "P_person_given_class_gt": gt_person,
            "P_person_given_class_weak": weak_person,
            "max_P_hub_given_class_gt": gt_max_hub,
            "max_gt_cooccurring_hub_raw_id": gt_max_hub_id,
            "max_gt_cooccurring_hub_name": gt_co.get("max_cooccurring_hub_name"),
            "max_P_hub_given_class_weak": weak_max_hub,
            "max_weak_cooccurring_hub_raw_id": weak_max_hub_id,
            "max_weak_cooccurring_hub_name": weak_co.get("max_cooccurring_hub_name"),
            "strong_hub_cooccurrence_level": strong_level,
            "failure_subtype": subtype,
            "is_person_cooccurrence": bool(max_person >= float(strong_hub_cooccurrence_threshold)),
            "is_nonperson_hub_cooccurrence": bool(max_hub >= float(strong_hub_cooccurrence_threshold) and max_person < float(strong_hub_cooccurrence_threshold)),
            "top_suppressor_is_hub": bool(top_suppressor_raw_id in hub_set) if top_suppressor_raw_id is not None else False,
            "top_suppressor_is_person": bool(top_suppressor_raw_id == 773) if top_suppressor_raw_id is not None else False,
        }
        out.append(item)
    out.sort(key=lambda r: (str(r.get("failure_subtype")), -(int(r.get("gt_trajectory_count", 0))), -(float(r.get("rank_gt20_rate") or 0.0))))
    return out


def _fully_missed_class_report_payload(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    subtype_counter = Counter(str(r.get("failure_subtype")) for r in rows)
    co_level_counter = Counter(str(r.get("strong_hub_cooccurrence_level")) for r in rows)
    trajectory_count_by_subtype: Counter = Counter()
    for r in rows:
        trajectory_count_by_subtype[str(r.get("failure_subtype"))] += int(r.get("gt_trajectory_count", 0) or 0)
    return {
        "status": "PASS",
        "definition": "Class-level report for classes whose blind_spot_type is fully_missed_blind_spot in formal_aligned_failure_taxonomy_by_class.csv.",
        "class_count": int(len(rows)),
        "trajectory_count": int(sum(int(r.get("gt_trajectory_count", 0) or 0) for r in rows)),
        "failure_subtype_histogram": dict(subtype_counter),
        "trajectory_count_by_failure_subtype": dict(trajectory_count_by_subtype),
        "strong_hub_cooccurrence_level_histogram": dict(co_level_counter),
        "person_cooccurrence_class_count": int(sum(1 for r in rows if bool(r.get("is_person_cooccurrence")))),
        "nonperson_hub_cooccurrence_class_count": int(sum(1 for r in rows if bool(r.get("is_nonperson_hub_cooccurrence")))),
        "rows": rows,
    }


def _weighted_mean_from_class_rows(rows: Sequence[Mapping[str, Any]], key: str, *, weight_key: str = "gt_trajectory_count") -> Optional[float]:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = _safe_float(row.get(key))
        weight = _safe_float(row.get(weight_key))
        if value is None or weight is None or weight <= 0:
            continue
        numerator += float(value) * float(weight)
        denominator += float(weight)
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _fully_missed_trajectory_weighted_payload(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize fully-missed class rows with trajectory-count weighting.

    This is deliberately a pure aggregation over formal_aligned_fully_missed_blind_spot_class_report rows:
    it does not rescore examples, does not touch training state, and preserves the existing class-level taxonomy.
    """
    rows_list = [dict(r) for r in rows]
    total_classes = int(len(rows_list))
    total_traj = int(sum(int(r.get("gt_trajectory_count", 0) or 0) for r in rows_list))

    def _subtype_of(row: Mapping[str, Any]) -> str:
        return str(row.get("failure_subtype") or "unknown")

    def _level_of(row: Mapping[str, Any]) -> str:
        return str(row.get("strong_hub_cooccurrence_level") or "unknown")

    def _aggregate_group(name: str, group_rows: Sequence[Mapping[str, Any]], *, group_type: str) -> Dict[str, Any]:
        cls_count = int(len(group_rows))
        traj_count = int(sum(int(r.get("gt_trajectory_count", 0) or 0) for r in group_rows))
        person_classes = int(sum(1 for r in group_rows if bool(r.get("is_person_cooccurrence"))))
        nonperson_classes = int(sum(1 for r in group_rows if bool(r.get("is_nonperson_hub_cooccurrence"))))
        suppressor_hub_weight = int(sum(int(r.get("gt_trajectory_count", 0) or 0) for r in group_rows if bool(r.get("top_suppressor_is_hub"))))
        suppressor_person_weight = int(sum(int(r.get("gt_trajectory_count", 0) or 0) for r in group_rows if bool(r.get("top_suppressor_is_person"))))
        return {
            "group_type": str(group_type),
            "name": str(name),
            "class_count": cls_count,
            "class_rate_among_fully_missed_classes": float(cls_count / max(total_classes, 1)),
            "trajectory_count_weighted": traj_count,
            "trajectory_weighted_rate_among_fully_missed_report": float(traj_count / max(total_traj, 1)),
            "weighted_active_raw_rate": _weighted_mean_from_class_rows(group_rows, "active_raw_rate"),
            "weighted_top1_rate": _weighted_mean_from_class_rows(group_rows, "top1_rate"),
            "weighted_R_GT_winner_rate": _weighted_mean_from_class_rows(group_rows, "R_GT_winner_rate"),
            "weighted_rank_top3_rate": _weighted_mean_from_class_rows(group_rows, "rank_top3_rate"),
            "weighted_rank_4_5_rate": _weighted_mean_from_class_rows(group_rows, "rank_4_5_rate"),
            "weighted_rank_6_20_rate": _weighted_mean_from_class_rows(group_rows, "rank_6_20_rate"),
            "weighted_rank_gt20_rate": _weighted_mean_from_class_rows(group_rows, "rank_gt20_rate"),
            "weighted_rank_missing_rate": _weighted_mean_from_class_rows(group_rows, "rank_missing_rate"),
            "weighted_gt_alone_rate": _weighted_mean_from_class_rows(group_rows, "gt_alone_rate"),
            "weighted_weak_alone_rate": _weighted_mean_from_class_rows(group_rows, "weak_alone_rate"),
            "weighted_P_person_given_class_gt": _weighted_mean_from_class_rows(group_rows, "P_person_given_class_gt"),
            "weighted_P_person_given_class_weak": _weighted_mean_from_class_rows(group_rows, "P_person_given_class_weak"),
            "weighted_max_P_hub_given_class_gt": _weighted_mean_from_class_rows(group_rows, "max_P_hub_given_class_gt"),
            "weighted_max_P_hub_given_class_weak": _weighted_mean_from_class_rows(group_rows, "max_P_hub_given_class_weak"),
            "person_cooccurrence_class_count": person_classes,
            "nonperson_hub_cooccurrence_class_count": nonperson_classes,
            "top_suppressor_is_hub_trajectory_weighted_count": suppressor_hub_weight,
            "top_suppressor_is_hub_trajectory_weighted_rate": float(suppressor_hub_weight / max(traj_count, 1)),
            "top_suppressor_is_person_trajectory_weighted_count": suppressor_person_weight,
            "top_suppressor_is_person_trajectory_weighted_rate": float(suppressor_person_weight / max(traj_count, 1)),
            "top_classes_by_trajectory_count": [
                {
                    "raw_id": r.get("raw_id"),
                    "name": r.get("name"),
                    "gt_trajectory_count": r.get("gt_trajectory_count"),
                    "active_raw_rate": r.get("active_raw_rate"),
                    "top1_rate": r.get("top1_rate"),
                    "rank_gt20_rate": r.get("rank_gt20_rate"),
                    "rank_missing_rate": r.get("rank_missing_rate"),
                    "top_suppressor_name": r.get("top_suppressor_name"),
                    "P_person_given_class_gt": r.get("P_person_given_class_gt"),
                    "max_gt_cooccurring_hub_name": r.get("max_gt_cooccurring_hub_name"),
                    "failure_subtype": r.get("failure_subtype"),
                    "strong_hub_cooccurrence_level": r.get("strong_hub_cooccurrence_level"),
                }
                for r in sorted(group_rows, key=lambda x: int(x.get("gt_trajectory_count", 0) or 0), reverse=True)[:40]
            ],
        }

    by_subtype: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_level: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows_list:
        by_subtype[_subtype_of(row)].append(row)
        by_level[_level_of(row)].append(row)

    cooccurrence_rows = [
        r for r in rows_list
        if str(r.get("failure_subtype")) in {"person_cooccurrence_blind_spot", "nonperson_hub_cooccurrence_blind_spot"}
    ]
    strong_or_very_rows = [
        r for r in rows_list
        if str(r.get("strong_hub_cooccurrence_level")) in {"strong", "very_strong"}
    ]
    moderate_or_above_rows = [
        r for r in rows_list
        if str(r.get("strong_hub_cooccurrence_level")) in {"moderate", "strong", "very_strong"}
    ]
    non_cooccurrence_rows = [r for r in rows_list if r not in cooccurrence_rows]

    subtype_rows = [
        _aggregate_group(name, group, group_type="failure_subtype")
        for name, group in sorted(by_subtype.items(), key=lambda kv: sum(int(r.get("gt_trajectory_count", 0) or 0) for r in kv[1]), reverse=True)
    ]
    co_level_rows = [
        _aggregate_group(name, group, group_type="strong_hub_cooccurrence_level")
        for name, group in sorted(by_level.items(), key=lambda kv: sum(int(r.get("gt_trajectory_count", 0) or 0) for r in kv[1]), reverse=True)
    ]
    big_mode_rows = [
        _aggregate_group("cooccurrence_hub_blind_spot", cooccurrence_rows, group_type="big_mode"),
        _aggregate_group("strong_or_very_strong_hub_cooccurrence", strong_or_very_rows, group_type="big_mode"),
        _aggregate_group("moderate_or_above_hub_cooccurrence", moderate_or_above_rows, group_type="big_mode"),
        _aggregate_group("non_cooccurrence_residual", non_cooccurrence_rows, group_type="big_mode"),
    ]

    return {
        "status": "PASS",
        "definition": "Trajectory-weighted aggregation of formal_aligned_fully_missed_blind_spot_class_report rows. Weights are gt_trajectory_count; no rescoring or training semantics are changed.",
        "class_count": total_classes,
        "trajectory_count_weighted_total": total_traj,
        "big_mode_rows": big_mode_rows,
        "by_failure_subtype": subtype_rows,
        "by_strong_hub_cooccurrence_level": co_level_rows,
        "all_rows_for_csv": big_mode_rows + subtype_rows + co_level_rows,
    }


def _hub_collapse_rescue_audit_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    gt_class_cooccurrence: Mapping[str, Any],
    weak_class_cooccurrence: Mapping[str, Any],
    hub_raw_ids: Sequence[int],
    records_by_raw: Mapping[int, Mapping[str, Any]],
    risk_threshold: float,
    low_alone_threshold: float,
    current_k: int,
    top_examples: int,
) -> Dict[str, Any]:
    """Find high co-occurrence-risk rows that are nevertheless rescued.

    This is a pure read-only aggregation over formal-aligned row diagnostics and class
    co-occurrence payloads. It does not rescore rows and does not alter training semantics.
    """
    joined = _rows_joined(rows)
    hub_set = {int(x) for x in hub_raw_ids}
    gt_co_map = _cooccurrence_map(gt_class_cooccurrence)
    weak_co_map = _cooccurrence_map(weak_class_cooccurrence)
    risk_threshold = float(risk_threshold)
    low_alone_threshold = float(low_alone_threshold)
    current_k = max(1, int(current_k))

    class_ids = sorted({int(x) for x in (_safe_int(r.get("gt_raw_id")) for r in joined) if x is not None})

    def _class_risk_meta(raw_id: int) -> Dict[str, Any]:
        gt_co = gt_co_map.get(int(raw_id), {})
        weak_co = weak_co_map.get(int(raw_id), {})
        gt_person = _safe_float(gt_co.get("P_person_given_class"))
        weak_person = _safe_float(weak_co.get("P_person_given_class"))
        gt_max_hub = _safe_float(gt_co.get("max_P_hub_given_class"))
        weak_max_hub = _safe_float(weak_co.get("max_P_hub_given_class"))
        gt_alone = _safe_float(gt_co.get("alone_rate"))
        weak_alone = _safe_float(weak_co.get("alone_rate"))
        gt_present = _safe_float(gt_co.get("present_rate"))
        weak_present = _safe_float(weak_co.get("present_rate"))
        max_person = max([x for x in (gt_person, weak_person) if x is not None] or [0.0])
        max_hub = max([x for x in (gt_max_hub, weak_max_hub) if x is not None] or [0.0])
        low_alone = bool(
            (gt_alone is not None and gt_alone <= low_alone_threshold)
            or (weak_alone is not None and weak_alone <= low_alone_threshold)
        )
        is_high_risk = bool(max_person >= risk_threshold or max_hub >= risk_threshold)
        risk_sources: List[str] = []
        if gt_person is not None and gt_person >= risk_threshold:
            risk_sources.append("gt_person")
        if weak_person is not None and weak_person >= risk_threshold:
            risk_sources.append("weak_person")
        if gt_max_hub is not None and gt_max_hub >= risk_threshold:
            risk_sources.append("gt_hub")
        if weak_max_hub is not None and weak_max_hub >= risk_threshold:
            risk_sources.append("weak_hub")
        if low_alone:
            risk_sources.append("low_alone")
        max_hub_id = _safe_int(gt_co.get("max_cooccurring_hub_raw_id"))
        max_hub_name = gt_co.get("max_cooccurring_hub_name")
        if max_hub_id is None:
            max_hub_id = _safe_int(weak_co.get("max_cooccurring_hub_raw_id"))
            max_hub_name = weak_co.get("max_cooccurring_hub_name")
        label = _class_label(int(raw_id), records_by_raw)
        return {
            "raw_id": int(raw_id),
            "name": label.get("name"),
            "is_high_risk_class": is_high_risk,
            "risk_sources": ";".join(risk_sources),
            "risk_threshold": risk_threshold,
            "low_alone_threshold": low_alone_threshold,
            "P_person_given_class_gt": gt_person,
            "P_person_given_class_weak": weak_person,
            "max_P_hub_given_class_gt": gt_max_hub,
            "max_P_hub_given_class_weak": weak_max_hub,
            "max_P_person_given_class": max_person,
            "max_P_hub_given_class": max_hub,
            "max_cooccurring_hub_raw_id": max_hub_id,
            "max_cooccurring_hub_name": max_hub_name,
            "gt_alone_rate": gt_alone,
            "weak_alone_rate": weak_alone,
            "gt_present_rate": gt_present,
            "weak_present_rate": weak_present,
            "low_alone_flag": low_alone,
        }

    risk_meta_by_class = {cid: _class_risk_meta(cid) for cid in class_ids}
    high_risk_class_ids = {cid for cid, meta in risk_meta_by_class.items() if bool(meta.get("is_high_risk_class"))}

    def _row_flags(row: Mapping[str, Any]) -> Dict[str, Any]:
        gt = _safe_int(row.get("gt_raw_id"))
        known_set = {int(x) for x in _unique_ints(row.get("candidate_ids_known"))}
        active_extra = _active_extra_set(row)
        extra_set = {int(x) for x in _unique_ints(row.get("candidate_ids_extra"))}
        final_winner = _safe_int(row.get("final_winner_raw_id"))
        suppressor = _safe_int(row.get("top_suppressor_raw_id"))
        candidate_hubs = sorted((known_set | active_extra | extra_set) & hub_set)
        pressure_types: List[str] = []
        if known_set & hub_set:
            pressure_types.append("hub_in_known_Yprime")
        if active_extra & hub_set:
            pressure_types.append("hub_in_active_extra")
        if extra_set & hub_set:
            pressure_types.append("hub_in_extra_candidates")
        if final_winner in hub_set:
            pressure_types.append("final_winner_is_hub")
        if suppressor in hub_set:
            pressure_types.append("top_suppressor_is_hub")
        if final_winner == 773:
            pressure_types.append("final_winner_is_person")
        if suppressor == 773:
            pressure_types.append("top_suppressor_is_person")
        rank = _safe_int(row.get("gt_mining_rank"))
        active_rescue = bool(_active_raw_contains(row) is True)
        rank_rescue = bool(rank is not None and int(rank) <= current_k)
        r_rescue = bool(row.get("r_final_gt_winner")) if row.get("r_final_gt_winner") is not None else False
        top1_rescue = bool(row.get("final_top1_is_gt"))
        pressure = bool(pressure_types)
        rescue_types: List[str] = []
        if active_rescue:
            rescue_types.append("active_rescue")
        if rank_rescue:
            rescue_types.append("rank_topK_rescue")
        if r_rescue:
            rescue_types.append("responsibility_rescue")
        if top1_rescue:
            rescue_types.append("strict_top1_rescue")
        if pressure and top1_rescue:
            rescue_types.append("true_hub_rescue")
        if (not pressure) and top1_rescue:
            rescue_types.append("no_pressure_success")
        return {
            "gt_raw_id": gt,
            "hub_pressure_present": pressure,
            "hub_pressure_types": ";".join(sorted(set(pressure_types))),
            "candidate_hub_raw_ids": candidate_hubs,
            "candidate_hub_count": int(len(candidate_hubs)),
            "active_rescue": active_rescue,
            "rank_topK_rescue": rank_rescue,
            "R_GT_winner_rescue": r_rescue,
            "strict_top1_rescue": top1_rescue,
            "any_rescue": bool(active_rescue or rank_rescue or r_rescue or top1_rescue),
            "rescue_types": ";".join(rescue_types),
            "gt_mining_rank": rank,
            "final_winner_raw_id": final_winner,
            "top_suppressor_raw_id": suppressor,
        }

    high_risk_rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in joined:
        gt = _safe_int(row.get("gt_raw_id"))
        if gt is None or int(gt) not in high_risk_class_ids:
            continue
        flags = _row_flags(row)
        meta = risk_meta_by_class.get(int(gt), {})
        label = _class_label(int(gt), records_by_raw)
        enriched = dict(row)
        enriched.update({
            "gt_raw_id": int(gt),
            "gt_name": label.get("name"),
            **{k: v for k, v in meta.items() if k not in {"raw_id", "name"}},
            **flags,
            "final_winner_name": _class_label(int(flags.get("final_winner_raw_id")), records_by_raw).get("name") if flags.get("final_winner_raw_id") is not None else None,
            "top_suppressor_name": _class_label(int(flags.get("top_suppressor_raw_id")), records_by_raw).get("name") if flags.get("top_suppressor_raw_id") is not None else None,
        })
        high_risk_rows.append(enriched)
        grouped[int(gt)].append(enriched)
        if bool(enriched.get("hub_pressure_present")) and bool(enriched.get("any_rescue")):
            examples.append(enriched)
        elif bool(enriched.get("strict_top1_rescue")):
            examples.append(enriched)

    def _rate(seq: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        return _rate_bools([bool(r.get(key)) for r in seq])

    def _summarize_seq(seq: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        ranks = [float(r.get("gt_mining_rank")) for r in seq if _safe_int(r.get("gt_mining_rank")) is not None]
        return {
            "count": int(len(seq)),
            "active_rescue_rate": _rate(seq, "active_rescue"),
            "rank_topK_rescue_rate": _rate(seq, "rank_topK_rescue"),
            "R_GT_winner_rescue_rate": _rate(seq, "R_GT_winner_rescue"),
            "strict_top1_rescue_rate": _rate(seq, "strict_top1_rescue"),
            "hub_pressure_rate": _rate(seq, "hub_pressure_present"),
            "mean_gt_mining_rank": _mean(ranks),
            "median_gt_mining_rank": _median(ranks),
            "rank_missing_rate": _rate_bools([_safe_int(r.get("gt_mining_rank")) is None for r in seq]),
        }

    pressure_rows = [r for r in high_risk_rows if bool(r.get("hub_pressure_present"))]
    no_pressure_rows = [r for r in high_risk_rows if not bool(r.get("hub_pressure_present"))]
    true_hub_rescue_rows = [r for r in pressure_rows if bool(r.get("strict_top1_rescue"))]
    responsibility_rescue_rows = [r for r in pressure_rows if bool(r.get("R_GT_winner_rescue"))]
    slot_rescue_rows = [r for r in pressure_rows if bool(r.get("active_rescue"))]
    rank_rescue_rows = [r for r in pressure_rows if bool(r.get("rank_topK_rescue"))]
    collapse_failure_rows = [r for r in pressure_rows if not bool(r.get("strict_top1_rescue"))]
    no_pressure_success_rows = [r for r in no_pressure_rows if bool(r.get("strict_top1_rescue"))]

    class_report: List[Dict[str, Any]] = []
    for gt in sorted(grouped):
        seq = grouped[int(gt)]
        meta = risk_meta_by_class.get(int(gt), {})
        pressure_seq = [r for r in seq if bool(r.get("hub_pressure_present"))]
        success_seq = [r for r in pressure_seq if bool(r.get("strict_top1_rescue"))]
        failure_seq = [r for r in pressure_seq if not bool(r.get("strict_top1_rescue"))]
        suppressors = Counter(_safe_int(r.get("top_suppressor_raw_id")) for r in failure_seq if _safe_int(r.get("top_suppressor_raw_id")) is not None)
        winners = Counter(_safe_int(r.get("final_winner_raw_id")) for r in success_seq if _safe_int(r.get("final_winner_raw_id")) is not None)
        top_supp_id = suppressors.most_common(1)[0][0] if suppressors else None
        top_win_id = winners.most_common(1)[0][0] if winners else None
        ranks_success = [float(r.get("gt_mining_rank")) for r in success_seq if _safe_int(r.get("gt_mining_rank")) is not None]
        ranks_failure = [float(r.get("gt_mining_rank")) for r in failure_seq if _safe_int(r.get("gt_mining_rank")) is not None]
        item = {
            "raw_id": int(gt),
            "name": meta.get("name") or _class_label(int(gt), records_by_raw).get("name"),
            "gt_trajectory_count": int(len(seq)),
            "risk_sources": meta.get("risk_sources"),
            "P_person_given_class_gt": meta.get("P_person_given_class_gt"),
            "P_person_given_class_weak": meta.get("P_person_given_class_weak"),
            "max_P_hub_given_class_gt": meta.get("max_P_hub_given_class_gt"),
            "max_P_hub_given_class_weak": meta.get("max_P_hub_given_class_weak"),
            "gt_alone_rate": meta.get("gt_alone_rate"),
            "weak_alone_rate": meta.get("weak_alone_rate"),
            "max_cooccurring_hub_raw_id": meta.get("max_cooccurring_hub_raw_id"),
            "max_cooccurring_hub_name": meta.get("max_cooccurring_hub_name"),
            "hub_pressure_row_count": int(len(pressure_seq)),
            "hub_pressure_row_rate": float(len(pressure_seq) / max(len(seq), 1)),
            "active_rescue_count": int(sum(1 for r in pressure_seq if bool(r.get("active_rescue")))),
            "active_rescue_rate_among_pressure": _rate(pressure_seq, "active_rescue"),
            "rank_topK_rescue_count": int(sum(1 for r in pressure_seq if bool(r.get("rank_topK_rescue")))),
            "rank_topK_rescue_rate_among_pressure": _rate(pressure_seq, "rank_topK_rescue"),
            "R_GT_winner_rescue_count": int(sum(1 for r in pressure_seq if bool(r.get("R_GT_winner_rescue")))),
            "R_GT_winner_rescue_rate_among_pressure": _rate(pressure_seq, "R_GT_winner_rescue"),
            "strict_top1_rescue_count": int(len(success_seq)),
            "strict_top1_rescue_rate_among_pressure": float(len(success_seq) / max(len(pressure_seq), 1)) if pressure_seq else None,
            "collapse_failure_count": int(len(failure_seq)),
            "collapse_failure_rate_among_pressure": float(len(failure_seq) / max(len(pressure_seq), 1)) if pressure_seq else None,
            "no_pressure_success_count": int(sum(1 for r in seq if (not bool(r.get("hub_pressure_present"))) and bool(r.get("strict_top1_rescue")))),
            "mean_gt_rank_success_pressure_top1": _mean(ranks_success),
            "mean_gt_rank_failure_pressure_non_top1": _mean(ranks_failure),
            "top_success_winner_raw_id": int(top_win_id) if top_win_id is not None else None,
            "top_success_winner_name": _class_label(int(top_win_id), records_by_raw).get("name") if top_win_id is not None else None,
            "top_failure_suppressor_raw_id": int(top_supp_id) if top_supp_id is not None else None,
            "top_failure_suppressor_name": _class_label(int(top_supp_id), records_by_raw).get("name") if top_supp_id is not None else None,
        }
        class_report.append(item)
    class_report.sort(key=lambda r: (-(int(r.get("strict_top1_rescue_count") or 0)), -(int(r.get("R_GT_winner_rescue_count") or 0)), -(int(r.get("gt_trajectory_count") or 0))))

    def _compact_example(row: Mapping[str, Any]) -> Dict[str, Any]:
        keys = [
            "clip_id", "video_id", "trajectory_id", "tid", "gt_raw_id", "gt_name",
            "risk_sources", "P_person_given_class_gt", "max_P_hub_given_class_gt", "gt_alone_rate", "weak_alone_rate",
            "hub_pressure_present", "hub_pressure_types", "candidate_hub_raw_ids", "active_raw_contains",
            "active_rescue", "rank_topK_rescue", "R_GT_winner_rescue", "strict_top1_rescue", "rescue_types",
            "gt_mining_rank", "final_winner_raw_id", "final_winner_name", "top_suppressor_raw_id", "top_suppressor_name",
            "candidate_ids_known", "candidate_ids_extra",
        ]
        out = {k: row.get(k) for k in keys if k in row}
        # Keep optional score/mass fields if the upstream diagnostics already exposed them.
        for k in ["score_gt", "score_person", "score_top_hub", "margin_gt_minus_person", "margin_gt_minus_top_hub", "r_mass_gt", "r_mass_person", "r_mass_top_hub"]:
            if k in row:
                out[k] = row.get(k)
        return out

    example_rows = [_compact_example(r) for r in sorted(
        examples,
        key=lambda r: (
            0 if bool(r.get("strict_top1_rescue")) and bool(r.get("hub_pressure_present")) else 1,
            0 if bool(r.get("R_GT_winner_rescue")) else 1,
            999999 if _safe_int(r.get("gt_mining_rank")) is None else int(r.get("gt_mining_rank")),
            str(r.get("gt_name")),
        ),
    )[:max(1, int(top_examples))]]

    contrast_rows = [
        {"group": "high_risk_all", **_summarize_seq(high_risk_rows)},
        {"group": "hub_pressure_rows", **_summarize_seq(pressure_rows)},
        {"group": "no_pressure_rows", **_summarize_seq(no_pressure_rows)},
        {"group": "true_hub_rescue_top1", **_summarize_seq(true_hub_rescue_rows)},
        {"group": "responsibility_rescue", **_summarize_seq(responsibility_rescue_rows)},
        {"group": "slot_active_rescue", **_summarize_seq(slot_rescue_rows)},
        {"group": "rank_topK_rescue", **_summarize_seq(rank_rescue_rows)},
        {"group": "collapse_failure_pressure_non_top1", **_summarize_seq(collapse_failure_rows)},
        {"group": "no_pressure_success_top1", **_summarize_seq(no_pressure_success_rows)},
    ]

    summary = {
        "status": "PASS",
        "definition": "High co-occurrence-risk classes are classes with max P(hub|class) or P(person|class) >= hub_collapse_risk_threshold in GT/weak co-occurrence. Rescues are counted on formal-aligned rows without rescoring.",
        "current_k": int(current_k),
        "risk_threshold": float(risk_threshold),
        "low_alone_threshold": float(low_alone_threshold),
        "row_count": int(len(joined)),
        "high_risk_class_count": int(len(high_risk_class_ids)),
        "high_risk_row_count": int(len(high_risk_rows)),
        "hub_pressure_row_count": int(len(pressure_rows)),
        "hub_pressure_rate_among_high_risk_rows": float(len(pressure_rows) / max(len(high_risk_rows), 1)),
        "active_rescue_count": int(len(slot_rescue_rows)),
        "active_rescue_rate_among_pressure_rows": float(len(slot_rescue_rows) / max(len(pressure_rows), 1)),
        "rank_topK_rescue_count": int(len(rank_rescue_rows)),
        "rank_topK_rescue_rate_among_pressure_rows": float(len(rank_rescue_rows) / max(len(pressure_rows), 1)),
        "R_GT_winner_rescue_count": int(len(responsibility_rescue_rows)),
        "R_GT_winner_rescue_rate_among_pressure_rows": float(len(responsibility_rescue_rows) / max(len(pressure_rows), 1)),
        "strict_top1_rescue_count": int(len(true_hub_rescue_rows)),
        "strict_top1_rescue_rate_among_pressure_rows": float(len(true_hub_rescue_rows) / max(len(pressure_rows), 1)),
        "collapse_failure_count": int(len(collapse_failure_rows)),
        "collapse_failure_rate_among_pressure_rows": float(len(collapse_failure_rows) / max(len(pressure_rows), 1)),
        "no_pressure_success_count": int(len(no_pressure_success_rows)),
        "no_pressure_success_rate_among_high_risk_rows": float(len(no_pressure_success_rows) / max(len(high_risk_rows), 1)),
        "success_mode_interpretation": {
            "active_rescue": "GT enters active extra under hub pressure.",
            "rank_topK_rescue": "GT mining rank is within current TopK under hub pressure.",
            "responsibility_rescue": "R_final/E-step winner is GT under hub pressure.",
            "strict_top1_rescue": "final top1 is GT under hub pressure; this is the strongest rescue evidence.",
            "no_pressure_success": "Class is globally high-risk but this row did not expose hub pressure; not counted as overcoming collapse.",
        },
    }
    return {
        "summary": summary,
        "class_rows": class_report,
        "example_rows": example_rows,
        "contrast_rows": contrast_rows,
    }


def _annotation_non_gt_hub_rescue_audit_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    annotation_units: Sequence[Mapping[str, Any]],
    gt_class_cooccurrence: Mapping[str, Any],
    weak_class_cooccurrence: Mapping[str, Any],
    hub_raw_ids: Sequence[int],
    records_by_raw: Mapping[int, Mapping[str, Any]],
    risk_threshold: float,
    low_alone_threshold: float,
    current_k: int,
    top_examples: int,
) -> Dict[str, Any]:
    """Audit whether strict rescue occurs when the GT annotation unit has non-GT hubs.

    This is stricter than the model-level hub pressure audit: hub_raw_ids are first
    stripped of the current gt_raw_id, then the annotation class set for the row's
    video/clip is checked. It answers whether successes are simply coming from
    clips with no non-GT hub in the GT annotation.
    """
    joined = _rows_joined(rows)
    hub_set = {int(x) for x in hub_raw_ids}
    current_k = max(1, int(current_k))
    risk_threshold = float(risk_threshold)
    low_alone_threshold = float(low_alone_threshold)
    unit_class_map: Dict[str, set[int]] = {}
    for unit in annotation_units:
        uid = unit.get("unit_id")
        if uid is None:
            continue
        vals = {int(x) for x in _unique_ints(unit.get("class_ids"))}
        if vals:
            unit_class_map[str(uid)] = vals

    gt_co_map = _cooccurrence_map(gt_class_cooccurrence)
    weak_co_map = _cooccurrence_map(weak_class_cooccurrence)
    class_ids = sorted({int(x) for x in (_safe_int(r.get("gt_raw_id")) for r in joined) if x is not None})

    def _class_risk_meta(raw_id: int) -> Dict[str, Any]:
        gt_co = gt_co_map.get(int(raw_id), {})
        weak_co = weak_co_map.get(int(raw_id), {})
        gt_person = _safe_float(gt_co.get("P_person_given_class"))
        weak_person = _safe_float(weak_co.get("P_person_given_class"))
        gt_max_hub = _safe_float(gt_co.get("max_P_hub_given_class"))
        weak_max_hub = _safe_float(weak_co.get("max_P_hub_given_class"))
        gt_alone = _safe_float(gt_co.get("alone_rate"))
        weak_alone = _safe_float(weak_co.get("alone_rate"))
        max_person = max([x for x in (gt_person, weak_person) if x is not None] or [0.0])
        max_hub = max([x for x in (gt_max_hub, weak_max_hub) if x is not None] or [0.0])
        low_alone = bool(
            (gt_alone is not None and gt_alone <= low_alone_threshold)
            or (weak_alone is not None and weak_alone <= low_alone_threshold)
        )
        is_high_risk = bool(max_person >= risk_threshold or max_hub >= risk_threshold)
        risk_sources: List[str] = []
        if gt_person is not None and gt_person >= risk_threshold:
            risk_sources.append("gt_person")
        if weak_person is not None and weak_person >= risk_threshold:
            risk_sources.append("weak_person")
        if gt_max_hub is not None and gt_max_hub >= risk_threshold:
            risk_sources.append("gt_hub")
        if weak_max_hub is not None and weak_max_hub >= risk_threshold:
            risk_sources.append("weak_hub")
        if low_alone:
            risk_sources.append("low_alone")
        label = _class_label(int(raw_id), records_by_raw)
        return {
            "raw_id": int(raw_id),
            "name": label.get("name"),
            "is_high_risk_class": is_high_risk,
            "risk_sources": ";".join(risk_sources),
            "P_person_given_class_gt": gt_person,
            "P_person_given_class_weak": weak_person,
            "max_P_hub_given_class_gt": gt_max_hub,
            "max_P_hub_given_class_weak": weak_max_hub,
            "gt_alone_rate": gt_alone,
            "weak_alone_rate": weak_alone,
            "low_alone_flag": low_alone,
        }

    risk_meta_by_class = {cid: _class_risk_meta(cid) for cid in class_ids}
    high_risk_class_ids = {cid for cid, meta in risk_meta_by_class.items() if bool(meta.get("is_high_risk_class"))}

    def _annotation_class_set_for_row(row: Mapping[str, Any]) -> Optional[set[int]]:
        keys = [row.get("video_id"), row.get("clip_id"), row.get("vid")]
        for key in keys:
            if key is None:
                continue
            vals = unit_class_map.get(str(key))
            if vals is not None:
                return set(vals)
        return None

    def _row_flags(row: Mapping[str, Any]) -> Dict[str, Any]:
        gt = _safe_int(row.get("gt_raw_id"))
        if gt is None:
            gt = -1
        gt_int = int(gt)
        non_gt_hub_set = set(hub_set) - {gt_int}
        annotation_class_set = _annotation_class_set_for_row(row)
        annotation_known = annotation_class_set is not None
        annotation_non_gt_hubs = sorted((annotation_class_set or set()) & non_gt_hub_set)
        annotation_has_non_gt_hub = bool(annotation_non_gt_hubs)
        annotation_has_person_non_gt = bool(gt_int != 773 and annotation_class_set is not None and 773 in annotation_class_set)

        known_set = {int(x) for x in _unique_ints(row.get("candidate_ids_known"))}
        active_extra = _active_extra_set(row)
        extra_set = {int(x) for x in _unique_ints(row.get("candidate_ids_extra"))}
        final_winner = _safe_int(row.get("final_winner_raw_id"))
        suppressor = _safe_int(row.get("top_suppressor_raw_id"))
        candidate_non_gt_hubs = sorted((known_set | active_extra | extra_set) & non_gt_hub_set)
        model_pressure_types: List[str] = []
        if known_set & non_gt_hub_set:
            model_pressure_types.append("non_gt_hub_in_known_Yprime")
        if active_extra & non_gt_hub_set:
            model_pressure_types.append("non_gt_hub_in_active_extra")
        if extra_set & non_gt_hub_set:
            model_pressure_types.append("non_gt_hub_in_extra_candidates")
        if final_winner in non_gt_hub_set:
            model_pressure_types.append("final_winner_is_non_gt_hub")
        if suppressor in non_gt_hub_set:
            model_pressure_types.append("top_suppressor_is_non_gt_hub")
        if gt_int != 773 and 773 in (known_set | active_extra | extra_set):
            model_pressure_types.append("person_in_candidate_domain")
        if gt_int != 773 and final_winner == 773:
            model_pressure_types.append("final_winner_is_person")
        if gt_int != 773 and suppressor == 773:
            model_pressure_types.append("top_suppressor_is_person")
        model_non_gt_hub_pressure = bool(model_pressure_types)

        rank = _safe_int(row.get("gt_mining_rank"))
        active_rescue = bool(_active_raw_contains(row) is True)
        rank_rescue = bool(rank is not None and int(rank) <= current_k)
        r_rescue = bool(row.get("r_final_gt_winner")) if row.get("r_final_gt_winner") is not None else False
        top1_rescue = bool(row.get("final_top1_is_gt"))
        return {
            "annotation_known": annotation_known,
            "annotation_class_ids": sorted(annotation_class_set) if annotation_class_set is not None else None,
            "annotation_non_gt_hub_raw_ids": annotation_non_gt_hubs,
            "annotation_non_gt_hub_count": int(len(annotation_non_gt_hubs)),
            "annotation_has_non_gt_hub": annotation_has_non_gt_hub,
            "annotation_has_person_non_gt": annotation_has_person_non_gt,
            "model_non_gt_hub_pressure_present": model_non_gt_hub_pressure,
            "model_non_gt_hub_pressure_types": ";".join(sorted(set(model_pressure_types))),
            "model_candidate_non_gt_hub_raw_ids": candidate_non_gt_hubs,
            "model_candidate_non_gt_hub_count": int(len(candidate_non_gt_hubs)),
            "annotation_and_model_non_gt_hub_pressure": bool(annotation_has_non_gt_hub and model_non_gt_hub_pressure),
            "active_rescue": active_rescue,
            "rank_topK_rescue": rank_rescue,
            "R_GT_winner_rescue": r_rescue,
            "strict_top1_rescue": top1_rescue,
            "gt_mining_rank": rank,
            "final_winner_raw_id": final_winner,
            "top_suppressor_raw_id": suppressor,
        }

    high_risk_rows: List[Dict[str, Any]] = []
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    examples: List[Dict[str, Any]] = []
    for row in joined:
        gt = _safe_int(row.get("gt_raw_id"))
        if gt is None or int(gt) not in high_risk_class_ids:
            continue
        meta = risk_meta_by_class.get(int(gt), {})
        flags = _row_flags(row)
        enriched = dict(row)
        enriched.update({
            "gt_raw_id": int(gt),
            "gt_name": _class_label(int(gt), records_by_raw).get("name"),
            **{k: v for k, v in meta.items() if k not in {"raw_id", "name"}},
            **flags,
            "final_winner_name": _class_label(int(flags.get("final_winner_raw_id")), records_by_raw).get("name") if flags.get("final_winner_raw_id") is not None else None,
            "top_suppressor_name": _class_label(int(flags.get("top_suppressor_raw_id")), records_by_raw).get("name") if flags.get("top_suppressor_raw_id") is not None else None,
        })
        high_risk_rows.append(enriched)
        grouped[int(gt)].append(enriched)
        if bool(enriched.get("strict_top1_rescue")) or bool(enriched.get("annotation_has_non_gt_hub")):
            examples.append(enriched)

    def _rate(seq: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        return _rate_bools([bool(r.get(key)) for r in seq])

    def _summarize_seq(seq: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        ranks = [float(r.get("gt_mining_rank")) for r in seq if _safe_int(r.get("gt_mining_rank")) is not None]
        return {
            "count": int(len(seq)),
            "annotation_known_rate": _rate(seq, "annotation_known"),
            "annotation_has_non_gt_hub_rate": _rate(seq, "annotation_has_non_gt_hub"),
            "annotation_has_person_non_gt_rate": _rate(seq, "annotation_has_person_non_gt"),
            "model_non_gt_hub_pressure_rate": _rate(seq, "model_non_gt_hub_pressure_present"),
            "annotation_and_model_non_gt_hub_pressure_rate": _rate(seq, "annotation_and_model_non_gt_hub_pressure"),
            "active_rescue_rate": _rate(seq, "active_rescue"),
            "rank_topK_rescue_rate": _rate(seq, "rank_topK_rescue"),
            "R_GT_winner_rescue_rate": _rate(seq, "R_GT_winner_rescue"),
            "strict_top1_rescue_rate": _rate(seq, "strict_top1_rescue"),
            "mean_gt_mining_rank": _mean(ranks),
            "median_gt_mining_rank": _median(ranks),
            "rank_missing_rate": _rate_bools([_safe_int(r.get("gt_mining_rank")) is None for r in seq]),
        }

    annotation_known_rows = [r for r in high_risk_rows if bool(r.get("annotation_known"))]
    ann_has_rows = [r for r in high_risk_rows if bool(r.get("annotation_has_non_gt_hub"))]
    ann_no_rows = [r for r in high_risk_rows if bool(r.get("annotation_known")) and not bool(r.get("annotation_has_non_gt_hub"))]
    ann_unknown_rows = [r for r in high_risk_rows if not bool(r.get("annotation_known"))]
    model_pressure_rows = [r for r in high_risk_rows if bool(r.get("model_non_gt_hub_pressure_present"))]
    ann_and_model_rows = [r for r in high_risk_rows if bool(r.get("annotation_and_model_non_gt_hub_pressure"))]
    strict_ann_has_rows = [r for r in ann_has_rows if bool(r.get("strict_top1_rescue"))]
    strict_ann_no_rows = [r for r in ann_no_rows if bool(r.get("strict_top1_rescue"))]
    strict_ann_and_model_rows = [r for r in ann_and_model_rows if bool(r.get("strict_top1_rescue"))]
    fail_ann_has_rows = [r for r in ann_has_rows if not bool(r.get("strict_top1_rescue"))]
    fail_ann_no_rows = [r for r in ann_no_rows if not bool(r.get("strict_top1_rescue"))]

    contrast_rows = [
        {"group": "high_risk_all", **_summarize_seq(high_risk_rows)},
        {"group": "annotation_known_rows", **_summarize_seq(annotation_known_rows)},
        {"group": "annotation_has_non_gt_hub", **_summarize_seq(ann_has_rows)},
        {"group": "annotation_no_non_gt_hub", **_summarize_seq(ann_no_rows)},
        {"group": "annotation_unknown_rows", **_summarize_seq(ann_unknown_rows)},
        {"group": "model_non_gt_hub_pressure", **_summarize_seq(model_pressure_rows)},
        {"group": "annotation_and_model_non_gt_hub_pressure", **_summarize_seq(ann_and_model_rows)},
        {"group": "strict_top1_success_annotation_has_non_gt_hub", **_summarize_seq(strict_ann_has_rows)},
        {"group": "strict_top1_success_annotation_no_non_gt_hub", **_summarize_seq(strict_ann_no_rows)},
        {"group": "strict_top1_success_annotation_and_model_non_gt_hub_pressure", **_summarize_seq(strict_ann_and_model_rows)},
        {"group": "collapse_failure_annotation_has_non_gt_hub", **_summarize_seq(fail_ann_has_rows)},
        {"group": "collapse_failure_annotation_no_non_gt_hub", **_summarize_seq(fail_ann_no_rows)},
    ]

    class_rows: List[Dict[str, Any]] = []
    for gt in sorted(grouped):
        seq = grouped[int(gt)]
        ann_has_seq = [r for r in seq if bool(r.get("annotation_has_non_gt_hub"))]
        ann_no_seq = [r for r in seq if bool(r.get("annotation_known")) and not bool(r.get("annotation_has_non_gt_hub"))]
        model_seq = [r for r in seq if bool(r.get("model_non_gt_hub_pressure_present"))]
        ann_model_seq = [r for r in seq if bool(r.get("annotation_and_model_non_gt_hub_pressure"))]
        meta = risk_meta_by_class.get(int(gt), {})
        item = {
            "raw_id": int(gt),
            "name": meta.get("name") or _class_label(int(gt), records_by_raw).get("name"),
            "gt_trajectory_count": int(len(seq)),
            "risk_sources": meta.get("risk_sources"),
            "P_person_given_class_gt": meta.get("P_person_given_class_gt"),
            "P_person_given_class_weak": meta.get("P_person_given_class_weak"),
            "max_P_hub_given_class_gt": meta.get("max_P_hub_given_class_gt"),
            "max_P_hub_given_class_weak": meta.get("max_P_hub_given_class_weak"),
            "gt_alone_rate": meta.get("gt_alone_rate"),
            "weak_alone_rate": meta.get("weak_alone_rate"),
            "annotation_non_gt_hub_row_count": int(len(ann_has_seq)),
            "annotation_non_gt_hub_row_rate": float(len(ann_has_seq) / max(len(seq), 1)),
            "annotation_no_non_gt_hub_row_count": int(len(ann_no_seq)),
            "annotation_no_non_gt_hub_row_rate": float(len(ann_no_seq) / max(len(seq), 1)),
            "model_non_gt_hub_pressure_row_count": int(len(model_seq)),
            "model_non_gt_hub_pressure_row_rate": float(len(model_seq) / max(len(seq), 1)),
            "annotation_and_model_non_gt_hub_pressure_row_count": int(len(ann_model_seq)),
            "annotation_and_model_non_gt_hub_pressure_row_rate": float(len(ann_model_seq) / max(len(seq), 1)),
            "strict_top1_success_under_annotation_non_gt_hub_count": int(sum(1 for r in ann_has_seq if bool(r.get("strict_top1_rescue")))),
            "strict_top1_success_under_annotation_non_gt_hub_rate": _rate(ann_has_seq, "strict_top1_rescue"),
            "strict_top1_success_without_annotation_non_gt_hub_count": int(sum(1 for r in ann_no_seq if bool(r.get("strict_top1_rescue")))),
            "strict_top1_success_without_annotation_non_gt_hub_rate": _rate(ann_no_seq, "strict_top1_rescue"),
            "strict_top1_success_under_annotation_and_model_non_gt_hub_pressure_count": int(sum(1 for r in ann_model_seq if bool(r.get("strict_top1_rescue")))),
            "strict_top1_success_under_annotation_and_model_non_gt_hub_pressure_rate": _rate(ann_model_seq, "strict_top1_rescue"),
        }
        class_rows.append(item)
    class_rows.sort(key=lambda r: (-(int(r.get("strict_top1_success_under_annotation_non_gt_hub_count") or 0)), -(int(r.get("gt_trajectory_count") or 0))))

    def _compact_example(row: Mapping[str, Any]) -> Dict[str, Any]:
        keys = [
            "clip_id", "video_id", "trajectory_id", "tid", "gt_raw_id", "gt_name",
            "risk_sources", "P_person_given_class_gt", "max_P_hub_given_class_gt", "gt_alone_rate", "weak_alone_rate",
            "annotation_known", "annotation_has_non_gt_hub", "annotation_has_person_non_gt", "annotation_non_gt_hub_raw_ids",
            "model_non_gt_hub_pressure_present", "model_non_gt_hub_pressure_types", "model_candidate_non_gt_hub_raw_ids",
            "annotation_and_model_non_gt_hub_pressure",
            "active_rescue", "rank_topK_rescue", "R_GT_winner_rescue", "strict_top1_rescue",
            "gt_mining_rank", "final_winner_raw_id", "final_winner_name", "top_suppressor_raw_id", "top_suppressor_name",
            "candidate_ids_known", "candidate_ids_extra",
        ]
        return {k: row.get(k) for k in keys if k in row}

    example_rows = [_compact_example(r) for r in sorted(
        examples,
        key=lambda r: (
            0 if bool(r.get("strict_top1_rescue")) and bool(r.get("annotation_has_non_gt_hub")) and bool(r.get("model_non_gt_hub_pressure_present")) else 1,
            0 if bool(r.get("strict_top1_rescue")) and bool(r.get("annotation_has_non_gt_hub")) else 1,
            999999 if _safe_int(r.get("gt_mining_rank")) is None else int(r.get("gt_mining_rank")),
            str(r.get("gt_name")),
        ),
    )[:max(1, int(top_examples))]]

    summary = {
        "status": "PASS",
        "definition": "Row-level audit that checks whether each high-risk GT row's annotation unit actually contains a non-GT hub class. hub_raw_ids are stripped of the current gt_raw_id before annotation/model pressure tests.",
        "current_k": int(current_k),
        "risk_threshold": float(risk_threshold),
        "low_alone_threshold": float(low_alone_threshold),
        "row_count": int(len(joined)),
        "annotation_unit_count": int(len(unit_class_map)),
        "high_risk_class_count": int(len(high_risk_class_ids)),
        "high_risk_row_count": int(len(high_risk_rows)),
        "annotation_known_row_count": int(len(annotation_known_rows)),
        "annotation_unknown_row_count": int(len(ann_unknown_rows)),
        "annotation_non_gt_hub_row_count": int(len(ann_has_rows)),
        "annotation_non_gt_hub_rate_among_high_risk_rows": float(len(ann_has_rows) / max(len(high_risk_rows), 1)),
        "annotation_no_non_gt_hub_row_count": int(len(ann_no_rows)),
        "annotation_no_non_gt_hub_rate_among_high_risk_rows": float(len(ann_no_rows) / max(len(high_risk_rows), 1)),
        "model_non_gt_hub_pressure_row_count": int(len(model_pressure_rows)),
        "model_non_gt_hub_pressure_rate_among_high_risk_rows": float(len(model_pressure_rows) / max(len(high_risk_rows), 1)),
        "annotation_and_model_non_gt_hub_pressure_row_count": int(len(ann_and_model_rows)),
        "annotation_and_model_non_gt_hub_pressure_rate_among_high_risk_rows": float(len(ann_and_model_rows) / max(len(high_risk_rows), 1)),
        "strict_top1_success_under_annotation_non_gt_hub_count": int(len(strict_ann_has_rows)),
        "strict_top1_success_under_annotation_non_gt_hub_rate": float(len(strict_ann_has_rows) / max(len(ann_has_rows), 1)) if ann_has_rows else None,
        "strict_top1_success_without_annotation_non_gt_hub_count": int(len(strict_ann_no_rows)),
        "strict_top1_success_without_annotation_non_gt_hub_rate": float(len(strict_ann_no_rows) / max(len(ann_no_rows), 1)) if ann_no_rows else None,
        "strict_top1_success_under_annotation_and_model_non_gt_hub_pressure_count": int(len(strict_ann_and_model_rows)),
        "strict_top1_success_under_annotation_and_model_non_gt_hub_pressure_rate": float(len(strict_ann_and_model_rows) / max(len(ann_and_model_rows), 1)) if ann_and_model_rows else None,
        "collapse_failure_under_annotation_non_gt_hub_count": int(len(fail_ann_has_rows)),
        "collapse_failure_under_annotation_non_gt_hub_rate": float(len(fail_ann_has_rows) / max(len(ann_has_rows), 1)) if ann_has_rows else None,
        "answer_key": {
            "if_success_without_annotation_non_gt_hub_dominates": "Successful rows mainly avoid annotation-level non-GT hub co-occurrence.",
            "if_success_under_annotation_non_gt_hub_exists": "Some rows truly succeed despite GT annotation non-GT hub co-occurrence.",
            "strictest_success": "strict_top1_success_under_annotation_and_model_non_gt_hub_pressure",
        },
    }
    return {
        "summary": summary,
        "contrast_rows": contrast_rows,
        "class_rows": class_rows,
        "example_rows": example_rows,
    }

def _text_semantic_confusion_payload(rows: Sequence[Mapping[str, Any]], *, records_by_raw: Mapping[int, Mapping[str, Any]], top_n: int, neighbor_topk: int, sim_threshold: float) -> Dict[str, Any]:
    joined = _rows_joined(rows)
    failure_rows = [r for r in joined if not bool(r.get("final_top1_is_gt")) and _safe_int(r.get("final_winner_raw_id")) is not None]

    def _rank_is_neighbor(r: Mapping[str, Any]) -> bool:
        rank = _safe_int(r.get("final_winner_text_neighbor_rank_to_gt"))
        return bool(rank is not None and int(rank) <= int(neighbor_topk))

    def _sim_above_threshold(r: Mapping[str, Any]) -> bool:
        sim = _safe_float(r.get("final_winner_text_sim_to_gt"))
        return bool(sim is not None and float(sim) >= float(sim_threshold))

    semantic_rows = [r for r in failure_rows if _rank_is_neighbor(r)]
    sim_threshold_rows = [r for r in failure_rows if _sim_above_threshold(r)]
    sim_threshold_only_rows = [r for r in failure_rows if _sim_above_threshold(r) and not _rank_is_neighbor(r)]
    sim_rank_inconsistent_rows = [
        r for r in failure_rows
        if (_safe_float(r.get("final_winner_text_sim_to_gt")) is not None and float(r.get("final_winner_text_sim_to_gt")) >= 0.999)
        and (not _rank_is_neighbor(r))
    ]

    pair_counter: Counter = Counter()
    pair_examples: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in semantic_rows:
        gt = _safe_int(r.get("gt_raw_id"))
        wrong = _safe_int(r.get("final_winner_raw_id"))
        if gt is None or wrong is None:
            continue
        key = (int(gt), int(wrong))
        pair_counter[key] += 1
        pair_examples.setdefault(key, {
            "gt_raw_id": int(gt),
            "gt_name": _class_label(int(gt), records_by_raw).get("name"),
            "wrong_raw_id": int(wrong),
            "wrong_name": _class_label(int(wrong), records_by_raw).get("name"),
            "text_sim": r.get("final_winner_text_sim_to_gt"),
            "text_neighbor_rank": r.get("final_winner_text_neighbor_rank_to_gt"),
        })

    pairs: List[Dict[str, Any]] = []
    for key, count in pair_counter.most_common(max(1, int(top_n))):
        item = dict(pair_examples.get(key, {}))
        item["count"] = int(count)
        item["rate_among_semantic_failures"] = float(count / max(len(semantic_rows), 1))
        pairs.append(item)

    def _rank_neighbor_rate(seq: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not seq:
            return None
        return _rate_bools([_rank_is_neighbor(r) for r in seq])

    def _sim_threshold_rate(seq: Sequence[Mapping[str, Any]]) -> Optional[float]:
        if not seq:
            return None
        return _rate_bools([_sim_above_threshold(r) for r in seq])

    person_failures = [r for r in failure_rows if _safe_int(r.get("final_winner_raw_id")) == 773]
    nonperson_failures = [r for r in failure_rows if _safe_int(r.get("final_winner_raw_id")) not in (None, 773)]

    sanity_examples: List[Dict[str, Any]] = []
    for r in sim_rank_inconsistent_rows[:max(1, min(int(top_n), 50))]:
        gt = _safe_int(r.get("gt_raw_id"))
        wrong = _safe_int(r.get("final_winner_raw_id"))
        sanity_examples.append({
            "gt_raw_id": gt,
            "gt_name": _class_label(int(gt), records_by_raw).get("name") if gt is not None else None,
            "wrong_raw_id": wrong,
            "wrong_name": _class_label(int(wrong), records_by_raw).get("name") if wrong is not None else None,
            "text_sim": r.get("final_winner_text_sim_to_gt"),
            "text_neighbor_rank": r.get("final_winner_text_neighbor_rank_to_gt"),
        })

    return {
        "status": "PASS",
        "definition": {
            "semantic_neighbor_failure": "final_winner_text_neighbor_rank_to_gt <= text_neighbor_topk",
            "sim_threshold": "reported only as sanity; it is not used to mark semantic_neighbor_failure",
        },
        "neighbor_topk": int(neighbor_topk),
        "sim_threshold": float(sim_threshold),
        "failure_count": int(len(failure_rows)),
        "semantic_neighbor_failure_count": int(len(semantic_rows)),
        "semantic_neighbor_failure_rate": float(len(semantic_rows) / max(len(failure_rows), 1)),
        "sim_threshold_count": int(len(sim_threshold_rows)),
        "sim_threshold_rate": float(len(sim_threshold_rows) / max(len(failure_rows), 1)),
        "sim_threshold_only_count": int(len(sim_threshold_only_rows)),
        "sim_threshold_only_rate": float(len(sim_threshold_only_rows) / max(len(failure_rows), 1)),
        "sim_rank_inconsistency_count": int(len(sim_rank_inconsistent_rows)),
        "sim_rank_inconsistency_rate": float(len(sim_rank_inconsistent_rows) / max(len(failure_rows), 1)),
        "person_error_text_neighbor_rate": _rank_neighbor_rate(person_failures),
        "nonperson_error_text_neighbor_rate": _rank_neighbor_rate(nonperson_failures),
        "person_error_sim_threshold_rate": _sim_threshold_rate(person_failures),
        "nonperson_error_sim_threshold_rate": _sim_threshold_rate(nonperson_failures),
        "top_confused_pairs": pairs,
        "sim_rank_inconsistency_examples": sanity_examples,
    }

def _hub_prior_beta_sweep_payload(*, rows: Sequence[Mapping[str, Any]], clip_mining: Mapping[int, Mapping[str, Any]], vocab_ids: Sequence[int], selected_extra_counter: Counter, beta_values: Sequence[float], hub_raw_ids: Sequence[int], current_k: int) -> Dict[str, Any]:
    joined = _rows_joined(rows)
    vocab_arr = np.asarray([int(x) for x in vocab_ids], dtype=np.int64)
    hub_set = {int(x) for x in hub_raw_ids}
    counts = np.asarray([float(selected_extra_counter.get(int(rid), 0)) for rid in vocab_arr.tolist()], dtype=np.float64)
    mean_count = float(np.mean(counts[counts > 0])) if np.any(counts > 0) else 1.0
    hub_prior = np.log1p(counts / max(mean_count, 1e-12))
    by_beta: List[Dict[str, Any]] = []
    for beta in beta_values:
        b = float(beta)
        active_hits = lost_existing = newly_gained = hub_selected = hub_active_gt_missing = 0
        selected_counter: Counter = Counter()
        for row in joined:
            gt = _safe_int(row.get("gt_raw_id"))
            clip_id = _safe_int(row.get("clip_id"))
            if gt is None or clip_id is None or int(clip_id) not in clip_mining:
                continue
            mining = clip_mining[int(clip_id)]
            scores = np.asarray(mining.get("scores"), dtype=np.float64).copy()
            mask = np.asarray(mining.get("candidate_mask"), dtype=bool)
            if scores.size != len(vocab_arr):
                continue
            adjusted = scores - b * hub_prior
            selected = _top_raw_ids(adjusted, vocab_arr.tolist(), mask, max(1, int(current_k)))
            selected_set = {int(x) for x in selected}
            for rid in selected_set:
                selected_counter[int(rid)] += 1
            current_hit = _active_raw_contains(row) is True
            beta_hit = int(gt) in selected_set
            active_hits += int(beta_hit)
            lost_existing += int(current_hit and not beta_hit)
            newly_gained += int((not current_hit) and beta_hit)
            hub_present = bool(any(int(h) in selected_set for h in hub_set)) if hub_set else False
            hub_selected += int(hub_present)
            hub_active_gt_missing += int(hub_present and not beta_hit)
        denom = max(len(joined), 1)
        by_beta.append({
            "beta": b,
            "target_count": int(len(joined)),
            "active_raw_membership_rate": float(active_hits / denom),
            "active_raw_contains_true": int(active_hits),
            "lost_existing": int(lost_existing),
            "newly_gained": int(newly_gained),
            "net_gain": int(newly_gained - lost_existing),
            "hub_selected_row_count": int(hub_selected),
            "hub_selected_row_rate": float(hub_selected / denom),
            "hub_active_and_gt_missing_count": int(hub_active_gt_missing),
            "hub_active_and_gt_missing_rate": float(hub_active_gt_missing / denom),
            "top_selected_after_rerank": [{"raw_id": int(rid), "count": int(cnt)} for rid, cnt in selected_counter.most_common(20)],
        })
    return {
        "status": "PASS",
        "definition": "post-hoc only: score_beta(c)=clip_max_score(c)-beta*log1p(selected_count(c)/mean_nonzero_selected_count)",
        "current_k": int(current_k),
        "hub_raw_ids": [int(x) for x in hub_raw_ids],
        "beta_values": [float(x) for x in beta_values],
        "by_beta": by_beta,
    }


def _cooccurrence_payload_from_units(
    *,
    units: Sequence[Mapping[str, Any]],
    hub_raw_ids: Sequence[int],
    records_by_raw: Mapping[int, Mapping[str, Any]],
    top_n: int,
    source: str,
    unit_level: str,
) -> Dict[str, Any]:
    hub_set = {int(x) for x in hub_raw_ids}
    class_sets: List[set[int]] = []
    for unit in units:
        vals = {int(x) for x in _unique_ints(unit.get("class_ids"))}
        if vals:
            class_sets.append(vals)
    total = len(class_sets)
    any_hub_units = [cs for cs in class_sets if cs & hub_set]
    all_hubs_count = sum(1 for cs in class_sets if hub_set and hub_set.issubset(cs))
    per_hub: List[Dict[str, Any]] = []
    for hub in sorted(hub_set):
        present_units = [cs for cs in class_sets if hub in cs]
        present = len(present_units)
        co_counter: Counter = Counter()
        other_hub_count = 0
        alone_count = 0
        num_classes_values: List[float] = []
        for cs in present_units:
            num_classes_values.append(float(len(cs)))
            if len(cs) == 1:
                alone_count += 1
            if (cs - {hub}) & hub_set:
                other_hub_count += 1
            for c in sorted(cs - {hub}):
                co_counter[int(c)] += 1
        top_co = []
        for cid, cnt in co_counter.most_common(max(1, int(top_n))):
            item = _class_label(int(cid), records_by_raw)
            item["count"] = int(cnt)
            item["P_other_given_hub"] = float(cnt / max(present, 1))
            other_present = sum(1 for cs in class_sets if int(cid) in cs)
            item["P_hub_given_other"] = float(cnt / max(other_present, 1)) if other_present else None
            top_co.append(item)
        hitem = _class_label(int(hub), records_by_raw)
        hitem.update({
            "present_count": int(present),
            "present_rate": float(present / max(total, 1)),
            "alone_count": int(alone_count),
            "alone_rate_among_hub_units": float(alone_count / max(present, 1)) if present else None,
            "with_other_classes_count": int(present - alone_count),
            "with_other_classes_rate_among_hub_units": float((present - alone_count) / max(present, 1)) if present else None,
            "with_other_hub_classes_count": int(other_hub_count),
            "with_other_hub_classes_rate_among_hub_units": float(other_hub_count / max(present, 1)) if present else None,
            "mean_num_classes_when_present": _mean(num_classes_values),
            "top_cooccurring_classes": top_co,
        })
        per_hub.append(hitem)
    return {
        "status": "PASS" if total else "NO_UNITS",
        "source": str(source),
        "unit_level": str(unit_level),
        "definition": "Each unit contributes a set of raw class ids; hub co-occurrence is measured within the same unit.",
        "hub_raw_ids": [int(x) for x in sorted(hub_set)],
        "unit_count": int(total),
        "any_hub_count": int(len(any_hub_units)),
        "any_hub_rate": float(len(any_hub_units) / max(total, 1)),
        "no_hub_count": int(total - len(any_hub_units)),
        "no_hub_rate": float((total - len(any_hub_units)) / max(total, 1)),
        "all_hubs_together_count": int(all_hubs_count),
        "all_hubs_together_rate": float(all_hubs_count / max(total, 1)),
        "per_hub": per_hub,
    }


def _units_from_rows(rows: Sequence[Mapping[str, Any]], *, unit_key: str, class_key: str = "gt_raw_id") -> List[Dict[str, Any]]:
    grouped: Dict[str, set[int]] = defaultdict(set)
    for r in rows:
        unit_val = r.get(unit_key)
        rid = _safe_int(r.get(class_key))
        if unit_val is None or rid is None:
            continue
        grouped[str(unit_val)].add(int(rid))
    return [{"unit_id": k, "class_ids": sorted(v)} for k, v in grouped.items() if v]


def _weak_units_from_examples(examples: Sequence[Mapping[str, Any]], *, unit_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, set[int]] = defaultdict(set)
    for ex in examples:
        unit_val = ex.get(unit_key)
        if unit_val is None:
            continue
        vals = _unique_ints(ex.get("observed_raw_ids")) or _unique_ints(ex.get("candidate_ids_known"))
        for rid in vals:
            grouped[str(unit_val)].add(int(rid))
    return [{"unit_id": k, "class_ids": sorted(v)} for k, v in grouped.items() if v]


def _annotation_cooccurrence_units(runtime_output_root: Path, dataset_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    split = "train" if str(dataset_name) == "lvvis_train_base" else "val"
    root = Path(runtime_output_root)
    annotation_roots = [
        root / "videocutler" / "datasets" / "LV-VIS" / "annotations",
        root / "datasets" / "LV-VIS" / "annotations",
        root.parent / "wsovvis_asserts" / "dataset" / "LV-VIS" / "annotations",
        root.parent / "wsovvis_asserts" / "datasets" / "LV-VIS" / "annotations",
        Path("/home/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations"),
        Path("/mnt/sda/zyy/code/wsovvis_asserts/dataset/LV-VIS/annotations"),
    ]
    names = [f"{split}_instances.json", f"instances_{split}.json", "train_instances.json" if split == "train" else "val_instances.json"]
    candidates: List[Path] = []
    for ar in annotation_roots:
        for name in names:
            candidates.append(ar / name)
    seen: set[str] = set()
    for path in candidates:
        if str(path) in seen or not path.is_file():
            continue
        seen.add(str(path))
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        annotations = payload.get("annotations") if isinstance(payload, Mapping) else None
        if not isinstance(annotations, list):
            continue
        grouped: Dict[str, set[int]] = defaultdict(set)
        used = 0
        for ann in annotations:
            if not isinstance(ann, Mapping):
                continue
            cid = _safe_int(ann.get("category_id", ann.get("raw_id", ann.get("class_id"))))
            vid = ann.get("video_id", ann.get("vid", ann.get("video", ann.get("video_idx"))))
            if cid is None or vid is None:
                continue
            grouped[str(vid)].add(int(cid))
            used += 1
        units = [{"unit_id": k, "class_ids": sorted(v)} for k, v in grouped.items() if v]
        if units:
            return units, {"status": "PASS", "path": str(path), "annotation_count_used": int(used), "unit_level": "video_id"}
    return [], {"status": "MISSING_OR_UNREADABLE_ANNOTATIONS", "searched": [str(p) for p in candidates], "unit_level": "video_id"}


def _model_hub_current_payload(rows: Sequence[Mapping[str, Any]], *, hub_raw_ids: Sequence[int], records_by_raw: Mapping[int, Mapping[str, Any]], top_n: int) -> Dict[str, Any]:
    joined = _rows_joined(rows)
    hub_set = {int(x) for x in hub_raw_ids}
    selected_counter: Counter = Counter()
    winner_counter: Counter = Counter()
    wrong_counter: Counter = Counter()
    suppressor_counter: Counter = Counter()
    for r in joined:
        for rid in _active_extra_set(r):
            if rid in hub_set:
                selected_counter[int(rid)] += 1
        fw = _safe_int(r.get("final_winner_raw_id"))
        if fw in hub_set:
            winner_counter[int(fw)] += 1
            if not bool(r.get("final_top1_is_gt")):
                wrong_counter[int(fw)] += 1
        sp = _safe_int(r.get("top_suppressor_raw_id"))
        if sp in hub_set:
            suppressor_counter[int(sp)] += 1
    denom = len(joined)
    per_hub = []
    for hub in sorted(hub_set):
        item = _class_label(int(hub), records_by_raw)
        item.update({
            "selected_count": int(selected_counter.get(hub, 0)),
            "selected_rate": float(selected_counter.get(hub, 0) / max(denom, 1)),
            "winner_count": int(winner_counter.get(hub, 0)),
            "winner_rate": float(winner_counter.get(hub, 0) / max(denom, 1)),
            "wrong_winner_count": int(wrong_counter.get(hub, 0)),
            "wrong_winner_rate": float(wrong_counter.get(hub, 0) / max(denom, 1)),
            "suppressor_count": int(suppressor_counter.get(hub, 0)),
            "suppressor_rate": float(suppressor_counter.get(hub, 0) / max(denom, 1)),
        })
        per_hub.append(item)
    return {
        "status": "PASS",
        "row_universe": "formal_aligned_rows_or_target_rows",
        "row_count": int(denom),
        "per_hub": per_hub,
        "top_hub_selected": _counter_payload_with_rates(selected_counter, records_by_raw=records_by_raw, top_n=top_n, denominator=denom),
        "top_hub_wrong_winner": _counter_payload_with_rates(wrong_counter, records_by_raw=records_by_raw, top_n=top_n, denominator=denom),
        "top_hub_suppressor": _counter_payload_with_rates(suppressor_counter, records_by_raw=records_by_raw, top_n=top_n, denominator=denom),
    }


def _hub_snapshot_timeline_payload(*, run_root: Path, stage: str, target_rows: Sequence[Mapping[str, Any]], hub_raw_ids: Sequence[int], records_by_raw: Mapping[int, Mapping[str, Any]], top_n: int) -> Dict[str, Any]:
    hub_set = {int(x) for x in hub_raw_ids}
    snap_dir = run_root / "train" / stage / "extra_snapshots"
    paths = sorted(snap_dir.glob("epoch_*.jsonl"))
    target_tids = {str(r.get("trajectory_id")) for r in target_rows if str(r.get("trajectory_id", ""))}
    target_clips = {_safe_int(r.get("clip_id")) for r in target_rows if _safe_int(r.get("clip_id")) is not None}
    epochs: List[Dict[str, Any]] = []
    for path in paths:
        row_count = 0
        target_row_count = 0
        any_hub_rows = 0
        hub_counter: Counter = Counter()
        for snap in _iter_jsonl(path):
            row_count += 1
            tid = str(snap.get("trajectory_id", "")).strip()
            clip_id = _safe_int(snap.get("clip_id"))
            is_target = (tid in target_tids) or (clip_id in target_clips)
            if not is_target:
                continue
            target_row_count += 1
            extras = {int(x) for x in _snapshot_extra_ids(snap)}
            hits = extras & hub_set
            if hits:
                any_hub_rows += 1
                for h in hits:
                    hub_counter[int(h)] += 1
        epochs.append({
            "snapshot_path": str(path),
            "snapshot_id": path.stem,
            "snapshot_row_count": int(row_count),
            "target_row_count": int(target_row_count),
            "any_hub_selected_count": int(any_hub_rows),
            "any_hub_selected_rate": float(any_hub_rows / max(target_row_count, 1)),
            "top_hub_selected": _counter_payload_with_rates(hub_counter, records_by_raw=records_by_raw, top_n=top_n, denominator=target_row_count),
        })
    return {"status": "PASS" if paths else "NO_EXTRA_SNAPSHOTS", "snapshot_dir": str(snap_dir), "epoch_count": int(len(paths)), "epochs": epochs}


def _hub_origin_classification_payload(*, gt_payload: Mapping[str, Any], weak_payload: Mapping[str, Any], model_payload: Mapping[str, Any], timeline_payload: Mapping[str, Any], hub_raw_ids: Sequence[int], records_by_raw: Mapping[int, Mapping[str, Any]]) -> Dict[str, Any]:
    def _per_map(payload: Mapping[str, Any], key: str) -> Dict[int, Mapping[str, Any]]:
        out: Dict[int, Mapping[str, Any]] = {}
        for item in payload.get(key, []) if isinstance(payload, Mapping) else []:
            rid = _safe_int(item.get("raw_id"))
            if rid is not None:
                out[int(rid)] = item
        return out
    gt_map = _per_map(gt_payload, "per_hub")
    weak_map = _per_map(weak_payload, "per_hub")
    model_map = _per_map(model_payload, "per_hub")
    rows: List[Dict[str, Any]] = []
    for hub in sorted({int(x) for x in hub_raw_ids}):
        gt = gt_map.get(hub, {})
        wk = weak_map.get(hub, {})
        md = model_map.get(hub, {})
        gt_present = _safe_float(gt.get("present_rate")) or 0.0
        gt_alone = _safe_float(gt.get("alone_rate_among_hub_units"))
        weak_present = _safe_float(wk.get("present_rate")) or 0.0
        model_selected = _safe_float(md.get("selected_rate")) or 0.0
        model_wrong = _safe_float(md.get("wrong_winner_rate")) or 0.0
        flags: List[str] = []
        if gt_present >= 0.20 and (gt_alone is None or gt_alone <= 0.25):
            flags.append("data_cooccurrence_hub")
        if weak_present >= gt_present + 0.05:
            flags.append("weak_label_amplified_hub")
        if model_selected >= max(gt_present, weak_present) + 0.10 or model_wrong >= 0.10:
            flags.append("model_amplified_hub")
        if not flags:
            flags.append("insufficient_evidence_or_low_hub_signal")
        item = _class_label(hub, records_by_raw)
        item.update({
            "gt_present_rate": gt_present,
            "gt_alone_rate_among_hub_units": gt_alone,
            "weak_present_rate": weak_present,
            "model_selected_rate": model_selected,
            "model_wrong_winner_rate": model_wrong,
            "evidence_flags": flags,
        })
        rows.append(item)
    return {
        "status": "PASS",
        "definition": "Heuristic origin flags comparing GT co-occurrence, weak-label co-occurrence, and current model hub selected/winner/suppressor rates. This is an audit, not a training change.",
        "timeline_status": timeline_payload.get("status") if isinstance(timeline_payload, Mapping) else None,
        "rows": rows,
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
    text_features_np = np.asarray(text_vocab_matrix, dtype=np.float64)
    text_norms = np.linalg.norm(text_features_np, axis=1, keepdims=True)
    text_norms[text_norms <= 0.0] = 1.0
    text_features_norm = text_features_np / text_norms
    text_sim_matrix = np.matmul(text_features_norm, text_features_norm.T)

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
        if clip_logits.size:
            argmax_local = np.argmax(clip_logits, axis=0)
            argmax_row_indices = np.asarray(row_indices, dtype=np.int64)[argmax_local]
        else:
            argmax_row_indices = np.full((vocab_count,), -1, dtype=np.int64)
        mask = np.ones((vocab_count,), dtype=bool)
        for raw_id in yprime_by_clip.get(int(clip_id), set()):
            idx = raw_to_index.get(int(raw_id))
            if idx is not None:
                mask[int(idx)] = False
        clip_mining[int(clip_id)] = {
            "scores": max_scores,
            "candidate_mask": mask,
            "argmax_row_indices": argmax_row_indices,
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

        top_suppressor_raw_id = final_winner_raw_id if (not final_top1_is_gt and final_winner_raw_id is not None) else None
        top_suppressor_idx = raw_to_index.get(int(top_suppressor_raw_id)) if top_suppressor_raw_id is not None else None
        top_suppressor_score = float(logits[int(top_suppressor_idx)]) if top_suppressor_idx is not None else None
        top_suppressor_clip_max_score = None
        top_suppressor_clip_argmax_row_index = None
        top_suppressor_clip_argmax_trajectory_id = None
        top_suppressor_clip_argmax_same_trajectory = None
        if top_suppressor_idx is not None and mining_scores.size:
            top_suppressor_clip_max_score = float(mining_scores[int(top_suppressor_idx)])
            argmax_rows = np.asarray(mining.get("argmax_row_indices", np.full((vocab_count,), -1, dtype=np.int64)), dtype=np.int64)
            if argmax_rows.size > int(top_suppressor_idx):
                top_suppressor_clip_argmax_row_index = int(argmax_rows[int(top_suppressor_idx)])
                if 0 <= top_suppressor_clip_argmax_row_index < len(effective_examples):
                    top_suppressor_clip_argmax_trajectory_id = str(effective_examples[top_suppressor_clip_argmax_row_index].get("trajectory_id", ""))
                top_suppressor_clip_argmax_same_trajectory = bool(top_suppressor_clip_argmax_row_index == int(row_index))
        suppressor_beats_gt_on_current_traj = bool(
            top_suppressor_score is not None and gt_score is not None and float(top_suppressor_score) > float(gt_score)
        )
        same_trajectory_confusion = bool(
            top_suppressor_raw_id is not None
            and suppressor_beats_gt_on_current_traj
            and top_suppressor_clip_argmax_same_trajectory is True
        )
        mixed_confusion = bool(
            top_suppressor_raw_id is not None
            and suppressor_beats_gt_on_current_traj
            and top_suppressor_clip_argmax_same_trajectory is False
        )
        other_trajectory_hijack = bool(
            top_suppressor_raw_id is not None
            and (not suppressor_beats_gt_on_current_traj)
            and top_suppressor_clip_argmax_same_trajectory is False
        )

        final_winner_text_sim_to_gt = None
        final_winner_text_neighbor_rank_to_gt = None
        final_winner_is_text_neighbor_topk = None
        if gt_idx is not None and final_winner_raw_id is not None and int(final_winner_raw_id) in raw_to_index:
            fw_idx = int(raw_to_index[int(final_winner_raw_id)])
            sims = np.asarray(text_sim_matrix[int(gt_idx)], dtype=np.float64).copy()
            final_winner_text_sim_to_gt = float(sims[fw_idx])
            sims[int(gt_idx)] = -np.inf
            order_sim = np.argsort(-sims, kind="stable")
            pos = np.where(order_sim == fw_idx)[0]
            if pos.size:
                final_winner_text_neighbor_rank_to_gt = int(pos[0]) + 1
                final_winner_is_text_neighbor_topk = bool(final_winner_text_neighbor_rank_to_gt <= int(config.text_neighbor_topk))

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
            "final_winner_text_sim_to_gt": final_winner_text_sim_to_gt,
            "final_winner_text_neighbor_rank_to_gt": final_winner_text_neighbor_rank_to_gt,
            "final_winner_is_text_neighbor_topk": final_winner_is_text_neighbor_topk,
            "top_suppressor_raw_id": top_suppressor_raw_id,
            "top_suppressor_score_on_current_traj": top_suppressor_score,
            "top_suppressor_clip_max_score": top_suppressor_clip_max_score,
            "top_suppressor_clip_argmax_row_index": top_suppressor_clip_argmax_row_index,
            "top_suppressor_clip_argmax_trajectory_id": top_suppressor_clip_argmax_trajectory_id,
            "top_suppressor_clip_argmax_same_trajectory": top_suppressor_clip_argmax_same_trajectory,
            "suppressor_beats_gt_on_current_traj": suppressor_beats_gt_on_current_traj,
            "same_trajectory_confusion": same_trajectory_confusion,
            "other_trajectory_hijack": other_trajectory_hijack,
            "mixed_confusion": mixed_confusion,
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
    weak_vocab_raw_ids = _weak_vocab_raw_ids_from_examples(effective_examples)
    formal_aligned_reachable = _reachable_unobserved_payload(
        formal_aligned_rows if formal_aligned_rows else target_rows,
        weak_vocab_raw_ids=weak_vocab_raw_ids,
    )

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

    taxonomy_rows = formal_aligned_rows if formal_aligned_rows else target_rows
    active_raw_conversion_payload = _active_raw_conversion_payload(taxonomy_rows, hub_raw_ids=config.hub_raw_ids)
    same_vs_other_payload = _same_vs_other_payload(taxonomy_rows, records_by_raw=records_by_raw, top_n=config.top_classes)
    rank_bucket_by_class_rows = _rank_bucket_payload_by_class(taxonomy_rows, records_by_raw=records_by_raw)
    blind_spot_toplist_rows = list(rank_bucket_by_class_rows)
    text_semantic_payload = _text_semantic_confusion_payload(
        taxonomy_rows,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
        neighbor_topk=config.text_neighbor_topk,
        sim_threshold=config.text_neighbor_sim_threshold,
    )
    hub_beta_payload = _hub_prior_beta_sweep_payload(
        rows=taxonomy_rows,
        clip_mining=clip_mining,
        vocab_ids=vocab_ids,
        selected_extra_counter=selected_extra_counter,
        beta_values=config.hub_beta_values,
        hub_raw_ids=config.hub_raw_ids,
        current_k=current_k,
    )

    annotation_units, annotation_meta = _annotation_cooccurrence_units(runtime_output_root, str(config.dataset_name))
    annotation_gt_hub_payload = _cooccurrence_payload_from_units(
        units=annotation_units,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
        source=f"lvvis_annotation:{annotation_meta.get('path', 'missing')}",
        unit_level=str(annotation_meta.get("unit_level", "video_id")),
    )
    annotation_gt_hub_payload["annotation_meta"] = annotation_meta
    sidecar_clip_hub_payload = _cooccurrence_payload_from_units(
        units=_units_from_rows(row_diagnostics, unit_key="clip_id", class_key="gt_raw_id"),
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
        source="trajectory_gt_sidecar_from_effective_examples",
        unit_level="clip_id",
    )
    weak_label_hub_payload = _cooccurrence_payload_from_units(
        units=_weak_units_from_examples(effective_examples, unit_key="clip_id"),
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
        source="stage_effective_examples_observed_or_candidate_known",
        unit_level="clip_id",
    )
    fully_missed_candidate_ids = [
        int(r["raw_id"])
        for r in blind_spot_toplist_rows
        if str(r.get("blind_spot_type")) == "fully_missed_blind_spot" and _safe_int(r.get("raw_id")) is not None
    ]
    full_class_gt_cooccurrence_payload = _class_cooccurrence_payload_from_units(
        units=annotation_units if annotation_units else _units_from_rows(row_diagnostics, unit_key="clip_id", class_key="gt_raw_id"),
        target_raw_ids=fully_missed_candidate_ids,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
        source=(f"lvvis_annotation:{annotation_meta.get('path', 'missing')}" if annotation_units else "trajectory_gt_sidecar_from_effective_examples"),
        unit_level=str(annotation_meta.get("unit_level", "video_id")) if annotation_units else "clip_id",
    )
    full_class_weak_cooccurrence_payload = _class_cooccurrence_payload_from_units(
        units=_weak_units_from_examples(effective_examples, unit_key="clip_id"),
        target_raw_ids=fully_missed_candidate_ids,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
        source="stage_effective_examples_observed_or_candidate_known",
        unit_level="clip_id",
    )
    fully_missed_class_report_rows = _fully_missed_class_report_rows(
        rows=taxonomy_rows,
        blind_spot_rows=blind_spot_toplist_rows,
        gt_class_cooccurrence=full_class_gt_cooccurrence_payload,
        weak_class_cooccurrence=full_class_weak_cooccurrence_payload,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        strong_hub_cooccurrence_threshold=float(config.strong_hub_cooccurrence_threshold),
        weak_unobservable_present_threshold=float(config.weak_unobservable_present_threshold),
        weak_unobservable_alone_threshold=float(config.weak_unobservable_alone_threshold),
    )
    fully_missed_class_report_payload = _fully_missed_class_report_payload(fully_missed_class_report_rows)
    fully_missed_trajectory_weighted_payload = _fully_missed_trajectory_weighted_payload(fully_missed_class_report_rows)

    all_taxonomy_gt_ids = sorted({
        int(x)
        for x in (_safe_int(r.get("gt_raw_id")) for r in taxonomy_rows)
        if x is not None
    })
    hub_collapse_all_class_gt_cooccurrence_payload = _class_cooccurrence_payload_from_units(
        units=annotation_units if annotation_units else _units_from_rows(row_diagnostics, unit_key="clip_id", class_key="gt_raw_id"),
        target_raw_ids=all_taxonomy_gt_ids,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
        source=(f"lvvis_annotation:{annotation_meta.get('path', 'missing')}" if annotation_units else "trajectory_gt_sidecar_from_effective_examples"),
        unit_level=str(annotation_meta.get("unit_level", "video_id")) if annotation_units else "clip_id",
    )
    hub_collapse_all_class_weak_cooccurrence_payload = _class_cooccurrence_payload_from_units(
        units=_weak_units_from_examples(effective_examples, unit_key="clip_id"),
        target_raw_ids=all_taxonomy_gt_ids,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
        source="stage_effective_examples_observed_or_candidate_known",
        unit_level="clip_id",
    )
    hub_collapse_rescue_payload = _hub_collapse_rescue_audit_payload(
        rows=taxonomy_rows,
        gt_class_cooccurrence=hub_collapse_all_class_gt_cooccurrence_payload,
        weak_class_cooccurrence=hub_collapse_all_class_weak_cooccurrence_payload,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        risk_threshold=float(config.hub_collapse_risk_threshold),
        low_alone_threshold=float(config.hub_collapse_low_alone_threshold),
        current_k=int(current_k),
        top_examples=int(config.hub_collapse_top_examples),
    )
    annotation_non_gt_hub_rescue_payload = _annotation_non_gt_hub_rescue_audit_payload(
        rows=taxonomy_rows,
        annotation_units=annotation_units if annotation_units else _units_from_rows(row_diagnostics, unit_key="clip_id", class_key="gt_raw_id"),
        gt_class_cooccurrence=hub_collapse_all_class_gt_cooccurrence_payload,
        weak_class_cooccurrence=hub_collapse_all_class_weak_cooccurrence_payload,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        risk_threshold=float(config.hub_collapse_risk_threshold),
        low_alone_threshold=float(config.hub_collapse_low_alone_threshold),
        current_k=int(current_k),
        top_examples=int(config.hub_collapse_top_examples),
    )

    model_hub_current_payload = _model_hub_current_payload(
        taxonomy_rows,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
    )
    hub_timeline_payload = _hub_snapshot_timeline_payload(
        run_root=run_root,
        stage=str(config.stage),
        target_rows=taxonomy_rows,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
        top_n=config.top_classes,
    )
    hub_origin_payload = _hub_origin_classification_payload(
        gt_payload=annotation_gt_hub_payload if annotation_gt_hub_payload.get("status") == "PASS" else sidecar_clip_hub_payload,
        weak_payload=weak_label_hub_payload,
        model_payload=model_hub_current_payload,
        timeline_payload=hub_timeline_payload,
        hub_raw_ids=config.hub_raw_ids,
        records_by_raw=records_by_raw,
    )

    failure_taxonomy_summary = {
        "status": "PASS",
        "row_universe": "formal_aligned_rows" if formal_aligned_rows else "diagnostic_base_unobserved_rows",
        "row_count": int(len(taxonomy_rows)),
        "active_raw_conversion": active_raw_conversion_payload,
        "same_vs_other_hijack": same_vs_other_payload,
        "text_semantic_confusion": text_semantic_payload,
        "hub_prior_beta_sweep": hub_beta_payload,
        "gt_hub_cooccurrence": annotation_gt_hub_payload,
        "trajectory_sidecar_gt_hub_cooccurrence": sidecar_clip_hub_payload,
        "weak_label_hub_cooccurrence": weak_label_hub_payload,
        "model_hub_current_stage": model_hub_current_payload,
        "hub_formation_timeline": hub_timeline_payload,
        "hub_origin_classification": hub_origin_payload,
        "full_class_gt_cooccurrence": full_class_gt_cooccurrence_payload,
        "full_class_weak_label_cooccurrence": full_class_weak_cooccurrence_payload,
        "fully_missed_blind_spot_class_report": fully_missed_class_report_payload,
        "fully_missed_blind_spot_trajectory_weighted_summary": fully_missed_trajectory_weighted_payload,
        "hub_collapse_rescue_audit": hub_collapse_rescue_payload.get("summary", {}),
        "annotation_non_gt_hub_rescue_audit": annotation_non_gt_hub_rescue_payload.get("summary", {}),
        "blind_spot_type_histogram": dict(Counter(str(r.get("blind_spot_type")) for r in blind_spot_toplist_rows)),
    }

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
        "formal_aligned_reachable_summary": formal_aligned_reachable,
        "class_name_mapping": class_name_meta,
        "formal_summary_existing": formal_summary,
        "extra_probe_summary_existing": existing_probe,
        "self_check": {
            "gt_in_extra_rate_vs_existing_probe": {
                "diagnosis": recall_curve["actual_active_gt_in_extra_rate"],
                "existing_probe_base_unobserved": (((existing_probe.get("summary") or {}).get("by_split") or {}).get("base_unobserved") or {}).get("gt_in_extra_candidate_rate") if isinstance(existing_probe, Mapping) else None,
            }
        },
        "failure_taxonomy_enabled": {
            "emit_failure_taxonomy": bool(config.emit_failure_taxonomy),
            "emit_active_raw_conversion": bool(config.emit_active_raw_conversion),
            "emit_same_vs_other_hijack": bool(config.emit_same_vs_other_hijack),
            "emit_text_semantic_confusion": bool(config.emit_text_semantic_confusion),
            "emit_hub_prior_beta_sweep": bool(config.emit_hub_prior_beta_sweep),
            "emit_hub_formation_timeline": bool(config.emit_hub_formation_timeline),
            "emit_gt_cooccurrence": bool(config.emit_gt_cooccurrence),
            "emit_weak_label_cooccurrence": bool(config.emit_weak_label_cooccurrence),
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
    takeaways.append(f"- primary base_unobserved_reachable count: `{formal_aligned_reachable.get('base_unobserved_reachable', {}).get('count')}`")
    takeaways.append(f"- primary base_unobserved_reachable top1: `{formal_aligned_reachable.get('base_unobserved_reachable', {}).get('final_top1_rate')}`")
    takeaways.append(f"- primary base_unobserved_reachable GT-in-extra: `{formal_aligned_reachable.get('base_unobserved_reachable', {}).get('gt_in_extra_rate')}`")
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
        "formal_aligned_reachable_summary": output_dir / "formal_aligned_reachable_summary.json",
        "formal_aligned_r_to_logit_transfer_gap": output_dir / "formal_aligned_r_to_logit_transfer_gap.json",
        "formal_aligned_row_diagnostics": output_dir / "formal_aligned_row_diagnostics.jsonl",
        "top_selected_extra_classes_named": output_dir / "top_selected_extra_classes_named.json",
        "top_wrong_extra_winner_classes_named": output_dir / "top_wrong_extra_winner_classes_named.json",
        "top_gt_suppressor_classes_named": output_dir / "top_gt_suppressor_classes_named.json",
        "iou_bucket_report": output_dir / "iou_bucket_report.json",
        "missing_gt_examples": output_dir / "missing_gt_examples.jsonl",
        "row_diagnostics": output_dir / "row_diagnostics.jsonl",
        "diagnosis_takeaways": output_dir / "diagnosis_takeaways.md",
        "formal_aligned_active_raw_conversion": output_dir / "formal_aligned_active_raw_conversion.json",
        "formal_aligned_failure_taxonomy_summary": output_dir / "formal_aligned_failure_taxonomy_summary.json",
        "formal_aligned_same_vs_other_hijack": output_dir / "formal_aligned_same_vs_other_hijack.json",
        "formal_aligned_text_semantic_confusion": output_dir / "formal_aligned_text_semantic_confusion.json",
        "formal_aligned_rank_bucket_by_class": output_dir / "formal_aligned_rank_bucket_by_class.csv",
        "formal_aligned_failure_taxonomy_by_class": output_dir / "formal_aligned_failure_taxonomy_by_class.csv",
        "formal_aligned_hub_prior_beta_sweep": output_dir / "formal_aligned_hub_prior_beta_sweep.json",
        "formal_aligned_gt_hub_cooccurrence": output_dir / "formal_aligned_gt_hub_cooccurrence.json",
        "formal_aligned_trajectory_sidecar_gt_hub_cooccurrence": output_dir / "formal_aligned_trajectory_sidecar_gt_hub_cooccurrence.json",
        "formal_aligned_weak_label_hub_cooccurrence": output_dir / "formal_aligned_weak_label_hub_cooccurrence.json",
        "formal_aligned_model_hub_current_stage": output_dir / "formal_aligned_model_hub_current_stage.json",
        "formal_aligned_hub_formation_timeline": output_dir / "formal_aligned_hub_formation_timeline.json",
        "formal_aligned_hub_origin_classification": output_dir / "formal_aligned_hub_origin_classification.json",
        "formal_aligned_full_class_gt_cooccurrence": output_dir / "formal_aligned_full_class_gt_cooccurrence.json",
        "formal_aligned_full_class_weak_label_cooccurrence": output_dir / "formal_aligned_full_class_weak_label_cooccurrence.json",
        "formal_aligned_fully_missed_blind_spot_class_report": output_dir / "formal_aligned_fully_missed_blind_spot_class_report.json",
        "formal_aligned_fully_missed_blind_spot_class_report_csv": output_dir / "formal_aligned_fully_missed_blind_spot_class_report.csv",
        "formal_aligned_fully_missed_blind_spot_trajectory_weighted_summary": output_dir / "formal_aligned_fully_missed_blind_spot_trajectory_weighted_summary.json",
        "formal_aligned_fully_missed_blind_spot_trajectory_weighted_summary_csv": output_dir / "formal_aligned_fully_missed_blind_spot_trajectory_weighted_summary.csv",
        "formal_aligned_hub_collapse_rescue_summary": output_dir / "formal_aligned_hub_collapse_rescue_summary.json",
        "formal_aligned_hub_collapse_rescue_class_report": output_dir / "formal_aligned_hub_collapse_rescue_class_report.json",
        "formal_aligned_hub_collapse_rescue_class_report_csv": output_dir / "formal_aligned_hub_collapse_rescue_class_report.csv",
        "formal_aligned_hub_collapse_rescue_row_examples": output_dir / "formal_aligned_hub_collapse_rescue_row_examples.jsonl",
        "formal_aligned_hub_collapse_rescue_success_failure_contrast": output_dir / "formal_aligned_hub_collapse_rescue_success_failure_contrast.json",
        "formal_aligned_hub_collapse_rescue_success_failure_contrast_csv": output_dir / "formal_aligned_hub_collapse_rescue_success_failure_contrast.csv",
        "formal_aligned_annotation_non_gt_hub_rescue_summary": output_dir / "formal_aligned_annotation_non_gt_hub_rescue_summary.json",
        "formal_aligned_annotation_non_gt_hub_rescue_contrast": output_dir / "formal_aligned_annotation_non_gt_hub_rescue_contrast.json",
        "formal_aligned_annotation_non_gt_hub_rescue_contrast_csv": output_dir / "formal_aligned_annotation_non_gt_hub_rescue_contrast.csv",
        "formal_aligned_annotation_non_gt_hub_rescue_class_report": output_dir / "formal_aligned_annotation_non_gt_hub_rescue_class_report.json",
        "formal_aligned_annotation_non_gt_hub_rescue_class_report_csv": output_dir / "formal_aligned_annotation_non_gt_hub_rescue_class_report.csv",
        "formal_aligned_annotation_non_gt_hub_rescue_row_examples": output_dir / "formal_aligned_annotation_non_gt_hub_rescue_row_examples.jsonl",
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
    _write_json(files["formal_aligned_reachable_summary"], formal_aligned_reachable)
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
    if bool(config.emit_active_raw_conversion) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_active_raw_conversion"], active_raw_conversion_payload)
    if bool(config.emit_same_vs_other_hijack) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_same_vs_other_hijack"], same_vs_other_payload)
    if bool(config.emit_text_semantic_confusion) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_text_semantic_confusion"], text_semantic_payload)
    if bool(config.emit_hub_prior_beta_sweep) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_hub_prior_beta_sweep"], hub_beta_payload)
    if bool(config.emit_gt_cooccurrence) or bool(config.emit_hub_formation_timeline) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_gt_hub_cooccurrence"], annotation_gt_hub_payload)
        _write_json(files["formal_aligned_trajectory_sidecar_gt_hub_cooccurrence"], sidecar_clip_hub_payload)
    if bool(config.emit_weak_label_cooccurrence) or bool(config.emit_hub_formation_timeline) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_weak_label_hub_cooccurrence"], weak_label_hub_payload)
    if bool(config.emit_hub_formation_timeline) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_model_hub_current_stage"], model_hub_current_payload)
        _write_json(files["formal_aligned_hub_formation_timeline"], hub_timeline_payload)
        _write_json(files["formal_aligned_hub_origin_classification"], hub_origin_payload)
    if bool(config.emit_full_class_cooccurrence) or bool(config.emit_fully_missed_class_report) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_full_class_gt_cooccurrence"], full_class_gt_cooccurrence_payload)
        _write_json(files["formal_aligned_full_class_weak_label_cooccurrence"], full_class_weak_cooccurrence_payload)
    if bool(config.emit_fully_missed_class_report) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_fully_missed_blind_spot_class_report"], fully_missed_class_report_payload)
        _write_csv(files["formal_aligned_fully_missed_blind_spot_class_report_csv"], fully_missed_class_report_rows)
    if bool(config.emit_fully_missed_trajectory_weighted_report) or bool(config.emit_fully_missed_class_report) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_fully_missed_blind_spot_trajectory_weighted_summary"], fully_missed_trajectory_weighted_payload)
        _write_csv(files["formal_aligned_fully_missed_blind_spot_trajectory_weighted_summary_csv"], fully_missed_trajectory_weighted_payload.get("all_rows_for_csv", []))
    if bool(config.emit_hub_collapse_rescue_audit) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_hub_collapse_rescue_summary"], hub_collapse_rescue_payload.get("summary", {}))
        _write_json(files["formal_aligned_hub_collapse_rescue_class_report"], {"status": "PASS", "rows": hub_collapse_rescue_payload.get("class_rows", [])})
        _write_csv(files["formal_aligned_hub_collapse_rescue_class_report_csv"], hub_collapse_rescue_payload.get("class_rows", []))
        _write_jsonl(files["formal_aligned_hub_collapse_rescue_row_examples"], hub_collapse_rescue_payload.get("example_rows", []))
        _write_json(files["formal_aligned_hub_collapse_rescue_success_failure_contrast"], {"status": "PASS", "rows": hub_collapse_rescue_payload.get("contrast_rows", [])})
        _write_csv(files["formal_aligned_hub_collapse_rescue_success_failure_contrast_csv"], hub_collapse_rescue_payload.get("contrast_rows", []))
    if bool(config.emit_annotation_non_gt_hub_rescue_audit) or bool(config.emit_hub_collapse_rescue_audit) or bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_annotation_non_gt_hub_rescue_summary"], annotation_non_gt_hub_rescue_payload.get("summary", {}))
        _write_json(files["formal_aligned_annotation_non_gt_hub_rescue_contrast"], {"status": "PASS", "rows": annotation_non_gt_hub_rescue_payload.get("contrast_rows", [])})
        _write_csv(files["formal_aligned_annotation_non_gt_hub_rescue_contrast_csv"], annotation_non_gt_hub_rescue_payload.get("contrast_rows", []))
        _write_json(files["formal_aligned_annotation_non_gt_hub_rescue_class_report"], {"status": "PASS", "rows": annotation_non_gt_hub_rescue_payload.get("class_rows", [])})
        _write_csv(files["formal_aligned_annotation_non_gt_hub_rescue_class_report_csv"], annotation_non_gt_hub_rescue_payload.get("class_rows", []))
        _write_jsonl(files["formal_aligned_annotation_non_gt_hub_rescue_row_examples"], annotation_non_gt_hub_rescue_payload.get("example_rows", []))
    if bool(config.emit_failure_taxonomy):
        _write_json(files["formal_aligned_failure_taxonomy_summary"], failure_taxonomy_summary)
        _write_csv(files["formal_aligned_rank_bucket_by_class"], rank_bucket_by_class_rows)
        _write_csv(files["formal_aligned_failure_taxonomy_by_class"], blind_spot_toplist_rows)

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
    p.add_argument("--emit_failure_taxonomy", type=_parse_bool, default=False)
    p.add_argument("--emit_active_raw_conversion", type=_parse_bool, default=False)
    p.add_argument("--emit_same_vs_other_hijack", type=_parse_bool, default=False)
    p.add_argument("--emit_text_semantic_confusion", type=_parse_bool, default=False)
    p.add_argument("--emit_hub_prior_beta_sweep", type=_parse_bool, default=False)
    p.add_argument("--hub_raw_ids", type=_parse_int_tuple, default=(773,))
    p.add_argument("--hub_beta_values", type=_parse_float_tuple, default=(0.0, 0.02, 0.05, 0.10, 0.20, 0.30))
    p.add_argument("--text_neighbor_topk", type=int, default=20)
    p.add_argument("--text_neighbor_sim_threshold", type=float, default=0.65)
    p.add_argument("--emit_hub_formation_timeline", type=_parse_bool, default=False)
    p.add_argument("--emit_gt_cooccurrence", type=_parse_bool, default=False)
    p.add_argument("--emit_weak_label_cooccurrence", type=_parse_bool, default=False)
    p.add_argument("--emit_fully_missed_class_report", type=_parse_bool, default=False)
    p.add_argument("--emit_fully_missed_trajectory_weighted_report", type=_parse_bool, default=False)
    p.add_argument("--emit_hub_collapse_rescue_audit", type=_parse_bool, default=False)
    p.add_argument("--emit_annotation_non_gt_hub_rescue_audit", type=_parse_bool, default=False)
    p.add_argument("--emit_full_class_cooccurrence", type=_parse_bool, default=False)
    p.add_argument("--hub_collapse_risk_threshold", type=float, default=0.5)
    p.add_argument("--hub_collapse_low_alone_threshold", type=float, default=0.1)
    p.add_argument("--hub_collapse_top_examples", type=int, default=200)
    p.add_argument("--strong_hub_cooccurrence_threshold", type=float, default=0.5)
    p.add_argument("--weak_unobservable_present_threshold", type=float, default=0.05)
    p.add_argument("--weak_unobservable_alone_threshold", type=float, default=0.05)
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
        emit_failure_taxonomy=bool(args.emit_failure_taxonomy),
        emit_active_raw_conversion=bool(args.emit_active_raw_conversion),
        emit_same_vs_other_hijack=bool(args.emit_same_vs_other_hijack),
        emit_text_semantic_confusion=bool(args.emit_text_semantic_confusion),
        emit_hub_prior_beta_sweep=bool(args.emit_hub_prior_beta_sweep),
        hub_raw_ids=tuple(int(x) for x in args.hub_raw_ids),
        hub_beta_values=tuple(float(x) for x in args.hub_beta_values),
        text_neighbor_topk=max(1, int(args.text_neighbor_topk)),
        text_neighbor_sim_threshold=float(args.text_neighbor_sim_threshold),
        emit_hub_formation_timeline=bool(args.emit_hub_formation_timeline),
        emit_gt_cooccurrence=bool(args.emit_gt_cooccurrence),
        emit_weak_label_cooccurrence=bool(args.emit_weak_label_cooccurrence),
        emit_fully_missed_class_report=bool(args.emit_fully_missed_class_report),
        emit_fully_missed_trajectory_weighted_report=bool(args.emit_fully_missed_trajectory_weighted_report),
        emit_hub_collapse_rescue_audit=bool(args.emit_hub_collapse_rescue_audit),
        emit_annotation_non_gt_hub_rescue_audit=bool(args.emit_annotation_non_gt_hub_rescue_audit),
        emit_full_class_cooccurrence=bool(args.emit_full_class_cooccurrence),
        hub_collapse_risk_threshold=float(args.hub_collapse_risk_threshold),
        hub_collapse_low_alone_threshold=float(args.hub_collapse_low_alone_threshold),
        hub_collapse_top_examples=max(1, int(args.hub_collapse_top_examples)),
        strong_hub_cooccurrence_threshold=float(args.strong_hub_cooccurrence_threshold),
        weak_unobservable_present_threshold=float(args.weak_unobservable_present_threshold),
        weak_unobservable_alone_threshold=float(args.weak_unobservable_alone_threshold),
    )
    result = run_diagnosis(config)
    print(result["summary_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
