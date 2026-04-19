from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from videocutler.ext_stageb_ovvis.audit.extra_recovery_audit import _load_or_generate_gt_sidecar_lookup
from videocutler.ext_stageb_ovvis.banks.text_bank import read_text_prototype_records
from videocutler.ext_stageb_ovvis.data.g7_phase1_materialization import (
    Phase1MaterializationConfig,
    materialize_phase1_training_samples,
)

Record = Dict[str, Any]
_STAGE_IDS: Tuple[str, ...] = ("prealign", "softem_base", "softem_aug")


def _load_jsonl(path: Path) -> List[Record]:
    if not path.is_file():
        return []
    rows: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Record]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _text_vocab_ids(output_root: Path) -> List[int]:
    records = read_text_prototype_records(output_root / "text_bank" / "text_prototype_records.jsonl")
    return [int(record["raw_id"]) for record in records]


def _resolve_stage_path(output_root: Path, stage_id: str) -> Path:
    if stage_id == "prealign":
        return output_root / "train" / "prealign" / "proxy_records.jsonl"
    if stage_id == "softem_base":
        return output_root / "train" / "softem_base" / "responsibility_records.jsonl"
    if stage_id == "softem_aug":
        return output_root / "train" / "softem_aug" / "responsibility_records.jsonl"
    raise ValueError(f"unsupported stage_id: {stage_id}")


def _sample_lookup(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Record]:
    lookup: Dict[str, Record] = {}
    for sample in samples:
        tid = str(sample.get("trajectory_id", "")).strip()
        if tid:
            lookup[tid] = dict(sample)
    return lookup


def _score_map_from_prealign(row: Mapping[str, Any]) -> Tuple[Dict[int, float], Optional[float], List[int]]:
    proxy_mass = dict(row.get("proxy_mass", {}))
    score_map: Dict[int, float] = {}
    for key, value in proxy_mass.items():
        if str(key) == "unknown":
            continue
        raw_id = _as_int(key)
        score = _as_float(value)
        if raw_id is None or score is None:
            continue
        score_map[int(raw_id)] = float(score)
    unknown_score = _as_float(proxy_mass.get("unknown"))
    domain_ids = sorted(score_map.keys())
    return score_map, unknown_score, domain_ids


def _score_map_from_softem(row: Mapping[str, Any]) -> Tuple[Dict[int, float], Optional[float], List[int]]:
    r_final = dict(row.get("r_final", {}))
    score_map: Dict[int, float] = {}
    for key, value in r_final.items():
        if str(key) == "unknown":
            continue
        raw_id = _as_int(key)
        score = _as_float(value)
        if raw_id is None or score is None:
            continue
        score_map[int(raw_id)] = float(score)
    unknown_score = _as_float(r_final.get("unknown"))
    domain_ids = sorted(score_map.keys())
    return score_map, unknown_score, domain_ids


def _rank_metrics_for_gt(
    *,
    gt_raw_id: int,
    score_map: Mapping[int, float],
    domain_ids: Sequence[int],
    full_vocab_ids: Sequence[int],
    base_vocab_ids: Sequence[int],
) -> Dict[str, Any]:
    domain_set = set(int(x) for x in domain_ids)
    full_vocab = [int(x) for x in full_vocab_ids]
    sentinel_rank = len(full_vocab) + 1
    gt_in_stage_domain = int(gt_raw_id) in domain_set
    sorted_domain = sorted(((int(raw_id), float(score)) for raw_id, score in score_map.items()), key=lambda item: (-item[1], item[0]))
    top1_id = int(sorted_domain[0][0]) if sorted_domain else None
    top1_score = float(sorted_domain[0][1]) if sorted_domain else None
    wrong_top1_is_base = bool(top1_id is not None and top1_id != int(gt_raw_id) and int(top1_id) in set(int(x) for x in base_vocab_ids))

    if not gt_in_stage_domain:
        return {
            "dropped_gt_in_stage_domain": False,
            "dropped_gt_rank": sentinel_rank,
            "dropped_gt_mrr": 0.0,
            "dropped_gt_top1": False,
            "dropped_gt_top5": False,
            "dropped_gt_top10": False,
            "dropped_gt_score": None,
            "dropped_gt_margin_to_best_wrong": None,
            "stage_top1_id": top1_id,
            "stage_top1_score": top1_score,
            "wrong_top1_is_base": wrong_top1_is_base,
        }

    gt_score = float(score_map[int(gt_raw_id)])
    rank = 1 + sum(1 for raw_id, score in sorted_domain if float(score) > gt_score and int(raw_id) != int(gt_raw_id))
    best_wrong = None
    for raw_id, score in sorted_domain:
        if int(raw_id) != int(gt_raw_id):
            best_wrong = float(score)
            break
    margin = float(gt_score - best_wrong) if best_wrong is not None else None
    return {
        "dropped_gt_in_stage_domain": True,
        "dropped_gt_rank": int(rank),
        "dropped_gt_mrr": float(1.0 / rank),
        "dropped_gt_top1": bool(rank <= 1),
        "dropped_gt_top5": bool(rank <= 5),
        "dropped_gt_top10": bool(rank <= 10),
        "dropped_gt_score": float(gt_score),
        "dropped_gt_margin_to_best_wrong": margin,
        "stage_top1_id": top1_id,
        "stage_top1_score": top1_score,
        "wrong_top1_is_base": wrong_top1_is_base,
    }


def build_dropped_gt_attribution_rows(
    *,
    stage_id: str,
    materialized_samples: Sequence[Mapping[str, Any]],
    stage_records: Sequence[Mapping[str, Any]],
    gt_sidecar_lookup: Mapping[str, Mapping[str, Any]],
    full_vocab_ids: Sequence[int],
    base_vocab_ids: Sequence[int],
) -> Tuple[List[Record], Dict[str, Any]]:
    if stage_id not in _STAGE_IDS:
        raise ValueError(f"unsupported stage_id: {stage_id}")
    samples_by_tid = _sample_lookup(materialized_samples)
    rows: List[Record] = []
    invalid_hist = Counter()

    for record in stage_records:
        tid = str(record.get("trajectory_id", "")).strip() or str(record.get("join_key", "")).strip()
        sample = samples_by_tid.get(tid)
        sidecar = gt_sidecar_lookup.get(tid, {})
        gt_raw_id = _as_int(sidecar.get("matched_gt_class_id"))
        observed_raw_ids = [int(x) for x in list(sample.get("observed_raw_ids", []))] if sample else []
        gt_available = bool(sidecar.get("audit_usable", False)) and gt_raw_id is not None
        gt_missing = bool(gt_available and gt_raw_id not in set(observed_raw_ids))

        if stage_id == "prealign":
            score_map, unknown_score, domain_ids = _score_map_from_prealign(record)
        else:
            score_map, unknown_score, domain_ids = _score_map_from_softem(record)

        invalid_reasons: List[str] = []
        if sample is None:
            invalid_reasons.append("missing_materialized_sample")
        if not gt_available:
            invalid_reasons.append("gt_unavailable_for_audit")
        if not gt_missing:
            invalid_reasons.append("gt_not_missing_from_observed")
        for reason in invalid_reasons:
            invalid_hist[str(reason)] += 1

        rank_payload = {
            "dropped_gt_in_stage_domain": False,
            "dropped_gt_rank": None,
            "dropped_gt_mrr": None,
            "dropped_gt_top1": False,
            "dropped_gt_top5": False,
            "dropped_gt_top10": False,
            "dropped_gt_score": None,
            "dropped_gt_margin_to_best_wrong": None,
            "stage_top1_id": None,
            "stage_top1_score": None,
            "wrong_top1_is_base": False,
        }
        if gt_missing and gt_raw_id is not None:
            rank_payload = _rank_metrics_for_gt(
                gt_raw_id=int(gt_raw_id),
                score_map=score_map,
                domain_ids=domain_ids,
                full_vocab_ids=full_vocab_ids,
                base_vocab_ids=base_vocab_ids,
            )

        rows.append(
            {
                "stage_id": str(stage_id),
                "trajectory_id": tid,
                "join_key": tid,
                "clip_id": _as_int(record.get("clip_id")) if _as_int(record.get("clip_id")) is not None else (int(sample.get("clip_id")) if sample and sample.get("clip_id") is not None else None),
                "video_id": _as_int(record.get("video_id")) if _as_int(record.get("video_id")) is not None else (int(sample.get("video_id")) if sample and sample.get("video_id") is not None else None),
                "observed_raw_ids": observed_raw_ids,
                "gt_available_for_audit": gt_available,
                "gt_class_id": gt_raw_id,
                "gt_missing_from_observed": gt_missing,
                "stage_domain_size": int(len(domain_ids)),
                "full_vocab_size": int(len(full_vocab_ids)),
                "unknown_score": unknown_score,
                "invalid_reasons": invalid_reasons,
                **rank_payload,
            }
        )

    summary = summarize_dropped_gt_attribution_rows(rows, stage_id=stage_id)
    summary["invalid_reason_histogram"] = dict(sorted(invalid_hist.items()))
    return rows, summary


def summarize_dropped_gt_attribution_rows(rows: Sequence[Mapping[str, Any]], *, stage_id: str) -> Dict[str, Any]:
    gt_available_rows = [row for row in rows if bool(row.get("gt_available_for_audit"))]
    dropped_rows = [row for row in gt_available_rows if bool(row.get("gt_missing_from_observed"))]
    in_domain_rows = [row for row in dropped_rows if bool(row.get("dropped_gt_in_stage_domain"))]
    ranks = [int(row["dropped_gt_rank"]) for row in dropped_rows if row.get("dropped_gt_rank") is not None]
    mrrs = [float(row["dropped_gt_mrr"]) for row in dropped_rows if row.get("dropped_gt_mrr") is not None]
    margins = [float(row["dropped_gt_margin_to_best_wrong"]) for row in dropped_rows if row.get("dropped_gt_margin_to_best_wrong") is not None]
    scores = [float(row["dropped_gt_score"]) for row in dropped_rows if row.get("dropped_gt_score") is not None]
    wrong_top1_base_count = sum(1 for row in dropped_rows if bool(row.get("wrong_top1_is_base")))
    top1_counter = Counter(str(row.get("stage_top1_id")) for row in dropped_rows if row.get("stage_top1_id") is not None and int(row.get("stage_top1_id")) != int(row.get("gt_class_id")) if row.get("gt_class_id") is not None)

    def _mean(values: Sequence[float]) -> Optional[float]:
        if not values:
            return None
        return float(sum(values) / len(values))

    return {
        "stage_id": str(stage_id),
        "status": "PASS" if rows else "EMPTY",
        "row_count": int(len(rows)),
        "gt_available_row_count": int(len(gt_available_rows)),
        "dropped_gt_count": int(len(dropped_rows)),
        "dropped_gt_in_stage_domain_rate": float(len(in_domain_rows) / len(dropped_rows)) if dropped_rows else None,
        "dropped_gt_mean_rank": _mean(ranks),
        "dropped_gt_mrr": _mean(mrrs),
        "dropped_gt_top1_hit_rate": float(sum(1 for row in dropped_rows if bool(row.get("dropped_gt_top1"))) / len(dropped_rows)) if dropped_rows else None,
        "dropped_gt_top5_hit_rate": float(sum(1 for row in dropped_rows if bool(row.get("dropped_gt_top5"))) / len(dropped_rows)) if dropped_rows else None,
        "dropped_gt_top10_hit_rate": float(sum(1 for row in dropped_rows if bool(row.get("dropped_gt_top10"))) / len(dropped_rows)) if dropped_rows else None,
        "wrong_top1_is_base_rate": float(wrong_top1_base_count / len(dropped_rows)) if dropped_rows else None,
        "dropped_gt_score_mean": _mean(scores),
        "dropped_gt_margin_to_best_wrong_mean": _mean(margins),
        "top_confusion_classes": [{"raw_id": key, "count": int(count)} for key, count in top1_counter.most_common(10)],
    }


def run_dropped_gt_attribution_audit(
    *,
    output_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    smoke: bool,
    smoke_max_trajectories: int,
    stage: str,
) -> Dict[str, Any]:
    if stage not in _STAGE_IDS and stage != "all":
        raise ValueError(f"unsupported stage selection: {stage}")

    output_root = Path(output_root)
    materialized = materialize_phase1_training_samples(
        output_root,
        Phase1MaterializationConfig(
            dataset_name=dataset_name,
            trajectory_source_branch=trajectory_source_branch,
            smoke=smoke,
            smoke_max_trajectories=smoke_max_trajectories,
        ),
    )
    samples = list(materialized["samples"])
    clip_ids = sorted({int(sample.get("clip_id", -1)) for sample in samples if sample.get("clip_id") is not None})
    gt_sidecar_lookup = _load_or_generate_gt_sidecar_lookup(
        output_root=output_root,
        dataset_name=dataset_name,
        clip_ids=clip_ids,
        generate_sidecars=True,
    )
    full_vocab_ids = _text_vocab_ids(output_root)
    base_vocab_ids = list(full_vocab_ids)

    selected_stages = list(_STAGE_IDS) if stage == "all" else [stage]
    stage_summaries: Dict[str, Any] = {}
    ledger_paths: Dict[str, str] = {}
    for stage_id in selected_stages:
        stage_path = _resolve_stage_path(output_root, stage_id)
        stage_records = _load_jsonl(stage_path)
        rows, summary = build_dropped_gt_attribution_rows(
            stage_id=stage_id,
            materialized_samples=samples,
            stage_records=stage_records,
            gt_sidecar_lookup=gt_sidecar_lookup,
            full_vocab_ids=full_vocab_ids,
            base_vocab_ids=base_vocab_ids,
        )
        ledger_path = output_root / "train" / stage_id / "dropped_gt_attribution_ledger.jsonl"
        _write_jsonl(ledger_path, rows)
        ledger_paths[stage_id] = str(ledger_path)
        stage_summaries[stage_id] = summary

    summary_payload: Dict[str, Any] = {
        "status": "PASS" if stage_summaries else "EMPTY",
        "dataset_name": str(dataset_name),
        "trajectory_source_branch": str(trajectory_source_branch),
        "smoke": bool(smoke),
        "smoke_max_trajectories": int(smoke_max_trajectories),
        "requested_stage": str(stage),
        "stage_summaries": stage_summaries,
        "ledger_paths": ledger_paths,
        "metric_definitions": {
            "dropped_gt_mean_rank": "mean full-vocabulary rank of GT classes missing from observed_raw_ids; for soft-EM stages, GT outside stage domain receives sentinel rank |V|+1",
            "dropped_gt_top1_hit_rate": "fraction of dropped GT classes ranked top-1",
            "dropped_gt_top5_hit_rate": "fraction of dropped GT classes ranked in top-5",
            "dropped_gt_top10_hit_rate": "fraction of dropped GT classes ranked in top-10",
            "dropped_gt_mrr": "mean reciprocal rank of dropped GT classes",
            "wrong_top1_is_base_rate": "fraction of dropped GT rows whose wrong top-1 prediction belongs to the base/text-bank domain",
            "dropped_gt_in_stage_domain_rate": "fraction of dropped GT rows whose GT class appears in the explicit stage candidate domain",
            "dropped_gt_margin_to_best_wrong_mean": "mean score margin between GT and best wrong class for rows where GT is in stage domain",
        },
    }
    summary_path = output_root / "train" / "audit" / "dropped_gt_attribution_summary.json"
    _write_json(summary_path, summary_payload)
    summary_payload["summary_path"] = str(summary_path)
    return summary_payload
