from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from .ws_metrics_reporting_v1 import build_ws_metrics_summary_v1


def _as_int_list(values: Any, *, field_path: str) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{field_path}: must be a sequence of ints")
    out: list[int] = []
    for idx, value in enumerate(values):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field_path}[{idx}]: must be int")
        out.append(int(value))
    return out


def _collect_missing_fields(round_summary: Mapping[str, Any]) -> list[str]:
    missing: list[str] = []
    ros = round_summary.get("round_output_summary")
    if not isinstance(ros, Mapping):
        return ["round_output_summary"]
    if not isinstance(ros.get("selected_video_id"), str) or not str(ros.get("selected_video_id")).strip():
        missing.append("round_output_summary.selected_video_id")
    if not isinstance(ros.get("positive_label_ids"), Sequence) or isinstance(ros.get("positive_label_ids"), (str, bytes)):
        missing.append("round_output_summary.positive_label_ids")
    if not isinstance(ros.get("candidate_label_ids"), Sequence) or isinstance(ros.get("candidate_label_ids"), (str, bytes)):
        missing.append("round_output_summary.candidate_label_ids")
    return missing


def _build_predictions_by_missing_rate(
    candidate_label_ids: Sequence[int],
    missing_rates: Sequence[float] = (0.0, 0.5, 1.0),
) -> dict[str, list[int]]:
    cand = list(candidate_label_ids)
    n = len(cand)
    out: dict[str, list[int]] = {}
    for m in missing_rates:
        m_float = float(m)
        if not (0.0 <= m_float <= 1.0):
            raise ValueError(f"missing_rate must be in [0,1], got {m_float}")
        if m_float >= 1.0:
            k = 0
        elif m_float <= 0.0:
            k = n
        else:
            k = int(math.ceil((1.0 - m_float) * float(n)))
            if k < 0:
                k = 0
            if k > n:
                k = n
        out[str(m_float)] = cand[:k]
    return out


def build_ws_metrics_summary_v1_from_stage_d_round_summary(
    round_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(round_summary, Mapping):
        raise ValueError("round_summary: must be mapping")
    missing_fields = _collect_missing_fields(round_summary)
    if missing_fields:
        raise ValueError(f"stage_d_round_summary missing required fields: {', '.join(missing_fields)}")

    round_output_summary = round_summary["round_output_summary"]
    assert isinstance(round_output_summary, Mapping)
    positive_label_ids = _as_int_list(
        round_output_summary["positive_label_ids"],
        field_path="round_output_summary.positive_label_ids",
    )
    candidate_label_ids = _as_int_list(
        round_output_summary["candidate_label_ids"],
        field_path="round_output_summary.candidate_label_ids",
    )

    predictions_by_missing_rate = _build_predictions_by_missing_rate(candidate_label_ids)
    ws_eval_bundle = {
        "gt_entities": positive_label_ids,
        "predicted_entities": candidate_label_ids,
        "predictions_by_missing_rate": predictions_by_missing_rate,
    }
    summary = build_ws_metrics_summary_v1(
        {
            "video_id": round_output_summary["selected_video_id"],
            "assignment_backend": round_output_summary.get("assignment_backend"),
            "steps": None,
            "seed": None,
            "ws_eval_bundle": ws_eval_bundle,
        }
    )
    gt_set = set(positive_label_ids)
    cand_set = set(candidate_label_ids)
    intersection_size = len(gt_set & cand_set)
    union_size = len(gt_set | cand_set)
    summary["stage_d_extras"] = {
        "candidate_size": int(len(candidate_label_ids)),
        "positive_size": int(len(positive_label_ids)),
        "jaccard": float(intersection_size / union_size) if union_size > 0 else 1.0,
        "overreach": int(len(candidate_label_ids) - intersection_size),
    }
    return summary


def build_ws_metrics_summary_v1_from_stage_d_round_summary_json(
    round_summary_json_path: str | Path,
) -> dict[str, Any]:
    path = Path(round_summary_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"round summary JSON must be object: {path}")
    return build_ws_metrics_summary_v1_from_stage_d_round_summary(payload)
