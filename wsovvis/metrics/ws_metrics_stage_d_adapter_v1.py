from __future__ import annotations

import json
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

    ws_eval_bundle = {
        "gt_entities": positive_label_ids,
        "predicted_entities": candidate_label_ids,
        "predictions_by_missing_rate": {
            "0.0": candidate_label_ids,
            "0.5": candidate_label_ids,
            "1.0": candidate_label_ids,
        },
    }
    return build_ws_metrics_summary_v1(
        {
            "video_id": round_output_summary["selected_video_id"],
            "assignment_backend": round_output_summary.get("assignment_backend"),
            "steps": None,
            "seed": None,
            "ws_eval_bundle": ws_eval_bundle,
        }
    )


def build_ws_metrics_summary_v1_from_stage_d_round_summary_json(
    round_summary_json_path: str | Path,
) -> dict[str, Any]:
    path = Path(round_summary_json_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"round summary JSON must be object: {path}")
    return build_ws_metrics_summary_v1_from_stage_d_round_summary(payload)
