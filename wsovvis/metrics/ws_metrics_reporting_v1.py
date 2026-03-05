from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .ws_metrics import aurc_from_curve, missing_rate_curve_from_predictions, set_coverage_recall

WS_METRICS_SUMMARY_SCHEMA_NAME = "wsovvis.ws_metrics_summary_v1"
WS_METRICS_SUMMARY_SCHEMA_VERSION = "1.0"


def _require(condition: bool, field_path: str, rule: str) -> None:
    if not condition:
        raise ValueError(f"{field_path}: {rule}")


def _normalize_missing_rate_map(raw: Mapping[str | float, Any]) -> dict[float, Sequence[int]]:
    out: dict[float, Sequence[int]] = {}
    for key, value in raw.items():
        m = float(key)
        _require(isinstance(value, Sequence), f"predictions_by_missing_rate[{key!r}]", "must be sequence")
        out[m] = [int(v) for v in value]
    return out


def _parse_eval_bundle(bundle: Mapping[str, Any]) -> tuple[list[int], list[int], dict[float, Sequence[int]]]:
    _require("gt_entities" in bundle, "gt_entities", "required")
    _require("predicted_entities" in bundle, "predicted_entities", "required")
    _require("predictions_by_missing_rate" in bundle, "predictions_by_missing_rate", "required")

    gt_entities_raw = bundle["gt_entities"]
    predicted_entities_raw = bundle["predicted_entities"]
    by_missing_raw = bundle["predictions_by_missing_rate"]

    _require(isinstance(gt_entities_raw, Sequence), "gt_entities", "must be sequence")
    _require(isinstance(predicted_entities_raw, Sequence), "predicted_entities", "must be sequence")
    _require(isinstance(by_missing_raw, Mapping), "predictions_by_missing_rate", "must be mapping")

    gt_entities = [int(v) for v in gt_entities_raw]
    predicted_entities = [int(v) for v in predicted_entities_raw]
    by_missing = _normalize_missing_rate_map(by_missing_raw)
    return gt_entities, predicted_entities, by_missing


def build_ws_metrics_summary_v1(source: Mapping[str, Any]) -> dict[str, Any]:
    """Attach C10a WS metrics to a tiny stage-level summary/eval bundle.

    Accepted input shapes:
    - direct bundle: {gt_entities, predicted_entities, predictions_by_missing_rate}
    - stage summary: { ..., ws_eval_bundle: {gt_entities, predicted_entities, predictions_by_missing_rate} }
    """

    _require(isinstance(source, Mapping), "source", "must be mapping")
    if "ws_eval_bundle" in source:
        raw_bundle = source["ws_eval_bundle"]
        _require(isinstance(raw_bundle, Mapping), "ws_eval_bundle", "must be mapping when present")
        bundle = raw_bundle
        source_metadata = {
            "video_id": source.get("video_id"),
            "assignment_backend": source.get("assignment_backend"),
            "steps": source.get("steps"),
            "seed": source.get("seed"),
        }
    else:
        bundle = source
        source_metadata = {
            "video_id": source.get("video_id"),
            "assignment_backend": source.get("assignment_backend"),
            "steps": source.get("steps"),
            "seed": source.get("seed"),
        }

    gt_entities, predicted_entities, by_missing = _parse_eval_bundle(bundle)
    scr = set_coverage_recall(gt_entities, predicted_entities)
    curve = missing_rate_curve_from_predictions(gt_entities, by_missing)
    aurc = aurc_from_curve(curve)

    curve_rows = [{"missing_rate": float(m), "scr": float(r)} for m, r in curve]
    return {
        "schema_name": WS_METRICS_SUMMARY_SCHEMA_NAME,
        "schema_version": WS_METRICS_SUMMARY_SCHEMA_VERSION,
        "metrics": {
            "scr": float(scr),
            "missing_rate_curve": curve_rows,
            "aurc": float(aurc),
        },
        "source_metadata": source_metadata,
    }
