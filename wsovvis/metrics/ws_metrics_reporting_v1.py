from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .ws_metrics import (
    aurc_from_curve,
    hidden_positive_recall,
    missing_rate_curve_from_predictions,
    set_coverage_recall,
    unknown_attribution_recall,
)

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


def _normalize_optional_int_sequence(bundle: Mapping[str, Any], key: str) -> list[int] | None:
    raw = bundle.get(key)
    if raw is None:
        return None
    _require(isinstance(raw, Sequence), key, "must be sequence when present")
    return [int(v) for v in raw]


def _resolve_hidden_positive_entities(bundle: Mapping[str, Any], gt_entities: Sequence[int]) -> list[int] | None:
    hidden_positive_entities = _normalize_optional_int_sequence(bundle, "hidden_positive_entities")
    observed_entities = _normalize_optional_int_sequence(bundle, "observed_entities")
    gt_set = set(int(v) for v in gt_entities)

    if hidden_positive_entities is not None:
        hidden_set = set(hidden_positive_entities)
        _require(hidden_set <= gt_set, "hidden_positive_entities", "must be subset of gt_entities")
    else:
        hidden_set = set()

    if observed_entities is None:
        return hidden_positive_entities

    observed_set = set(observed_entities)
    _require(observed_set <= gt_set, "observed_entities", "must be subset of gt_entities")
    derived_hidden = [int(v) for v in gt_entities if int(v) not in observed_set]
    if hidden_positive_entities is None:
        return derived_hidden

    _require(hidden_set == set(derived_hidden), "hidden_positive_entities", "must equal gt_entities - observed_entities when both are provided")
    return hidden_positive_entities


def _parse_eval_bundle(
    bundle: Mapping[str, Any],
) -> tuple[list[int], list[int], dict[float, Sequence[int]], list[int] | None, list[int] | None]:
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
    hidden_positive_entities = _resolve_hidden_positive_entities(bundle, gt_entities)
    unknown_attributed_entities = _normalize_optional_int_sequence(bundle, "unknown_attributed_entities")
    return gt_entities, predicted_entities, by_missing, hidden_positive_entities, unknown_attributed_entities


def build_ws_metrics_summary_v1(source: Mapping[str, Any]) -> dict[str, Any]:
    """Attach C10a WS metrics to a tiny stage-level summary/eval bundle.

    Accepted input shapes:
    - direct bundle: {gt_entities, predicted_entities, predictions_by_missing_rate, ...optional hidden-positive inputs}
    - stage summary: { ..., ws_eval_bundle: {gt_entities, predicted_entities, predictions_by_missing_rate, ...optional hidden-positive inputs} }

    Optional hidden-positive inputs:
    - observed_entities: observed/visible positive subset used to derive hidden positives as gt - observed
    - hidden_positive_entities: explicit hidden-positive subset; if observed_entities is also present, both must agree
    - unknown_attributed_entities: subset preserved via explicit unknown attribution for UAR
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

    gt_entities, predicted_entities, by_missing, hidden_positive_entities, unknown_attributed_entities = _parse_eval_bundle(bundle)
    scr = set_coverage_recall(gt_entities, predicted_entities)
    curve = missing_rate_curve_from_predictions(gt_entities, by_missing)
    aurc = aurc_from_curve(curve)

    curve_rows = [{"missing_rate": float(m), "scr": float(r)} for m, r in curve]
    metrics: dict[str, Any] = {
        "scr": float(scr),
        "missing_rate_curve": curve_rows,
        "aurc": float(aurc),
    }
    if hidden_positive_entities is not None:
        metrics["hpr"] = float(hidden_positive_recall(hidden_positive_entities, predicted_entities))
        if unknown_attributed_entities is not None:
            metrics["uar"] = float(unknown_attribution_recall(hidden_positive_entities, unknown_attributed_entities))
    return {
        "schema_name": WS_METRICS_SUMMARY_SCHEMA_NAME,
        "schema_version": WS_METRICS_SUMMARY_SCHEMA_VERSION,
        "metrics": metrics,
        "source_metadata": source_metadata,
    }
