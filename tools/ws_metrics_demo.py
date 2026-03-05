#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsovvis.metrics import aurc_from_curve, missing_rate_curve_from_predictions, set_coverage_recall


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Demo SCR + missing-rate curve + AURC on tiny JSON input.")
    p.add_argument("--input-json", type=Path, required=True, help="Path to tiny synthetic metrics JSON.")
    return p


def main() -> int:
    args = _build_parser().parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))

    gt_entities = payload["gt_entities"]
    predicted_entities = payload["predicted_entities"]
    predictions_by_missing_rate = {float(k): v for k, v in payload["predictions_by_missing_rate"].items()}

    scr = set_coverage_recall(gt_entities, predicted_entities)
    curve = missing_rate_curve_from_predictions(gt_entities, predictions_by_missing_rate)
    aurc = aurc_from_curve(curve)

    print(json.dumps({"scr": scr, "curve": curve, "aurc": aurc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
