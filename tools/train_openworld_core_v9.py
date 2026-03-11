#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsovvis.attribution.openworld_core_v9 import (
    OpenWorldCoreConfig,
    build_openworld_core_v9,
    build_openworld_core_v9_worked_example,
    summarize_openworld_core_v9,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bounded v9 core open-world attribution artifact.")
    parser.add_argument("--semantic-cache-root", required=True, help="Input G4 semantic-cache split root")
    parser.add_argument("--text-map-root", required=True, help="Input G5 text-map root")
    parser.add_argument("--protocol-output-json", required=True, help="Input G1/G5 protocol output JSON")
    parser.add_argument("--protocol-manifest-json", required=True, help="Input G1/G5 protocol manifest JSON")
    parser.add_argument("--output-root", required=True, help="Output G6 attribution root")
    parser.add_argument("--summary-json", help="Optional summary JSON output path")
    parser.add_argument("--worked-example-json", help="Optional worked-example JSON output path")
    parser.add_argument("--selected-video-id", help="Optional worked-example video override")
    parser.add_argument("--bg-score-threshold", type=float, default=0.34)
    parser.add_argument("--observed-min-score", type=float, default=0.34)
    parser.add_argument("--unknown-min-score", type=float, default=0.40)
    parser.add_argument("--unknown-margin", type=float, default=0.08)
    parser.add_argument("--unknown-min-objectness", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output paths")
    return parser.parse_args()


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    config = OpenWorldCoreConfig(
        bg_score_threshold=float(args.bg_score_threshold),
        observed_min_score=float(args.observed_min_score),
        unknown_min_score=float(args.unknown_min_score),
        unknown_margin=float(args.unknown_margin),
        unknown_min_objectness=float(args.unknown_min_objectness),
    )
    output_root = build_openworld_core_v9(
        semantic_cache_root=Path(args.semantic_cache_root),
        text_map_root=Path(args.text_map_root),
        protocol_output_json=Path(args.protocol_output_json),
        protocol_manifest_json=Path(args.protocol_manifest_json),
        output_root=Path(args.output_root),
        overwrite=bool(args.overwrite),
        config=config,
        selected_video_id=args.selected_video_id,
    )
    summary = summarize_openworld_core_v9(output_root)
    worked_example = build_openworld_core_v9_worked_example(output_root, selected_video_id=args.selected_video_id)
    _write_json(args.summary_json, summary)
    _write_json(args.worked_example_json, worked_example)
    print(f"built open-world core attribution: {output_root}")
    print(f"selected video: {worked_example['selected_video_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
