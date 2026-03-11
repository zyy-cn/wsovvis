#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsovvis.inference.bagfree_inference_v9 import (
    BagFreeInferenceConfig,
    build_bagfree_inference_v9,
    build_bagfree_inference_v9_worked_example,
    summarize_bagfree_inference_v9,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bounded v9 bag-free inference/evaluation artifact.")
    parser.add_argument("--semantic-cache-root", required=True, help="Input G4 semantic-cache split root")
    parser.add_argument("--text-map-root", required=True, help="Input G5 text-map root")
    parser.add_argument("--protocol-output-json", required=True, help="Input protocol output JSON")
    parser.add_argument("--protocol-manifest-json", required=True, help="Input protocol manifest JSON")
    parser.add_argument("--output-root", required=True, help="Output G7 evaluation root")
    parser.add_argument("--summary-json", help="Optional summary JSON output path")
    parser.add_argument("--worked-example-json", help="Optional worked-example JSON output path")
    parser.add_argument("--selected-video-id", help="Optional worked-example video override")
    parser.add_argument("--num-qualitative-videos", type=int, default=3)
    parser.add_argument("--bg-score-threshold", type=float, default=0.34)
    parser.add_argument("--direct-min-score", type=float, default=0.40)
    parser.add_argument("--direct-margin", type=float, default=0.10)
    parser.add_argument("--unknown-min-score", type=float, default=0.34)
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
    config = BagFreeInferenceConfig(
        bg_score_threshold=float(args.bg_score_threshold),
        direct_min_score=float(args.direct_min_score),
        direct_margin=float(args.direct_margin),
        unknown_min_score=float(args.unknown_min_score),
        unknown_min_objectness=float(args.unknown_min_objectness),
    )
    output_root = build_bagfree_inference_v9(
        semantic_cache_root=Path(args.semantic_cache_root),
        text_map_root=Path(args.text_map_root),
        protocol_output_json=Path(args.protocol_output_json),
        protocol_manifest_json=Path(args.protocol_manifest_json),
        output_root=Path(args.output_root),
        overwrite=bool(args.overwrite),
        config=config,
        selected_video_id=args.selected_video_id,
        num_qualitative_videos=int(args.num_qualitative_videos),
    )
    summary = summarize_bagfree_inference_v9(output_root)
    worked_example = build_bagfree_inference_v9_worked_example(output_root, selected_video_id=args.selected_video_id)
    _write_json(args.summary_json, summary)
    _write_json(args.worked_example_json, worked_example)
    print(f"built bag-free evaluation: {output_root}")
    print(f"selected video: {worked_example['selected_video_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
