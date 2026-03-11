from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsovvis.tracking.global_track_bank_v9 import (
    StitchingConfig,
    build_global_track_bank_v9,
    build_global_track_bank_v9_worked_example,
    render_global_track_bank_coverage_svg,
    summarize_global_track_bank_v9,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bounded v9 clip-level global track bank.")
    parser.add_argument("--input-split-root", required=True, help="Stage-B local-tracklet split root")
    parser.add_argument("--output-split-root", required=True, help="Output global-track-bank split root")
    parser.add_argument("--summary-json", help="Optional summary JSON output path")
    parser.add_argument("--worked-example-json", help="Optional worked-example JSON output path")
    parser.add_argument("--coverage-svg", help="Optional clip-level coverage SVG output path")
    parser.add_argument("--selected-video-id", help="Optional worked-example / SVG video id override")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output paths")
    parser.add_argument("--weight-temporal-iou", type=float, default=0.5)
    parser.add_argument("--weight-query-cosine", type=float, default=0.5)
    parser.add_argument("--min-temporal-iou", type=float, default=1.0)
    parser.add_argument("--min-query-cosine", type=float, default=0.995)
    parser.add_argument("--min-match-score", type=float, default=None)
    return parser.parse_args()


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    config = StitchingConfig(
        weight_temporal_iou=float(args.weight_temporal_iou),
        weight_query_cosine=float(args.weight_query_cosine),
        min_temporal_iou=float(args.min_temporal_iou),
        min_query_cosine=float(args.min_query_cosine),
        min_match_score=None if args.min_match_score is None else float(args.min_match_score),
    )
    output_root = build_global_track_bank_v9(
        source_split_root=Path(args.input_split_root),
        output_split_root=Path(args.output_split_root),
        overwrite=bool(args.overwrite),
        config=config,
    )
    summary = summarize_global_track_bank_v9(output_root)
    worked_example = build_global_track_bank_v9_worked_example(
        output_root,
        selected_video_id=args.selected_video_id or summary["selected_video_id"],
    )
    _write_json(args.summary_json, summary)
    _write_json(args.worked_example_json, worked_example)
    if args.coverage_svg:
        render_global_track_bank_coverage_svg(
            output_root,
            Path(args.coverage_svg),
            selected_video_id=worked_example["selected_video_id"],
        )
    print(f"built global track bank: {output_root}")
    print(f"selected worked-example video: {worked_example['selected_video_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
