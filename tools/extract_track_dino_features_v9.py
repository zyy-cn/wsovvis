#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsovvis.features.track_dino_feature_v9 import (
    SemanticCacheConfig,
    build_track_dino_feature_cache_v9,
    build_track_dino_feature_cache_v9_worked_example,
    render_track_dino_feature_provenance_svg,
    summarize_track_dino_feature_cache_v9,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bounded v9 DINO-only track semantic cache.")
    parser.add_argument("--global-track-bank-root", required=True, help="Input G3 global-track-bank split root")
    parser.add_argument("--run-root", required=True, help="Stage-B run root with real predictions and LV-VIS config")
    parser.add_argument("--output-split-root", required=True, help="Output semantic-cache split root")
    parser.add_argument("--summary-json", help="Optional summary JSON output path")
    parser.add_argument("--worked-example-json", help="Optional worked-example JSON output path")
    parser.add_argument("--provenance-svg", help="Optional crop/pooling provenance SVG path")
    parser.add_argument("--selected-video-id", help="Optional worked-example video id override")
    parser.add_argument("--selected-global-track-id", type=int, help="Optional worked-example global_track_id override")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output paths")
    parser.add_argument("--crop-padding-ratio", type=float, default=0.1)
    parser.add_argument("--resize-edge", type=int, default=224)
    parser.add_argument("--max-visible-frames-per-track", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dino-repo-path", default="third_party/dinov2")
    parser.add_argument("--dino-weights-path", default="weights/DINOv2/dinov2_vitb14_pretrain.pth")
    return parser.parse_args()


def _write_json(path: str | None, payload: dict) -> None:
    if not path:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = _parse_args()
    config = SemanticCacheConfig(
        dino_repo_path=str(args.dino_repo_path),
        dino_weights_path=str(args.dino_weights_path),
        crop_padding_ratio=float(args.crop_padding_ratio),
        resize_edge=int(args.resize_edge),
        max_visible_frames_per_track=int(args.max_visible_frames_per_track),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    output_root = build_track_dino_feature_cache_v9(
        global_track_bank_root=Path(args.global_track_bank_root),
        run_root=Path(args.run_root),
        output_split_root=Path(args.output_split_root),
        overwrite=bool(args.overwrite),
        config=config,
    )
    summary = summarize_track_dino_feature_cache_v9(output_root)
    worked_example = build_track_dino_feature_cache_v9_worked_example(
        output_root,
        selected_video_id=args.selected_video_id or summary["selected_video_id"],
        selected_global_track_id=args.selected_global_track_id or summary["selected_global_track_id"],
    )
    _write_json(args.summary_json, summary)
    _write_json(args.worked_example_json, worked_example)
    if args.provenance_svg:
        render_track_dino_feature_provenance_svg(
            output_root,
            Path(args.provenance_svg),
            selected_video_id=worked_example["selected_video_id"],
            selected_global_track_id=worked_example["selected_global_track_id"],
        )
    print(f"built track semantic cache: {output_root}")
    print(f"selected worked-example video: {worked_example['selected_video_id']}")
    print(f"selected worked-example global_track_id: {worked_example['selected_global_track_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
