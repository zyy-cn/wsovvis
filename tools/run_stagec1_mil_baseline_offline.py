#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsovvis.track_feature_export import (
    StageC1MilConfig,
    run_stagec1_mil_baseline_offline,
)


def _parse_supported_statuses(raw: str) -> tuple[str, ...]:
    statuses = tuple(s.strip() for s in raw.split(",") if s.strip())
    if not statuses:
        raise argparse.ArgumentTypeError("--supported-video-statuses must provide at least one status")
    return statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage C1 MIL-first offline attribution baseline scoring")
    parser.add_argument("--split-root", type=Path, required=True, help="Stage B export split root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for Stage C1 artifacts")
    parser.add_argument("--embedding-abs-mean-weight", type=float, default=1.0)
    parser.add_argument("--objectness-weight", type=float, default=1.0)
    parser.add_argument("--length-log-weight", type=float, default=0.25)
    parser.add_argument("--top-k-per-video", type=int, default=3)
    parser.add_argument(
        "--supported-video-statuses",
        type=_parse_supported_statuses,
        default=("processed_with_tracks", "processed_zero_tracks"),
        help="Comma-separated statuses accepted for processed videos",
    )
    parser.add_argument("--no-eager-validate", action="store_true", help="Disable eager Stage C0 shard validation")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = StageC1MilConfig(
        embedding_abs_mean_weight=args.embedding_abs_mean_weight,
        objectness_weight=args.objectness_weight,
        length_log_weight=args.length_log_weight,
        top_k_per_video=args.top_k_per_video,
        supported_video_statuses=args.supported_video_statuses,
    )

    report = run_stagec1_mil_baseline_offline(
        split_root=args.split_root,
        output_dir=args.output_dir,
        config=config,
        eager_validate=not args.no_eager_validate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
