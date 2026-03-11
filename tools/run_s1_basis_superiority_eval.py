#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsovvis.scientific import run_s1_basis_superiority_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the S1 structure-only basis superiority comparator on unified YTVIS-style inputs."
    )
    parser.add_argument("--gt-json", type=Path, required=True, help="Class-agnostic LV-VIS GT JSON on val")
    parser.add_argument("--raw-json", type=Path, required=True, help="Raw VideoCutLER pseudo tube JSON")
    parser.add_argument("--refined-json", type=Path, required=True, help="SeqFormer refined predictions JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for S1 pilot artifacts")
    parser.add_argument("--raw-label", default="raw_pseudo_tube", help="Comparator label for raw tubes")
    parser.add_argument("--refined-label", default="refined_basis", help="Comparator label for refined basis")
    parser.add_argument(
        "--frame-match-iou-threshold",
        type=float,
        default=0.5,
        help="Per-frame IoU threshold used to count GT fragments",
    )
    parser.add_argument(
        "--short-track-threshold",
        type=int,
        default=5,
        help="Active-frame threshold used by short_track_ratio",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_s1_basis_superiority_eval(
        gt_json_path=args.gt_json,
        raw_json_path=args.raw_json,
        refined_json_path=args.refined_json,
        output_root=args.output_dir,
        raw_label=args.raw_label,
        refined_label=args.refined_label,
        frame_match_iou_threshold=args.frame_match_iou_threshold,
        short_track_threshold=args.short_track_threshold,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
