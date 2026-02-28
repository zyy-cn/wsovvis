#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsovvis.track_feature_export import ExportContractError, validate_track_feature_export_v1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage B track feature export v1 artifact invariants (schema/paths/status/count/alignment)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split-root", type=Path, help="Split root directory containing manifest.v1.json")
    group.add_argument("--manifest", type=Path, help="Path to manifest.v1.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validated_root = validate_track_feature_export_v1(
            split_root=args.split_root,
            manifest_path=args.manifest,
        )
    except ExportContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Validation OK: {validated_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
