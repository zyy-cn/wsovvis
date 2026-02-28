#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_enablement_module():
    module_path = REPO_ROOT / "wsovvis" / "track_feature_export" / "feature_export_enablement_v1.py"
    spec = importlib.util.spec_from_file_location("feature_export_enablement_v1", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage B feature-export enablement artifact (contract v1) from a "
            "runner-side JSON payload with explicit per-track embeddings."
        )
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Feature-export input JSON path")
    parser.add_argument("--run-root", type=Path, required=True, help="Stage B run root directory")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing feature_export_v1 output")
    parser.add_argument(
        "--emit-video-index",
        action="store_true",
        help="Also write optional d2/inference/feature_export_v1/video_index.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = _load_enablement_module()
    try:
        payload = module.load_feature_export_enablement_input(args.input_json)
        output_root = module.build_feature_export_enablement_v1(
            input_payload=payload,
            run_root=args.run_root,
            overwrite=args.overwrite,
            emit_video_index=args.emit_video_index,
        )
    except module.ExportContractError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote Stage B feature-export enablement artifact: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
