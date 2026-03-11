#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsovvis.semantics.prototype_bank_v9 import (
    PrototypeBankConfig,
    build_prototype_bank_v9,
    summarize_prototype_bank_v9,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the bounded v9 seen-visual prototype bank.")
    parser.add_argument("--semantic-cache-root", required=True, help="Input G4 semantic-cache split root")
    parser.add_argument("--protocol-output-json", required=True, help="Input clip-level protocol JSON")
    parser.add_argument("--protocol-manifest-json", required=True, help="Input clip-level protocol manifest JSON")
    parser.add_argument("--output-root", required=True, help="Output prototype-bank root")
    parser.add_argument("--summary-json", help="Optional summary JSON output path")
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
    output_root = build_prototype_bank_v9(
        semantic_cache_root=Path(args.semantic_cache_root),
        protocol_output_json=Path(args.protocol_output_json),
        protocol_manifest_json=Path(args.protocol_manifest_json),
        output_root=Path(args.output_root),
        overwrite=bool(args.overwrite),
        config=PrototypeBankConfig(),
    )
    summary = summarize_prototype_bank_v9(output_root)
    _write_json(args.summary_json, summary)
    print(f"built prototype bank: {output_root}")
    print(f"selected label: {summary['selected_label_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
