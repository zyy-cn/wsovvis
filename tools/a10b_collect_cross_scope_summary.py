#!/usr/bin/env python3
"""Collect lightweight A10B cross-scope summary artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _csv_rows(path: Path) -> int:
    if not path.is_file():
        return -1
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(sum(1 for _ in f) - 1, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output_root", required=True)
    args = ap.parse_args()
    out = Path(args.output_root).expanduser().resolve()
    ana = out / "analysis"
    files = [
        "cross_scope_availability_inventory.csv",
        "cross_scope_class_proto_summary.csv",
        "cross_scope_row_level_summary.csv",
        "cross_scope_anchor_ratio_curve.csv",
        "cross_scope_projector_selection_by_calib.csv",
        "cross_scope_row_level_per_class.csv",
    ]
    manifest: List[Dict[str, Any]] = []
    for rel in files:
        p = ana / rel
        if p.is_file():
            manifest.append({"artifact": str(p.relative_to(out)), "row_count": _csv_rows(p), "size_bytes": p.stat().st_size})
    # Small aggregate heads for quick review without reading large files.
    payload = {
        "status": "PASS" if manifest else "WARN",
        "output_root": str(out),
        "analysis_root": str(ana),
        "artifact_count": len(manifest),
        "artifacts": manifest,
    }
    _write_json(ana / "A10B_collected_summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
