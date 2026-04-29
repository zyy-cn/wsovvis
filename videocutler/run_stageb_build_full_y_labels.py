#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from videocutler.ext_stageb_ovvis.data.oracle_clean_ablation_sources import (
    build_full_y_records,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Materialize full-Y raw-id labels for LV-VIS train base.")
    p.add_argument("--dataset_name", default="lvvis_train_base")
    p.add_argument("--annotation_json", required=True)
    p.add_argument("--weak_labels_json", required=True)
    p.add_argument("--official_split_json", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--summary_path", required=True)
    p.add_argument("--asset_root", default="/home/zyy/code/wsovvis_asserts")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    annotation_json = Path(args.annotation_json).expanduser().resolve()
    weak_labels_json = Path(args.weak_labels_json).expanduser().resolve()
    official_split_json = Path(args.official_split_json).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    summary_path = Path(args.summary_path).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()

    asset, summary = build_full_y_records(
        annotation_json=annotation_json,
        weak_labels_json=weak_labels_json,
        official_split_json=official_split_json,
        text_bank_root=asset_root if asset_root.is_dir() else None,
    )

    write_json(output_path, asset)
    summary = dict(summary)
    summary.update(
        {
            "status": "PASS" if int(summary.get("yprime_subset_full_y_violation_count", 0)) == 0 and int(summary.get("missing_full_y_count", 0)) == 0 else "PARTIAL",
            "output_path": str(output_path),
            "record_count": int(len(asset.get("records", []))),
        }
    )
    write_json(summary_path, summary)

    print(json.dumps({"status": summary["status"], "output_path": str(output_path), "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
