#!/usr/bin/env python3
"""Make a YouTube-VIS/LV-VIS style annotation JSON class-agnostic.

This is used so we can evaluate a class-agnostic model (1 category) against LV-VIS GT.

Input is expected to have keys like: videos, annotations, categories.
We keep the structure and only rewrite:
  - categories -> single category
  - annotations[*].category_id -> that category id

We do NOT touch segmentations/bboxes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--category_id", type=int, default=1)
    ap.add_argument("--category_name", default="object")
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["categories"] = [{"id": args.category_id, "name": args.category_name}]
    for ann in data.get("annotations", []):
        ann["category_id"] = args.category_id

    os.makedirs(Path(args.out_json).parent, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(
        f"[OK] wrote {args.out_json} | videos={len(data.get('videos', []))} anns={len(data.get('annotations', []))}"
    )


if __name__ == "__main__":
    main()
