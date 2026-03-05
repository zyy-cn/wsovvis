#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsovvis.metrics import build_ws_metrics_summary_v1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build ws_metrics_summary_v1 from tiny summary/eval input JSON")
    p.add_argument("--input-json", type=Path, required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    summary = build_ws_metrics_summary_v1(payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
