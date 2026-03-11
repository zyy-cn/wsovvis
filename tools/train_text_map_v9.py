#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsovvis.semantics.text_map_v9 import (
    TextMapConfig,
    build_text_map_v9,
    build_text_map_v9_worked_example,
    render_text_map_alignment_svg,
    summarize_text_map_v9,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bounded v9 class-level text map A and mapped text prototypes.")
    parser.add_argument("--prototype-bank-root", required=True, help="Input G5 prototype-bank root")
    parser.add_argument("--output-root", required=True, help="Output text-map root")
    parser.add_argument("--summary-json", help="Optional combined G5 summary JSON output path")
    parser.add_argument("--worked-example-json", help="Optional class-level worked-example JSON output path")
    parser.add_argument("--alignment-svg", help="Optional mapped-text alignment SVG output path")
    parser.add_argument("--selected-label-id", help="Optional worked-example label override")
    parser.add_argument("--text-model-name", default="ViT-B-32")
    parser.add_argument("--text-model-pretrained", default="openai")
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--cache-dir", default=None)
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
    config = TextMapConfig(
        text_model_name=str(args.text_model_name),
        text_model_pretrained=str(args.text_model_pretrained),
        device=str(args.device),
        batch_size=int(args.batch_size),
        ridge_lambda=float(args.ridge_lambda),
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
    )
    output_root = build_text_map_v9(
        prototype_bank_root=Path(args.prototype_bank_root),
        output_root=Path(args.output_root),
        overwrite=bool(args.overwrite),
        config=config,
    )
    summary = summarize_text_map_v9(output_root)
    worked_example = build_text_map_v9_worked_example(
        output_root,
        selected_label_id=args.selected_label_id,
    )
    _write_json(args.summary_json, summary)
    _write_json(args.worked_example_json, worked_example)
    if args.alignment_svg:
        render_text_map_alignment_svg(
            output_root,
            Path(args.alignment_svg),
            selected_label_id=worked_example["selected_label_id"],
        )
    print(f"built text map: {output_root}")
    print(f"selected label: {worked_example['selected_label_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
