#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tiny compare for Stage C unknown-handling diagnostics")
    p.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more run_stagec_c5_micro_training summary JSON files",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Optional path to write normalized compare JSON")
    return p


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be JSON object: {path}")
    return payload


def _diag_row(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    diag = payload.get("unknown_handling_diagnostics_v1")
    if not isinstance(diag, dict):
        raise ValueError(f"missing unknown_handling_diagnostics_v1 in {path}")
    mass = diag.get("mass", {})
    dist = diag.get("distribution", {})
    coverage = diag.get("coverage", {})
    losses = diag.get("losses", {})
    guardrail = diag.get("risk_guardrail_v1", {})
    guardrail_reasons = guardrail.get("reasons", [])
    guardrail_reason_text = "; ".join(str(x) for x in guardrail_reasons if isinstance(x, str))
    row = {
        "source_json": str(path),
        "video_id": str(diag.get("video_id", payload.get("selected_video_id", ""))),
        "selected_num_positive_labels": int(diag.get("selected_num_positive_labels", payload.get("selected_num_positive_labels", 0))),
        "backend": str(diag.get("assignment_backend", payload.get("assignment_backend_requested", ""))),
        "bg_mass": float(mass.get("bg_mass", 0.0)),
        "unk_fg_mass": float(mass.get("unk_fg_mass", 0.0)),
        "non_special_mass": float(mass.get("non_special_mass", 0.0)),
        "unk_vs_bg_ratio": float(mass.get("unk_vs_bg_ratio", 0.0)),
        "entropy": float(dist.get("entropy", 0.0)),
        "top1_mass": float(dist.get("top1_mass", 0.0)),
        "coverage_ratio": float(coverage.get("coverage_ratio", 0.0)),
        "coverage_loss": float(coverage.get("coverage_loss", 0.0)),
        "alignment_loss": float(losses.get("alignment_loss", 0.0)),
        "fg_not_bg_loss": float(losses.get("fg_not_bg_loss", 0.0)),
        "total_loss": float(losses.get("total_loss", 0.0)),
        "risk_guardrail_triggered": bool(guardrail.get("triggered", False)),
        "risk_guardrail_score": int(guardrail.get("risk_score", 0)),
        "risk_guardrail_level": str(guardrail.get("risk_level", "none")),
        "risk_guardrail_reasons": guardrail_reason_text,
        "backend_config_echo": diag.get("backend_config_echo", {}),
    }
    return row


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(
        "backend|video_id|selected_num_positive_labels|bg_mass|unk_fg_mass|non_special_mass|"
        "unk_vs_bg_ratio|entropy|top1_mass|coverage_ratio|coverage_loss|alignment_loss|fg_not_bg_loss|total_loss|"
        "risk_guardrail_triggered|risk_guardrail_score|risk_guardrail_level|risk_guardrail_reasons"
    )
    for row in rows:
        print(
            f"{row['backend']}|{row['video_id']}|{row['selected_num_positive_labels']}|"
            f"{row['bg_mass']:.12f}|{row['unk_fg_mass']:.12f}|{row['non_special_mass']:.12f}|"
            f"{row['unk_vs_bg_ratio']:.6f}|{row['entropy']:.12f}|{row['top1_mass']:.12f}|"
            f"{row['coverage_ratio']:.12f}|{row['coverage_loss']:.12f}|{row['alignment_loss']:.12f}|"
            f"{row['fg_not_bg_loss']:.12f}|{row['total_loss']:.12f}|"
            f"{int(row['risk_guardrail_triggered'])}|{row['risk_guardrail_score']}|"
            f"{row['risk_guardrail_level']}|{row['risk_guardrail_reasons']}"
        )


def main() -> int:
    args = _build_parser().parse_args()
    rows = [_diag_row(path, _load_json(path)) for path in args.inputs]
    _print_table(rows)
    compare = {
        "schema_name": "wsovvis.stagec_unknown_handling_compare_v1",
        "schema_version": "1.0",
        "num_rows": len(rows),
        "rows": rows,
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(compare, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"UNKNOWN_COMPARE_OUT_JSON {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
