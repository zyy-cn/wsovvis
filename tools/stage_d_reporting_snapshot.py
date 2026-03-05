#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a compact read-only Stage D snapshot from existing round/loop summaries."
    )
    p.add_argument("--loop-summary-json", type=Path, required=True, help="Path to Stage D loop summary JSON")
    p.add_argument("--round0-summary-json", type=Path, default=None, help="Optional explicit round0 summary JSON")
    p.add_argument("--round1-summary-json", type=Path, default=None, help="Optional explicit round1 summary JSON")
    p.add_argument(
        "--round-summary-root",
        type=Path,
        default=None,
        help="Optional root directory containing round{0,1}_summary.json",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Optional path to write compact snapshot JSON")
    p.add_argument("--out-text", type=Path, default=None, help="Optional path to write concise text snapshot")
    return p


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _require_int(payload: dict[str, Any], key: str, field_path: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Missing or non-integer field: {field_path}")
    return int(value)


def _read_int_list(payload: dict[str, Any], key: str) -> list[int]:
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        if isinstance(item, int) and not isinstance(item, bool):
            out.append(int(item))
    return sorted(set(out))


def _resolve_maybe_relative(base_path: Path, candidate: str) -> Path:
    raw = Path(candidate)
    if raw.is_absolute():
        return raw
    return (base_path.parent / raw).resolve()


def _resolve_round_path(
    *,
    loop_payload: dict[str, Any],
    loop_path: Path,
    round_index: int,
    explicit_path: Path | None,
    round_summary_root: Path | None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path.resolve())

    round_paths = loop_payload.get("round_paths")
    if isinstance(round_paths, list) and round_index < len(round_paths):
        row = round_paths[round_index]
        if isinstance(row, dict):
            round_summary_path = row.get("round_summary_path")
            if isinstance(round_summary_path, str) and round_summary_path.strip():
                candidates.append(_resolve_maybe_relative(loop_path.resolve(), round_summary_path.strip()))

    if round_summary_root is not None:
        candidates.append((round_summary_root.resolve() / f"round{round_index}_summary.json").resolve())
    candidates.append((loop_path.resolve().parent / "rounds" / f"round{round_index}_summary.json").resolve())
    candidates.append((loop_path.resolve().parent / f"round{round_index}_summary.json").resolve())

    for path in candidates:
        if path.is_file():
            return path
    joined = ", ".join(str(p) for p in candidates)
    raise ValueError(f"Unable to locate round{round_index} summary JSON. Tried: {joined}")


def _build_round_compact(round_payload: dict[str, Any], round_label: str) -> dict[str, Any]:
    count_before = _require_int(
        round_payload,
        "candidate_label_ids_count_before",
        f"{round_label}.candidate_label_ids_count_before",
    )
    count_after = _require_int(
        round_payload,
        "candidate_label_ids_count_after",
        f"{round_label}.candidate_label_ids_count_after",
    )
    count_delta = _require_int(
        round_payload,
        "candidate_label_ids_count_delta",
        f"{round_label}.candidate_label_ids_count_delta",
    )
    additions_count = round_payload.get("round_refine_additions_count")
    if additions_count is None:
        additions_count = 0
    if isinstance(additions_count, bool) or not isinstance(additions_count, int):
        raise ValueError(f"Missing or non-integer field: {round_label}.round_refine_additions_count")

    kept_count = round_payload.get("round_policy_kept_count")
    if kept_count is None:
        kept_count = 0
    if isinstance(kept_count, bool) or not isinstance(kept_count, int):
        raise ValueError(f"Missing or non-integer field: {round_label}.round_policy_kept_count")

    dropped_count = round_payload.get("round_policy_dropped_count")
    if dropped_count is None:
        dropped_count = 0
    if isinstance(dropped_count, bool) or not isinstance(dropped_count, int):
        raise ValueError(f"Missing or non-integer field: {round_label}.round_policy_dropped_count")

    risk_guardrail = round_payload.get("upstream_risk_guardrail_v1")
    return {
        "candidate_label_ids_count_before": count_before,
        "candidate_label_ids_count_after": count_after,
        "candidate_label_ids_count_delta": count_delta,
        "round_refine_additions_count": int(additions_count),
        "round_refine_added_label_ids": _read_int_list(round_payload, "round_refine_added_label_ids"),
        "round_policy_kept_count": int(kept_count),
        "round_policy_dropped_count": int(dropped_count),
        "upstream_risk_guardrail_v1_present": isinstance(risk_guardrail, dict),
        "upstream_risk_guardrail_v1": risk_guardrail if isinstance(risk_guardrail, dict) else None,
    }


def _build_text(snapshot: dict[str, Any]) -> str:
    loop = snapshot["loop"]
    round0 = snapshot["round0"]
    round1 = snapshot.get("round1")
    lines = [
        "STAGE_D_SNAPSHOT",
        (
            "loop"
            f" rounds_executed={loop['rounds_executed']}"
            f" additions_total={loop['round_refine_additions_count_total']}"
            f" policy_kept_total={loop['round_policy_kept_count_total']}"
            f" policy_dropped_total={loop['round_policy_dropped_count_total']}"
            f" candidate_delta_total={loop['candidate_label_ids_count_delta_total']}"
        ),
        (
            "round0"
            f" candidate_before={round0['candidate_label_ids_count_before']}"
            f" candidate_after={round0['candidate_label_ids_count_after']}"
            f" delta={round0['candidate_label_ids_count_delta']}"
            f" additions={round0['round_refine_additions_count']}"
            f" kept={round0['round_policy_kept_count']}"
            f" dropped={round0['round_policy_dropped_count']}"
        ),
    ]
    if isinstance(round1, dict):
        lines.append(
            (
                "round1"
                f" candidate_before={round1['candidate_label_ids_count_before']}"
                f" candidate_after={round1['candidate_label_ids_count_after']}"
                f" delta={round1['candidate_label_ids_count_delta']}"
                f" additions={round1['round_refine_additions_count']}"
                f" kept={round1['round_policy_kept_count']}"
                f" dropped={round1['round_policy_dropped_count']}"
                f" additions_ids={round1['round_refine_added_label_ids']}"
            )
        )
    lines.extend(
        [
            (
                "paths"
                f" loop={snapshot['artifact_paths']['loop_summary_json']}"
                f" round0={snapshot['artifact_paths']['round0_summary_json']}"
                f" round1={snapshot['artifact_paths'].get('round1_summary_json')}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _main_impl() -> int:
    args = _build_parser().parse_args()
    loop_path = args.loop_summary_json.resolve()
    loop_payload = _load_json(loop_path)

    round0_path = _resolve_round_path(
        loop_payload=loop_payload,
        loop_path=loop_path,
        round_index=0,
        explicit_path=args.round0_summary_json,
        round_summary_root=args.round_summary_root,
    )
    round1_path: Path | None = None
    rounds_executed = _require_int(loop_payload, "rounds_executed", "loop.rounds_executed")
    if rounds_executed >= 2:
        round1_path = _resolve_round_path(
            loop_payload=loop_payload,
            loop_path=loop_path,
            round_index=1,
            explicit_path=args.round1_summary_json,
            round_summary_root=args.round_summary_root,
        )

    round0_payload = _load_json(round0_path)
    round1_payload = _load_json(round1_path) if round1_path is not None else None

    loop_risk_guardrail = loop_payload.get("upstream_risk_guardrail_v1")
    snapshot: dict[str, Any] = {
        "schema_name": "wsovvis.stage_d_reporting_snapshot_v1",
        "schema_version": "1.0",
        "artifact_paths": {
            "loop_summary_json": str(loop_path),
            "round0_summary_json": str(round0_path),
            "round1_summary_json": str(round1_path) if round1_path is not None else None,
        },
        "round0": _build_round_compact(round0_payload, "round0"),
        "round1": _build_round_compact(round1_payload, "round1") if isinstance(round1_payload, dict) else None,
        "loop": {
            "rounds_executed": rounds_executed,
            "round_refine_additions_count_total": int(loop_payload.get("round_refine_additions_count_total", 0)),
            "round_policy_kept_count_total": int(loop_payload.get("round_policy_kept_count_total", 0)),
            "round_policy_dropped_count_total": int(loop_payload.get("round_policy_dropped_count_total", 0)),
            "candidate_label_ids_count_before_start": int(loop_payload.get("candidate_label_ids_count_before_start", 0)),
            "candidate_label_ids_count_after_end": int(loop_payload.get("candidate_label_ids_count_after_end", 0)),
            "candidate_label_ids_count_delta_total": int(loop_payload.get("candidate_label_ids_count_delta_total", 0)),
            "upstream_risk_guardrail_v1_present": isinstance(loop_risk_guardrail, dict),
            "upstream_risk_guardrail_v1": loop_risk_guardrail if isinstance(loop_risk_guardrail, dict) else None,
        },
    }

    text = _build_text(snapshot)
    print(text, end="")
    if args.out_json is not None:
        _write_json(args.out_json.resolve(), snapshot)
        print(f"SNAPSHOT_JSON_PATH {args.out_json.resolve()}")
    if args.out_text is not None:
        _write_text(args.out_text.resolve(), text)
        print(f"SNAPSHOT_TEXT_PATH {args.out_text.resolve()}")
    return 0


def main() -> int:
    try:
        return _main_impl()
    except Exception as exc:  # pragma: no cover - top-level fail-fast path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
