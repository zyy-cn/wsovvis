#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Stage D0 self-training loop skeleton (round0/round1)")
    p.add_argument("--round-index", type=int, default=0, help="Starting round index")
    p.add_argument("--max-rounds", type=int, default=1, help="Exclusive max round index")
    p.add_argument(
        "--refine-mode",
        choices=("none", "minimal"),
        default="none",
        help="Refine mode applied before rounds after round0",
    )
    p.add_argument(
        "--round-policy",
        choices=("none", "minimal_curriculum_v1"),
        default="none",
        help="Optional round-level policy gate applied during round input construction (round>=1)",
    )
    p.add_argument(
        "--tiny-pinned",
        action="store_true",
        help="Enable tiny/canonical smoke mode with synthetic fallback seed",
    )
    p.add_argument(
        "--stagec-summary-json",
        type=Path,
        default=None,
        help="Optional Stage C summary JSON used as round0 seed input",
    )
    p.add_argument(
        "--stagec-summary-in-json",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--round-summary-root",
        type=Path,
        default=Path("outputs/stage_d0_round_summaries"),
        help="Directory for round input/output/summary JSON files",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional path to write top-level Stage D0 run summary JSON",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed for bounded smoke reproducibility",
    )
    return p


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON must be object: {path}")
    return payload


def _as_int_list(values: Any) -> list[int]:
    out: list[int] = []
    if not isinstance(values, (list, tuple)):
        return out
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            out.append(int(value))
    return out


def _seed_from_stagec_summary(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    selected_video_id = payload.get("selected_video_id")
    if not isinstance(selected_video_id, str) or not selected_video_id.strip():
        raise ValueError(f"Stage C summary missing non-empty selected_video_id: {path}")
    selected_positive_ids = _as_int_list(payload.get("selected_positive_label_ids"))
    final = payload.get("final")
    final_candidate_ids = _as_int_list(final.get("candidate_label_ids")) if isinstance(final, dict) else []
    candidate_ids = final_candidate_ids or selected_positive_ids
    if not candidate_ids:
        raise ValueError(
            f"Stage C summary must provide candidate labels via final.candidate_label_ids "
            f"or selected_positive_label_ids: {path}"
        )
    ws_metrics = payload.get("ws_metrics_summary_v1")
    unknown_diag = payload.get("unknown_handling_diagnostics_v1")
    risk_guardrail = unknown_diag.get("risk_guardrail_v1") if isinstance(unknown_diag, dict) else None
    return {
        "source_kind": "stagec_summary_json",
        "source_path": str(path.resolve()),
        "selected_video_id": selected_video_id,
        "positive_label_ids": selected_positive_ids,
        "candidate_label_ids": candidate_ids,
        "assignment_backend": str(payload.get("assignment_backend_requested", "")),
        "ws_metrics_summary_v1": ws_metrics if isinstance(ws_metrics, dict) else None,
        "upstream_risk_guardrail_v1": risk_guardrail if isinstance(risk_guardrail, dict) else None,
    }


def _seed_from_tiny_pinned() -> dict[str, Any]:
    return {
        "source_kind": "tiny_pinned_synthetic_seed",
        "source_path": None,
        "selected_video_id": "tiny_pinned_synthetic",
        "positive_label_ids": [101, 202],
        "candidate_label_ids": [101, 202, 303],
        "assignment_backend": "d0_tiny_seed_v1",
        "ws_metrics_summary_v1": None,
        "upstream_risk_guardrail_v1": None,
    }


def _minimal_refine(previous_round_output: dict[str, Any]) -> dict[str, Any]:
    base_ids = _as_int_list(previous_round_output.get("candidate_label_ids"))
    if not base_ids:
        base_ids = _as_int_list(previous_round_output.get("positive_label_ids"))
    refined = sorted(set(base_ids + [909001]))
    return {
        "schema_name": "wsovvis.stage_d_refine_summary_v1",
        "schema_version": "1.0",
        "refine_mode": "minimal",
        "applied": True,
        "rule_id": "append_refine_marker_label_v1",
        "input_candidate_label_ids": base_ids,
        "output_candidate_label_ids": refined,
        "added_label_ids": [x for x in refined if x not in set(base_ids)],
    }


def _apply_round_policy(
    *,
    round_index: int,
    round_policy: str,
    previous_candidate_ids: list[int],
    refined_candidate_ids: list[int],
) -> tuple[list[int], bool, str, dict[str, Any]]:
    if round_policy == "none":
        return refined_candidate_ids, False, "round_policy=none; no round-policy gating applied", {}
    if round_policy != "minimal_curriculum_v1":
        raise ValueError(f"unsupported --round-policy: {round_policy}")
    if round_index < 1:
        return (
            refined_candidate_ids,
            False,
            "round_policy=minimal_curriculum_v1 is active only for round>=1",
            {"k": 1, "reason": "round<1"},
        )

    k = 1
    base_set = set(previous_candidate_ids)
    additions = [label_id for label_id in refined_candidate_ids if label_id not in base_set]
    kept_additions = additions[:k]
    kept_set = set(kept_additions)
    gated_candidate_ids = [label_id for label_id in refined_candidate_ids if label_id in base_set or label_id in kept_set]
    dropped_additions = [label_id for label_id in additions if label_id not in kept_set]
    notes = (
        "round_policy=minimal_curriculum_v1 cap_additions_k=1 "
        f"(input_additions={len(additions)}, kept={len(kept_additions)}, dropped={len(dropped_additions)})"
    )
    stats: dict[str, Any] = {
        "k": k,
        "input_additions_count": len(additions),
        "kept_addition_ids": kept_additions,
        "dropped_addition_ids": dropped_additions,
    }
    return gated_candidate_ids, True, notes, stats


def _build_round_input_summary(
    *,
    round_index: int,
    max_rounds: int,
    tiny_pinned: bool,
    refine_mode: str,
    state: dict[str, Any],
    refine_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    ws_metrics = state.get("ws_metrics_summary_v1")
    upstream_risk_guardrail = state.get("upstream_risk_guardrail_v1")
    return {
        "schema_name": "wsovvis.stage_d_round_input_summary_v1",
        "schema_version": "1.0",
        "round_index": int(round_index),
        "max_rounds": int(max_rounds),
        "tiny_pinned": bool(tiny_pinned),
        "refine_mode": str(refine_mode),
        "source_kind": str(state.get("source_kind", "unknown")),
        "source_path": state.get("source_path"),
        "selected_video_id": str(state.get("selected_video_id", "unknown")),
        "positive_label_ids": _as_int_list(state.get("positive_label_ids")),
        "candidate_label_ids": _as_int_list(state.get("candidate_label_ids")),
        "assignment_backend": str(state.get("assignment_backend", "")),
        "ws_metrics_summary_v1_present": isinstance(ws_metrics, dict),
        "upstream_risk_guardrail_v1_present": isinstance(upstream_risk_guardrail, dict),
        "refine_summary": refine_summary,
    }


def _build_round_output_summary(
    *,
    round_input: dict[str, Any],
) -> dict[str, Any]:
    candidate_ids = _as_int_list(round_input.get("candidate_label_ids"))
    return {
        "schema_name": "wsovvis.stage_d_round_output_summary_v1",
        "schema_version": "1.0",
        "round_index": int(round_input["round_index"]),
        "selected_video_id": str(round_input["selected_video_id"]),
        "assignment_backend": str(round_input["assignment_backend"]),
        "positive_label_ids": _as_int_list(round_input.get("positive_label_ids")),
        "candidate_label_ids": candidate_ids,
        "num_candidate_labels": int(len(candidate_ids)),
        "orchestration_status": "pass_through_baseline",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _build_parser().parse_args()
    if args.seed is not None:
        random.seed(int(args.seed))
    if args.round_index < 0:
        raise ValueError("--round-index must be >= 0")
    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be >= 1")
    if args.round_index >= args.max_rounds:
        raise ValueError("--round-index must be < --max-rounds")

    stagec_summary_path = args.stagec_summary_json or args.stagec_summary_in_json
    if stagec_summary_path is not None:
        seed_state = _seed_from_stagec_summary(stagec_summary_path.resolve())
    elif args.tiny_pinned:
        seed_state = _seed_from_tiny_pinned()
    else:
        raise ValueError("provide --stagec-summary-in-json or enable --tiny-pinned")

    round_paths: list[dict[str, str]] = []
    round_summaries: list[dict[str, Any]] = []
    current_state = seed_state
    previous_round_output: dict[str, Any] | None = None

    for round_index in range(int(args.round_index), int(args.max_rounds)):
        refine_summary: dict[str, Any] | None = None
        candidate_count_before = 0
        round_policy_kept_count = 0
        round_policy_dropped_count = 0
        if previous_round_output is not None:
            if args.refine_mode == "minimal":
                refine_summary = _minimal_refine(previous_round_output)
                refined_ids = _as_int_list(refine_summary["output_candidate_label_ids"])
            else:
                refined_ids = _as_int_list(previous_round_output.get("candidate_label_ids"))

            previous_ids = _as_int_list(previous_round_output.get("candidate_label_ids"))
            candidate_count_before = int(len(previous_ids))
            gated_ids, round_policy_applied, round_policy_notes, round_policy_stats = _apply_round_policy(
                round_index=round_index,
                round_policy=str(args.round_policy),
                previous_candidate_ids=previous_ids,
                refined_candidate_ids=refined_ids,
            )
            round_policy_kept_count = int(len(_as_int_list(round_policy_stats.get("kept_addition_ids"))))
            round_policy_dropped_count = int(len(_as_int_list(round_policy_stats.get("dropped_addition_ids"))))
            current_state = {
                "source_kind": "previous_round_output",
                "source_path": None,
                "selected_video_id": previous_round_output.get("selected_video_id"),
                "positive_label_ids": _as_int_list(previous_round_output.get("positive_label_ids")),
                "candidate_label_ids": gated_ids,
                "assignment_backend": previous_round_output.get("assignment_backend", ""),
                "ws_metrics_summary_v1": current_state.get("ws_metrics_summary_v1"),
                "upstream_risk_guardrail_v1": current_state.get("upstream_risk_guardrail_v1"),
            }
        else:
            round_policy_applied = False
            round_policy_notes = (
                "round_policy=minimal_curriculum_v1 is active only for round>=1"
                if args.round_policy == "minimal_curriculum_v1"
                else "round_policy=none; no round-policy gating applied"
            )
            round_policy_stats = {"reason": "round0_seed"}
            candidate_count_before = int(len(_as_int_list(current_state.get("candidate_label_ids"))))

        round_input = _build_round_input_summary(
            round_index=round_index,
            max_rounds=int(args.max_rounds),
            tiny_pinned=bool(args.tiny_pinned),
            refine_mode=str(args.refine_mode),
            state=current_state,
            refine_summary=refine_summary,
        )
        round_output = _build_round_output_summary(round_input=round_input)
        round_refine_added_label_ids = (
            _as_int_list(refine_summary.get("added_label_ids")) if isinstance(refine_summary, dict) else []
        )
        candidate_count_after = int(len(_as_int_list(round_output.get("candidate_label_ids"))))
        round_summary = {
            "schema_name": "wsovvis.stage_d_round_summary_v1",
            "schema_version": "1.0",
            "round_index": int(round_index),
            "max_rounds": int(args.max_rounds),
            "tiny_pinned": bool(args.tiny_pinned),
            "refine_mode_requested": str(args.refine_mode),
            "refine_applied": bool(refine_summary is not None and refine_summary.get("applied") is True),
            "round_policy_name": str(args.round_policy),
            "round_policy_applied": bool(round_policy_applied),
            "round_policy_notes": str(round_policy_notes),
            "round_policy_stats": round_policy_stats,
            "round_refine_additions_count": int(len(round_refine_added_label_ids)),
            "round_refine_added_label_ids": round_refine_added_label_ids,
            "round_policy_kept_count": int(round_policy_kept_count),
            "round_policy_dropped_count": int(round_policy_dropped_count),
            "candidate_label_ids_count_before": int(candidate_count_before),
            "candidate_label_ids_count_after": int(candidate_count_after),
            "candidate_label_ids_count_delta": int(candidate_count_after - candidate_count_before),
            "round_input_summary": round_input,
            "round_output_summary": round_output,
            "ws_metrics_summary_v1": current_state.get("ws_metrics_summary_v1"),
            "upstream_risk_guardrail_v1": current_state.get("upstream_risk_guardrail_v1"),
        }

        input_path = args.round_summary_root / f"round{round_index}_input_summary.json"
        output_path = args.round_summary_root / f"round{round_index}_output_summary.json"
        summary_path = args.round_summary_root / f"round{round_index}_summary.json"
        _write_json(input_path, round_input)
        _write_json(output_path, round_output)
        _write_json(summary_path, round_summary)
        print(f"D0_ROUND_INPUT_PATH {input_path}")
        print(f"D0_ROUND_OUTPUT_PATH {output_path}")
        print(f"D0_ROUND_SUMMARY_PATH {summary_path}")

        round_paths.append(
            {
                "round_input_summary_path": str(input_path),
                "round_output_summary_path": str(output_path),
                "round_summary_path": str(summary_path),
            }
        )
        round_summaries.append(round_summary)
        previous_round_output = round_output

    result = {
        "status": "PASS",
        "schema_name": "wsovvis.stage_d_loop_summary_v1",
        "schema_version": "1.0",
        "seed": int(args.seed) if args.seed is not None else None,
        "round_index_start": int(args.round_index),
        "max_rounds": int(args.max_rounds),
        "refine_mode": str(args.refine_mode),
        "round_policy_name": str(args.round_policy),
        "round_policy_applied": any(bool(s.get("round_policy_applied")) for s in round_summaries),
        "round_policy_notes": (
            "round_policy=none; no round-policy gating applied"
            if args.round_policy == "none"
            else "round_policy=minimal_curriculum_v1 cap_additions_k=1 on round>=1"
        ),
        "round_policy_stats": {
            "rounds_with_policy_applied": [
                int(s.get("round_index"))
                for s in round_summaries
                if bool(s.get("round_policy_applied"))
            ],
            "policy_applied_round_count": int(
                sum(1 for s in round_summaries if bool(s.get("round_policy_applied")))
            ),
        },
        "round_refine_additions_count_total": int(
            sum(int(s.get("round_refine_additions_count", 0)) for s in round_summaries)
        ),
        "round_policy_kept_count_total": int(sum(int(s.get("round_policy_kept_count", 0)) for s in round_summaries)),
        "round_policy_dropped_count_total": int(
            sum(int(s.get("round_policy_dropped_count", 0)) for s in round_summaries)
        ),
        "candidate_label_ids_count_before_start": (
            int(round_summaries[0].get("candidate_label_ids_count_before")) if round_summaries else 0
        ),
        "candidate_label_ids_count_after_end": (
            int(round_summaries[-1].get("candidate_label_ids_count_after")) if round_summaries else 0
        ),
        "candidate_label_ids_count_delta_total": (
            int(round_summaries[-1].get("candidate_label_ids_count_after", 0))
            - int(round_summaries[0].get("candidate_label_ids_count_before", 0))
            if round_summaries
            else 0
        ),
        "upstream_risk_guardrail_v1_present": isinstance(seed_state.get("upstream_risk_guardrail_v1"), dict),
        "upstream_risk_guardrail_v1": seed_state.get("upstream_risk_guardrail_v1"),
        "tiny_pinned": bool(args.tiny_pinned),
        "rounds_executed": int(len(round_summaries)),
        "round_paths": round_paths,
        "round_summaries": round_summaries,
    }
    print("D0_LOOP_SUMMARY", json.dumps(result, sort_keys=True))
    if args.out_json is not None:
        _write_json(args.out_json, result)
        print(f"D0_LOOP_SUMMARY_PATH {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
