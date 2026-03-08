#!/usr/bin/env python3
"""Bounded document-driven supervisor prompt generator for WS-OVVIS.

This helper keeps the automation flow conservative:
- before the terminal gate, it prepares one bounded supervisor prompt
- at the accepted terminal gate, it stops automatically and writes a
  terminal summary instead of proposing a new coding step
- it can optionally prepare a bounded terminal revalidation prompt
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
READ_ORDER = [
    "AGENTS.md",
    "START_AUTOMATION.md",
    "docs/mainline/INDEX.md",
    "docs/mainline/EXECUTION_SOURCE.md",
    "docs/mainline/PLAN.md",
    "docs/mainline/IMPLEMENT.md",
    "docs/mainline/STATUS.md",
    "docs/mainline/METRICS_ACCEPTANCE.md",
    "docs/mainline/FAILURE_PLAYBOOK.md",
    "docs/mainline/ENVIRONMENT_AND_VALIDATION.md",
    "docs/mainline/CODEBASE_MAP.md",
    "docs/mainline/STAGEB_INTERFACE_CONTRACT.md",
    "docs/runbooks/mainline_phase_gate_runbook.md",
    "docs/mainline/SUPERVISOR_STATE_MACHINE.md",
    "docs/mainline/SUPERVISOR_DEPLOYMENT.md",
]

TERMINAL_GATE_TOKEN = "G6"
TERMINAL_STATUS_LABEL = "Terminal mainline mode"
TERMINAL_STATUS_ACTIVE = "active"

TERMINAL_REGRESSION_SUITE = {
    "g4": [
        "tests/test_stagec2_labelset_proto_baseline_v1.py::test_decoder_independent_matches_scorer_predictions",
        "tests/test_stagec4_sinkhorn_scorer_v1.py::test_sinkhorn_c43_unk_fg_gating_schema_and_behavior",
        "tests/test_ws_metrics_reporting_v1.py::test_ws_metrics_summary_v1_exposes_hpr_and_uar_when_hidden_positive_inputs_exist",
    ],
    "g5": [
        "tests/test_g5_bounded_policy_v1.py",
    ],
    "g6": [
        "tests/test_stage_d0_self_training_loop_v1.py::test_d0_round0_round1_with_minimal_refine_and_stagec_seed",
        "tests/test_stage_d0_self_training_loop_v1.py::test_d0_emit_ws_metrics_preserves_hidden_positive_fields_for_hpr_uar",
        "tests/test_stage_d_reporting_snapshot_v1.py::test_stage_d_snapshot_happy_path_detects_round_paths",
        "tests/test_stage_d_ws_metrics_adapter_v1.py::test_stage_d_ws_metrics_adapter_exposes_hpr_and_uar_when_optional_fields_exist",
    ],
}


@dataclass(frozen=True)
class Paths:
    root: Path
    status: Path
    acceptance: Path
    reports: Path
    prompt_latest: Path
    terminal_summary: Path


@dataclass(frozen=True)
class SupervisorState:
    active_gate_line: str
    active_gate_token: str
    gate_judgment: str
    terminal_mode_explicit: bool
    terminal_detected: bool
    terminal_reason: str
    no_further_gate_reported: bool


def _build_paths(repo_root: Path) -> Paths:
    reports = repo_root / "docs/mainline/reports"
    return Paths(
        root=repo_root,
        status=repo_root / "docs/mainline/STATUS.md",
        acceptance=reports / "acceptance_latest.txt",
        reports=reports,
        prompt_latest=reports / "supervisor_prompt_latest.txt",
        terminal_summary=reports / "mainline_terminal_summary.txt",
    )


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _find_status_value(text: str, label: str) -> str:
    label_low = label.lower()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped_low = stripped.lower()
        if stripped_low.startswith("- "):
            stripped = stripped[2:].strip()
            stripped_low = stripped.lower()
        if stripped_low.startswith(label_low):
            _, _, value = stripped.partition(":")
            return value.strip()
    return ""


def _extract_gate_token(raw_gate: str) -> str:
    match = re.search(r"\bG\d+\b", raw_gate)
    return match.group(0) if match else "UNKNOWN"


def infer_state(paths: Paths) -> SupervisorState:
    status_text = _read_text(paths.status)
    acceptance_text = _read_text(paths.acceptance)

    active_gate_line = _find_status_value(status_text, "Active gate")
    active_gate_token = _extract_gate_token(active_gate_line)
    gate_judgment = _find_status_value(status_text, f"Current {active_gate_token} judgment") if active_gate_token != "UNKNOWN" else ""

    explicit_terminal_value = _find_status_value(status_text, TERMINAL_STATUS_LABEL)
    terminal_mode_explicit = explicit_terminal_value.strip("`").lower() == TERMINAL_STATUS_ACTIVE
    no_further_gate_reported = "no further gate" in acceptance_text.lower()

    terminal_detected = False
    terminal_reason = "not_terminal"
    if terminal_mode_explicit and active_gate_token == TERMINAL_GATE_TOKEN and gate_judgment.strip("`").upper() == "PASS":
        terminal_detected = True
        terminal_reason = "explicit_terminal_status"
    elif active_gate_token == TERMINAL_GATE_TOKEN and gate_judgment.strip("`").upper() == "PASS":
        terminal_detected = True
        terminal_reason = "inferred_from_terminal_gate_pass"

    return SupervisorState(
        active_gate_line=active_gate_line or "UNKNOWN",
        active_gate_token=active_gate_token,
        gate_judgment=gate_judgment or "UNKNOWN",
        terminal_mode_explicit=terminal_mode_explicit,
        terminal_detected=terminal_detected,
        terminal_reason=terminal_reason,
        no_further_gate_reported=no_further_gate_reported,
    )


def _render_read_order() -> str:
    return "\n".join(f"- {path}" for path in READ_ORDER)


def _terminal_pytest_targets() -> list[str]:
    targets: list[str] = []
    for key in ("g4", "g5", "g6"):
        targets.extend(TERMINAL_REGRESSION_SUITE[key])
    return targets


def build_progress_prompt(state: SupervisorState) -> str:
    return f"""$wsovvis-mainline-supervisor

Read the following files in this exact order before doing any work:
{_render_read_order()}

Current known gate from STATUS.md: {state.active_gate_line}

Run exactly one bounded supervisor iteration.
Tasks:
1. Determine the active gate and blocker.
2. Identify the smallest next valid step.
3. If a scoped implementation step is required, perform only that step.
4. Run acceptance evaluation for the current gate.
5. Update docs/mainline/STATUS.md if needed.
6. Write a full loop report to docs/mainline/reports/supervisor_latest.txt
7. Also write a timestamped archival copy under docs/mainline/reports/archive/

Hard constraints:
- Do not widen scope.
- Do not enable default-off modules.
- Do not introduce new algorithmic mechanisms during gate plumbing.
- Prefer canonical remote validation when local execution is blocked.
- During G0, prioritize canonical environment inheritance verification.
- Record remote HEAD consistency whenever canonical validation is used.
- Stop after one bounded iteration and report the result.
"""


def build_terminal_notice(state: SupervisorState, summary_path: Path) -> str:
    return f"""WSOVVIS terminal mainline detected.

Active terminal gate: {state.active_gate_line}
Current judgment: {state.gate_judgment}
Terminal detection reason: {state.terminal_reason}

No new coding step was generated.
Terminal summary written to: {summary_path}

To prepare a bounded terminal revalidation prompt instead of a new gate-progression prompt, run:
`python tools/run_mainline_loop.py --terminal-revalidate --dry-run`
"""


def build_terminal_revalidation_prompt(state: SupervisorState) -> str:
    g4_targets = "\n".join(f"  - {target}" for target in TERMINAL_REGRESSION_SUITE["g4"])
    g5_targets = "\n".join(f"  - {target}" for target in TERMINAL_REGRESSION_SUITE["g5"])
    g6_targets = "\n".join(f"  - {target}" for target in TERMINAL_REGRESSION_SUITE["g6"])
    pytest_targets = " ".join(_terminal_pytest_targets())
    return f"""$wsovvis-mainline-supervisor

Read the following files in this exact order before doing any work:
{_render_read_order()}

Current known state:
- terminal mainline mode is active
- accepted terminal gate from STATUS.md: {state.active_gate_line}
- current judgment: {state.gate_judgment}
- no further gate should activate under the current documented scope

Operate strictly in bounded terminal revalidation mode.
Do not continue algorithm development.
Do not open a new gate.
Do not expand scope.

Tasks:
1. Re-run only the authoritative terminal regression suite.
2. Preserve the accepted G4 bounded attribution path.
3. Preserve the accepted G5 bounded linking / quality-weighted classification policy.
4. Preserve the accepted G6 single-round bounded refinement path.
5. Use canonical `gpu4090d` validation semantics and record remote HEAD consistency.
6. Update `docs/mainline/STATUS.md`, `docs/mainline/reports/phase_gate_latest.txt`, `docs/mainline/reports/acceptance_latest.txt`, and `docs/mainline/reports/mainline_terminal_summary.txt` only if revalidation evidence changes.
7. Stop after the bounded revalidation run.

Authoritative terminal regression suite:
- G4 bounded attribution anchors:
{g4_targets}
- G5 bounded linking / classification anchors:
{g5_targets}
- G6 bounded refinement anchors:
{g6_targets}

Canonical validation semantics to preserve:
- use `bash tools/remote_verify_wsovvis.sh`
- remote host alias: `gpu4090d`
- canonical repo dir: `/home/zyy/code/wsovvis_runner`
- bootstrap preflight when canonical validation is used:
  `python tools/check_canonical_runner_bootstrap_links.py --check`
- bounded pytest subset for terminal revalidation:
  `pytest -q {pytest_targets}`
- remote PASS counts only if `remote HEAD == intended local commit`

Hard constraints:
- no new research scope
- no label-set expansion
- no second-round refinement
- no enhanced memory aggregation
- no broader Stage D continuation
- no algorithmic expansion
- if the terminal regression suite passes, write/update the terminal summary and stop
- if terminal revalidation is blocked by infra or remote mismatch, classify it as `BLOCKED` / `INCONCLUSIVE`, not as algorithmic failure
"""


def build_terminal_summary(state: SupervisorState, now_utc: str) -> str:
    g4_targets = "\n".join(f"- {target}" for target in TERMINAL_REGRESSION_SUITE["g4"])
    g5_targets = "\n".join(f"- {target}" for target in TERMINAL_REGRESSION_SUITE["g5"])
    g6_targets = "\n".join(f"- {target}" for target in TERMINAL_REGRESSION_SUITE["g6"])
    no_further_gate_text = "yes" if state.no_further_gate_reported else "not explicitly recorded in acceptance_latest.txt"
    return f"""WSOVVIS terminal mainline summary
Timestamp (UTC): {now_utc}

1. Terminal status
- active gate: {state.active_gate_line}
- current gate judgment: {state.gate_judgment}
- terminal gate token: `{TERMINAL_GATE_TOKEN}`
- terminal detection explicit in STATUS.md: `{"yes" if state.terminal_mode_explicit else "no"}`
- terminal detection result: `{"active" if state.terminal_detected else "inactive"}`
- terminal detection reason: `{state.terminal_reason}`
- acceptance_latest reports no further gate: `{no_further_gate_text}`

2. Terminal automation behavior
- default supervisor behavior at terminal mainline:
  stop automatically
  do not generate a new coding step
  write/update `docs/mainline/reports/mainline_terminal_summary.txt`
- allowed bounded follow-up mode:
  `python tools/run_mainline_loop.py --terminal-revalidate --dry-run`

3. Authoritative bounded terminal regression suite
- G4 bounded attribution regression anchors:
{g4_targets}
- G5 bounded linking / quality-weighted classification regression anchors:
{g5_targets}
- G6 single-round bounded refinement regression anchors:
{g6_targets}

4. Canonical validation semantics for terminal revalidation
- canonical host alias: `gpu4090d`
- canonical repo dir: `/home/zyy/code/wsovvis_runner`
- canonical wrapper: `tools/remote_verify_wsovvis.sh`
- bootstrap preflight:
  `python tools/check_canonical_runner_bootstrap_links.py --check`
- remote PASS counts only if `remote HEAD == intended local commit`

5. Scope guardrails
- no new gate activation under the current documented mainline
- no algorithm development after terminal acceptance
- no label-set expansion
- no second-round refinement
- no enhanced memory aggregation
- no broader Stage D continuation
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--terminal-revalidate", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    paths = _build_paths(repo_root)
    paths.reports.mkdir(parents=True, exist_ok=True)

    state = infer_state(paths)
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if state.terminal_detected:
        terminal_summary = build_terminal_summary(state, now_utc)
        paths.terminal_summary.write_text(terminal_summary, encoding="utf-8")
        if args.terminal_revalidate:
            payload = build_terminal_revalidation_prompt(state)
            print(
                f"[wsovvis-mainline-supervisor] prepared bounded terminal revalidation prompt for {state.active_gate_line}"
            )
        else:
            payload = build_terminal_notice(state, paths.terminal_summary)
            print(f"[wsovvis-mainline-supervisor] terminal mainline detected at {state.active_gate_line}")
        paths.prompt_latest.write_text(payload, encoding="utf-8")
        print(f"saved: {paths.prompt_latest}")
        print(f"saved terminal summary: {paths.terminal_summary}")
        print("---")
        print(payload)
        return

    prompt = build_progress_prompt(state)
    paths.prompt_latest.write_text(prompt, encoding="utf-8")

    print(f"[wsovvis-mainline-supervisor] prepared prompt for gate: {state.active_gate_line}")
    print(f"saved: {paths.prompt_latest}")
    print("---")
    print(prompt)
    if args.dry_run:
        return
    print("\nNo direct Codex invocation is performed by this script.")
    print("Use the generated prompt with `codex exec` or in an interactive Codex session.")
    print(f"Requested max iterations: {args.max_iterations}")
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
