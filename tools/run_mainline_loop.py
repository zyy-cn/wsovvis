#!/usr/bin/env python3
"""Prepare a bounded WS-OVVIS supervisor prompt for one iteration."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse

ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "docs/mainline/STATUS.md"
REPORTS = ROOT / "docs/mainline/reports"

READ_ORDER = [
    "AGENTS.md",
    "START_AUTOMATION.md",
    "docs/mainline/INDEX.md",
    "docs/mainline/EXECUTION_SOURCE.md",
    "docs/mainline/PLAN.md",
    "docs/mainline/IMPLEMENT.md",
    "docs/mainline/STATUS.md",
    "docs/mainline/METRICS_ACCEPTANCE.md",
    "docs/mainline/EVIDENCE_REQUIREMENTS.md",
    "docs/mainline/FAILURE_PLAYBOOK.md",
    "docs/mainline/ENVIRONMENT_AND_VALIDATION.md",
    "docs/mainline/CODEBASE_MAP.md",
    "docs/mainline/STAGEB_INTERFACE_CONTRACT.md",
    "docs/mainline/SUPERVISOR_STATE_MACHINE.md",
    "docs/mainline/SUPERVISOR_DEPLOYMENT.md",
    "docs/runbooks/mainline_phase_gate_runbook.md",
]


def read_status_gate() -> str:
    if not STATUS.exists():
        return "UNKNOWN"
    for line in STATUS.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "Active gate:" in line:
            return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def terminal_active() -> bool:
    text = STATUS.read_text(encoding="utf-8", errors="ignore") if STATUS.exists() else ""
    return "Terminal mainline mode: `active`" in text or "Terminal mainline mode: active" in text


def build_prompt(revalidate: bool = False) -> str:
    gate = read_status_gate()
    ro = "\n".join(f"- {x}" for x in READ_ORDER)
    if terminal_active() or revalidate:
        return f"""$mainline-supervisor

Read the following files in this exact order before doing any work:
{ro}

authoritative current gate from STATUS.md: {gate}
terminal-mainline mode is active or requested.

Task:
1. Do not activate a new gate.
2. Run only bounded terminal revalidation.
3. Refresh STATUS.md and terminal report files if evidence changes.
4. Write or update docs/mainline/reports/mainline_terminal_summary.txt
5. Keep evidence artifacts current if the terminal judgment depends on them.
6. Stop after this bounded revalidation loop.
"""
    return f"""$mainline-supervisor

Read the following files in this exact order before doing any work:
{ro}

authoritative current gate from STATUS.md: {gate}

Run exactly one bounded supervisor iteration:
1. determine the current active gate and blocker,
2. identify the smallest next valid step,
3. if a scoped implementation or evidence-closing step is needed, perform only that step,
4. run acceptance evaluation and evidence evaluation,
5. update docs/mainline/STATUS.md,
6. write or update docs/mainline/reports/phase_gate_latest.txt, acceptance_latest.txt, evidence_latest.txt, and any required worked-example outputs,
7. stop.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--terminal-revalidate", action="store_true")
    args = ap.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(revalidate=args.terminal_revalidate)
    out = REPORTS / "supervisor_prompt_latest.txt"
    out.write_text(prompt, encoding="utf-8")
    print(f"saved: {out}")
    print(prompt)
    if not args.dry_run:
        print("\nNo direct Codex invocation is performed by this script.")
        print(f"timestamp: {datetime.now().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
