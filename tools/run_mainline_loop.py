#!/usr/bin/env python3
"""Prepare a bounded WS-OVVIS supervisor prompt for one iteration."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

READ_ORDER = [
    'AGENTS.md',
    'START_AUTOMATION.md',
    'docs/mainline/INDEX.md',
    'docs/mainline/EXECUTION_SOURCE.md',
    'docs/mainline/PLAN.md',
    'docs/mainline/IMPLEMENT.md',
    'docs/mainline/STATUS.md',
    'docs/mainline/METRICS_ACCEPTANCE.md',
    'docs/mainline/EVIDENCE_REQUIREMENTS.md',
    'docs/mainline/FAILURE_PLAYBOOK.md',
    'docs/mainline/ENVIRONMENT_AND_VALIDATION.md',
    'docs/mainline/CODEBASE_MAP.md',
    'docs/mainline/STAGEB_INTERFACE_CONTRACT.md',
    'docs/mainline/SUPERVISOR_STATE_MACHINE.md',
    'docs/mainline/SUPERVISOR_DEPLOYMENT.md',
    'docs/runbooks/mainline_phase_gate_runbook.md',
]

TERMINAL_REFERENCES = [
    'tools/remote_verify_wsovvis.sh',
    'tests/test_stage_d11_canonical_replay_v1.py',
    'tests/test_stage_d_reporting_snapshot_v1.py',
    'tests/test_stage_d13_ci_quick_pipeline_v1.py',
]


def _status_path(repo_root: Path) -> Path:
    return repo_root / 'docs/mainline/STATUS.md'


def _reports_dir(repo_root: Path) -> Path:
    return repo_root / 'docs/mainline/reports'


def read_status_gate(repo_root: Path) -> str:
    status = _status_path(repo_root)
    if not status.exists():
        return 'UNKNOWN'
    for line in status.read_text(encoding='utf-8', errors='ignore').splitlines():
        if 'Active gate:' in line:
            return line.split(':', 1)[1].strip()
    return 'UNKNOWN'


def terminal_active(repo_root: Path) -> bool:
    status = _status_path(repo_root)
    text = status.read_text(encoding='utf-8', errors='ignore') if status.exists() else ''
    return 'Terminal mainline mode: `active`' in text or 'Terminal mainline mode: active' in text


def _terminal_summary_text(gate: str) -> str:
    refs = '\n'.join(f'- {ref}' for ref in TERMINAL_REFERENCES)
    return (
        '# WS-OVVIS terminal mainline summary\n\n'
        f'Accepted gate: {gate}\n\n'
        'Terminal policy:\n'
        '- terminal mainline mode is active\n'
        '- no new algorithm development is authorized under the current mainline\n'
        '- only bounded terminal revalidation may run\n\n'
        f'Key terminal references:\n{refs}\n'
    )


def build_prompt(repo_root: Path, revalidate: bool = False) -> str:
    gate = read_status_gate(repo_root)
    ro = '\n'.join(f'- {x}' for x in READ_ORDER)
    if terminal_active(repo_root) or revalidate:
        return (
            '$mainline-supervisor\n\n'
            'Read the following files in this exact order before doing any work:\n'
            f'{ro}\n\n'
            f'authoritative current gate from STATUS.md: {gate}\n'
            'terminal-mainline mode is active or requested.\n'
            'canonical remote profile: gpu4090d\n\n'
            'Task:\n'
            '1. Operate strictly in bounded terminal revalidation mode.\n'
            '2. Do not activate a new gate.\n'
            '3. Refresh STATUS.md and terminal report files if evidence changes.\n'
            '4. Write or update docs/mainline/reports/mainline_terminal_summary.txt.\n'
            '5. Keep evidence artifacts current if the terminal judgment depends on them.\n'
            '6. Do not continue algorithm development.\n'
            '7. Stop after this bounded revalidation loop.\n'
        )
    return (
        '$mainline-supervisor\n\n'
        'Read the following files in this exact order before doing any work:\n'
        f'{ro}\n\n'
        f'authoritative current gate from STATUS.md: {gate}\n\n'
        'Run exactly one bounded supervisor iteration.\n'
        '1. Determine the current active gate and blocker.\n'
        '2. Identify the smallest next valid step.\n'
        '3. If a scoped implementation or evidence-closing step is required, perform only that step.\n'
        '4. Run acceptance evaluation and evidence evaluation.\n'
        '5. Update docs/mainline/STATUS.md.\n'
        '6. Write or update docs/mainline/reports/phase_gate_latest.txt, acceptance_latest.txt, evidence_latest.txt, and any required worked-example outputs.\n'
        '7. Stop.\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--terminal-revalidate', action='store_true')
    args = ap.parse_args()

    repo_root = args.repo_root.resolve()
    reports = _reports_dir(repo_root)
    reports.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt(repo_root, revalidate=args.terminal_revalidate)
    out = reports / 'supervisor_prompt_latest.txt'
    out.write_text(prompt, encoding='utf-8')
    print(f'saved: {out}')
    print(prompt)

    if terminal_active(repo_root) or args.terminal_revalidate:
        summary = reports / 'mainline_terminal_summary.txt'
        summary.write_text(_terminal_summary_text(read_status_gate(repo_root)), encoding='utf-8')
        print(f'saved: {summary}')

    if not args.dry_run:
        print('\nNo direct Codex invocation is performed by this script.')
        print(f'timestamp: {datetime.now().isoformat(timespec="seconds")}')


if __name__ == '__main__':
    main()
