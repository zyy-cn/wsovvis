# WS-OVVIS Mainline Implement Runbook

This file defines how Codex should work inside the repository.

## 1. Before editing code
Always do these first:
1. Read `AGENTS.md` and all files in `docs/mainline/` referenced from `INDEX.md`.
2. Determine the active gate from `STATUS.md`.
3. Identify the blocking acceptance condition from `METRICS_ACCEPTANCE.md`.
4. Identify the required evidence pack from `EVIDENCE_REQUIREMENTS.md`.
5. Identify the smallest valid step toward that gate.
6. Identify the fallback path if this step fails.

Do not code before these are clear.

## 2. Scope control
- Edit only files needed for the current gate.
- Do not enable default-off modules.
- Do not broaden a failing experiment into a larger redesign.
- Do not reinterpret environment failures as algorithmic failures.

## 3. Command discipline
- Local checks are informative.
- Canonical PASS depends on `ENVIRONMENT_AND_VALIDATION.md`.
- A gate PASS depends on both the acceptance contract and the required evidence pack.
- Record files changed, commands run, local results, remote results, intended commit, remote HEAD, and whether they match.

## 4. Output discipline
- Task artifacts belong under `codex/<task_dir>/`.
- Mainline gate and acceptance artifacts belong under `docs/mainline/reports/`.
- Do not write loose `*_output.txt` files in the repo root.

## 5. Mandatory report outputs
- every phase/gate check must write `docs/mainline/reports/phase_gate_latest.txt`
- every acceptance evaluation must write `docs/mainline/reports/acceptance_latest.txt`
- every evidence review must write `docs/mainline/reports/evidence_latest.txt`
- every gate requiring a worked example must write `docs/mainline/reports/worked_example_verification_latest.md` and `.json`
- all of the above should also write timestamped archival copies under `docs/mainline/reports/archive/`
- report outputs must be sufficient for a future session to understand why the gate was or was not passed
- all state-bearing outputs must update `docs/mainline/STATUS.md` if state materially changed

## 6. Entry-point discipline by gate
- G0: docs, startup instructions, skills, and supervisor/tooling only
- G1: protocol tooling and contract parsing only
- G2: pseudo-tube / class-agnostic basis / local-tracklet entrypoints only
- G3: global-track-bank entrypoints only
- G4: DINO semantic-cache entrypoints only
- G5: prototype-bank and text-map entrypoints only
- G6: core attribution and aligned hidden-positive metrics only
- G7: bag-free inference and evaluation entrypoints only

Do not mix later-gate algorithmic work into earlier-gate loops.

## 7. Terminal accepted gate rule
When the accepted terminal gate is active and already `PASS`:
- stop automatically
- do not propose or execute a new coding step
- write/update `docs/mainline/reports/mainline_terminal_summary.txt`
- treat bounded terminal revalidation as the only allowed follow-up mode under the current mainline
