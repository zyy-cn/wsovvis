# WS-OVVIS G0 Worked Example — Canonical Connectivity Retry

## Goal
Show how the current bounded supervisor loop rechecked the first `G0` prerequisite, kept the repository in `G0`, and deferred the Stage B contract wording fix until canonical replay becomes recordable.

## Inputs inspected
- `AGENTS.md`
- `START_AUTOMATION.md`
- `docs/mainline/INDEX.md`
- `docs/mainline/EXECUTION_SOURCE.md`
- `docs/mainline/PLAN.md`
- `docs/mainline/STATUS.md`
- `docs/mainline/METRICS_ACCEPTANCE.md`
- `docs/mainline/EVIDENCE_REQUIREMENTS.md`
- `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
- `docs/mainline/CODEBASE_MAP.md`
- `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`
- `docs/mainline/SUPERVISOR_STATE_MACHINE.md`
- `docs/mainline/SUPERVISOR_DEPLOYMENT.md`
- `docs/runbooks/mainline_phase_gate_runbook.md`
- `tools/remote_verify_wsovvis.sh`
- `tools/check_canonical_runner_bootstrap_links.py`
- `docs/mainline/reports/archive/phase_gate_20260311T032428Z.txt`
- `docs/mainline/reports/archive/acceptance_20260311T032428Z.txt`
- `docs/mainline/reports/archive/evidence_20260311T032428Z.txt`
- `docs/mainline/reports/archive/worked_example_verification_20260311T032428Z.md`

## Concrete evidence
- The prior G0 authority-switch evidence remained valid from the archived `20260311T032428Z` report bundle.
- The current intended local commit remained `23cb9f2e2b0d1267ff33494c478d3d2a069bc1a3` on branch `mainline-v9-g0`.
- The exact retry of `git ls-remote origin refs/heads/mainline-v9-g0` on `20260311T034147Z` failed with `ssh: Could not resolve hostname github.com: Temporary failure in name resolution`.
- The exact retry of the approved wrapper command on `20260311T034147Z` failed with `ssh: connect to host 172.29.112.1 port 2222: No route to host`.
- Because the wrapper never reached the remote runner, there is still no canonical bootstrap output from `/home/zyy/code/wsovvis_runner`, no inherited conda or `PYTHONPATH` record, and no observed remote `HEAD`.

## Decision trace
1. The authoritative docs still keep the repository at `G0`, and no later gate may activate without an evidence-backed pass.
2. The current user-directed sequence makes canonical connectivity restoration the first blocking prerequisite.
3. The loop therefore reran the exact same `git ls-remote` reachability check and the exact same canonical wrapper/bootstrap command before attempting any doc reconciliation.
4. Both commands failed again, so the canonical replay record remains unavailable.
5. Because the prerequisite did not clear, the planned doc-only reconciliation in `docs/mainline/STAGEB_INTERFACE_CONTRACT.md` was intentionally deferred to preserve strict `G0` sequencing.

## Output
- active gate: `G0`
- final bounded-loop status: `BLOCKED`
- blocking acceptance condition: the canonical connectivity prerequisite still fails, so bootstrap, inherited environment, and remote-`HEAD` evidence cannot be recorded
- smallest next valid step: restore DNS and SSH reachability, rerun the same wrapper/bootstrap path, then and only then perform the doc-only Stage B contract reconciliation

## Why this example is sufficient
This worked example ties the archived local G0 evidence to the current exact retry of the canonical path, showing why the gate must remain `BLOCKED` without widening scope beyond the first unresolved prerequisite.
