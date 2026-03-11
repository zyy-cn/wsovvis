# WS-OVVIS G0 Worked Example — Remote Host Reachable, Intended Branch Still Missing

## Goal
Show how the current bounded supervisor loop retried the exact canonical replay path, confirmed that `gpu4090d` is now reachable, and still kept the repository in `G0` because the intended branch or commit is not yet recordable from the approved replay path.

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
- `docs/mainline/reports/archive/phase_gate_20260311T034147Z.txt`
- `docs/mainline/reports/archive/acceptance_20260311T034147Z.txt`
- `docs/mainline/reports/archive/evidence_20260311T034147Z.txt`
- `docs/mainline/reports/archive/worked_example_verification_20260311T034147Z.md`

## Concrete evidence
- The prior G0 blocked-connectivity evidence remained valid from the archived `20260311T034147Z` report bundle.
- The current intended local commit remained `23cb9f2e2b0d1267ff33494c478d3d2a069bc1a3` on branch `mainline-v9-g0`.
- The exact retry of `git ls-remote origin refs/heads/mainline-v9-g0` on `20260311T040711Z` still failed with `ssh: Could not resolve hostname github.com: Temporary failure in name resolution`.
- The exact retry of the approved wrapper command on `20260311T040711Z` reached the remote runner and then failed with `fatal: 'origin/mainline-v9-g0' is not a commit and a branch 'mainline-v9-g0' cannot be created from it`.
- Because the wrapper failed before the verification command, there is still no canonical bootstrap output from `/home/zyy/code/wsovvis_runner`, no inherited conda or `PYTHONPATH` record, and no observed remote `HEAD`.

## Decision trace
1. The authoritative docs still keep the repository at `G0`, and no later gate may activate without an evidence-backed pass.
2. The current user-directed sequence makes the exact canonical replay path the first prerequisite before any doc-only reconciliation.
3. The loop therefore reran the exact same `git ls-remote` reachability check and the exact same canonical wrapper/bootstrap command before attempting any doc reconciliation.
4. The wrapper now reaches `gpu4090d`, so raw remote host reachability improved, but the replay still cannot proceed because `origin/mainline-v9-g0` is not available for checkout and the direct local origin check still fails.
5. Because the replay still cannot record bootstrap, inherited environment, or remote `HEAD`, the planned doc-only reconciliation in `docs/mainline/STAGEB_INTERFACE_CONTRACT.md` was intentionally deferred to preserve strict `G0` sequencing.

## Output
- active gate: `G0`
- final bounded-loop status: `BLOCKED`
- blocking acceptance condition: the canonical replay path cannot yet reach the intended remote branch or commit, so bootstrap, inherited environment, and remote-`HEAD` evidence cannot be recorded
- smallest next valid step: make `mainline-v9-g0` reachable on origin from the approved route, rerun the same wrapper/bootstrap path, then and only then perform the doc-only Stage B contract reconciliation

## Why this example is sufficient
This worked example ties the archived blocked-connectivity evidence to the current exact retry of the canonical path, showing why the gate must remain `BLOCKED` without widening scope beyond the first unresolved replay prerequisite.
