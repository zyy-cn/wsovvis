# WSOVVIS Mainline Implement Runbook

This file defines how Codex should work inside the repository.
It is the authoritative execution runbook for the clean automation kit.

## 1. Before editing code
Always do these first:
1. Read `AGENTS.md` and all files in `docs/mainline/` referenced from `INDEX.md`.
2. Determine the active gate from `STATUS.md`.
3. Identify the blocking acceptance condition from `METRICS_ACCEPTANCE.md`.
4. Identify the smallest valid step toward that gate.
5. Identify the fallback path if this step fails.

Do not code before these are clear.

## 2. Scope control
- Edit only files needed for the current gate.
- Do not enable default-off modules.
- Do not broaden a failing experiment into a larger redesign.
- Do not change Stage B schema or bridge semantics without explicitly updating `STAGEB_INTERFACE_CONTRACT.md`.
- Do not reinterpret environment failures as algorithmic failures.

## 3. Command discipline
### Local checks
Local checks are informative and should be run when practical.

### Canonical checks
Canonical PASS depends on remote validation described in `ENVIRONMENT_AND_VALIDATION.md`.

### Required reporting after each implementation step
Record:
- files changed
- commands run
- local results
- remote results, if used
- intended local commit hash
- remote checked-out HEAD hash
- whether they match
- whether the result is canonical, informational, blocked, or inconclusive

## 4. Canonical remote validation procedure
Use this procedure whenever a gate requires canonical evidence.

### 4.1 Preflight
Before remote validation that depends on live links or artifacts:
1. confirm `tools/remote_verify_wsovvis.sh` exists
2. confirm canonical runner repo dir is `/home/zyy/code/wsovvis_runner`
3. run canonical runner bootstrap preflight when required by `ENVIRONMENT_AND_VALIDATION.md`
4. confirm the intended branch/commit is ready to be pushed

### 4.2 Environment activation
The canonical remote environment should use the project-approved environment recipe, including:
- `source ~/software/miniconda3/etc/profile.d/conda.sh`
- `conda activate wsovvis`
- minimum `PYTHONPATH` for `third_party/VNext`

### 4.3 Remote validation result counting
A remote result counts toward PASS only if:
- the intended local commit was pushed or otherwise made reachable
- remote validation used the canonical runner
- remote `HEAD == intended local commit`
- the exact command and outcome were recorded

If any of the above are missing, the run is informational only or `BLOCKED` / `INCONCLUSIVE`.

## 5. Git discipline
- Use one task-specific branch per task.
- Commit only scoped changes.
- Push before canonical remote validation.
- Never count a remote run as PASS if remote HEAD does not equal the intended local commit.
- If push routing is blocked, classify the gate as `BLOCKED` rather than manufacturing a non-canonical substitute.

## 6. Output discipline
Task artifacts belong under `codex/<task_dir>/`.
Mainline gate and acceptance artifacts belong under `docs/mainline/reports/`.
Do not write loose `*_output.txt` files in the repo root.

### Mandatory report outputs
- every phase/gate check must write `docs/mainline/reports/phase_gate_latest.txt`
- every acceptance evaluation must write `docs/mainline/reports/acceptance_latest.txt`
- both should also write timestamped archival copies under `docs/mainline/reports/archive/`
- both must update `docs/mainline/STATUS.md` if state materially changed

## 7. When a step fails
Use `FAILURE_PLAYBOOK.md`.
Do not add new modules before trying the documented fallback for the current gate.

### Classification reminder
- infra, routing, wrapper, bootstrap, or remote mismatch problems => `BLOCKED` / `INCONCLUSIVE`
- verified contract/code violation under canonical evaluation => `FAIL`

## 8. After each step
Update `STATUS.md` with:
- active gate
- latest evidence
- next smallest step
- current blockers
- canonical runner evidence status
- remote validation evidence status

## 9. Terminal accepted gate rule
When the accepted terminal gate is active and already `PASS`:
- stop automatically
- do not propose or execute a new coding step
- write/update `docs/mainline/reports/mainline_terminal_summary.txt`
- treat bounded terminal revalidation as the only allowed follow-up mode under the current mainline

### Terminal revalidation scope
Bounded terminal revalidation may:
- rerun the accepted G4 / G5 / G6 regression suite
- use canonical `gpu4090d` validation semantics
- refresh terminal reports if evidence changes

Bounded terminal revalidation may not:
- activate a new gate
- reopen algorithm design
- widen into default-off branches
