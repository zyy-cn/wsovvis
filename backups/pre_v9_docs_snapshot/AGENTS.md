# WSOVVIS Automation AGENTS

This repository runs in **document-driven automation mode**.

Read in this exact order before planning or coding:
1. `docs/mainline/INDEX.md`
2. `docs/mainline/PLAN.md`
3. `docs/mainline/IMPLEMENT.md`
4. `docs/mainline/STATUS.md`
5. `docs/mainline/METRICS_ACCEPTANCE.md`
6. `docs/mainline/FAILURE_PLAYBOOK.md`
7. `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
8. `docs/mainline/CODEBASE_MAP.md`
9. `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`

## Mission
The current mainline exists to prove the paper's core claim:

> Under incomplete positive label sets, open-world set-to-track attribution should preserve hidden positives better than closed-world weak supervision.

Any code change that does not directly strengthen that claim should **not** enter the mainline by default.

## Authority and precedence
Priority order:
1. This `AGENTS.md`
2. `docs/mainline/*`
3. repo-specific skills under `.agents/skills/*`
4. code and tests
5. user prompt for the current session, as long as it does not conflict with 1-4

Historical task packs, old workflow docs, handoff notes, milestone memos, and historical prompts are **not authoritative** in this workflow.

## Default-on mainline scope
Default-on:
- WS-OVVIS protocol generation (`uniform`, `long_tail`)
- Stage B export/bridge/consumer compatibility
- mixed representation required for Stage C semantics
- open-world attribution mainline
- cross-window linking baseline
- single-round bounded refinement only

## Default-off modules
Do **not** enable these unless the current phase/gate explicitly allows them:
- label-set expansion
- second-round refinement
- scenario/domain-missing as mainline protocol
- enhanced memory aggregation for global classification
- decoder-comparison research branches as default mainline
- Stage D multi-round guarded continuation as default mainline

## Operating style
- Make the **smallest valid step** toward the current active gate.
- Do not widen scope to make a failing result look better.
- Prefer the documented fallback path over adding a new mechanism.
- If docs and code disagree, report the conflict before patching.

## Validation model
- Local checks are informative.
- Canonical PASS is determined by the remote validation model in `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`.
- A remote PASS only counts if remote `HEAD == intended local commit`.
- Environment mismatch, wrapper failure, push-route failure, or missing canonical bootstrap evidence must be classified as `BLOCKED` or `INCONCLUSIVE`, not as algorithmic failure.

## Git discipline
- Use task-specific branches.
- Commit only scoped changes.
- Push before canonical remote validation.
- Preserve task outputs under `codex/<task_dir>/...`; do not write loose `*_output.txt` files at repo root.
- If direct push is unreliable, use the project-approved route such as `github-via-gpu`.

## Reporting discipline
- Every phase/gate check must write `docs/mainline/reports/phase_gate_latest.txt` plus a timestamped archive copy.
- Every acceptance evaluation must write `docs/mainline/reports/acceptance_latest.txt` plus a timestamped archive copy.
- `docs/mainline/STATUS.md` must be updated whenever the active gate, blockers, or canonical evidence materially change.

## When blocked
If current acceptance cannot be reached:
1. identify the blocking acceptance
2. follow `docs/mainline/FAILURE_PLAYBOOK.md`
3. update `docs/mainline/STATUS.md`
4. do not open default-off modules unless the docs explicitly authorize it
