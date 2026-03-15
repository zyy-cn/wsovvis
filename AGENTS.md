# WS-OVVIS Automation AGENTS

This repository runs in **document-driven automation mode**.

Read in this exact order before planning or coding:
1. `docs/mainline/INDEX.md`
2. `docs/mainline/PLAN.md`
3. `docs/mainline/IMPLEMENT.md`
4. `docs/mainline/STATUS.md`
5. `docs/mainline/METRICS_ACCEPTANCE.md`
6. `docs/mainline/EVIDENCE_REQUIREMENTS.md`
7. `docs/mainline/FAILURE_PLAYBOOK.md`
8. `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
9. `docs/mainline/CODEBASE_MAP.md`
10. `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`
11. `docs/runbooks/mainline_phase_gate_runbook.md`

## Mission
The current mainline exists to prove the project's core claim:

> Under clip-level incomplete positive evidence Y'(v), prove that a frozen video-native class-agnostic basis, materialized query-trajectory and semantic-carrier banks, seen visual prototypes, a class-level text map, and coverage-aware open-world attribution are sufficient for bag-free text-conditioned open-vocabulary video instance inference without using the observed label bag at test time.

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
- clip-level weak-label protocol generation for incomplete positive evidence `Y'(v)`
- frozen DINOv2 + lightweight adapter + VideoMask2Former as the video-native class-agnostic basis path
- direct query-trajectory export as the atomic basis interface
- persistent Query-Trajectory Bank and Semantic Carrier Bank
- seen-class prototype initialization and class-level text map training
- coverage-aware open-world attribution on `Y'(v) + bg + unk`
- bag-free full-vocabulary inference and ws-metrics reporting

## Default-off modules
Do **not** enable these unless the current phase/gate explicitly allows them:
- cross-query stitching, tracklet refinement, duplicate suppression, or cardinality control as default mainline logic
- threshold sweeps derived from the retired SeqFormer route
- prototype EM / momentum refresh beyond bounded gate-specific evidence collection
- unrestricted prompt ensemble, synonym expansion, or free-form text grounding branches
- multi-round continuation or self-training beyond a documented gate-approved bounded replay
- any new semantic defense mechanism opened before the active gate evidence says it is necessary

## Operating style
- Make the **smallest valid step** toward the current active gate.
- Do not widen scope to make a failing result look better.
- Prefer the documented fallback path over adding a new mechanism.
- If docs and code disagree, report the conflict before patching.

## Validation model
- Local checks are informative.
- Canonical PASS is determined by the remote validation model in `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`.
- A remote PASS only counts if remote `HEAD == intended local commit`.
- A gate PASS only counts if the acceptance contract and the required evidence pack are both complete.
- Environment mismatch, wrapper failure, push-route failure, or missing canonical bootstrap evidence must be classified as `BLOCKED` or `INCONCLUSIVE`, not as algorithmic failure.

## Reporting discipline
- Every phase/gate check must write `docs/mainline/reports/phase_gate_latest.txt` plus a timestamped archive copy.
- Every acceptance evaluation must write `docs/mainline/reports/acceptance_latest.txt` plus a timestamped archive copy.
- Every evidence review must write `docs/mainline/reports/evidence_latest.txt` plus a timestamped archive copy.
- Every gate requiring a worked example must also write `docs/mainline/reports/worked_example_verification_latest.md` and `.json` plus timestamped archive copies.
- `docs/mainline/STATUS.md` must be updated whenever the active gate, blockers, acceptance status, evidence status, or canonical evidence materially change.
