# WSOVVIS

WSOVVIS is an ongoing research/engineering project for weakly-supervised open-vocabulary video instance segmentation (and related protocol/tooling experiments).

## Current project status
- ✅ P1 protocol tooling v2 completed (`uniform` + `long_tail` missing protocols; canonical remote validation passed)
- ✅ P3-Prep completed (Stage B track feature export schema v1 / consumer contract / manifest stub)
- 📘 Canonical Stage B export contract docs: `docs/contracts/stageb_track_feature_export_v1/`
- ✅ P3 core completed (Stage B track feature export v1 producer + validator + focused tests; canonical remote validation passed)
- ✅ Stage C0-C4 completed (offline attribution data-plane + scorer expansion closure under additive/default-OFF constraints)
- ✅ Stage D1-D12 completed (unified training-loop wiring, helper hardening, and quick-check/runbook reinforcement)
- 📘 Stage D closure memo: `docs/STAGE_D_CLOSURE_MEMO_D1_D12.md`
- 📘 Stage D quick-check runbook (D10/D11/D12/N4 zero+nonzero modes): `docs/STAGE_D_SMOKE_HELPER_QUICKCHECK.md`
- 📘 Session handoff (post-closure): `docs/SESSION_HANDOFF_STAGE_D_CLOSURE.md`
- ▶️ Next recommended direction: **research-facing nonzero supervision semantics** under new gated milestone planning
- Progress log: `docs/PROJECT_PROGRESS.md`
- Long-term Codex workflow reference (including Git/remote validation + `github-via-gpu` push routing): `codex/WSOVVIS_CODEX_WORKFLOW_README.md`
- Reusable prompt/policy specs for concise future prompts: `codex/specs/`
- Cold-start Codex workflow entry (new session startup order / precedence rules): `codex/START_HERE.md`

## Notes
This README intentionally keeps the status summary brief. Detailed milestone history and workflow operating procedures are maintained in the linked documents above.
