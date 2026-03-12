# Start WS-OVVIS Automation

Deploy this overlay at the **repository root** of a clean `wsovvis` checkout.

Required paths after deployment:
- `AGENTS.md`
- `.codex/config.toml`
- `docs/mainline/*`
- `docs/scientific/*`
- `docs/runbooks/mainline_phase_gate_runbook.md`
- `.agents/skills/mainline-phase-gate-check/SKILL.md`
- `.agents/skills/mainline-eval-acceptance/SKILL.md`
- optional but recommended: `.agents/skills/mainline-supervisor/SKILL.md`
- `tools/run_mainline_loop.py`
- existing project wrapper: `tools/remote_verify_wsovvis.sh`
- existing bootstrap preflight checker: `tools/check_canonical_runner_bootstrap_links.py`

## Scientific overlay v2 activation
When the engineering control plane is already deployed and a scientific gate or scientific migration task is active, Codex must also read:
- `docs/scientific/INDEX.md`
- `docs/scientific/V10_RECONCILIATION_MEMO.md`
- `docs/scientific/P0_EXPERIMENTAL_CHARTER.md`
- `docs/scientific/STATUS.md`
- the active scientific gate spec

Scientific gates do not replace engineering gates. They sit on top of them.
Engineering PASS and Scientific PASS must be tracked separately.
Formal scientific progression and diagnostic-only probing must also be tracked separately.

## Important
Do not assume a scientific claim is proven merely because the engineering path is runnable.
Scientific PASS requires the comparator, metric, evidence, and sign-off rules frozen by `docs/scientific/P0_EXPERIMENTAL_CHARTER.md` and the active scientific gate spec.

## Migration note
The prior strict `S1` / `S1R` path is superseded by the refined v10 scientific overlay. Do not continue `S1R` unless an explicit legacy replay is requested. Use the new scientific overlay activation prompt instead.
