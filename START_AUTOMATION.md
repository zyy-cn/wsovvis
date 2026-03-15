# Start WS-OVVIS Automation

Deploy this overlay at the repository root of the WS-OVVIS repository.

Required paths after deployment:
- `AGENTS.md`
- `.codex/config.toml`
- `docs/mainline/*`, including `EVIDENCE_REQUIREMENTS.md`
- `docs/runbooks/mainline_phase_gate_runbook.md`
- `.agents/skills/mainline-phase-gate-check/SKILL.md`
- `.agents/skills/mainline-eval-acceptance/SKILL.md`
- `.agents/skills/mainline-supervisor/SKILL.md`
- `tools/run_mainline_loop.py`
- existing project wrapper: `tools/remote_verify_wsovvis.sh`
- existing bootstrap checker: `tools/check_canonical_runner_bootstrap_links.py`

## First run
From the repo root:
```bash
codex
```
Then paste `prompts/02_project_specific_first_run_prompt.md`.

## Important
Do not assume the environment inheritance is correct until `G0` records:
- canonical runner facts,
- wrapper availability,
- bootstrap preflight evidence,
- intended-commit / remote-HEAD tracking policy,
- the required G0 evidence bundle and worked example defined in `docs/mainline/EVIDENCE_REQUIREMENTS.md`.
