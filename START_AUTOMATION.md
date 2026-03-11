# Start WS-OVVIS Automation

Deploy this overlay at the **repository root** of a clean `wsovvis` checkout.

Required paths after deployment:
- `AGENTS.md`
- `.codex/config.toml`
- `docs/mainline/*`, including `EVIDENCE_REQUIREMENTS.md` and `STAGEB_INTERFACE_CONTRACT.md`
- `docs/runbooks/mainline_phase_gate_runbook.md`
- `.agents/skills/mainline-phase-gate-check/SKILL.md`
- `.agents/skills/mainline-eval-acceptance/SKILL.md`
- optional but recommended: `.agents/skills/mainline-supervisor/SKILL.md`
- `tools/run_mainline_loop.py`
- existing project wrapper: `tools/remote_verify_wsovvis.sh`
- existing bootstrap preflight checker: `tools/check_canonical_runner_bootstrap_links.py`

## First run
From the repo root:

```bash
codex
```

Then paste the project-specific Prompt 2 from:

- `prompts/02_project_specific_first_run_prompt.md`

Or prepare the same bounded prompt from the repo helper:

```bash
python tools/run_mainline_loop.py --dry-run
```

## Important
Do not assume the environment is inherited correctly until `G0` records:
- canonical runner facts
- wrapper availability
- bootstrap preflight evidence
- remote HEAD consistency evidence
- v9 authority-switch evidence under `docs/mainline/*`
- the required G0 evidence bundle and worked example defined in `docs/mainline/EVIDENCE_REQUIREMENTS.md`
