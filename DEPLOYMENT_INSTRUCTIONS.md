# WS-OVVIS Deployment Instructions

## What this overlay does
This overlay privatizes the generic workflow kit for WS-OVVIS and re-establishes the v9 mainline under `docs/mainline/*` while preserving:
- document-driven control
- skills-based gate checking and acceptance
- supervisor-driven bounded loops
- canonical validation semantics
- terminal-mode stop and bounded terminal revalidation

## Deployment steps
1. Unzip this overlay at the repository root of a clean `wsovvis` checkout.
2. Keep the existing project files in place, especially:
   - `tools/remote_verify_wsovvis.sh`
   - `tools/check_canonical_runner_bootstrap_links.py`
   - all project code and tests already in the repo
3. Restore or add the overlay files exactly at their relative paths.
4. Confirm these paths now exist:
   - `AGENTS.md`
   - `START_AUTOMATION.md`
   - `.codex/config.toml`
   - `.agents/skills/mainline-phase-gate-check/SKILL.md`
   - `.agents/skills/mainline-eval-acceptance/SKILL.md`
   - `.agents/skills/mainline-supervisor/SKILL.md`
   - `docs/mainline/*`
   - `docs/runbooks/mainline_phase_gate_runbook.md`
   - `prompts/02_project_specific_first_run_prompt.md`
5. Replace or update `tools/run_mainline_loop.py` with the overlay version if you want the helper to reflect terminal-at-`G7` semantics.
6. Start Codex from the repo root with `codex`.
7. Paste `prompts/02_project_specific_first_run_prompt.md`.

## Recommended first validation
After deployment, but before any algorithmic work:
```bash
python tools/run_mainline_loop.py --dry-run
```
Then verify that the generated prompt still targets `G0`.

## Notes
- This overlay intentionally treats existing Stage D continuation assets as legacy/default-off.
- The first bounded loop is for authority switch and environment inheritance, not model development.
- `G7` is the terminal gate for this privatized control plane.
