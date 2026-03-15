# WS-OVVIS Deployment Instructions

## What this package provides
This overlay installs a self-consistent project-private control plane for WS-OVVIS while preserving the generic workflow model:
- document-driven control plane
- skills-based gate checking and acceptance
- supervisor-driven bounded loops
- canonical validation semantics
- terminal-mode stop and bounded terminal revalidation
- explicit evidence-backed gate judgments

## Deployment steps
1. Unzip this overlay at the **repository root** of a clean WS-OVVIS code checkout.
2. Keep the existing project code, especially these paths from the code repository:
   - `docs/outline/WS_OVVIS_outline_v13_with_gates.tex`
   - `tools/remote_verify_wsovvis.sh`
   - `tools/check_canonical_runner_bootstrap_links.py`
   - `tools/build_wsovvis_labelset_protocol.py`
   - `tools/run_s1_basis_superiority_eval.py`
   - `tools/build_global_track_bank_v9.py`
   - `tools/extract_track_dino_features_v9.py`
   - `tools/init_prototype_bank_v9.py`
   - `tools/train_text_map_v9.py`
   - `tools/train_openworld_core_v9.py`
   - `tools/run_bagfree_eval_v9.py`
   - `tests/`
   - `wsovvis/`
3. Confirm these overlay paths now exist:
   - `AGENTS.md`
   - `START_AUTOMATION.md`
   - `.codex/config.toml`
   - `.agents/skills/mainline-phase-gate-check/SKILL.md`
   - `.agents/skills/mainline-eval-acceptance/SKILL.md`
   - `.agents/skills/mainline-supervisor/SKILL.md`
   - `docs/mainline/*`
   - `docs/runbooks/mainline_phase_gate_runbook.md`
   - `prompts/02_project_specific_first_run_prompt.md`
   - `tools/run_mainline_loop.py`
4. Start Codex from the repository root.
5. Paste `prompts/02_project_specific_first_run_prompt.md`.

## Recommended first local sanity check
From the repo root:
```bash
python tools/run_mainline_loop.py --dry-run
```
This should write `docs/mainline/reports/supervisor_prompt_latest.txt`.

## Notes
- This adapted package intentionally uses the uploaded v13 outline as the design source, not the older v9 outline referenced by the partial prior overlay.
- The first bounded loop is a G0 authority/environment step, not an algorithm-development step.
- A later gate may activate only after evidence-backed PASS for the current gate.
