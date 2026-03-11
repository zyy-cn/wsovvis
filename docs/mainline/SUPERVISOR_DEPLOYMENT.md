# WS-OVVIS Supervisor Deployment

## Goal
Deploy a conservative supervisor layer on top of the document-driven automation kit.

## Components
- repo skills: `.agents/skills/mainline-*/`
- helper script: `tools/run_mainline_loop.py`
- state machine: `docs/mainline/SUPERVISOR_STATE_MACHINE.md`
- environment authority: `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
- execution runbook: `docs/mainline/IMPLEMENT.md`
- evidence contract: `docs/mainline/EVIDENCE_REQUIREMENTS.md`

## Deployment steps
1. Place all overlay files at the repository root according to their relative paths.
2. Ensure the existing project wrapper `tools/remote_verify_wsovvis.sh` and bootstrap checker `tools/check_canonical_runner_bootstrap_links.py` remain present.
3. Create reports directories if missing.
4. Confirm the repository still contains the authoritative entrypoints named in `CODEBASE_MAP.md`.
5. Treat the first supervisor run as `G0` authority-switch and environment inheritance verification.

## Recommended usage
### Interactive
Run:
```bash
python tools/run_mainline_loop.py --dry-run
```
Then paste the generated prompt into Codex.

### Non-interactive
Run:
```bash
python tools/run_mainline_loop.py
```
Then use the saved prompt from `docs/mainline/reports/supervisor_prompt_latest.txt`.

## Mandatory G0 verification
The automation layer is not considered active until the first bounded loop records:
1. control-plane readout proving `docs/mainline/*` is the active interpretation source
2. canonical host alias `gpu4090d`
3. canonical remote repo dir `/home/zyy/code/wsovvis_runner`
4. wrapper `tools/remote_verify_wsovvis.sh`
5. bootstrap preflight availability from `tools/check_canonical_runner_bootstrap_links.py`
6. remote environment activation recipe and `PYTHONPATH` inheritance
7. remote `HEAD == intended local commit`

If any item above is missing, classify `G0` as `BLOCKED` or `INCONCLUSIVE`; do **not** continue deeper into the mainline as if the environment were already inherited.
