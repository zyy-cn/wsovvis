# Stage-D CI Wiring Examples

This folder contains CI-ready templates for Stage-D quick pipeline gate wiring.

## Template
- `stage_d_quick_pipeline.github_actions.yml`

## Usage
1. Copy the template into an actual CI location in a CI-enabled mirror repo:
   - `.github/workflows/stage_d_quick_pipeline.yml`
2. Keep the gate command unchanged:
   - `bash tools/run_stage_d13_ci_quick_pipeline.sh`
3. Ensure the workflow environment provides:
   - `wsovvis` conda environment activation
   - `python -m pytest --version` preflight
   - `PYTHONPATH` export with `${PYTHONPATH:-}`

This repo keeps the file under `docs/` as a lightweight CI-ready mirror artifact when live CI config is not present.
