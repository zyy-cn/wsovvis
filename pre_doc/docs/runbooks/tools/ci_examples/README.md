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

## Replay (`workflow_dispatch`) smoke
Use one of the following in a CI-enabled mirror after the workflow file is present:

- Operator note:
  - export `GITHUB_TOKEN` in the same shell/environment that runs Codex and `gh`/`curl` dispatch commands; otherwise token inheritance failures can appear as false auth blockers.
- CI-hosted behavior note:
  - `tools/run_stage_d13_ci_quick_pipeline.sh` may skip canonical replay gracefully when checkpoint assets are absent on GitHub-hosted runners; this is expected compatibility behavior, not a Stage D semantic failure.

- GitHub CLI:
  - `gh workflow run stage_d_quick_pipeline.yml --ref staged-nonzero-semantics`
- GitHub API:
  - `curl -X POST -H "Accept: application/vnd.github+json" -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com/repos/zyy-cn/wsovvis/actions/workflows/stage_d_quick_pipeline.yml/dispatches -d '{"ref":"staged-nonzero-semantics"}'`
