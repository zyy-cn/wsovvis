# Stage D Quick Pipeline: CI-Ready Invocation + Gate Policy (N14)

## Purpose
Provide a formal-CI-ready, deterministic policy for choosing Stage D quick gates without changing Stage D semantics.

This document is tooling/docs/policy only and reuses existing wrappers:
- `tools/run_stage_d9_helper_tests_quick.sh`
- `tools/run_stage_d13_ci_quick_pipeline.sh`
- `tools/run_stage_d11_canonical_replay.sh`

## Preconditions
- Run from repo root.
- Use conda-first environment on canonical runner paths when validating remotely:
  - `source ~/software/miniconda3/etc/profile.d/conda.sh`
  - `conda activate wsovvis`
  - `python -m pytest --version`
- Preserve `PYTHONPATH` safely under `set -u`:
  - `export PYTHONPATH="$PWD/third_party/VNext:$PWD/third_party/CutLER:$PWD:${PYTHONPATH:-}"`

## Gate Policy (Stage D Quick Pipeline)
Use the cheapest gate that matches risk:

1. `helper-only fast gate` (default cheapest)
- Command: `tools/run_stage_d9_helper_tests_quick.sh`
- Use when:
  - docs/tooling-only changes
  - wrapper help text / argument plumbing edits
  - no Stage D execution-path edits

2. `N13 quick pipeline` (helper + replay)
- Command: `bash tools/run_stage_d13_ci_quick_pipeline.sh`
- Use when:
  - changing Stage D quick pipeline wrappers/runbooks
  - touching helper/replay command composition
  - preparing CI wiring confidence for branch-local checks

3. `broader checks escalation` (more expensive)
- Suggested commands:
  - `bash tools/run_stage_d10_layered_fast_gate.sh --with-pilot-smoke --pilot-on-mode pilot --pilot-on-weight 0.25 --pilot-scale 1e-6`
  - `tools/run_stage_d10_quick_checks.sh --on-mode nonzero --on-weight 0.25`
  - canonical remote replay using `tools/remote_verify_wsovvis.sh` + hash match checks
- Use when:
  - helper or replay behavior changes beyond argument wiring
  - failures/flakiness appear in quick pipeline
  - release-candidate or merge-critical confidence is required

## CI-Ready Command Snippets
Minimal shell job step:

```bash
set -euo pipefail
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate wsovvis
python -m pytest --version
export PYTHONPATH="$PWD/third_party/VNext:$PWD/third_party/CutLER:$PWD:${PYTHONPATH:-}"
bash tools/run_stage_d13_ci_quick_pipeline.sh
```

Optional split-gate form:

```bash
set -euo pipefail
tools/run_stage_d9_helper_tests_quick.sh
bash tools/run_stage_d13_ci_quick_pipeline.sh
```

## N15 Platform CI Wiring Example
Repository-local CI-ready example (GitHub Actions format):
- `docs/runbooks/tools/ci_examples/stage_d_quick_pipeline.github_actions.yml`

Usage:
1. Copy template to `.github/workflows/stage_d_quick_pipeline.yml` in CI-enabled mirrors.
2. Keep the gate command unchanged:
   - `bash tools/run_stage_d13_ci_quick_pipeline.sh`
3. Preserve deterministic prerequisites in the job:
   - conda-first `wsovvis`
   - `python -m pytest --version`
   - `PYTHONPATH` export with `${PYTHONPATH:-}`

## Output Artifact Path Discipline (copy/paste)
Use this exact clause in prompts/runbooks:

```text
Write the full report to codex/<task_dir>/xx_output.txt (sibling to the prompt file). Writing xx_output.txt at repo root is not allowed.
```

## Canonical Remote Verification Reminder
If remote verification is claimed for quick pipeline gates, include all of:
- conda-first `wsovvis`
- `python -m pytest --version` preflight
- dual `${PYTHONPATH:-}` handling in env and command contexts
- canonical runner path `/home/zyy/code/wsovvis_runner`
- local intended commit hash equals remote `HEAD`
