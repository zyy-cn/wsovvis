# Stage D Smoke Helper Quick-Check Runbook (D10/D11/D12/N4/N7/N13/N15)

## Purpose
`tools/run_stage_d9_smoke_helper.py` is the Stage D10 helper that wraps the real `train_seqformer_pseudo.py` entrypoint for OFF/ON smoke verification.

Stage D11 reinforced this helper with:
- parser/assertion extraction for GPU-free tests
- `--dry-run` / `--print-commands-only` to print resolved OFF/ON commands without training

Stage D12 adds a reproducible quick-check command path:
- `tools/run_stage_d10_quick_checks.sh`

N9 adds a local/CI mirror entry for helper coverage:
- `tools/run_stage_d9_helper_tests_quick.sh`

N10 adds a layered fast-gate entry that composes helper coverage first, then optional pilot-capable quick-check smoke:
- `tools/run_stage_d10_layered_fast_gate.sh`

N12 adds a replay wrapper for the N11 canonical sequence:
- `tools/run_stage_d11_canonical_replay.sh`

N13 adds a branch-local CI mirror pipeline:
- `tools/run_stage_d13_ci_quick_pipeline.sh`

N14 adds formal-CI-ready policy guidance for gate selection:
- `docs/STAGE_D_CI_QUICK_PIPELINE_GATE_POLICY.md`

N15 adds a platform CI wiring example for quick-pipeline adoption:
- `docs/runbooks/tools/ci_examples/stage_d_quick_pipeline.github_actions.yml`

N4+N7 extend this same path with first-class ON-mode selection:
- zero-mode quick check (compatibility/regression sentinel)
- nonzero-mode quick check (semantic validation of constant nonzero additive-loss path)
- pilot-mode quick check (wiring validation for `gradient_coupled_pilot_v1`)

## Canonical quick-check commands
From repo root:

Helper coverage fast path (local/CI, pytest-capable, recommended first gate):

```bash
tools/run_stage_d9_helper_tests_quick.sh
```

Layered fast gate (runs helper coverage first, and optionally pilot-capable quick-check smoke):

```bash
# default: helper coverage only (fastest)
bash tools/run_stage_d10_layered_fast_gate.sh

# include pilot quick-check smoke after helper coverage
bash tools/run_stage_d10_layered_fast_gate.sh --with-pilot-smoke --pilot-on-mode pilot --pilot-on-weight 0.25 --pilot-scale 1e-6
```

Canonical N11 replay sequence (N12 bundled entry):

```bash
bash tools/run_stage_d11_canonical_replay.sh
```

Branch-local CI mirror for quick pipeline wiring (N13):

```bash
bash tools/run_stage_d13_ci_quick_pipeline.sh
```

N14 gate policy reference (formal-CI-ready decision criteria):

```bash
cat docs/STAGE_D_CI_QUICK_PIPELINE_GATE_POLICY.md
```

N15 platform CI wiring template (copy into CI-enabled mirror as workflow file):

```bash
cat docs/runbooks/tools/ci_examples/stage_d_quick_pipeline.github_actions.yml
```

Zero-mode (compatibility sentinel, default):

```bash
tools/run_stage_d10_quick_checks.sh
```

Equivalent explicit commands:

```bash
python tools/run_stage_d9_smoke_helper.py --help
python tools/run_stage_d9_smoke_helper.py --repo-root "$PWD" --dry-run --on-mode zero
python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
```

Nonzero-mode (constant nonzero semantic validation path):

```bash
tools/run_stage_d10_quick_checks.sh --on-mode nonzero --on-weight 0.25
```

Equivalent explicit commands:

```bash
python tools/run_stage_d9_smoke_helper.py --help
python tools/run_stage_d9_smoke_helper.py --repo-root "$PWD" --dry-run --on-mode nonzero --on-weight 0.25
python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
```

Pilot-mode (gradient-coupled pilot wiring smoke):

```bash
tools/run_stage_d10_quick_checks.sh --on-mode pilot --on-weight 0.25 --pilot-scale 1e-6
```

Equivalent explicit commands:

```bash
python tools/run_stage_d9_smoke_helper.py --help
python tools/run_stage_d9_smoke_helper.py --repo-root "$PWD" --dry-run --on-mode pilot --on-weight 0.25 --pilot-scale 1e-6
python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
```

## Prompt-template output clause (copy/paste)
For task prompts and runbooks that request `xx_output.txt`, include this exact rule:

```text
Write the full report to codex/<task_dir>/xx_output.txt (sibling to the prompt file). Writing xx_output.txt at repo root is not allowed.
```

## When to run which mode
- Run zero-mode when you need a compatibility/regression sentinel that preserves OFF/ON no-op expectations (`loss_stage_d_attr` remains effectively zero in ON path).
- Run nonzero-mode when you need semantic validation that ON path is wired for constant nonzero additive-loss behavior (`nonzero_semantics.enabled=True` and `weight>0` present in command wiring).
- Run pilot-mode when you need a lightweight quick-check smoke that ON command wiring explicitly requests `nonzero_semantics.mode=gradient_coupled_pilot_v1` (plus optional `gradient_coupled_scale`).

## OFF-path vs ON-path smoke expectations
When running real OFF/ON smoke (non-dry-run), expectations depend on ON mode:
- OFF path (`stage_d_attribution.enabled=False`): no `loss_stage_d_attr` metric key
- ON zero-mode (`--on-mode zero`, `weight=0.0`): `loss_stage_d_attr` appears and remains ~0, and OFF/ON total-loss parity remains within `--parity-tol`
- ON nonzero-mode (`--on-mode nonzero`, `weight>0`): `loss_stage_d_attr` is nonzero and ON total loss increases vs OFF path
- ON pilot-mode (`--on-mode pilot`, `weight>0`): helper requests `gradient_coupled_pilot_v1` and now asserts ON-run `cfg_runtime.json` pilot diagnostics (`nonzero_semantics_mode`, pilot applied/skip state, skip reasons, and scale/weight consistency)

These checks are reported as `D10_*` lines by the helper.

## Canonical remote verification discipline (conda-first)
For canonical runner checks (e.g. `gpu4090d:/home/zyy/code/wsovvis_runner`), always do:

```bash
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate wsovvis
python -m pytest --version
export PYTHONPATH="$PWD/third_party/VNext:$PWD/third_party/CutLER:$PWD:${PYTHONPATH:-}"
tools/run_stage_d9_helper_tests_quick.sh
bash tools/run_stage_d10_layered_fast_gate.sh
bash tools/run_stage_d10_layered_fast_gate.sh --with-pilot-smoke --pilot-on-mode pilot --pilot-on-weight 0.25 --pilot-scale 1e-6
bash tools/run_stage_d11_canonical_replay.sh
bash tools/run_stage_d13_ci_quick_pipeline.sh
python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
tools/run_stage_d10_quick_checks.sh
tools/run_stage_d10_quick_checks.sh --on-mode nonzero --on-weight 0.25
tools/run_stage_d10_quick_checks.sh --on-mode pilot --on-weight 0.25 --pilot-scale 1e-6
```

Notes:
- Use `${PYTHONPATH:-}` (not `$PYTHONPATH`) to avoid shell failures under `set -u`.
- Keep quick checks targeted and GPU-free unless a task explicitly requires full D9/D10 smoke.

## Common failure modes and quick fixes
1. Path resolution failures (`FileNotFoundError` for weights/data)
- Cause: running non-dry-run helper in an environment without expected artifacts.
- Fix: use `--dry-run` for wiring checks, or pass explicit `--weights-path` / `--train-json` / image roots.

2. Remote branch or commit mismatch
- Cause: remote runner validated stale `origin/<branch>` commit.
- Fix: commit + push first; assert remote `git rev-parse HEAD` equals intended local commit.

3. Quoting pitfalls in remote commands
- Cause: nested quoting around overrides and `PYTHONPATH`.
- Fix: prefer `tools/remote_verify_wsovvis.sh` and keep commands simple; use `${PYTHONPATH:-}` in both env and command contexts.

4. Task output written to repo root (`xx_output.txt`)
- Cause: using relative output names without task-directory prefix.
- Fix: always write outputs to `codex/<task_dir>/xx_output.txt` in the same folder as the prompt.
- Example (current task style): `codex/2026030304_staged_nonzero_semantics/13_output.txt`
