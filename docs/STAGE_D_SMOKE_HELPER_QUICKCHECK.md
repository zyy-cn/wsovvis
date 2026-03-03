# Stage D Smoke Helper Quick-Check Runbook (D10/D11/D12/N4)

## Purpose
`tools/run_stage_d9_smoke_helper.py` is the Stage D10 helper that wraps the real `train_seqformer_pseudo.py` entrypoint for OFF/ON smoke verification.

Stage D11 reinforced this helper with:
- parser/assertion extraction for GPU-free tests
- `--dry-run` / `--print-commands-only` to print resolved OFF/ON commands without training

Stage D12 adds a reproducible quick-check command path:
- `tools/run_stage_d10_quick_checks.sh`

N4 extends this same path with first-class ON-mode selection:
- zero-mode quick check (compatibility/regression sentinel)
- nonzero-mode quick check (semantic validation of nonzero additive-loss path)

## Canonical quick-check commands
From repo root:

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

Nonzero-mode (semantic validation path):

```bash
tools/run_stage_d10_quick_checks.sh --on-mode nonzero --on-weight 0.25
```

Equivalent explicit commands:

```bash
python tools/run_stage_d9_smoke_helper.py --help
python tools/run_stage_d9_smoke_helper.py --repo-root "$PWD" --dry-run --on-mode nonzero --on-weight 0.25
python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
```

## When to run which mode
- Run zero-mode when you need a compatibility/regression sentinel that preserves OFF/ON no-op expectations (`loss_stage_d_attr` remains effectively zero in ON path).
- Run nonzero-mode when you need semantic validation that ON path is wired for nonzero additive-loss behavior (`nonzero_semantics.enabled=True` and `weight>0` present in command wiring).

## OFF-path vs ON-path smoke expectations
When running real OFF/ON smoke (non-dry-run), expectations depend on ON mode:
- OFF path (`stage_d_attribution.enabled=False`): no `loss_stage_d_attr` metric key
- ON zero-mode (`--on-mode zero`, `weight=0.0`): `loss_stage_d_attr` appears and remains ~0, and OFF/ON total-loss parity remains within `--parity-tol`
- ON nonzero-mode (`--on-mode nonzero`, `weight>0`): `loss_stage_d_attr` is nonzero and ON total loss increases vs OFF path

These checks are reported as `D10_*` lines by the helper.

## Canonical remote verification discipline (conda-first)
For canonical runner checks (e.g. `gpu4090d:/home/zyy/code/wsovvis_runner`), always do:

```bash
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate wsovvis
python -m pytest --version
export PYTHONPATH="$PWD/third_party/VNext:$PWD/third_party/CutLER:$PWD:${PYTHONPATH:-}"
python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
tools/run_stage_d10_quick_checks.sh
tools/run_stage_d10_quick_checks.sh --on-mode nonzero --on-weight 0.25
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
