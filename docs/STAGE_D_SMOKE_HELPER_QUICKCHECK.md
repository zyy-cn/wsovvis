# Stage D Smoke Helper Quick-Check Runbook (D10/D11/D12)

## Purpose
`tools/run_stage_d9_smoke_helper.py` is the Stage D10 helper that wraps the real `train_seqformer_pseudo.py` entrypoint for OFF/ON smoke verification with strict no-op assertions.

Stage D11 reinforced this helper with:
- parser/assertion extraction for GPU-free tests
- `--dry-run` / `--print-commands-only` to print resolved OFF/ON commands without training

Stage D12 adds a reproducible quick-check command path:
- `tools/run_stage_d10_quick_checks.sh`

## Canonical quick-check commands
From repo root:

```bash
tools/run_stage_d10_quick_checks.sh
```

Equivalent explicit commands:

```bash
python tools/run_stage_d9_smoke_helper.py --help
python tools/run_stage_d9_smoke_helper.py --repo-root "$PWD" --dry-run
python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
```

## OFF-path vs ON-path smoke expectations
When running real OFF/ON smoke (non-dry-run):
- OFF path (`stage_d_attribution.enabled=False`): no `loss_stage_d_attr` metric key
- ON path (`stage_d_attribution.enabled=True` and coupling/loss-key `weight=0.0`): `loss_stage_d_attr` appears and remains ~0
- OFF vs ON `total_loss` remains parity-equal within tolerance (`--parity-tol`, default `1e-9`)

These checks are reported as `D10_*` lines by the helper.

## Canonical remote verification discipline (conda-first)
For canonical runner checks (e.g. `gpu4090d:/home/zyy/code/wsovvis_runner`), always do:

```bash
source ~/software/miniconda3/etc/profile.d/conda.sh
conda activate wsovvis
python -m pytest --version
export PYTHONPATH="$PWD/third_party/VNext:$PWD/third_party/CutLER:$PWD:${PYTHONPATH:-}"
python -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
python tools/run_stage_d9_smoke_helper.py --help
python tools/run_stage_d9_smoke_helper.py --repo-root "$PWD" --dry-run
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
