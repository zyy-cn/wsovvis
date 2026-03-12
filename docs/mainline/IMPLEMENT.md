# WS-OVVIS Mainline Implement Runbook

This file defines how Codex should work inside the repository at the engineering layer.

## Scientific overlay interaction
When a scientific gate is active:
- do not treat engineering PASS as scientific PASS
- read `docs/scientific/INDEX.md`, `docs/scientific/V10_RECONCILIATION_MEMO.md`, `docs/scientific/STATUS.md`, and the active scientific gate spec before doing scientific work
- distinguish formal scientific progression, diagnostic-only probing, and recovery-mode execution
- keep engineering reports under `docs/mainline/reports/`
- keep scientific reports under `docs/scientific/reports/`
- do not merge engineering and scientific judgments into a single report

All other engineering-layer rules remain governed by the existing full `docs/mainline/IMPLEMENT.md` already present in the repository.
