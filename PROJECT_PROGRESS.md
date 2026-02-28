# Project Progress

This file tracks milestone-level project progress (engineering + protocol/tooling), while the root `README.md` remains the primary entry point for setup and usage.

## 2026-02-28 — P1 Protocol v2 completed (WS-OVVIS labelset protocol tooling)

### Summary
- Extended `tools/build_wsovvis_labelset_protocol.py` from **v1 / P1-mini** (`uniform` only) to **P1 v2** with `uniform + long_tail` protocol support.
- Preserved `uniform` behavior and deterministic output expectations.
- Added richer manifest statistics for protocol auditing and later experiment reporting.
- Expanded unit tests for long-tail behavior and determinism.

### Implemented items
- `--protocol long_tail` support (frequency-aware label dropping; deterministic with fixed seed)
- `protocol_metadata` in manifest (drop rule, frequency-count basis, long-tail formula metadata)
- `class_stats` in manifest (annotation/full/observed counts, observed rate, drop probability)
- `missing_count_stats` in manifest (min/max/mean missing labels over non-empty clips)
- Long-tail tests (happy path, determinism, statistical tendency, min-label cap guarantee)

### Canonical validation (authoritative)
- **Validation plane:** remote runner on `gpu4090d`
- **Command:** `python -m pytest -q tests/test_build_wsovvis_labelset_protocol.py`
- **Result:** `13 passed`
- **Verified remote target commit:** `7c0c277` (after branch sync/push)

### Notes / workflow lessons (useful for future tasks)
- In this repository workflow, local WSL checks are useful but **non-canonical**.
- Final PASS should be counted only after remote canonical validation on `gpu4090d` (and after confirming the remote runner validated the intended commit).

### Next planned step
- **P3-Prep (Tier 1, doc-only):** define Stage B track feature export schema for Stage C consumption (stable Stage B→Stage C input contract).

---

## Status snapshot
- **Current phase:** P1 completed; preparing P3-Prep
- **Near-term focus:** Stage B track feature export schema (doc-only) → Stage B export implementation
