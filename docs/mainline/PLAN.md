# WSOVVIS Mainline Plan

## 1. Purpose
This file is the source of truth for the current WSOVVIS automation mainline.

Main claim to prove:

> In incomplete positive-label supervision, open-world set-to-track attribution reduces hidden-positive-to-background collapse better than closed-world weak supervision.

The automation system must prioritize proving this claim over expanding the system.

## 2. Mainline phases and gates

### G0 — Repository bootstrap and environment verification
Goal:
- verify the codebase can be interpreted under this new control plane
- verify canonical environment facts and remote validation wrapper

Exit criteria:
- `ENVIRONMENT_AND_VALIDATION.md` matches the real environment
- `CODEBASE_MAP.md` matches real code entrypoints
- Stage B interface files referenced in `STAGEB_INTERFACE_CONTRACT.md` exist

### G1 — Protocol and baseline alignment
Goal:
- ensure the WS-OVVIS protocol build path is working
- ensure at least one closed-world baseline path is runnable

Required focus:
- `tools/build_wsovvis_labelset_protocol.py`
- `tests/test_build_wsovvis_labelset_protocol.py`
- baseline attribution entrypoint(s) in `tools/` / `wsovvis/track_feature_export/`

Exit criteria:
- protocol generation checks pass
- baseline path is identifiable and smoke-runnable

### G2 — Stage B export/bridge/consumer integrity
Goal:
- preserve the current Stage B data plane as a stable basis for Stage C

Exit criteria:
- Stage B schema, bridge, and consumer requirements are satisfied
- related tests pass locally or canonically as appropriate

### G3 — Mixed representation and Stage C semantic plumbing
Goal:
- preserve or restore the representation bridge required for open-vocabulary attribution

Exit criteria:
- Stage C loader / semantic slice / prototype cache path is coherent
- no regression in required representation interfaces

### G4 — Open-world attribution validation (core gate)
Goal:
- establish a working mainline for open-world attribution
- prioritize hidden-positive handling evidence over system breadth

Required evidence hierarchy:
1. closed-world baseline
2. open-world without coverage enhancement
3. open-world with coverage-aware mainline
4. retrieved-candidate enhancement only if needed

Exit criteria:
- acceptance in `METRICS_ACCEPTANCE.md` for the active attribution gate is met

### G5 — Full-video linking and inference closure
Goal:
- move from window-level semantics to full-video output without destabilizing the attribution mainline

Default policy:
- linking uses geometry + query as primary signals
- semantics is weak auxiliary only
- global classification defaults to quality-weighted logit averaging

Exit criteria:
- linking/inference gate acceptance is met

### G6 — Single-round bounded refinement
Goal:
- keep only a bounded, low-risk refinement step

Default policy:
- single round only
- mask/temporal quality dominates
- semantic score is secondary
- label-set expansion remains default-off

Exit criteria:
- refinement yields non-negative value under acceptance contract

Terminal rule:
- the current documented mainline ends at `G6`
- when `G6` has an evidence-backed `PASS`, the supervisor must enter terminal-mainline mode
- no further gate activates unless the authoritative docs themselves are updated

## 3. Default-off research branches
These are not part of the mainline unless `STATUS.md` explicitly activates them:
- label-set expansion
- second-round refinement
- scenario/domain-missing as mainline protocol
- stronger memory aggregation for global classification
- decoder-comparison branches as mainline
- Stage D multi-round guarded continuation as mainline

## 4. Exploration policy
Exploration is allowed only when:
- the current gate explicitly requires it, and
- `STATUS.md` records the uncertainty, the smallest proposed experiment, and the fallback path.

If a result is inconclusive, prefer narrowing the current design rather than opening another branch.
