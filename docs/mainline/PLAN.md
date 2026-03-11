# WS-OVVIS Mainline Plan

## 1. Purpose
This file is the source of truth for the current automation mainline.

Main claim to prove:

> Under clip-level incomplete positive evidence `Y'(v)`, a class-agnostic video instance basis plus clip-level global tracks, DINO-only track semantics, seen visual prototypes, a class-level text map, and core open-world attribution should support bag-free open-vocabulary video instance inference without relying on the observed bag at test time.

The automation system must prioritize proving this claim over expanding the system.
A gate may only receive `PASS` when its acceptance contract and its required evidence pack are both satisfied.

## 2. Mainline phases and gates

### G0 — v9 control-plane bootstrap and inheritance verification
Goal:
- establish `docs/mainline/*` as the repository execution source without changing the interaction model
- verify that the control docs match the real repository and preserve the inherited canonical environment semantics

Exit criteria:
- `ENVIRONMENT_AND_VALIDATION.md` matches the real environment
- `CODEBASE_MAP.md` matches the real repository layout, current reusable anchors, and planned v9 entrypoints
- `STAGEB_INTERFACE_CONTRACT.md` matches the Stage B to v9 artifact transition plan
- the repository can be interpreted unambiguously under this control plane
- the minimum G0 evidence pack exists

### G1 — Clip-level protocol and artifact-contract alignment
Goal:
- fix the clip-level weak-supervision protocol `Y'(v)`
- fix the artifact contracts needed by the v9 mainline

Required focus:
- `tools/build_wsovvis_labelset_protocol.py`
- protocol tests and contract documentation
- v9 artifact contract definitions

Exit criteria:
- clip-level protocol generation checks pass
- v9 artifact contracts are documented and minimally parseable
- window-level `Y'(w)` is not the mainline supervision object
- the minimum G1 evidence pack exists

### G2 — Class-agnostic instance basis closure
Goal:
- keep the structure layer independent from the semantic bridge
- obtain local tracklets from pseudo tubes and class-agnostic SeqFormer training/inference

Exit criteria:
- local tracklet export is identifiable, contract-stable, and smoke-runnable
- structure-side diagnostics exist
- the minimum G2 evidence pack exists

### G3 — Clip-level global track bank closure
Goal:
- move from local tracklets to a fixed clip-level global track bank

Default policy:
- linking uses overlap IoU + local-query consistency as primary signals
- semantics does not dominate track structure

Exit criteria:
- global track bank can be built offline and reloaded deterministically
- stitching diagnostics exist
- the minimum G3 evidence pack exists

### G4 — DINO-only track semantic carrier closure
Goal:
- establish `z_tau` as the only mainline semantic carrier
- compute track objectness `o_tau`

Exit criteria:
- a track semantic cache exists and is stable
- `z_tau` and `o_tau` are available for downstream consumption
- mixed Stage-C representation no longer defines the active mainline path
- the minimum G4 evidence pack exists

### G5 — Seen visual prototypes and class-level text map
Goal:
- construct seen visual prototypes in DINO space
- train the class-level text map `A`
- produce mapped text prototypes for attribution and inference

Exit criteria:
- prototype bank exists and is stable
- mapped text prototype cache exists and is consumable
- deterministic pseudo text-prototype cache is not the active mainline backend
- the minimum G5 evidence pack exists

### G6 — Core open-world attribution
Goal:
- train the v9 core attribution path on `Y'(v) + bg + unk`

Required evidence hierarchy:
1. closed-world comparator on the same protocol
2. v9 core open-world attribution without optional enhancements
3. hidden-positive evidence under the core path
4. optional enhancement only if required by the failure playbook

Exit criteria:
- acceptance in `METRICS_ACCEPTANCE.md` for the active attribution gate is met
- the minimum G6 evidence pack exists

### G7 — Bag-free inference and terminal mainline closure
Goal:
- produce full-vocabulary bag-free inference using the accepted v9 core path
- evaluate AP / HPR / UAR / robustness under canonical validation

Exit criteria:
- bag-free inference is functioning
- canonical evaluation evidence is present
- terminal acceptance is met
- the minimum G7 evidence pack exists

Terminal rule:
- the documented mainline ends at `G7`
- when `G7` has an evidence-backed `PASS`, the supervisor must enter terminal-mainline mode
- no further gate activates unless the authoritative docs themselves are updated

## 3. Default-off research branches
These are not part of the mainline unless `STATUS.md` explicitly activates them:
- prototype EM / momentum refresh
- candidate retrieval
- warm-up BCE
- temporal consistency
- unknown fallback
- one-round quality-aware refinement
- any multi-round continuation
- scenario/domain-missing protocol
- unrestricted or held-out text expansion

## 4. Exploration policy
Exploration is allowed only when:
- the current gate explicitly requires it, and
- `STATUS.md` records the uncertainty, the smallest proposed experiment, and the fallback path.

If a result is inconclusive, prefer narrowing the current design rather than opening another branch.
