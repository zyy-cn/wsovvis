# WSOVVIS V9 Metrics and Acceptance

This file defines gate evaluation semantics for the v9 mainline.
A gate cannot receive `PASS` unless both the acceptance contract and the required evidence pack are complete.
See `EVIDENCE_REQUIREMENTS.md` for the evidence-pack requirements.

## 1. Evaluation layers
Every gate review must evaluate two layers.

### Acceptance layer
Does the gate satisfy the technical contract for the current mainline phase?

### Evidence layer
Is there enough reviewable quantitative, visual, and worked-example evidence to justify `PASS`?

If the acceptance layer is satisfied but the evidence layer is incomplete, the correct result is `INCONCLUSIVE`.

## 2. Gate-level acceptance contracts

### G0
Acceptance contract:
- the v9 control docs are internally consistent,
- the real repository can be mapped to the v9 control plane,
- the authority-switch patch scope is identifiable,
- the active-gate interpretation under the v9 docs is unambiguous.

Evidence minimum:
- dry-run or equivalent control-plane resolution output,
- authority-chain comparison,
- one worked example showing how the gate is resolved from the docs.

### G1
Acceptance contract:
- clip-level `Y'(v)` protocol generation is defined and testable,
- artifact contracts are documented and parseable,
- the mainline no longer depends on window-level `Y'(w)`.

Evidence minimum:
- actual `Y'(v)` statistics,
- full-vs-observed worked example(s),
- contract readback proof for the declared artifacts.

### G2
Acceptance contract:
- class-agnostic basis path is structurally runnable,
- local-tracklet artifacts are identifiable and contract-stable,
- structure-side diagnostics exist.

Evidence minimum:
- local-tracklet count/duration/filtering statistics,
- pseudo-tube to local-tracklet worked example,
- structure visualizations.

### G3
Acceptance contract:
- clip-level global-track-bank construction is deterministic enough for downstream reuse,
- linking remains structure-dominant,
- stitching diagnostics exist.

Evidence minimum:
- stitching success/fragmentation statistics,
- score-matrix-to-merge worked example,
- global-track coverage visualization.

### G4
Acceptance contract:
- `z_tau` is available as the mainline semantic carrier,
- `o_tau` is available or derivable,
- semantic cache artifacts are stable enough for downstream use,
- mixed Stage-C semantics no longer define the active mainline path.

Evidence minimum:
- semantic-cache coverage statistics,
- objectness distribution,
- one per-track DINO extraction and aggregation worked example,
- crop/pooling provenance visualization.

### G5
Acceptance contract:
- seen visual prototypes exist,
- the class-level text map `A` is trainable and consumable,
- mapped text prototypes are available for downstream attribution/inference,
- deterministic pseudo text-prototype cache is not the active backend.

Evidence minimum:
- prototype-bank coverage metrics,
- text-map alignment metrics,
- one class-level worked example,
- prototype/text-map visualization.

### G6
Acceptance contract:
`PASS` only if all are true:
- the v9 core open-world variant is functioning,
- `HPR` improves relative to the closed-world comparator,
- `UAR` improves relative to the closed-world comparator,
- positive-evidence coverage is not invalidated,
- standard metrics (`AP` or `AURC`) are not catastrophically worse,
- the active result comes from the bounded core path rather than a default-off enhancement.

Evidence minimum:
- known/background/unknown allocation statistics,
- clip-level attribution worked example with cost matrix and assignment summary,
- aligned closed-world comparison,
- hidden-positive evidence (`HPR`, `UAR`) or an explicit evidence-gap note.

`INCONCLUSIVE` if:
- the core logic runs but hidden-positive metrics are missing,
- hidden-positive metrics are present but evidence is too noisy to decide,
- the comparison setup is not protocol-aligned,
- optional enhancements were required before the core path itself was understood,
- evidence is incomplete even though the contract looks promising.

`FAIL` if:
- hidden positives still collapse predominantly to background,
- `HPR`/`UAR` clearly fail to improve,
- the implementation drifted into a default-off branch before proving the core path,
- the evidence directly contradicts the PASS claim.

### G7
Acceptance contract:
- bag-free inference works,
- the canonical evaluation path is runnable,
- AP / HPR / UAR / robustness evidence is recorded,
- the accepted core path does not require observed-label bags at test time.

Evidence minimum:
- canonical evaluation record,
- full-vocab bag-free worked example,
- qualitative prediction visualizations,
- main result table/summary with robustness evidence.

## 3. Missing metric implementation policy
If `HPR` and `UAR` are not yet implemented in code, the current attribution gate cannot be fully passed.
In that case, the correct status is `INCONCLUSIVE`, and the smallest valid next step is to implement or expose these metrics.

## 4. PASS / FAIL / INCONCLUSIVE semantics
- `PASS`: gate is satisfied, evidence is complete, and the next gate may become active.
- `FAIL`: gate is not satisfied and the fallback path must be used.
- `INCONCLUSIVE`: evidence or acceptance is insufficient; do not widen scope, first close the contract or evidence gap.
