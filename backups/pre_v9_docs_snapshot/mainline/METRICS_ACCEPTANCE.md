# WSOVVIS Metrics and Acceptance

This file defines gate-level acceptance for the mainline.

## 1. Mainline metric priority
Primary metrics:
- `AP`
- `AP_base`
- `AP_novel`
- `AURC`
- `SCR`
- `HPR`
- `UAR`

Secondary / supporting metrics:
- tracking-oriented auxiliaries
- additional ablation-only metrics

## 2. Gate acceptance policy
### G0
PASS if:
- environment facts are confirmed,
- code entrypoint map is confirmed,
- Stage B interface contract matches code,
- canonical remote validation semantics are executable.

FAIL if:
- the environment model is wrong,
- codebase map points to missing or incorrect files,
- the validation wrapper path is invalid.

### G1
PASS if:
- protocol generation is runnable,
- protocol test path passes,
- one closed-world baseline path is smoke-runnable.

### G2
PASS if:
- Stage B schema, bridge, and consumer requirements match the code,
- Stage B validation tests pass.

### G3
PASS if:
- the Stage C representation bridge is coherent,
- representation-critical tests pass,
- no regression breaks loader/prototype/semantic-slice coherence.

### G4 (core claim gate)
The active attribution gate should be judged in this order:
1. Is there a closed-world baseline result?
2. Is there an open-world result without broader enhancements?
3. Does hidden-positive handling improve?
4. Is positive-evidence coverage preserved or improved?

PASS if all are true:
- the open-world variant is functioning,
- `HPR` improves relative to the closed-world baseline,
- `UAR` improves relative to the closed-world baseline,
- `SCR` is not degraded in a way that invalidates positive-evidence coverage,
- standard metrics (`AP` or `AURC`) are not catastrophically worse.

INCONCLUSIVE if:
- open-world logic runs but hidden-positive metrics are missing,
- hidden-positive metrics are present but evidence is too noisy to decide,
- the comparison setup is not protocol-aligned.

FAIL if:
- hidden positives still collapse predominantly to background,
- `HPR`/`UAR` clearly fail to improve,
- the implementation drifted into a default-off branch.

### G5
PASS if:
- full-video linking works under the bounded policy,
- linking does not destabilize the accepted attribution mainline,
- global classification closure remains on bounded defaults.

### G6
PASS if:
- single-round refinement provides non-negative net value,
- no regression in hidden-positive handling,
- default-off branches remain off.

## 3. Missing metric implementation policy
If `HPR` and `UAR` are not yet implemented in code, the current attribution gate cannot be fully passed.
In that case, the correct status is `INCONCLUSIVE`, and the smallest valid next step is to implement or expose these metrics.

## 4. PASS / FAIL / INCONCLUSIVE semantics
- `PASS`: gate is satisfied and the next gate may become active.
- `FAIL`: gate is not satisfied and the fallback path must be used.
- `INCONCLUSIVE`: evidence is insufficient; do not widen scope, first close the evidence gap.
