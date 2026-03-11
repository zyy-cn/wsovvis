# WS-OVVIS Failure Playbook

Use this file when a gate does not pass.
If a gate is technically promising but lacks the required evidence pack, keep the judgment at `INCONCLUSIVE` and close the evidence gap before widening scope.

## 1. Control-plane bootstrap failure modes (G0)
### Symptom: the repository is still interpreted through legacy or draft control files
Fallback order:
1. re-check the authority chain from `AGENTS.md` to `docs/mainline/*`
2. correct startup instructions, skills, and supervisor helper read order
3. verify that `STATUS.md` resolves unambiguously to `G0`
4. write the missing G0 worked example and authority-chain comparison before declaring `PASS`
5. do not start algorithmic work before the control plane is stable

### Symptom: canonical environment inheritance cannot be trusted
Fallback order:
1. verify wrapper path and invocation syntax
2. verify bootstrap-preflight output
3. verify push route and remote runner directory
4. verify `remote HEAD == intended local commit`
5. classify unresolved infra issues as `BLOCKED`, not as algorithmic failure

## 2. Protocol / contract failure modes (G1)
### Symptom: clip-level protocol is inconsistent with downstream artifacts
Fallback order:
1. simplify to the minimum clip/video-level observed-label representation
2. verify contract parsing before widening metadata
3. keep artifact additions backward-compatible where possible
4. produce the missing protocol statistics and worked example before declaring `PASS`
5. do not open algorithmic work before the contract is stable

## 3. Structure failure modes (G2 / G3)
### Symptom A: local tracklets are unstable or under-specified
Fallback order:
1. verify pseudo-tube filtering and Stage B export wiring
2. narrow to the minimum local-tracklet contract
3. fix structure diagnostics before adding semantics
4. add the missing worked example and visual evidence before declaring `PASS`

### Symptom B: global track stitching is unstable
Fallback order:
1. reduce stitching to overlap IoU + local-query consistency
2. tune only bounded thresholds or weights
3. keep semantics out of linking
4. produce score-matrix and merge worked examples before declaring `PASS`
5. if still unstable, stay in `G3` and do not proceed to semantic gates

## 4. Semantic-carrier failure modes (G4)
### Symptom: `z_tau` cache is unstable or objectness is non-informative
Fallback order:
1. simplify cropping and pooling
2. simplify frame weighting
3. simplify the objectness formula
4. close the missing cache/objectness evidence gap
5. do not reopen mixed Stage-C representation as the mainline carrier

## 5. Prototype / text-map failure modes (G5)
### Symptom: prototype bank or text map is unstable
Fallback order:
1. simplify to the conservative seen-prototype initialization
2. reduce refresh frequency
3. enable prototype EM initialization only if needed
4. enable momentum refresh only if needed
5. produce class-level alignment evidence before declaring `PASS`
6. do not promote deterministic pseudo text-prototypes back into the mainline

## 6. Attribution failure modes (G6)
### Symptom A: core open-world attribution runs, but hidden positives still collapse to background
Fallback order:
1. verify protocol alignment and the closed-world comparator
2. reduce to the bounded core: `Y'(v) + bg + unk`
3. simplify the attribution backend before adding new mechanisms
4. generate the required clip-level worked example and hidden-positive evidence
5. enable warm-up BCE only if the evidence gap persists
6. enable retrieval only after the core path is understood

### Symptom B: `unk` absorbs too much mass and semantics become non-informative
Fallback order:
1. reduce candidate breadth to the bounded core
2. reduce coverage pressure
3. simplify the attribution backend
4. preserve the open-world distinction, but reduce complexity
5. provide quantitative mass-allocation evidence before declaring `PASS`

### Symptom C: no stable evidence because `HPR` or `UAR` are missing
Fallback order:
1. implement or expose the missing metrics
2. rerun the same protocol-aligned comparison
3. complete the evidence pack
4. do not widen scope before the evidence gap is closed

## 7. Bag-free inference failure modes (G7)
### Symptom: bag-free inference fails for structure-related reasons
Fallback order:
1. verify `G3` global-track stability
2. verify `G4` semantic cache integrity
3. verify `G5` mapped prototype integrity
4. close the inference worked-example and qualitative-evidence gap
5. consider one-round quality-aware refinement only if the structure-side failure is well localized and the docs explicitly authorize it

## 8. Evidence-gap rule
If acceptance looks satisfied but the required evidence pack is incomplete, do not open a new branch.
Keep the gate `INCONCLUSIVE` and close the evidence gap first.
