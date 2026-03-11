# WSOVVIS Failure Playbook

Use this file when a gate does not pass.

## 1. Attribution failure modes (G4)
### Symptom A: open-world attribution runs, but hidden positives still collapse to background
Fallback order:
1. verify protocol alignment and closed-world comparator
2. disable broader candidate-set expansion ideas
3. reduce to the bounded open-world core: positive labels + `bg` + `unk`
4. simplify attribution before adding new mechanisms

### Symptom B: `unk` absorbs too much mass and semantics become non-informative
Fallback order:
1. reduce candidate breadth
2. reduce coverage pressure
3. simplify the attribution backend
4. preserve the open-world distinction, but reduce complexity

### Symptom C: no stable evidence because `HPR` / `UAR` are missing
Fallback order:
1. implement or expose the missing metrics
2. rerun the same comparison
3. do not widen scope before the evidence gap is closed

## 2. Refinement failure modes (G6)
### Symptom: refinement introduces instability or negative gain
Fallback order:
1. keep only single-round refinement
2. drop semantic-heavy gating
3. retain mask/temporal denoising only
4. if still unstable, remove refinement from the active gate and keep it out of the mainline

## 3. Linking / inference failure modes (G5)
### Symptom: semantic signals destabilize linking
Fallback order:
1. reduce semantics in the linking score
2. use geometry + query as primary linking signals
3. revert global classification to quality-weighted logit averaging
4. keep stronger memory aggregation off

## 4. Environment / validation failure modes
### Symptom: canonical validation cannot be trusted
Fallback order:
1. verify remote runner path and wrapper command
2. verify push succeeded
3. verify remote HEAD == intended local commit
4. treat mismatch or infra errors as `BLOCKED`, not as algorithmic failure
