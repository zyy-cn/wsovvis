# Route A Five-Arm Pilot Addendum

This addendum updates the bounded Route A strengthening pilot while preserving:
- formal PASS of refined S1 via Route B
- inactivity of S2
- the previously frozen Route A primary metric surface

Additions:
- C0_dupselect_control: no-training export/selection-side duplicate suppression control
- C1_cardinality_control: short-horizon training-side control that activates loss_cardinality only

Interpretation rule:
- A0/A1/A2 probe target-side confidence sensitivity
- C0 probes export/selection-side count mismatch
- C1 probes training-side cardinality mismatch

Decision rule:
- If C0 is best: prioritize export/selection-side stabilization
- If C1 is best: prioritize cardinality-aware training-side strengthening
- If A1/A2 dominate: prioritize target-confidence / pseudo-target surface refinement
- If none are promising: do not authorize full-scale Route A training yet
