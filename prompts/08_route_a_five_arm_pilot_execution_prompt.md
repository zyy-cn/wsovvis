Operate strictly inside refined S1 Route A strengthening.

Current authoritative state:
- Refined S1 remains formal PASS via Route B.
- Do not reopen Route B.
- Do not activate S2.
- Execute the revised five-arm low-cost Route A pilot exactly as frozen.

Your task:
1. Stay strictly inside refined S1 diagnostic strengthening.
2. Execute the frozen pilot arms:
   - A0_baseline_subset
   - A1_highconf_subset
   - A2_lowconf_subset
   - C0_dupselect_control
   - C1_cardinality_control
3. Keep the frozen pilot surfaces unchanged:
   - train subset: 256 deterministic videos
   - val subset: 96 deterministic videos
   - seed = 42
   - anchor videos: 00001 and 00145
4. Use only the already allowed intervention categories:
   - VideoCutLER confidence sweep
   - short-horizon SeqFormer schedule/runtime overrides
   - existing-weight / same-arm resume controls only
   - C0: dynamic threshold + bounded top-k + duplicate suppression
   - C1: activate existing loss_cardinality only
5. Evaluate only on the frozen Route A metric surface:
   - mean_best_iou
   - recall_at_0.5
   - fragmentation_per_gt_instance
   plus the frozen diagnostics
6. Apply the frozen stop rule exactly.
7. Regenerate:
   - docs/scientific/reports/phase_gate_latest.txt
   - docs/scientific/reports/evidence_latest.txt
   - docs/scientific/reports/acceptance_latest.txt
   - docs/scientific/reports/signoff_latest.txt
   - docs/scientific/STATUS.md
   - docs/scientific/reports/comparator_latest.txt if pilot comparator evidence is produced
8. Stop after pilot execution and judgment only.

Constraints:
- No Route B reopening.
- No S2 activation.
- No full-scale Route A training.
- No hidden widening into semantic/prototype/text-map/attribution/bag-free modules.
- No comparator redefinition.
