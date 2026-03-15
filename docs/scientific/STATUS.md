# WS-OVVIS Scientific Overlay Status (v2 / v10-refined)

This file tracks the current authoritative state for the scientific validation overlay.

## Schema
- Scientific overlay version
- Formal active gate
- Diagnostic active gate
- Scientific mode (`formal` / `diagnostic` / `recovery` / `migration`)
- Last formal PASS
- Last formal FAIL
- Current recovery gate
- Later formal gate activation allowed
- Diagnostic-only probing allowed
- Human sign-off status
- Notes on historical evidence carried over from prior overlay versions

## Current formal state
- Scientific overlay version: `v2 / v10-refined`
- Scientific mode: `diagnostic`
- Formal active gate: `none; refined S1 PASS already recorded and no later formal gate activated in this loop`
- Diagnostic active gate: `S1 Route A strengthening fresh exclusion-policy A2 full-scale validation complete; aggregate comparator negative`
- Last formal PASS: `S1 — Basis Utility PASS via Route B recorded on 2026-03-12 using the aligned canonical packet at codex/s1_routeb_closure_20260312T130255Z/`
- Last formal FAIL: `none under the refined overlay; historical strict S1 FAIL remains legacy evidence only`
- Current recovery gate: `none`
- Later formal gate activation allowed: `yes; refined S1 already has an evidence-backed PASS, but S2 and all later gates remain inactive until explicitly activated in a later loop`
- Diagnostic-only probing allowed: `yes; the completed exclusion-policy A2 full-scale comparator is recorded inside refined S1 without reopening Route B or activating S2`
- Human sign-off status: `final human scientific sign-off for refined S1 Route B remains authoritative; the fresh exclusion-policy A2 full-scale Route A result is recorded as aggregate-negative; S2 remains inactive`

## Notes
- The refined v10 overlay docs under `docs/scientific/*` are the active scientific source of truth.
- Historical strict `S1 FAIL` / `S1R` materials remain preserved as legacy evidence only.
- Authoritative formal refined `S1` PASS packet: `codex/s1_routeb_closure_20260312T130255Z/`.
- Route B remains the authoritative formal evidence route for refined `S1`; it is closed and is not reopened here.
- Completed fresh exclusion-policy Route A packet: `codex/s1_routea_a2_fullscale_exclude_mixed_fresh_20260313T110810Z/`
- Full-scale Route A comparator exists at `codex/s1_routea_a2_fullscale_exclude_mixed_fresh_20260313T110810Z/full_eval/summary.json` and `codex/s1_routea_a2_fullscale_exclude_mixed_fresh_20260313T110810Z/full_eval/comparator_table.md`
- Aggregate Route A outcome is negative and does not support direct-superiority strengthening:
  - `mean_best_iou`: `-0.032327`
  - `recall_at_0.5`: `-0.023394`
  - `fragmentation_per_gt_instance`: `+0.773057`
- No `S2` activation follows from this result.
