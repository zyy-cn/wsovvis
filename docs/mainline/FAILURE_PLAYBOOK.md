# WS-OVVIS Failure Playbook

Use this file when a gate does not pass.

## G0 failures
- If the adapted docs and repository layout disagree, fix the control-plane docs first; do not start algorithm work.
- If `tools/run_mainline_loop.py --dry-run` fails, repair only the control-plane helper and report paths needed for bounded loops.
- If wrapper or bootstrap facts are unavailable, classify as `BLOCKED` / `INCONCLUSIVE` rather than inferring algorithmic failure.

## G1 failures
- Narrow to the label-protocol builder, manifest schema, and deterministic sampling behavior.
- Do not open semantic or basis-generator branches to compensate for a protocol-contract failure.

## G2 failures
- First check DINOv2 integration plumbing, export schema, and original-threshold comparator inputs.
- Do not open refinement, lowconf, or threshold-scan branches until the raw-threshold evidence shows they are needed.

## G3 failures
- Fix stable IDs, manifests, or replay paths before touching prototype or attribution logic.
- Preserve the principle that banks are durable artifacts, not transient caches.

## G4 failures
- Narrow to prototype initialization, class-level text-map serialization, or manifest alignment.
- Do not widen into free-form text or larger mapping architectures during the gate.

## G5 failures
- Narrow to bounded attribution behavior, unknown-foreground routing, and ws-metric reporting.
- Only after the gate evidence isolates the defect may a documented semantic-defense fallback be opened.

## G6 failures
- First verify the bag-free path proof and test-time interface assumptions.
- Do not fall back to bag-constrained inference and still call the mainline claim proven.

## G7 failures
- Treat remote environment, wrapper, bootstrap-link, or commit-consistency issues as validation blockers.
- Do not continue algorithm development while canonical replay is unresolved.


## Evidence-gap rule
If acceptance looks satisfied but the required evidence pack is incomplete, do not open a new branch. Keep the gate `INCONCLUSIVE` and close the evidence gap first.
