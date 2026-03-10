# WSOVVIS V9 Execution Source

This file is the compact execution-source summary for the v9 mainline.

## What this replaces
It replaces old workflow control documents, handoff notes, and historical task packs as the authority for active **v9** development decisions once the repository switches authority.

## What this does not replace
It does not replace the research outline as the design source.
The v9 outline remains the high-level research document.

## Current execution model
- research/design source: v9 outline
- execution source: `docs/mainline_v9/*`
- repo rules: `AGENTS.md`
- repeatable workflows: `.agents/skills/*`

## Mainline objective
Prove the v9 core path from clip-level weak supervision to bag-free inference.

## Mainline gates
See `PLAN.md`.

## Gate evaluation
See `METRICS_ACCEPTANCE.md`.

## Evidence-backed PASS rule
A gate does not receive `PASS` unless the acceptance contract and the required evidence pack are both complete.
See `EVIDENCE_REQUIREMENTS.md`.

## Fallback policy
See `FAILURE_PLAYBOOK.md`.

## Environment model
See `ENVIRONMENT_AND_VALIDATION.md`.
