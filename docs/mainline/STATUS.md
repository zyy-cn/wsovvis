# WS-OVVIS Mainline Status

This file tracks the current authoritative state for document-driven automation.
It is the shared memory for new Codex sessions and supervisor-driven loops.

## Current state
- Current code snapshot status: partially mature codebase detected; uploaded package already contains substantial WS-OVVIS code and tests, but the shipped private control plane is incomplete and not authoritative for this adaptation
- Active gate: `G0 - Authority switch and canonical environment inheritance`
- Terminal mainline mode: `inactive`
- Mainline authority: `docs/mainline/*`
- Default-off modules: `stitching`, `refinement`, `duplicate suppression`, `cardinality control`, `SeqFormer-derived lowconf tuning`, `prototype EM`, `multi-round continuation`, `free-form text branches`
- Acceptance status: `unknown`
- Evidence status: `unknown`
- Evidence bundle reviewed: `no`

## Why the current gate is active
The repository already contains substantial implementation artifacts, but the uploaded package does **not** contain a complete self-consistent `docs/mainline/*` control plane. The visible prior overlay points to missing `docs/scientific/*` content and an older v9 design source, while the repository now ships `docs/outline/WS_OVVIS_outline_v13_with_gates.tex`. Because there is no current evidence-backed status bundle proving a later gate, the adapted workflow must restart from `G0` and re-establish authority, environment facts, and report discipline.

## Current blockers
- no current evidence-backed `docs/mainline/STATUS.md` existed in the uploaded project package
- prior overlay references missing documents and therefore cannot be treated as the live execution source
- canonical remote facts (`gpu4090d`, remote repo dir, wrapper health, bootstrap-link state, remote HEAD consistency) have not yet been re-evidenced under this adapted control plane

## Canonical environment evidence tracker
- remote host alias inferred: `gpu4090d`
- canonical remote repo dir inferred: `/home/zyy/code/wsovvis_runner`
- canonical wrapper inferred: `tools/remote_verify_wsovvis.sh`
- bootstrap preflight checker inferred: `tools/check_canonical_runner_bootstrap_links.py --check`
- current canonical judgment: `unknown` until the first G0 loop records command output and commit-consistency evidence

## Next smallest valid step
1. deploy this adapted overlay at the repository root
2. run `python tools/run_mainline_loop.py --dry-run`
3. inspect `tools/check_canonical_runner_bootstrap_links.py --check` usage and wrapper availability
4. write the initial G0 phase-gate, acceptance, evidence, and worked-example stubs
5. keep the gate at `INCONCLUSIVE` or `BLOCKED` until canonical environment facts are actually evidenced

## Latest evidence
- design source identified: `docs/outline/WS_OVVIS_outline_v13_with_gates.tex`
- canonical project wrapper present: `tools/remote_verify_wsovvis.sh`
- canonical bootstrap checker present: `tools/check_canonical_runner_bootstrap_links.py`
- representative mainline entrypoints detected for G1–G6 under `tools/` and `tests/`
- no authoritative current gate reports were present in the uploaded project package

## Latest evidence artifact pointers
- Phase/gate report: `docs/mainline/reports/phase_gate_latest.txt`
- Acceptance report: `docs/mainline/reports/acceptance_latest.txt`
- Evidence report: `docs/mainline/reports/evidence_latest.txt`
- Worked example (md): `docs/mainline/reports/worked_example_verification_latest.md`
- Worked example (json): `docs/mainline/reports/worked_example_verification_latest.json`
