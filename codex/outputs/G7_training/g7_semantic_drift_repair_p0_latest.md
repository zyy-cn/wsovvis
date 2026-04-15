# G7 Semantic Drift Repair P0

Status: PASS

This round applied the package-authorized P0 repair for:
- extra proposal semantics
- coverage-aware refinement semantics
- true EM subiterations

Current control-plane task after recompile:
- `G7_training-task`

Preflight:
- local git clean: PASS
- sync/parity: PASS

Bounded remote smoke:
- prealign smoke: PASS
- softem smoke: PASS
- `em_subiterations=2` exercised in softem smoke

Key repaired behaviors:
- extra proposal candidates are evidence-driven and exclude observed classes
- coverage-aware refinement uses current responsibility mass
- softem subiterations are now iterative rather than no-op

Canonical small artifacts pulled back:
- `codex/outputs/g7_training/prealign_contract_check.json`
- `codex/outputs/g7_training/softem_contract_check.json`
- `train/prealign/train_state.json`
- `train/prealign/proxy_records.jsonl`
- `train/softem_base/train_state.json`
- `train/softem_base/responsibility_records.jsonl`
- `train/softem_aug/train_state.json`
- `train/softem_aug/responsibility_records.jsonl`

Audit-only infrastructure remains intact:
- Audit-v1 preserved
- extra recovery audit preserved

Formal closure:
- not claimed in this artifact
- `formal_training_ready: false`

Next recommended experiment:
- stage-wise projector quality audit and/or small closed-set controlled-drop recovery
