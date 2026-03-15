# WS-OVVIS Codebase Map

This file maps the project repository to the automation control plane.

## Top-level code domains
- `docs/outline/` — authoritative design source; use `WS_OVVIS_outline_v13_with_gates.tex` first when interpreting scope and gates.
- `tools/` — operational entrypoints for label protocol generation, basis evaluation, bank construction, prototype/text-map training, open-world attribution, bag-free evaluation, canonical replay, and supervisor prompt generation.
- `tests/` — authoritative unit/integration checks for each mainline gate; the most relevant gate tests are named directly in `PLAN.md`.
- `wsovvis/` — core library code, including tracking, features, semantics, attribution, metrics, and track-feature export utilities.
- `configs/` + `train_seqformer_pseudo.py` — basis-generator training configuration and launch surfaces.
- `scripts/` — helper execution surfaces for subset or remote runs; informative unless a gate explicitly elevates them.
- `third_party/VNext/` — required upstream code on the current `PYTHONPATH` contract; environment-critical but not the authority for the WS-OVVIS control plane.

## Mainline entrypoint policy
If multiple code paths exist, the active one must be declared in `STATUS.md`; it must not be chosen ad hoc from historical code variety.

## Evidence-producing entrypoint policy
For each active gate, the authoritative scripts, tests, or notebooks that produce the required evidence pack should be identifiable from this map or explicitly recorded in `STATUS.md`.
