# WS-OVVIS Supervisor Deployment

## Goal
Deploy a conservative supervisor layer on top of the document-driven automation kit.
The supervisor must be able to operate **without old workflow docs**, while preserving the old canonical environment semantics.

## Components
- repo skill: `.agents/skills/wsovvis-mainline-supervisor/`
- helper script: `tools/run_mainline_loop.py`
- state machine: `docs/mainline/SUPERVISOR_STATE_MACHINE.md`
- environment authority: `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
- execution runbook: `docs/mainline/IMPLEMENT.md`

## Deployment steps
1. Place all files in the repository root according to their relative paths.
2. Ensure the base automation kit is already deployed:
   - `AGENTS.md`
   - `.codex/config.toml`
   - `docs/mainline/*`
   - `.agents/skills/wsovvis-phase-gate-check/`
   - `.agents/skills/wsovvis-eval-acceptance/`
3. Create the reports directory if missing:
   - `docs/mainline/reports/`
   - `docs/mainline/reports/archive/`
4. Confirm the current code repo still contains the canonical wrapper and Stage B / Stage C entrypoints named in `CODEBASE_MAP.md`.
5. Treat the first supervisor run as **G0 environment inheritance verification**, not as algorithm development.

## Mandatory G0 environment verification
The clean kit is not considered fully active until all of these are checked:
1. canonical host alias `gpu4090d` is reachable
2. canonical remote repo dir is `/home/zyy/code/wsovvis_runner`
3. `tools/remote_verify_wsovvis.sh` is present and usable
4. canonical runner bootstrap preflight is available or the absence is explicitly recorded as a blocker
5. remote environment activation recipe is confirmed:
   - `source ~/software/miniconda3/etc/profile.d/conda.sh`
   - `conda activate wsovvis`
   - `PYTHONPATH` includes `third_party/VNext`
6. if push routing is required, `github-via-gpu` or equivalent is confirmed
7. the first canonical run records `remote HEAD == intended local commit`

If any item above is missing, classify G0 as `BLOCKED` or `INCONCLUSIVE`; do **not** continue deeper into the mainline as if the environment were already inherited.

## Recommended usage
### Interactive
Run:
```bash
python tools/run_mainline_loop.py --dry-run
```
Then paste the generated prompt into Codex.

### Non-interactive
Run:
```bash
python tools/run_mainline_loop.py
```
Then use the saved prompt from `docs/mainline/reports/supervisor_prompt_latest.txt` with `codex exec`.

### Terminal accepted mainline
When the accepted terminal gate has already passed, run:
```bash
python tools/run_mainline_loop.py --dry-run
```
The helper must stop automatically, avoid emitting a new coding-step prompt, and write:
- `docs/mainline/reports/mainline_terminal_summary.txt`

To prepare a bounded terminal revalidation prompt instead, run:
```bash
python tools/run_mainline_loop.py --terminal-revalidate --dry-run
```

## Scope
This v1 supervisor does not directly invoke Codex itself. It prepares and standardizes one bounded supervisor iteration.
This is intentional to keep the first deployment conservative and auditable.

## What counts as a successful supervisor deployment
A successful deployment means:
- the supervisor skill is discoverable via `/skills`
- the supervisor can read the full mainline document set
- G0 can be judged using only the new kit
- no old workflow doc is required to interpret canonical remote validation
- phase/gate and acceptance outputs are written into `docs/mainline/reports/`
- terminal-mainline detection is explicit when the accepted terminal gate is reached
- bounded terminal revalidation can be prepared without reopening scope
