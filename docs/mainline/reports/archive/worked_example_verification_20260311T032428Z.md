# WS-OVVIS G0 Worked Example — Authority-Switch and Blocked Inheritance Resolution

## Goal
Show how one bounded supervisor loop resolves the active gate and final `G0` status from the deployed docs plus the current repository/tooling state.

## Inputs inspected
- `AGENTS.md`
- `START_AUTOMATION.md`
- `docs/mainline/INDEX.md`
- `docs/mainline/EXECUTION_SOURCE.md`
- `docs/mainline/PLAN.md`
- `docs/mainline/STATUS.md`
- `docs/mainline/METRICS_ACCEPTANCE.md`
- `docs/mainline/EVIDENCE_REQUIREMENTS.md`
- `docs/mainline/ENVIRONMENT_AND_VALIDATION.md`
- `docs/mainline/CODEBASE_MAP.md`
- `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`
- `docs/mainline/SUPERVISOR_STATE_MACHINE.md`
- `docs/mainline/SUPERVISOR_DEPLOYMENT.md`
- `docs/runbooks/mainline_phase_gate_runbook.md`
- `tools/run_mainline_loop.py --dry-run`
- `tools/remote_verify_wsovvis.sh`
- `tools/check_canonical_runner_bootstrap_links.py`

## Concrete evidence
- The dry-run supervisor helper printed `authoritative current gate from STATUS.md: \`G0\``.
- The execution-source docs and active skill/read-order files all point to `docs/mainline/*`; `docs/mainline/EXECUTION_SOURCE.md` explicitly says it replaces `docs/mainline_v9/*`.
- The working tree shows `docs/mainline_v9/*` deleted, so the current interpretation cannot legitimately come from that draft tree.
- `docs/mainline/CODEBASE_MAP.md` matched all listed current anchors in the repository, and every listed planned v9 entrypoint was still absent as planned.
- `wsovvis/track_feature_export/v1_core.py` currently emits `track_id`, `row_index`, `start_frame_idx`, `end_frame_idx`, `num_active_frames`, `objectness_score`, embeddings, `manifest.v1.json`, `track_metadata.v1.json`, and `track_arrays.v1.npz`.
- `docs/mainline/STAGEB_INTERFACE_CONTRACT.md` currently requires `local_track_id`, mask or mask-reference fields, and local-query feature or equivalent as minimum payload fields, so the doc is ahead of the current export contract.
- The canonical wrapper invocation failed with `ssh: connect to host 172.29.112.1 port 2222: No route to host`.
- `git ls-remote origin refs/heads/mainline-v9-g0` failed with `Temporary failure in name resolution`, so remote branch or commit reachability was not recordable.

## Decision trace
1. `PLAN.md`, `STATUS.md`, `SUPERVISOR_STATE_MACHINE.md`, and the runbook keep the repository at `G0` until an evidence-backed pass is recorded.
2. The dry-run helper confirms the active gate is still `G0`, so no later gate may activate.
3. The authority switch itself is locally proven because the active read orders point to `docs/mainline/*` and the draft `docs/mainline_v9/*` tree is deleted from the working tree.
4. `CODEBASE_MAP.md` is locally consistent with the files that exist.
5. `STAGEB_INTERFACE_CONTRACT.md` is not yet fully consistent with the current Stage B export artifact fields, so the local contract portion of `G0` is not complete.
6. Canonical inheritance evidence is blocked because the approved remote wrapper could not reach `gpu4090d`, and remote branch or commit reachability could not be established.

## Output
- active gate: `G0`
- final bounded-loop status: `BLOCKED`
- blocking acceptance condition: canonical inheritance evidence is unavailable from the current environment, and the Stage B transition contract still needs a doc-only reconciliation
- smallest next valid step: restore canonical connectivity, rerun the same wrapper/bootstrap checks, then make only the minimal G0 contract wording fix if needed

## Why this example is sufficient
This worked example ties the authoritative docs, the local repository state, and the actual canonical-tooling outcomes into a single traceable `G0` decision without widening scope beyond control-plane verification.
