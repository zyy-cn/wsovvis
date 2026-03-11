# WS-OVVIS Mainline Status

This file tracks the current authoritative state for document-driven automation.
It is the shared memory for new Codex sessions and supervisor-driven loops.

## Current state
- Current code snapshot status: partially structured repository with mature Stage B / Stage C / Stage D assets, deployed `docs/mainline/*` control-plane files, deleted draft `docs/mainline_v9/*` files in the working tree, and a bounded `G0` canonical replay retry completed on `20260311T040711Z`
- Active gate: `G0`
- Terminal mainline mode: `inactive`
- Mainline authority: `docs/mainline/*`
- Default-off modules: `prototype EM / momentum refresh`, `candidate retrieval`, `warm-up BCE`, `temporal consistency`, `unknown fallback`, `one-round quality-aware refinement`, `multi-round continuation`, `scenario/domain-missing protocol`, `prompt ensemble / synonym expansion`, `held-out / unrestricted text evaluation`
- Acceptance status: `BLOCKED`
- Evidence status: `partial locally; canonical inheritance blocked`
- Evidence bundle reviewed: `20260311T040711Z bounded G0 canonical replay retry`

## Why the current gate is active
The deployed docs now resolve the repository through `AGENTS.md` plus `docs/mainline/*`, and the active-gate interpretation is locally unambiguous as `G0`.
`G0` remains active because the exact canonical wrapper path still cannot reach the intended remote branch or commit, so canonical environment inheritance and remote-`HEAD` evidence cannot yet be recorded.
The Stage B transition contract reconciliation remains intentionally deferred until the canonical replay prerequisite succeeds, matching the current bounded `G0` task order.
No later gate may activate until both the G0 acceptance contract and the G0 evidence pack are complete.

## Current blockers
- exact canonical wrapper replay on `20260311T040711Z` reached `gpu4090d` but failed during remote checkout: `fatal: 'origin/mainline-v9-g0' is not a commit and a branch 'mainline-v9-g0' cannot be created from it`
- direct local reachability to the origin fetch route is still not recordable: `git ls-remote origin refs/heads/mainline-v9-g0` failed again on `20260311T040711Z` with `ssh: Could not resolve hostname github.com: Temporary failure in name resolution`
- `docs/mainline/STAGEB_INTERFACE_CONTRACT.md` still needs the planned doc-only reconciliation to the current `track_id`-based export contract, but that step remains deferred until canonical replay is recorded

## Canonical environment evidence tracker
- remote alias `gpu4090d`: `exact retry on 20260311T040711Z reached the remote runner and advanced to remote git checkout`
- canonical runner dir `/home/zyy/code/wsovvis_runner`: `documented and reached by the wrapper, but bootstrap inspection did not run because remote branch checkout failed`
- wrapper `bash tools/remote_verify_wsovvis.sh`: `present in repo; exact retry recorded in G0 evidence; remote execution now reaches checkout stage but still blocks before the verification command`
- bootstrap preflight `python tools/check_canonical_runner_bootstrap_links.py --check`: `present in repo; checker semantics verified locally; canonical runner check not recorded because remote execution blocked`
- remote HEAD == intended local commit: `not recorded; remote branch checkout failed and local origin reachability check still failed`
- push route alias `github-via-gpu`: `verified from local git config as origin push URL git@github-via-gpu:zyy-cn/wsovvis.git`

## Next smallest valid step
Make the intended branch or commit reachable on origin from the approved path, then rerun the same bounded `G0` loop to record:
1. canonical runner bootstrap output from `/home/zyy/code/wsovvis_runner`,
2. remote environment activation and `PYTHONPATH` inheritance,
3. remote `HEAD == intended local commit`,
4. only after that, a minimal doc-only reconciliation of `STAGEB_INTERFACE_CONTRACT.md` if the current `track_id`-based export wording remains mismatched.

## Latest evidence
- authoritative design source identified: `docs/outline/WS_OVVIS_outline_v9.tex`
- `python tools/run_mainline_loop.py --dry-run` resolved the current gate from `STATUS.md` as `G0`
- authority chain now resolves through `AGENTS.md` + `docs/mainline/*`; `docs/mainline/EXECUTION_SOURCE.md` explicitly replaces `docs/mainline_v9/*`, and the working tree shows `docs/mainline_v9/*` deleted
- `docs/mainline/CODEBASE_MAP.md` matches the current repository anchors; all listed planned v9 entrypoints remain absent as planned
- current Stage B export code writes `manifest.v1.json`, `track_metadata.v1.json`, and `track_arrays.v1.npz` with `track_id`, `row_index`, temporal fields, `objectness_score`, embeddings, and producer provenance
- exact retry on `20260311T040711Z` of `git ls-remote origin refs/heads/mainline-v9-g0` still failed with GitHub name-resolution errors from the local environment
- exact retry on `20260311T040711Z` of the canonical wrapper/bootstrap command reached `gpu4090d` but failed because `origin/mainline-v9-g0` was not available for checkout on the remote runner

## Latest evidence artifact pointers
- Phase/gate report: `docs/mainline/reports/phase_gate_latest.txt`
- Acceptance report: `docs/mainline/reports/acceptance_latest.txt`
- Evidence report: `docs/mainline/reports/evidence_latest.txt`
- Worked example (md): `docs/mainline/reports/worked_example_verification_latest.md`
- Worked example (json): `docs/mainline/reports/worked_example_verification_latest.json`
