$mainline-supervisor

Read the following files in this exact order before doing any work:
- AGENTS.md
- START_AUTOMATION.md
- docs/mainline/INDEX.md
- docs/mainline/EXECUTION_SOURCE.md
- docs/mainline/PLAN.md
- docs/mainline/IMPLEMENT.md
- docs/mainline/STATUS.md
- docs/mainline/METRICS_ACCEPTANCE.md
- docs/mainline/EVIDENCE_REQUIREMENTS.md
- docs/mainline/FAILURE_PLAYBOOK.md
- docs/mainline/ENVIRONMENT_AND_VALIDATION.md
- docs/mainline/CODEBASE_MAP.md
- docs/mainline/STAGEB_INTERFACE_CONTRACT.md
- docs/mainline/SUPERVISOR_STATE_MACHINE.md
- docs/mainline/SUPERVISOR_DEPLOYMENT.md
- docs/runbooks/mainline_phase_gate_runbook.md

Repository-specific facts already established:
- design source: `docs/outline/WS_OVVIS_outline_v9.tex`
- execution source after deployment: `docs/mainline/*`
- active initial gate: `G0 — v9 control-plane bootstrap and inheritance verification`
- terminal gate: `G7`
- canonical remote alias: `gpu4090d`
- canonical runner repo dir: `/home/zyy/code/wsovvis_runner`
- canonical wrapper: `bash tools/remote_verify_wsovvis.sh`
- bootstrap preflight checker: `python tools/check_canonical_runner_bootstrap_links.py --check`
- conda activation: `source ~/software/miniconda3/etc/profile.d/conda.sh && conda activate wsovvis`
- canonical `PYTHONPATH`: `/home/zyy/code/wsovvis_runner/third_party/VNext`
- origin clone URL: `git@github.com:zyy-cn/wsovvis.git`
- push-route alias: `github-via-gpu`

Run exactly one bounded supervisor iteration for `G0`.

Tasks:
1. Determine the active gate and blocker from the deployed docs, without widening scope.
2. Verify that the repository is now interpreted through `AGENTS.md` + `docs/mainline/*`, not through deleted legacy docs or draft `docs/mainline_v9/*`.
3. Verify that `docs/mainline/CODEBASE_MAP.md`, `docs/mainline/STAGEB_INTERFACE_CONTRACT.md`, and `docs/mainline/ENVIRONMENT_AND_VALIDATION.md` match the real repository files that currently exist.
4. Verify the inherited canonical validation semantics using the existing project tooling:
   - `tools/remote_verify_wsovvis.sh`
   - `tools/check_canonical_runner_bootstrap_links.py`
   - local git config remote facts when needed
5. Produce the smallest valid evidence-backed `G0` result.
6. Write or update all required durable outputs:
   - `docs/mainline/reports/phase_gate_latest.txt`
   - `docs/mainline/reports/acceptance_latest.txt`
   - `docs/mainline/reports/evidence_latest.txt`
   - `docs/mainline/reports/worked_example_verification_latest.md`
   - `docs/mainline/reports/worked_example_verification_latest.json`
   - timestamped archive copies under `docs/mainline/reports/archive/`
   - `docs/mainline/STATUS.md` if state materially changes
7. Treat `PASS` as valid only if both the `G0` acceptance contract and the `G0` evidence pack are complete.
8. If canonical remote replay is not yet executable or remote HEAD consistency cannot be recorded in this loop, classify the result as `INCONCLUSIVE` or `BLOCKED`, not as algorithmic failure.
9. Stop after this one bounded loop.

Hard constraints:
- no algorithmic Stage B / C / D changes during `G0`
- no default-off modules
- no scope expansion
- no contract-only PASS
- preserve terminal-mode stop semantics and bounded terminal revalidation semantics with terminal gate fixed at `G7`

When choosing exact verification actions, prefer the smallest set that can still leave a durable, reviewable G0 evidence bundle.
