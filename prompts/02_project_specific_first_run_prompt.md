Read the privatized WS-OVVIS control plane first, in this exact order:
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

Then inspect the actual authoritative repository entrypoints for the first two layers of the mainline:
- docs/outline/WS_OVVIS_outline_v13_with_gates.tex
- tools/run_mainline_loop.py
- tools/check_canonical_runner_bootstrap_links.py
- tools/remote_verify_wsovvis.sh
- tools/build_wsovvis_labelset_protocol.py
- tests/test_run_mainline_loop_v1.py
- tests/test_build_wsovvis_labelset_protocol.py

Your task is to run exactly one bounded first-loop iteration for this privatized project.

Hard requirements:
1. Treat `G0 - Authority switch and canonical environment inheritance` as the active gate unless the freshly deployed `docs/mainline/STATUS.md` itself contains stronger evidence-backed state.
2. Do not open any algorithm-development branch during this first loop.
3. Verify that the v13 outline is the design source and report any conflict with older v9 references if found.
4. Verify the presence and intended role of:
   - `tools/remote_verify_wsovvis.sh`
   - `tools/check_canonical_runner_bootstrap_links.py`
   - `tools/run_mainline_loop.py`
5. Run the smallest valid G0 checks you can without widening scope. At minimum, generate or refresh the supervisor dry-run prompt and capture the G0 evidence bundle.
6. Gate judgment must be evidence-backed. If canonical remote facts are not actually evidenced yet, keep the gate `INCONCLUSIVE` or `BLOCKED`; do not claim `PASS` on documentation alone.
7. Write or update all required state-bearing outputs:
   - docs/mainline/STATUS.md
   - docs/mainline/reports/phase_gate_latest.txt
   - docs/mainline/reports/acceptance_latest.txt
   - docs/mainline/reports/evidence_latest.txt
   - docs/mainline/reports/worked_example_verification_latest.md
   - docs/mainline/reports/worked_example_verification_latest.json
8. Stop after that one bounded loop.

The expected result of this prompt is not model progress. The expected result is a trustworthy G0 authority/environment status with durable reports.
