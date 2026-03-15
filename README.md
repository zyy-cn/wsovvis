# WS-OVVIS Project-Private Workflow Kit (v13 evidence-backed adaptation)

This package privatizes the generic Codex mainline workflow kit for **WS-OVVIS**.

## Adaptation summary
- project name: `WS-OVVIS`
- project slug: `wsovvis`
- authoritative design source: `docs/outline/WS_OVVIS_outline_v13_with_gates.tex`
- authoritative execution source after deployment: `AGENTS.md` + `docs/mainline/*`
- initial active gate: `G0 - Authority switch and canonical environment inheritance`
- terminal gate: `G7 - Canonical replay, evidence freeze, and terminal closure`
- canonical environment profile: `gpu4090d`
- canonical wrapper: `tools/remote_verify_wsovvis.sh`
- canonical bootstrap checker: `tools/check_canonical_runner_bootstrap_links.py --check`

## Why this adaptation supersedes the partial prior overlay
The uploaded project package already contained fragments of an older private overlay, but that overlay was not self-consistent inside the uploaded archive: it referenced missing `docs/scientific/*` files and still pointed to an older v9 outline. This package rebuilds the project-private layer as a self-contained evidence-backed control plane centered on the currently uploaded v13 outline.

## Confirmation round
No user confirmation round was required because the project name, objective, design source, canonical profile, wrapper, and appropriate starting gate were all inferable from the uploaded kit and project package.

See `DEPLOYMENT_INSTRUCTIONS.md` and `prompts/02_project_specific_first_run_prompt.md`.
