# WS-OVVIS Project-Private Kit Overlay

This overlay is meant to be copied into the repository root of the WS-OVVIS code checkout.
It renders the generic mainline workflow kit into a WS-OVVIS-private control plane.

Key decisions already baked into this overlay:
- authoritative design source: `docs/outline/WS_OVVIS_outline_v9.tex`
- authoritative execution source after deployment: `docs/mainline/*`
- active initial gate: `G0`
- terminal gate: `G7`
- canonical remote profile: `gpu4090d`
- canonical repo dir: `/home/zyy/code/wsovvis_runner`
- canonical wrapper: `bash tools/remote_verify_wsovvis.sh`

See:
- `DEPLOYMENT_INSTRUCTIONS.md`
- `prompts/02_project_specific_first_run_prompt.md`
