# Session Handoff: Stage C1.r1

## Current status snapshot
- Stage C1 (MIL-first offline baseline attribution) is PASS.
- Canonical PASS evidence: `codex/2026030105_progress_sync_and_stagec0_consumer_loader_taskpack_meta_tier1_2/16_output.txt` (`6 passed`, branch/commit match confirmed).
- Next target: **Stage C1.r1 (real-artifact smoke + baseline diagnostics hardening)**.

## Already stable
- Bridge pipeline through P3.1c.2 is stable (real Stage B + sidecar -> bridge-input -> adapter/export -> validator).
- Stage C0 loader/offline data-plane is stable and canonically validated (`10/10` PASS).
- Stage C1 MIL-first offline baseline is implemented and canonically validated.
- Canonical remote validation discipline is established (branch/commit equality required for PASS claims).

## Workflow/environment requirements (authoritative summary)
Use `codex/WSOVVIS_CODEX_WORKFLOW_README.md` as source of truth.

- Canonical remote validation target:
  - host alias: `gpu4090d`
  - repo path: `/home/zyy/code/wsovvis_runner`
- `remote_verify` branch/commit rule:
  - `tools/remote_verify_wsovvis.sh` validates `origin/<branch>` state on remote; unpushed local edits are not included.
  - Require: intended local/pushed commit hash equals remote checked-out HEAD hash.
- Remote import environment requirement (VNext/CutLER/detectron2 path wiring):
  - `export PYTHONPATH="$PWD/third_party/VNext:$PWD/third_party/CutLER:$PWD:$PYTHONPATH"`
  - Apply in `--env-cmd` (and duplicate in `--cmd` if command context may override env).
- Validation strategy:
  - local-first checks when dependencies exist
  - canonical remote fallback/expected path when local deps are missing or real-run data is remote
  - always report exact commands, paths, and outcomes
- Real-run path handling:
  - if repo-relative artifact paths do not exist in remote clone, use remote absolute equivalent paths and report exact path used.

## Recommended Stage C1.r1 scope
- Real export artifact smoke using Stage C1 CLI/pipeline on real artifact inputs.
- Output artifact checks:
  - `track_scores.jsonl`
  - `per_video_summary.json`
  - `run_summary.json`
- Baseline diagnostics hardening:
  - score distribution checks
  - non-empty scored video checks
  - track-count distribution checks
  - deterministic ordering/identity checks under real-artifact conditions
- Keep strict non-goals:
  - no EM baseline yet
  - no OT/Sinkhorn
  - no training/loss/Stage D loop integration
  - no contract/schema redesign

## Suggested branch for next implementation
- `codex/stagec1-r1-real-artifact-smoke-v1`

## Pointers for next session
- Progress anchor: `docs/PROJECT_PROGRESS.md`
- Stage C0 task pack: `codex/2026030106_stagec0_consumer_loader_offline_dataplane_tier2/`
- Stage C1 task pack: `codex/2026030107_stagec1_attribution_baseline_offline_milfirst_tier2/`
- Key PASS outputs:
  - `codex/2026030105_progress_sync_and_stagec0_consumer_loader_taskpack_meta_tier1_2/10_output.txt`
  - `codex/2026030105_progress_sync_and_stagec0_consumer_loader_taskpack_meta_tier1_2/16_output.txt`
- Authoritative workflow docs:
  - `codex/START_HERE.md`
  - `codex/WSOVVIS_CODEX_WORKFLOW_README.md`
  - `codex/common/WORKFLOW_TIERS.md`
  - `codex/common/CODEX_COLLABORATION_RULES.md`
  - `codex/common/REVIEW_CHECKLISTS.md`
  - `codex/specs/doc_only_and_impl_prompt_contract.md`
