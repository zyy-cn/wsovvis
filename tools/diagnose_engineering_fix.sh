#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-/mnt/sda/zyy/code/wsovvis}"
RUN_ROOT="${2:-$REPO_ROOT/codex/outputs/G8_inference_and_eval/engineering_fix_smoke}"

cd "$REPO_ROOT"

echo "==================== ENV ===================="
echo "REPO_ROOT=$REPO_ROOT"
echo "RUN_ROOT=$RUN_ROOT"
python -V

echo
echo "==================== FILE PRESENCE ===================="
for f in \
  "videocutler/ext_stageb_ovvis/metrics/collector.py" \
  "videocutler/ext_stageb_ovvis/audit/gt_attribution_rank_audit.py" \
  "videocutler/ext_stageb_ovvis/utils/unknown_metrics.py" \
  "videocutler/ext_stageb_ovvis/algorithms/reservoir_v1.py" \
  "$RUN_ROOT/train/pipeline_train_summary.json" \
  "$RUN_ROOT/final_summary.json"
do
  if [[ -f "$f" ]]; then
    echo "[OK] $f"
  else
    echo "[MISS] $f"
  fi
done

echo
echo "==================== SUMMARY SNAPSHOT ===================="
python - <<PY
import json, pathlib
run_root = pathlib.Path("$RUN_ROOT")
train_p = run_root / "train" / "pipeline_train_summary.json"
final_p = run_root / "final_summary.json"

if train_p.exists():
    train = json.loads(train_p.read_text())
    print("[train] pipeline =", train.get("pipeline"))
    print("[train] stage_scope =", train.get("stage_scope"))
    pre = train.get("stages", {}).get("prealign", {})
    soft = train.get("stages", {}).get("softem", {})
    print("[train] prealign.unknown_metrics =", pre.get("unknown_metrics"))
    print("[train] softem.unknown_metrics =", soft.get("unknown_metrics"))
else:
    print("[train] missing")

if final_p.exists():
    final = json.loads(final_p.read_text())
    gar = final.get("gt_attribution_rank", {}).get("stages", {})
    for k in ["prealign", "softem_base", "softem_aug"]:
        st = gar.get(k, {})
        print(f"[gar] {k}: gt_available_row_count={st.get('gt_available_row_count')} gt_count={st.get('gt_count')}")
    print("[external_eval] =", final.get("external_eval", {}).get("metrics"))
else:
    print("[final] missing")
PY

echo
echo "==================== GT AUDIT CALL SITE ===================="
python - <<'PY'
from pathlib import Path
p = Path("videocutler/ext_stageb_ovvis/metrics/collector.py")
text = p.read_text(encoding="utf-8")
needles = [
    "run_gt_attribution_rank_audit",
    "all_gt_only",
    "generate_sidecars_if_missing",
    "gt_sidecar",
]
for n in needles:
    print(f"\n---- grep: {n} ----")
    for i, line in enumerate(text.splitlines(), 1):
        if n in line:
            lo = max(1, i-3); hi = min(len(text.splitlines()), i+4)
            for j in range(lo, hi+1):
                print(f"{j:04d}: {text.splitlines()[j-1]}")
PY

echo
echo "==================== GT AUDIT DEFAULTS ===================="
python - <<'PY'
from pathlib import Path
p = Path("videocutler/ext_stageb_ovvis/audit/gt_attribution_rank_audit.py")
text = p.read_text(encoding="utf-8")
needles = [
    "GTAttributionRankAuditConfig",
    "all_gt_generate_sidecars_if_missing",
    "_load_or_generate_gt_sidecar_lookup_cached",
    "gt_available_row_count",
]
for n in needles:
    print(f"\n---- grep: {n} ----")
    for i, line in enumerate(text.splitlines(), 1):
        if n in line:
            lo = max(1, i-4); hi = min(len(text.splitlines()), i+8)
            for j in range(lo, hi+1):
                print(f"{j:04d}: {text.splitlines()[j-1]}")
PY

echo
echo "==================== UNKNOWN METRICS IMPLEMENTATION ===================="
python - <<'PY'
from pathlib import Path
p = Path("videocutler/ext_stageb_ovvis/utils/unknown_metrics.py")
text = p.read_text(encoding="utf-8")
needles = [
    "class UnknownMetricsAccumulator",
    "update_prealign",
    "update_base",
    "finalize",
    "unknown_retention_rate",
]
for n in needles:
    print(f"\n---- grep: {n} ----")
    for i, line in enumerate(text.splitlines(), 1):
        if n in line:
            lo = max(1, i-4); hi = min(len(text.splitlines()), i+12)
            for j in range(lo, hi+1):
                print(f"{j:04d}: {text.splitlines()[j-1]}")
PY

echo
echo "==================== UNKNOWN METRICS CALL SITES ===================="
python - <<'PY'
from pathlib import Path
for fp in [
    Path("videocutler/ext_stageb_ovvis/algorithms/reservoir_v1.py"),
    Path("videocutler/ext_stageb_ovvis/pipeline/train_orchestrator.py"),
]:
    text = fp.read_text(encoding="utf-8")
    print(f"\n===== {fp} =====")
    for n in ["UnknownMetricsAccumulator", "update_prealign", "update_base", "finalize", "unknown_metrics"]:
        print(f"\n---- grep: {n} ----")
        for i, line in enumerate(text.splitlines(), 1):
            if n in line:
                lo = max(1, i-3); hi = min(len(text.splitlines()), i+6)
                for j in range(lo, hi+1):
                    print(f"{j:04d}: {text.splitlines()[j-1]}")
PY

echo
echo "==================== RESERVOIR PREALIGN BAG TARGET ===================="
python - <<'PY'
from pathlib import Path
p = Path("videocutler/ext_stageb_ovvis/algorithms/reservoir_v1.py")
text = p.read_text(encoding="utf-8")
needles = [
    "clip_examples[0]",
    "observed_raw_ids",
    "candidate_ids_known",
    "clip_id",
    "bag",
]
for n in needles:
    print(f"\n---- grep: {n} ----")
    for i, line in enumerate(text.splitlines(), 1):
        if n in line:
            lo = max(1, i-5); hi = min(len(text.splitlines()), i+10)
            for j in range(lo, hi+1):
                print(f"{j:04d}: {text.splitlines()[j-1]}")
PY

echo
echo "==================== TRAINABLE SAMPLE SHAPE CHECK ===================="
python - <<'PY'
from pathlib import Path
from videocutler.ext_stageb_ovvis.pipeline.plans import TrainPlan
from videocutler.ext_stageb_ovvis.pipeline.train_orchestrator import _materialize

plan = TrainPlan(
    exp_name="diag_shape",
    output_root=Path("/tmp/diag_shape"),
    device="cuda:0",
    seed=3407,
    smoke=True,
    dataset_name="lvvis_train_base",
    trajectory_source_branch="mainline",
    pipeline="reservoir_v1",
    stage_scope="prealign_base_aug",
    smoke_max_trajectories=8,
    repo_root=Path("/mnt/sda/zyy/code/wsovvis"),
    asset_root=Path("/home/zyy/code/wsovvis_asserts"),
    prealign_epochs=1,
    base_epochs=1,
    aug_epochs=1,
    prealign_learning_rate=1e-4,
    base_learning_rate=1e-4,
    aug_learning_rate=1e-4,
    weight_decay=0.0,
    t_dis_init=1.0,
    lambda_frame=1.0,
    lambda_cov=1.0,
    subset_fraction=None,
    batch_budget=None,
    show_progress=False,
    log_every=1,
    write_runtime_metrics_jsonl=False,
    print_epoch_summary=False,
)

m = _materialize(plan)
print("top_keys =", list(m.keys()))
print("valid_samples =", len(m.get("valid_samples", [])))
if m.get("valid_samples"):
    s = m["valid_samples"][0]
    print("first_valid_keys =", list(s.keys()))
    print("first_valid.clip_id =", s.get("clip_id"))
    print("first_valid.candidate_ids_known[:10] =", (s.get("candidate_ids_known") or [])[:10])
    print("first_valid.observed_raw_ids[:10] =", (s.get("observed_raw_ids") or [])[:10])
PY