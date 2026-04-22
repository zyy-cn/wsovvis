#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"
cd "$REPO_ROOT"
if [[ -x "$REPO_ROOT/tools/bootstrap_repo_assets_from_asserts.sh" ]]; then
  bash "$REPO_ROOT/tools/bootstrap_repo_assets_from_asserts.sh" "$REPO_ROOT" "${ASSERTS_ROOT:-$REPO_ROOT/../wsovvis_asserts}"
fi
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-3407}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_MAX_TRAJ="${SMOKE_MAX_TRAJ:-8}"
LOGIT_CHUNK_SIZE="${LOGIT_CHUNK_SIZE:-64}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/codex/outputs/G8_inference_and_eval/overlay_validate_${TS}}"
LOG_DIR="$OUT_ROOT/_validate_logs"

mkdir -p "$LOG_DIR"

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] DEVICE=$DEVICE"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] RUN_SMOKE=$RUN_SMOKE"

need_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] missing file: $p" >&2
    exit 2
  fi
}

echo "[STEP] checking required overlay files"
need_file "$REPO_ROOT/videocutler/run_stageb_train.py"
need_file "$REPO_ROOT/videocutler/run_stageb_test.py"
need_file "$REPO_ROOT/videocutler/ext_stageb_ovvis/pipeline/plans.py"
need_file "$REPO_ROOT/videocutler/ext_stageb_ovvis/pipeline/train_orchestrator.py"
need_file "$REPO_ROOT/videocutler/ext_stageb_ovvis/pipeline/test_orchestrator.py"
need_file "$REPO_ROOT/videocutler/ext_stageb_ovvis/metrics/collector.py"
need_file "$REPO_ROOT/videocutler/ext_stageb_ovvis/utils/unknown_metrics.py"
need_file "$REPO_ROOT/videocutler/ext_stageb_ovvis/algorithms/reservoir_v1.py"
need_file "$REPO_ROOT/package/reference/g8_metric_collection_policy.json"

echo "[STEP] py_compile"
python -m py_compile \
  "$REPO_ROOT/videocutler/run_stageb_train.py" \
  "$REPO_ROOT/videocutler/run_stageb_test.py" \
  "$REPO_ROOT/videocutler/ext_stageb_ovvis/pipeline/plans.py" \
  "$REPO_ROOT/videocutler/ext_stageb_ovvis/pipeline/train_orchestrator.py" \
  "$REPO_ROOT/videocutler/ext_stageb_ovvis/pipeline/test_orchestrator.py" \
  "$REPO_ROOT/videocutler/ext_stageb_ovvis/metrics/collector.py" \
  "$REPO_ROOT/videocutler/ext_stageb_ovvis/utils/unknown_metrics.py" \
  "$REPO_ROOT/videocutler/ext_stageb_ovvis/algorithms/reservoir_v1.py"

echo "[STEP] import check"
python - <<'PY'
import importlib
mods = [
    "videocutler.run_stageb_train",
    "videocutler.run_stageb_test",
    "videocutler.ext_stageb_ovvis.pipeline.plans",
    "videocutler.ext_stageb_ovvis.pipeline.train_orchestrator",
    "videocutler.ext_stageb_ovvis.pipeline.test_orchestrator",
    "videocutler.ext_stageb_ovvis.metrics.collector",
    "videocutler.ext_stageb_ovvis.utils.unknown_metrics",
    "videocutler.ext_stageb_ovvis.algorithms.reservoir_v1",
]
for m in mods:
    importlib.import_module(m)
print("IMPORT_OK")
PY

echo "[STEP] CLI help check"
python "$REPO_ROOT/videocutler/run_stageb_train.py" --help > "$LOG_DIR/train_help.txt"
python "$REPO_ROOT/videocutler/run_stageb_test.py" --help > "$LOG_DIR/test_help.txt"

echo "STATIC_OK"

if [[ "$RUN_SMOKE" != "1" ]]; then
  echo "[INFO] RUN_SMOKE=0, static validation only."
  exit 0
fi

echo "[STEP] reservoir_v1 train smoke: prealign_base_aug"
python "$REPO_ROOT/videocutler/run_stageb_train.py" \
  --exp_name overlay_validate \
  --output_root "$OUT_ROOT" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --smoke \
  --dataset_name lvvis_train_base \
  --trajectory_source_branch mainline \
  --pipeline reservoir_v1 \
  --stage_scope prealign_base_aug \
  --smoke_max_trajectories "$SMOKE_MAX_TRAJ" \
  --prealign_epochs 1 \
  --base_epochs 1 \
  --aug_epochs 1 \
  --log_every 1 \
  > "$LOG_DIR/train_stdout.txt" 2> "$LOG_DIR/train_stderr.txt"

if [[ ! -f "$OUT_ROOT/train/pipeline_train_summary.json" ]]; then
  echo "[ERROR] train summary missing: $OUT_ROOT/train/pipeline_train_summary.json" >&2
  exit 3
fi

echo "[STEP] validate train summary schema-lite"
python - <<PY
import json, pathlib, sys
p = pathlib.Path("$OUT_ROOT/train/pipeline_train_summary.json")
obj = json.loads(p.read_text(encoding="utf-8"))
assert obj["pipeline"] == "reservoir_v1"
assert obj["stage_scope"] == "prealign_base_aug"
assert "stages" in obj
for key in ("prealign", "softem_base", "softem_aug"):
    assert key in obj["stages"], f"missing stage: {key}"
print("TRAIN_SUMMARY_OK")
PY

echo "TRAIN_SMOKE_OK"

echo "[STEP] reservoir_v1 test smoke"
python "$REPO_ROOT/videocutler/run_stageb_test.py" \
  --exp_name overlay_validate \
  --output_root "$OUT_ROOT" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --smoke \
  --pipeline reservoir_v1 \
  --stage_scope prealign_base_aug \
  --dataset_name lvvis_val \
  --benchmark lvvis \
  --metrics_profile default \
  --logit_chunk_size "$LOGIT_CHUNK_SIZE" \
  > "$LOG_DIR/test_stdout.txt" 2> "$LOG_DIR/test_stderr.txt"

if [[ ! -f "$OUT_ROOT/final_summary.json" ]]; then
  echo "[ERROR] final summary missing: $OUT_ROOT/final_summary.json" >&2
  exit 4
fi

echo "[STEP] validate final summary schema-lite"
python - <<PY
import json, pathlib
p = pathlib.Path("$OUT_ROOT/final_summary.json")
obj = json.loads(p.read_text(encoding="utf-8"))
assert obj["pipeline"] == "reservoir_v1"
assert obj["stage_scope"] == "prealign_base_aug"
assert "train" in obj
assert "gt_attribution_rank" in obj
assert "external_eval" in obj
print("FINAL_SUMMARY_OK")
PY

echo "TEST_SMOKE_OK"
echo "[DONE] overlay validation passed"
echo "[DONE] logs: $LOG_DIR"
echo "[DONE] output_root: $OUT_ROOT"