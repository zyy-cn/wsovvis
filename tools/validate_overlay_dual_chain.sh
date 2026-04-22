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

LEGACY_STAGE_SCOPE="${LEGACY_STAGE_SCOPE:-prealign_base_aug}"
RESERVOIR_STAGE_SCOPE="${RESERVOIR_STAGE_SCOPE:-prealign_base_aug}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/codex/outputs/G8_inference_and_eval/overlay_validate_dual_${TS}}"
LOG_DIR="$OUT_ROOT/_validate_logs"

mkdir -p "$LOG_DIR"

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] DEVICE=$DEVICE"
echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] RUN_SMOKE=$RUN_SMOKE"
echo "[INFO] LEGACY_STAGE_SCOPE=$LEGACY_STAGE_SCOPE"
echo "[INFO] RESERVOIR_STAGE_SCOPE=$RESERVOIR_STAGE_SCOPE"

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

run_train() {
  local pipeline="$1"
  local stage_scope="$2"
  local run_root="$3"
  local stdout_log="$4"
  local stderr_log="$5"

  echo "[STEP] train smoke: pipeline=$pipeline stage_scope=$stage_scope"
  python "$REPO_ROOT/videocutler/run_stageb_train.py" \
    --exp_name "validate_${pipeline}" \
    --output_root "$run_root" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --smoke \
    --dataset_name lvvis_train_base \
    --trajectory_source_branch mainline \
    --pipeline "$pipeline" \
    --stage_scope "$stage_scope" \
    --smoke_max_trajectories "$SMOKE_MAX_TRAJ" \
    --prealign_epochs 1 \
    --base_epochs 1 \
    --aug_epochs 1 \
    --log_every 1 \
    > "$stdout_log" 2> "$stderr_log"

  local summary="$run_root/train/pipeline_train_summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "[ERROR] train summary missing: $summary" >&2
    exit 3
  fi

  python - <<PY
import json, pathlib
p = pathlib.Path("$summary")
obj = json.loads(p.read_text(encoding="utf-8"))
assert obj["pipeline"] == "$pipeline"
assert obj["stage_scope"] == "$stage_scope"
assert "stages" in obj
assert "prealign" in obj["stages"]
if "$stage_scope" != "prealign_only":
    # legacy path writes 'softem'; reservoir path may split base/aug in its stage payloads
    assert any(k in obj["stages"] for k in ("softem", "softem_base", "softem_aug")), obj["stages"].keys()
print("TRAIN_SUMMARY_OK", "$pipeline")
PY

  echo "TRAIN_SMOKE_OK $pipeline"
}

run_test() {
  local pipeline="$1"
  local stage_scope="$2"
  local run_root="$3"
  local stdout_log="$4"
  local stderr_log="$5"

  echo "[STEP] test smoke: pipeline=$pipeline stage_scope=$stage_scope"
  python "$REPO_ROOT/videocutler/run_stageb_test.py" \
    --exp_name "validate_${pipeline}" \
    --output_root "$run_root" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --smoke \
    --pipeline "$pipeline" \
    --stage_scope "$stage_scope" \
    --dataset_name lvvis_val \
    --benchmark lvvis \
    --metrics_profile default \
    --logit_chunk_size "$LOGIT_CHUNK_SIZE" \
    > "$stdout_log" 2> "$stderr_log"

  local summary="$run_root/final_summary.json"
  if [[ ! -f "$summary" ]]; then
    echo "[ERROR] final summary missing: $summary" >&2
    exit 4
  fi

  python - <<PY
import json, pathlib
p = pathlib.Path("$summary")
obj = json.loads(p.read_text(encoding="utf-8"))
assert obj["pipeline"] == "$pipeline"
assert obj["stage_scope"] == "$stage_scope"
assert "train" in obj
assert "gt_attribution_rank" in obj
assert "external_eval" in obj
print("FINAL_SUMMARY_OK", "$pipeline")
PY

  echo "TEST_SMOKE_OK $pipeline"
}

LEGACY_ROOT="$OUT_ROOT/legacy"
RESERVOIR_ROOT="$OUT_ROOT/reservoir_v1"

mkdir -p "$LEGACY_ROOT" "$RESERVOIR_ROOT"

run_train "legacy" "$LEGACY_STAGE_SCOPE" \
  "$LEGACY_ROOT" \
  "$LOG_DIR/legacy_train_stdout.txt" \
  "$LOG_DIR/legacy_train_stderr.txt"

run_test "legacy" "$LEGACY_STAGE_SCOPE" \
  "$LEGACY_ROOT" \
  "$LOG_DIR/legacy_test_stdout.txt" \
  "$LOG_DIR/legacy_test_stderr.txt"

run_train "reservoir_v1" "$RESERVOIR_STAGE_SCOPE" \
  "$RESERVOIR_ROOT" \
  "$LOG_DIR/reservoir_train_stdout.txt" \
  "$LOG_DIR/reservoir_train_stderr.txt"

run_test "reservoir_v1" "$RESERVOIR_STAGE_SCOPE" \
  "$RESERVOIR_ROOT" \
  "$LOG_DIR/reservoir_test_stdout.txt" \
  "$LOG_DIR/reservoir_test_stderr.txt"

echo "[STEP] dual-chain compare summary"
python - <<PY
import json, pathlib
legacy = json.loads(pathlib.Path("$LEGACY_ROOT/final_summary.json").read_text(encoding="utf-8"))
reservoir = json.loads(pathlib.Path("$RESERVOIR_ROOT/final_summary.json").read_text(encoding="utf-8"))

def safe_get(d, *ks):
    cur = d
    for k in ks:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

rows = {
    "legacy.pipeline": legacy.get("pipeline"),
    "legacy.stage_scope": legacy.get("stage_scope"),
    "reservoir.pipeline": reservoir.get("pipeline"),
    "reservoir.stage_scope": reservoir.get("stage_scope"),
    "legacy.external.AP": safe_get(legacy, "external_eval", "AP"),
    "reservoir.external.AP": safe_get(reservoir, "external_eval", "AP"),
}
for k,v in rows.items():
    print(f"{k}={v}")
PY

echo "[DONE] dual-chain overlay validation passed"
echo "[DONE] logs: $LOG_DIR"
echo "[DONE] output_root: $OUT_ROOT"
echo "[DONE] legacy_root: $LEGACY_ROOT"
echo "[DONE] reservoir_root: $RESERVOIR_ROOT"