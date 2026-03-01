#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-runs/wsovvis_seqformer/18}"
SMALL_N="${SMALL_N:-20}"
MEDIUM_N="${MEDIUM_N:-75}"
WORK_ROOT="${WORK_ROOT:-tmp/stagec3r3a_compare}"
OUT_ROOT="${OUT_ROOT:-/tmp/stagec3r3a_compare}"
SCORER_BACKEND="${SCORER_BACKEND:-labelset_proto_v1}"
LABELSET_KEY="${LABELSET_KEY:-label_set_observed_ids}"
EMPTY_LABELSET_POLICY="${EMPTY_LABELSET_POLICY:-use_all_prototypes}"

emit_json() {
  local marker="$1"
  local payload="$2"
  printf '%s::%s\n' "$marker" "$payload"
}

json_escape() {
  python - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1]))
PY
}

cmd_to_string() {
  local out=""
  local part
  for part in "$@"; do
    printf -v part '%q' "$part"
    out+="$part "
  done
  printf '%s' "${out% }"
}

ensure_runner_symlinks() {
  local p target
  for p in runs outputs; do
    target="../wsovvis_live/${p}/"
    if [ -L "$p" ]; then
      if [ "$(readlink "$p")" != "$target" ]; then
        rm "$p"
        ln -s "$target" "$p"
      fi
    elif [ -e "$p" ]; then
      echo "CONFLICT: $p exists and is not a symlink" >&2
      exit 2
    else
      ln -s "$target" "$p"
    fi
  done
}

build_cohort() {
  local tier="$1"
  local nvideos="$2"
  local cohort_root="$WORK_ROOT/${tier}_${nvideos}"
  local split_root="$cohort_root/export_split"
  local input_root="$cohort_root/inputs"
  local bridge_json="$cohort_root/bridge_input.json"
  local builder_summary="$cohort_root/qa_summary_builder.json"
  local adapter_summary="$cohort_root/qa_summary_adapter.json"

  rm -rf "$cohort_root"
  mkdir -p "$cohort_root" "$input_root"

  local cmd_builder=(
    python tools/build_stageb_bridge_input_from_real_stageb_sidecar_v1.py
    --run-root "$RUN_ROOT"
    --sample-video-limit "$nvideos"
    --output-json "$bridge_json"
    --join-summary-json "$builder_summary"
  )
  "${cmd_builder[@]}"

  local cmd_export=(
    python tools/build_stageb_track_feature_export_from_stageb_bridge_v1.py
    --input-json "$bridge_json"
    --output-split-root "$split_root"
    --overwrite
    --bridge-summary-json "$adapter_summary"
  )
  "${cmd_export[@]}"

  local cmd_validate=(
    python tools/validate_stageb_track_feature_export_v1.py
    --split-root "$split_root"
  )
  "${cmd_validate[@]}"

  python - "$split_root" "$input_root" <<'PY'
import json
import numpy as np
import sys
from pathlib import Path

split = Path(sys.argv[1])
input_root = Path(sys.argv[2])
manifest = json.loads((split / "manifest.v1.json").read_text(encoding="utf-8"))
video_ids = [entry["video_id"] for entry in manifest["videos"]]
emb_dim = int(manifest["embedding_dim"])
labels = [
    {"label_id": 1, "row_index": 0},
    {"label_id": 2, "row_index": 1},
    {"label_id": "3", "row_index": 2},
]
proto = np.zeros((len(labels), emb_dim), dtype=np.float32)
if emb_dim > 0:
    proto[0, 0] = 1.0
if emb_dim > 1:
    proto[1, 1] = 1.0
if emb_dim > 2:
    proto[2, 2] = 1.0
np.savez(input_root / "prototypes.npz", prototypes=proto)
manifest_obj = {
    "schema_name": "wsovvis.stagec.label_prototypes.v1",
    "schema_version": "1.0.0",
    "embedding_dim": emb_dim,
    "dtype": "float32",
    "labels": labels,
    "arrays_path": "prototypes.npz",
    "array_key": "prototypes",
}
(input_root / "prototype_manifest.json").write_text(
    json.dumps(manifest_obj, indent=2),
    encoding="utf-8",
)
labelset = {
    "videos": [
        {"video_id": video_id, "label_set_observed_ids": [1, 2]}
        for video_id in video_ids
    ]
}
(input_root / "labelset.json").write_text(json.dumps(labelset, indent=2), encoding="utf-8")
print(json.dumps({
    "video_count": len(video_ids),
    "embedding_dim": emb_dim,
}))
PY

  local cohort_meta
  cohort_meta="$(python - "$tier" "$nvideos" "$RUN_ROOT" "$cohort_root" "$split_root" "$input_root" <<'PY'
import json
import sys
print(json.dumps({
    "tier": sys.argv[1],
    "requested_video_count": int(sys.argv[2]),
    "source_run_root": sys.argv[3],
    "cohort_root": sys.argv[4],
    "split_root": sys.argv[5],
    "input_root": sys.argv[6],
}))
PY
)"
  emit_json "C3R3A_COHORT" "$cohort_meta"
}

run_decoder_once() {
  local tier="$1"
  local nvideos="$2"
  local decoder="$3"
  local run_id="$4"

  local cohort_root="$WORK_ROOT/${tier}_${nvideos}"
  local split_root="$cohort_root/export_split"
  local input_root="$cohort_root/inputs"
  local out_dir="$OUT_ROOT/${tier}_${nvideos}/${decoder}/${run_id}"

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  local cmd=(
    python tools/run_stagec1_mil_baseline_offline.py
    --split-root "$split_root"
    --output-dir "$out_dir"
    --scorer-backend "$SCORER_BACKEND"
    --decoder-backend "$decoder"
    --labelset-json "$input_root/labelset.json"
    --prototype-manifest-json "$input_root/prototype_manifest.json"
    --labelset-key "$LABELSET_KEY"
    --empty-labelset-policy "$EMPTY_LABELSET_POLICY"
  )

  local cmd_str
  cmd_str="$(cmd_to_string "${cmd[@]}")"
  emit_json "C3R3A_RUN_COMMAND" "$(python - "$tier" "$nvideos" "$decoder" "$run_id" "$cmd_str" <<'PY'
import json
import sys
print(json.dumps({
    "tier": sys.argv[1],
    "cohort_size": int(sys.argv[2]),
    "decoder": sys.argv[3],
    "run_id": sys.argv[4],
    "command": sys.argv[5],
}))
PY
)"

  /usr/bin/time -p "${cmd[@]}" >"$out_dir/cli_stdout.log" 2>"$out_dir/time.log"

  local metric_json
  metric_json="$(python - "$tier" "$nvideos" "$decoder" "$run_id" "$out_dir" "$cmd_str" <<'PY'
import json
import pathlib
import re
import sys


def read_time_seconds(path: pathlib.Path) -> float:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^real\s+([0-9.]+)$", text, flags=re.MULTILINE)
    if not m:
        return 0.0
    return float(m.group(1))


def per_video_max(rows, key):
    vals = []
    for row in rows:
        if key in row and row[key] is not None:
            vals.append(float(row[key]))
    if not vals:
        return None
    return max(vals)


tier = sys.argv[1]
cohort_size = int(sys.argv[2])
decoder = sys.argv[3]
run_id = sys.argv[4]
out_dir = pathlib.Path(sys.argv[5])
command = sys.argv[6]

run_summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
per_video = json.loads((out_dir / "per_video_summary.json").read_text(encoding="utf-8"))
decoder_summary = run_summary.get("decoder_summary", {})

payload = {
    "tier": tier,
    "cohort_size": cohort_size,
    "decoder": decoder,
    "run_id": run_id,
    "output_dir": str(out_dir),
    "command": command,
    "wall_seconds": read_time_seconds(out_dir / "time.log"),
    "num_videos_total": int(run_summary.get("num_videos_total", 0)),
    "num_tracks_scored": int(run_summary.get("num_tracks_scored", 0)),
    "coverage_target_count": int(decoder_summary.get("coverage_target_count", 0)),
    "coverage_hit_count": int(decoder_summary.get("coverage_hit_count", 0)),
    "coverage_ratio": float(decoder_summary.get("coverage_ratio", 0.0)),
    "num_tracks_fg": int(decoder_summary.get("num_tracks_fg", 0)),
    "num_tracks_bg": int(decoder_summary.get("num_tracks_bg", 0)),
    "fill_bg_count": int(decoder_summary.get("fill_bg_count", 0)),
    "tie_break_count": int(decoder_summary.get("tie_break_count", 0)),
    "coverage_skip_fg_min_count": int(decoder_summary.get("coverage_skip_fg_min_count", 0)),
    "coverage_skip_bg_gate_count": int(decoder_summary.get("coverage_skip_bg_gate_count", 0)),
    "bg_reason_counts": decoder_summary.get("bg_reason_counts", {}),
    "otlite_iters": decoder_summary.get("otlite_iters"),
    "otlite_col_mass_l1_error_mean": decoder_summary.get("otlite_col_mass_l1_error_mean"),
    "otlite_col_mass_l1_error_max": decoder_summary.get("otlite_col_mass_l1_error_max"),
    "otlite_row_mass_l1_error_mean": decoder_summary.get("otlite_row_mass_l1_error_mean"),
    "otlite_row_mass_l1_error_max": decoder_summary.get("otlite_row_mass_l1_error_max"),
    "otlite_bg_ot_prob_count": decoder_summary.get("otlite_bg_ot_prob_count"),
    "per_video_otlite_col_mass_l1_error_max": per_video_max(per_video, "decoder_otlite_col_mass_l1_error"),
    "per_video_otlite_row_mass_l1_error_max": per_video_max(per_video, "decoder_otlite_row_mass_l1_error"),
    "per_video_otlite_bg_ot_prob_count_max": per_video_max(per_video, "decoder_otlite_bg_ot_prob_count"),
}
print(json.dumps(payload, sort_keys=True))
PY
)"
  emit_json "C3R3A_RUN_METRIC" "$metric_json"
}

run_determinism_check() {
  local tier="$1"
  local nvideos="$2"
  local decoder="$3"

  local base_dir="$OUT_ROOT/${tier}_${nvideos}/${decoder}"
  local run_a="$base_dir/run_a"
  local run_b="$base_dir/run_b"
  local files=(track_scores.jsonl per_video_summary.json run_summary.json)

  local cmp_pass="true"
  local f
  for f in "${files[@]}"; do
    if ! cmp "$run_a/$f" "$run_b/$f" >/dev/null 2>&1; then
      cmp_pass="false"
    fi
  done

  local payload
  payload="$(python - "$tier" "$nvideos" "$decoder" "$run_a" "$run_b" "$cmp_pass" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

tier = sys.argv[1]
cohort_size = int(sys.argv[2])
decoder = sys.argv[3]
run_a = Path(sys.argv[4])
run_b = Path(sys.argv[5])
cmp_pass = sys.argv[6].lower() == "true"
files = ["track_scores.jsonl", "per_video_summary.json", "run_summary.json"]

hashes = {}
for name in files:
    data = (run_a / name).read_bytes()
    hashes[name] = hashlib.sha256(data).hexdigest()

print(json.dumps({
    "tier": tier,
    "cohort_size": cohort_size,
    "decoder": decoder,
    "run_a": str(run_a),
    "run_b": str(run_b),
    "cmp_pass": cmp_pass,
    "hashes_run_a": hashes,
}, sort_keys=True))
PY
)"
  emit_json "C3R3A_DETERMINISM" "$payload"
}

main() {
  ensure_runner_symlinks

  local remote_meta
  remote_meta="$(python - <<'PY'
import json
import os
import subprocess

branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(json.dumps({
    "remote_repo_dir": os.getcwd(),
    "remote_branch": branch,
    "remote_head": head,
    "scorer_backend": os.environ.get("SCORER_BACKEND", "labelset_proto_v1"),
    "run_root": os.environ.get("RUN_ROOT", "runs/wsovvis_seqformer/18"),
}, sort_keys=True))
PY
)"
  emit_json "C3R3A_REMOTE_META" "$remote_meta"

  rm -rf "$WORK_ROOT" "$OUT_ROOT"
  mkdir -p "$WORK_ROOT" "$OUT_ROOT"

  build_cohort "small" "$SMALL_N"
  build_cohort "medium" "$MEDIUM_N"

  local tier n dec
  for tier in small medium; do
    if [ "$tier" = "small" ]; then
      n="$SMALL_N"
    else
      n="$MEDIUM_N"
    fi
    for dec in independent coverage_greedy_v1 otlite_v1; do
      run_decoder_once "$tier" "$n" "$dec" "run_a"
    done
  done

  for dec in independent coverage_greedy_v1 otlite_v1; do
    run_decoder_once "small" "$SMALL_N" "$dec" "run_b"
    run_determinism_check "small" "$SMALL_N" "$dec"
  done

  emit_json "C3R3A_PROTOCOL_DONE" "$(python - <<'PY'
import json
print(json.dumps({"status": "ok"}))
PY
)"
}

main "$@"
