#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_stage_d11_canonical_replay.sh [--repo-root <path>] [--python-bin <path>] [--output-root <path>] [--config-path <path>] [--stagec-artifact-root <path>] [--on-weight <float>] [--pilot-scale <float>] [--keep-output]

Replay the N11 canonical validation sequence in order:
  1) N10 layered fast gate: tools/run_stage_d10_layered_fast_gate.sh
  2) Real helper entrypoint pilot-mode smoke: tools/run_stage_d9_smoke_helper.py --on-mode pilot

Defaults:
  - --pilot-scale defaults to 1e-6.
  - ON-path weight is helper default unless --on-weight is provided.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python"
output_root=""
config_path=""
stagec_artifact_root=""
on_weight=""
pilot_scale="1e-6"
keep_output=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --repo-root" >&2; exit 2; }
      repo_root="$2"
      shift 2
      ;;
    --python-bin)
      [[ $# -ge 2 ]] || { echo "Missing value for --python-bin" >&2; exit 2; }
      python_bin="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --output-root" >&2; exit 2; }
      output_root="$2"
      shift 2
      ;;
    --config-path)
      [[ $# -ge 2 ]] || { echo "Missing value for --config-path" >&2; exit 2; }
      config_path="$2"
      shift 2
      ;;
    --stagec-artifact-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --stagec-artifact-root" >&2; exit 2; }
      stagec_artifact_root="$2"
      shift 2
      ;;
    --on-weight)
      [[ $# -ge 2 ]] || { echo "Missing value for --on-weight" >&2; exit 2; }
      on_weight="$2"
      shift 2
      ;;
    --pilot-scale)
      [[ $# -ge 2 ]] || { echo "Missing value for --pilot-scale" >&2; exit 2; }
      pilot_scale="$2"
      shift 2
      ;;
    --keep-output)
      keep_output=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

run() {
  echo "RUN: $*"
  "$@"
}

cd "$repo_root"

layered_cmd=(
  bash tools/run_stage_d10_layered_fast_gate.sh
  --repo-root "$repo_root"
  --python-bin "$python_bin"
)

pilot_cmd=(
  "$python_bin" tools/run_stage_d9_smoke_helper.py
  --repo-root "$repo_root"
  --on-mode pilot
  --pilot-scale "$pilot_scale"
)
if [[ -n "$output_root" ]]; then
  pilot_cmd+=(--output-root "$output_root")
fi
if [[ -n "$config_path" ]]; then
  pilot_cmd+=(--config-path "$config_path")
fi
if [[ -n "$stagec_artifact_root" ]]; then
  pilot_cmd+=(--stagec-artifact-root "$stagec_artifact_root")
fi
if [[ -n "$on_weight" ]]; then
  pilot_cmd+=(--on-weight "$on_weight")
fi
if [[ "$keep_output" -eq 1 ]]; then
  pilot_cmd+=(--keep-output)
fi

echo "D11_CANONICAL_REPLAY_STAGE=n10_layered_fast_gate START"
run "${layered_cmd[@]}"
echo "D11_CANONICAL_REPLAY_STAGE=n10_layered_fast_gate PASS"

echo "D11_CANONICAL_REPLAY_STAGE=pilot_helper_smoke START"
run "${pilot_cmd[@]}"
echo "D11_CANONICAL_REPLAY_STAGE=pilot_helper_smoke PASS"

echo "D11_CANONICAL_REPLAY=PASS"
