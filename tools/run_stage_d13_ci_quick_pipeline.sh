#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_stage_d13_ci_quick_pipeline.sh [--repo-root <path>] [--python-bin <path>] [--on-weight <float>] [--pilot-scale <float>]

Runs a branch-local CI mirror pipeline in order:
  1) helper fast gate: tools/run_stage_d9_helper_tests_quick.sh
  2) canonical replay wrapper: tools/run_stage_d11_canonical_replay.sh

This script is a lightweight CI/quick-check wiring surface for branches
without a dedicated CI config change.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python"
on_weight=""
pilot_scale=""

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

echo "D13_CI_QUICK_PIPELINE_STAGE=helper_fast_gate START"
run bash tools/run_stage_d9_helper_tests_quick.sh --repo-root "$repo_root" --python-bin "$python_bin"
echo "D13_CI_QUICK_PIPELINE_STAGE=helper_fast_gate PASS"

replay_cmd=(
  bash tools/run_stage_d11_canonical_replay.sh
  --repo-root "$repo_root"
  --python-bin "$python_bin"
)
if [[ -n "$on_weight" ]]; then
  replay_cmd+=(--on-weight "$on_weight")
fi
if [[ -n "$pilot_scale" ]]; then
  replay_cmd+=(--pilot-scale "$pilot_scale")
fi

echo "D13_CI_QUICK_PIPELINE_STAGE=canonical_replay START"
run "${replay_cmd[@]}"
echo "D13_CI_QUICK_PIPELINE_STAGE=canonical_replay PASS"

echo "D13_CI_QUICK_PIPELINE=PASS"
