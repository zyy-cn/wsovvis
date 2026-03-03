#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_stage_d10_layered_fast_gate.sh [--repo-root <path>] [--python-bin <path>] [--with-pilot-smoke] [--pilot-on-mode <zero|nonzero|pilot>] [--pilot-on-weight <float>] [--pilot-scale <float>]

Runs layered fast-gate validation in order:
  1) helper coverage fast gate (N9): tools/run_stage_d9_helper_tests_quick.sh
  2) optional pilot-capable quick-check smoke (N7): tools/run_stage_d10_quick_checks.sh

Defaults:
  - Step 2 is disabled unless --with-pilot-smoke is provided.
  - --pilot-on-mode defaults to pilot when Step 2 is enabled.
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python"
with_pilot_smoke=0
pilot_on_mode="pilot"
pilot_on_weight=""
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
    --with-pilot-smoke)
      with_pilot_smoke=1
      shift
      ;;
    --pilot-on-mode)
      [[ $# -ge 2 ]] || { echo "Missing value for --pilot-on-mode" >&2; exit 2; }
      pilot_on_mode="$2"
      if [[ "$pilot_on_mode" != "zero" && "$pilot_on_mode" != "nonzero" && "$pilot_on_mode" != "pilot" ]]; then
        echo "Invalid --pilot-on-mode '$pilot_on_mode' (expected: zero|nonzero|pilot)" >&2
        exit 2
      fi
      shift 2
      ;;
    --pilot-on-weight)
      [[ $# -ge 2 ]] || { echo "Missing value for --pilot-on-weight" >&2; exit 2; }
      pilot_on_weight="$2"
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

if [[ "$with_pilot_smoke" -eq 0 ]]; then
  if [[ -n "$pilot_on_weight" || -n "$pilot_scale" || "$pilot_on_mode" != "pilot" ]]; then
    echo "Pilot quick-check args require --with-pilot-smoke" >&2
    exit 2
  fi
fi

if [[ "$pilot_on_mode" != "pilot" && -n "$pilot_scale" ]]; then
  echo "--pilot-scale is only valid when --pilot-on-mode pilot" >&2
  exit 2
fi

run() {
  echo "RUN: $*"
  "$@"
}

cd "$repo_root"

echo "D10_LAYERED_FAST_GATE_STAGE=helper_coverage START"
run bash tools/run_stage_d9_helper_tests_quick.sh --repo-root "$repo_root" --python-bin "$python_bin"
echo "D10_LAYERED_FAST_GATE_STAGE=helper_coverage PASS"

if [[ "$with_pilot_smoke" -eq 1 ]]; then
  quick_cmd=(
    bash tools/run_stage_d10_quick_checks.sh
    --repo-root "$repo_root"
    --python-bin "$python_bin"
    --on-mode "$pilot_on_mode"
  )
  if [[ -n "$pilot_on_weight" ]]; then
    quick_cmd+=(--on-weight "$pilot_on_weight")
  fi
  if [[ -n "$pilot_scale" ]]; then
    quick_cmd+=(--pilot-scale "$pilot_scale")
  fi

  echo "D10_LAYERED_FAST_GATE_STAGE=pilot_quick_check START"
  run "${quick_cmd[@]}"
  echo "D10_LAYERED_FAST_GATE_STAGE=pilot_quick_check PASS"
else
  echo "D10_LAYERED_FAST_GATE_STAGE=pilot_quick_check SKIP"
fi

echo "D10_LAYERED_FAST_GATE=PASS"
