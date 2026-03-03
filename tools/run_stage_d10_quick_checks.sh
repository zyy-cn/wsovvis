#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_stage_d10_quick_checks.sh [--repo-root <path>] [--python-bin <path>] [--on-mode <zero|nonzero|pilot>] [--on-weight <float>] [--pilot-scale <float>]

Runs Stage D10/D11 helper quick checks:
  1) helper --help
  2) helper --dry-run (zero compatibility sentinel, constant nonzero mode, or gradient-coupled pilot mode)
  3) GPU-free pytest: tests/test_stage_d9_smoke_helper_v1.py
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python"
on_mode="zero"
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
    --on-mode)
      [[ $# -ge 2 ]] || { echo "Missing value for --on-mode" >&2; exit 2; }
      on_mode="$2"
      if [[ "$on_mode" != "zero" && "$on_mode" != "nonzero" && "$on_mode" != "pilot" ]]; then
        echo "Invalid --on-mode '$on_mode' (expected: zero|nonzero|pilot)" >&2
        exit 2
      fi
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

helper_dry_run_cmd=(
  "$python_bin" tools/run_stage_d9_smoke_helper.py --repo-root "$repo_root" --dry-run --on-mode "$on_mode"
)
if [[ -n "$on_weight" ]]; then
  helper_dry_run_cmd+=(--on-weight "$on_weight")
fi
if [[ -n "$pilot_scale" ]]; then
  helper_dry_run_cmd+=(--pilot-scale "$pilot_scale")
fi

run "$python_bin" tools/run_stage_d9_smoke_helper.py --help
run "${helper_dry_run_cmd[@]}"
run "$python_bin" -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
