#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_stage_d10_quick_checks.sh [--repo-root <path>] [--python-bin <path>]

Runs Stage D10/D11 helper quick checks:
  1) helper --help
  2) helper --dry-run
  3) GPU-free pytest: tests/test_stage_d9_smoke_helper_v1.py
USAGE
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="python"

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

run "$python_bin" tools/run_stage_d9_smoke_helper.py --help
run "$python_bin" tools/run_stage_d9_smoke_helper.py --repo-root "$repo_root" --dry-run
run "$python_bin" -m pytest -q tests/test_stage_d9_smoke_helper_v1.py
