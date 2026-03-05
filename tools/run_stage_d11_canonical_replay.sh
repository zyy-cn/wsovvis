#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/run_stage_d11_canonical_replay.sh [--repo-root <path>] [--python-bin <path>] [--output-root <path>] [--config-path <path>] [--stagec-artifact-root <path>] [--on-weight <float>] [--pilot-scale <float>] [--keep-output] [--bootstrap-link-check] [--bootstrap-link-fix] [--bootstrap-runner-root <path>]

Replay the N11 canonical validation sequence in order:
  0) Optional bootstrap preflight (opt-in): tools/check_canonical_runner_bootstrap_links.py
  1) N10 layered fast gate: tools/run_stage_d10_layered_fast_gate.sh
  2) Real helper entrypoint pilot-mode smoke: tools/run_stage_d9_smoke_helper.py --on-mode pilot

Defaults:
  - --pilot-scale defaults to 1e-6.
  - ON-path weight is helper default unless --on-weight is provided.
  - Bootstrap preflight is disabled unless explicitly requested.
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
bootstrap_link_check=0
bootstrap_link_fix=0
bootstrap_runner_root=""

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
    --bootstrap-link-check)
      bootstrap_link_check=1
      shift
      ;;
    --bootstrap-link-fix)
      bootstrap_link_fix=1
      shift
      ;;
    --bootstrap-runner-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --bootstrap-runner-root" >&2; exit 2; }
      bootstrap_runner_root="$2"
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

bootstrap_cmd_base=(
  "$python_bin" tools/check_canonical_runner_bootstrap_links.py
)
if [[ -n "$bootstrap_runner_root" ]]; then
  bootstrap_cmd_base+=(--runner-root "$bootstrap_runner_root")
fi

run_bootstrap_stage() {
  local stage_label="$1"
  shift
  local tmp_log
  tmp_log="$(mktemp)"

  echo "D11_CANONICAL_REPLAY_STAGE=${stage_label} START"
  echo "RUN: ${bootstrap_cmd_base[*]} $*"
  set +e
  "${bootstrap_cmd_base[@]}" "$@" 2>&1 | tee "$tmp_log"
  local cmd_rc=${PIPESTATUS[0]}
  set -e

  if grep -F -q $'\tSKIPPED\t' "$tmp_log"; then
    echo "ERROR: bootstrap link preflight reported SKIPPED status (risky/non-symlink path). Resolve manually and rerun with --bootstrap-link-check before replay." >&2
    rm -f "$tmp_log"
    exit 1
  fi
  if [[ "$cmd_rc" -ne 0 ]]; then
    echo "ERROR: bootstrap link preflight stage '${stage_label}' failed. Fix reported link issues and retry." >&2
    rm -f "$tmp_log"
    exit "$cmd_rc"
  fi

  rm -f "$tmp_log"
  echo "D11_CANONICAL_REPLAY_STAGE=${stage_label} PASS"
}

if [[ "$bootstrap_link_fix" -eq 1 ]]; then
  run_bootstrap_stage "bootstrap_link_preflight_fix" --fix
  run_bootstrap_stage "bootstrap_link_preflight_recheck" --check
elif [[ "$bootstrap_link_check" -eq 1 ]]; then
  run_bootstrap_stage "bootstrap_link_preflight_check" --check
fi

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
