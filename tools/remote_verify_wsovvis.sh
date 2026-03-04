#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/remote_verify_wsovvis.sh \
    --remote <ssh-host-alias> \
    --repo-dir <remote-runner-path> \
    --branch <branch-name> \
    --env-cmd "<shell command>" \
    --cmd "<shell command>" \
    [--clone-url <git-url>] \
    [--allow-suspicious-repo-dir] \
    [--keep-untracked]
USAGE
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_suspicious_repo_dir() {
  local path_lower
  path_lower=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  [[ "$path_lower" == *live* || "$path_lower" == *pycharm* ]]
}

remote=""
repo_dir=""
branch=""
env_cmd=""
verify_cmd=""
clone_url=""
allow_suspicious_repo_dir=0
keep_untracked=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      [[ $# -ge 2 ]] || die "Missing value for --remote"
      remote="$2"
      shift 2
      ;;
    --repo-dir)
      [[ $# -ge 2 ]] || die "Missing value for --repo-dir"
      repo_dir="$2"
      shift 2
      ;;
    --branch)
      [[ $# -ge 2 ]] || die "Missing value for --branch"
      branch="$2"
      shift 2
      ;;
    --env-cmd)
      [[ $# -ge 2 ]] || die "Missing value for --env-cmd"
      env_cmd="$2"
      shift 2
      ;;
    --cmd)
      [[ $# -ge 2 ]] || die "Missing value for --cmd"
      verify_cmd="$2"
      shift 2
      ;;
    --clone-url)
      [[ $# -ge 2 ]] || die "Missing value for --clone-url"
      clone_url="$2"
      shift 2
      ;;
    --allow-suspicious-repo-dir)
      allow_suspicious_repo_dir=1
      shift
      ;;
    --keep-untracked)
      keep_untracked=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ -n "$remote" ]] || die "--remote is required"
[[ -n "$repo_dir" ]] || die "--repo-dir is required"
[[ -n "$branch" ]] || die "--branch is required"
[[ -n "$env_cmd" ]] || die "--env-cmd is required"
[[ -n "$verify_cmd" ]] || die "--cmd is required"

if [[ "$allow_suspicious_repo_dir" -ne 1 ]] && is_suspicious_repo_dir "$repo_dir"; then
  die "Refusing suspicious --repo-dir '$repo_dir'. Use --allow-suspicious-repo-dir to override."
fi

if [[ -z "$clone_url" ]]; then
  if ! clone_url=$(git remote get-url origin 2>/dev/null); then
    die "--clone-url not provided and failed to resolve local origin URL"
  fi
fi

echo "=== Remote Verify Config ==="
echo "remote     : $remote"
echo "repo-dir   : $repo_dir"
echo "branch     : $branch"
echo "clone-url  : $clone_url"
echo "env-cmd    : $env_cmd"
echo "cmd        : $verify_cmd"
echo "============================"

encode_b64() {
  printf '%s' "$1" | base64 | tr -d '\n'
}

repo_dir_b64=$(encode_b64 "$repo_dir")
branch_b64=$(encode_b64 "$branch")
clone_url_b64=$(encode_b64 "$clone_url")
env_cmd_b64=$(encode_b64 "$env_cmd")
verify_cmd_b64=$(encode_b64 "$verify_cmd")

set +e
ssh "$remote" bash -s -- "$repo_dir_b64" "$branch_b64" "$clone_url_b64" "$env_cmd_b64" "$verify_cmd_b64" "$keep_untracked" <<'REMOTE_BLOCK'
set -euo pipefail

decode_b64() {
  printf '%s' "$1" | base64 --decode 2>/dev/null || printf '%s' "$1" | base64 -d
}

resolve_conda_sh() {
  local candidates=(
    "$HOME/software/miniconda3"
    "$HOME/miniconda3"
    "$HOME/anaconda3"
    "/opt/conda"
    "/usr/share/miniconda"
  )
  local base_path=""
  local conda_sh=""

  for base_path in "${candidates[@]}"; do
    conda_sh="$base_path/etc/profile.d/conda.sh"
    if [[ -f "$conda_sh" ]]; then
      printf '%s\n' "$conda_sh"
      return 0
    fi
  done

  return 1
}

repo_dir="$(decode_b64 "$1")"
branch="$(decode_b64 "$2")"
clone_url="$(decode_b64 "$3")"
env_cmd="$(decode_b64 "$4")"
verify_cmd="$(decode_b64 "$5")"
keep_untracked="${6-0}"

if [[ -e "$repo_dir" && ! -d "$repo_dir" ]]; then
  echo "ERROR: repo-dir exists but is not a directory: $repo_dir" >&2
  exit 2
fi

if [[ ! -d "$repo_dir/.git" ]]; then
  if [[ -d "$repo_dir" ]]; then
    echo "ERROR: repo-dir exists but is not a git repo: $repo_dir" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$repo_dir")"
  git clone "$clone_url" "$repo_dir"
fi

cd "$repo_dir"
git remote set-url origin "$clone_url"
git fetch origin
git checkout -B "$branch" "origin/$branch"
git reset --hard "origin/$branch"
if [[ "$keep_untracked" != "1" ]]; then
  git clean -fd
fi

conda_sh="$(resolve_conda_sh)" || {
  echo "ERROR: unable to resolve conda.sh from candidates: $HOME/software/miniconda3, $HOME/miniconda3, $HOME/anaconda3, /opt/conda, /usr/share/miniconda" >&2
  exit 3
}
conda_base="$(dirname "$(dirname "$(dirname "$conda_sh")")")"
# shellcheck disable=SC1090
source "$conda_sh"
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found after sourcing $conda_sh" >&2
  exit 3
fi
conda activate wsovvis
echo "CONDA_BASE=$conda_base"
echo "CONDA_SH=$conda_sh"
echo "PY=$(command -v python)"

eval "$env_cmd"
eval "$verify_cmd"
REMOTE_BLOCK
rc=$?
set -e

if [[ $rc -eq 0 ]]; then
  echo "PASS: remote verification succeeded"
else
  echo "FAIL: remote verification failed (exit=$rc)" >&2
fi

echo "remote     : $remote"
echo "repo-dir   : $repo_dir"
echo "branch     : $branch"
echo "cmd        : $verify_cmd"

exit "$rc"
