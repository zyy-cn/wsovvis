#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"
ASSERTS_ROOT="${2:-$REPO_ROOT/../wsovvis_asserts}"

cd "$REPO_ROOT"

echo "[INFO] REPO_ROOT=$REPO_ROOT"
echo "[INFO] ASSERTS_ROOT=$ASSERTS_ROOT"

need_dir() {
  local p="$1"
  if [[ ! -d "$p" ]]; then
    echo "[ERROR] missing directory: $p" >&2
    exit 2
  fi
}

need_file() {
  local p="$1"
  if [[ ! -f "$p" ]]; then
    echo "[ERROR] missing file: $p" >&2
    exit 3
  fi
}

need_dir "$ASSERTS_ROOT"
need_dir "$ASSERTS_ROOT/exports"
need_dir "$ASSERTS_ROOT/frame_bank"
need_dir "$ASSERTS_ROOT/text_bank"
need_dir "$ASSERTS_ROOT/gt_sidecar_bank"
need_dir "$ASSERTS_ROOT/train"
need_dir "$ASSERTS_ROOT/predictions"

need_file "$ASSERTS_ROOT/exports/lvvis_train_base/trajectory_records.jsonl"
need_file "$ASSERTS_ROOT/exports/lvvis_val/trajectory_records.jsonl"
need_file "$ASSERTS_ROOT/text_bank/text_prototype_records.jsonl"

link_dir() {
  local name="$1"
  local src="$ASSERTS_ROOT/$name"
  local dst="$REPO_ROOT/$name"

  if [[ -L "$dst" ]]; then
    local target
    target="$(readlink -f "$dst" || true)"
    local source
    source="$(readlink -f "$src" || true)"
    if [[ "$target" == "$source" ]]; then
      echo "[OK] symlink already correct: $dst -> $src"
      return 0
    fi
    echo "[INFO] updating symlink: $dst -> $src"
    ln -sfn "$src" "$dst"
    return 0
  fi

  if [[ -e "$dst" ]]; then
    echo "[WARN] path exists and is not a symlink, skip: $dst" >&2
    return 0
  fi

  echo "[INFO] creating symlink: $dst -> $src"
  ln -s "$src" "$dst"
}

for name in exports frame_bank text_bank gt_sidecar_bank train predictions; do
  link_dir "$name"
done

echo "[STEP] sanity check repo-visible assets"
need_file "$REPO_ROOT/exports/lvvis_train_base/trajectory_records.jsonl"
need_file "$REPO_ROOT/exports/lvvis_val/trajectory_records.jsonl"
need_file "$REPO_ROOT/text_bank/text_prototype_records.jsonl"

echo "[DONE] asset bootstrap complete"
