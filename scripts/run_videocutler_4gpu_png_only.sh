#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-train}"   # train or val
WSOVVIS_ROOT="${WSOVVIS_ROOT:-$HOME/code/wsovvis}"

DATA_ROOT="${WSOVVIS_ROOT}/data/LV-VIS/${SPLIT}/JPEGImages"
OUT_ROOT="${WSOVVIS_ROOT}/outputs/videocutler_lvvis_png/${SPLIT}/masks"
LOG_ROOT="${WSOVVIS_ROOT}/outputs/videocutler_lvvis_png/${SPLIT}/logs"

CFG="${WSOVVIS_ROOT}/third_party/CutLER/videocutler/configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml"
WEIGHTS="${WSOVVIS_ROOT}/third_party/CutLER/videocutler/pretrain/videocutler_m2f_rn50.pth"
CONF="${CONF:-0.6}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "${WSOVVIS_ROOT}/third_party/CutLER/videocutler"

for gpu in 0 1 2 3; do
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "[GPU ${gpu}] start $(date)"
    ls -1 "${DATA_ROOT}" | sort | awk -v r="${gpu}" '((NR-1)%4)==r {print}' | while read -r vid; do
      echo "[GPU ${gpu}] ${vid}"
      python demo_video/demo_masks_only.py \
        --config-file "${CFG}" \
        --input "${DATA_ROOT}/${vid}/*.jpg" \
        --output "${OUT_ROOT}/${vid}" \
        --save-masks True \
        --confidence-threshold "${CONF}" \
        --opts MODEL.WEIGHTS "${WEIGHTS}"

      # Keep only mask PNGs
      rm -f "${OUT_ROOT}/${vid}"/*.jpg "${OUT_ROOT}/${vid}"/*.mp4 || true
    done
    echo "[GPU ${gpu}] done $(date)"
  ) > "${LOG_ROOT}/gpu${gpu}.log" 2>&1 &
done

wait
echo "[ALL DONE] masks at: ${OUT_ROOT}"
