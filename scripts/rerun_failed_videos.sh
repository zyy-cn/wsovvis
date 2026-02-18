#!/usr/bin/env bash
set -u  # 不用 -e，避免单个视频失败就退出

LIST=${1:?need a video list file, e.g. failed_all.txt}
GPU=${2:-0}

CFG="configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml"
WEIGHTS="pretrain/videocutler_m2f_rn50.pth"

IMG_ROOT="data/LV-VIS/train/JPEGImages"
OUT_ROOT="outputs/videocutler_lvvis_png/train/masks"
LOG_DIR="outputs/videocutler_lvvis_png/train/logs_rerun"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 稳定显存：降测试分辨率（你可以更保守一些）
MIN_TEST=480
MAX_TEST=864

# 先用较低阈值跑一遍，提高覆盖率；你也可以对 empty 单独用更低阈值
CONF1=0.6
CONF2=0.6

while read -r vid; do
  [ -z "${vid}" ] && continue

  echo "[rerun] ${vid}"

  # 清理旧输出（避免混合残留）
  rm -rf "${OUT_ROOT}/${vid}"
  mkdir -p "${OUT_ROOT}/${vid}"

  # 第一遍：常规阈值 + 降分辨率
  python third_party/CutLER/videocutler/demo_video/demo_masks_only.py \
    --config-file "${CFG}" \
    --input "${IMG_ROOT}/${vid}/*.jpg" \
    --output "${OUT_ROOT}" \
    --save-frames False \
    --save-masks True \
    --confidence-threshold "${CONF1}" \
    --opts MODEL.WEIGHTS "${WEIGHTS}" INPUT.MIN_SIZE_TEST ${MIN_TEST} INPUT.MAX_SIZE_TEST ${MAX_TEST} \
    > "${LOG_DIR}/${vid}.log" 2>&1

  # 如果没产出 mask_*.png，再用更低阈值重试一次（主要针对 empty）
  if ! ls "${OUT_ROOT}/${vid}"/mask_*.png >/dev/null 2>&1; then
    echo "[rerun] ${vid} -> retry with conf=${CONF2}" | tee -a "${LOG_DIR}/${vid}.log"
    python third_party/CutLER/videocutler/demo_video/demo_masks_only.py \
      --config-file "${CFG}" \
      --input "${IMG_ROOT}/${vid}/*.jpg" \
      --output "${OUT_ROOT}" \
      --save-frames False \
      --save-masks True \
      --confidence-threshold "${CONF2}" \
      --opts MODEL.WEIGHTS "${WEIGHTS}" INPUT.MIN_SIZE_TEST ${MIN_TEST} INPUT.MAX_SIZE_TEST ${MAX_TEST} \
      >> "${LOG_DIR}/${vid}.log" 2>&1
  fi

  # 最终仍无 mask：记下（供后续再降分辨率或人工处理）
  if ! ls "${OUT_ROOT}/${vid}"/mask_*.png >/dev/null 2>&1; then
    echo "${vid}" >> "${LOG_DIR}/still_failed.txt"
  fi

done < "${LIST}"

echo "[rerun] done. see ${LOG_DIR}/still_failed.txt (if exists)"
