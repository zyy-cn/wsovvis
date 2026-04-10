This overlay adds a detectron2-side launcher and wrapper for using the native DINOv2 ViT-L register-aware backbone
from `third_party/dinov2/dinov2/models/vision_transformer.py` together with a minimally patched copy of the
original segmentation_m2f ViT-Adapter logic.

Use:
  train_net_video_dinov2_regofficial.py
  configs/imagenet_video/video_mask2former_dinov2_vitl14_reg4_official_cls_agnostic.yaml
