# regofficial overlay audit

This overlay is built against the latest uploaded snapshot and assumes the snapshot's `third_party/dinov2` already has native register-token support.

## Directly reused from original project code (verbatim copy only to avoid package-init side effects)

These files are copied verbatim from the snapshot because importing them through `third_party.dinov2.dinov2.eval.segmentation_m2f...` would execute `segmentation_m2f/__init__.py` and pull `mmseg` into the runtime. Their algorithmic logic is unchanged.

- `videocutler/ext_dinov2_regofficial/vendor/drop_path.py`
  - source: `third_party/dinov2/dinov2/eval/segmentation_m2f/models/backbones/drop_path.py`
  - effect on algorithm: none
- `videocutler/ext_dinov2_regofficial/vendor/ops/modules/ms_deform_attn.py`
  - source: `third_party/dinov2/dinov2/eval/segmentation_m2f/ops/modules/ms_deform_attn.py`
  - effect on algorithm: none
- `videocutler/ext_dinov2_regofficial/vendor/adapter_modules.py`
  - source: `third_party/dinov2/dinov2/eval/segmentation_m2f/models/backbones/adapter_modules.py`
  - change: only the import path is rewritten from relative `...ops.modules` to local vendor copy
  - effect on algorithm: none

## Directly imported from original project code

These are not copied; the overlay imports them from the snapshot's native DINOv2 implementation.

- `dinov2.models.vision_transformer.vit_large`
- the full native register-aware `DinoVisionTransformer` implementation behind it

Effect on algorithm:
- backbone architecture is the snapshot's native official DINOv2 ViT-L register-aware backbone
- register-token semantics are those of the native implementation

## Newly added minimal files

- `config_dinov2_regofficial.py`
  - purpose: adds a separate detectron2 config namespace without touching original configs
  - effect on algorithm: none
- `d2_dinov2_vit_adapter_regofficial.py`
  - purpose: detectron2 `Backbone` wrapper returning `res2/res3/res4/res5`
  - effect on algorithm: none; interface only
- `train_net_video_dinov2_regofficial.py`
  - purpose: separate launcher so original launcher stays untouched
  - effect on algorithm: none; launcher only
- `video_mask2former_dinov2_vitl14_reg4_official_cls_agnostic.yaml`
  - purpose: separate config entrypoint
  - effect on algorithm: only config values; no code-path change beyond selecting this backbone
- `adapter_modules_prefix.py`
  - purpose: minimal replacement for `InteractionBlockWithCls` that treats the prefix length as `1 + num_register_tokens` instead of hard-coding `1`
  - effect on algorithm: the only behavior change needed for register-token correctness in the adapter path; no new branch is added
- `vit_adapter_regofficial.py`
  - purpose: bridge the original segmentation_m2f ViT-Adapter logic onto the native official reg backbone, avoiding `segmentation_m2f/vit.py`
  - effect on algorithm:
    - backbone path changes from old segmentation-specific ViT skeleton to native official DINOv2 reg backbone
    - adapter path still follows the original SPM + interaction + feature-fusion logic
    - patch-token slicing is corrected to skip register tokens

## Key design decision

The old `segmentation_m2f/vit.py` path is not used as the backbone anymore. Instead, the overlay uses the snapshot's native official DINOv2 register-aware backbone directly and only keeps the original adapter-side logic where it is still valid.
