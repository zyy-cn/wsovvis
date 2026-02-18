import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling import BACKBONE_REGISTRY, Backbone
from detectron2.structures import ImageList
import torch.nn.functional as F


@BACKBONE_REGISTRY.register()
class DINOv2PseudoFPN(Backbone):
    """
    DINOv2 frozen backbone + simple multi-scale feature adapter (pseudo-FPN).
    Produces a dict of feature maps with expected keys/strides for SeqFormer.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        model_name = cfg.MODEL.DINOV2.MODEL_NAME  # e.g. "dinov2_vitb14"
        out_channels = cfg.MODEL.DINOV2.OUT_CHANNELS  # e.g. 256
        self.out_features = cfg.MODEL.DINOV2.OUT_FEATURES  # e.g. ["res2","res3","res4","res5"]

        # 1) Load dinov2 (fast path)
        # NOTE: this will download weights if not cached.
        # self.dino = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        import os
        import torch

        def _load_state_dict_any(path: str):
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict):
                # 常见几种 key
                for k in ["model", "state_dict", "teacher", "student"]:
                    if k in ckpt and isinstance(ckpt[k], dict):
                        sd = ckpt[k]
                        break
                else:
                    # 可能已经就是 state_dict
                    sd = ckpt
            else:
                sd = ckpt

            # 去掉可能的 "module." 前缀
            if any(key.startswith("module.") for key in sd.keys()):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

            return sd

        # ---- build model from LOCAL dinov2 repo ----
        repo_path = cfg.MODEL.DINOV2.REPO_PATH
        assert repo_path and os.path.isdir(repo_path), f"Invalid DINOV2.REPO_PATH: {repo_path}"

        # 用本地 hubconf 注册的 model_name（dinov2_vitb14 等）
        self.dino = torch.hub.load(repo_path, model_name, source="local", pretrained=False)

        # ---- load local weights ----
        w = cfg.MODEL.DINOV2.WEIGHTS
        assert w and os.path.isfile(w), f"Invalid DINOV2.WEIGHTS: {w}"
        sd = _load_state_dict_any(w)
        missing, unexpected = self.dino.load_state_dict(sd, strict=False)

        # 可选：打印一次，确认确实加载上了
        # print("DINOv2 load_state_dict missing:", missing[:10], "..., unexpected:", unexpected[:10], "...")

        # 2) Freeze
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)

        # 3) Patch size (dinov2 vit*14 -> 14)
        # Some dinov2 exposes patch_embed.patch_size; keep robust
        ps = getattr(getattr(self.dino, "patch_embed", None), "patch_size", 14)
        self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)

        # 4) DINO output dim: depends on model (vitb14=768, vitl14=1024, etc.)
        # We'll infer once by running a tiny forward in build, but simplest:
        dino_dim = cfg.MODEL.DINOV2.DINO_DIM

        # 5) 1x1 projections for each scale
        self.proj = nn.ModuleDict({
            "res2": nn.Conv2d(dino_dim, out_channels, 1),
            "res3": nn.Conv2d(dino_dim, out_channels, 1),
            "res4": nn.Conv2d(dino_dim, out_channels, 1),
            "res5": nn.Conv2d(dino_dim, out_channels, 1),
        })

        # Declare strides to downstream (match your SeqFormer expectation)
        self._out_feature_strides = {
            "res2": 8,
            "res3": 16,
            "res4": 32,
            "res5": 64,
        }
        self._out_feature_channels = {k: out_channels for k in self._out_feature_strides.keys()}

    def forward(self, x: ImageList):
        if isinstance(x, torch.Tensor):
            images = x
        elif hasattr(x, "tensor"):  # detectron2 ImageList
            images = x.tensor
        elif hasattr(x, "tensors"):  # NestedTensor in SeqFormer
            images = x.tensors
        else:
            raise TypeError(f"Unsupported input type for backbone: {type(x)}")

        # DINOv2 expects normalized input; ensure cfg.MODEL.PIXEL_MEAN/STD set correctly.
        with torch.no_grad():


            # ...
            H, W = images.shape[-2:]
            ps = int(self.patch_size)
            pad_h = (ps - H % ps) % ps
            pad_w = (ps - W % ps) % ps
            if pad_h or pad_w:
                # pad format: (left, right, top, bottom)
                images = F.pad(images, (0, pad_w, 0, pad_h), value=0.0)

            feats = self.dino.forward_features(images)
            # Common key: 'x_norm_patchtokens' (B, N, C)
            pt = feats["x_norm_patchtokens"]

        B, N, C = pt.shape
        H, W = images.shape[-2], images.shape[-1]
        Hp, Wp = H // self.patch_size, W // self.patch_size

        # (B, N, C) -> (B, C, Hp, Wp)
        f = pt.transpose(1, 2).reshape(B, C, Hp, Wp).contiguous()

        # Build 4 scales by interpolation to target spatial sizes
        # sizes derived from desired stride: e.g. H/8, H/16, H/32, H/64
        out = {}

        def make_level(name, stride):
            th, tw = (H + stride - 1) // stride, (W + stride - 1) // stride
            g = F.interpolate(f, size=(th, tw), mode="bilinear", align_corners=False)
            g = self.proj[name](g)
            return g

        # Use only required features
        for name in self.out_features:
            stride = self._out_feature_strides[name]
            out[name] = make_level(name, stride)

        return out

    def output_shape(self):
        from detectron2.layers import ShapeSpec
        return {
            k: ShapeSpec(channels=self._out_feature_channels[k], stride=self._out_feature_strides[k])
            for k in self.out_features
        }
