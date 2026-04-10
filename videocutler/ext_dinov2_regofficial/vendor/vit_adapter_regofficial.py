import math
import sys
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DINOV2_REPO = _REPO_ROOT / "third_party" / "dinov2"
if str(_DINOV2_REPO) not in sys.path:
    sys.path.insert(0, str(_DINOV2_REPO))

from dinov2.models.vision_transformer import vit_base, vit_large

from .adapter_modules import SpatialPriorModule, deform_inputs
from .adapter_modules_prefix import InteractionBlockWithPrefix
from .ops.modules import MSDeformAttn


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "teacher", "student"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _strip_prefixes(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in ("module.", "backbone."):
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


def _resize_pos_embed_if_needed(state_dict, model):
    if "pos_embed" not in state_dict:
        return state_dict
    src = state_dict["pos_embed"]
    dst = model.pos_embed
    if src.shape == dst.shape:
        return state_dict
    if src.ndim != 3 or dst.ndim != 3 or src.shape[0] != 1 or dst.shape[0] != 1 or src.shape[2] != dst.shape[2]:
        return state_dict
    src_n = src.shape[1] - 1
    dst_n = dst.shape[1] - 1
    src_hw = int(math.sqrt(src_n))
    dst_hw = int(math.sqrt(dst_n))
    if src_hw * src_hw != src_n or dst_hw * dst_hw != dst_n:
        return state_dict
    cls = src[:, :1]
    patch = src[:, 1:].reshape(1, src_hw, src_hw, -1).permute(0, 3, 1, 2)
    patch = F.interpolate(patch, size=(dst_hw, dst_hw), mode="bicubic", align_corners=False)
    patch = patch.permute(0, 2, 3, 1).reshape(1, dst_hw * dst_hw, -1)
    state_dict = dict(state_dict)
    state_dict["pos_embed"] = torch.cat((cls, patch), dim=1)
    return state_dict


def _load_official_backbone(backbone, pretrained, strict=True):
    if not pretrained:
        return
    state_dict = torch.load(pretrained, map_location="cpu")
    state_dict = _resize_pos_embed_if_needed(_strip_prefixes(_extract_state_dict(state_dict)), backbone)
    incompatible = backbone.load_state_dict(state_dict, strict=False)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            "Official reg backbone checkpoint mismatch: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )


class ViTAdapterRegOfficial(nn.Module):
    def __init__(
        self,
        img_size=592,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        drop_path_rate=0.4,
        init_values=1.0,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        pretrained=None,
        use_extra_extractor=True,
        freeze_vit=False,
        strict_pretrain_load=True,
    ):
        super().__init__()
        if embed_dim == 768 and depth == 12 and num_heads == 12:
            backbone_ctor = vit_base
        elif embed_dim == 1024 and depth == 24 and num_heads == 16:
            backbone_ctor = vit_large
        else:
            raise ValueError(
                f"Unsupported DINOv2 backbone spec: embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}"
            )

        self.backbone = backbone_ctor(
            img_size=img_size,
            patch_size=patch_size,
            init_values=init_values,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset,
            drop_path_rate=drop_path_rate,
            block_chunks=0,
        )
        _load_official_backbone(self.backbone, pretrained, strict=strict_pretrain_load)
        self.freeze_vit = freeze_vit
        if self.freeze_vit:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size if isinstance(self.backbone.patch_size, int) else self.backbone.patch_size[0]
        self.num_register_tokens = self.backbone.num_register_tokens
        self.interaction_indexes = interaction_indexes or [[0, 5], [6, 11], [12, 17], [18, 23]]
        self.add_vit_feature = add_vit_feature

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False)
        self.interactions = nn.Sequential(
            *[
                InteractionBlockWithPrefix(
                    dim=self.embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=0.0,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((True if i == len(self.interaction_indexes) - 1 else False) and use_extra_extractor),
                    with_cp=False,
                )
                for i in range(len(self.interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(self.embed_dim)
        self.norm2 = nn.SyncBatchNorm(self.embed_dim)
        self.norm3 = nn.SyncBatchNorm(self.embed_dim)
        self.norm4 = nn.SyncBatchNorm(self.embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)


    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_vit", False):
            # Keep the official DINOv2 trunk frozen and deterministic even when
            # the outer detectron2 model enters train() mode.
            self.backbone.eval()
        return self

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        H_toks = x.shape[2] // self.patch_size
        W_toks = x.shape[3] // self.patch_size
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16

        x_tokens = self.backbone.prepare_tokens_with_masks(x)
        prefix_len = 1 + self.num_register_tokens
        prefix, x_tokens = x_tokens[:, :prefix_len], x_tokens[:, prefix_len:]

        bs, _, dim = x_tokens.shape
        outs = []
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            blocks = self.backbone.blocks[indexes[0] : indexes[-1] + 1]
            x_tokens, c, prefix = layer(
                x_tokens,
                c,
                prefix,
                blocks,
                deform_inputs1,
                deform_inputs2,
                H_c,
                W_c,
            )
            outs.append(x_tokens.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]
        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(H_c, W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
