#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(
    description="Convert DINOv2 patch14 checkpoint to patch16 for current wsovvis/DINO loader."
)
parser.add_argument("filename", type=str, help="Input checkpoint path (.pth)")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Optional output path. Default: <input>_14to16_wsovvis.pth",
)
parser.add_argument(
    "--keep-module-prefix",
    action="store_true",
    help="Keep leading 'module.' prefix if present. Default: strip it.",
)
args = parser.parse_args()

in_path = Path(args.filename)
if args.output is None:
    out_path = in_path.with_name(in_path.stem + "_14to16_wsovvis.pth")
else:
    out_path = Path(args.output)

obj = torch.load(str(in_path), map_location=torch.device("cpu"))

# Support plain state_dict or wrapped dict
if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
    state_dict = obj["model"]
    wrapper = "model"
elif isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
    state_dict = obj["state_dict"]
    wrapper = "state_dict"
elif isinstance(obj, dict):
    state_dict = obj
    wrapper = None
else:
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)}")

# Optionally strip a leading 'module.' prefix to better match current wsovvis loader
new_state = {}
for k, v in state_dict.items():
    nk = k
    if not args.keep_module_prefix and nk.startswith("module."):
        nk = nk[len("module."):]
    new_state[nk] = v
state_dict = new_state

if "patch_embed.proj.weight" not in state_dict:
    raise KeyError("Expected key 'patch_embed.proj.weight' not found in checkpoint")

patch_embed = state_dict["patch_embed.proj.weight"]
if not isinstance(patch_embed, torch.Tensor):
    raise TypeError("patch_embed.proj.weight is not a torch.Tensor")

if patch_embed.ndim != 4:
    raise ValueError(
        f"Expected patch_embed.proj.weight to be 4D [out,in,h,w], got shape {tuple(patch_embed.shape)}"
    )

if tuple(patch_embed.shape[-2:]) != (14, 14):
    raise ValueError(
        f"Expected patch size 14x14 before conversion, got {tuple(patch_embed.shape[-2:])}"
    )

# Official ViT-Adapter does this step too. We keep all other keys unchanged.
state_dict["patch_embed.proj.weight"] = F.interpolate(
    patch_embed, size=(16, 16), mode="bilinear", align_corners=False
)

# IMPORTANT:
# - keep mask_token if present
# - keep official DINOv2 names ls1.gamma / ls2.gamma
# This is the only intended difference from ViT-Adapter's official convert_14to16.py

if wrapper is None:
    out_obj = state_dict
else:
    out_obj = dict(obj)
    out_obj[wrapper] = state_dict

torch.save(out_obj, str(out_path))

print(f"[done] input : {in_path}")
print(f"[done] output: {out_path}")
print("[done] patch_embed.proj.weight resized: 14x14 -> 16x16")
print("[done] mask_token preserved if present")
print("[done] ls1.gamma / ls2.gamma names preserved")
