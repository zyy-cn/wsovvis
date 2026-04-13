from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, normalize, pil_to_tensor, resize

from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import FrameGeometry, build_valid_token_mask, keep_aspect_pad_geometry

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_dinov2_import_path() -> None:
    repo_root = _repo_root()
    vendor_root = repo_root / "third_party" / "dinov2"
    path_value = str(vendor_root)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)


_ensure_dinov2_import_path()

from dinov2.hub.backbones import dinov2_vitb14_reg  # noqa: E402


@dataclass(frozen=True)
class EncodedFrame:
    features: np.ndarray
    geometry: FrameGeometry
    valid_token_mask: np.ndarray


class DinoV2VisualEncoder:
    def __init__(
        self,
        *,
        dinov2_ckpt: str,
        device: str = "cuda:0",
        batch_size_frames: int = 4,
        preprocess_mode: str = "keep_aspect_pad",
        resize_short_side: int = 672,
        max_long_side: int = 1008,
        pad_to_multiple: int = 14,
        patch_size: int = 14,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size_frames = max(1, int(batch_size_frames))
        self.preprocess_mode = str(preprocess_mode)
        self.resize_short_side = int(resize_short_side)
        self.max_long_side = int(max_long_side)
        self.pad_to_multiple = int(pad_to_multiple)
        self.patch_size = int(patch_size)
        if self.preprocess_mode != "keep_aspect_pad":
            raise ValueError("unsupported preprocess_mode; only keep_aspect_pad is supported")
        if dinov2_ckpt == "vitb14_reg":
            self.model = dinov2_vitb14_reg(pretrained=True)
        else:
            self.model = dinov2_vitb14_reg(pretrained=True, weights=dinov2_ckpt)
        self.model.eval()
        self.model.to(self.device)

    def _load_image_tensor(self, image_path: Path) -> tuple[torch.Tensor, FrameGeometry, np.ndarray]:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            orig_w, orig_h = rgb.size
            geometry = keep_aspect_pad_geometry(
                orig_h=orig_h,
                orig_w=orig_w,
                resize_short_side=self.resize_short_side,
                max_long_side=self.max_long_side,
                pad_to_multiple=self.pad_to_multiple,
                patch_size=self.patch_size,
            )
            resized = resize(rgb, [int(geometry.resized_h), int(geometry.resized_w)], interpolation=InterpolationMode.BICUBIC, antialias=True)
            canvas = Image.new("RGB", (int(geometry.padded_w), int(geometry.padded_h)), color=(0, 0, 0))
            canvas.paste(resized, (0, 0))

        x = pil_to_tensor(canvas).float() / 255.0
        x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        valid_mask = build_valid_token_mask(geometry)
        return x, geometry, valid_mask

    @torch.inference_mode()
    def encode_image_paths(self, image_paths: Iterable[Path]) -> List[EncodedFrame]:
        paths: List[Path] = [Path(p) for p in image_paths]
        if not paths:
            return []
        packed = [self._load_image_tensor(path) for path in paths]
        shape_buckets: Dict[Tuple[int, int, int], List[int]] = OrderedDict()
        for idx, (tensor, _, _) in enumerate(packed):
            shape_key = (int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2]))
            shape_buckets.setdefault(shape_key, []).append(idx)

        out: List[EncodedFrame | None] = [None] * len(packed)
        for bucket_indices in shape_buckets.values():
            for start in range(0, len(bucket_indices), self.batch_size_frames):
                chunk_indices = bucket_indices[start : start + self.batch_size_frames]
                tensors = [packed[i][0] for i in chunk_indices]
                geoms = [packed[i][1] for i in chunk_indices]
                masks = [packed[i][2] for i in chunk_indices]

                batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
                features = self.model.forward_features(batch)
                patch_tokens = features["x_norm_patchtokens"].detach().cpu().float().numpy()
                for local_idx, global_idx in enumerate(chunk_indices):
                    geom = geoms[local_idx]
                    token_count = int(geom.grid_h) * int(geom.grid_w)
                    if int(patch_tokens[local_idx].shape[0]) != token_count:
                        raise ValueError(
                            f"patch token count mismatch: got {patch_tokens[local_idx].shape[0]}, expected {token_count}"
                        )
                    out[global_idx] = EncodedFrame(
                        features=np.asarray(patch_tokens[local_idx], dtype=np.float32),
                        geometry=geom,
                        valid_token_mask=np.asarray(masks[local_idx], dtype=np.uint8),
                    )
        return [item for item in out if item is not None]
