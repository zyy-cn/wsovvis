from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_dinov2_import_path() -> None:
    repo_root = _repo_root()
    vendor_root = repo_root / "third_party" / "dinov2"
    path_value = str(vendor_root)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)


_ensure_dinov2_import_path()

from dinov2.data.transforms import make_classification_eval_transform  # noqa: E402
from dinov2.hub.backbones import dinov2_vitb14_reg  # noqa: E402


class DinoV2VisualEncoder:
    def __init__(
        self,
        *,
        dinov2_ckpt: str,
        device: str = "cuda:0",
        batch_size_frames: int = 4,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size_frames = max(1, int(batch_size_frames))
        self.transform = make_classification_eval_transform()
        if dinov2_ckpt == "vitb14_reg":
            self.model = dinov2_vitb14_reg(pretrained=True)
        else:
            self.model = dinov2_vitb14_reg(pretrained=True, weights=dinov2_ckpt)
        self.model.eval()
        self.model.to(self.device)

    def _load_image_tensor(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
        return self.transform(rgb)

    @torch.inference_mode()
    def encode_image_paths(self, image_paths: Iterable[Path]) -> np.ndarray:
        paths: List[Path] = [Path(p) for p in image_paths]
        if not paths:
            return np.zeros((0, 0), dtype=np.float32)
        all_vectors: List[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size_frames):
            batch_paths = paths[start : start + self.batch_size_frames]
            tensors = [self._load_image_tensor(path) for path in batch_paths]
            batch = torch.stack(tensors, dim=0).to(self.device, non_blocking=True)
            features = self.model.forward_features(batch)
            vectors = features["x_norm_clstoken"].detach().cpu().float().numpy()
            all_vectors.append(vectors)
        return np.concatenate(all_vectors, axis=0).astype(np.float32, copy=False)
