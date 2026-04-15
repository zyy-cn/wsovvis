from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch


_ENV_VAR = "WSOVVIS_OPENAI_CLIP_VITB16_WEIGHTS"
_FALLBACK_REL = Path("weights/CLIP/ViT-B-16.pt")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _import_vendored_clip():
    vendor_root = _repo_root() / "third_party" / "openai_clip"
    vendor_path = str(vendor_root)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    import clip  # type: ignore

    return clip


def resolve_openai_clip_vitb16_weights() -> Path:
    env_value = os.environ.get(_ENV_VAR, "").strip()
    if env_value:
        path = Path(env_value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"{_ENV_VAR} points to a missing file: {path}"
            )
        return path

    fallback = (_repo_root() / _FALLBACK_REL).resolve()
    if fallback.is_file():
        return fallback

    raise FileNotFoundError(
        f"OpenAI CLIP ViT-B/16 weights not found; set {_ENV_VAR} or provide {fallback}"
    )


def _prepare_download_root(weights_path: Path) -> Path:
    cache_root = Path(tempfile.gettempdir()) / "wsovvis_openai_clip_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / "ViT-B-16.pt"
    if target.exists() or target.is_symlink():
        if target.resolve() == weights_path.resolve():
            return cache_root
        target.unlink()
    target.symlink_to(weights_path.resolve())
    return cache_root


@dataclass(frozen=True)
class ClipTextEncoderConfig:
    clip_ckpt: str
    device: str


class OpenAIClipTextEncoder:
    def __init__(self, config: ClipTextEncoderConfig) -> None:
        if config.clip_ckpt != "openai_clip_vit_b16":
            raise ValueError(f"unsupported clip_ckpt: {config.clip_ckpt}")
        self.config = config
        self.weights_path = resolve_openai_clip_vitb16_weights()
        self._clip = _import_vendored_clip()
        download_root = _prepare_download_root(self.weights_path)
        self.model, _ = self._clip.load(
            "ViT-B/16",
            device=config.device,
            jit=False,
            download_root=str(download_root),
        )
        self.model.eval()

    def encode_texts(self, texts: Iterable[str], *, batch_size: int = 256) -> np.ndarray:
        text_list = [str(text) for text in texts]
        if not text_list:
            return np.zeros((0, 512), dtype=np.float32)

        encoded: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(text_list), max(1, int(batch_size))):
                batch = text_list[start : start + max(1, int(batch_size))]
                tokens = self._clip.tokenize(batch).to(self.config.device)
                features = self.model.encode_text(tokens)
                encoded.append(features.detach().float().cpu().numpy().astype(np.float32, copy=False))
        return np.concatenate(encoded, axis=0)
