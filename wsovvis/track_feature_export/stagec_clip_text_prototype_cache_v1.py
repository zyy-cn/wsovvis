from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .stagec_semantic_slice_v1 import StageCSemanticSliceError


def _err(field_path: str, rule_summary: str) -> StageCSemanticSliceError:
    return StageCSemanticSliceError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _is_valid_label_id(label_id: object) -> bool:
    return (isinstance(label_id, int) and not isinstance(label_id, bool)) or (
        isinstance(label_id, str) and bool(label_id)
    )


def _serialize_label_id(label_id: int | str) -> str:
    return f"i:{label_id}" if isinstance(label_id, int) else f"s:{label_id}"


def _deserialize_label_id(value: str) -> int | str:
    _require(isinstance(value, str) and len(value) >= 3 and value[1] == ":", "label_id", "invalid serialized form")
    prefix, raw = value[0], value[2:]
    _require(prefix in ("i", "s"), "label_id", "invalid serialized prefix")
    if prefix == "i":
        return int(raw)
    return raw


def _resolve_label_text(label_id: int | str, label_text_by_id: Mapping[int | str, str] | None) -> str:
    if label_text_by_id is not None and label_id in label_text_by_id:
        text = label_text_by_id[label_id]
        _require(isinstance(text, str) and bool(text), f"label_text_by_id[{label_id}]", "must be non-empty string")
        return text
    if isinstance(label_id, str):
        return label_id
    return f"class_{label_id}"


def _format_prompt_text(label_text: str, prompt_variant: str) -> str:
    if prompt_variant == "default":
        return f"a photo of {label_text}"
    if prompt_variant == "label_only":
        return label_text
    return f"[{prompt_variant}] {label_text}"


def _deterministic_vector(*, seed_material: str, embedding_dim: int) -> np.ndarray:
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    seed = int(digest[:16], 16) & ((1 << 63) - 1)
    rng = np.random.default_rng(seed=seed)
    vec = rng.standard_normal(size=(embedding_dim,), dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        vec[0] = 1.0
        norm = 1.0
    return (vec / norm).astype(np.float32)


@dataclass(frozen=True)
class StageCClipTextPrototypeCacheEntryV1:
    cache_key: str
    metadata_path: Path
    tensor_path: Path
    model_name: str
    prompt_variant: str
    candidate_label_ids: tuple[int | str, ...]
    label_texts: tuple[str, ...]
    prototype_features: np.ndarray
    cache_hit: bool


def compute_stagec_clip_text_cache_key_v1(
    *,
    candidate_label_ids: Sequence[int | str],
    label_text_by_id: Mapping[int | str, str] | None,
    model_name: str,
    prompt_variant: str,
    embedding_dim: int,
) -> str:
    _require(isinstance(model_name, str) and bool(model_name), "model_name", "must be non-empty string")
    _require(isinstance(prompt_variant, str) and bool(prompt_variant), "prompt_variant", "must be non-empty string")
    _require(isinstance(embedding_dim, int) and embedding_dim > 0, "embedding_dim", "must be positive integer")
    serialized_labels: list[dict[str, str]] = []
    for idx, raw_label in enumerate(candidate_label_ids):
        _require(_is_valid_label_id(raw_label), f"candidate_label_ids[{idx}]", "must be non-empty string or integer")
        label_text = _resolve_label_text(raw_label, label_text_by_id)
        serialized_labels.append({"id": _serialize_label_id(raw_label), "text": label_text})
    payload = {
        "version": "stagec_clip_text_cache_v1",
        "model_name": model_name,
        "prompt_variant": prompt_variant,
        "embedding_dim": embedding_dim,
        "labels": serialized_labels,
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_stagec_clip_text_prototypes_v1(
    *,
    candidate_label_ids: Sequence[int | str],
    label_text_by_id: Mapping[int | str, str] | None,
    model_name: str,
    prompt_variant: str,
    embedding_dim: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    _require(isinstance(embedding_dim, int) and embedding_dim > 0, "embedding_dim", "must be positive integer")
    features: list[np.ndarray] = []
    texts: list[str] = []
    for idx, raw_label in enumerate(candidate_label_ids):
        _require(_is_valid_label_id(raw_label), f"candidate_label_ids[{idx}]", "must be non-empty string or integer")
        label_text = _resolve_label_text(raw_label, label_text_by_id)
        prompt_text = _format_prompt_text(label_text, prompt_variant)
        seed_material = f"{model_name}|{prompt_variant}|{prompt_text}|{embedding_dim}"
        features.append(_deterministic_vector(seed_material=seed_material, embedding_dim=embedding_dim))
        texts.append(label_text)
    if features:
        stacked = np.stack(features, axis=0).astype(np.float32)
    else:
        stacked = np.zeros((0, embedding_dim), dtype=np.float32)
    return stacked, tuple(texts)


def save_stagec_clip_text_prototype_cache_v1(
    *,
    cache_root: str | Path,
    cache_key: str,
    model_name: str,
    prompt_variant: str,
    candidate_label_ids: Sequence[int | str],
    label_texts: Sequence[str],
    prototype_features: np.ndarray,
) -> tuple[Path, Path]:
    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    metadata_path = root / f"{cache_key}.json"
    tensor_path = root / f"{cache_key}.npz"
    _require(isinstance(prototype_features, np.ndarray), "prototype_features", "must be numpy ndarray")
    _require(prototype_features.ndim == 2, "prototype_features", "must be rank-2 [N_cand, D]")
    _require(prototype_features.dtype == np.float32, "prototype_features", "must be float32")
    _require(np.isfinite(prototype_features).all(), "prototype_features", "must be finite")
    _require(len(candidate_label_ids) == prototype_features.shape[0], "candidate_label_ids", "length must match N_cand")
    _require(len(label_texts) == prototype_features.shape[0], "label_texts", "length must match N_cand")
    meta = {
        "version": "stagec_clip_text_cache_v1",
        "cache_key": cache_key,
        "model_name": model_name,
        "prompt_variant": prompt_variant,
        "embedding_dim": int(prototype_features.shape[1]),
        "n_cand": int(prototype_features.shape[0]),
        "candidate_label_ids": [_serialize_label_id(v) for v in candidate_label_ids],
        "label_texts": list(label_texts),
    }
    metadata_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    np.savez_compressed(tensor_path, prototype_features=prototype_features)
    return metadata_path, tensor_path


def load_stagec_clip_text_prototype_cache_v1(
    *,
    cache_root: str | Path,
    cache_key: str,
) -> StageCClipTextPrototypeCacheEntryV1 | None:
    root = Path(cache_root)
    metadata_path = root / f"{cache_key}.json"
    tensor_path = root / f"{cache_key}.npz"
    if not metadata_path.exists() or not tensor_path.exists():
        return None
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    arr = np.load(tensor_path, allow_pickle=False)["prototype_features"].astype(np.float32)
    labels = tuple(_deserialize_label_id(v) for v in meta["candidate_label_ids"])
    texts = tuple(meta["label_texts"])
    _require(arr.ndim == 2, "prototype_features", "must be rank-2 [N_cand, D]")
    _require(arr.shape[0] == len(labels), "prototype_features.shape[0]", "must match label count")
    _require(arr.shape[0] == len(texts), "prototype_features.shape[0]", "must match text count")
    _require(np.isfinite(arr).all(), "prototype_features", "must be finite")
    return StageCClipTextPrototypeCacheEntryV1(
        cache_key=cache_key,
        metadata_path=metadata_path,
        tensor_path=tensor_path,
        model_name=meta["model_name"],
        prompt_variant=meta["prompt_variant"],
        candidate_label_ids=labels,
        label_texts=texts,
        prototype_features=arr,
        cache_hit=True,
    )


def get_or_build_stagec_clip_text_prototype_cache_v1(
    *,
    cache_root: str | Path,
    candidate_label_ids: Sequence[int | str],
    label_text_by_id: Mapping[int | str, str] | None = None,
    model_name: str = "clip-vit-b32",
    prompt_variant: str = "default",
    embedding_dim: int = 512,
) -> StageCClipTextPrototypeCacheEntryV1:
    cache_key = compute_stagec_clip_text_cache_key_v1(
        candidate_label_ids=candidate_label_ids,
        label_text_by_id=label_text_by_id,
        model_name=model_name,
        prompt_variant=prompt_variant,
        embedding_dim=embedding_dim,
    )
    hit = load_stagec_clip_text_prototype_cache_v1(cache_root=cache_root, cache_key=cache_key)
    if hit is not None:
        return hit
    prototype_features, label_texts = build_stagec_clip_text_prototypes_v1(
        candidate_label_ids=candidate_label_ids,
        label_text_by_id=label_text_by_id,
        model_name=model_name,
        prompt_variant=prompt_variant,
        embedding_dim=embedding_dim,
    )
    metadata_path, tensor_path = save_stagec_clip_text_prototype_cache_v1(
        cache_root=cache_root,
        cache_key=cache_key,
        model_name=model_name,
        prompt_variant=prompt_variant,
        candidate_label_ids=candidate_label_ids,
        label_texts=label_texts,
        prototype_features=prototype_features,
    )
    return StageCClipTextPrototypeCacheEntryV1(
        cache_key=cache_key,
        metadata_path=metadata_path,
        tensor_path=tensor_path,
        model_name=model_name,
        prompt_variant=prompt_variant,
        candidate_label_ids=tuple(candidate_label_ids),
        label_texts=label_texts,
        prototype_features=prototype_features,
        cache_hit=False,
    )
