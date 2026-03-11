from __future__ import annotations

import json
import os
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .prototype_bank_v9 import (
    PROTOTYPE_SCHEMA_NAME,
    PrototypeBankError,
    load_prototype_bank_v9,
)


SCHEMA_VERSION = "1.0.0"
TEXT_MAP_SCHEMA_NAME = "wsovvis.text_map_v9"
SUMMARY_SCHEMA_NAME = "wsovvis.g5_text_map_summary_v9"
WORKED_EXAMPLE_SCHEMA_NAME = "wsovvis.g5_text_map_worked_example_v9"


class TextMapError(RuntimeError):
    """Raised when the bounded G5 text-map artifact is invalid."""


TextFeatureBackend = Callable[[Sequence[str], "TextMapConfig"], np.ndarray]


@dataclass(frozen=True)
class TextMapConfig:
    text_model_name: str = "ViT-B-32"
    text_model_pretrained: str = "openai"
    prompt_variant: str = "default"
    ridge_lambda: float = 1e-2
    device: str = "cpu"
    batch_size: int = 128
    cache_dir: str | None = None

    def canonical_dict(self) -> Dict[str, Any]:
        _require(
            isinstance(self.text_model_name, str) and bool(self.text_model_name),
            "config.text_model_name",
            "must be a non-empty string",
        )
        _require(
            isinstance(self.text_model_pretrained, str) and bool(self.text_model_pretrained),
            "config.text_model_pretrained",
            "must be a non-empty string",
        )
        _require(
            isinstance(self.prompt_variant, str) and self.prompt_variant == "default",
            "config.prompt_variant",
            "must equal 'default'",
        )
        _require(
            isinstance(self.ridge_lambda, (int, float)) and float(self.ridge_lambda) >= 0.0,
            "config.ridge_lambda",
            "must be numeric >= 0",
        )
        _require(
            isinstance(self.device, str) and self.device in {"cpu", "cuda", "auto"},
            "config.device",
            "must be one of {'cpu','cuda','auto'}",
        )
        _require(
            isinstance(self.batch_size, int) and self.batch_size >= 1,
            "config.batch_size",
            "must be integer >= 1",
        )
        if self.cache_dir is not None:
            _require(isinstance(self.cache_dir, str) and bool(self.cache_dir), "config.cache_dir", "must be a non-empty string")
        return {
            "text_model_name": self.text_model_name,
            "text_model_pretrained": self.text_model_pretrained,
            "prompt_variant": self.prompt_variant,
            "ridge_lambda": float(self.ridge_lambda),
            "device": self.device,
            "batch_size": int(self.batch_size),
            "cache_dir": self.cache_dir,
        }


def _err(field_path: str, rule_summary: str) -> TextMapError:
    return TextMapError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path, rule_summary)


def _normalize_label_id(value: Any, *, field_path: str) -> int | str:
    _require(
        (isinstance(value, int) and not isinstance(value, bool)) or (isinstance(value, str) and bool(value)),
        field_path,
        "must be non-empty string or integer",
    )
    return value  # type: ignore[return-value]


def _canonical_label_key(label_id: int | str) -> tuple[int, int | str]:
    if isinstance(label_id, int) and not isinstance(label_id, bool):
        return (0, int(label_id))
    return (1, str(label_id))


def _canonical_label_sort_key(label_id: int | str) -> Tuple[int, str]:
    key = _canonical_label_key(label_id)
    return (key[0], str(key[1]).zfill(12) if key[0] == 0 else str(key[1]))


def _parse_major_version(value: Any, field_path: str) -> int:
    _require(isinstance(value, str), field_path, "must be a string")
    parts = value.split(".")
    _require(len(parts) == 3 and all(part.isdigit() for part in parts), field_path, "must follow MAJOR.MINOR.PATCH")
    return int(parts[0])


def _load_json(path: Path, file_label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TextMapError(f"{file_label} missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TextMapError(f"{file_label} invalid JSON at {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{file_label}.$", "top-level value must be an object")
    return payload


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _require_relative_path(path_value: Any, field_path: str) -> str:
    _require(isinstance(path_value, str) and bool(path_value), field_path, "must be a non-empty relative path")
    rel = PurePosixPath(path_value)
    _require(not rel.is_absolute(), field_path, "absolute path is forbidden")
    _require(".." not in rel.parts, field_path, "must not contain '..'")
    return str(rel)


def _discover_manifest_path(root: Path) -> Path:
    manifest_v1 = root / "text_map_manifest.v1.json"
    manifest_compat = root / "text_map_manifest.json"
    if manifest_v1.exists():
        return manifest_v1
    if manifest_compat.exists():
        return manifest_compat
    raise TextMapError(f"text-map manifest missing under root {root}; expected text_map_manifest.v1.json or text_map_manifest.json")


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    _require(arr.ndim == 2, "matrix", "must be rank-2")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0.0, norms, 1.0)
    return np.asarray(arr / norms, dtype=np.float32)


def _prompt_text(label_text: str, prompt_variant: str) -> str:
    normalized = label_text.replace("_", " ")
    if prompt_variant == "default":
        return f"a photo of {normalized}"
    raise TextMapError(f"unsupported prompt_variant: {prompt_variant}")


def fit_ridge_text_map_v9(text_features: np.ndarray, visual_prototypes: np.ndarray, ridge_lambda: float) -> np.ndarray:
    x = np.asarray(text_features, dtype=np.float64)
    y = np.asarray(visual_prototypes, dtype=np.float64)
    _require(x.ndim == 2, "text_features", "must be rank-2 [C, D_text]")
    _require(y.ndim == 2, "visual_prototypes", "must be rank-2 [C, D_visual]")
    _require(x.shape[0] == y.shape[0], "text_features.shape[0]", "must match visual prototype count")
    _require(x.shape[0] > 0, "text_features.shape[0]", "must be > 0")
    _require(np.isfinite(x).all(), "text_features", "must be finite")
    _require(np.isfinite(y).all(), "visual_prototypes", "must be finite")
    regularizer = float(ridge_lambda) * np.eye(x.shape[1], dtype=np.float64)
    lhs = x.T @ x + regularizer
    rhs = x.T @ y
    try:
        solution = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.pinv(lhs) @ rhs
    return np.asarray(solution, dtype=np.float32)


def compute_text_map_alignment_metrics_v9(visual_prototypes: np.ndarray, mapped_text_prototypes: np.ndarray) -> Dict[str, Any]:
    visual = _l2_normalize_rows(visual_prototypes)
    mapped = _l2_normalize_rows(mapped_text_prototypes)
    similarities = np.asarray(mapped @ visual.T, dtype=np.float64)
    diagonal = np.diag(similarities)
    top1 = np.argmax(similarities, axis=1)
    margins: List[float] = []
    for row_index in range(similarities.shape[0]):
        if similarities.shape[1] == 1:
            margins.append(float(diagonal[row_index]))
            continue
        others = np.delete(similarities[row_index], row_index)
        margins.append(float(diagonal[row_index] - np.max(others)))
    margins_np = np.asarray(margins, dtype=np.float64)
    return {
        "num_labels_aligned": int(similarities.shape[0]),
        "mean_diagonal_cosine": float(diagonal.mean()) if diagonal.size else 0.0,
        "median_diagonal_cosine": float(np.median(diagonal)) if diagonal.size else 0.0,
        "min_diagonal_cosine": float(diagonal.min()) if diagonal.size else 0.0,
        "max_diagonal_cosine": float(diagonal.max()) if diagonal.size else 0.0,
        "top1_retrieval_accuracy": float(np.mean(top1 == np.arange(similarities.shape[0]))) if diagonal.size else 0.0,
        "mean_margin": float(margins_np.mean()) if margins_np.size else 0.0,
        "median_margin": float(np.median(margins_np)) if margins_np.size else 0.0,
    }


def encode_label_texts_open_clip_v9(label_texts: Sequence[str], config: TextMapConfig) -> np.ndarray:
    try:
        import open_clip  # type: ignore[import-not-found]
        import torch  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        raise TextMapError(f"open_clip backend unavailable: {exc}") from exc
    prompts = [_prompt_text(text, config.prompt_variant) for text in label_texts]
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_spec = str(config.text_model_pretrained)
    pretrained_path = Path(pretrained_spec).expanduser()
    model_kwargs: Dict[str, Any] = {}
    if pretrained_path.exists():
        # Locally managed OpenAI checkpoints on the canonical runner may be TorchScript
        # archives; PyTorch 2.6 requires weights_only=False for that trusted local case.
        model_kwargs["weights_only"] = False
    model = open_clip.create_model(
        config.text_model_name,
        pretrained=pretrained_spec,
        device=device,
        cache_dir=config.cache_dir,
        **model_kwargs,
    )
    tokenizer = open_clip.get_tokenizer(config.text_model_name)
    model.eval()
    all_features: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(prompts), int(config.batch_size)):
            batch = prompts[start : start + int(config.batch_size)]
            tokens = tokenizer(batch).to(device)
            encoded = model.encode_text(tokens)
            encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            all_features.append(encoded.detach().cpu().float().numpy())
    _require(all_features, "text_features", "no prompts were encoded")
    return np.asarray(np.concatenate(all_features, axis=0), dtype=np.float32)


def build_text_map_v9(
    prototype_bank_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    config: TextMapConfig | None = None,
    text_feature_backend: TextFeatureBackend | None = None,
) -> Path:
    cfg = config or TextMapConfig()
    cfg_dict = cfg.canonical_dict()
    prototype_view = load_prototype_bank_v9(prototype_bank_root)
    label_rows = [record.metadata for record in prototype_view.iter_records()]
    label_texts = [record.label_text for record in label_rows]
    label_ids = [record.label_id for record in label_rows]
    prompt_texts = [_prompt_text(text, cfg.prompt_variant) for text in label_texts]
    backend = text_feature_backend or encode_label_texts_open_clip_v9
    text_features = np.asarray(backend(label_texts, cfg), dtype=np.float32)
    _require(text_features.ndim == 2, "text_features", "must be rank-2 [C, D_text]")
    _require(text_features.shape[0] == len(label_rows), "text_features.shape[0]", "must match prototype rows")
    _require(np.isfinite(text_features).all(), "text_features", "must be finite")
    text_features = _l2_normalize_rows(text_features)
    visual_prototypes = np.asarray(prototype_view.prototypes, dtype=np.float32)
    matrix_a = fit_ridge_text_map_v9(text_features, visual_prototypes, float(cfg.ridge_lambda))
    mapped_text = _l2_normalize_rows(np.asarray(text_features @ matrix_a, dtype=np.float32))
    alignment = compute_text_map_alignment_metrics_v9(visual_prototypes, mapped_text)

    output_root = Path(output_root)
    if output_root.exists():
        if not overwrite:
            raise TextMapError(f"output root already exists: {output_root}")
        shutil.rmtree(output_root)
    temp_dir = Path(tempfile.mkdtemp(prefix="text_map_v9.", dir=str(output_root.parent if output_root.parent.exists() else Path.cwd())))
    try:
        np.savez_compressed(
            temp_dir / "text_map_state.v1.npz",
            A=matrix_a.astype(np.float32),
            text_features=text_features.astype(np.float32),
            mapped_text_prototypes=mapped_text.astype(np.float32),
        )
        np.savez_compressed(
            temp_dir / "mapped_text_prototype_arrays.v1.npz",
            prototypes=mapped_text.astype(np.float32),
        )
        prototype_bank_rel = os.path.relpath(Path(prototype_bank_root), start=Path(output_root))
        mapped_text_manifest = {
            "schema_name": PROTOTYPE_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "prototype_source": "g5_mapped_text_prototypes_v9",
            "split": prototype_view.split,
            "embedding_dim": int(prototype_view.embedding_dim),
            "dtype": "float32",
            "array_key": "prototypes",
            "arrays_path": "mapped_text_prototype_arrays.v1.npz",
            "producer": {
                "prototype_bank_root_rel": prototype_bank_rel,
                **cfg_dict,
            },
            "labels": [
                {
                    "label_id": label_id,
                    "label_text": label_text,
                    "row_index": row_index,
                    "prompt_text": prompt_text,
                    "support_video_count": int(label_rows[row_index].support_video_count),
                    "support_track_count": int(label_rows[row_index].support_track_count),
                }
                for row_index, (label_id, label_text, prompt_text) in enumerate(zip(label_ids, label_texts, prompt_texts))
            ],
        }
        selected_label_id = max(
            zip(label_rows, np.diag(mapped_text @ visual_prototypes.T)),
            key=lambda item: (
                int(item[0].support_video_count),
                float(item[1]),
                -_canonical_label_sort_key(item[0].label_id)[0],
                str(item[0].label_id),
            ),
        )[0].label_id
        text_map_manifest = {
            "schema_name": TEXT_MAP_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "split": prototype_view.split,
            "text_embedding_dim": int(text_features.shape[1]),
            "mapped_embedding_dim": int(mapped_text.shape[1]),
            "dtype": "float32",
            "state_arrays_path": "text_map_state.v1.npz",
            "mapped_text_manifest_path": "mapped_text_prototype_manifest.v1.json",
            "prototype_bank_root_rel": prototype_bank_rel,
            "selected_label_id": selected_label_id,
            "producer": cfg_dict,
            "alignment_metrics": alignment,
            "labels": [
                {
                    "label_id": label_id,
                    "label_text": label_text,
                    "row_index": row_index,
                    "prompt_text": prompt_text,
                    "support_video_count": int(label_rows[row_index].support_video_count),
                    "support_track_count": int(label_rows[row_index].support_track_count),
                }
                for row_index, (label_id, label_text, prompt_text) in enumerate(zip(label_ids, label_texts, prompt_texts))
            ],
        }
        _dump_json(temp_dir / "mapped_text_prototype_manifest.v1.json", mapped_text_manifest)
        _dump_json(temp_dir / "text_map_manifest.v1.json", text_map_manifest)
        temp_dir.replace(output_root)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return output_root


def summarize_text_map_v9(output_root: Path) -> Dict[str, Any]:
    view = _load_text_map_state(output_root)
    return {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split": view["manifest"]["split"],
        "prototype_bank_coverage": dict(view["prototype_summary"]["prototype_bank_coverage"]),
        "text_map_alignment": dict(view["manifest"]["alignment_metrics"]),
        "selected_label_id": view["manifest"]["selected_label_id"],
        "selected_label_text": view["labels_by_row"][view["selected_row_index"]]["label_text"],
        "text_embedding_dim": int(view["text_features"].shape[1]),
        "mapped_embedding_dim": int(view["mapped_text_prototypes"].shape[1]),
    }


def _load_text_map_state(output_root: Path) -> Dict[str, Any]:
    manifest_path = _discover_manifest_path(Path(output_root))
    manifest = _load_json(manifest_path, "text_map_manifest.v1.json")
    _require(
        manifest.get("schema_name") == TEXT_MAP_SCHEMA_NAME,
        "text_map_manifest.schema_name",
        f"must equal '{TEXT_MAP_SCHEMA_NAME}'",
    )
    major = _parse_major_version(manifest.get("schema_version"), "text_map_manifest.schema_version")
    _require(major == 1, "text_map_manifest.schema_version", "unsupported major version")
    state_rel = _require_relative_path(manifest.get("state_arrays_path"), "text_map_manifest.state_arrays_path")
    mapped_rel = _require_relative_path(
        manifest.get("mapped_text_manifest_path"), "text_map_manifest.mapped_text_manifest_path"
    )
    state_path = Path(output_root) / Path(state_rel)
    mapped_manifest_path = Path(output_root) / Path(mapped_rel)
    _require(state_path.exists(), "text_map_manifest.state_arrays_path", f"missing file: {state_path}")
    _require(mapped_manifest_path.exists(), "text_map_manifest.mapped_text_manifest_path", f"missing file: {mapped_manifest_path}")
    arrays = np.load(state_path, allow_pickle=False)
    for key in ("A", "text_features", "mapped_text_prototypes"):
        _require(key in arrays.files, f"text_map_state.v1.npz.{key}", "required array missing")
    matrix_a = np.asarray(arrays["A"], dtype=np.float32)
    text_features = np.asarray(arrays["text_features"], dtype=np.float32)
    mapped_text = np.asarray(arrays["mapped_text_prototypes"], dtype=np.float32)
    labels = manifest.get("labels")
    _require(isinstance(labels, list) and labels, "text_map_manifest.labels", "must be a non-empty list")
    _require(text_features.ndim == 2, "text_map_state.v1.npz.text_features", "must be rank-2")
    _require(mapped_text.ndim == 2, "text_map_state.v1.npz.mapped_text_prototypes", "must be rank-2")
    _require(text_features.shape[0] == len(labels), "text_map_state.v1.npz.text_features", "row count must match labels")
    _require(mapped_text.shape[0] == len(labels), "text_map_state.v1.npz.mapped_text_prototypes", "row count must match labels")
    prototype_bank_root_rel = _require_relative_path(
        manifest.get("prototype_bank_root_rel"), "text_map_manifest.prototype_bank_root_rel"
    )
    prototype_bank_root = (Path(output_root) / Path(prototype_bank_root_rel)).resolve()
    prototype_view = load_prototype_bank_v9(prototype_bank_root)
    prototype_summary = summarize_prototype_bank_v9(prototype_bank_root)
    selected_label_id = _normalize_label_id(manifest.get("selected_label_id"), field_path="text_map_manifest.selected_label_id")
    selected_row_index = prototype_view.get_record(selected_label_id).metadata.row_index
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "state_path": state_path,
        "mapped_manifest_path": mapped_manifest_path,
        "A": matrix_a,
        "text_features": text_features,
        "mapped_text_prototypes": mapped_text,
        "prototype_view": prototype_view,
        "prototype_summary": prototype_summary,
        "labels_by_row": labels,
        "selected_row_index": selected_row_index,
    }


def build_text_map_v9_worked_example(
    output_root: Path,
    *,
    selected_label_id: int | str | None = None,
) -> Dict[str, Any]:
    state = _load_text_map_state(output_root)
    prototype_view = state["prototype_view"]
    label_id = selected_label_id if selected_label_id is not None else state["manifest"]["selected_label_id"]
    prototype_record = prototype_view.get_record(label_id)
    row_index = prototype_record.metadata.row_index
    mapped_vector = state["mapped_text_prototypes"][row_index]
    visual_vector = prototype_record.prototype
    similarities = np.asarray(state["mapped_text_prototypes"][row_index] @ prototype_view.prototypes.T, dtype=np.float64)
    ranked_indices = np.argsort(-similarities)
    top_rows = [int(index) for index in ranked_indices[:5]]
    representative_support = min(
        prototype_record.metadata.support_refs,
        key=lambda support: (-float(support.o_tau), -int(support.member_count), str(support.video_id), int(support.global_track_id)),
    )
    prompt_text = state["labels_by_row"][row_index]["prompt_text"]
    diagonal_cosine = float(np.dot(mapped_vector.astype(np.float64), visual_vector.astype(np.float64)))
    return {
        "schema_name": WORKED_EXAMPLE_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "selected_label_id": prototype_record.metadata.label_id,
        "selected_label_text": prototype_record.metadata.label_text,
        "prompt_text": prompt_text,
        "prototype_bank": {
            "support_video_count": int(prototype_record.metadata.support_video_count),
            "support_track_count": int(prototype_record.metadata.support_track_count),
            "support_weight_sum": float(prototype_record.metadata.support_weight_sum),
        },
        "representative_support": {
            "video_id": representative_support.video_id,
            "global_track_id": int(representative_support.global_track_id),
            "representative_source_track_id": representative_support.representative_source_track_id,
            "o_tau": float(representative_support.o_tau),
            "member_count": int(representative_support.member_count),
            "num_active_frames": int(representative_support.num_active_frames),
            "start_frame_idx": int(representative_support.start_frame_idx),
            "end_frame_idx": int(representative_support.end_frame_idx),
        },
        "text_map": {
            "visual_prototype_dim": int(visual_vector.shape[0]),
            "mapped_text_dim": int(mapped_vector.shape[0]),
            "visual_prototype_l2_norm": float(np.linalg.norm(visual_vector)),
            "mapped_text_l2_norm": float(np.linalg.norm(mapped_vector)),
            "diagonal_cosine": diagonal_cosine,
            "top1_label_id": prototype_view._rows[top_rows[0]].label_id,
            "top1_label_text": prototype_view._rows[top_rows[0]].label_text,
            "top1_is_correct": bool(top_rows[0] == row_index),
            "nearest_labels": [
                {
                    "label_id": prototype_view._rows[idx].label_id,
                    "label_text": prototype_view._rows[idx].label_text,
                    "cosine_similarity": float(similarities[idx]),
                    "is_selected_label": bool(idx == row_index),
                }
                for idx in top_rows
            ],
        },
    }


def render_text_map_alignment_svg(
    output_root: Path,
    output_path: Path,
    *,
    selected_label_id: int | str | None = None,
) -> Path:
    worked_example = build_text_map_v9_worked_example(output_root, selected_label_id=selected_label_id)
    bars = worked_example["text_map"]["nearest_labels"]
    width = 960
    height = 320
    margin = 24
    chart_x = 280
    chart_w = width - chart_x - margin
    row_h = 42
    bar_h = 18
    origin_y = 88
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F8F7F2"/>',
        f'<text x="{margin}" y="28" font-family="monospace" font-size="18" fill="#102A43">G5 mapped-text alignment for label {worked_example["selected_label_id"]}</text>',
        f'<text x="{margin}" y="50" font-family="monospace" font-size="13" fill="#486581">{worked_example["selected_label_text"]} | prompt={worked_example["prompt_text"]}</text>',
    ]
    for row_index, row in enumerate(bars):
        ypos = origin_y + (row_index * row_h)
        cosine = max(-1.0, min(1.0, float(row["cosine_similarity"])))
        bar_length = max(4.0, ((cosine + 1.0) / 2.0) * chart_w)
        fill = "#0B6E4F" if row["is_selected_label"] else "#9FB3C8"
        lines.append(
            f'<text x="{margin}" y="{ypos + 14}" font-family="monospace" font-size="12" fill="#102A43">{row["label_id"]}: {row["label_text"]}</text>'
        )
        lines.append(f'<rect x="{chart_x}" y="{ypos}" width="{chart_w}" height="{bar_h}" fill="#E4E7EB" rx="4"/>')
        lines.append(f'<rect x="{chart_x}" y="{ypos}" width="{bar_length:.2f}" height="{bar_h}" fill="{fill}" rx="4"/>')
        lines.append(
            f'<text x="{chart_x + bar_length + 8:.2f}" y="{ypos + 14}" font-family="monospace" font-size="12" fill="#243B53">{cosine:.4f}</text>'
        )
    lines.append("</svg>")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
