from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from wsovvis.tracking import load_global_track_bank_v9


SCHEMA_VERSION = "1.0.0"
MANIFEST_SCHEMA_NAME = "wsovvis.track_dino_semantic_cache"
VIDEO_SCHEMA_NAME = "wsovvis.track_dino_semantic_cache_video"
SUMMARY_SCHEMA_NAME = "wsovvis.track_dino_semantic_cache_summary"
WORKED_EXAMPLE_SCHEMA_NAME = "wsovvis.track_dino_semantic_cache_worked_example"
PROCESSED_STATUSES = {"processed_with_tracks", "processed_zero_tracks"}
ALL_STATUSES = {"processed_with_tracks", "processed_zero_tracks", "failed", "unprocessed"}


class TrackDinoFeatureCacheError(RuntimeError):
    """Raised when the bounded G4 semantic-cache artifact is invalid."""


@dataclass(frozen=True)
class SemanticCacheConfig:
    model_name: str = "dinov2_vitb14"
    dino_repo_path: str = "third_party/dinov2"
    dino_weights_path: str = "weights/DINOv2/dinov2_vitb14_pretrain.pth"
    crop_padding_ratio: float = 0.1
    resize_edge: int = 224
    max_visible_frames_per_track: int = 3
    batch_size: int = 16
    device: str = "auto"
    frame_weighting_rule: str = "visible_area_weighted_mean"
    patch_pooling_rule: str = "mask_aware_patch_mean"
    objectness_formula: str = "sigmoid(mask_score_mean + duration_ratio + temporal_consistency)"

    def canonical_dict(self) -> Dict[str, Any]:
        _require(self.model_name == "dinov2_vitb14", "config.model_name", "must equal 'dinov2_vitb14'")
        _require(
            isinstance(self.dino_repo_path, str) and self.dino_repo_path,
            "config.dino_repo_path",
            "must be a non-empty string",
        )
        _require(
            isinstance(self.dino_weights_path, str) and self.dino_weights_path,
            "config.dino_weights_path",
            "must be a non-empty string",
        )
        _require(
            0.0 <= float(self.crop_padding_ratio) <= 0.5,
            "config.crop_padding_ratio",
            "must be in [0, 0.5]",
        )
        _require(
            isinstance(self.resize_edge, int) and self.resize_edge > 0 and self.resize_edge % 14 == 0,
            "config.resize_edge",
            "must be a positive multiple of 14",
        )
        _require(
            isinstance(self.max_visible_frames_per_track, int) and self.max_visible_frames_per_track >= 1,
            "config.max_visible_frames_per_track",
            "must be integer >= 1",
        )
        _require(
            isinstance(self.batch_size, int) and self.batch_size >= 1,
            "config.batch_size",
            "must be integer >= 1",
        )
        _require(
            isinstance(self.device, str) and self.device in {"auto", "cpu", "cuda"},
            "config.device",
            "must be one of ['auto', 'cpu', 'cuda']",
        )
        _require(
            self.frame_weighting_rule == "visible_area_weighted_mean",
            "config.frame_weighting_rule",
            "must equal 'visible_area_weighted_mean'",
        )
        _require(
            self.patch_pooling_rule == "mask_aware_patch_mean",
            "config.patch_pooling_rule",
            "must equal 'mask_aware_patch_mean'",
        )
        _require(
            self.objectness_formula == "sigmoid(mask_score_mean + duration_ratio + temporal_consistency)",
            "config.objectness_formula",
            "must equal the bounded G4 objectness formula",
        )
        return {
            "model_name": self.model_name,
            "dino_repo_path": self.dino_repo_path,
            "dino_weights_path": self.dino_weights_path,
            "crop_padding_ratio": float(self.crop_padding_ratio),
            "resize_edge": int(self.resize_edge),
            "max_visible_frames_per_track": int(self.max_visible_frames_per_track),
            "batch_size": int(self.batch_size),
            "device": self.device,
            "frame_weighting_rule": self.frame_weighting_rule,
            "patch_pooling_rule": self.patch_pooling_rule,
            "objectness_formula": self.objectness_formula,
            "representative_member_policy": "highest_objectness_then_longest_then_lowest_track_id",
            "frame_selection_rule": "uniform_over_visible_frames_of_representative_member",
            "aggregation_rule": "visible_area_weighted_mean_over_frame_features",
        }


@dataclass(frozen=True)
class TrackCropRequest:
    video_id: str
    global_track_id: int
    source_track_id: str | int
    frame_idx: int
    image_path: Path
    image_width: int
    image_height: int
    segmentation: Any
    mask_area: float
    mask_bbox_xyxy: Tuple[float, float, float, float]
    crop_box_xyxy: Tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackDinoFeatureVideoRecord:
    video_id: str
    status: str
    num_global_tracks: int
    semantic_track_metadata_path: Optional[str]
    semantic_track_arrays_path: Optional[str]


@dataclass(frozen=True)
class TrackDinoFeatureMetadata:
    video_id: str
    row_index: int
    global_track_id: int
    start_frame_idx: int
    end_frame_idx: int
    num_active_frames: int
    member_count: int
    member_track_ids: Tuple[str | int, ...]
    representative_source_track_id: str | int
    source_track_objectness_score: float
    o_tau: float
    mask_score_mean: float
    duration_ratio: float
    temporal_consistency: float
    selected_frame_indices: Tuple[int, ...]
    frame_weights: Tuple[float, ...]
    mask_bbox_xyxy: Tuple[Tuple[float, float, float, float], ...]
    crop_box_xyxy: Tuple[Tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class TrackDinoFeatureRecord:
    metadata: TrackDinoFeatureMetadata
    z_tau: np.ndarray


@dataclass
class _LoadedVideoPayload:
    track_rows: Tuple[TrackDinoFeatureMetadata, ...]
    track_index_by_id: Mapping[int, int]
    embeddings: np.ndarray


@dataclass(frozen=True)
class _VideoContext:
    video_id: str
    length: int
    width: int
    height: int
    file_names: Tuple[str, ...]


def _err(field_path: str, rule_summary: str) -> TrackDinoFeatureCacheError:
    return TrackDinoFeatureCacheError(f"{field_path}: {rule_summary}")


def _require(condition: bool, field_path: str, rule_summary: str) -> None:
    if not condition:
        raise _err(field_path=field_path, rule_summary=rule_summary)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse_major_version(version: Any, field_path: str) -> int:
    _require(isinstance(version, str), field_path, "must be a string")
    parts = version.split(".")
    _require(len(parts) == 3 and all(part.isdigit() for part in parts), field_path, "must follow MAJOR.MINOR.PATCH")
    return int(parts[0])


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_json(path: Path, file_label: str) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrackDinoFeatureCacheError(f"{file_label} missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise TrackDinoFeatureCacheError(f"{file_label} invalid JSON at {path}: {exc}") from exc
    _require(isinstance(payload, dict), f"{file_label}.$", "top-level value must be an object")
    return payload


def _discover_manifest_path(split_root: Path) -> Path:
    manifest_v1 = split_root / "manifest.v1.json"
    manifest_compat = split_root / "manifest.json"
    if manifest_v1.exists():
        return manifest_v1
    if manifest_compat.exists():
        return manifest_compat
    raise TrackDinoFeatureCacheError(
        f"manifest missing under split root {split_root}; expected manifest.v1.json or manifest.json"
    )


def _require_relative_path(path_value: Any, field_path: str) -> str:
    _require(isinstance(path_value, str) and path_value, field_path, "must be a non-empty relative path")
    rel = PurePosixPath(path_value)
    _require(not rel.is_absolute(), field_path, "absolute path is forbidden")
    _require(".." not in rel.parts, field_path, "must not contain '..'")
    return str(rel)


def _canonical_track_id_order(value: str | int) -> Tuple[int, str]:
    if _is_int(value):
        return (0, str(int(value)).zfill(12))
    return (1, str(value))


def _format_track_id_list(values: Sequence[str | int]) -> List[str | int]:
    return [value for value in values]


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=np.float64)
    denom = float(np.linalg.norm(vec))
    if denom == 0.0:
        return np.zeros_like(vec, dtype=np.float32)
    return np.asarray(vec / denom, dtype=np.float32)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value))))


def _uniform_indices(count: int, max_count: int) -> List[int]:
    if count <= max_count:
        return list(range(count))
    if max_count == 1:
        return [count // 2]
    positions = np.linspace(0, count - 1, num=max_count)
    chosen: List[int] = []
    used = set()
    for raw in positions:
        index = int(round(float(raw)))
        while index in used and index + 1 < count:
            index += 1
        while index in used and index - 1 >= 0:
            index -= 1
        used.add(index)
        chosen.append(index)
    chosen.sort()
    return chosen


def _xywh_to_xyxy(xywh: Sequence[float]) -> Tuple[float, float, float, float]:
    _require(len(xywh) == 4, "bbox", "must contain four values")
    x, y, w, h = [float(value) for value in xywh]
    return (x, y, x + max(0.0, w), y + max(0.0, h))


def _pad_bbox_xyxy(
    bbox_xyxy: Tuple[float, float, float, float],
    *,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    crop_x0 = max(0, int(math.floor(x0 - pad_x)))
    crop_y0 = max(0, int(math.floor(y0 - pad_y)))
    crop_x1 = min(int(image_width), int(math.ceil(x1 + pad_x)))
    crop_y1 = min(int(image_height), int(math.ceil(y1 + pad_y)))
    crop_x1 = max(crop_x1, crop_x0 + 1)
    crop_y1 = max(crop_y1, crop_y0 + 1)
    return (crop_x0, crop_y0, crop_x1, crop_y1)


def _resolve_split_paths(run_root: Path, split: str) -> Tuple[Path, Path]:
    config = _load_json(run_root / "config.json", "config.json")
    data = config.get("data")
    _require(isinstance(data, dict), "config.json.data", "must be an object")
    json_key = f"{split}_json"
    img_key = f"{split}_img_root"
    _require(isinstance(data.get(json_key), str) and data[json_key], f"config.json.data.{json_key}", "must be a non-empty string")
    _require(isinstance(data.get(img_key), str) and data[img_key], f"config.json.data.{img_key}", "must be a non-empty string")
    repo_root = run_root
    while repo_root != repo_root.parent:
        if (repo_root / "tools").exists() and (repo_root / "wsovvis").exists():
            break
        repo_root = repo_root.parent
    def _resolve_relative(raw_path: str, field_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path.resolve()
        candidates: List[Path] = [repo_root / path]
        probe = run_root
        while probe != probe.parent:
            candidates.append(probe / path)
            probe = probe.parent
        candidates.append(Path.cwd() / path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve()

    annotation_path = _resolve_relative(data[json_key], f"config.json.data.{json_key}")
    image_root = _resolve_relative(data[img_key], f"config.json.data.{img_key}")
    return annotation_path, image_root


def _load_video_index(run_root: Path, split: str) -> Tuple[Dict[str, _VideoContext], Path]:
    annotation_path, image_root = _resolve_split_paths(run_root, split)
    payload = _load_json(annotation_path, "split_annotation_json")
    videos = payload.get("videos")
    _require(isinstance(videos, list), "split_annotation_json.videos", "must be a list")
    result: Dict[str, _VideoContext] = {}
    for idx, video in enumerate(videos):
        vpath = f"split_annotation_json.videos[{idx}]"
        _require(isinstance(video, dict), vpath, "must be an object")
        _require("id" in video, f"{vpath}.id", "required field missing")
        _require("file_names" in video, f"{vpath}.file_names", "required field missing")
        raw_id = video["id"]
        _require((_is_int(raw_id) or (isinstance(raw_id, str) and raw_id)), f"{vpath}.id", "must be int or non-empty string")
        file_names = video["file_names"]
        _require(isinstance(file_names, list), f"{vpath}.file_names", "must be a list")
        canonical = str(raw_id)
        result[canonical] = _VideoContext(
            video_id=canonical,
            length=int(video.get("length", len(file_names))),
            width=int(video.get("width", 0)),
            height=int(video.get("height", 0)),
            file_names=tuple(str(name) for name in file_names),
        )
    return result, image_root


def _coerce_predictions_json(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("predictions"), list):
            return [row for row in payload["predictions"] if isinstance(row, dict)]
        if isinstance(payload.get("annotations"), list):
            return [row for row in payload["annotations"] if isinstance(row, dict)]
    raise _err("results.json", "unsupported prediction payload shape")


def _load_predictions(run_root: Path) -> Dict[Tuple[str, str | int], Dict[str, Any]]:
    inference_root = run_root / "d2" / "inference"
    pth_path = inference_root / "instances_predictions.pth"
    records: List[Dict[str, Any]]
    if pth_path.exists():
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise _err("instances_predictions.pth", "torch is required to load .pth predictions") from exc
        payload = torch.load(pth_path, map_location="cpu")
        _require(isinstance(payload, list), "instances_predictions.pth", "top-level value must be a list")
        records = [row for row in payload if isinstance(row, dict)]
    else:
        payload = _load_json(inference_root / "results.json", "results.json")
        records = _coerce_predictions_json(payload)

    result: Dict[Tuple[str, str | int], Dict[str, Any]] = {}
    for idx, row in enumerate(records):
        if "video_id" not in row or "track_id" not in row:
            continue
        video_id = str(row["video_id"])
        track_id = row["track_id"]
        _require(
            _is_int(track_id) or (isinstance(track_id, str) and track_id),
            f"predictions[{idx}].track_id",
            "must be integer or non-empty string",
        )
        key = (video_id, track_id)
        _require(key not in result, f"predictions[{idx}]", f"duplicate prediction key {key}")
        result[key] = dict(row)
    return result


def _select_representative_track(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def _key(row: Dict[str, Any]) -> Tuple[float, int, Tuple[int, str]]:
        score = float(row.get("score", row.get("objectness_score", 0.0)))
        active = int(row.get("num_active_frames", 0))
        return (-score, -active, _canonical_track_id_order(row["track_id"]))

    ordered = sorted(candidates, key=_key)
    _require(bool(ordered), "representative_track", "no candidate tracks available")
    return ordered[0]


def _mask_utils():
    try:
        import pycocotools.mask as mask_utils
    except Exception as exc:  # pragma: no cover
        raise _err("pycocotools.mask", "required for segmentation decoding and bbox extraction") from exc
    return mask_utils


def _segmentation_bbox_and_area(segmentation: Any) -> Tuple[Tuple[float, float, float, float], float]:
    mask_utils = _mask_utils()
    bbox = mask_utils.toBbox(segmentation).tolist()
    area_value = mask_utils.area(segmentation)
    if hasattr(area_value, "tolist"):
        area_list = area_value.tolist()
        if isinstance(area_list, list):
            area_float = float(area_list[0])
        else:
            area_float = float(area_list)
    else:
        area_float = float(area_value)
    return _xywh_to_xyxy(bbox), area_float


def _build_crop_requests(
    *,
    video_id: str,
    global_track_id: int,
    representative_track: Dict[str, Any],
    video_ctx: _VideoContext,
    image_root: Path,
    config: SemanticCacheConfig,
) -> List[TrackCropRequest]:
    segmentations = representative_track.get("segmentations")
    _require(isinstance(segmentations, list) and segmentations, "representative_track.segmentations", "must be a non-empty list")
    start_frame_idx = int(representative_track.get("start_frame_idx", 0))
    visible: List[TrackCropRequest] = []
    for offset, segmentation in enumerate(segmentations):
        if not segmentation:
            continue
        frame_idx = start_frame_idx + offset
        _require(
            0 <= frame_idx < len(video_ctx.file_names),
            "representative_track.segmentations",
            f"frame_idx {frame_idx} out of bounds for video {video_id}",
        )
        bbox_xyxy, mask_area = _segmentation_bbox_and_area(segmentation)
        crop_box = _pad_bbox_xyxy(
            bbox_xyxy,
            image_width=video_ctx.width,
            image_height=video_ctx.height,
            padding_ratio=float(config.crop_padding_ratio),
        )
        visible.append(
            TrackCropRequest(
                video_id=video_id,
                global_track_id=global_track_id,
                source_track_id=representative_track["track_id"],
                frame_idx=frame_idx,
                image_path=(image_root / video_ctx.file_names[frame_idx]).resolve(),
                image_width=video_ctx.width,
                image_height=video_ctx.height,
                segmentation=segmentation,
                mask_area=float(mask_area),
                mask_bbox_xyxy=bbox_xyxy,
                crop_box_xyxy=crop_box,
            )
        )
    _require(bool(visible), "representative_track.segmentations", "must contain at least one visible frame")
    selected = _uniform_indices(len(visible), int(config.max_visible_frames_per_track))
    return [visible[index] for index in selected]


class _DefaultDinoFeatureExtractor:
    def __init__(self, *, repo_root: Path, config: SemanticCacheConfig) -> None:
        try:
            import torch
            from PIL import Image
        except Exception as exc:  # pragma: no cover
            raise _err("g4.dino_runtime", "torch and pillow are required for real DINO extraction") from exc

        self._torch = torch
        self._Image = Image
        self._mask_utils = _mask_utils()
        self._config = config
        self._repo_root = repo_root

        repo_path = (repo_root / config.dino_repo_path).resolve()
        weights_path = (repo_root / config.dino_weights_path).resolve()
        _require(repo_path.exists(), "config.dino_repo_path", f"path not found: {repo_path}")
        _require(weights_path.exists(), "config.dino_weights_path", f"path not found: {weights_path}")

        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        _require(device in {"cpu", "cuda"}, "config.device", "resolved device must be cpu or cuda")
        self._device = device

        model = torch.hub.load(str(repo_path), config.model_name, source="local", pretrained=False)
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            elif "state_dict" in state and isinstance(state["state_dict"], dict):
                state = state["state_dict"]
        if isinstance(state, dict) and any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        _require(not unexpected, "dino.load_state_dict", f"unexpected keys present: {sorted(unexpected)[:5]}")
        _require(not missing, "dino.load_state_dict", f"missing keys present: {sorted(missing)[:5]}")
        model.eval()
        model.to(device)
        self._model = model

        patch_size = getattr(getattr(model, "patch_embed", None), "patch_size", 14)
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self._patch_size = int(patch_size)
        self._grid_edge = int(config.resize_edge) // self._patch_size
        _require(self._grid_edge > 0, "config.resize_edge", "must produce a positive patch grid size")
        self._image_mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        self._image_std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)

    def __call__(
        self,
        requests: Sequence[TrackCropRequest],
        repo_root: Path,
        config: SemanticCacheConfig,
    ) -> List[np.ndarray]:
        _require(repo_root.resolve() == self._repo_root.resolve(), "repo_root", "must match initialized DINO extractor repo_root")
        _require(config == self._config, "config", "must match initialized DINO extractor config")
        torch = self._torch
        Image = self._Image

        batches: List[np.ndarray] = []
        patch_weight_rows: List[np.ndarray] = []
        for request in requests:
            image = Image.open(request.image_path).convert("RGB")
            crop = image.crop(tuple(request.crop_box_xyxy)).resize(
                (config.resize_edge, config.resize_edge),
                resample=Image.Resampling.BILINEAR,
            )
            crop_np = np.asarray(crop, dtype=np.float32) / 255.0
            crop_np = (crop_np - self._image_mean) / self._image_std
            batches.append(np.transpose(crop_np, (2, 0, 1)))

            mask = self._mask_utils.decode(request.segmentation)
            if mask.ndim == 3:
                mask = mask[..., 0]
            mask_uint8 = np.asarray(mask, dtype=np.uint8) * 255
            mask_crop = Image.fromarray(mask_uint8, mode="L").crop(tuple(request.crop_box_xyxy)).resize(
                (config.resize_edge, config.resize_edge),
                resample=Image.Resampling.NEAREST,
            )
            patch_mask = mask_crop.resize((self._grid_edge, self._grid_edge), resample=Image.Resampling.BILINEAR)
            patch_weights = np.asarray(patch_mask, dtype=np.float32).reshape(-1) / 255.0
            if float(patch_weights.sum()) <= 0.0:
                patch_weights = np.ones((self._grid_edge * self._grid_edge,), dtype=np.float32)
            patch_weight_rows.append(patch_weights.astype(np.float32))

        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for batch_start in range(0, len(batches), int(config.batch_size)):
                batch_np = np.stack(batches[batch_start : batch_start + int(config.batch_size)], axis=0)
                patch_np = np.stack(patch_weight_rows[batch_start : batch_start + int(config.batch_size)], axis=0)
                images = torch.from_numpy(batch_np).to(device=self._device, dtype=torch.float32)
                patch_weights = torch.from_numpy(patch_np).to(device=self._device, dtype=torch.float32)
                feature_dict = self._model.forward_features(images)
                patch_tokens = feature_dict["x_norm_patchtokens"]
                weights = patch_weights / patch_weights.sum(dim=1, keepdim=True)
                pooled = (patch_tokens * weights.unsqueeze(-1)).sum(dim=1)
                pooled = torch.nn.functional.normalize(pooled, dim=1)
                outputs.extend(np.asarray(row.detach().cpu(), dtype=np.float32) for row in pooled)
        return outputs


def build_track_dino_feature_cache_v9(
    *,
    global_track_bank_root: Path,
    run_root: Path,
    output_split_root: Path,
    overwrite: bool = False,
    config: SemanticCacheConfig | None = None,
    frame_feature_extractor: Optional[Callable[[Sequence[TrackCropRequest], Path, SemanticCacheConfig], Sequence[np.ndarray]]] = None,
) -> Path:
    resolved_config = config or SemanticCacheConfig()
    config_dict = resolved_config.canonical_dict()
    global_view = load_global_track_bank_v9(global_track_bank_root, eager_validate=True)
    video_index, image_root = _load_video_index(run_root=run_root.resolve(), split=global_view.split)
    prediction_index = _load_predictions(run_root.resolve())
    repo_root = run_root.resolve()
    while repo_root != repo_root.parent:
        if (repo_root / "tools").exists() and (repo_root / "wsovvis").exists():
            break
        repo_root = repo_root.parent

    extractor = frame_feature_extractor or _DefaultDinoFeatureExtractor(repo_root=repo_root, config=resolved_config)
    output_split_root = output_split_root.resolve()
    if output_split_root.exists():
        _require(overwrite, "output_split_root", "exists already; pass overwrite=True to replace it")
        shutil.rmtree(output_split_root)

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"track_dino_feature_v9_{global_view.split}_", dir=str(output_split_root.parent))
    )
    videos_payload: List[Dict[str, Any]] = []
    embedding_dim: Optional[int] = None

    try:
        for video in global_view.iter_videos():
            rel_dir = PurePosixPath("videos") / video.video_id
            metadata_rel = str(rel_dir / "semantic_track_metadata.v1.json")
            arrays_rel = str(rel_dir / "semantic_track_arrays.v1.npz")
            metadata_out = temp_dir / metadata_rel
            arrays_out = temp_dir / arrays_rel
            video_ctx = video_index.get(video.video_id)
            _require(video_ctx is not None, "split_annotation_json.videos", f"missing video_id '{video.video_id}' from annotation json")

            if video.status == "processed_zero_tracks":
                _write_video_payload(
                    metadata_path=metadata_out,
                    arrays_path=arrays_out,
                    split=global_view.split,
                    video_id=video.video_id,
                    tracks=[],
                    embeddings=np.zeros((0, 0), dtype=np.float32),
                )
                videos_payload.append(
                    {
                        "video_id": video.video_id,
                        "status": video.status,
                        "num_global_tracks": 0,
                        "semantic_track_metadata_path": metadata_rel,
                        "semantic_track_arrays_path": arrays_rel,
                    }
                )
                continue

            if video.status not in PROCESSED_STATUSES:
                videos_payload.append(
                    {
                        "video_id": video.video_id,
                        "status": video.status,
                        "num_global_tracks": 0,
                        "semantic_track_metadata_path": None,
                        "semantic_track_arrays_path": None,
                    }
                )
                continue

            metadata_rows: List[Dict[str, Any]] = []
            embedding_rows: List[np.ndarray] = []
            for record in global_view.iter_global_tracks(video.video_id):
                prediction_candidates: List[Dict[str, Any]] = []
                for member_track_id in record.metadata.member_track_ids:
                    key = (video.video_id, member_track_id)
                    _require(key in prediction_index, "prediction_index", f"missing source track {key}")
                    prediction_candidates.append(prediction_index[key])
                representative = _select_representative_track(prediction_candidates)
                requests = _build_crop_requests(
                    video_id=video.video_id,
                    global_track_id=int(record.metadata.global_track_id),
                    representative_track=representative,
                    video_ctx=video_ctx,
                    image_root=image_root,
                    config=resolved_config,
                )
                frame_features = extractor(requests, repo_root, resolved_config)
                _require(
                    len(frame_features) == len(requests),
                    "frame_feature_extractor",
                    "must return exactly one feature vector per request",
                )
                per_frame = [np.asarray(feature, dtype=np.float32) for feature in frame_features]
                if embedding_dim is None:
                    _require(per_frame[0].ndim == 1 and per_frame[0].shape[0] > 0, "z_tau", "must be a 1-D non-empty vector")
                    embedding_dim = int(per_frame[0].shape[0])
                for feature in per_frame:
                    _require(feature.ndim == 1 and feature.shape[0] == embedding_dim, "z_tau", "embedding dimension must stay constant")
                    _require(np.isfinite(feature).all(), "z_tau", "feature vector must contain only finite values")

                raw_weights = np.asarray([request.mask_area for request in requests], dtype=np.float64)
                if float(raw_weights.sum()) <= 0.0:
                    raw_weights = np.ones((len(requests),), dtype=np.float64)
                frame_weights = raw_weights / raw_weights.sum()
                z_tau = np.zeros((embedding_dim,), dtype=np.float64)
                for weight, feature in zip(frame_weights, per_frame):
                    z_tau += float(weight) * np.asarray(feature, dtype=np.float64)
                z_tau = _l2_normalize(z_tau)

                duration_ratio = float(record.metadata.num_active_frames) / max(1, int(video_ctx.length))
                support_span = max(1, int(record.metadata.end_frame_idx) - int(record.metadata.start_frame_idx) + 1)
                temporal_consistency = float(record.metadata.num_active_frames) / float(support_span)
                mask_score_mean = float(record.metadata.objectness_score_mean)
                o_tau = _sigmoid(mask_score_mean + duration_ratio + temporal_consistency)

                metadata_rows.append(
                    {
                        "row_index": int(record.metadata.row_index),
                        "global_track_id": int(record.metadata.global_track_id),
                        "start_frame_idx": int(record.metadata.start_frame_idx),
                        "end_frame_idx": int(record.metadata.end_frame_idx),
                        "num_active_frames": int(record.metadata.num_active_frames),
                        "member_count": int(record.metadata.member_count),
                        "member_track_ids": _format_track_id_list(record.metadata.member_track_ids),
                        "representative_source_track_id": representative["track_id"],
                        "source_track_objectness_score": float(representative.get("score", representative.get("objectness_score", 0.0))),
                        "o_tau": float(o_tau),
                        "o_tau_components": {
                            "mask_score_mean": float(mask_score_mean),
                            "duration_ratio": float(duration_ratio),
                            "temporal_consistency": float(temporal_consistency),
                        },
                        "provenance": {
                            "selected_frame_indices": [int(request.frame_idx) for request in requests],
                            "frame_weights": [float(value) for value in frame_weights.tolist()],
                            "mask_bbox_xyxy": [list(map(float, request.mask_bbox_xyxy)) for request in requests],
                            "crop_box_xyxy": [list(map(int, request.crop_box_xyxy)) for request in requests],
                            "crop_padding_ratio": float(resolved_config.crop_padding_ratio),
                            "frame_selection_rule": config_dict["frame_selection_rule"],
                            "patch_pooling_rule": config_dict["patch_pooling_rule"],
                            "frame_weighting_rule": config_dict["frame_weighting_rule"],
                            "image_paths": [
                                str(request.image_path.relative_to(image_root))
                                if request.image_path.is_relative_to(image_root)
                                else str(request.image_path)
                                for request in requests
                            ],
                            "mask_areas": [float(request.mask_area) for request in requests],
                        },
                    }
                )
                embedding_rows.append(z_tau.astype(np.float32))

            if embedding_dim is None:
                embedding_dim = 0
            embeddings_np = np.stack(embedding_rows, axis=0) if embedding_rows else np.zeros((0, embedding_dim), dtype=np.float32)
            _write_video_payload(
                metadata_path=metadata_out,
                arrays_path=arrays_out,
                split=global_view.split,
                video_id=video.video_id,
                tracks=metadata_rows,
                embeddings=embeddings_np,
            )
            videos_payload.append(
                {
                    "video_id": video.video_id,
                    "status": video.status,
                    "num_global_tracks": int(video.num_global_tracks),
                    "semantic_track_metadata_path": metadata_rel,
                    "semantic_track_arrays_path": arrays_rel,
                }
            )

        videos_payload.sort(key=lambda row: str(row["video_id"]))
        manifest = {
            "schema_name": MANIFEST_SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "split": global_view.split,
            "embedding_dim": int(embedding_dim or 0),
            "embedding_dtype": "float32",
            "embedding_normalization": "l2",
            "embedding_pooling": "visible_area_weighted_mean",
            "producer": {
                **config_dict,
                "global_track_bank_ref": str(global_track_bank_root),
                "run_root_ref": str(run_root),
                "annotation_json_ref": str(_resolve_split_paths(run_root, global_view.split)[0]),
                "image_root_ref": str(image_root),
            },
            "videos": videos_payload,
        }
        _dump_json(temp_dir / "manifest.v1.json", manifest)
        load_track_dino_feature_cache_v9(temp_dir, eager_validate=True)
        output_split_root.parent.mkdir(parents=True, exist_ok=True)
        if output_split_root.exists():
            shutil.rmtree(output_split_root)
        temp_dir.replace(output_split_root)
        return output_split_root
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _write_video_payload(
    *,
    metadata_path: Path,
    arrays_path: Path,
    split: str,
    video_id: str,
    tracks: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
) -> None:
    metadata_payload = {
        "schema_name": VIDEO_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "video_id": video_id,
        "num_global_tracks": len(tracks),
        "semantic_tracks": list(tracks),
    }
    _dump_json(metadata_path, metadata_payload)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    row_index = np.asarray([int(track["row_index"]) for track in tracks], dtype=np.int64)
    np.savez(
        arrays_path,
        z_tau=np.asarray(embeddings, dtype=np.float32),
        semantic_track_row_index=row_index,
    )


class TrackDinoFeatureSplitView:
    def __init__(
        self,
        *,
        split_root: Path,
        manifest_path: Path,
        manifest: Mapping[str, Any],
        eager_validate: bool,
    ) -> None:
        self.split_root = split_root
        self.manifest_path = manifest_path
        self.manifest = dict(manifest)
        self.split = str(self.manifest["split"])
        self.embedding_dim = int(self.manifest["embedding_dim"])
        self.embedding_dtype = str(self.manifest["embedding_dtype"])
        self.embedding_normalization = str(self.manifest["embedding_normalization"])
        self.embedding_pooling = str(self.manifest["embedding_pooling"])
        self.producer = dict(self.manifest["producer"])

        self._video_order: Tuple[str, ...] = tuple()
        self._videos_by_id: Dict[str, TrackDinoFeatureVideoRecord] = {}
        self._loaded_by_video_id: Dict[str, _LoadedVideoPayload] = {}
        self._validate_manifest()
        if eager_validate:
            self._eager_validate_processed_videos()

    def _validate_manifest(self) -> None:
        _require(
            self.manifest.get("schema_name") == MANIFEST_SCHEMA_NAME,
            "manifest.schema_name",
            f"must equal '{MANIFEST_SCHEMA_NAME}'",
        )
        major = _parse_major_version(self.manifest.get("schema_version"), "manifest.schema_version")
        _require(major == 1, "manifest.schema_version", "unsupported major version")
        _require(isinstance(self.manifest.get("split"), str) and self.manifest["split"], "manifest.split", "must be non-empty")
        _require(_is_int(self.manifest.get("embedding_dim")) and self.manifest["embedding_dim"] >= 0, "manifest.embedding_dim", "must be integer >= 0")
        _require(self.manifest.get("embedding_dtype") == "float32", "manifest.embedding_dtype", "must equal 'float32'")
        _require(self.manifest.get("embedding_normalization") == "l2", "manifest.embedding_normalization", "must equal 'l2'")
        _require(
            self.manifest.get("embedding_pooling") == "visible_area_weighted_mean",
            "manifest.embedding_pooling",
            "must equal 'visible_area_weighted_mean'",
        )
        _require(isinstance(self.manifest.get("producer"), dict), "manifest.producer", "must be an object")

        videos = self.manifest.get("videos")
        _require(isinstance(videos, list), "manifest.videos", "must be a list")
        seen = set()
        order: List[str] = []
        for index, video in enumerate(videos):
            vpath = f"manifest.videos[{index}]"
            _require(isinstance(video, dict), vpath, "must be an object")
            for key in ("video_id", "status", "num_global_tracks", "semantic_track_metadata_path", "semantic_track_arrays_path"):
                _require(key in video, f"{vpath}.{key}", "required field missing")
            video_id = video["video_id"]
            _require(isinstance(video_id, str) and video_id, f"{vpath}.video_id", "must be a non-empty string")
            _require(video_id not in seen, f"{vpath}.video_id", f"duplicate video_id '{video_id}'")
            seen.add(video_id)
            order.append(video_id)
            status = video["status"]
            _require(status in ALL_STATUSES, f"{vpath}.status", f"must be one of {sorted(ALL_STATUSES)}")
            num_global_tracks = video["num_global_tracks"]
            _require(_is_int(num_global_tracks) and num_global_tracks >= 0, f"{vpath}.num_global_tracks", "must be integer >= 0")
            metadata_path = video["semantic_track_metadata_path"]
            arrays_path = video["semantic_track_arrays_path"]
            if status in PROCESSED_STATUSES:
                metadata_path = _require_relative_path(metadata_path, f"{vpath}.semantic_track_metadata_path")
                arrays_path = _require_relative_path(arrays_path, f"{vpath}.semantic_track_arrays_path")
            else:
                _require(num_global_tracks == 0, f"{vpath}.num_global_tracks", "must equal 0 for failed/unprocessed")
                _require(metadata_path is None, f"{vpath}.semantic_track_metadata_path", "must be null for failed/unprocessed")
                _require(arrays_path is None, f"{vpath}.semantic_track_arrays_path", "must be null for failed/unprocessed")
            self._videos_by_id[video_id] = TrackDinoFeatureVideoRecord(
                video_id=video_id,
                status=status,
                num_global_tracks=num_global_tracks,
                semantic_track_metadata_path=metadata_path,
                semantic_track_arrays_path=arrays_path,
            )
        _require(order == sorted(order), "manifest.videos", "must be lexicographically sorted by video_id")
        self._video_order = tuple(order)

    def _eager_validate_processed_videos(self) -> None:
        for video_id in self._video_order:
            video = self._videos_by_id[video_id]
            if video.status in PROCESSED_STATUSES:
                self._load_processed_video(video)

    def iter_videos(self, include_statuses: Tuple[str, ...] | None = None) -> Iterator[TrackDinoFeatureVideoRecord]:
        accepted = None if include_statuses is None else set(include_statuses)
        if accepted is not None:
            unknown = accepted - ALL_STATUSES
            _require(not unknown, "iter_videos.include_statuses", f"contains unknown statuses: {sorted(unknown)}")
        for video_id in self._video_order:
            video = self._videos_by_id[video_id]
            if accepted is not None and video.status not in accepted:
                continue
            yield video

    def iter_tracks(self, video_id: str) -> Iterator[TrackDinoFeatureRecord]:
        video = self._get_video(video_id)
        if video.status not in PROCESSED_STATUSES:
            return iter(())
        payload = self._load_processed_video(video)
        return (
            TrackDinoFeatureRecord(metadata=meta, z_tau=payload.embeddings[meta.row_index])
            for meta in payload.track_rows
        )

    def get_track_metadata(self, video_id: str, global_track_id: int) -> TrackDinoFeatureMetadata:
        _require(_is_int(global_track_id) and global_track_id >= 0, "global_track_id", "must be integer >= 0")
        video = self._get_video(video_id)
        payload = self._load_processed_video(video)
        _require(global_track_id in payload.track_index_by_id, "global_track_id", f"unknown global_track_id '{global_track_id}'")
        return payload.track_rows[payload.track_index_by_id[global_track_id]]

    def get_track_embedding(self, video_id: str, global_track_id: int) -> np.ndarray:
        metadata = self.get_track_metadata(video_id, global_track_id)
        payload = self._load_processed_video(self._videos_by_id[video_id])
        return payload.embeddings[metadata.row_index]

    def _get_video(self, video_id: str) -> TrackDinoFeatureVideoRecord:
        _require(isinstance(video_id, str) and video_id, "video_id", "must be a non-empty string")
        _require(video_id in self._videos_by_id, "video_id", f"unknown video_id '{video_id}'")
        return self._videos_by_id[video_id]

    def _load_processed_video(self, video: TrackDinoFeatureVideoRecord) -> _LoadedVideoPayload:
        cached = self._loaded_by_video_id.get(video.video_id)
        if cached is not None:
            return cached
        _require(video.semantic_track_metadata_path is not None, "semantic_track_metadata_path", "required for processed video")
        _require(video.semantic_track_arrays_path is not None, "semantic_track_arrays_path", "required for processed video")

        metadata_path = self.split_root / Path(video.semantic_track_metadata_path)
        arrays_path = self.split_root / Path(video.semantic_track_arrays_path)
        metadata = _load_json(metadata_path, "semantic_track_metadata.v1.json")
        _require(metadata.get("schema_name") == VIDEO_SCHEMA_NAME, "semantic_track_metadata.v1.json.schema_name", f"must equal '{VIDEO_SCHEMA_NAME}'")
        major = _parse_major_version(metadata.get("schema_version"), "semantic_track_metadata.v1.json.schema_version")
        _require(major == 1, "semantic_track_metadata.v1.json.schema_version", "unsupported major version")
        _require(metadata.get("split") == self.split, "semantic_track_metadata.v1.json.split", "must match manifest split")
        _require(metadata.get("video_id") == video.video_id, "semantic_track_metadata.v1.json.video_id", "must match manifest video_id")
        tracks = metadata.get("semantic_tracks")
        _require(isinstance(tracks, list), "semantic_track_metadata.v1.json.semantic_tracks", "must be a list")
        _require(len(tracks) == video.num_global_tracks, "semantic_track_metadata.v1.json.semantic_tracks", "length must match manifest num_global_tracks")

        arrays = np.load(arrays_path)
        _require("z_tau" in arrays.files, "semantic_track_arrays.v1.npz", "missing z_tau")
        _require("semantic_track_row_index" in arrays.files, "semantic_track_arrays.v1.npz", "missing semantic_track_row_index")
        embeddings = np.asarray(arrays["z_tau"], dtype=np.float32)
        row_index = np.asarray(arrays["semantic_track_row_index"], dtype=np.int64)
        _require(embeddings.ndim == 2, "semantic_track_arrays.v1.npz.z_tau", "must be rank-2")
        _require(row_index.ndim == 1, "semantic_track_arrays.v1.npz.semantic_track_row_index", "must be rank-1")
        _require(embeddings.shape[0] == len(tracks), "semantic_track_arrays.v1.npz.z_tau", "row count must match metadata")
        _require(row_index.shape[0] == len(tracks), "semantic_track_arrays.v1.npz.semantic_track_row_index", "row count must match metadata")

        rows: List[TrackDinoFeatureMetadata] = []
        track_index_by_id: Dict[int, int] = {}
        seen_global_ids = set()
        for index, track in enumerate(tracks):
            tpath = f"semantic_track_metadata.v1.json.semantic_tracks[{index}]"
            _require(isinstance(track, dict), tpath, "must be an object")
            for key in (
                "row_index",
                "global_track_id",
                "start_frame_idx",
                "end_frame_idx",
                "num_active_frames",
                "member_count",
                "member_track_ids",
                "representative_source_track_id",
                "source_track_objectness_score",
                "o_tau",
                "o_tau_components",
                "provenance",
            ):
                _require(key in track, f"{tpath}.{key}", "required field missing")
            _require(_is_int(track["row_index"]) and track["row_index"] == int(row_index[index]), f"{tpath}.row_index", "must match arrays row_index")
            global_track_id = track["global_track_id"]
            _require(_is_int(global_track_id) and global_track_id >= 0, f"{tpath}.global_track_id", "must be integer >= 0")
            _require(global_track_id not in seen_global_ids, f"{tpath}.global_track_id", f"duplicate global_track_id '{global_track_id}'")
            seen_global_ids.add(global_track_id)
            member_track_ids = track["member_track_ids"]
            _require(isinstance(member_track_ids, list) and len(member_track_ids) == int(track["member_count"]), f"{tpath}.member_track_ids", "must be a list with member_count entries")
            provenance = track["provenance"]
            _require(isinstance(provenance, dict), f"{tpath}.provenance", "must be an object")
            frame_indices = provenance.get("selected_frame_indices")
            frame_weights = provenance.get("frame_weights")
            mask_bbox_xyxy = provenance.get("mask_bbox_xyxy")
            crop_box_xyxy = provenance.get("crop_box_xyxy")
            _require(isinstance(frame_indices, list), f"{tpath}.provenance.selected_frame_indices", "must be a list")
            _require(isinstance(frame_weights, list) and len(frame_weights) == len(frame_indices), f"{tpath}.provenance.frame_weights", "must match selected_frame_indices length")
            _require(isinstance(mask_bbox_xyxy, list) and len(mask_bbox_xyxy) == len(frame_indices), f"{tpath}.provenance.mask_bbox_xyxy", "must match selected_frame_indices length")
            _require(isinstance(crop_box_xyxy, list) and len(crop_box_xyxy) == len(frame_indices), f"{tpath}.provenance.crop_box_xyxy", "must match selected_frame_indices length")
            rows.append(
                TrackDinoFeatureMetadata(
                    video_id=video.video_id,
                    row_index=int(track["row_index"]),
                    global_track_id=int(global_track_id),
                    start_frame_idx=int(track["start_frame_idx"]),
                    end_frame_idx=int(track["end_frame_idx"]),
                    num_active_frames=int(track["num_active_frames"]),
                    member_count=int(track["member_count"]),
                    member_track_ids=tuple(member_track_ids),
                    representative_source_track_id=track["representative_source_track_id"],
                    source_track_objectness_score=float(track["source_track_objectness_score"]),
                    o_tau=float(track["o_tau"]),
                    mask_score_mean=float(track["o_tau_components"]["mask_score_mean"]),
                    duration_ratio=float(track["o_tau_components"]["duration_ratio"]),
                    temporal_consistency=float(track["o_tau_components"]["temporal_consistency"]),
                    selected_frame_indices=tuple(int(value) for value in frame_indices),
                    frame_weights=tuple(float(value) for value in frame_weights),
                    mask_bbox_xyxy=tuple(tuple(float(item) for item in row) for row in mask_bbox_xyxy),
                    crop_box_xyxy=tuple(tuple(int(item) for item in row) for row in crop_box_xyxy),
                )
            )
            track_index_by_id[int(global_track_id)] = index

        payload = _LoadedVideoPayload(
            track_rows=tuple(rows),
            track_index_by_id=track_index_by_id,
            embeddings=embeddings,
        )
        self._loaded_by_video_id[video.video_id] = payload
        return payload


def load_track_dino_feature_cache_v9(split_root: Path, *, eager_validate: bool = True) -> TrackDinoFeatureSplitView:
    manifest_path = _discover_manifest_path(split_root)
    manifest = _load_json(manifest_path, "manifest.v1.json")
    return TrackDinoFeatureSplitView(
        split_root=split_root,
        manifest_path=manifest_path,
        manifest=manifest,
        eager_validate=eager_validate,
    )


def summarize_track_dino_feature_cache_v9(split_root: Path) -> Dict[str, Any]:
    view = load_track_dino_feature_cache_v9(split_root, eager_validate=True)
    objectness_values: List[float] = []
    feature_norms: List[float] = []
    member_counts: List[int] = []
    processed_videos = 0
    total_tracks = 0
    selected_video_id: Optional[str] = None
    selected_global_track_id: Optional[int] = None
    best_selection_key: Optional[Tuple[int, float, str, int]] = None
    for video in view.iter_videos():
        if video.status not in PROCESSED_STATUSES:
            continue
        processed_videos += 1
        for record in view.iter_tracks(video.video_id):
            total_tracks += 1
            objectness_values.append(float(record.metadata.o_tau))
            feature_norms.append(float(np.linalg.norm(record.z_tau)))
            member_counts.append(int(record.metadata.member_count))
            selection_key = (
                int(record.metadata.member_count),
                float(record.metadata.o_tau),
                record.video_id,
                int(record.metadata.global_track_id),
            )
            if best_selection_key is None or selection_key > best_selection_key:
                best_selection_key = selection_key
                selected_video_id = record.video_id
                selected_global_track_id = int(record.metadata.global_track_id)

    objectness_np = np.asarray(objectness_values, dtype=np.float64) if objectness_values else np.zeros((0,), dtype=np.float64)
    norms_np = np.asarray(feature_norms, dtype=np.float64) if feature_norms else np.zeros((0,), dtype=np.float64)
    return {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split": view.split,
        "num_videos_total": len(tuple(view.iter_videos())),
        "num_processed_videos": processed_videos,
        "num_global_tracks_total": total_tracks,
        "semantic_cache_coverage": {
            "tracks_with_z_tau": total_tracks,
            "coverage_ratio": 1.0 if total_tracks > 0 else 0.0,
            "embedding_dim": int(view.embedding_dim),
        },
        "objectness_distribution": {
            "count": int(objectness_np.size),
            "min": float(objectness_np.min()) if objectness_np.size else 0.0,
            "max": float(objectness_np.max()) if objectness_np.size else 0.0,
            "mean": float(objectness_np.mean()) if objectness_np.size else 0.0,
            "median": float(np.median(objectness_np)) if objectness_np.size else 0.0,
            "p10": float(np.percentile(objectness_np, 10)) if objectness_np.size else 0.0,
            "p90": float(np.percentile(objectness_np, 90)) if objectness_np.size else 0.0,
        },
        "feature_norm_distribution": {
            "count": int(norms_np.size),
            "min": float(norms_np.min()) if norms_np.size else 0.0,
            "max": float(norms_np.max()) if norms_np.size else 0.0,
            "mean": float(norms_np.mean()) if norms_np.size else 0.0,
        },
        "member_count_distribution": {
            "videos_with_tracks": processed_videos,
            "tracks_with_merged_members": int(sum(1 for value in member_counts if value > 1)),
            "max_member_count": int(max(member_counts) if member_counts else 0),
        },
        "selected_video_id": selected_video_id,
        "selected_global_track_id": selected_global_track_id,
    }


def build_track_dino_feature_cache_v9_worked_example(
    split_root: Path,
    *,
    selected_video_id: Optional[str] = None,
    selected_global_track_id: Optional[int] = None,
) -> Dict[str, Any]:
    view = load_track_dino_feature_cache_v9(split_root, eager_validate=True)
    summary = summarize_track_dino_feature_cache_v9(split_root)
    video_id = selected_video_id or summary["selected_video_id"]
    _require(isinstance(video_id, str) and video_id, "selected_video_id", "must resolve to a non-empty string")
    if selected_global_track_id is None:
        selected_global_track_id = int(summary["selected_global_track_id"])
    metadata = view.get_track_metadata(video_id, int(selected_global_track_id))
    embedding = view.get_track_embedding(video_id, int(selected_global_track_id))
    return {
        "schema_name": WORKED_EXAMPLE_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "selected_video_id": video_id,
        "selected_global_track_id": int(selected_global_track_id),
        "global_track": {
            "global_track_id": int(metadata.global_track_id),
            "member_track_ids": _format_track_id_list(metadata.member_track_ids),
            "member_count": int(metadata.member_count),
            "start_frame_idx": int(metadata.start_frame_idx),
            "end_frame_idx": int(metadata.end_frame_idx),
            "num_active_frames": int(metadata.num_active_frames),
        },
        "semantic_carrier": {
            "representative_source_track_id": metadata.representative_source_track_id,
            "z_tau_dim": int(embedding.shape[0]),
            "z_tau_l2_norm": float(np.linalg.norm(embedding)),
            "o_tau": float(metadata.o_tau),
            "o_tau_components": {
                "mask_score_mean": float(metadata.mask_score_mean),
                "duration_ratio": float(metadata.duration_ratio),
                "temporal_consistency": float(metadata.temporal_consistency),
            },
        },
        "provenance": {
            "selected_frame_indices": [int(value) for value in metadata.selected_frame_indices],
            "frame_weights": [float(value) for value in metadata.frame_weights],
            "mask_bbox_xyxy": [list(row) for row in metadata.mask_bbox_xyxy],
            "crop_box_xyxy": [list(row) for row in metadata.crop_box_xyxy],
            "source_track_objectness_score": float(metadata.source_track_objectness_score),
            "pooling_rule": view.producer["patch_pooling_rule"],
            "frame_weighting_rule": view.producer["frame_weighting_rule"],
            "crop_padding_ratio": float(view.producer["crop_padding_ratio"]),
        },
    }


def render_track_dino_feature_provenance_svg(
    split_root: Path,
    output_path: Path,
    *,
    selected_video_id: Optional[str] = None,
    selected_global_track_id: Optional[int] = None,
) -> Path:
    worked_example = build_track_dino_feature_cache_v9_worked_example(
        split_root,
        selected_video_id=selected_video_id,
        selected_global_track_id=selected_global_track_id,
    )
    mask_boxes = worked_example["provenance"]["mask_bbox_xyxy"]
    crop_boxes = worked_example["provenance"]["crop_box_xyxy"]
    frame_indices = worked_example["provenance"]["selected_frame_indices"]
    frame_weights = worked_example["provenance"]["frame_weights"]
    width = 1024
    height = 260
    card_w = 300
    card_h = 180
    margin = 20
    title = (
        f"G4 crop/pooling provenance for video {worked_example['selected_video_id']} "
        f"global track {worked_example['selected_global_track_id']}"
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#F7F7F2"/>',
        f'<text x="{margin}" y="24" font-family="monospace" font-size="16" fill="#102A43">{title}</text>',
    ]
    for index, (frame_idx, weight, mask_box, crop_box) in enumerate(zip(frame_indices, frame_weights, mask_boxes, crop_boxes)):
        origin_x = margin + index * (card_w + 16)
        origin_y = 48
        parts.append(f'<rect x="{origin_x}" y="{origin_y}" width="{card_w}" height="{card_h}" rx="12" fill="#FFFFFF" stroke="#CBD2D9"/>')
        parts.append(f'<text x="{origin_x + 12}" y="{origin_y + 20}" font-family="monospace" font-size="13" fill="#243B53">frame {frame_idx}</text>')
        parts.append(f'<text x="{origin_x + 12}" y="{origin_y + 38}" font-family="monospace" font-size="12" fill="#486581">weight={weight:.3f}</text>')
        frame_box_x = origin_x + 12
        frame_box_y = origin_y + 52
        frame_box_w = card_w - 24
        frame_box_h = card_h - 70
        parts.append(
            f'<rect x="{frame_box_x}" y="{frame_box_y}" width="{frame_box_w}" height="{frame_box_h}" fill="#E4E7EB" stroke="#9FB3C8"/>'
        )
        crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
        mask_x0, mask_y0, mask_x1, mask_y1 = mask_box
        denom_x = max(1.0, float(crop_x1 - crop_x0))
        denom_y = max(1.0, float(crop_y1 - crop_y0))
        inner_x = frame_box_x + ((float(mask_x0) - float(crop_x0)) / denom_x) * frame_box_w
        inner_y = frame_box_y + ((float(mask_y0) - float(crop_y0)) / denom_y) * frame_box_h
        inner_w = max(2.0, ((float(mask_x1) - float(mask_x0)) / denom_x) * frame_box_w)
        inner_h = max(2.0, ((float(mask_y1) - float(mask_y0)) / denom_y) * frame_box_h)
        parts.append(f'<rect x="{frame_box_x}" y="{frame_box_y}" width="{frame_box_w}" height="{frame_box_h}" fill="none" stroke="#D64545" stroke-width="2"/>')
        parts.append(f'<rect x="{inner_x}" y="{inner_y}" width="{inner_w}" height="{inner_h}" fill="rgba(31, 119, 180, 0.18)" stroke="#1F77B4" stroke-width="2"/>')
        parts.append(f'<text x="{origin_x + 12}" y="{origin_y + card_h - 10}" font-family="monospace" font-size="11" fill="#486581">crop={list(crop_box)}</text>')
    parts.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return output_path
