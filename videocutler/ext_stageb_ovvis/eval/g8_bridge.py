from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    fuse_carrier_frame_logits_torch,
    load_combined_evidence,
)
from videocutler.ext_stageb_ovvis.algorithms.soft_em import _load_projector_from_checkpoint
from videocutler.ext_stageb_ovvis.banks.carrier_bank import read_carrier_records
from videocutler.ext_stageb_ovvis.banks.text_bank import load_text_vocab as load_text_bank_vocab


ALLOWED_SELECTED_FOR_INFER = ("prealign_only", "base_only", "augmented")
ALLOWED_DATASET_NAMES = ("lvvis_val", "ytvis_2019_val")


@dataclass(frozen=True)
class InferResolution:
    selected_for_infer: str
    checkpoint_path: Path
    source: str
    train_state_path: Path | None
    train_state_payload: Dict[str, Any] | None


@dataclass(frozen=True)
class G8Paths:
    output_root: Path
    dataset_name: str

    @property
    def pred_main_path(self) -> Path:
        return self.output_root / "predictions" / self.dataset_name / "pred_main.json"

    @property
    def pred_diag_path(self) -> Path:
        return self.output_root / "predictions" / self.dataset_name / "pred_diag.json"

    @property
    def external_lvvis_metrics_path(self) -> Path:
        return self.output_root / "eval" / "lvvis" / "external_metrics.lvvis.json"

    @property
    def external_ytvis2019_metrics_path(self) -> Path:
        return self.output_root / "eval" / "ytvis2019" / "external_metrics.ytvis2019.json"

    @property
    def internal_metrics_path(self) -> Path:
        return self.output_root / "eval" / "internal" / "internal_metrics.lvvis.json"

    @property
    def internal_companion_metrics_path(self) -> Path:
        return self.output_root / "eval" / "internal" / "internal_metrics_companion.lvvis.json"


@dataclass(frozen=True)
class InferenceAssetRoots:
    asset_root: Path
    trajectory_records_path: Path
    carrier_records_path: Path
    text_records_path: Path


@dataclass(frozen=True)
class ProjectorBundle:
    projector: Any
    temperature: float
    unknown_logit: float
    checkpoint_path: Path
    stage_id: str
    checkpoint_payload: Dict[str, Any]


Record = Dict[str, Any]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def package_root() -> Path:
    return repo_root() / "package"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Record]:
    rows: List[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_dataset_name(dataset_name: str, *, allowed: tuple[str, ...] = ALLOWED_DATASET_NAMES) -> str:
    if dataset_name not in allowed:
        raise ValueError(f"dataset_name must be one of {allowed}, got {dataset_name!r}")
    return dataset_name


def load_train_infer_handoff_rule() -> Dict[str, Any]:
    return load_json(package_root() / "reference" / "train_infer_handoff_rule.json")


def canonical_checkpoint_relpath(selected_for_infer: str) -> str:
    rule = load_train_infer_handoff_rule()
    allowed = tuple(rule.get("selected_for_infer_allowed_values", []))
    if selected_for_infer not in allowed:
        raise ValueError(f"selected_for_infer must be one of {allowed}, got {selected_for_infer!r}")
    examples = dict(rule["canonical_examples"])
    return str(examples[selected_for_infer])


def resolve_selected_for_infer(output_root: Path, *, ckpt_path: str | None = None) -> InferResolution:
    rule = load_train_infer_handoff_rule()
    if ckpt_path:
        return InferResolution(
            selected_for_infer="augmented",
            checkpoint_path=Path(ckpt_path).expanduser().resolve(),
            source="run_meta_override_only",
            train_state_path=None,
            train_state_payload=None,
        )

    state_relpaths = rule["state_files"]
    selected_hits: list[tuple[Path, str, Dict[str, Any]]] = []
    for relpath in state_relpaths:
        path = output_root / relpath
        if not path.exists():
            continue
        payload = load_json(path)
        selected = str(payload.get("selected_for_infer", "")).strip()
        if selected:
            selected_hits.append((path, selected, payload))

    if not selected_hits:
        raise FileNotFoundError(
            "no train_state.json with explicit selected_for_infer found under output_root; "
            "package forbids implicit infer defaults"
        )

    last_path, selected, payload = selected_hits[-1]
    if selected not in ALLOWED_SELECTED_FOR_INFER:
        raise ValueError(f"invalid selected_for_infer={selected!r} in {last_path}")
    checkpoint_rel = str(payload.get("checkpoint_selected", "")).strip() or canonical_checkpoint_relpath(selected)
    checkpoint_path = (output_root / checkpoint_rel).resolve()
    return InferResolution(
        selected_for_infer=selected,
        checkpoint_path=checkpoint_path,
        source="explicit_train_state_field",
        train_state_path=last_path,
        train_state_payload=payload,
    )


def load_schema(schema_name: str) -> Dict[str, Any]:
    return load_json(package_root() / "schemas" / schema_name)


def validate_json_artifact(payload: Dict[str, Any] | list[Any], schema_name: str) -> None:
    try:
        import jsonschema  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("jsonschema is required to validate G8 canonical artifacts") from exc
    jsonschema.validate(payload, load_schema(schema_name))


def build_cli_contract_summary(cli_contract_relpath: str) -> Dict[str, Any]:
    payload = load_json(package_root() / cli_contract_relpath)
    return {
        "cli_id": payload["cli_id"],
        "entrypoint": payload["entrypoint"],
        "required_args": payload["required_args"],
        "optional_args": payload.get("optional_args", []),
        "canonical_evaluation_policy_ref": payload.get("canonical_evaluation_policy_ref"),
    }


def _trajectory_records_relpath(dataset_name: str, trajectory_source_branch: str) -> Path:
    if trajectory_source_branch == "mainline":
        return Path("exports") / dataset_name / "trajectory_records.jsonl"
    if trajectory_source_branch == "gt_upper_bound":
        return Path("exports_gt") / dataset_name / "trajectory_records.jsonl"
    raise ValueError(f"unsupported trajectory_source_branch: {trajectory_source_branch}")


def _carrier_records_relpath(dataset_name: str, trajectory_source_branch: str) -> Path:
    if trajectory_source_branch == "mainline":
        return Path("carrier_bank") / dataset_name / "carrier_records.jsonl"
    if trajectory_source_branch == "gt_upper_bound":
        return Path("carrier_bank_gt") / dataset_name / "carrier_records.jsonl"
    raise ValueError(f"unsupported trajectory_source_branch: {trajectory_source_branch}")


_REQUIRED_ASSET_RELPATHS = (
    lambda dataset_name, branch: _trajectory_records_relpath(dataset_name, branch),
    lambda dataset_name, branch: _carrier_records_relpath(dataset_name, branch),
    lambda dataset_name, branch: Path("text_bank") / "text_prototype_records.jsonl",
)


def _candidate_asset_roots(output_root: Path, resolution: InferResolution) -> List[Path]:
    candidates: List[Path] = [output_root.resolve()]
    payload = resolution.train_state_payload or {}
    runtime_asset_output_root = str(payload.get("runtime_asset_output_root", "")).strip()
    if runtime_asset_output_root:
        candidates.append(Path(runtime_asset_output_root).expanduser())
    env_output_root = str(os.environ.get("WSOVVIS_AUTHORITATIVE_REMOTE_OUTPUT_ROOT", "")).strip()
    if env_output_root:
        candidates.append(Path(env_output_root).expanduser())
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_inference_asset_roots(
    output_root: Path,
    *,
    dataset_name: str,
    trajectory_source_branch: str,
    resolution: InferResolution,
) -> InferenceAssetRoots:
    errors: List[str] = []
    for root in _candidate_asset_roots(output_root, resolution):
        required_paths = [root / rel_builder(dataset_name, trajectory_source_branch) for rel_builder in _REQUIRED_ASSET_RELPATHS]
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            errors.append(f"{root}: missing {missing}")
            continue
        return InferenceAssetRoots(
            asset_root=root,
            trajectory_records_path=root / _trajectory_records_relpath(dataset_name, trajectory_source_branch),
            carrier_records_path=root / _carrier_records_relpath(dataset_name, trajectory_source_branch),
            text_records_path=root / "text_bank" / "text_prototype_records.jsonl",
        )
    joined = "; ".join(errors) if errors else "no asset roots checked"
    raise FileNotFoundError(f"unable to resolve G8 inference asset root for dataset={dataset_name}: {joined}")


def load_projector_bundle(checkpoint_path: Path, *, device: torch.device) -> ProjectorBundle:
    projector, theta_t, b_u, checkpoint = _load_projector_from_checkpoint(checkpoint_path, device=device)
    projector.eval()
    temperature = float(F.softplus(theta_t.detach()).item() + 1e-4)
    return ProjectorBundle(
        projector=projector,
        temperature=temperature,
        unknown_logit=float(b_u.detach().item()),
        checkpoint_path=checkpoint_path,
        stage_id=str(checkpoint.get("stage_id", "")),
        checkpoint_payload=dict(checkpoint),
    )


def _resolve_lvvis_root() -> Path:
    env_value = str(os.environ.get("WSOVVIS_LVVIS_ROOT", "")).strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (repo_root() / "videocutler" / "datasets" / "LV-VIS").resolve()


def _load_lvvis_annotation_json(split: str) -> Dict[str, Any]:
    root = _resolve_lvvis_root()
    ann_name = "train_instances.json" if split == "train" else "val_instances.json"
    ann_path = root / "annotations" / ann_name
    if not ann_path.is_file():
        raise FileNotFoundError(f"LV-VIS annotation json not found: {ann_path}")
    return load_json(ann_path)


def _load_ytvis2019_annotation_json() -> Dict[str, Any]:
    explicit_root = str(os.environ.get("WSOVVIS_YTVIS2019_ROOT", "")).strip()
    candidates: List[Path] = []
    if explicit_root:
        explicit = Path(explicit_root).expanduser().resolve()
        candidates.extend([explicit / "valid.json", explicit / "ytvis_2019" / "valid.json"])
    datasets_root = Path(os.environ.get("DETECTRON2_DATASETS", repo_root() / "datasets")).expanduser().resolve()
    candidates.append(datasets_root / "ytvis_2019" / "valid.json")
    for ann_path in candidates:
        if ann_path.is_file():
            return load_json(ann_path)
    checked = [str(path) for path in candidates]
    raise FileNotFoundError(f"YTVIS2019 annotation json not found; checked: {checked}")


def load_class_name_map(dataset_name: str) -> Dict[int, str]:
    if dataset_name == "lvvis_val":
        class_map: Dict[int, str] = {}
        for split in ("train", "val"):
            try:
                payload = _load_lvvis_annotation_json(split)
            except FileNotFoundError:
                continue
            for category in payload.get("categories", []):
                raw_id = int(category["id"])
                class_map[raw_id] = str(category.get("name", raw_id))
        if class_map:
            return class_map
    if dataset_name == "ytvis_2019_val":
        try:
            payload = _load_ytvis2019_annotation_json()
        except FileNotFoundError:
            payload = {}
        class_map = {
            int(category["id"]): str(category.get("name", category["id"]))
            for category in payload.get("categories", [])
        }
        if class_map:
            return class_map
    return {}


def load_text_vocab_with_names(asset_root: Path, dataset_name: str) -> Tuple[List[int], List[Record], np.ndarray, Dict[int, str]]:
    raw_ids, records, matrix = load_text_bank_vocab(asset_root)
    class_name_map = load_class_name_map(dataset_name)
    if not class_name_map:
        class_name_map = {int(raw_id): f"raw_id_{int(raw_id)}" for raw_id in raw_ids}
    return raw_ids, records, matrix, class_name_map


def _load_video_meta_lvvis(dataset_name: str) -> Dict[int, Dict[str, int]]:
    if dataset_name != "lvvis_val":
        raise ValueError(f"unsupported LV-VIS dataset name: {dataset_name}")
    payload = _load_lvvis_annotation_json("val")
    meta: Dict[int, Dict[str, int]] = {}
    for video in payload.get("videos", []):
        video_id = int(video["id"])
        file_names = video.get("file_names") or video.get("filenames") or []
        meta[video_id] = {
            "video_id": video_id,
            "clip_id": int(video.get("id", video_id)),
            "length": int(video.get("length", len(file_names))),
            "height": int(video.get("height", 0) or 0),
            "width": int(video.get("width", 0) or 0),
        }
    return meta


def _load_video_meta_ytvis2019() -> Dict[int, Dict[str, int]]:
    root = Path(os.environ.get("DETECTRON2_DATASETS", "datasets")).expanduser()
    ann_path = root / "ytvis_2019" / "valid.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"YTVIS2019 annotation json not found: {ann_path}")
    payload = load_json(ann_path)
    meta: Dict[int, Dict[str, int]] = {}
    for video in payload.get("videos", []):
        video_id = int(video["id"])
        file_names = video.get("file_names") or video.get("filenames") or []
        meta[video_id] = {
            "video_id": video_id,
            "clip_id": int(video.get("id", video_id)),
            "length": int(video.get("length", len(file_names))),
            "height": int(video.get("height", 0) or 0),
            "width": int(video.get("width", 0) or 0),
        }
    return meta


def load_video_meta(dataset_name: str) -> Dict[int, Dict[str, int]]:
    if dataset_name == "lvvis_val":
        return _load_video_meta_lvvis(dataset_name)
    if dataset_name == "ytvis_2019_val":
        return _load_video_meta_ytvis2019()
    raise ValueError(f"unsupported dataset_name for video meta: {dataset_name}")


def build_infer_rows(
    asset_roots: InferenceAssetRoots,
    *,
    dataset_name: str,
) -> Tuple[List[Record], Dict[str, Record], Dict[str, Record]]:
    trajectory_rows = load_jsonl(asset_roots.trajectory_records_path)
    carrier_rows = read_carrier_records(asset_roots.carrier_records_path)
    carrier_by_tid = {str(row.get("trajectory_id", "")): row for row in carrier_rows}

    rows: List[Record] = []
    skipped: Dict[str, int] = {}

    def _bump(reason: str) -> None:
        skipped[reason] = int(skipped.get(reason, 0)) + 1

    for traj in sorted(trajectory_rows, key=lambda row: str(row.get("trajectory_id", ""))):
        trajectory_id = str(traj.get("trajectory_id", "")).strip()
        if not trajectory_id:
            _bump("missing_trajectory_id")
            continue
        carrier_record = carrier_by_tid.get(trajectory_id)
        if carrier_record is None:
            _bump("missing_carrier_record")
            continue
        frame_paths = list(carrier_record.get("frame_carriers_norm_paths", [])) if isinstance(carrier_record, Mapping) else []
        if not frame_paths:
            _bump("missing_runtime_frame_paths")
            continue
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "clip_id": str(traj.get("clip_id", traj.get("video_id", ""))),
                "video_id": int(traj.get("video_id", traj.get("clip_id", 0))),
                "trajectory_record": traj,
                "carrier_record": carrier_record,
            }
        )
    return rows, skipped, {
        "trajectory_count": {"total": len(trajectory_rows), "retained": len(rows)},
        "carrier_count": {"total": len(carrier_rows)},
    }


def _chunk_slices(total: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    if chunk_size <= 0:
        chunk_size = total
    start = 0
    while start < total:
        end = min(total, start + chunk_size)
        yield start, end
        start = end


def compute_fused_logits_chunked(
    *,
    projector: Any,
    carrier_vec: np.ndarray,
    frame_vec: np.ndarray,
    candidate_matrix: np.ndarray,
    temperature: float,
    frame_vectors: Sequence[np.ndarray],
    logit_chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    carrier_parts: List[np.ndarray] = []
    frame_parts: List[np.ndarray] = []
    fused_parts: List[np.ndarray] = []
    total = int(candidate_matrix.shape[0])
    for start, end in _chunk_slices(total, logit_chunk_size):
        chunk = np.asarray(candidate_matrix[start:end], dtype=np.float32)
        carrier_logits_t, frame_logits_t, fused_logits_t = fuse_carrier_frame_logits_torch(
            projector=projector,
            carrier_vec=carrier_vec,
            frame_vec=frame_vec,
            candidate_matrix=chunk,
            temperature=float(temperature),
            frame_vectors=frame_vectors if frame_vectors else None,
        )
        carrier_parts.append(np.asarray(carrier_logits_t.detach().cpu().numpy(), dtype=np.float32))
        frame_parts.append(np.asarray(frame_logits_t.detach().cpu().numpy(), dtype=np.float32))
        fused_parts.append(np.asarray(fused_logits_t.detach().cpu().numpy(), dtype=np.float32))
    return (
        np.concatenate(carrier_parts, axis=0).astype(np.float32) if carrier_parts else np.zeros((0,), dtype=np.float32),
        np.concatenate(frame_parts, axis=0).astype(np.float32) if frame_parts else np.zeros((0,), dtype=np.float32),
        np.concatenate(fused_parts, axis=0).astype(np.float32) if fused_parts else np.zeros((0,), dtype=np.float32),
    )


def score_infer_row(
    row: Mapping[str, Any],
    *,
    bundle: ProjectorBundle,
    asset_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    text_vocab_ids: Sequence[int],
    text_matrix: np.ndarray,
    class_name_map: Mapping[int, str],
    logit_chunk_size: int,
) -> Dict[str, Any]:
    carrier_vec, frame_vectors, frame_vec, _combined = load_combined_evidence(
        row,
        output_root=asset_root,
        dataset_name=dataset_name,
        trajectory_source_branch=trajectory_source_branch,
    )
    _carrier_logits, _frame_logits, fused_logits = compute_fused_logits_chunked(
        projector=bundle.projector,
        carrier_vec=carrier_vec,
        frame_vec=frame_vec,
        candidate_matrix=text_matrix,
        temperature=bundle.temperature,
        frame_vectors=frame_vectors,
        logit_chunk_size=logit_chunk_size,
    )
    if fused_logits.size == 0:
        raise ValueError("empty fused logits")
    all_logits = np.concatenate([fused_logits, np.asarray([bundle.unknown_logit], dtype=np.float32)], axis=0)
    logits_tensor = torch.from_numpy(all_logits.astype(np.float32))
    probs = torch.softmax(logits_tensor, dim=0).detach().cpu().numpy().astype(np.float32)
    known_probs = probs[:-1]
    unknown_prob = float(probs[-1])
    top1_idx = int(np.argmax(fused_logits))
    top1_raw_id = int(text_vocab_ids[top1_idx])
    top1_prob = float(known_probs[top1_idx])
    if fused_logits.size >= 2:
        top2_idx = int(np.argsort(-fused_logits, kind="mergesort")[1])
        margin_top1_top2 = float(fused_logits[top1_idx] - fused_logits[top2_idx])
    else:
        margin_top1_top2 = float(fused_logits[top1_idx] - bundle.unknown_logit)
    margin_top1_vs_unknown = float(fused_logits[top1_idx] - bundle.unknown_logit)
    trajectory_record = row.get("trajectory_record") if isinstance(row.get("trajectory_record"), Mapping) else {}
    generator_score = float(trajectory_record.get("pred_score", 1.0) or 1.0)
    valid_carrier = bool(trajectory_record.get("valid_carrier", True))
    return {
        "trajectory_id": str(row.get("trajectory_id", "")),
        "clip_id": int(row.get("clip_id", row.get("video_id", 0))),
        "video_id": int(row.get("video_id", row.get("clip_id", 0))),
        "generator_score": float(generator_score),
        "score": float(max(0.0, min(1.0, generator_score * top1_prob))),
        "category_id": top1_raw_id,
        "top1_known_raw_id": top1_raw_id,
        "top1_known_name": str(class_name_map.get(top1_raw_id, f"raw_id_{top1_raw_id}")),
        "top1_known_prob": float(top1_prob),
        "unknown_prob": float(unknown_prob),
        "margin_top1_top2": float(margin_top1_top2),
        "margin_top1_vs_unknown": float(margin_top1_vs_unknown),
        "valid_carrier": bool(valid_carrier),
        "trajectory_record": dict(trajectory_record),
    }


def densify_segmentations(trajectory_record: Mapping[str, Any], *, video_length: int) -> List[Any]:
    frame_indices = [int(x) for x in list(trajectory_record.get("frame_indices", []))]
    masks = list(trajectory_record.get("masks_rle", []))
    if len(frame_indices) != len(masks):
        raise ValueError("trajectory frame_indices/masks_rle length mismatch")
    dense: List[Any] = [None for _ in range(max(0, int(video_length)))]
    for frame_index, mask in zip(frame_indices, masks):
        if frame_index < 0:
            raise ValueError(f"negative frame index: {frame_index}")
        if frame_index >= len(dense):
            dense.extend([None for _ in range(frame_index + 1 - len(dense))])
        dense[frame_index] = mask
    return dense


def build_pred_rows(
    scored_rows: Sequence[Mapping[str, Any]],
    *,
    video_meta: Mapping[int, Mapping[str, int]],
) -> Tuple[List[Record], List[Record]]:
    pred_main: List[Record] = []
    pred_diag: List[Record] = []
    for index, row in enumerate(scored_rows):
        video_id = int(row["video_id"])
        meta = video_meta.get(video_id, {})
        trajectory_record = row.get("trajectory_record") if isinstance(row.get("trajectory_record"), Mapping) else {}
        video_length = int(meta.get("length", len(list(trajectory_record.get("frame_indices", [])))))
        pred_main.append(
            {
                "trajectory_id": str(row["trajectory_id"]),
                "video_id": video_id,
                "score": float(row["score"]),
                "category_id": int(row["category_id"]),
                "segmentations": densify_segmentations(trajectory_record, video_length=video_length),
            }
        )
        pred_diag.append(
            {
                "pred_main_index": int(index),
                "trajectory_id": str(row["trajectory_id"]),
                "clip_id": int(row["clip_id"]),
                "video_id": video_id,
                "generator_score": float(row["generator_score"]),
                "top1_known_raw_id": int(row["top1_known_raw_id"]),
                "top1_known_name": str(row["top1_known_name"]),
                "top1_known_prob": float(row["top1_known_prob"]),
                "unknown_prob": float(row["unknown_prob"]),
                "margin_top1_top2": float(row["margin_top1_top2"]),
                "margin_top1_vs_unknown": float(row["margin_top1_vs_unknown"]),
                "valid_carrier": bool(row["valid_carrier"]),
            }
        )
    return pred_main, pred_diag
