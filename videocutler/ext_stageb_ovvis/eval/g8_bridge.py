from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from videocutler.ext_stageb_ovvis.algorithms._g7_semantics import (
    _coerce_temperature_tensor,
    _project_candidate_matrix,
    _resolve_module_device,
    load_carrier_evidence,
    score_carrier_logits_torch,
)
from tqdm.auto import tqdm
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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + '\n')


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
        checkpoint_path = Path(ckpt_path).expanduser().resolve()
        checkpoint_payload: Dict[str, Any] | None = None
        selected_for_infer = "augmented"
        try:
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
            pipeline = str(checkpoint_payload.get("pipeline", "")).strip()
            label_source = str(checkpoint_payload.get("label_source", "")).strip()
            if bool(checkpoint_payload.get("vc_full_y_validation", False)) or pipeline == "videocutler_full_y_clean" or label_source == "full_Y_base":
                selected_for_infer = "prealign_only"
        except Exception:
            checkpoint_payload = None
        return InferResolution(
            selected_for_infer=selected_for_infer,
            checkpoint_path=checkpoint_path,
            source="run_meta_override_only",
            train_state_path=None,
            train_state_payload=checkpoint_payload,
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




def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_npz_first_array(path: Path) -> Tuple[np.ndarray, str]:
    z = np.load(path)
    keys = list(z.keys())
    if not keys:
        raise RuntimeError(f"empty npz payload: {path}")
    for key in ("protos", "features", "arr_0", "llama_hidden_mean", "clip_of_llm_mean", "llama_direct_concept_mean"):
        if key in z:
            return np.asarray(z[key]), str(key)
    key0 = str(keys[0])
    return np.asarray(z[key0]), key0


def _safe_int_for_text_bank(x: Any, default: int | None = None) -> int | None:
    try:
        if x is None or x == "":
            return default
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default


def _load_external_text_bank_from_checkpoint_payload(
    checkpoint_payload: Mapping[str, Any],
) -> Tuple[List[int], np.ndarray, Dict[int, str], Dict[str, Any]] | None:
    """Load the external A8 text bank declared by a checkpoint.

    This is intentionally checkpoint-driven: inference/evaluation must not accept
    a separate text-bank CLI value that could drift from the trained projector.
    It verifies payload/manifest hashes recorded at train time when present.
    """
    tb_raw = checkpoint_payload.get("text_bank", {})
    tb = tb_raw if isinstance(tb_raw, Mapping) else {}
    variant = str(tb.get("variant", "clip_current") or "clip_current")
    if variant == "clip_current":
        return None

    root_s = str(tb.get("root", "")).strip()
    if not root_s:
        raise RuntimeError(f"checkpoint text_bank variant={variant!r} has empty root")
    root = Path(root_s).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint requested external text bank root but it is missing: {root}")

    class_path = root / "lvvis_class_names.json"
    if not class_path.is_file():
        raise FileNotFoundError(class_path)
    payload = json.loads(class_path.read_text(encoding="utf-8"))
    rows = payload.get("classes", payload if isinstance(payload, list) else [])
    ids: List[int] = []
    names: Dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        rid = _safe_int_for_text_bank(row.get("raw_id"))
        if rid is None:
            continue
        ids.append(int(rid))
        names[int(rid)] = str(row.get("name", row.get("class_name", rid)))
    if not ids:
        raise RuntimeError(f"no class ids in external text bank: {class_path}")
    if ids != sorted(ids):
        raise RuntimeError(f"external text bank raw ids are not ascending: {class_path}")

    payload_path_s = str(tb.get("payload_path", "")).strip()
    payload_path = Path(payload_path_s).expanduser().resolve() if payload_path_s else Path()
    if not payload_path.is_file():
        fallback = {
            "clip_of_llm_mean": root / "payload" / "clip_of_llm_mean.fp16.npz",
            "llama_hidden_mean": root / "payload" / "llama_hidden_mean.fp16.npz",
            "llama_direct_concept_mean": root / "payload" / "llama_direct_concept_mean.fp16.npz",
        }.get(variant)
        if fallback is None or not fallback.is_file():
            raise FileNotFoundError(f"missing external text bank payload for {variant}: {payload_path}")
        payload_path = fallback

    expected_payload_sha = str(tb.get("payload_sha256", "")).strip()
    actual_payload_sha = _sha256_file(payload_path)
    if expected_payload_sha and actual_payload_sha != expected_payload_sha:
        raise RuntimeError(
            f"external text bank payload sha256 mismatch for {variant}: "
            f"expected={expected_payload_sha} actual={actual_payload_sha} path={payload_path}"
        )

    manifest_path_s = str(tb.get("manifest_path", "")).strip()
    manifest_path = Path(manifest_path_s).expanduser().resolve() if manifest_path_s else (root / "manifest.json")
    expected_manifest_sha = str(tb.get("manifest_sha256", "")).strip()
    actual_manifest_sha = ""
    if manifest_path.is_file():
        actual_manifest_sha = _sha256_file(manifest_path)
        if expected_manifest_sha and actual_manifest_sha != expected_manifest_sha:
            raise RuntimeError(
                f"external text bank manifest sha256 mismatch for {variant}: "
                f"expected={expected_manifest_sha} actual={actual_manifest_sha} path={manifest_path}"
            )

    arr, arr_key = _load_npz_first_array(payload_path)
    if arr.ndim != 2 or int(arr.shape[0]) != len(ids):
        raise RuntimeError(f"invalid external text bank payload shape={arr.shape}; class_count={len(ids)}")
    arr = np.asarray(arr, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"non-finite external text bank payload: {payload_path}")
    arr = arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)

    summary = dict(tb)
    summary.update({
        "status": "PASS",
        "loaded_by_checkpoint_text_bank_loader": True,
        "variant": variant,
        "root": str(root),
        "payload_path": str(payload_path),
        "payload_array_key": str(arr_key),
        "payload_sha256": actual_payload_sha,
        "payload_sha256_verified_against_checkpoint": bool(expected_payload_sha),
        "manifest_path": str(manifest_path) if manifest_path else "",
        "manifest_sha256": actual_manifest_sha or expected_manifest_sha,
        "manifest_sha256_verified_against_checkpoint": bool(expected_manifest_sha and actual_manifest_sha),
        "feature_dim": int(arr.shape[1]),
        "class_count": int(len(ids)),
        "replaces_only_text_anchor_source": True,
    })
    return ids, arr, names, summary


def load_text_vocab_for_checkpoint(
    asset_root: Path,
    dataset_name: str,
    checkpoint_payload: Mapping[str, Any],
) -> Tuple[List[int], List[Record], np.ndarray, Dict[int, str], Dict[str, Any]]:
    """Load the exact text vocab/matrix that must be paired with a checkpoint.

    For normal CLIP checkpoints this returns the canonical text bank.  For A8
    external-text-bank checkpoints it loads the checkpoint-declared payload and
    verifies train-time hashes, preventing AP/rank audits from silently falling
    back to the current CLIP text bank.
    """
    external = _load_external_text_bank_from_checkpoint_payload(checkpoint_payload)
    if external is None:
        raw_ids, records, matrix, class_name_map = load_text_vocab_with_names(asset_root, dataset_name)
        return raw_ids, records, matrix, class_name_map, {
            "variant": "clip_current",
            "status": "PASS",
            "loaded_by_checkpoint_text_bank_loader": False,
        }
    ids, matrix, names, summary = external
    _canon_ids, records, _canon_matrix, class_name_map = load_text_vocab_with_names(asset_root, dataset_name)
    class_name_map = dict(class_name_map)
    class_name_map.update({int(k): str(v) for k, v in names.items()})
    # Preserve record shape for callers that only need ids/names; external matrix
    # is authoritative for scoring.
    return ids, records, matrix, class_name_map, summary

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
    total = int(candidate_matrix.shape[0])
    for start, end in _chunk_slices(total, logit_chunk_size):
        chunk = np.asarray(candidate_matrix[start:end], dtype=np.float32)
        carrier_logits_t = score_carrier_logits_torch(
            projector=projector,
            carrier_vec=carrier_vec,
            candidate_matrix=chunk,
            temperature=float(temperature),
        )
        carrier_parts.append(np.asarray(carrier_logits_t.detach().cpu().numpy(), dtype=np.float32))
    logits = np.concatenate(carrier_parts, axis=0).astype(np.float32) if carrier_parts else np.zeros((0,), dtype=np.float32)
    return logits, logits, logits


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
    carrier_vec = load_carrier_evidence(
        row,
        output_root=asset_root,
        dataset_name=dataset_name,
        trajectory_source_branch=trajectory_source_branch,
    )
    _carrier_logits, _frame_logits, fused_logits = compute_fused_logits_chunked(
        projector=bundle.projector,
        carrier_vec=carrier_vec,
        frame_vec=np.asarray(carrier_vec, dtype=np.float32),
        candidate_matrix=text_matrix,
        temperature=bundle.temperature,
        frame_vectors=(),
        logit_chunk_size=logit_chunk_size,
    )
    if fused_logits.size == 0:
        raise ValueError('empty fused logits')
    all_logits = np.concatenate([fused_logits, np.asarray([bundle.unknown_logit], dtype=np.float32)], axis=0)
    logits_tensor = torch.from_numpy(all_logits.astype(np.float32))
    probs = torch.softmax(logits_tensor, dim=0).detach().cpu().numpy().astype(np.float32)
    known_probs = probs[:-1]
    unknown_prob = float(probs[-1])
    top1_idx = int(np.argmax(fused_logits))
    top1_raw_id = int(text_vocab_ids[top1_idx])
    top1_prob = float(known_probs[top1_idx])
    if fused_logits.size >= 2:
        top2_idx = int(np.argsort(-fused_logits, kind='mergesort')[1])
        margin_top1_top2 = float(fused_logits[top1_idx] - fused_logits[top2_idx])
    else:
        margin_top1_top2 = float(fused_logits[top1_idx] - bundle.unknown_logit)
    margin_top1_vs_unknown = float(fused_logits[top1_idx] - bundle.unknown_logit)
    trajectory_record = row.get('trajectory_record') if isinstance(row.get('trajectory_record'), Mapping) else {}
    generator_score = float(trajectory_record.get('pred_score', 1.0) or 1.0)
    valid_carrier = bool(trajectory_record.get('valid_carrier', True))
    return {
        'trajectory_id': str(row.get('trajectory_id', '')),
        'clip_id': int(row.get('clip_id', row.get('video_id', 0))),
        'video_id': int(row.get('video_id', row.get('clip_id', 0))),
        'generator_score': float(generator_score),
        'score': float(max(0.0, min(1.0, generator_score * top1_prob))),
        'category_id': top1_raw_id,
        'top1_known_raw_id': top1_raw_id,
        'top1_known_name': str(class_name_map.get(top1_raw_id, f'raw_id_{top1_raw_id}')),
        'top1_known_prob': float(top1_prob),
        'unknown_prob': float(unknown_prob),
        'margin_top1_top2': float(margin_top1_top2),
        'margin_top1_vs_unknown': float(margin_top1_vs_unknown),
        'valid_carrier': bool(valid_carrier),
        'trajectory_record': dict(trajectory_record),
    }


def build_carrier_only_infer_pack(
    infer_rows: Sequence[Mapping[str, Any]],
    *,
    asset_root: Path,
    dataset_name: str,
    trajectory_source_branch: str,
    show_progress: bool = True,
) -> Dict[str, Any]:
    carrier_vectors: List[np.ndarray] = []
    row_manifest: List[Record] = []
    trajectory_records: List[Dict[str, Any]] = []
    iterator = tqdm(infer_rows, desc='infer: load carrier', unit='traj', leave=True) if show_progress else infer_rows
    try:
        for row_idx, row in enumerate(iterator):
            carrier_vec = load_carrier_evidence(
                row,
                output_root=asset_root,
                dataset_name=dataset_name,
                trajectory_source_branch=trajectory_source_branch,
            )
            carrier_vectors.append(np.asarray(carrier_vec, dtype=np.float32))
            trajectory_record = dict(row.get('trajectory_record', {})) if isinstance(row.get('trajectory_record'), Mapping) else {}
            trajectory_records.append(trajectory_record)
            row_manifest.append({
                'row_idx': int(row_idx),
                'trajectory_id': str(row.get('trajectory_id', '')),
                'join_key': str(row.get('trajectory_id', '')),
                'clip_id': int(row.get('clip_id', row.get('video_id', 0))),
                'video_id': int(row.get('video_id', row.get('clip_id', 0))),
                'generator_score': float(trajectory_record.get('pred_score', 1.0) or 1.0),
                'valid_carrier': bool(trajectory_record.get('valid_carrier', True)),
            })
    finally:
        if show_progress and hasattr(iterator, 'close'):
            iterator.close()
    if not carrier_vectors:
        raise ValueError('no carrier vectors for inference')
    carrier_matrix = np.stack(carrier_vectors, axis=0).astype(np.float32)
    return {
        'carrier_matrix': carrier_matrix,
        'row_manifest': row_manifest,
        'trajectory_records': trajectory_records,
    }


def score_infer_rows_matrix(
    *,
    carrier_matrix: np.ndarray,
    bundle: ProjectorBundle,
    text_matrix: np.ndarray,
    show_progress: bool = True,
) -> Dict[str, np.ndarray]:
    progress = tqdm(total=3, desc='infer: matrix score', unit='step', leave=True) if show_progress else None
    try:
        device = _resolve_module_device(bundle.projector)
        carrier_tensor = torch.from_numpy(np.asarray(carrier_matrix, dtype=np.float32)).to(device=device, dtype=torch.float32)
        carrier_tensor = F.normalize(carrier_tensor, p=2.0, dim=-1)
        if progress is not None:
            progress.set_postfix_str('carrier tensor ready')
            progress.update(1)
        candidate_tensor = _project_candidate_matrix(projector=bundle.projector, candidate_matrix=text_matrix, device=device)
        if progress is not None:
            progress.set_postfix_str('text tensor projected')
            progress.update(1)
        temperature_tensor = _coerce_temperature_tensor(bundle.temperature, device=device)
        fused_logits_t = torch.matmul(carrier_tensor, candidate_tensor.t()) / temperature_tensor
        unknown_col = torch.full((int(fused_logits_t.shape[0]), 1), float(bundle.unknown_logit), device=device, dtype=fused_logits_t.dtype)
        probs_t = torch.softmax(torch.cat([fused_logits_t, unknown_col], dim=1), dim=1)
        if progress is not None:
            progress.set_postfix_str('logits/probs ready')
            progress.update(1)
        return {
            'fused_logits': np.asarray(fused_logits_t.detach().cpu().numpy(), dtype=np.float32),
            'known_probs': np.asarray(probs_t[:, :-1].detach().cpu().numpy(), dtype=np.float32),
            'unknown_probs': np.asarray(probs_t[:, -1].detach().cpu().numpy(), dtype=np.float32),
        }
    finally:
        if progress is not None:
            progress.close()


def materialize_scored_rows_from_matrix(
    *,
    row_manifest: Sequence[Mapping[str, Any]],
    trajectory_records: Sequence[Mapping[str, Any]],
    text_vocab_ids: Sequence[int],
    class_name_map: Mapping[int, str],
    fused_logits: np.ndarray,
    known_probs: np.ndarray,
    unknown_probs: np.ndarray,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    if int(fused_logits.shape[0]) != len(row_manifest):
        raise ValueError('row_manifest length does not match fused_logits rows')
    if int(known_probs.shape[0]) != len(row_manifest) or int(unknown_probs.shape[0]) != len(row_manifest):
        raise ValueError('probability rows do not match row_manifest length')
    if int(fused_logits.shape[1]) != len(text_vocab_ids):
        raise ValueError('text vocab axis does not match fused_logits width')
    rows: List[Dict[str, Any]] = []
    iterator = tqdm(range(len(row_manifest)), desc='infer: materialize rows', unit='traj', leave=True) if show_progress else range(len(row_manifest))
    try:
        for row_idx in iterator:
            manifest = row_manifest[row_idx]
            trajectory_record = dict(trajectory_records[row_idx]) if row_idx < len(trajectory_records) else {}
            row_logits = np.asarray(fused_logits[row_idx], dtype=np.float32)
            row_known_probs = np.asarray(known_probs[row_idx], dtype=np.float32)
            row_unknown_prob = float(unknown_probs[row_idx])
            top1_idx = int(np.argmax(row_logits))
            top1_raw_id = int(text_vocab_ids[top1_idx])
            top1_prob = float(row_known_probs[top1_idx])
            if row_logits.size >= 2:
                top2_idx = int(np.argsort(-row_logits, kind='mergesort')[1])
                margin_top1_top2 = float(row_logits[top1_idx] - row_logits[top2_idx])
            else:
                margin_top1_top2 = float(row_logits[top1_idx])
            rows.append({
                'trajectory_id': str(manifest.get('trajectory_id', '')),
                'clip_id': int(manifest.get('clip_id', manifest.get('video_id', 0))),
                'video_id': int(manifest.get('video_id', manifest.get('clip_id', 0))),
                'generator_score': float(manifest.get('generator_score', 1.0) or 1.0),
                'score': float(max(0.0, min(1.0, float(manifest.get('generator_score', 1.0) or 1.0) * top1_prob))),
                'category_id': top1_raw_id,
                'top1_known_raw_id': top1_raw_id,
                'top1_known_name': str(class_name_map.get(top1_raw_id, f'raw_id_{top1_raw_id}')),
                'top1_known_prob': float(top1_prob),
                'unknown_prob': float(row_unknown_prob),
                'margin_top1_top2': float(margin_top1_top2),
                'margin_top1_vs_unknown': float(row_logits[top1_idx]),
                'valid_carrier': bool(manifest.get('valid_carrier', True)),
                'trajectory_record': trajectory_record,
            })
    finally:
        if show_progress and hasattr(iterator, 'close'):
            iterator.close()
    return rows

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
