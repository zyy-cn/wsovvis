from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .g8_bridge import G8Paths, load_json, validate_json_artifact, write_json


ROOT_ENV_VAR = "WSOVVIS_YTVIS2019_ROOT"
DATASETS_ENV_VAR = "DETECTRON2_DATASETS"


@dataclass(frozen=True)
class ExternalYTVIS2019EvalConfig:
    exp_name: str
    output_root: Path
    seed: int
    smoke: bool = False


@dataclass(frozen=True)
class YTVIS2019AnnotationPaths:
    root: Path
    val_json: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _candidate_dataset_roots() -> List[Path]:
    candidates: List[Path] = []
    explicit_root = str(os.environ.get(ROOT_ENV_VAR, "")).strip()
    if explicit_root:
        explicit = Path(explicit_root).expanduser().resolve()
        candidates.extend([explicit, explicit / "ytvis_2019"])
    datasets_root = str(os.environ.get(DATASETS_ENV_VAR, "")).strip()
    if datasets_root:
        candidates.append(Path(datasets_root).expanduser().resolve() / "ytvis_2019")
    else:
        candidates.append((_repo_root() / "datasets" / "ytvis_2019").resolve())
    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def resolve_ytvis2019_annotation_paths() -> YTVIS2019AnnotationPaths:
    checked: List[str] = []
    for root in _candidate_dataset_roots():
        val_json = root / "valid.json"
        checked.append(str(val_json))
        if val_json.is_file():
            return YTVIS2019AnnotationPaths(root=root, val_json=val_json)
    raise FileNotFoundError(f"YTVIS2019 annotation json not found; checked: {checked}")


def _read_pred_main(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_json_artifact(payload, "pred_main.schema.json")
    return [dict(row) for row in payload]


def _subset_annotation_payload(payload: Mapping[str, Any], video_ids: Iterable[int]) -> Dict[str, Any]:
    keep = {int(v) for v in video_ids}
    return {
        **dict(payload),
        "videos": [dict(video) for video in payload.get("videos", []) if int(video.get("id", -1)) in keep],
        "annotations": [dict(ann) for ann in payload.get("annotations", []) if int(ann.get("video_id", -1)) in keep],
    }


def load_ytvis2019_backend_classes():
    try:
        from videocutler.mask2former_video.data_video.datasets.ytvis_api.ytvos import YTVOS
        from videocutler.mask2former_video.data_video.datasets.ytvis_api.ytvoseval import YTVOSeval
    except Exception as exc:  # pragma: no cover - depends on runtime evaluator stack
        raise RuntimeError(
            "YTVIS2019 evaluator backend is unavailable; ensure pycocotools and canonical evaluator deps are installed"
        ) from exc
    return YTVOS, YTVOSeval


def _load_gt_for_mode(paths: YTVIS2019AnnotationPaths, *, smoke: bool, pred_rows: Sequence[Mapping[str, Any]]) -> tuple[Any, Dict[str, Any]]:
    YTVOS, _ = load_ytvis2019_backend_classes()
    val_payload = load_json(paths.val_json)
    if smoke:
        pred_video_ids = [int(row["video_id"]) for row in pred_rows]
        subset = _subset_annotation_payload(val_payload, pred_video_ids)
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            json.dump(subset, handle, ensure_ascii=False)
            handle.flush()
            ann_path = Path(handle.name)
        with contextlib.redirect_stdout(io.StringIO()):
            api = YTVOS(str(ann_path))
        return api, subset
    with contextlib.redirect_stdout(io.StringIO()):
        api = YTVOS(str(paths.val_json))
    return api, val_payload


def _run_ytvis_eval(gt_api: Any, pred_rows: Sequence[Mapping[str, Any]]) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        _YTVOS, YTVOSeval = load_ytvis2019_backend_classes()
        dt_api = gt_api.loadRes(list(pred_rows))
        evaluator = YTVOSeval(gt_api, dt_api, "segm")
        evaluator.params.useCats = 1
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return evaluator


def _per_category_ap(evaluator: Any, *, category_ids: Sequence[int]) -> List[Dict[str, Any]]:
    precisions = evaluator.eval.get("precision")
    if precisions is None:
        return []
    category_ids = [int(x) for x in category_ids]
    per_category: List[Dict[str, Any]] = []
    for category_index, raw_category_id in enumerate(category_ids):
        precision = precisions[:, :, category_index, 0, -1]
        precision = precision[precision > -1]
        ap = float(np.mean(precision) * 100.0) if precision.size else 0.0
        per_category.append({"raw_category_id": int(raw_category_id), "AP": float(ap)})
    return per_category


def _build_external_metrics_payload(*, evaluator: Any, gt_payload: Mapping[str, Any], smoke: bool) -> Dict[str, Any]:
    category_ids = [int(cat["id"]) for cat in gt_payload.get("categories", [])]
    per_category_ap = _per_category_ap(evaluator, category_ids=category_ids)
    return {
        "benchmark_name": "ytvis2019",
        "dataset_name": "ytvis_2019_val",
        "split_tag": "aux_val_smoke" if smoke else "aux_val",
        "AP": float(evaluator.stats[0] * 100.0),
        "AP50": float(evaluator.stats[1] * 100.0),
        "AP75": float(evaluator.stats[2] * 100.0),
        "mAPb": 0.0,
        "mAPn": 0.0,
        "base_raw_ids": [],
        "novel_raw_ids": [],
        "per_category_ap": per_category_ap,
    }


def run_external_ytvis2019_eval(config: ExternalYTVIS2019EvalConfig) -> Dict[str, Any]:
    paths = G8Paths(config.output_root, "ytvis_2019_val")
    pred_main_path = paths.pred_main_path
    if not pred_main_path.is_file():
        raise FileNotFoundError(f"missing canonical pred_main artifact: {pred_main_path}")
    pred_rows = _read_pred_main(pred_main_path)
    ann_paths = resolve_ytvis2019_annotation_paths()
    gt_api, gt_payload = _load_gt_for_mode(ann_paths, smoke=config.smoke, pred_rows=pred_rows)
    evaluator = _run_ytvis_eval(gt_api, pred_rows)
    payload = _build_external_metrics_payload(evaluator=evaluator, gt_payload=gt_payload, smoke=config.smoke)
    validate_json_artifact(payload, "external_metrics.schema.json")
    write_json(paths.external_ytvis2019_metrics_path, payload)
    return {
        "status": "OK",
        "artifact": str(paths.external_ytvis2019_metrics_path),
        "pred_main_path": str(pred_main_path),
        "video_count_evaluated": int(len(gt_payload.get("videos", []))),
        "annotation_root": str(ann_paths.root),
        "metrics": {
            "AP": payload["AP"],
            "AP50": payload["AP50"],
            "AP75": payload["AP75"],
            "AR1": float(evaluator.stats[6] * 100.0) if len(getattr(evaluator, "stats", [])) > 6 else 0.0,
        },
    }
