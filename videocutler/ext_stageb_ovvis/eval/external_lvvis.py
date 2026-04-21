from __future__ import annotations

import contextlib
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .g8_bridge import G8Paths, load_json, validate_json_artifact, write_json
from videocutler.ext_stageb_ovvis.data.datasets.lvvis_official_split import load_lvvis_base_and_novel_raw_ids, validate_lvvis_annotation_categories


ROOT_ENV_VAR = "WSOVVIS_LVVIS_ROOT"
ROOT_FALLBACK = "videocutler/datasets/LV-VIS"


@dataclass(frozen=True)
class ExternalLVVISEvalConfig:
    exp_name: str
    output_root: Path
    seed: int
    smoke: bool = False


@dataclass(frozen=True)
class LVVISAnnotationPaths:
    root: Path
    train_json: Path
    val_json: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_lvvis_annotation_paths(*, validate_official_authority: bool = False) -> LVVISAnnotationPaths:
    env_value = str(__import__("os").environ.get(ROOT_ENV_VAR, "")).strip()
    root = Path(env_value).expanduser().resolve() if env_value else (_repo_root() / ROOT_FALLBACK).resolve()
    train_json = root / "annotations" / "train_instances.json"
    val_json = root / "annotations" / "val_instances.json"
    if not train_json.is_file() or not val_json.is_file():
        missing = [str(path) for path in (train_json, val_json) if not path.is_file()]
        raise FileNotFoundError(f"LV-VIS annotation files not found: {missing}")
    if validate_official_authority:
        validate_lvvis_annotation_categories(train_json, val_json)
    return LVVISAnnotationPaths(root=root, train_json=train_json, val_json=val_json)


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


def load_lvvis_backend_classes():
    try:
        from third_party.lvvis_official.evaluate.lvvis import LVVIS
        from third_party.lvvis_official.evaluate.lvviseval import LVVISEval
    except Exception as exc:  # pragma: no cover - depends on runtime evaluator stack
        raise RuntimeError(
            "LV-VIS official evaluator backend is unavailable; ensure pycocotools and canonical evaluator deps are installed"
        ) from exc
    return LVVIS, LVVISEval


def _load_gt_for_mode(paths: LVVISAnnotationPaths, *, smoke: bool, pred_rows: Sequence[Mapping[str, Any]]) -> tuple[Any, Dict[str, Any]]:
    LVVIS, _ = load_lvvis_backend_classes()
    val_payload = load_json(paths.val_json)
    if smoke:
        pred_video_ids = [int(row["video_id"]) for row in pred_rows]
        subset = _subset_annotation_payload(val_payload, pred_video_ids)
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
            json.dump(subset, handle, ensure_ascii=False)
            handle.flush()
            ann_path = Path(handle.name)
        with contextlib.redirect_stdout(io.StringIO()):
            api = LVVIS(str(ann_path))
        return api, subset
    with contextlib.redirect_stdout(io.StringIO()):
        api = LVVIS(str(paths.val_json))
    return api, val_payload


def _run_lvvis_eval(gt_api: Any, pred_rows: Sequence[Mapping[str, Any]]) -> Any:
    with contextlib.redirect_stdout(io.StringIO()):
        _LVVIS, LVVISEval = load_lvvis_backend_classes()
        dt_api = gt_api.loadRes(list(pred_rows))
        evaluator = LVVISEval(gt_api, dt_api, "segm")
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


def _derive_base_and_novel_ids(paths: LVVISAnnotationPaths) -> tuple[List[int], List[int]]:
    _ = paths
    return load_lvvis_base_and_novel_raw_ids()


def _mean_ap(raw_ids: Sequence[int], per_category_ap: Sequence[Mapping[str, Any]]) -> float:
    if not raw_ids:
        return 0.0
    ap_by_id = {int(row["raw_category_id"]): float(row["AP"]) for row in per_category_ap}
    values = [ap_by_id[int(raw_id)] for raw_id in raw_ids if int(raw_id) in ap_by_id]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_external_metrics_payload(
    *,
    evaluator: Any,
    gt_payload: Mapping[str, Any],
    base_raw_ids: Sequence[int],
    novel_raw_ids: Sequence[int],
    smoke: bool,
) -> Dict[str, Any]:
    category_ids = [int(cat["id"]) for cat in gt_payload.get("categories", [])]
    per_category_ap = _per_category_ap(evaluator, category_ids=category_ids)
    return {
        "benchmark_name": "lvvis",
        "dataset_name": "lvvis_val",
        "split_tag": "val_smoke" if smoke else "val",
        "AP": float(evaluator.stats[0] * 100.0),
        "AP50": float(evaluator.stats[1] * 100.0),
        "AP75": float(evaluator.stats[2] * 100.0),
        "mAPb": _mean_ap(base_raw_ids, per_category_ap),
        "mAPn": _mean_ap(novel_raw_ids, per_category_ap),
        "base_raw_ids": [int(x) for x in base_raw_ids],
        "novel_raw_ids": [int(x) for x in novel_raw_ids],
        "per_category_ap": per_category_ap,
    }


def run_external_lvvis_eval(config: ExternalLVVISEvalConfig) -> Dict[str, Any]:
    paths = G8Paths(config.output_root, "lvvis_val")
    pred_main_path = paths.pred_main_path
    if not pred_main_path.is_file():
        raise FileNotFoundError(f"missing canonical pred_main artifact: {pred_main_path}")
    pred_rows = _read_pred_main(pred_main_path)
    ann_paths = resolve_lvvis_annotation_paths(validate_official_authority=not config.smoke)
    gt_api, gt_payload = _load_gt_for_mode(ann_paths, smoke=config.smoke, pred_rows=pred_rows)
    evaluator = _run_lvvis_eval(gt_api, pred_rows)
    base_raw_ids, novel_raw_ids = _derive_base_and_novel_ids(ann_paths)
    payload = _build_external_metrics_payload(
        evaluator=evaluator,
        gt_payload=gt_payload,
        base_raw_ids=base_raw_ids,
        novel_raw_ids=novel_raw_ids,
        smoke=config.smoke,
    )
    validate_json_artifact(payload, "external_metrics.schema.json")
    write_json(paths.external_lvvis_metrics_path, payload)
    return {
        "status": "OK",
        "artifact": str(paths.external_lvvis_metrics_path),
        "pred_main_path": str(pred_main_path),
        "video_count_evaluated": int(len(gt_payload.get("videos", []))),
        "annotation_root": str(ann_paths.root),
        "metrics": {key: payload[key] for key in ("AP", "AP50", "AP75", "mAPb", "mAPn")},
    }
