from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from videocutler.ext_stageb_ovvis.banks.trajectory_bank import (
    GENERATOR_CFG_PATH,
    GENERATOR_TAG,
    materialize_trajectory_bank,
    validate_trajectory_record,
)

DATASET_CHOICES = ("lvvis_train_base", "lvvis_val", "ytvis_2019_val")
SPEC_VERSION = "v20.1.3"
CONTRACT_VERSION = "v4.8.9-cf3"
TEST_SPEC_VERSION = "v1.4.0-cf3"
ASSET_VERSION = "v1.2.6-cf2"
SMOKE_FIXTURE_RELS = {
    "lvvis_train_base": "fixtures/tiny_lvvis_pipeline_case/trajectory_records_train_min.jsonl",
    "lvvis_val": "fixtures/tiny_lvvis_pipeline_case/trajectory_records_val_min.jsonl",
}
RAW_RESULTS_DEFAULTS = {
    "lvvis_train_base": "OUTPUT/lvvis_train_base_videocutler_r50/inference/results.json",
    "lvvis_val": "OUTPUT/lvvis_val_videocutler_r50/inference/results.json",
}
SUMMARY_FIELDS = {
    "lvvis_train_base": {
        "input_source_type": "official_lvvis_train_annotations",
        "data_scope_smoke": "train_smoke",
        "data_scope_full": "train",
        "consumer_target": "trajectory_sample_view_readable|run_stageb_build_carrier_bank|run_stageb_train_prealign|run_stageb_train_softem",
    },
    "lvvis_val": {
        "input_source_type": "official_lvvis_val_annotations",
        "data_scope_smoke": "val_smoke",
        "data_scope_full": "val",
        "consumer_target": "run_stageb_extract_dinov2_frames|run_stageb_build_carrier_bank|run_stageb_infer_ov",
    },
}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _merge_unique(values: list[str]) -> str:
    seen: list[str] = []
    for value in values:
        value = str(value).strip()
        if value and value not in seen:
            seen.append(value)
    return "|".join(seen)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_run_root(output_root: str, exp_name: str) -> Path:
    root = Path(output_root).expanduser().resolve()
    if root.name == exp_name:
        return root
    return root / exp_name


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_lvvis_root() -> Path:
    env_value = os.environ.get("WSOVVIS_LVVIS_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (_repo_root() / "videocutler" / "datasets" / "LV-VIS").resolve()


def _annotation_json_path(dataset_name: str) -> Path:
    if dataset_name == "lvvis_train_base":
        ann_rel = "annotations/train_instances.json"
    elif dataset_name == "lvvis_val":
        ann_rel = "annotations/val_instances.json"
    else:
        raise ValueError(f"dataset does not support full raw export: {dataset_name}")
    return _resolve_lvvis_root() / ann_rel


def _load_video_metadata(dataset_name: str) -> Dict[int, Dict[str, int]]:
    ann_path = _annotation_json_path(dataset_name)
    if not ann_path.exists():
        raise FileNotFoundError(f"LV-VIS annotation json not found: {ann_path}")
    payload = json.loads(ann_path.read_text(encoding="utf-8"))
    videos = payload.get("videos", [])
    meta: Dict[int, Dict[str, int]] = {}
    for video in videos:
        video_id = int(video["id"])
        file_names = video.get("file_names") or video.get("filenames") or []
        length = int(video.get("length", len(file_names)))
        width = int(video.get("width", 0) or 0)
        height = int(video.get("height", 0) or 0)
        meta[video_id] = {
            "video_id": video_id,
            "clip_id": video_id,
            "length": length,
            "width": width,
            "height": height,
        }
    return meta


def _load_raw_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"raw results json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"raw results json must be a list: {path}")
    records: List[Dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"raw result {idx} is not an object")
        records.append(item)
    return records


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _encode_for_sort(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _bbox_from_rle(rle: Dict[str, Any]) -> List[float] | None:
    try:
        from pycocotools import mask as mask_utils

        bbox_xywh = mask_utils.toBbox(rle)
        if bbox_xywh is None or len(bbox_xywh) != 4:
            return None
        x, y, w, h = [float(v) for v in bbox_xywh]
        if w <= 0 or h <= 0:
            return None
        return [x, y, x + w, y + h]
    except Exception:
        return None


def _iter_valid_segmentations(segmentations: Iterable[Any]) -> Iterable[Tuple[int, Dict[str, Any]]]:
    for frame_index, seg in enumerate(segmentations):
        if seg is None:
            continue
        if not isinstance(seg, dict):
            continue
        if "counts" not in seg or "size" not in seg:
            continue
        size = seg.get("size")
        if not isinstance(size, list) or len(size) != 2:
            continue
        yield frame_index, {"counts": seg["counts"], "size": [int(size[0]), int(size[1])]}


def _split_tag(dataset_name: str, smoke: bool) -> str:
    if dataset_name == "lvvis_train_base":
        return "train_smoke" if smoke else "train"
    if dataset_name == "lvvis_val":
        return "val_smoke" if smoke else "val"
    return "aux_val_smoke" if smoke else "aux_val"


def _default_raw_results_path(dataset_name: str) -> Path:
    rel = RAW_RESULTS_DEFAULTS.get(dataset_name)
    if not rel:
        raise ValueError(f"dataset does not support full raw export: {dataset_name}")
    return _repo_root() / rel


def _load_smoke_fixture(dataset_name: str) -> str:
    fixture_rel = SMOKE_FIXTURE_RELS.get(dataset_name)
    if not fixture_rel:
        raise ValueError(f"unsupported dataset for smoke export: {dataset_name}")
    zip_path = _repo_root() / "package" / "assets" / "WSOVVIS_stageb_test_assets_v1.2.6.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"smoke fixture archive missing: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        try:
            return zf.read(fixture_rel).decode("utf-8")
        except KeyError as exc:
            raise FileNotFoundError(f"smoke fixture missing in archive: {fixture_rel}") from exc


def _records_from_smoke_fixture(dataset_name: str) -> List[Dict[str, Any]]:
    fixture_text = _load_smoke_fixture(dataset_name)
    return [json.loads(line) for line in fixture_text.splitlines() if line.strip()]


def _records_from_raw_results(args: argparse.Namespace) -> List[Dict[str, Any]]:
    raw_path = Path(args.raw_results_json).expanduser().resolve() if args.raw_results_json else _default_raw_results_path(args.dataset_name)
    raw_results = _load_raw_results(raw_path)
    video_meta = _load_video_metadata(args.dataset_name)

    grouped: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for raw_idx, raw in enumerate(raw_results):
        video_id = _safe_int(raw.get("video_id"), -1)
        if video_id < 0:
            continue
        grouped.setdefault(video_id, []).append((raw_idx, raw))

    split_tag = _split_tag(args.dataset_name, smoke=False)
    records: List[Dict[str, Any]] = []
    for video_id in sorted(grouped.keys()):
        meta = video_meta.get(video_id, {"video_id": video_id, "clip_id": video_id, "width": 0, "height": 0})
        per_video: List[Dict[str, Any]] = []
        for raw_idx, raw in grouped[video_id]:
            frame_indices: List[int] = []
            masks_rle: List[Dict[str, Any]] = []
            boxes_xyxy: List[List[float] | None] = []
            image_size: List[int] | None = None
            segmentations = raw.get("segmentations", [])
            if not isinstance(segmentations, list):
                segmentations = []
            for frame_index, seg in _iter_valid_segmentations(segmentations):
                frame_indices.append(int(frame_index))
                masks_rle.append(seg)
                boxes_xyxy.append(_bbox_from_rle(seg))
                if image_size is None:
                    image_size = [int(seg["size"][0]), int(seg["size"][1])]

            if image_size is None:
                fallback_h = _safe_int(meta.get("height"), 0)
                fallback_w = _safe_int(meta.get("width"), 0)
                if fallback_h > 0 and fallback_w > 0:
                    image_size = [fallback_h, fallback_w]
                elif masks_rle:
                    first_size = masks_rle[0]["size"]
                    image_size = [int(first_size[0]), int(first_size[1])]
                else:
                    image_size = [1, 1]

            valid_carrier = len(frame_indices) > 0
            per_video.append(
                {
                    "dataset_name": args.dataset_name,
                    "split_tag": split_tag,
                    "clip_id": _safe_int(meta.get("clip_id"), video_id),
                    "video_id": _safe_int(meta.get("video_id"), video_id),
                    "pred_score": _safe_float(raw.get("score"), 0.0),
                    "frame_indices": frame_indices,
                    "masks_rle": masks_rle,
                    "boxes_xyxy": boxes_xyxy,
                    "valid_carrier": valid_carrier,
                    "invalid_reason": None if valid_carrier else "no_valid_mask_frames",
                    "generator_tag": GENERATOR_TAG,
                    "generator_cfg_path": GENERATOR_CFG_PATH,
                    "generator_ckpt_path": str(Path(args.generator_ckpt).expanduser()),
                    "image_size": image_size,
                    "pred_label_raw": _safe_int(raw.get("category_id"), 1) - 1,
                    "_raw_index": int(raw_idx),
                }
            )

        per_video = sorted(
            per_video,
            key=lambda item: (
                -float(item["pred_score"]),
                list(item["frame_indices"]),
                _encode_for_sort(item["masks_rle"]),
                int(item["_raw_index"]),
            ),
        )
        for rank_in_clip, item in enumerate(per_video):
            item["rank_in_clip"] = int(rank_in_clip)
            item.pop("_raw_index", None)
            records.append(item)

    materialized = materialize_trajectory_bank(records)
    materialized = sorted(materialized, key=lambda item: str(item["trajectory_id"]))
    return materialized


def _ensure_image_size(records: List[Dict[str, Any]]) -> None:
    for idx, record in enumerate(records):
        image_size = record.get("image_size")
        if not isinstance(image_size, list) or len(image_size) != 2:
            inferred = None
            masks = list(record.get("masks_rle", []))
            for item in masks:
                if isinstance(item, dict) and isinstance(item.get("size"), list) and len(item["size"]) == 2:
                    inferred = [int(item["size"][0]), int(item["size"][1])]
                    break
            if inferred is None:
                raise ValueError(f"trajectory record {idx} missing required image_size and cannot infer it")
            record["image_size"] = inferred


def _write_exports(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.smoke:
        records = _records_from_smoke_fixture(args.dataset_name)
    else:
        records = _records_from_raw_results(args)

    _ensure_image_size(records)
    export_rel = Path("exports") / args.dataset_name / "trajectory_records.jsonl"
    export_path = _repo_root() / export_rel
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    return records


def _lvvis_root_binding() -> Dict[str, str]:
    env_value = os.environ.get("WSOVVIS_LVVIS_ROOT")
    if env_value:
        return {
            "lvvis_root_binding_mode": "env",
            "lvvis_root_resolved": str(Path(env_value).expanduser().resolve()),
        }
    return {
        "lvvis_root_binding_mode": "repo_fallback",
        "lvvis_root_resolved": str((Path(__file__).resolve().parent / "datasets" / "LV-VIS").resolve()),
    }


def build_resolved_config(args: argparse.Namespace, run_root: Path) -> Dict[str, Any]:
    return {
        "runtime": {
            "seed": args.seed,
            "device": args.device,
            "smoke": bool(args.smoke),
        },
        "data": {
            "dataset_name": args.dataset_name,
            "train_dataset": "lvvis_train_base",
            "main_val_dataset": "lvvis_val",
            "aux_val_dataset": "ytvis_2019_val",
        },
        "observation_protocol_id": args.protocol_id,
        "generator_ckpt": args.generator_ckpt,
        "dinov2_ckpt": args.dinov2_ckpt,
        "clip_backend": "openai_clip",
        "clip_ckpt": args.clip_ckpt,
        "output_root": str(run_root),
        "benchmark": {
            "main": "lvvis",
            "aux_validation": "ytvis2019",
        },
        "bindings": {
            "dinov2_source_binding": "third_party/dinov2",
            "clip_source_binding": "third_party/openai_clip",
            "lvvis_evaluator_binding": "third_party/lvvis_official/evaluate",
            "ytvis_evaluator_binding": "videocutler/mask2former_video/data_video/ytvis_eval.py",
            "lvvis_registration_binding": "videocutler/ext_stageb_ovvis/data/datasets/lvvis.py",
            **_lvvis_root_binding(),
        },
        "layout": {
            "output_root_layout_version": "v1",
            "prediction_dirs": {
                "main": "predictions/lvvis_val",
                "aux": "predictions/ytvis_2019_val",
            },
            "eval_dirs": {
                "internal": "eval/internal",
                "main": "eval/lvvis",
                "aux": "eval/ytvis2019",
            },
            "default_selected_for_infer": "softem_aug_last.pth",
        },
    }


def build_run_meta(args: argparse.Namespace, run_root: Path) -> Dict[str, Any]:
    summary = SUMMARY_FIELDS[args.dataset_name]
    run_scope = "smoke" if args.smoke else "full"
    meta: Dict[str, Any] = {
        "run_id": f"{args.exp_name}_{args.dataset_name}_seed{args.seed}",
        "exp_name": args.exp_name,
        "dataset_name": args.dataset_name,
        "run_scope": run_scope,
        "input_source_type": summary["input_source_type"],
        "data_scope": summary["data_scope_smoke"] if args.smoke else summary["data_scope_full"],
        "consumer_target": summary["consumer_target"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec_version": SPEC_VERSION,
        "contract_version": CONTRACT_VERSION,
        "test_spec_version": TEST_SPEC_VERSION,
        "asset_version": ASSET_VERSION,
        "main_prediction_path": str(run_root / "predictions" / "lvvis_val" / "pred_main.json"),
        "aux_prediction_path": str(run_root / "predictions" / "ytvis_2019_val" / "pred_main.json"),
        "main_eval_path": str(run_root / "eval" / "lvvis" / "external_metrics.lvvis.json"),
        "aux_eval_path": str(run_root / "eval" / "ytvis2019" / "external_metrics.ytvis2019.json"),
    }
    if args.ckpt_path:
        meta["override_ckpt_path"] = args.ckpt_path
    return meta


def _merge_contract_check(
    contract_path: Path,
    *,
    payload_path: Path,
    records: list[dict],
    dataset_name: str,
    protocol_id: str,
    split_tag: str,
    run_scope: str,
) -> Dict[str, Any]:
    existing = _load_json_if_exists(contract_path)
    dataset_entries: Dict[str, Dict[str, Any]] = {}
    for item in existing.get("datasets", []) if isinstance(existing.get("datasets", []), list) else []:
        if isinstance(item, dict) and str(item.get("dataset_name", "")).strip():
            dataset_entries[str(item["dataset_name"]).strip()] = item

    summary = SUMMARY_FIELDS[dataset_name]
    payload_exists = payload_path.exists()
    dataset_summary = {
        "dataset_name": dataset_name,
        "split_tag": split_tag,
        "observation_protocol_id": protocol_id,
        "run_scope": run_scope,
        "input_source_type": summary["input_source_type"],
        "data_scope": summary["data_scope_full"] if run_scope == "full" else summary["data_scope_smoke"],
        "record_count": len(records),
        "coverage_ratio": 1.0 if records else 0.0,
        "consumer_target": summary["consumer_target"],
        "consumer_ready": bool(payload_exists and records and run_scope == "full"),
        "payload_output": payload_path.as_posix(),
        "payload_exists": payload_exists,
        "payload_record_count": len(records),
        "payload_sha256": _sha256_file(payload_path) if payload_exists else "",
    }
    dataset_entries[dataset_name] = dataset_summary
    merged_datasets = [dataset_entries[key] for key in sorted(dataset_entries)]
    aggregate_input_source_type = _merge_unique([str(item.get("input_source_type", "")) for item in merged_datasets])
    aggregate_data_scope = _merge_unique([str(item.get("data_scope", "")) for item in merged_datasets])
    aggregate_consumer_target = _merge_unique([str(item.get("consumer_target", "")) for item in merged_datasets])
    aggregate_run_scope = _merge_unique([str(item.get("run_scope", "")) for item in merged_datasets]) or run_scope
    aggregate_record_count = sum(int(item.get("record_count", 0) or 0) for item in merged_datasets)
    aggregate_coverage = min([float(item.get("coverage_ratio", 0.0) or 0.0) for item in merged_datasets]) if merged_datasets else 0.0
    aggregate_consumer_ready = all(bool(item.get("consumer_ready")) for item in merged_datasets)
    contract = {
        "gate_id": "G2_exports",
        "status": "PASS" if aggregate_consumer_ready and aggregate_record_count > 0 else "FAIL",
        "run_scope": aggregate_run_scope,
        "input_source_type": aggregate_input_source_type,
        "data_scope": aggregate_data_scope,
        "record_count": aggregate_record_count,
        "coverage_ratio": aggregate_coverage,
        "consumer_target": aggregate_consumer_target,
        "consumer_ready": aggregate_consumer_ready,
        "dataset_requirements": ["lvvis_train_base", "lvvis_val"],
        "datasets": merged_datasets,
        "deliverables": {
            "trajectory_bank": "videocutler/ext_stageb_ovvis/banks/trajectory_bank.py",
            "trajectory_dataset": "videocutler/ext_stageb_ovvis/data/trajectory_dataset.py",
        },
        "evidence_refs": {
            "implemented_tests": [
                "tests/stageb/unit/data/test_trajectory_dataset.py",
                "tests/stageb/unit/banks/test_trajectory_bank.py",
                "tests/stageb/integration/g2_exports/test_run_stageb_export_train.py",
                "tests/stageb/integration/g2_exports/test_run_stageb_export_val.py",
            ],
            "smoke_command": "python videocutler/run_stageb_export.py --help",
        },
        "primary_artifacts": [
            "exports/lvvis_train_base/trajectory_records.jsonl",
            "exports/lvvis_val/trajectory_records.jsonl",
        ],
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage B LV-VIS export orchestration entrypoint.")
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--dataset_name", required=True, choices=DATASET_CHOICES)
    parser.add_argument("--generator_ckpt", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--protocol_id",
        default="keep60_seed42",
        choices=("keep80_seed42", "keep60_seed42", "keep40_seed42"),
    )
    parser.add_argument("--dinov2_ckpt", default="vitb14_reg")
    parser.add_argument("--clip_ckpt", default="openai_clip_vit_b16")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--contract_check_json")
    parser.add_argument(
        "--raw_results_json",
        help=(
            "Path to official VideoCutLER raw inference results.json for full export. "
            "If omitted, defaults to OUTPUT/<dataset>_videocutler_r50/inference/results.json."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    run_root = _resolved_run_root(args.output_root, args.exp_name)
    manifest_root = run_root / "manifests"
    _write_json(manifest_root / "resolved_config.json", build_resolved_config(args, run_root))
    _write_json(manifest_root / "run_meta.json", build_run_meta(args, run_root))
    records = _write_exports(args)
    validation_errors: List[str] = []
    for index, record in enumerate(records):
        errs = validate_trajectory_record(record)
        if errs:
            validation_errors.append(f"record {index}: {', '.join(errs)}")
            if len(validation_errors) >= 5:
                break
    if validation_errors:
        raise ValueError("exported trajectory record validation failed: " + "; ".join(validation_errors))
    if args.contract_check_json:
        split_tag = _split_tag(args.dataset_name, args.smoke)
        payload_path = _repo_root() / Path("exports") / args.dataset_name / "trajectory_records.jsonl"
        _merge_contract_check(
            Path(args.contract_check_json),
            payload_path=payload_path,
            records=records,
            dataset_name=args.dataset_name,
            protocol_id=args.protocol_id,
            split_tag=split_tag,
            run_scope="smoke" if args.smoke else "full",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
