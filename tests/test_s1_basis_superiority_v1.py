from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from wsovvis.scientific import run_s1_basis_superiority_eval


mask_utils = pytest.importorskip("pycocotools.mask")


def _rle(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(encoded["counts"], (bytes, bytearray)):
        encoded["counts"] = encoded["counts"].decode("ascii")
    return encoded


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _dataset_payload() -> tuple[dict, dict, list[dict]]:
    empty = None
    gt_mask_a = _rle(np.array([[1, 1], [0, 0]], dtype=np.uint8))
    gt_mask_b = _rle(np.array([[0, 1], [0, 0]], dtype=np.uint8))
    raw_mask = _rle(np.array([[0, 1], [0, 0]], dtype=np.uint8))
    refined_mask = _rle(np.array([[1, 1], [0, 0]], dtype=np.uint8))

    gt_json = {
        "videos": [
            {
                "id": 101,
                "name": "vid_101",
                "file_names": ["vid_101/00000.jpg", "vid_101/00001.jpg"],
                "height": 2,
                "width": 2,
                "length": 2,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "video_id": 101,
                "category_id": 1,
                "segmentations": [gt_mask_a, gt_mask_a],
                "areas": [2, 2],
                "bboxes": [[0, 0, 2, 1], [0, 0, 2, 1]],
            },
            {
                "id": 2,
                "video_id": 101,
                "category_id": 1,
                "segmentations": [empty, gt_mask_b],
                "areas": [0, 1],
                "bboxes": [None, [1, 0, 1, 1]],
            },
        ],
    }

    raw_json = {
        "videos": [
            {
                "id": 1,
                "name": "vid_101",
                "file_names": ["vid_101/00000.jpg", "vid_101/00001.jpg"],
                "height": 2,
                "width": 2,
                "length": 2,
            }
        ],
        "annotations": [
            {
                "id": 10,
                "video_id": 1,
                "category_id": 1,
                "segmentations": [raw_mask, raw_mask],
            },
            {
                "id": 11,
                "video_id": 1,
                "category_id": 1,
                "segmentations": [empty, gt_mask_b],
            },
        ],
    }

    refined_json = [
        {
            "track_id": 20,
            "video_id": 101,
            "category_id": 1,
            "score": 0.9,
            "segmentations": [refined_mask, refined_mask],
        },
        {
            "track_id": 21,
            "video_id": 101,
            "category_id": 1,
            "score": 0.7,
            "segmentations": [empty, gt_mask_b],
        },
    ]
    return gt_json, raw_json, refined_json


def test_s1_eval_runs_and_refined_improves_structure_metrics(tmp_path: Path) -> None:
    gt_json, raw_json, refined_json = _dataset_payload()
    gt_path = tmp_path / "gt.json"
    raw_path = tmp_path / "raw.json"
    refined_path = tmp_path / "refined.json"
    out_root = tmp_path / "out"

    _write_json(gt_path, gt_json)
    _write_json(raw_path, raw_json)
    _write_json(refined_path, refined_json)

    manifest = run_s1_basis_superiority_eval(
        gt_json_path=gt_path,
        raw_json_path=raw_path,
        refined_json_path=refined_path,
        output_root=out_root,
        raw_label="raw",
        refined_label="refined",
    )

    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))
    assert manifest["schema_name"] == "wsovvis.s1_basis_superiority_eval"
    assert summary["refined"]["mean_best_iou"] > summary["raw"]["mean_best_iou"]
    assert summary["refined"]["recall_at_0.5"] >= summary["raw"]["recall_at_0.5"]
    assert summary["refined"]["fragmentation_per_gt_instance"] <= summary["raw"]["fragmentation_per_gt_instance"]
    assert (out_root / "paired_worked_example.json").exists()
    assert (out_root / "failure_case_example.json").exists()
    assert (out_root / "paired_worked_example.svg").exists()
    assert (out_root / "failure_case_example.svg").exists()
