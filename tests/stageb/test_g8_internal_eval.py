from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from videocutler.ext_stageb_ovvis.eval.internal_val import InternalEvalConfig, run_internal_eval


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_g8_internal_eval_smoke(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "LV-VIS"
    ann_root = dataset_root / "annotations"
    ann_root.mkdir(parents=True, exist_ok=True)

    payload = {
        "videos": [{"id": 101, "width": 2, "height": 2, "length": 2, "file_names": ["a.jpg", "b.jpg"]}],
        "categories": [
            {"id": 1, "name": "base_cat"},
            {"id": 2, "name": "other_cat"},
        ],
        "annotations": [
            {
                "id": 1,
                "video_id": 101,
                "category_id": 1,
                "segmentations": [{"size": [2, 2], "counts": "mask_a"}, None],
                "areas": [2.0, None],
                "bboxes": [[0.0, 0.0, 2.0, 1.0], None],
                "avg_area": 2.0,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "video_id": 101,
                "category_id": 2,
                "segmentations": [None, {"size": [2, 2], "counts": "mask_b"}],
                "areas": [None, 2.0],
                "bboxes": [None, [0.0, 0.0, 2.0, 1.0]],
                "avg_area": 2.0,
                "iscrowd": 0,
            },
        ],
    }
    _write_json(ann_root / "train_instances.json", payload)
    _write_json(ann_root / "val_instances.json", payload)

    output_root = tmp_path / "formal_chain"
    pred_main = [
        {
            "trajectory_id": "traj-good-1",
            "video_id": 101,
            "score": 0.95,
            "category_id": 1,
            "segmentations": [{"size": [2, 2], "counts": "mask_a"}, None],
        },
        {
            "trajectory_id": "traj-good-dup",
            "video_id": 101,
            "score": 0.90,
            "category_id": 1,
            "segmentations": [{"size": [2, 2], "counts": "mask_a"}, None],
        },
        {
            "trajectory_id": "traj-unmatched",
            "video_id": 101,
            "score": 0.20,
            "category_id": 2,
            "segmentations": [{"size": [2, 2], "counts": "mask_c"}, None],
        },
    ]
    pred_diag = [
        {
            "pred_main_index": 0,
            "trajectory_id": "traj-good-1",
            "clip_id": 101,
            "video_id": 101,
            "generator_score": 0.95,
            "top1_known_raw_id": 1,
            "top1_known_name": "base_cat",
            "top1_known_prob": 0.9,
            "unknown_prob": 0.1,
            "margin_top1_top2": 0.4,
            "margin_top1_vs_unknown": 0.5,
            "valid_carrier": True,
        },
        {
            "pred_main_index": 1,
            "trajectory_id": "traj-good-dup",
            "clip_id": 101,
            "video_id": 101,
            "generator_score": 0.90,
            "top1_known_raw_id": 1,
            "top1_known_name": "base_cat",
            "top1_known_prob": 0.85,
            "unknown_prob": 0.15,
            "margin_top1_top2": 0.3,
            "margin_top1_vs_unknown": 0.4,
            "valid_carrier": True,
        },
        {
            "pred_main_index": 2,
            "trajectory_id": "traj-unmatched",
            "clip_id": 101,
            "video_id": 101,
            "generator_score": 0.20,
            "top1_known_raw_id": 2,
            "top1_known_name": "other_cat",
            "top1_known_prob": 0.4,
            "unknown_prob": 0.6,
            "margin_top1_top2": 0.1,
            "margin_top1_vs_unknown": -0.2,
            "valid_carrier": True,
        },
    ]
    _write_json(output_root / "predictions" / "lvvis_val" / "pred_main.json", pred_main)
    _write_json(output_root / "predictions" / "lvvis_val" / "pred_diag.json", pred_diag)

    monkeypatch.setenv("WSOVVIS_LVVIS_ROOT", str(dataset_root))
    result = run_internal_eval(
        InternalEvalConfig(
            exp_name="toy_internal_eval",
            dataset_name="lvvis_val",
            output_root=output_root,
            seed=0,
            smoke=True,
        )
    )

    assert result["status"] == "OK"
    metrics_path = output_root / "eval" / "internal" / "internal_metrics.lvvis.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "lvvis"
    assert payload["dataset_name"] == "lvvis_val"
    assert payload["split_tag"] == "val_smoke"
    assert payload["matched_predictions"] == 2
    assert payload["correct_predictions"] == 2
    assert payload["top1_known_acc"] == 1.0
    assert payload["mean_best_iou"] == 1.0
    assert payload["metric_valid"] is True
    assert payload["metric_status"] == "ok"

    companion = json.loads((output_root / "eval" / "internal" / "internal_metrics_companion.lvvis.json").read_text(encoding="utf-8"))
    assert companion["fragmentation_per_gt_instance"] == 0.5
    assert companion["split_rate"] == 0.5
    assert companion["merge_rate"] == 0.0
    assert companion["duplicate_rate"] == 0.5
    assert abs(companion["unmatched_trajectory_ratio"] - (1.0 / 3.0)) < 1e-9
