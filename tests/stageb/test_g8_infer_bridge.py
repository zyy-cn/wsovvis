from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig
from videocutler.run_stageb_infer_ov import main as run_infer_main


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_lvvis_annotations(root: Path) -> None:
    categories = [{"id": 1, "name": "cls_one"}, {"id": 3, "name": "cls_three"}]
    payload = {
        "videos": [{"id": 101, "length": 2, "height": 28, "width": 28, "file_names": ["000.jpg", "001.jpg"]}],
        "categories": categories,
        "annotations": [],
    }
    _write_json(root / "annotations" / "train_instances.json", payload)
    _write_json(root / "annotations" / "val_instances.json", payload)


def _prepare_infer_fixture(root: Path) -> None:
    carrier_dir = root / "carrier_bank" / "lvvis_val"
    frame_dir = root / "frame_bank" / "lvvis_val"
    text_dir = root / "text_bank"
    for path in (carrier_dir, frame_dir / "payload", text_dir / "payload"):
        path.mkdir(parents=True, exist_ok=True)

    traj = np.zeros((1, 768), dtype=np.float16)
    traj[0, 0] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj.npz", z_norm=traj)
    _write_jsonl(
        carrier_dir / "carrier_records.jsonl",
        [
            {
                "trajectory_id": "traj-101-0",
                "clip_id": "101",
                "z_norm_path": "carrier_vectors_traj.npz#z_norm[0]",
                "frame_indices": [0],
                "frame_carriers_norm_paths": [],
                "path_base_mode": "artifact_parent_dir",
            }
        ],
    )

    pooled = np.zeros((1, 768), dtype=np.float16)
    pooled[0, 1] = 1.0
    np.savez(frame_dir / "payload" / "clip_101_pooled.npz", frame_pooled=pooled)
    _write_jsonl(
        frame_dir / "pooled_frame_records.jsonl",
        [
            {
                "trajectory_id": "traj-101-0",
                "clip_id": "101",
                "trajectory_source_branch": "mainline",
                "frame_count": 1,
                "frame_pooled_path": "payload/clip_101_pooled.npz#frame_pooled[0]",
                "path_base_mode": "artifact_parent_dir",
            }
        ],
    )

    protos = np.zeros((2, 512), dtype=np.float32)
    protos[0, 0] = 1.0
    protos[1, 1] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [
            {"raw_id": 1, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"},
            {"raw_id": 3, "proto_path": "payload/text_prototypes.npz#protos[1]", "path_base_mode": "artifact_parent_dir"},
        ],
    )

    _write_jsonl(
        root / "exports" / "lvvis_val" / "trajectory_records.jsonl",
        [
            {
                "dataset_name": "lvvis_val",
                "split_tag": "val",
                "clip_id": 101,
                "video_id": 101,
                "rank_in_clip": 0,
                "trajectory_id": "traj-101-0",
                "generator_tag": "videocutler_r50_native",
                "pred_score": 0.9,
                "frame_indices": [0],
                "masks_rle": [{"size": [28, 28], "counts": "abc"}],
                "boxes_xyxy": [[0, 0, 10, 10]],
                "valid_carrier": True,
                "invalid_reason": None,
                "image_size": [28, 28],
            }
        ],
    )

    ckpt_dir = root / "train" / "softem_aug" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    projector = Projector(ProjectorConfig())
    torch.save(
        {
            "stage_id": "softem_aug",
            "epoch": 1,
            "text_projector_state_dict": projector.state_dict(),
            "text_projector_config": {
                "input_dim": 512,
                "hidden_dim": 1024,
                "output_dim": 768,
                "dropout": 0.0,
                "use_layernorm": True,
            },
            "theta_T": 0.07,
            "b_u": 0.0,
        },
        ckpt_dir / "softem_aug_last.pth",
    )
    _write_json(
        root / "train" / "softem_aug" / "train_state.json",
        {
            "stage_id": "softem_aug",
            "epoch": 1,
            "selected_for_infer": "augmented",
            "selected_for_infer_authority": "explicit_train_state_field",
            "checkpoint_selected": "train/softem_aug/checkpoints/softem_aug_last.pth",
            "runtime_asset_source": "local_canonical_assets",
            "runtime_asset_source_local_incomplete": False,
            "runtime_asset_output_root": str(root),
        },
    )


def test_run_stageb_infer_ov_emits_canonical_prediction_artifacts(tmp_path: Path, monkeypatch) -> None:
    lvvis_root = tmp_path / "lvvis_root"
    _prepare_lvvis_annotations(lvvis_root)
    _prepare_infer_fixture(tmp_path)
    monkeypatch.setenv("WSOVVIS_LVVIS_ROOT", str(lvvis_root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stageb_infer_ov.py",
            "--exp_name",
            "toy_g8_infer",
            "--dataset_name",
            "lvvis_val",
            "--output_root",
            str(tmp_path),
            "--device",
            "cpu",
            "--seed",
            "0",
            "--logit_chunk_size",
            "16",
        ],
    )
    assert run_infer_main() == 0

    pred_main = json.loads((tmp_path / "predictions" / "lvvis_val" / "pred_main.json").read_text(encoding="utf-8"))
    pred_diag = json.loads((tmp_path / "predictions" / "lvvis_val" / "pred_diag.json").read_text(encoding="utf-8"))

    assert len(pred_main) == 1
    assert len(pred_diag) == 1
    assert pred_main[0]["trajectory_id"] == "traj-101-0"
    assert pred_main[0]["video_id"] == 101
    assert len(pred_main[0]["segmentations"]) == 2
    assert pred_main[0]["segmentations"][1] is None
    assert pred_diag[0]["pred_main_index"] == 0
    assert pred_diag[0]["top1_known_raw_id"] in {1, 3}
    assert pred_diag[0]["top1_known_name"] in {"cls_one", "cls_three"}
    assert 0.0 <= float(pred_diag[0]["top1_known_prob"]) <= 1.0
    assert 0.0 <= float(pred_diag[0]["unknown_prob"]) <= 1.0
