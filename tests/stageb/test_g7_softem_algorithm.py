from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from videocutler.ext_stageb_ovvis.algorithms.soft_em import SoftEMConfig, run_soft_em
from videocutler.ext_stageb_ovvis.models.projector import Projector, ProjectorConfig


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _setup_minimal_softem_fixture(tmp_path: Path) -> list[dict]:
    carrier_dir = tmp_path / "carrier_bank" / "lvvis_train_base"
    carrier_dir.mkdir(parents=True, exist_ok=True)
    vec = np.zeros((1, 768), dtype=np.float16)
    vec[0, 0] = 1.0
    np.savez(carrier_dir / "carrier_vectors_traj.npz", z_norm=vec)

    text_dir = tmp_path / "text_bank"
    (text_dir / "payload").mkdir(parents=True, exist_ok=True)
    protos = np.zeros((1, 512), dtype=np.float32)
    protos[0, 0] = 1.0
    np.savez(text_dir / "payload" / "text_prototypes.npz", protos=protos)
    _write_jsonl(
        text_dir / "text_prototype_records.jsonl",
        [{"raw_id": 1, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"}],
    )

    prealign_dir = tmp_path / "train" / "prealign"
    (prealign_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    projector = Projector(ProjectorConfig())
    torch.save(
        {
            "stage_id": "prealign",
            "epoch": 1,
            "projector_state_dict": projector.state_dict(),
            "projector_config": {
                "input_dim": 768,
                "hidden_dim": 512,
                "output_dim": 512,
                "dropout": 0.0,
                "use_layernorm": True,
            },
            "seed": 0,
        },
        prealign_dir / "checkpoints" / "prealign_last.pth",
    )
    _write_jsonl(
        prealign_dir / "proxy_records.jsonl",
        [
            {
                "dataset_name": "lvvis_train_base",
                "clip_id": 7,
                "video_id": 8,
                "trajectory_id": "traj-1",
                "observed_raw_ids": [1],
                "proxy_mass": {"unknown": 0.1, "1": 0.9},
                "join_key": "traj-1",
            }
        ],
    )

    return [
        {
            "trajectory_id": "traj-1",
            "clip_id": "7",
            "trajectory_record": {"video_id": 8},
            "carrier_record": {"z_norm_path": "carrier_vectors_traj.npz#z_norm[0]"},
            "weak_label_record": {"observed_raw_ids": [1]},
            "frame_feature_rows": [],
            "frame_geometry_rows": [],
            "candidate_text_prototypes": [
                {"raw_id": 1, "proto_path": "payload/text_prototypes.npz#protos[0]", "path_base_mode": "artifact_parent_dir"}
            ],
            "observed_raw_ids": [1],
            "candidate_ids_known": [1],
            "candidate_ids_extra": [],
            "missing_views": [],
            "invalid_reasons": [],
            "sample_valid": True,
        }
    ]


def test_softem_base_then_aug_writes_canonical_artifacts(tmp_path: Path) -> None:
    samples = _setup_minimal_softem_fixture(tmp_path)
    result = run_soft_em(
        output_root=tmp_path,
        materialized_samples=samples,
        config=SoftEMConfig(
            dataset_name="lvvis_train_base",
            trajectory_source_branch="mainline",
            mode="base_then_aug",
            device="cpu",
            seed=0,
            smoke=True,
            base_epochs=1,
            aug_epochs=1,
            base_learning_rate=1e-4,
            aug_learning_rate=1e-4,
        ),
    )

    assert result["record_count_input"] == 1
    assert result["record_count_trainable"] == 1
    assert result["record_count_output"] == 1
    assert (tmp_path / "train" / "softem_base" / "train_state.json").is_file()
    assert (tmp_path / "train" / "softem_base" / "responsibility_records.jsonl").is_file()
    assert (tmp_path / "train" / "softem_base" / "checkpoints" / "softem_base_last.pth").is_file()
    assert (tmp_path / "train" / "softem_aug" / "train_state.json").is_file()
    assert (tmp_path / "train" / "softem_aug" / "responsibility_records.jsonl").is_file()
    assert (tmp_path / "train" / "softem_aug" / "checkpoints" / "softem_aug_last.pth").is_file()
