import json
from pathlib import Path

import pytest

from wsovvis.track_feature_export import (
    ExportContractError,
    export_feature_enablement_from_real_run,
)


torch = pytest.importorskip("torch")


def _write_run_layout(tmp_path: Path) -> Path:
    run_root = tmp_path / "runs" / "sample_run"
    (run_root / "d2" / "inference").mkdir(parents=True, exist_ok=True)
    (run_root / "d2" / "model_final.pth").write_bytes(b"checkpoint")
    (run_root / "d2" / "last_checkpoint").write_text("model_final.pth\n", encoding="utf-8")
    (run_root / "config.json").write_text(json.dumps({"run": "sample"}), encoding="utf-8")
    return run_root


def _write_predictions(path: Path, predictions):
    torch.save(predictions, path)


def test_real_run_predictions_to_enablement_artifact(tmp_path: Path):
    run_root = _write_run_layout(tmp_path)
    pseudo_path = tmp_path / "pseudo.json"
    pseudo_path.write_text(json.dumps({"videos": []}), encoding="utf-8")
    _write_predictions(
        run_root / "d2" / "inference" / "instances_predictions.pth",
        [
            {
                "video_id": 101,
                "track_id": 7,
                "score": 0.88,
                "embedding": [0.1, 0.2, 0.3, 0.4],
                "embedding_normalization": "none",
                "start_frame_idx": 0,
                "end_frame_idx": 2,
                "num_active_frames": 3,
            }
        ],
    )

    output_root = export_feature_enablement_from_real_run(
        run_root=run_root,
        repo_root=tmp_path,
        split="val",
        pseudo_tube_manifest_path="pseudo.json",
        d2_cfg_ref="configs/x.yaml",
        d2_opts=["A", "B"],
    )

    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    shard = json.loads((output_root / "videos" / "101.json").read_text(encoding="utf-8"))
    assert manifest["embedding_dim"] == 4
    assert shard["tracks"][0]["track_id"] == 7
    assert shard["tracks"][0]["embedding"] == [0.1, 0.2, 0.3, 0.4]


def test_real_run_missing_embedding_hard_fails(tmp_path: Path):
    run_root = _write_run_layout(tmp_path)
    pseudo_path = tmp_path / "pseudo.json"
    pseudo_path.write_text(json.dumps({"videos": []}), encoding="utf-8")
    _write_predictions(
        run_root / "d2" / "inference" / "instances_predictions.pth",
        [
            {
                "video_id": 101,
                "track_id": 7,
                "score": 0.88,
            }
        ],
    )

    with pytest.raises(ExportContractError, match="missing per-track embedding"):
        export_feature_enablement_from_real_run(
            run_root=run_root,
            repo_root=tmp_path,
            split="val",
            pseudo_tube_manifest_path="pseudo.json",
            d2_cfg_ref="configs/x.yaml",
            d2_opts=[],
        )
