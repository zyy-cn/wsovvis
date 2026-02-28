import json
from copy import deepcopy
from pathlib import Path

import pytest

from wsovvis.track_feature_export import (
    ExportContractError,
    build_feature_export_enablement_v1,
)


def _base_payload() -> dict:
    return {
        "run_id": "run15",
        "split": "val",
        "embedding_dim": 4,
        "embedding_normalization": "none",
        "stageb_checkpoint_ref": "d2/model_final.pth",
        "stageb_checkpoint_hash": "sha256:abc",
        "stageb_config_ref": "config.json",
        "stageb_config_hash": "sha256:def",
        "pseudo_tube_manifest_ref": "data/pseudo_tube_ytvis.json",
        "pseudo_tube_manifest_hash": "sha256:ghi",
        "extraction_settings": {
            "frame_sampling_rule": "uniform_stride_2",
            "pooling_rule": "track_feature_vector_direct",
            "min_track_length": 1,
        },
        "videos": [
            {
                "video_id": 2,
                "runtime_evidence": {
                    "stageb_completion_marker": "completed",
                    "evidence_source": "run.json.status",
                    "evidence_confidence": "explicit",
                },
                "tracks": [
                    {
                        "track_id": 9,
                        "embedding": [0.1, 0.2, 0.3, 0.4],
                        "embedding_normalization": "none",
                        "start_frame_idx": 10,
                        "end_frame_idx": 20,
                    },
                    {
                        "track_id": 4,
                        "embedding": [0.0, 0.0, 0.0, 0.0],
                        "embedding_normalization": "none",
                        "start_frame_idx": 0,
                        "end_frame_idx": 8,
                    },
                ],
            },
            {
                "video_id": 1,
                "runtime_evidence": {
                    "stageb_completion_marker": "failed",
                    "evidence_source": "d2/log.txt",
                    "failure_reason": "video decode failure",
                },
                "tracks": [],
            },
        ],
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_happy_path_emits_contract_topology_and_required_fields(tmp_path: Path):
    payload = _base_payload()

    output_root = build_feature_export_enablement_v1(
        input_payload=payload,
        run_root=tmp_path,
        emit_video_index=True,
    )

    manifest_path = output_root / "manifest.json"
    shard_1 = output_root / "videos" / "1.json"
    shard_2 = output_root / "videos" / "2.json"
    index_path = output_root / "video_index.json"
    assert manifest_path.exists()
    assert shard_1.exists()
    assert shard_2.exists()
    assert index_path.exists()

    manifest = _read_json(manifest_path)
    assert manifest["contract_name"] == "stageb_feature_export_enablement_contract_v1"
    assert manifest["contract_version"] == "v1"
    assert manifest["embedding_dtype"] == "float32"
    assert manifest["embedding_dim"] == 4
    assert manifest["video_shards"] == ["videos/1.json", "videos/2.json"]

    video2 = _read_json(shard_2)
    assert video2["video_id"] == 2
    assert "runtime_evidence" in video2
    assert "tracks" in video2
    assert [t["track_id"] for t in video2["tracks"]] == [4, 9]


def test_missing_required_embedding_hard_fails(tmp_path: Path):
    payload = _base_payload()
    del payload["videos"][0]["tracks"][0]["embedding"]

    with pytest.raises(ExportContractError, match="embedding"):
        build_feature_export_enablement_v1(input_payload=payload, run_root=tmp_path)


def test_non_finite_embedding_hard_fails(tmp_path: Path):
    payload = _base_payload()
    payload["videos"][0]["tracks"][0]["embedding"] = [0.0, float("inf"), 1.0, 2.0]

    with pytest.raises(ExportContractError, match="finite"):
        build_feature_export_enablement_v1(input_payload=payload, run_root=tmp_path)


def test_track_id_required_hard_fails(tmp_path: Path):
    payload = _base_payload()
    del payload["videos"][0]["tracks"][0]["track_id"]

    with pytest.raises(ExportContractError, match="track_id"):
        build_feature_export_enablement_v1(input_payload=payload, run_root=tmp_path)


def test_runtime_evidence_presence_and_unprocessed_boundary_hard_fail(tmp_path: Path):
    payload = _base_payload()
    del payload["videos"][0]["runtime_evidence"]
    with pytest.raises(ExportContractError, match="runtime_evidence"):
        build_feature_export_enablement_v1(input_payload=payload, run_root=tmp_path)

    payload = _base_payload()
    payload["videos"][0]["runtime_evidence"]["stageb_completion_marker"] = "unprocessed"
    with pytest.raises(ExportContractError, match="unprocessed"):
        build_feature_export_enablement_v1(input_payload=payload, run_root=tmp_path)


def test_deterministic_ordering_for_video_and_track_shards(tmp_path: Path):
    payload_a = _base_payload()
    payload_b = deepcopy(payload_a)
    payload_b["videos"] = list(reversed(payload_b["videos"]))
    payload_b["videos"][0]["tracks"] = list(reversed(payload_b["videos"][0]["tracks"]))

    out_a = build_feature_export_enablement_v1(input_payload=payload_a, run_root=tmp_path / "a")
    out_b = build_feature_export_enablement_v1(input_payload=payload_b, run_root=tmp_path / "b")

    manifest_a = _read_json(out_a / "manifest.json")
    manifest_b = _read_json(out_b / "manifest.json")
    assert manifest_a == manifest_b
    assert _read_json(out_a / "videos" / "2.json") == _read_json(out_b / "videos" / "2.json")
