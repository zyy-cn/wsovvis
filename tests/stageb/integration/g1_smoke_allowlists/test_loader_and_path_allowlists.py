from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_smoke_allowlist_fixtures_match_g1_contract():
    lvvis = json.loads(Path("package/assets/fixtures/tiny_lvvis_pipeline_case/allowlist_min.json").read_text())
    ytvis = json.loads(Path("package/assets/fixtures/tiny_ytvis_validation_case/allowlist_min.json").read_text())

    assert lvvis["lvvis_train_smoke_rule"] == "first_4_video_ids_sorted"
    assert lvvis["lvvis_val_smoke_rule"] == "first_2_video_ids_sorted"
    assert lvvis["ytvis_2019_val_smoke_ids"] == ["0062f687f1", "00f88c4f0a"]
    assert ytvis["ytvis_2019_val_smoke_ids"] == lvvis["ytvis_2019_val_smoke_ids"]


def test_builtin_dataset_files_are_not_modified():
    protected = [
        "videocutler/mask2former_video/data_video/datasets/builtin.py",
        "videocutler/mask2former_video/data_video/datasets/ytvis.py",
        "videocutler/mask2former_video/data_video/ytvis_eval.py",
    ]
    result = subprocess.run(
        ["git", "diff", "--name-only", "--", *protected],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_allowlist_contract_check_artifact_is_grounded():
    path = Path("outputs/g1_smoke_allowlists/allowlist_contract_check.json")
    assert path.exists()
    data = json.loads(path.read_text())

    assert data["gate_id"] == "G1_smoke_allowlists"
    assert data["task_id"] == "G1_smoke_allowlists-task"
    assert data["status"] == "PASS"
    assert data["builtin_dataset_files_touched"] is False
    assert data["smoke_allowlists"]["ytvis_2019_val_smoke_ids"] == ["0062f687f1", "00f88c4f0a"]
