from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.stageb.test_g8_infer_bridge import _prepare_infer_fixture, _write_json
from videocutler.run_stageb_infer_ov import main as run_infer_main


def _prepare_ytvis2019_annotations(root: Path) -> None:
    payload = {
        "videos": [{"id": 101, "length": 2, "height": 28, "width": 28, "file_names": ["000.jpg", "001.jpg"]}],
        "categories": [{"id": 1, "name": "yt_cls_one"}, {"id": 3, "name": "yt_cls_three"}],
        "annotations": [],
    }
    _write_json(root / "ytvis_2019" / "valid.json", payload)


def test_run_stageb_infer_ov_ytvis2019_emits_canonical_prediction_artifacts(tmp_path: Path, monkeypatch) -> None:
    datasets_root = tmp_path / "datasets"
    _prepare_ytvis2019_annotations(datasets_root)
    _prepare_infer_fixture(tmp_path)

    # Mirror the same authoritative exports/banks under ytvis_2019_val for a minimal compatibility smoke.
    source_paths = [
        (tmp_path / "carrier_bank" / "lvvis_val", tmp_path / "carrier_bank" / "ytvis_2019_val"),
        (tmp_path / "frame_bank" / "lvvis_val", tmp_path / "frame_bank" / "ytvis_2019_val"),
        (tmp_path / "exports" / "lvvis_val", tmp_path / "exports" / "ytvis_2019_val"),
    ]
    for src, dst in source_paths:
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            target = dst / child.name
            if child.is_file():
                target.write_bytes(child.read_bytes())
            elif child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                for nested in child.rglob('*'):
                    rel = nested.relative_to(child)
                    out = target / rel
                    if nested.is_dir():
                        out.mkdir(parents=True, exist_ok=True)
                    else:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_bytes(nested.read_bytes())

    monkeypatch.setenv("DETECTRON2_DATASETS", str(datasets_root))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_stageb_infer_ov.py",
            "--exp_name",
            "toy_g8_infer_ytvis2019",
            "--dataset_name",
            "ytvis_2019_val",
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

    pred_main = json.loads((tmp_path / "predictions" / "ytvis_2019_val" / "pred_main.json").read_text(encoding="utf-8"))
    pred_diag = json.loads((tmp_path / "predictions" / "ytvis_2019_val" / "pred_diag.json").read_text(encoding="utf-8"))

    assert len(pred_main) == 1
    assert len(pred_diag) == 1
    assert pred_main[0]["trajectory_id"] == "traj-101-0"
    assert pred_diag[0]["top1_known_name"] in {"yt_cls_one", "yt_cls_three"}
