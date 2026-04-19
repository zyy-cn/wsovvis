from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from videocutler.ext_stageb_ovvis.eval import external_ytvis2019 as ext
from videocutler.ext_stageb_ovvis.eval.external_ytvis2019 import ExternalYTVIS2019EvalConfig, run_external_ytvis2019_eval


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class _FakeYTVOS:
    def __init__(self, annotation_file: str):
        self.annotation_file = annotation_file
        self.dataset = json.loads(Path(annotation_file).read_text(encoding="utf-8"))

    def loadRes(self, rows):
        return {"rows": list(rows)}


class _FakeYTVOSeval:
    def __init__(self, gt_api, dt_api, iou_type: str):
        self.gt_api = gt_api
        self.dt_api = dt_api
        self.iou_type = iou_type
        self.params = SimpleNamespace(useCats=1)
        num_categories = len(gt_api.dataset.get("categories", []))
        precision = np.full((10, 101, max(1, num_categories), 1, 3), -1.0, dtype=np.float32)
        for index in range(num_categories):
            precision[:, :, index, 0, -1] = 1.0
        self.eval = {"precision": precision}
        self.stats = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    def evaluate(self):
        return None

    def accumulate(self):
        return None

    def summarize(self):
        return None


def test_g8_external_ytvis2019_eval_smoke(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "ytvis_2019"
    dataset_root.mkdir(parents=True, exist_ok=True)

    common_payload = {
        "videos": [{"id": 11, "width": 2, "height": 2, "length": 2, "file_names": ["a.jpg", "b.jpg"]}],
        "categories": [{"id": 4, "name": "yt_cat"}],
        "annotations": [
            {
                "id": 1,
                "video_id": 11,
                "category_id": 4,
                "segmentations": [{"size": [2, 2], "counts": "stub"}, None],
                "areas": [2.0, None],
                "bboxes": [[0.0, 0.0, 2.0, 1.0], None],
                "avg_area": 2.0,
                "iscrowd": 0,
            }
        ],
    }
    _write_json(dataset_root / "valid.json", common_payload)

    pred_main = [{
        "trajectory_id": "traj-11-0",
        "video_id": 11,
        "score": 0.95,
        "category_id": 4,
        "segmentations": [{"size": [2, 2], "counts": "stub"}, None],
    }]
    output_root = tmp_path / "formal_chain"
    _write_json(output_root / "predictions" / "ytvis_2019_val" / "pred_main.json", pred_main)

    monkeypatch.setenv("WSOVVIS_YTVIS2019_ROOT", str(dataset_root))
    monkeypatch.setattr(ext, "load_ytvis2019_backend_classes", lambda: (_FakeYTVOS, _FakeYTVOSeval))
    result = run_external_ytvis2019_eval(
        ExternalYTVIS2019EvalConfig(
            exp_name="toy_eval",
            output_root=output_root,
            seed=0,
            smoke=True,
        )
    )

    assert result["status"] == "OK"
    metrics_path = output_root / "eval" / "ytvis2019" / "external_metrics.ytvis2019.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["benchmark_name"] == "ytvis2019"
    assert payload["dataset_name"] == "ytvis_2019_val"
    assert payload["split_tag"] == "aux_val_smoke"
    assert payload["AP"] == 100.0
    assert payload["AP50"] == 100.0
    assert payload["AP75"] == 100.0
    assert payload["mAPb"] == 0.0
    assert payload["mAPn"] == 0.0
    assert payload["base_raw_ids"] == []
    assert payload["novel_raw_ids"] == []
    assert payload["per_category_ap"] == [{"raw_category_id": 4, "AP": 100.0}]
