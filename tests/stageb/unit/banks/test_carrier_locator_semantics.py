from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np


TARGET_TRAJECTORY_ID = "videocutler_r50_native:lvvis_train_base:1013:000009"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _find_trajectory(root: Path):
    path = root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl"
    for row in _load_jsonl(path):
        if str(row.get("trajectory_id")) == TARGET_TRAJECTORY_ID:
            return row
    raise AssertionError(f"trajectory not found: {TARGET_TRAJECTORY_ID}")


def _find_carrier(root: Path):
    path = root / "carrier_bank" / "lvvis_train_base" / "carrier_records.jsonl"
    for row in _load_jsonl(path):
        if str(row.get("trajectory_id")) == TARGET_TRAJECTORY_ID:
            return row
    raise AssertionError(f"carrier not found: {TARGET_TRAJECTORY_ID}")


def _load_train_mainline_pairs(root: Path):
    traj_path = root / "exports" / "lvvis_train_base" / "trajectory_records.jsonl"
    carrier_path = root / "carrier_bank" / "lvvis_train_base" / "carrier_records.jsonl"
    carrier_by_tid = {str(row["trajectory_id"]): row for row in _load_jsonl(carrier_path)}
    paired = []
    for traj in _load_jsonl(traj_path):
        tid = str(traj.get("trajectory_id"))
        carrier = carrier_by_tid.get(tid)
        if carrier is not None:
            paired.append((traj, carrier))
    return paired


class CarrierLocatorSemanticsTest(unittest.TestCase):
    def test_train_mainline_carrier_locator_uses_valid_frame_subsequence(self) -> None:
        root = _repo_root()
        traj = _find_trajectory(root)
        carrier = _find_carrier(root)

        raw_frame_indices = [int(x) for x in traj.get("frame_indices", [])]
        valid_positions = [
            pos
            for pos, box in enumerate(list(traj.get("boxes_xyxy", [])))
            if box is not None
        ]
        valid_frame_indices = [raw_frame_indices[pos] for pos in valid_positions]

        carrier_frame_indices = [int(x) for x in carrier.get("frame_indices", [])]
        carrier_frame_paths = list(carrier.get("frame_carriers_norm_paths", []))

        self.assertEqual(len(raw_frame_indices), 24)
        self.assertEqual(len(valid_frame_indices), 16)
        self.assertEqual(carrier_frame_indices, valid_frame_indices)
        self.assertEqual(len(carrier_frame_paths), len(valid_frame_indices))
        self.assertTrue(all(path.endswith("]") for path in carrier_frame_paths))
        self.assertEqual(list(carrier_frame_indices), sorted(carrier_frame_indices))

    def test_train_mainline_bounded_subset_uses_valid_frame_subsequence(self) -> None:
        root = _repo_root()
        paired = _load_train_mainline_pairs(root)
        ranked = sorted(
            paired,
            key=lambda item: (
                len(list(item[0].get("frame_indices", []))),
                int(item[0].get("clip_id", 0)),
                str(item[0].get("trajectory_id", "")),
            ),
        )
        sample_count = min(8, len(ranked))
        sample_indices = np.linspace(0, len(ranked) - 1, sample_count, dtype=int).tolist() if ranked else []
        sampled = [ranked[idx] for idx in sample_indices]

        self.assertEqual(len(sampled), sample_count)
        for traj, carrier in sampled:
            raw_frame_indices = [int(x) for x in traj.get("frame_indices", [])]
            valid_positions = [
                pos
                for pos, box in enumerate(list(traj.get("boxes_xyxy", [])))
                if box is not None
            ]
            valid_frame_indices = [raw_frame_indices[pos] for pos in valid_positions]
            carrier_frame_indices = [int(x) for x in carrier.get("frame_indices", [])]
            carrier_frame_paths = list(carrier.get("frame_carriers_norm_paths", []))

            self.assertGreater(len(valid_frame_indices), 0)
            self.assertEqual(carrier_frame_indices, valid_frame_indices)
            self.assertEqual(len(carrier_frame_paths), len(valid_frame_indices))
            self.assertTrue(all(path.endswith("]") for path in carrier_frame_paths))
            self.assertEqual(list(carrier_frame_indices), sorted(carrier_frame_indices))
