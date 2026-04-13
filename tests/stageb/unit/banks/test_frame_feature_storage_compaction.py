from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


@dataclass
class FakeArray:
    data: object
    dtype: str = "float32"

    @property
    def ndim(self) -> int:
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return 2
        if isinstance(self.data, list):
            return 1
        return 0

    @property
    def shape(self):
        if self.ndim == 2:
            return (len(self.data), len(self.data[0]))
        if self.ndim == 1:
            return (len(self.data),)
        return ()

    def astype(self, dtype):
        return FakeArray(self.data, dtype=str(dtype))

    def __getitem__(self, idx):
        if isinstance(self.data, list):
            return self.data[idx]
        raise TypeError("not indexable")

    def __setitem__(self, idx, value):
        if not isinstance(self.data, list):
            raise TypeError("not assignable")
        rows = range(*idx[0].indices(len(self.data))) if isinstance(idx, tuple) else range(*idx.indices(len(self.data)))
        if isinstance(idx, tuple):
            row_slice, col_slice = idx
            row_indices = range(*row_slice.indices(len(self.data)))
            col_indices = range(*col_slice.indices(len(self.data[0])))
            for r in row_indices:
                for c in col_indices:
                    self.data[r][c] = value
        else:
            for r in rows:
                self.data[r] = value


def _install_fake_numpy(payload_store):
    np_mod = ModuleType("numpy")
    np_mod.float16 = "float16"
    np_mod.float32 = "float32"
    np_mod.uint8 = "uint8"
    np_mod.ndarray = FakeArray
    np_mod.asarray = lambda x, dtype=None: x.astype(dtype) if hasattr(x, "astype") and dtype is not None else x
    np_mod.expand_dims = lambda x, axis=0: x
    np_mod.array_equal = lambda a, b: a.data == b.data and a.dtype == b.dtype
    np_mod.ceil = __import__("math").ceil
    np_mod.floor = __import__("math").floor

    def zeros(shape, dtype=None):
        h, w = shape
        return FakeArray([[0 for _ in range(w)] for _ in range(h)], dtype=dtype or "uint8")

    def savez_compressed(path, **kwargs):
        payload_store[str(path)] = kwargs

    class _FakeLoadResult(dict):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @property
        def files(self):
            return list(self.keys())

    def load(path, allow_pickle=False):
        payload = payload_store[str(path)]
        return _FakeLoadResult(payload)

    np_mod.zeros = zeros
    np_mod.savez_compressed = savez_compressed
    np_mod.load = load
    sys.modules["numpy"] = np_mod


def run_regression() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    payload_store = {}
    _install_fake_numpy(payload_store)

    from videocutler.ext_stageb_ovvis.banks.frame_feature_bank import (
        FrameBankStreamWriter,
        FrameSample,
        FrameGeometry,
        read_feature_vector,
        read_valid_token_mask,
    )

    with tempfile.TemporaryDirectory(prefix="g4_compaction_") as td:
        root = Path(td)
        frame_records = root / "frame_records.jsonl"
        geom_records = root / "frame_geom_records.jsonl"
        sample_a = FrameSample(clip_id="7", frame_index=0, image_path=root / "img.jpg")
        sample_b = FrameSample(clip_id="8", frame_index=0, image_path=root / "img.jpg")
        geom = FrameGeometry(
            orig_h=1080,
            orig_w=1920,
            resized_h=574,
            resized_w=1008,
            padded_h=574,
            padded_w=1008,
            scale_y=574 / 1080,
            scale_x=1008 / 1920,
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
            patch_size=14,
            grid_h=41,
            grid_w=72,
        )
        valid_mask = FakeArray([[1] * 72 for _ in range(41)], dtype="uint8")
        feature = FakeArray([[0.1] * 8 for _ in range(41 * 72)], dtype="float32")

        with FrameBankStreamWriter(root, frame_records, geom_records) as writer:
            writer.write_sample(sample_a, feature, geom, valid_mask)
            writer.write_sample(sample_b, feature, geom, valid_mask)

        assert any("clip_7_feats.npz" in path for path in payload_store)
        assert any("clip_8_feats.npz" in path for path in payload_store)
        first_payload_path = next(path for path in payload_store if path.endswith("clip_7_feats.npz"))
        second_payload_path = next(path for path in payload_store if path.endswith("clip_8_feats.npz"))
        first_payload = payload_store[first_payload_path]
        second_payload = payload_store[second_payload_path]
        assert first_payload["slot_0"].dtype == "float16"
        assert second_payload["slot_0"].dtype == "float16"
        assert all("valid_token_mask" not in path for path in payload_store)

        feat_vec = read_feature_vector(root, "payload/clip_7_feats.npz#0")
        assert feat_vec.dtype == "float32"
        feat_vec_2 = read_feature_vector(root, "payload/clip_8_feats.npz#0")
        assert feat_vec_2.dtype == "float32"

        geom_lines = [json.loads(line) for line in geom_records.read_text(encoding="utf-8").splitlines() if line.strip()]
        reconstructed = read_valid_token_mask(root, "frame_geom_records.jsonl#1")
        assert reconstructed.shape == (geom_lines[1]["grid_h"], geom_lines[1]["grid_w"])


if __name__ == "__main__":
    run_regression()
    print("PASS: storage compaction regression")
