from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Sample:
    clip_id: str
    frame_index: int
    image_path: Path


@dataclass(frozen=True)
class _Geom:
    grid_h: int
    grid_w: int


class _Feat:
    def __init__(self, token_count: int) -> None:
        self.shape = (int(token_count), 8)


class _Mask:
    def __init__(self, h: int, w: int) -> None:
        self.shape = (int(h), int(w))


@dataclass(frozen=True)
class _Encoded:
    features: _Feat
    geometry: _Geom
    valid_token_mask: _Mask


class _FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode_image_paths(self, paths):
        self.calls.append(len(paths))
        out = []
        for p in paths:
            idx = int(p.stem.replace("f", ""))
            grid_h = 41 if idx % 2 == 0 else 48
            grid_w = 72 if idx % 2 == 0 else 64
            out.append(_Encoded(features=_Feat(grid_h * grid_w), geometry=_Geom(grid_h=grid_h, grid_w=grid_w), valid_token_mask=_Mask(grid_h, grid_w)))
        return out


class _FakeStreamWriter:
    def __init__(self) -> None:
        self.calls = 0
        self.ordered_frame_indices: list[int] = []

    def write_sample(self, sample, feature, geometry, valid_token_mask):
        self.calls += 1
        self.ordered_frame_indices.append(int(sample.frame_index))
        assert int(feature.shape[0]) == int(geometry.grid_h) * int(geometry.grid_w)
        assert tuple(valid_token_mask.shape) == (int(geometry.grid_h), int(geometry.grid_w))


def run_regression() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from videocutler.run_stageb_extract_dinov2_frames import _stream_encode_and_write

    samples = [_Sample(clip_id="0", frame_index=i, image_path=Path(f"f{i}.jpg")) for i in range(10)]
    encoder = _FakeEncoder()
    writer = _FakeStreamWriter()
    total = _stream_encode_and_write(samples, encoder, writer, chunk_size=3)

    assert total == 10
    assert writer.calls == 10
    assert writer.ordered_frame_indices == list(range(10))
    assert max(encoder.calls) <= 3
    assert len(encoder.calls) == 4


if __name__ == "__main__":
    run_regression()
    print("PASS: streaming control-flow regression")
