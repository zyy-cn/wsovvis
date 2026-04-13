from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


class _FakeTokenMatrix:
    def __init__(self, token_count: int, dim: int) -> None:
        self.shape = (int(token_count), int(dim))


class _FakePatchTokenBatch:
    def __init__(self, batch_size: int, token_count: int, dim: int) -> None:
        self._rows = [_FakeTokenMatrix(token_count, dim) for _ in range(int(batch_size))]

    def __getitem__(self, index: int):
        return self._rows[index]


class _FakeTorchTensor:
    def __init__(self, value) -> None:
        self._value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._value


class _FakeInputTensor:
    def __init__(self, c: int, h: int, w: int) -> None:
        self.shape = (int(c), int(h), int(w))

    def to(self, device, non_blocking=True):
        return self


class _FakeBatchTensor:
    def __init__(self, batch_size: int, c: int, h: int, w: int) -> None:
        self.shape = (int(batch_size), int(c), int(h), int(w))

    def to(self, device, non_blocking=True):
        return self


def _install_fake_modules() -> None:
    pil_mod = ModuleType("PIL")
    pil_image_mod = ModuleType("PIL.Image")

    class _FakeImageModule:
        @staticmethod
        def open(path):
            raise RuntimeError("PIL open not expected in stub regression")

        @staticmethod
        def new(mode, size, color):
            return object()

    pil_image_mod.Image = object
    pil_image_mod.open = _FakeImageModule.open
    pil_image_mod.new = _FakeImageModule.new
    pil_mod.Image = pil_image_mod
    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = pil_image_mod

    np_mod = ModuleType("numpy")
    np_mod.float32 = "float32"
    np_mod.uint8 = "uint8"
    np_mod.ndarray = object
    np_mod.asarray = lambda x, dtype=None: x
    sys.modules["numpy"] = np_mod

    torch_mod = ModuleType("torch")
    torch_mod.device = lambda x: x

    def _inference_mode():
        def _decorator(fn):
            return fn

        return _decorator

    def _stack(tensors, dim=0):
        if not tensors:
            return _FakeBatchTensor(0, 0, 0, 0)
        base = tensors[0].shape
        for tensor in tensors[1:]:
            if tensor.shape != base:
                raise RuntimeError("stack expects each tensor to be equal size")
        c, h, w = base
        return _FakeBatchTensor(len(tensors), c, h, w)

    torch_mod.inference_mode = _inference_mode
    torch_mod.stack = _stack
    torch_mod.Tensor = _FakeInputTensor
    sys.modules["torch"] = torch_mod

    tv_mod = ModuleType("torchvision")
    tv_tf_mod = ModuleType("torchvision.transforms")
    tv_f_mod = ModuleType("torchvision.transforms.functional")

    class _Interp:
        BICUBIC = "bicubic"

    tv_f_mod.InterpolationMode = _Interp
    tv_f_mod.normalize = lambda x, mean, std: x
    tv_f_mod.pil_to_tensor = lambda x: _FakeInputTensor(3, 224, 224)
    tv_f_mod.resize = lambda img, size, interpolation=None, antialias=True: img
    tv_tf_mod.functional = tv_f_mod
    tv_mod.transforms = tv_tf_mod
    sys.modules["torchvision"] = tv_mod
    sys.modules["torchvision.transforms"] = tv_tf_mod
    sys.modules["torchvision.transforms.functional"] = tv_f_mod

    dinov2_mod = ModuleType("dinov2")
    hub_mod = ModuleType("dinov2.hub")
    backbones_mod = ModuleType("dinov2.hub.backbones")

    class _FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def forward_features(self, batch):
            bsz, _, h, w = batch.shape
            token_count = int((h // 14) * (w // 14))
            fake = _FakePatchTokenBatch(bsz, token_count, 8)
            return {"x_norm_patchtokens": _FakeTorchTensor(fake)}

    backbones_mod.dinov2_vitb14_reg = lambda pretrained=True, weights=None: _FakeModel()
    hub_mod.backbones = backbones_mod
    dinov2_mod.hub = hub_mod
    sys.modules["dinov2"] = dinov2_mod
    sys.modules["dinov2.hub"] = hub_mod
    sys.modules["dinov2.hub.backbones"] = backbones_mod


@dataclass(frozen=True)
class _Geom:
    grid_h: int
    grid_w: int


class _Mask:
    def __init__(self, h: int, w: int) -> None:
        self.shape = (int(h), int(w))


def run_regression() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    _install_fake_modules()
    from videocutler.ext_stageb_ovvis.models.visual_encoder_dinov2 import DinoV2VisualEncoder

    encoder = DinoV2VisualEncoder(
        dinov2_ckpt="vitb14_reg",
        device="cpu",
        batch_size_frames=4,
        preprocess_mode="keep_aspect_pad",
        resize_short_side=672,
        max_long_side=1008,
        pad_to_multiple=14,
        patch_size=14,
    )

    per_path = {
        "a.jpg": (3, 574, 1008),
        "b.jpg": (3, 672, 896),
        "c.jpg": (3, 574, 1008),
    }

    def _fake_load_image_tensor(path: Path):
        c, h, w = per_path[path.name]
        grid_h = int(h // 14)
        grid_w = int(w // 14)
        return _FakeInputTensor(c, h, w), _Geom(grid_h=grid_h, grid_w=grid_w), _Mask(grid_h, grid_w)

    encoder._load_image_tensor = _fake_load_image_tensor  # type: ignore[method-assign]
    paths = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]

    outputs = encoder.encode_image_paths(paths)
    assert len(outputs) == 3
    assert len({per_path[p.name][1:] for p in paths}) >= 2
    for item in outputs:
        token_count = int(item.geometry.grid_h) * int(item.geometry.grid_w)
        assert int(item.features.shape[0]) == token_count
        assert tuple(item.valid_token_mask.shape) == (int(item.geometry.grid_h), int(item.geometry.grid_w))


if __name__ == "__main__":
    run_regression()
    print("PASS: mixed-shape keep-aspect-pad batch regression")
