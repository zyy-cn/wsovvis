#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

# ---- Embedded minimal DeepLab + PointRend (offline) ----

DEEPLAB_CONFIG = r"""# Minimal DeepLab project for Detectron2 projects loader (offline)
def add_deeplab_config(cfg):
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE = "hard_pixel_mining"
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_FEATURES = ["res2"]
    cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS = [48]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS = 256
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS = [6, 12, 18]
    cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT = 0.1
    cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV = False
    cfg.MODEL.RESNETS.RES4_DILATION = 1
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 2, 4]
    cfg.MODEL.RESNETS.STEM_TYPE = "deeplab"
"""

DEEPLAB_INIT = r"""from .config import add_deeplab_config
__all__ = ["add_deeplab_config"]
"""

POINTREND_CONFIG = r"""def add_pointrend_config(cfg):
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.SOLVER.POLY_LR_POWER = 0.9
    cfg.SOLVER.POLY_LR_CONSTANT_ENDING = 0.0
    cfg.MODEL.ROI_MASK_HEAD.POINT_HEAD_ON = False
    # Minimal placeholder node
    class _CN: pass
    cfg.MODEL.POINT_HEAD = _CN()
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = 80
    cfg.MODEL.POINT_HEAD.IN_FEATURES = ["p2"]
    cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 196
    cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO = 3
    cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO = 0.75
"""

POINTREND_COLOR_AUG = r"""import numpy as np
from detectron2.data.transforms.augmentation import Augmentation
from detectron2.data.transforms.transform import Transform

class ColorAugSSDTransform(Transform):
    def __init__(self, img_format="RGB"):
        super().__init__()
        self.img_format = img_format

    def apply_image(self, img):
        if self.img_format == "BGR":
            img = img[..., ::-1]
        img = img.astype(np.float32)
        img = self._brightness(img)
        img = self._contrast(img)
        img = self._saturation(img)
        img = self._hue(img)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if self.img_format == "BGR":
            img = img[..., ::-1]
        return img

    def apply_coords(self, coords):
        return coords

    def _brightness(self, img, delta=32):
        if np.random.rand() < 0.5:
            img = img + np.random.uniform(-delta, delta)
        return img

    def _contrast(self, img, lower=0.5, upper=1.5):
        if np.random.rand() < 0.5:
            img = img * np.random.uniform(lower, upper)
        return img

    def _saturation(self, img, lower=0.5, upper=1.5):
        if np.random.rand() < 0.5:
            gray = img.mean(axis=2, keepdims=True)
            a = np.random.uniform(lower, upper)
            img = img * a + gray * (1 - a)
        return img

    def _hue(self, img):
        if np.random.rand() < 0.5:
            k = np.random.randint(0, 3)
            img = np.roll(img, shift=k, axis=2)
        return img

class ColorAugSSD(Augmentation):
    def __init__(self, img_format="RGB"):
        super().__init__()
        self.img_format = img_format

    def get_transform(self, image):
        return ColorAugSSDTransform(img_format=self.img_format)
"""

POINTREND_POINT_FEATURES = r"""import torch
import torch.nn.functional as F

def point_sample(input, point_coords, **kwargs):
    if point_coords.dim() == 3:
        point_coords = point_coords.unsqueeze(2)  # (N,P,1,2)
    grid = point_coords * 2.0 - 1.0
    out = F.grid_sample(input, grid, align_corners=False, **kwargs)
    return out.squeeze(3)

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    assert oversample_ratio >= 1
    assert 0 <= importance_sample_ratio <= 1
    N, C, H, W = coarse_logits.shape
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(N, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords)
    uncertainties = uncertainty_func(point_logits)
    if uncertainties.dim() == 3:
        uncertainties = uncertainties.squeeze(1)
    num_uncertain = int(num_points * importance_sample_ratio)
    num_random = num_points - num_uncertain
    idx = torch.topk(uncertainties, k=num_uncertain, dim=1).indices
    batch_idx = torch.arange(N, device=coarse_logits.device)[:, None]
    uncertain_coords = point_coords[batch_idx, idx]
    if num_random > 0:
        rand_coords = torch.rand(N, num_random, 2, device=coarse_logits.device)
        return torch.cat([uncertain_coords, rand_coords], dim=1)
    return uncertain_coords
"""

POINTREND_INIT = r"""from .config import add_pointrend_config
from .color_augmentation import ColorAugSSDTransform, ColorAugSSD
__all__ = ["add_pointrend_config", "ColorAugSSDTransform", "ColorAugSSD"]
"""

def ensure_projects_map(projects_init: Path, key: str, value: str) -> bool:
    txt = projects_init.read_text(encoding="utf-8")
    if re.search(rf'["\']{re.escape(key)}["\']\s*:\s*["\']{re.escape(value)}["\']', txt):
        return False
    m = re.search(r"_PROJECTS\s*=\s*\{", txt)
    if not m:
        raise RuntimeError(f"Cannot find _PROJECTS dict in {projects_init}")
    insert_pos = m.end()
    insertion = f'\n    "{key}": "{value}",'
    projects_init.write_text(txt[:insert_pos] + insertion + txt[insert_pos:], encoding="utf-8")
    return True

def write_if_missing(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return False
    path.write_text(content, encoding="utf-8")
    return True

def patch_pillow_linear(vnext_root: Path):
    target = vnext_root / "detectron2" / "data" / "transforms" / "transform.py"
    if not target.exists():
        return False, f"skip: {target} not found"
    txt = target.read_text(encoding="utf-8")
    if "Image.LINEAR" not in txt:
        return False, "skip: no Image.LINEAR"
    target.write_text(txt.replace("Image.LINEAR", "Image.BILINEAR"), encoding="utf-8")
    return True, f"patched: {target}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsovvis-root", required=True)
    args = ap.parse_args()

    wsovvis = Path(args.wsovvis_root).expanduser().resolve()
    vnext = wsovvis / "third_party" / "VNext"
    if not vnext.exists():
        raise SystemExit(f"VNext not found at: {vnext}")

    projects_init = vnext / "detectron2" / "projects" / "__init__.py"
    if not projects_init.exists():
        raise SystemExit(f"detectron2 projects init not found: {projects_init}")

    changes = []

    changes.append(("projects_map:deeplab", ensure_projects_map(projects_init, "deeplab", "DeepLab")))
    changes.append(("projects_map:point_rend", ensure_projects_map(projects_init, "point_rend", "PointRend")))

    deep = vnext / "projects" / "DeepLab" / "deeplab"
    changes.append(("DeepLab/config.py", write_if_missing(deep / "config.py", DEEPLAB_CONFIG)))
    changes.append(("DeepLab/__init__.py", write_if_missing(deep / "__init__.py", DEEPLAB_INIT)))

    pr = vnext / "projects" / "PointRend" / "point_rend"
    changes.append(("PointRend/config.py", write_if_missing(pr / "config.py", POINTREND_CONFIG)))
    changes.append(("PointRend/color_augmentation.py", write_if_missing(pr / "color_augmentation.py", POINTREND_COLOR_AUG)))
    changes.append(("PointRend/point_features.py", write_if_missing(pr / "point_features.py", POINTREND_POINT_FEATURES)))
    changes.append(("PointRend/__init__.py", write_if_missing(pr / "__init__.py", POINTREND_INIT)))

    did, msg = patch_pillow_linear(vnext)
    changes.append(("Pillow Image.LINEAR fix", did))

    print("[patch summary]")
    for name, did in changes:
        print(f" - {name}: {'CHANGED' if did else 'ok/skip'}")
    print(f" - pillow_detail: {msg}")

    print("\n[verify imports]")
    print("python - <<'PY'")
    print("from detectron2.projects.deeplab import add_deeplab_config")
    print("from detectron2.projects.point_rend import ColorAugSSDTransform")
    print("from detectron2.projects.point_rend.point_features import point_sample")
    print("print('OK imports')")
    print("PY")

if __name__ == "__main__":
    main()
