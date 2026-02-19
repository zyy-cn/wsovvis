# WS-OVVIS (LV-VIS)
**VideoCutLER pseudo tubes → SeqFormer training (DDP + Sacred)**  
**(New) Optional: replace SeqFormer backbone with a frozen DINOv2 ViT**

This repo provides a reproducible minimal pipeline to:

1. Run **VideoCutLER** to generate per-video `mask_*.png` pseudo masks  
2. Convert masks into **YTVIS/LV-VIS-style JSON** tubes  
3. Train **SeqFormer (VNext/Detectron2)** on pseudo tubes using **DDP + Sacred**  
4. Evaluate on **LV-VIS ground-truth masks** with a **class-agnostic** JSON  

> The current pipeline is **class-agnostic (1 class)** by design: pseudo tubes do not contain reliable category labels yet.  
> Once you add tube→category attribution, you can switch to standard LV-VIS multi-class evaluation.

---

## 0) Repository + `third_party` status (important)

This pipeline depends on third-party components under `third_party/`:

- `third_party/CutLER` — VideoCutLER  
- `third_party/VNext` — Detectron2 fork + SeqFormer project  
- **(New)** `third_party/dinov2` — DINOv2 repository (only needed if you enable the DINOv2 backbone)

### 0.1 Clone this repo
```bash
git clone --recursive <YOUR_FORK_OR_THIS_REPO_URL> wsovvis
cd wsovvis
git submodule update --init --recursive
```

If you do **not** use submodules, place the correct code manually under `third_party/` and make sure paths match the sections below.

### 0.2 (New) Clone DINOv2 into `third_party/` (optional)
Only needed for the **DINOv2 frozen-backbone** mode.

From repo root:
```bash
mkdir -p third_party
git clone --recursive https://github.com/facebookresearch/dinov2.git third_party/dinov2
```

Sanity check:
```bash
test -f third_party/dinov2/hubconf.py && echo "hubconf OK"
test -f third_party/dinov2/dinov2/__init__.py && echo "dinov2 pkg OK"
```

### 0.3 Required patches already tracked in GitHub
To reproduce, use a repo revision where the following patches are present:

**VideoCutLER stability (0 instances videos)**
- `third_party/CutLER/videocutler/demo_video/demo_masks_only.py`  
  Safety fix so videos with **0 instances** do not crash a GPU worker.

**(New) DINOv2 backbone integration**
- `wsovvis/modeling/backbone/dinov2_backbone.py`  
  Frozen DINOv2 backbone wrapper + pseudo multi-scale outputs.
- `train_seqformer_pseudo.py`  
  Config extension `MODEL.DINOV2` before `cfg.merge_from_file(...)` and forced import to register the backbone.

**(New) SeqFormer compatibility fixes for DINOv2**
- `third_party/VNext/projects/SeqFormer/seqformer/seqformer.py`  
  Pad `tensors` and `mask` to a multiple of DINOv2 patch size (14) when using DINOv2.
- `third_party/VNext/projects/SeqFormer/seqformer/models/deformable_detr.py`  
  Resize `pred_masks` to GT mask resolution before BCE/Focal loss to avoid shape mismatch.

---

## 1) Environment

### 1.1 Create conda env
```bash
conda create -n wsovvis python=3.10 -y
conda activate wsovvis

python -m pip install -U pip wheel
# Sacred imports pkg_resources; pin setuptools to keep pkg_resources available
python -m pip install -U "setuptools<81"
python -c "import pkg_resources; print('pkg_resources OK')"
```

### 1.2 Install PyTorch (example: CUDA 11.8)
```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
```
> Use a PyTorch build that matches your CUDA runtime/driver.

### 1.3 Install Python dependencies
```bash
python -m pip install -r requirements.txt
```


### 1.4 Notes on evaluation dependencies (YTVIS / LV-VIS)

This repo uses **YTVIS-style** evaluation (via `pycocotools.ytvos` / `pycocotools.ytvoseval`) when ground-truth annotations are available.

Common pitfalls:

- **NumPy compatibility (`np.float` error):** some YTVIS/YTVOS forks still use `np.float` (removed in NumPy ≥ 1.24).
  - Recommended: keep `numpy<1.24` (already pinned in `requirements.txt`).
  - If you must use newer NumPy, apply a small compatibility patch (see `patches/vnext_ytvis_eval_enable_eval.patch` which includes a safe NumPy shim).

- **Evaluator behaviour:** the SeqFormer evaluator will:
  - always **dump predictions** to `.../inference/results.json`;
  - compute metrics **only if** the dataset metadata provides a valid `json_file` and the JSON contains an `annotations` field (see §10 “Evaluation / metrics”).



If you see `ModuleNotFoundError: pycocotools`, install it:
```bash
python -m pip install pycocotools
```

---

## 2) Install / expose `third_party` code

You need both `CutLER` and `VNext` on `PYTHONPATH`. From repo root:
```bash
export PYTHONPATH="$PWD/third_party/VNext:$PWD/third_party/CutLER:$PWD:$PYTHONPATH"
```

> For DINOv2 mode, `torch.hub.load(..., source="local")` loads from `third_party/dinov2` directly.  
> You usually do **not** need to add it to `PYTHONPATH`, but it is harmless to do so:
```bash
export PYTHONPATH="$PWD/third_party/dinov2:$PYTHONPATH"
```

If your `VNext` fork requires building detectron2 CUDA ops, follow `third_party/VNext/INSTALL.md`.

---

## 3) Dataset layout (LV-VIS)

Expected layout:
```text
data/LV-VIS/
  train/JPEGImages/<video_name>/*.jpg
  val/JPEGImages/<video_name>/*.jpg
  annotations/
    lvvis_train.json
    lvvis_val.json
```

---

## 4) Step A — Generate pseudo masks (PNG tubes) with VideoCutLER

This repo includes a 4-GPU launcher script:
```bash
bash scripts/run_videocutler_4gpu_png_only.sh train
```

Outputs:
- masks: `outputs/videocutler_lvvis_png/train/masks/<video>/mask_*.png`
- logs:  `outputs/videocutler_lvvis_png/train/logs/gpu*.log`

Environment variables:
- `WSOVVIS_ROOT` (default: `$HOME/code/wsovvis`)
- `CONF` confidence threshold (default 0.6)

Example:
```bash
WSOVVIS_ROOT=$PWD CONF=0.6 bash scripts/run_videocutler_4gpu_png_only.sh train
```

> Some videos may produce **0 instances** (no mask PNGs). This is normal and will be handled in Step B.

---

## 5) Step B — Convert mask PNGs → standard YTVIS JSON

Convert pseudo masks to a tube JSON (videos + per-instance segmentations/bboxes/areas):
```bash
python tools/convert_videocutler_png_to_json.py \
  --mask_root outputs/videocutler_lvvis_png/train/masks \
  --img_root data/LV-VIS/train/JPEGImages \
  --out_json outputs/videocutler_lvvis_png/train/pseudo_tube_ytvis.json \
  --split_name train
```

### Skipping bad/empty videos (recommended)
Options:
- `--skip_list <txt>`: one `video_id` per line
- `--min_masks N` (default 3): skip if mask PNG count < N
- `--min_coverage X` (default 0.2): skip if (#mask_png / #frames) < X
- `--skip_report <json>`: write skip report JSON (default: `.skipped.json`)

Example (more permissive):
```bash
python tools/convert_videocutler_png_to_json.py \
  --mask_root outputs/videocutler_lvvis_png/train/masks \
  --img_root data/LV-VIS/train/JPEGImages \
  --out_json outputs/videocutler_lvvis_png/train/pseudo_tube_ytvis.json \
  --split_name train \
  --min_masks 1 --min_coverage 0.0
```

---

## 6) Step C — Create class-agnostic LV-VIS GT JSON (for evaluation)

Convert LV-VIS GT to a single category (id=1, name=`object`):
```bash
python tools/make_class_agnostic_ytvis_json.py \
  --in_json data/LV-VIS/annotations/lvvis_val.json \
  --out_json data/LV-VIS/annotations/lvvis_val_agnostic.json \
  --category_id 1 --category_name object
```

---

## 7) Step D — Train SeqFormer on pseudo tubes (DDP + Sacred)

### 7.1 Configure `configs/seqformer_pseudo_sacred.yaml`
Key fields you must set:
- `d2_cfg_path`: a config YAML under `third_party/VNext/projects/SeqFormer/configs/`
  - baseline: `.../base_ytvis.yaml`
  - (New) DINOv2 mode: point to your DINO overlay YAML (see below)
- `data.*` paths:
  - `train_json`: pseudo tube JSON from Step B
  - `train_img_root`: LV-VIS train frames root
  - `val_json`: class-agnostic LV-VIS val JSON from Step C
  - `val_img_root`: LV-VIS val frames root
- Dataset names (important for VNext SeqFormer):
  - set `train_name` and `val_name` to start with `ytvis_` (e.g. `ytvis_wsovvis_pseudo_train`)
  - this makes VNext’s SeqFormer select the YTVIS mapper.

### 7.2 Weights (baseline R50 configs)
Some SeqFormer configs refer to a local weight path (e.g. `weights/d2_seqformer_pretrain_r50.pth`) that may not exist.

Two reproducible options:
- **Option A:** override to Detectron2’s ImageNet R50 backbone (auto-download) via `d2_opts`:
  ```yaml
  d2_opts:
    - MODEL.WEIGHTS
    - "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  ```
- **Option B:** provide the weight file on disk at the path referenced by your config.

### 7.3 (New) DINOv2 frozen-backbone mode
This mode **replaces the backbone** with a **frozen DINOv2 ViT** and keeps the rest of SeqFormer unchanged.

#### 7.3.1 Create a DINOv2 overlay YAML (inherits `base_ytvis.yaml`)
Create a new file, e.g.
`third_party/VNext/projects/SeqFormer/configs/dinov2_fix_base_ytvis.yaml`:

```yaml
_BASE_: "base_ytvis.yaml"

MODEL:
  BACKBONE:
    NAME: "DINOv2PseudoFPN"

  RESNETS:
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]

  DINOV2:
    REPO_PATH: "third_party/dinov2"          # local dinov2 repo root
    WEIGHTS: "/ABS/PATH/TO/dinov2_weights.pth"
    MODEL_NAME: "dinov2_vitl14"              # vits14/vitb14/vitl14/vitg14
    OUT_CHANNELS: 256
    OUT_FEATURES: ["res2", "res3", "res4", "res5"]
    FREEZE: True

  # IMPORTANT: do not load R50 weights
  WEIGHTS: ""

SOLVER:
  BACKBONE_MULTIPLIER: 0.0
```

**DINOv2 dim mapping (for debugging):**
- `dinov2_vits14` → 384  
- `dinov2_vitb14` → 768  
- `dinov2_vitl14` → 1024  
- `dinov2_vitg14` → 1536  

> The wrapper can be implemented to auto-infer the embed dim; if you see a Conv channel mismatch, your `MODEL_NAME` and `WEIGHTS` likely refer to different variants.

#### 7.3.2 Sacred config changes (recommended)
In `configs/seqformer_pseudo_sacred.yaml`:
- set `d2_cfg_path` to the overlay YAML above
- ensure **1-class** setting matches SeqFormer config:
  ```yaml
  d2_opts:
    - MODEL.SeqFormer.NUM_CLASSES
    - 1
    - MODEL.WEIGHTS
    - ""
  ```

### 7.4 Run training
```bash
python train_seqformer_pseudo.py with configs/seqformer_pseudo_sacred.yaml -F runs/wsovvis_seqformer -c no
```

Notes:
- **Evaluation interval:** set `TEST.EVAL_PERIOD` (in iterations) in Detectron2 config to run eval during training.
- **Checkpoint interval:** set `SOLVER.CHECKPOINT_PERIOD` (in iterations) to control how often checkpoints are saved.

With Sacred, the cleanest way is to edit `d2_opts` in `configs/seqformer_pseudo_sacred.yaml`, e.g.:

```yaml
d2_opts:
  - TEST.EVAL_PERIOD
  - 2000
  - SOLVER.CHECKPOINT_PERIOD
  - 2000
```

(Use `0` to disable periodic evaluation.)
- `-F runs/...` enables Sacred `FileStorageObserver`
- `-c no` disables Sacred stdout capture (recommended for DDP stability)

---

## 8) Eval-only

In `configs/seqformer_pseudo_sacred.yaml` set:
```yaml
eval_only: true
resume: true
```

Then run the same command as training.

---

## 9) Outputs

- Sacred runs: `runs/wsovvis_seqformer/<run_id>/`
- Detectron2 output: under the run folder (see logs for `OUTPUT_DIR`)
- SeqFormer evaluator predictions: `runs/.../d2/inference/results.json`

---

## 10) Common failure modes (quick checks)

### Data / preprocessing


### Evaluation / metrics

#### 1) Evaluator prints `Annotations are not available for evaluation.`

This means **metric computation is disabled**, usually because:

- `MetadataCatalog.get(DATASETS.TEST[0]).json_file` is missing / wrong path; or
- the JSON file exists but does **not** contain the `annotations` field (common when using an “images-only” JSON).

**Quick sanity check (temporary debug snippet)**  
Add this near **`build_evaluator(...)`** in `train_seqformer_pseudo.py`:

```python
from detectron2.data import DatasetCatalog, MetadataCatalog
import os

test_name = d2_cfg.DATASETS.TEST[0]
meta = MetadataCatalog.get(test_name)

print("[DBG] test_name =", test_name)
print("[DBG] meta.json_file =", getattr(meta, "json_file", None))
print("[DBG] json exists =", os.path.exists(getattr(meta, "json_file", "")))

ds = DatasetCatalog.get(test_name)
print("[DBG] dataset len =", len(ds))
print("[DBG] record[0] keys =", list(ds[0].keys()))
print("[DBG] has annotations key =", "annotations" in ds[0])
if "annotations" in ds[0]:
    print("[DBG] len(record[0]['annotations']) =", len(ds[0]["annotations"]))
```

Expected output should show `json exists = True` and `has annotations key = True`.

#### 2) `NameError: name 'ytvis_results' is not defined`

You are likely on a partially-modified SeqFormer evaluator.  
Apply the patch `patches/vnext_ytvis_eval_enable_eval.patch` (included in this reply) to restore a consistent evaluator implementation.

#### 3) NumPy error from `ytvoseval.py` such as `AttributeError: module 'numpy' has no attribute 'float'`

Either:
- keep `numpy<1.24` (recommended); or
- apply the patch above (it adds a backward-compatible NumPy shim before calling YTVIS evaluation).

#### 4) Warnings like `destroy_process_group() was not called` or Sacred `tee_stdout.wait timeout`

These are **usually secondary effects** after an exception in one worker (DDP teardown). Fix the root error first; the warnings typically disappear.

1) **No `mask_*.png` in some videos**: OK — Step B will skip them (see skip report).
2) **JSON conversion errors**: check `pycocotools` install; validate PNG id format.

### SeqFormer / configs
3) **SeqFormer config path not found**: confirm `d2_cfg_path` exists under `third_party/VNext/projects/SeqFormer/configs/`.
4) **Weights path not found**: either set `MODEL.WEIGHTS: ""` (DINOv2 mode) or use `detectron2://...` (baseline).
5) **Dataset mapper crashes**: ensure dataset names start with `ytvis_`.

### DINOv2 mode specific
6) **`KeyError: Non-existent config key: MODEL.DINOV2`**: ensure `train_seqformer_pseudo.py` extends cfg (`add_dinov2_config(cfg)`) *before* `merge_from_file`.
7) **`ModuleNotFoundError: No module named 'dinov2'`**: your `REPO_PATH` must point to the dinov2 repo root containing both `hubconf.py` and `dinov2/`.
8) **`AssertionError: Input image height ... is not a multiple of patch height 14`**: ensure padding-to-14 logic exists (both tensors and masks).
9) **Conv channel mismatch (768 vs 1024 etc.)**: `MODEL_NAME` and `WEIGHTS` variant mismatch (vitb vs vitl).
10) **Mask loss size mismatch (`Target size ... must be the same as input size ...`)**: ensure the `loss_masks` resize patch exists.

---

## Citation / Acknowledgements
- VideoCutLER / CutLER  
- SeqFormer / VNext Detectron2 fork  
- DINOv2 (optional backbone)
