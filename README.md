# Installation & End-to-End Pipeline (from scratch)

This README is **purely installation + pipeline instructions**. It consolidates all setup/compat issues encountered in this project (CUDA/toolchain mismatches, Detectron2 project modules missing in VNext, Pillow incompatibilities, etc.) and provides a reproducible path to:

1) Build the environment (CUDA 11.8 stack)  
2) Install third-party deps (VNext, CutLER/VideoCutLER)  
3) Patch VNext so VideoCutLER can run (DeepLab / PointRend modules + Pillow fix)  
4) Generate LV-VIS tube masks (PNG only) with **4 GPUs**  
5) Convert mask PNGs to **tube JSON**  
6) (Optional) Run WS-OVVIS training with multi-GPU (DDP) + Sacred FileStorageObserver

---

## 0. System prerequisites

Tested assumptions (match your current setup):

- Linux x86_64 (Ubuntu 20.04/22.04)
- NVIDIA driver supports **CUDA 11.8** runtime
- You have / can install **CUDA Toolkit 11.8** (`nvcc` from 11.8 is required to compile CUDA ops)
- GCC/G++ >= 9 recommended
- `conda` (Miniconda/Anaconda)

Install system packages (Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y \
  git wget curl unzip \
  build-essential cmake pkg-config \
  ninja-build \
  ffmpeg \
  libgl1 libglib2.0-0 \
  libgeos-dev
```

Why `libgeos-dev`: Shapely can error with:
`OSError: Could not find library geos_c ...`

---

## 1. Create a clean conda environment

```bash
conda create -n wsovvis python=3.10 -y
conda activate wsovvis

python -m pip install -U pip setuptools wheel
```

---

## 2. Install PyTorch (CUDA 11.8)

**Install PyTorch first** (this avoids build-isolation errors like "No module named torch" when installing editable packages).

Example (official PyTorch index for cu118):

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118
```

Sanity check:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda:", torch.version.cuda)
PY
```

> IMPORTANT: CUDA version mismatch can break Detectron2 builds.  
> Error you saw previously:
> `The detected CUDA version (12.2) mismatches the version that was used to compile PyTorch (11.8).`
>
> Fix: ensure `nvcc` points to CUDA **11.8** when compiling ops:
>
> ```bash
> which nvcc
> nvcc --version
> ```
>
> If you have multiple CUDA toolkits, force CUDA 11.8:
> ```bash
> export CUDA_HOME=/usr/local/cuda-11.8
> export PATH=$CUDA_HOME/bin:$PATH
> export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
> ```

---

## 3. Install core Python requirements (order matters)

Install NumPy early (prevents `pycocotools` build failing with `No module named numpy`):

```bash
python -m pip install "numpy<2.5" cython
```

Then install remaining deps:

```bash
python -m pip install -r requirements.txt
```

If you still see `geos_c` issues (Shapely), either:
- keep `libgeos-dev` installed (apt), or
- prefer conda-forge:
  ```bash
  conda install -c conda-forge geos shapely -y
  ```

---

## 4. Clone third-party repos

From your project root `~/code/wsovvis`:

```bash
mkdir -p third_party
```

### 4.1 CutLER / VideoCutLER

```bash
git clone --recursive https://github.com/facebookresearch/CutLER.git third_party/CutLER
python -m pip install -r third_party/CutLER/videocutler/requirements.txt
```

### 4.2 VNext

```bash
git clone https://github.com/FoundationVision/VNext.git third_party/VNext
python -m pip install -r third_party/VNext/requirements.txt

# IMPORTANT: avoid build isolation so torch is visible during editable build
python -m pip install -e third_party/VNext --no-build-isolation
```

---

## 5. Fix VNext ↔ VideoCutLER conflicts (one-command patch)

### 5.1 What conflicts exist (and what this patch fixes)

In your environment, `detectron2` is imported from:

```
third_party/VNext/detectron2
```

However, VideoCutLER / Mask2Former expects Detectron2 "projects" modules that are **not shipped** in VNext's Detectron2 fork by default:

- `detectron2.projects.deeplab`
- `detectron2.projects.point_rend`
  - plus `detectron2.projects.point_rend.point_features`

Also, VNext Detectron2 may contain Pillow-incompatible usage:

- `Image.LINEAR` does not exist in Pillow>=10 (you have Pillow 12), causing:
  `AttributeError: module 'PIL.Image' has no attribute 'LINEAR'`

**This project provides an offline patch tool** that:
- injects minimal DeepLab + PointRend files into `third_party/VNext/projects/`
- updates `third_party/VNext/detectron2/projects/__init__.py` so the projects loader can find them
- patches `Image.LINEAR` → `Image.BILINEAR` for Pillow>=10

### 5.2 Apply the patch

```bash
cd ~/code/wsovvis
chmod +x tools/patch_vnext_for_videocutler
./tools/patch_vnext_for_videocutler --wsovvis-root ~/code/wsovvis
```

Verify:

```bash
python - <<'PY'
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.projects.point_rend.point_features import point_sample
print("OK imports")
PY
```

---

## 6. Build CUDA ops for VideoCutLER (Mask2Former ms_deform_attn)

Required for transformer-based VIS models.

```bash
cd ~/code/wsovvis/third_party/CutLER/videocutler/mask2former/modeling/pixel_decoder/ops
rm -rf build/
sh make.sh
```

You should see a `.so` built/installed, e.g.
`MultiScaleDeformableAttention.cpython-310-x86_64-linux-gnu.so`

Warnings about deprecated APIs are OK.

---

## 7. Run VideoCutLER demo (single video smoke test)

LV-VIS layout (**do not merge train/val**):

```
data/LV-VIS/
  train/JPEGImages/<video_id>/*.jpg
  val/JPEGImages/<video_id>/*.jpg
```

Example:

```bash
cd ~/code/wsovvis/third_party/CutLER/videocutler

VID=00000
python demo_video/demo_masks_only.py \
  --config-file configs/imagenet_video/video_mask2former_R50_cls_agnostic.yaml \
  --input ~/code/wsovvis/data/LV-VIS/train/JPEGImages/${VID}/*.jpg \
  --output ~/code/wsovvis/outputs/videocutler_smoke/${VID} \
  --save-masks True \
  --confidence-threshold 0.6 \
  --opts MODEL.WEIGHTS pretrain/videocutler_m2f_rn50.pth
```

The demo may also output per-frame `.jpg` visualizations.  
Next step runs in parallel and **deletes jpg/mp4**, keeping only mask PNGs.

---

## 8. 4-GPU parallel tube generation (PNG-only)

Run:

```bash
cd ~/code/wsovvis
bash scripts/run_videocutler_4gpu_png_only.sh train
# or:
bash scripts/run_videocutler_4gpu_png_only.sh val
```

Outputs:

```
outputs/videocutler_lvvis_png/<split>/masks/<video_id>/mask_*.png
outputs/videocutler_lvvis_png/<split>/logs/gpu*.log
```

---

## 9. Convert mask PNGs → tube JSON

Train:

```bash
python tools/convert_videocutler_png_to_json.py \
  --mask_root outputs/videocutler_lvvis_png/train/masks \
  --out_json outputs/videocutler_lvvis_png/train/videocutler_tubes_train.json \
  --split_name train
```

Val:

```bash
python tools/convert_videocutler_png_to_json.py \
  --mask_root outputs/videocutler_lvvis_png/val/masks \
  --out_json outputs/videocutler_lvvis_png/val/videocutler_tubes_val.json \
  --split_name val
```

Notes:
- The converter treats each distinct integer ID in `mask_*.png` as one track-id.
- If IDs are not consistent across frames, tracks will be fragmented (you can add an IoU linker later).

---

## 10. (Optional) WS-OVVIS training (Detectron2 DDP + Sacred)

If your entrypoint is `train_wsovvis.py` (Detectron2 trainer + Sacred), run DDP via torchrun:

```bash
cd ~/code/wsovvis
torchrun --nproc_per_node=4 train_wsovvis.py with configs/wsovvis_sacred.yaml
```

### Sacred FileStorageObserver
Use local file storage:

```
runs/
  <run_id>/
    config.json
    metrics.json
    ...
```

In code:

```python
from sacred.observers import FileStorageObserver
ex.observers.append(FileStorageObserver("runs"))
```

---

## 11. Recommended directory layout

```
wsovvis/
  data/
    LV-VIS/
      train/JPEGImages/<video_id>/*.jpg
      val/JPEGImages/<video_id>/*.jpg
  third_party/
    VNext/
    CutLER/
  outputs/
    videocutler_lvvis_png/
      train/masks/<video_id>/mask_*.png
      train/logs/gpu*.log
      val/...
  runs/                 # Sacred FileStorageObserver
  tools/
    patch_vnext_for_videocutler   # executable (offline patch)
    convert_videocutler_png_to_json.py
  scripts/
    run_videocutler_4gpu_png_only.sh
```

---

## 12. Troubleshooting (consolidated)

### A) `OSError: Could not find library geos_c`
Install GEOS:
```bash
sudo apt-get install -y libgeos-dev
# or:
conda install -c conda-forge geos shapely -y
```

### B) `ModuleNotFoundError: No module named numpy` when building pycocotools
Install numpy first:
```bash
python -m pip install "numpy<2.5"
```

### C) `ModuleNotFoundError: No module named torch` during `pip install -e ...`
Install torch first, and use:
```bash
python -m pip install -e third_party/VNext --no-build-isolation
```

### D) `AttributeError: PIL.Image has no attribute LINEAR`
Run the patch tool:
```bash
./tools/patch_vnext_for_videocutler --wsovvis-root ~/code/wsovvis
```

### E) Detectron2 build fails with CUDA mismatch (12.x vs 11.8)
Ensure `nvcc --version` is 11.8 when building ops, set `CUDA_HOME` to 11.8.

---

# End
