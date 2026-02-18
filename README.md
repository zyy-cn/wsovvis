# WS-OVVIS (LV-VIS) — VideoCutLER pseudo tubes → SeqFormer training

This repo provides a **reproducible minimal pipeline** to:

1. Run **VideoCutLER** to generate per-video `mask_*.png` pseudo masks
2. Convert masks into **YTVIS/LV-VIS-style JSON** tubes
3. Train **SeqFormer (VNext/Detectron2)** on pseudo tubes using **DDP + Sacred**
4. Evaluate on **LV-VIS ground-truth masks** with a **class-agnostic** JSON

> This stage is **class-agnostic (1 class)** by design: pseudo tubes do not contain reliable category labels yet.
> Once you add tube→category attribution, you can switch to standard LV-VIS multi-class evaluation.

---

## 0) Repository + third_party status (important)

This pipeline depends on two third-party components:

- `third_party/CutLER` (VideoCutLER)
- `third_party/VNext` (Detectron2 fork + SeqFormer project)

Make sure you clone with submodules (or otherwise place the correct code under `third_party/`):

```bash
git clone --recursive <YOUR_REPO_URL> wsovvis
cd wsovvis
git submodule update --init --recursive
```

### Required patches already tracked in GitHub
To reproduce, you **must** use a repo revision where the following third_party file contains the safety fix for “0 instances” videos (otherwise VideoCutLER may crash and stop a GPU worker):

- `third_party/CutLER/videocutler/demo_video/demo_masks_only.py`

You said you already backed up the file; for reproduction, simply ensure your checkout matches the GitHub version used by this README.

---

## 1) Environment

### 1.1 Create conda env

```bash
conda create -n wsovvis python=3.10 -y
conda activate wsovvis
python -m pip install -U pip wheel
# Sacred currently imports pkg_resources; pin setuptools to keep pkg_resources available
python -m pip install -U "setuptools<81"
python -c "import pkg_resources; print('pkg_resources OK')"
```

### 1.2 Install PyTorch (example: CUDA 11.8)

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu118   torch torchvision torchaudio
```

> Use a PyTorch build that matches your CUDA runtime/driver.

### 1.3 Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 2) Install / expose third_party code

You need both `CutLER` and `VNext` on `PYTHONPATH`.

From repo root:

```bash
export PYTHONPATH="$PWD/third_party/VNext:$PWD/third_party/CutLER:$PWD:$PYTHONPATH"
```

If your `VNext` fork requires building detectron2 CUDA ops, follow `third_party/VNext/INSTALL.md`.

### Optional: patch VNext so VideoCutLER runs
Some VNext layouts miss Detectron2 project stubs (e.g., DeepLab / PointRend) that VideoCutLER expects.
If you see import errors like `detectron2.projects.deeplab` / `point_rend`, run:

```bash
python tools/patch_vnext_for_videocutler.py --vnext_root third_party/VNext
```

---

## 3) Dataset layout (LV-VIS)

Expected layout:

```
data/LV-VIS/
  train/JPEGImages/<video_id>/*.jpg
  val/JPEGImages/<video_id>/*.jpg
  annotations/
    lvvis_val.json
```

---

## 4) Step A — Generate pseudo masks (PNG tubes) with VideoCutLER

This repo includes a 4-GPU launcher script:

```bash
bash scripts/run_videocutler_4gpu_png_only.sh train
```

Outputs:

- masks: `outputs/videocutler_lvvis_png/train/masks/<video_id>/mask_*.png`
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

Convert pseudo masks to a **tube JSON** (videos + per-instance segmentations/bboxes/areas):

```bash
python tools/convert_videocutler_png_to_json.py   --mask_root outputs/videocutler_lvvis_png/train/masks   --img_root  data/LV-VIS/train/JPEGImages   --out_json  outputs/videocutler_lvvis_png/train/pseudo_tube_ytvis.json   --split_name train
```

### Skipping bad/empty videos (recommended)
Use these options to skip problematic cases:

- `--skip_list <txt>`: one `video_id` per line
- `--min_masks N` (default 3): skip if mask PNG count < N
- `--min_coverage X` (default 0.2): skip if (#mask_png / #frames) < X
- `--skip_report <path>`: write skip report JSON (default: `<out_json>.skipped.json`)

Example (more permissive):

```bash
python tools/convert_videocutler_png_to_json.py   --mask_root outputs/videocutler_lvvis_png/train/masks   --img_root  data/LV-VIS/train/JPEGImages   --out_json  outputs/videocutler_lvvis_png/train/pseudo_tube_ytvis.json   --split_name train   --min_masks 1 --min_coverage 0.0
```

---

## 6) Step C — Create class-agnostic LV-VIS GT JSON (for evaluation)

Convert LV-VIS GT to a single category (id=1, name=`object`):

```bash
python tools/make_class_agnostic_ytvis_json.py   --in_json  data/LV-VIS/annotations/lvvis_val.json   --out_json data/LV-VIS/annotations/lvvis_val_agnostic.json   --category_id 1 --category_name object
```

---

## 7) Step D — Train SeqFormer on pseudo tubes (DDP + Sacred)

### 7.1 Configure `configs/seqformer_pseudo_sacred.yaml`

Key fields you must set:

- `d2_cfg_path`: use a config that exists in your VNext checkout, e.g.:

  - `third_party/VNext/projects/SeqFormer/configs/base_ytvis.yaml`
  - or `third_party/VNext/projects/SeqFormer/configs/large_model/swin_ytvis.yaml`

- `data.*` paths:
  - `train_json`: pseudo tube JSON from Step B
  - `train_img_root`: LV-VIS train frames root
  - `val_json`: class-agnostic LV-VIS val JSON from Step C
  - `val_img_root`: LV-VIS val frames root

- **Dataset names** (important for VNext SeqFormer):
  - Set `train_name` and `val_name` to start with `ytvis_` (e.g., `ytvis_wsovvis_pseudo_train`)
  - This makes VNext’s SeqFormer `train_net.py` select the YTVIS mapper.

### 7.2 Weights (important)
Some SeqFormer configs refer to a local weight path (e.g. `weights/d2_seqformer_pretrain_r50.pth`) that may not exist.
Two reproducible options:

**Option A (recommended): override to Detectron2’s ImageNet R50 backbone (auto-download)**

Add to `d2_opts`:

```yaml
d2_opts:
  - MODEL.WEIGHTS
  - "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
```

**Option B: provide the weight file on disk**

Put the expected checkpoint at the path in your config (e.g. `weights/d2_seqformer_pretrain_r50.pth`).

### 7.3 Run training

```bash
python train_seqformer_pseudo.py with configs/seqformer_pseudo_sacred.yaml -F runs/wsovvis_seqformer -c no
```

Notes:
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

---

## 10) Common failure modes (quick checks)

1) **No `mask_*.png` in some videos**: OK — Step B will skip them (see skip report).
2) **SeqFormer config path not found**: confirm `d2_cfg_path` points to an existing YAML under `third_party/VNext/projects/SeqFormer/configs/`.
3) **Weights path not found**: use Option A (`detectron2://...`) or put the file at the expected path.
4) **Dataset mapper crashes**: ensure dataset names start with `ytvis_` and your loader outputs per-frame `annotations` with `bbox_mode`.

