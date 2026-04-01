
# Installation

## Requirements
- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.8 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. 
  Note, please check PyTorch version matches that is required by Detectron2.
- Detectron2: follow Detectron2 installation instructions.
- OpenCV ≥ 4.6 is needed by demo and visualization.

## Example conda environment setup

```bash
conda create --name cutler python=3.8 -y
conda activate cutler
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
#conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.9"

# under your working directory
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
python -m pip install -U pip wheel ninja
python -m pip install -e . --no-build-isolation
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd ..
git clone --recursive git@github.com:facebookresearch/CutLER.git
cd CutLER
pip install -r requirements.txt
```

## datasets
If you want to train/evaluate on the datasets, please see [datasets/README.md](datasets/README.md) to see how we prepare datasets for this project.