#!/bin/bash
# Install the SAM 3D Body environment (Python 3.11, torch 2.5.1 cu124) used by mv_collect.py, mv_fit.py, marker_fit.py, sap_fit.py ... Adjust SAM3D_REPO / the venv path.
# Requires: git clone https://github.com/yangtiming/Fast-SAM-3D-Body  (the repo folder below)
set -x
SAM3D_REPO=${SAM3D_REPO:-/c/dev/Fast-SAM-3D-Body}
cd "$SAM3D_REPO"
uv venv --python 3.11 /c/dev/sam3d-venv
source /c/dev/sam3d-venv/Scripts/activate
uv pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
uv pip install pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn appdirs cython jsonlines pytest loguru optree fvcore pycocotools huggingface_hub ultralytics smplx scipy tqdm pyzmq onnx
uv pip install git+https://github.com/microsoft/MoGe.git
python - <<'P'
from huggingface_hub import snapshot_download
snapshot_download("facebook/sam-3d-body-dinov3", local_dir="checkpoints/sam-3d-body-dinov3")
import os; os.makedirs("checkpoints/yolo",exist_ok=True)
from ultralytics import YOLO
m=YOLO("yolo11m-pose.pt"); import shutil; shutil.copy("yolo11m-pose.pt","checkpoints/yolo/yolo11m-pose.pt") if os.path.exists("yolo11m-pose.pt") else None
print("DOWNLOADS DONE")
P
