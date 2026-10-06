#!/bin/bash
# Install the Sapiens2 environment (Python 3.12, torch >= 2.7 cu128) and download the 0.4B pose and body-part checkpoints.
# Requires: git clone --depth 1 https://github.com/facebookresearch/sapiens2   (folder below)
set -x
SAPIENS2_REPO=${SAPIENS2_REPO:-/c/dev/sapiens2}; VENV=${SAPIENS2_VENV:-/c/dev/sapiens2-venv}; PY=$VENV/Scripts/python.exe
uv venv $VENV --python 3.12
uv pip install --python $PY torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY -e "$SAPIENS2_REPO"
$PY - <<'P'
import os
from huggingface_hub import hf_hub_download
root = os.environ.get("SAPIENS_CHECKPOINT_ROOT", "C:/dev/sapiens2_host")
for rid, fn, d in [("facebook/sapiens2-pose-0.4b", "sapiens2_0.4b_pose.safetensors", "pose"), ("facebook/sapiens2-seg-0.4b", "sapiens2_0.4b_seg.safetensors", "seg")]:
    hf_hub_download(rid, fn, local_dir=f"{root}/{d}")
print("DOWNLOADS DONE")
P
