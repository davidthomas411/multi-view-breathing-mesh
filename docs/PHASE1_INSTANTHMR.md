# Phase 1: InstantHMR on a six-camera treatment-room data set (live Rerun viewer)

> Early work on a different data set (not included); kept for the record. Paths are examples.

Environment (outside the project folder): `.venv`  (Python 3.12, torch 2.6+cu124,
onnxruntime-gpu 1.22, rfdetr 1.11, rerun-sdk 0.38). Model: `C:\dev\InstantHMR\models\instanthmr.onnx`.

## Run the live demo (PowerShell)

    cd mv
    $py = ".venv\Scripts\python.exe"

    # 1. one-off: decode the 5.3K videos to 1920x1080 JPEGs (tmp\frames\01..06). ~3 min. Already done.
    & $py cache_frames.py

    # 2. live viewer: opens the Rerun window, plays at 30 fps, loops
    & $py live_viewer.py --loop

    # variants
    & $py live_viewer.py --fast                       # no real-time pacing
    & $py live_viewer.py --mode detect                # RF-DETR on every view every frame (slow, ~170-230 ms)
    & $py live_viewer.py --start 560 --stop 706       # supine segment only
    & $py live_viewer.py --no-spawn --save ..\tmp\run.rrd   # record, then:  rerun ..\tmp\run.rrd

    # numbers without the GUI
    & $py run_seq.py 0 706 1 track                    # accuracy proxies + latency table, whole sequence
    & $py test_frame.py 705                           # one synced frame set -> tmp\overlay_f705.jpg

Close the previous viewer window (and its python process) before starting another.

## Files
- `mvcal.py`       calibration from `cuvault2\vault2\videos\intri.yml / extri.yml`, distortion, triangulation
- `mvhmr.py`       MultiViewHMR: boxes (detector or reprojected previous skeleton) -> one batch-6 ONNX call -> triangulate
- `live_viewer.py` Rerun UI: 6 camera feeds + 2D fits, fused 3D skeleton in the room, latency + quality plots
- `run_seq.py`, `test_frame.py`, `single_vs_multi.py`, `sync_search.py`, `sync_drift.py`  analysis scripts
- `tmp\`           frame cache, scene_meshes.npz (TrueBeam mesh from scene.rrd), recordings, overlays

## Rebuilding the environment from scratch
    uv venv --python 3.12 .venv   (after `git clone https://github.com/mohamdev/InstantHMR C:\dev\InstantHMR`)
    uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
    uv pip install "onnxruntime-gpu==1.22.0" numpy opencv-python pillow rfdetr rerun-sdk huggingface_hub
    python -c "from huggingface_hub import hf_hub_download as h; h('momolesang/InstantHMR','instanthmr.onnx',local_dir='models')"
Gotchas: onnxruntime-gpu >= 1.23 needs CUDA 13 (pin 1.22); import torch before onnxruntime; the old
`*_frame50.jpg` stills are not time-synced.
