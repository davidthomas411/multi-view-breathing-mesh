# Setup

The code was developed on Windows 11 (RTX 4060, 8 GB; 6-core CPU, 32 GB RAM) with three Python environments. Paths are Windows-style defaults and can be overridden with environment variables; the scripts are plain Python modules in `mv/` that import each other, so run them from that folder.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `SGRT_ROOT` | parent folder of `mv/` | project root: expects `CUTrial/` (data) and writes `tmp/`, `reports/`, `recordings/` |
| `SAM3D_REPO` | `C:/dev/Fast-SAM-3D-Body` | clone of Fast-SAM-3D-Body (SAM 3D Body + MHR) with its `checkpoints/` |
| `INSTANTHMR_REPO` | `C:/dev/InstantHMR` | clone of InstantHMR with `models/instanthmr.onnx` (phase 1 only) |
| `SAPIENS2_REPO` | `C:/dev/sapiens2` | clone of Sapiens2 |
| `SAPIENS_CHECKPOINT_ROOT` | `C:/dev/sapiens2_host` | Sapiens2 checkpoints (`pose/`, `seg/`) |
| `TRIAL` | `TDB` | which breathing trial a script works on (`TDB` or `ADB`); outputs go to `tmp/<name>` or `tmp/<name>_ADB` |

## The three environments

`requirements/*.freeze.txt` are the exact package lists. The PyTorch wheels need their CUDA index (`--index-url https://download.pytorch.org/whl/cu124` or `cu128`).

```bash
# 1. "InstantHMR env": geometry, tracking, analysis, reports, Rerun.  Python 3.12
uv venv --python 3.12 .venv-main
uv pip install --python .venv-main/Scripts/python.exe --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
uv pip install --python .venv-main/Scripts/python.exe numpy scipy opencv-python matplotlib pymupdf ezc3d "rerun-sdk==0.38.*" "onnxruntime-gpu==1.22.0" rfdetr huggingface_hub

# 2. "SAM3D env": SAM 3D Body / MHR.  Python 3.11, torch 2.5.1 cu124
git clone https://github.com/yangtiming/Fast-SAM-3D-Body   # see scripts/install_sam3d.sh for the package list and checkpoint download
bash scripts/install_sam3d.sh

# 3. "Sapiens2 env": Python 3.12, torch >= 2.7
uv venv --python 3.12 .venv-sapiens2
git clone --depth 1 https://github.com/facebookresearch/sapiens2
bash scripts/install_sapiens2.sh
```

Checkpoints: `facebook/sam-3d-body-dinov3` (Hugging Face), YOLO11 pose weights (ultralytics), `facebook/sapiens2-pose-0.4b` and `facebook/sapiens2-seg-0.4b` (not gated; the Sapiens2 License applies).

GPU memory is the constraint on an 8 GB card: one SAM 3D Body job (~4 GB) at a time, and close viewers (Rerun, browsers) while it runs, or Windows pages video memory and everything slows down by an order of magnitude.

## Expected data layout (not included)

```
CUTrial/
  MM17/
    TDB/Gyroflow/MMC17_TDB_T1_stabilized.mp4 ... T6      Gyroflow-stabilised 4K videos
    ADB/MMC17_ADB_T1.mov ... T6                          raw GoPro video (lens-corrected on the fly by adb_geometry.py)
    DBECal/MMC17_DBECal_T1.mov ...                       checkerboard clip, raw
    DBECal/post_processing/Gyroflow/*_stabilized.mp4     checkerboard clip, stabilised
  Trials_marker_positions_by_image_frame_all_cases/
    MMC17_TDB_marker_positions_by_image_frame_wide.csv   Vicon markers per image frame (mm): marker names, X/Y/Z columns
```

Vicon `image_frame_number` = video frame index + 1.

## Calibration chain (MMC17)

1. `calib_find_board.py`, `calib_extrinsics.py`: detect the 6 x 5 inner-corner sub-grid of the checkerboard in the (stabilised) calibration clip; shared intrinsics + one board pose per camera (world = board, units = squares).
2. `calib_offsets.py`, `calib_ring.py`, `calib_final.py`: choose the sub-grid alignment (offset / flip) per camera. The image-based offset and the ring-layout prior are *not* enough: the reprojection error cannot see a wrong alignment.
3. `vicon_calib.py`, `vicon_calib2.py`, `calib_v3.py`: detect the reflective balls (`marker_detect.py`), match them to the Vicon markers with a Vicon-to-board transform search, settle the alignment of every camera from the marker matches, then per-camera PnP in the Vicon frame -> `tmp/mmc17_vicon_calib.json` (metres). `calib_v3.py` writes `tmp/mmc17_calib_v3.json`, which was copied to `tmp/mmc17_calib.json` (the file the SAM 3D Body scripts read).
4. `vicon_validate.py`: in-sample reprojection and triangulation of the balls against Vicon (including leave-one-camera-out).
5. `t6_search.py`, `t6_finalize.py`: recover camera T6 (it moved after the checkerboard clip); it stays excluded because it is not on the common clock.
6. Raw trials (ADB): `adb_geometry.py fit | align`, `adb_extrinsics.py`, check with `adb_check_markers.py`.

## Rerun recordings

`make_tracking_rrd.py`, `make_tracking_all_rrd.py`, `make_skin_rrd.py`, `make_skin_dense_rrd.py`, `make_sapiens_rrd.py` write Rerun 0.38 files (open with the `rerun` executable of the main environment). They embed camera frames and are not distributed.
