"""Sapiens2 (Meta) pose (308 keypoints, top-down) and 29-class body-part segmentation on MMC17 frames, one person crop per camera.
Run with the Sapiens2 env:   C:\\dev\\sapiens2-venv\\Scripts\\python.exe sapiens2_run.py --frames 250 --cams 1 2 3 4 5 [--trial TDB] [--size 0.4b] [--rot auto|0] [--device cuda|cpu]
Person boxes: the YOLO boxes saved by mv_collect.py (tmp/mv_sam3d[_TRIAL]/<frame>.npz).  The supine person is a very elongated box, so by default the crop is rotated by 90 degrees
(head up) before it goes into the 1024x768 (HxW) network and the results are rotated back.
Output:  tmp/sapiens2[_TRIAL]/<frame>_T<cam>_<size>_<rot>.npz  (kpts (308,2) px in the 4K stabilised-video frame, scores (308,), seg labels at half resolution (1080x1920 uint8), crop box)
         tmp/sapiens2[_TRIAL]/vis/<frame>_T<cam>_<size>_<rot>.jpg   (keypoints + segmentation overlay)"""
import argparse, json, os, sys, time
os.environ.setdefault("SAPIENS_CHECKPOINT_ROOT", "C:/dev/sapiens2_host")
import cv2, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trial_io as TI
from sapiens.pose.datasets import parse_pose_metainfo, UDPHeatmap
from sapiens.pose.models import init_model as init_pose
from sapiens.dense.models import init_model as init_seg
SP = os.environ.get("SAPIENS2_REPO", "C:/dev/sapiens2") + "/sapiens"; CK = os.environ["SAPIENS_CHECKPOINT_ROOT"]
PAL = np.random.default_rng(3).integers(40, 255, size=(29, 3)).astype(np.uint8); PAL[0] = 0


def load_models(size, device, do_seg=True):
    pose = init_pose(f"{SP}/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_{size}_keypoints308_shutterstock_goliath_3po-1024x768.py", f"{CK}/pose/sapiens2_{size}_pose.safetensors", device=device)
    pose.pose_metainfo = parse_pose_metainfo(dict(from_file=f"{SP}/pose/configs/_base_/keypoints308.py")); cod = dict(pose.cfg.codec); cod.pop("type"); pose.codec = UDPHeatmap(**cod)
    seg = init_seg(f"{SP}/dense/configs/seg/shutterstock_goliath/sapiens2_{size}_seg_shutterstock_goliath-1024x768.py", f"{CK}/seg/sapiens2_{size}_seg.safetensors", device=device) if do_seg else None
    return pose, seg


def orientation(z, v, K):
    """head side in the image from SAM 3D Body's 3D skeleton (nose vs mean ankle): 'cw' if the head is on the left (rotate clockwise to put it on top), else 'ccw'."""
    X = z["j3d"][v] + z["cam_t"][v]; uv = (K @ X.T).T; uv = uv[:, :2] / uv[:, 2:3]; return "cw" if uv[0, 0] < 0.5 * (uv[13, 0] + uv[14, 0]) else "ccw"


def crop_person(img, box, orient, margin=0.10):
    H, W = img.shape[:2]; x1, y1, x2, y2 = [float(v) for v in box]; m = margin * max(x2 - x1, y2 - y1)
    X1, Y1, X2, Y2 = int(max(x1 - m, 0)), int(max(y1 - m, 0)), int(min(x2 + m, W)), int(min(y2 + m, H)); crop = img[Y1:Y2, X1:X2]
    if orient == "cw": crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    elif orient == "ccw": crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return np.ascontiguousarray(crop), (X1, Y1, X2, Y2)


def to_original(pts, orient, crop_box):
    """points in the rotated crop -> pixel coordinates in the full frame."""
    X1, Y1, X2, Y2 = crop_box; hc, wc = Y2 - Y1, X2 - X1; u, v = pts[:, 0], pts[:, 1]
    if orient == "cw": x, y = v, hc - u                 # clockwise: (x, y) -> (hc - y, x)
    elif orient == "ccw": x, y = wc - v, u              # counter-clockwise: (x, y) -> (y, wc - x)
    else: x, y = u, v
    return np.stack([x + X1, y + Y1], 1)


@torch.no_grad()
def run_pose(model, crop_bgr, device):
    h, w = crop_bgr.shape[:2]; data_info = dict(img=crop_bgr, bbox=np.array([[0, 0, w - 1, h - 1]], np.float32), bbox_score=np.ones(1, np.float32))
    data = model.data_preprocessor(model.pipeline(data_info)); inputs = data["inputs"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device != "cpu")): pred = model(inputs)
    pred = pred.float().cpu().numpy(); kp, sc = model.codec.decode(pred[0]); m = data["data_samples"]["meta"]
    kp = kp / m["input_size"] * m["bbox_scale"] + m["bbox_center"] - 0.5 * m["bbox_scale"]
    return np.asarray(kp[0], np.float64), np.asarray(sc[0], np.float64)


@torch.no_grad()
def run_seg(model, crop_bgr, device):
    data = model.data_preprocessor(model.pipeline(dict(img=crop_bgr))); inputs = data["inputs"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device != "cpu")): logits = model(inputs)
    h2, w2 = max(crop_bgr.shape[0] // 2, 1), max(crop_bgr.shape[1] // 2, 1)                                   # half-resolution labels (what is stored), 4x less memory than the full crop
    logits = F.interpolate(logits.float(), size=(h2, w2), mode="bilinear"); lab = logits.argmax(1)[0].cpu().numpy().astype(np.uint8); del logits; return lab


def label_to_frame(lab, orient, crop_box, shape, scale=0.5):
    """lab = half-resolution labels of the (rotated) crop -> half-resolution label map of the whole frame."""
    if orient == "cw": lab = cv2.rotate(lab, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif orient == "ccw": lab = cv2.rotate(lab, cv2.ROTATE_90_CLOCKWISE)
    X1, Y1, X2, Y2 = crop_box; H2, W2 = int(round(shape[0] * scale)), int(round(shape[1] * scale)); full = np.zeros((H2, W2), np.uint8); x0, y0 = int(round(X1 * scale)), int(round(Y1 * scale)); hh, ww = min(lab.shape[0], H2 - y0), min(lab.shape[1], W2 - x0)
    full[y0:y0 + hh, x0:x0 + ww] = lab[:hh, :ww]; return full


def draw(img, kp, sc, seg_half, thr=0.3):
    v = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA); col = PAL[seg_half]; msk = seg_half > 0; v[msk] = (0.55 * v[msk] + 0.45 * col[msk][:, ::-1]).astype(np.uint8)
    for (x, y), s in zip(kp, sc):
        if s > thr: cv2.circle(v, (int(x * 0.5), int(y * 0.5)), 3, (0, 255, 255) if s > 0.6 else (0, 140, 255), -1)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, nargs="+", default=[250]); ap.add_argument("--cams", type=int, nargs="+", default=[1, 2, 3, 4, 5]); ap.add_argument("--trial", default=TI.TRIAL)
    ap.add_argument("--size", default="0.4b"); ap.add_argument("--rot", default="auto"); ap.add_argument("--device", default="cuda"); ap.add_argument("--no-seg", action="store_true"); a = ap.parse_args()
    TI.TRIAL = a.trial; OUT = TI.tdir("sapiens2", a.trial); os.makedirs(OUT + "/vis", exist_ok=True); MVS = TI.tdir("mv_sam3d", a.trial, make=False)
    K = np.array(json.load(open(TI.calib_path(a.trial)))["K"]); t0 = time.time(); pose, seg = load_models(a.size, a.device, not a.no_seg); print("models loaded in %.1fs" % (time.time() - t0), flush=True)
    caps = {c: TI.Capture(c, a.trial) for c in a.cams}
    for f in a.frames:
        z = np.load(f"{MVS}/{f:04d}.npz")
        for c in a.cams:
            caps[c].set(cv2.CAP_PROP_POS_FRAMES, f); ok, img = caps[c].read()
            if not ok or not z["ok"][c - 1]: print(f"frame {f} T{c}: no image or no person box", flush=True); continue
            orient = orientation(z, c - 1, K) if a.rot == "auto" else "none"; crop, cb = crop_person(img, z["box"][c - 1], orient); t1 = time.time()
            kp_r, sc = run_pose(pose, crop, a.device); kp = to_original(kp_r, orient, cb); t2 = time.time()
            lab = run_seg(seg, crop, a.device) if seg is not None else np.zeros(crop.shape[:2], np.uint8); seg_half = label_to_frame(lab, orient, cb, img.shape); t3 = time.time()
            tag = f"{f:04d}_T{c}_{a.size}_{'rot' if orient != 'none' else 'norot'}"
            np.savez_compressed(f"{OUT}/{tag}.npz", kpts=kp, scores=sc, seg=seg_half, crop_box=np.array(cb), orient=orient)
            cv2.imwrite(f"{OUT}/vis/{tag}.jpg", draw(img, kp, sc, seg_half), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if a.device != "cpu": torch.cuda.empty_cache()
            print(f"frame {f} T{c} ({orient}): pose {t2 - t1:.1f}s, seg {t3 - t2:.1f}s, {int((sc > 0.3).sum())}/308 keypoints > 0.3, mean score of the first 70 {sc[:70].mean():.2f}", flush=True)
