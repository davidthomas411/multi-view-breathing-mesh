"""Multi-view SAM 3D Body: per-view SAM3D (calibrated K, YOLO box) -> robust world-frame triangulation of the 70 keypoints.
Run with the SAM3D env:  C:\dev\sam3d-venv\Scripts\python.exe sam3d_mv.py [start stop step]
Caches per-frame per-view results in tmp/sam3d_cache/ so the benchmark scripts can reuse them."""
import os, sys, time
import cv2, numpy as np, torch
sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvcal import load_cameras, CAMS, FRAME_DIR, OUT, triangulate_joints

BODY = list(range(0, 21)) + [41, 62, 63, 64, 65, 66, 67, 68, 69]
CACHE = f"{OUT}/sam3d_cache"
os.makedirs(CACHE, exist_ok=True)

_est = _yolo = None
def models():
    global _est, _yolo
    if _est is None:
        cwd = os.getcwd(); os.chdir(os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        from ultralytics import YOLO
        m, cfg = load_sam_3d_body("checkpoints/sam-3d-body-dinov3/model.ckpt", device="cuda",
                                  mhr_path="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")
        _est = SAM3DBodyEstimator(sam_3d_body_model=m, model_cfg=cfg, human_detector=None, human_segmentor=None, fov_estimator=None)
        _yolo = YOLO("checkpoints/yolo/yolo11m-pose.pt")
        os.chdir(cwd)
    return _est, _yolo

def run_view(img_rgb, K, keep_mesh=False):
    est, yolo = models()
    r = yolo(img_rgb, verbose=False)[0]
    if len(r.boxes) == 0: return None
    b = r.boxes.xyxy.cpu().numpy(); a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    b = b[int(np.argmax(a))][None].astype(np.float32)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        o = est.process_one_image(img_rgb, bboxes=b, cam_int=torch.tensor(K[None], dtype=torch.float32))
    if not o: return None
    o = o[0]
    d = dict(j2d=np.asarray(o["pred_keypoints_2d"], np.float32), j3d=np.asarray(o["pred_keypoints_3d"], np.float32),
             cam_t=np.asarray(o["pred_cam_t"], np.float32), box=b[0])
    if keep_mesh: d["verts"] = np.asarray(o["pred_vertices"], np.float32)
    return d

def reproject(j3d, cam_t, cams):
    """2D observations = SAM3D's 3D skeleton + cam_t projected through the calibrated K (its own pred_keypoints_2d proved less
    reliable here: 9-21% of body height from YOLO-Pose vs 1-9% for this), then moved onto the ideal pinhole model."""
    out = np.full((6, 70, 2), np.nan, np.float32)
    for i, c in enumerate(CAMS):
        if np.isnan(j3d[i]).any(): continue
        X = j3d[i] + cam_t[i]; uv = (cams[c].K @ X.T).T
        out[i] = cams[c].undistort(uv[:, :2] / uv[:, 2:3])
    return out

def frame_views(f, cams, keep_mesh=False, use_cache=True):
    p = f"{CACHE}/{f:04d}.npz"
    if use_cache and os.path.exists(p) and not keep_mesh:
        z = np.load(p, allow_pickle=True); return reproject(z["j3d"], z["cam_t"], cams), z["j3d"], z["cam_t"], z["box"]
    j2d = np.full((6, 70, 2), np.nan, np.float32); j3d = np.full((6, 70, 3), np.nan, np.float32)
    ct = np.full((6, 3), np.nan, np.float32); bx = np.full((6, 4), np.nan, np.float32)
    for i, c in enumerate(CAMS):
        img = cv2.cvtColor(cv2.imread(f"{FRAME_DIR}/{c}/{f:04d}.jpg"), cv2.COLOR_BGR2RGB)
        d = run_view(img, cams[c].K, keep_mesh)
        if d is None: continue
        j3d[i] = d["j3d"]; ct[i] = d["cam_t"]; bx[i] = d["box"]
    np.savez(p, j3d=j3d, cam_t=ct, box=bx)
    return reproject(j3d, ct, cams), j3d, ct, bx

def fuse(j2d, cams_l):
    views = [i for i in range(6) if not np.isnan(j2d[i, BODY]).any()]
    if len(views) < 2: return None
    X, err, used = triangulate_joints([cams_l[i] for i in views], j2d[views][:, BODY])
    loo = []
    if len(views) >= 3:
        for h in views:
            rest = [v for v in views if v != h]
            Xl, _, _ = triangulate_joints([cams_l[v] for v in rest], j2d[rest][:, BODY])
            loo.append(np.linalg.norm(cams_l[h].project(Xl) - j2d[h][BODY], axis=1))
    return X, err, views, (np.nanmedian(np.concatenate(loo)) if loo else np.nan)

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:4]] if len(sys.argv) > 3 else [0, 706, 10]
    cams = load_cameras(); cl = [cams[c] for c in CAMS]
    rep, loo, nv, tt = [], [], [], []
    for f in range(*a):
        t = time.time(); j2d, j3d, ct, bx = frame_views(f, cams); tt.append(time.time() - t)
        r = fuse(j2d, cl)
        if r is None: rep.append(np.nan); loo.append(np.nan); nv.append(0); continue
        X, err, views, l = r
        rep.append(np.nanmedian(err)); loo.append(l); nv.append(len(views))
        if f % 50 == 0: print(f"frame {f}: views {len(views)} reproj {rep[-1]:.1f}px loo {l:.1f}px  ({tt[-1]:.1f}s for 6 views)", flush=True)
    print("== SAM3D multi-view summary (px at 1080p) ==")
    print("frames", len(rep), "views mean %.2f" % np.mean(nv), "median reproj %.2f  median loo %.2f" % (np.nanmedian(rep), np.nanmedian(loo)))
    print("time per 6-view frame (uncached steady): median %.1fs" % np.median(tt[1:] if len(tt) > 1 else tt))
    np.savez(f"{OUT}/sam3d_seq_summary.npz", rep=rep, loo=loo, nv=nv)
