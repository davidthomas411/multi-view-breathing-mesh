"""Render 6-camera mosaics with the triangulated SAM 3D Body skeleton reprojected into every view (tmp/overlays/*.jpg + meta json).
green skeleton = fused multi-view skeleton | yellow dots = each view's own prediction (3D+cam_t projected with K) | red dots = >threshold px from fused
Run with the SAM3D env (reads the cached per-view results; no model inference).   C:\dev\sam3d-venv\Scripts\python.exe make_overlays.py"""
import os, sys, json, glob, cv2, numpy as np
sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvcal import load_cameras, CAMS, FRAME_DIR, OUT, triangulate_joints
import sam3d_mv as S
BODY = S.BODY
EDGES = [(0, 69), (69, 5), (69, 6), (5, 9), (6, 10), (9, 10), (5, 7), (7, 62), (6, 8), (8, 41), (9, 11), (11, 13), (13, 17), (10, 12), (12, 14), (14, 20)]
OD = OUT + "/overlays"; os.makedirs(OD, exist_ok=True)

def draw(img, cam, X, own, err_thr, label, used, tw=640):
    s = tw / img.shape[1]; v = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA); th = v.shape[0]
    def P(p): return (int(p[0] * s), int(p[1] * s))
    if X is not None:
        pr = cam.project(X)
        for a, b in EDGES:
            if not np.isnan(pr[[a, b]]).any(): cv2.line(v, P(pr[a]), P(pr[b]), (60, 230, 60), 2, cv2.LINE_AA)
        for j in BODY:
            if not np.isnan(pr[j]).any(): cv2.circle(v, P(pr[j]), 3, (60, 230, 60), -1, cv2.LINE_AA)
    if own is not None:
        for j in BODY:
            if np.isnan(own[j]).any(): continue
            bad = X is not None and np.linalg.norm(cam.project(X)[j] - own[j]) > err_thr
            cv2.circle(v, P(own[j]), 3, (40, 40, 255) if bad else (0, 215, 255), -1, cv2.LINE_AA)
    cv2.rectangle(v, (0, 0), (150 if not used else 112, 20), (0, 0, 0), -1)
    cv2.putText(v, label + ("" if used else "  (not fused)"), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255) if used else (120, 160, 255), 1, cv2.LINE_AA)
    return v

def mosaic(tiles): return np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])

def vault():
    cams = load_cameras(); cl = [cams[c] for c in CAMS]; meta = []
    for p in sorted(glob.glob(S.CACHE + "/*.npz")):
        f = int(os.path.basename(p)[:4]); j2d, j3d, ct, bx = S.frame_views(f, cams)
        r = S.fuse(j2d, cl)
        imgs = [cv2.imread(f"{FRAME_DIR}/{c}/{f:04d}.jpg") for c in CAMS]
        X = r[0] if r else None; views = r[2] if r else []
        Xf = None
        if X is not None: Xf = np.full((70, 3), np.nan); Xf[BODY] = X
        tiles = [draw(imgs[i], cl[i], Xf, j2d[i], 25, "cam %s" % CAMS[i], i in views) for i in range(6)]
        cv2.imwrite(f"{OD}/vault_{f:04d}.jpg", mosaic(tiles), [cv2.IMWRITE_JPEG_QUALITY, 70])
        meta.append(dict(frame=f, views=len(views), reproj=float(np.nanmedian(r[1])) if r else None, loo=float(r[3]) if r and not np.isnan(r[3]) else None))
    json.dump(meta, open(OD + "/vault_meta.json", "w")); print("vault frames:", len(meta))

def mmc17():
    import mmc17_sam3d as M
    meta = []
    for p in sorted(glob.glob(M.OUT + "/mmc17_sam3d/TDB_*.npz")):
        f = int(p[-8:-4])
        if f % 10: continue                                  # uniform 10-frame spacing (0.33 s)
        z = np.load(p); j2d = M.obs2d(z["j3d"], z["cam_t"])
        fuse_views = [i for i in range(5) if not np.isnan(j2d[i][BODY]).any()]            # T1-T5; T6 shown but not fused
        Xf = None; rep = None
        if len(fuse_views) >= 3:
            X, err, _ = triangulate_joints([M.CAMS17[i] for i in fuse_views], j2d[fuse_views][:, BODY]); Xf = np.full((70, 3), np.nan); Xf[BODY] = X; rep = float(np.nanmedian(err))
        imgs = [cv2.cvtColor(M.read("TDB", i + 1, f), cv2.COLOR_RGB2BGR) for i in range(6)]
        tiles = [draw(imgs[i], M.CAMS17[i], Xf, j2d[i], 120, "T%d" % (i + 1), i in fuse_views) for i in range(6)]
        cv2.imwrite(f"{OD}/mmc17_{f:04d}.jpg", mosaic(tiles), [cv2.IMWRITE_JPEG_QUALITY, 70])
        cam_err = [float(np.nanmedian(np.linalg.norm(M.CAMS17[i].project(Xf)[BODY] - j2d[i][BODY], axis=1))) if Xf is not None and not np.isnan(j2d[i][BODY]).any() else None for i in range(6)]
        meta.append(dict(frame=f, views=len(fuse_views), reproj=rep, cam_err=cam_err)); print("mmc17", f, flush=True)
    json.dump(meta, open(OD + "/mmc17_meta.json", "w")); print("mmc17 frames:", len(meta))

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("both", "vault"): vault()
    if which in ("both", "mmc17"): mmc17()
