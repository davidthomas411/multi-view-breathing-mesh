"""Vicon -> GoPro calibration for MMC17 from the reflective markers themselves (no pose model involved).

1. detect marker balls in the 4K frames (marker_detect.py)             -> unlabelled 2D points per camera/frame
2. find the Vicon->room similarity that makes the projected labelled Vicon markers land on the detections in all cameras
   (random search with a wide Gaussian capture radius, then simplex, then annealed robust least squares)
3. mutual-nearest matching -> labelled 2D-3D correspondences -> per-camera PnP (Ransac + LM) with fixed intrinsics, exactly like
   pnp_gopro_vicon_code/gopro_final.py, but on automatically found markers.  Extrinsics are then in the Vicon frame (mm).
4. validation: leave-one-camera-out triangulation of the matched markers vs the Vicon positions (mm).

usage (SAM3D env): python vicon_calib.py [frames...]   default frames: steady supine, 260..540 step 40
"""
import os, sys, csv, json, time
import cv2, numpy as np
from scipy.optimize import least_squares, minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M
import marker_detect as MD

CAMS_USE = [0, 1, 2, 3, 4]                       # T1-T5 (T6 is not synchronized)
CSV = M.ROOT + "/CUTrial/Trials_marker_positions_by_image_frame_all_cases/MMC17_TDB_marker_positions_by_image_frame_wide.csv"
CAL = json.load(open(M.OUT + "/vicon_to_board.json"))
K, DIST = M.K, M.DIST
FLIP = np.diag([1.0, -1.0, -1.0])               # Vicon (Z up) -> board (z down), proper rotation

# ----------------------------------------------------------------- data
rows = list(csv.reader(open(CSV))); NAMES = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]
VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}
def vicon(f): return VIC[f + 1]                  # image_frame_number = video frame index + 1

def detections(f, cache={}):
    p = f"{M.OUT}/marker_dets/{f:04d}.npz"; os.makedirs(os.path.dirname(p), exist_ok=True)
    if os.path.exists(p): z = np.load(p, allow_pickle=True); return list(z["d"])
    z = np.load(f"{M.OUT}/mmc17_sam3d/TDB_{f:04d}.npz"); out = []
    for i in range(6):
        im = M.read("TDB", i + 1, f); b = z["box"][i]; pad = 80
        d = MD.detect(im, [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]) if not np.isnan(b).any() else np.zeros((0, 4))
        if len(d): d[:, :2] = cv2.undistortPoints(d[:, :2].reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2)
        out.append(d)
    np.savez(p, d=np.array(out, dtype=object), allow_pickle=True); return out

# ----------------------------------------------------------------- transform  Vicon(mm) -> board(squares)
def build(params, sq_mm_scale=True):
    yaw, roll, pitch, tx, ty, tz, ls = params
    R = cv2.Rodrigues(np.array([roll, pitch, yaw]))[0]          # small tilts + yaw about board z
    s = np.exp(ls)
    return s, R
def to_board(params, V):
    s, R = build(params); return s * (V @ FLIP.T) @ R.T + np.array(params[3:6])
def init_params():
    s0 = 1.0 / CAL["square_mm"]; th = CAL["yaw_rad"]; cq = np.array(CAL["cq"]); ca = np.array(CAL["ca"])
    R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]); t_xy = ca - s0 * R2 @ cq
    return np.array([th, 0.0, 0.0, t_xy[0], t_xy[1], CAL["tz"], np.log(s0)])

def project(cam, X):                              # pinhole (points are undistorted)
    Xc = X @ cam.R_wc.T + cam.t_wc; uv = Xc @ K.T; return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

def score(params, frames, dets, sigma, V):
    sc = 0.0
    for f in frames:
        Xb = to_board(params, V[f])
        for i in CAMS_USE:
            d = dets[f][i]
            if len(d) == 0: continue
            ok = ~np.isnan(Xb).any(1); uv, z = project(M.CAMS17[i], Xb[ok]); m = z > 0
            if not m.any(): continue
            dd = ((uv[m][:, None, :] - d[None, :, :2]) ** 2).sum(2).min(1); sc += np.exp(-dd / (2 * sigma ** 2)).sum()
    return sc

def lm_refine(params, frames, dets, V, gate):
    """Robust least squares on nearest-detection residuals within a gate (px)."""
    def res(p):
        out = []
        for f in frames:
            Xb = to_board(p, V[f]); ok = ~np.isnan(Xb).any(1)
            for i in CAMS_USE:
                d = dets[f][i]
                if len(d) == 0: continue
                uv, z = project(M.CAMS17[i], Xb[ok]); dist = np.linalg.norm(uv[:, None, :] - d[None, :, :2], axis=2); j = dist.argmin(1); dm = dist.min(1)
                sel = (dm < gate) & (z > 0)
                out.append(np.where(sel[:, None], uv - d[j, :2], 0.0).ravel())
        return np.concatenate(out)
    r = least_squares(res, params, loss="soft_l1", f_scale=gate / 3, max_nfev=60); return r.x

def matches(params, f, i, V, dets, gate=6.0):
    Xb = to_board(params, V[f]); idx = np.where(~np.isnan(Xb).any(1))[0]; uv, z = project(M.CAMS17[i], Xb[idx]); d = dets[f][i]
    if len(d) == 0: return []
    dist = np.linalg.norm(uv[:, None, :] - d[None, :, :2], axis=2); out = []
    for a in range(len(idx)):
        j = dist[a].argmin()
        if z[a] > 0 and dist[a, j] < gate and dist[:, j].argmin() == a: out.append((idx[a], j, dist[a, j]))
    return out

if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[1:]] or list(range(260, 541, 40))
    V = {f: vicon(f) for f in frames}; t0 = time.time(); dets = {f: detections(f) for f in frames}
    print("detections per camera (mean over frames):", np.round([np.mean([len(dets[f][i]) for f in frames]) for i in range(6)], 1), " [%.0fs]" % (time.time() - t0))
    p0 = init_params(); print("initial score (sigma 30px): %.1f" % score(p0, frames, dets, 30.0, V))
    rng = np.random.default_rng(0)
    lo = np.array([-0.35, -0.07, -0.07, -2.0, -2.0, -0.8, -0.2]); hi = -lo
    best = []
    for k in range(7000):
        p = p0 + rng.uniform(lo, hi) * (0.15 + 0.85 * (k / 7000))      # progressively wider search
        p = p0 + rng.uniform(lo, hi)
        best.append((score(p, frames, dets, 30.0, V), p))
    best.sort(key=lambda x: -x[0]); print("random search best scores:", [round(b[0], 1) for b in best[:5]])
    cands = []
    for sc, p in best[:6]:
        for sg, gate in ((20.0, 60.0), (10.0, 25.0)):
            r = minimize(lambda q: -score(q, frames, dets, sg, V), p, method="Nelder-Mead", options=dict(maxiter=500, xatol=1e-4, fatol=1e-4)); p = r.x
        for gate in (40, 20, 10, 6): p = lm_refine(p, frames, dets, V, gate)
        n = sum(len(matches(p, f, i, V, dets)) for f in frames for i in CAMS_USE); cands.append((n, score(p, frames, dets, 8.0, V), p))
    cands.sort(key=lambda x: (-x[0], -x[1])); print("candidate solutions (matches within 6 px, score):", [(c[0], round(c[1], 1)) for c in cands])
    n, sc, p = cands[0]; s, R = build(p)
    print("BEST: matches %d | square size %.1f mm (scale %.5f) | yaw %.2f deg, tilt roll/pitch %.2f/%.2f deg" % (n, 1 / s, s, np.degrees(p[0]), np.degrees(p[1]), np.degrees(p[2])))
    np.save(M.OUT + "/vicon_to_board_params.npy", p)
    # ---- per-camera PnP in the Vicon frame (mm), as in gopro_final.py
    out = {}
    print("per-camera PnP (Vicon frame, fixed intrinsics):")
    for i in CAMS_USE:
        P3, P2, lab = [], [], []
        for f in frames:
            for a, j, e in matches(p, f, i, V, dets): P3.append(V[f][a]); P2.append(dets[f][i][j, :2]); lab.append((f, NAMES[a]))
        if len(P3) < 8: print("  T%d: only %d matches" % (i + 1, len(P3))); continue
        P3 = np.array(P3, np.float64) * 1e-3; P2 = np.array(P2, np.float64)
        ok, rv, tv, inl = cv2.solvePnPRansac(P3, P2, K, None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=4.0, iterationsCount=2000)
        inl = inl.ravel(); rv, tv = cv2.solvePnPRefineLM(P3[inl], P2[inl], K, None, rv, tv)
        pr = cv2.projectPoints(P3[inl], rv, tv, K, None)[0].reshape(-1, 2); rms = float(np.sqrt(np.mean(np.sum((pr - P2[inl]) ** 2, 1))))
        print("  T%d: %d matches, %d RANSAC inliers, reprojection RMS %.2f px" % (i + 1, len(P3), inl.size, rms))
        out["T%d" % (i + 1)] = dict(rvec=rv.ravel().tolist(), tvec=tv.ravel().tolist(), n=int(inl.size), rms=rms)
    json.dump(dict(frames=frames, cams=out, K=K.tolist(), dist=DIST.tolist(), units="metres (Vicon frame), pinhole on Gyroflow-undistorted frames"), open(M.OUT + "/mmc17_vicon_calib.json", "w"), indent=1)
    print("saved tmp/mmc17_vicon_calib.json  [%.0fs total]" % (time.time() - t0))
