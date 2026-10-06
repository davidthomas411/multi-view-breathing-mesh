"""Geometry of the RAW GoPro videos (ADB trial, .mov) relative to the Gyroflow-stabilised videos the calibration (tmp/mmc17_vicon_calib.json) was made on.

The ADB data only exist as raw wide-angle video, the TDB data and the ECal clips exist as raw AND Gyroflow-stabilised video.  The stabilisation of a tripod-mounted camera is a fixed warp
(lens undistortion + a small rotation + crop; the static-background drift of the stabilised TDB video is 0.03-0.05 px), so for each camera we fit that warp from the ECal raw/stabilised pair:
    SIFT matches (raw frame <-> stabilised frame of the same clip)  ->  ideal pinhole pixel = K [R * unproject_fisheye(raw pixel)]
with a fisheye radial model (f, cx, cy, k1..k4) and a rotation R, robust least squares.  Usage:
    python adb_geometry.py fit            # fits cameras 1-5 -> tmp/adb_geometry.json
    python adb_geometry.py check <cam>    # warps a raw ADB frame and compares with the stabilised TDB frame (static background)
and, from other scripts:   import adb_geometry as G;  G.Warper(cam)(raw_frame_bgr) -> frame in the stabilised-video geometry (same K, DIST as tmp/mmc17_vicon_calib.json)
"""
import json, os, sys, time
import cv2, numpy as np
from scipy.optimize import least_squares
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; MM = ROOT + "/CUTrial/MM17"
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"]); W, H = 3840, 2160
GEO = TMP + "/adb_geometry.json"


def grab(path, idx):
    c = cv2.VideoCapture(path); c.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, im = c.read(); c.release(); return im if ok else None


def unproject(p, th):
    """raw pixel (N,2) -> unit ray (N,3) under the fisheye model th = [f, cx, cy, k1, k2, k3, k4]."""
    f, cx, cy, k1, k2, k3, k4 = th; x = (p[:, 0] - cx) / f; y = (p[:, 1] - cy) / f; r = np.sqrt(x * x + y * y) + 1e-12
    t = r.copy()
    for _ in range(12):                                                                              # solve  t + k1 t^3 + k2 t^5 + k3 t^7 + k4 t^9 = r  (Newton)
        g = t + k1 * t ** 3 + k2 * t ** 5 + k3 * t ** 7 + k4 * t ** 9 - r; dg = 1 + 3 * k1 * t ** 2 + 5 * k2 * t ** 4 + 7 * k3 * t ** 6 + 9 * k4 * t ** 8; t = t - g / dg
    s = np.sin(t) / r; return np.stack([x * s, y * s, np.cos(t)], 1)


def project_fisheye(d, th):
    """unit ray (N,3) -> raw pixel."""
    f, cx, cy, k1, k2, k3, k4 = th; t = np.arccos(np.clip(d[:, 2] / np.linalg.norm(d, axis=1), -1, 1)); ph = np.arctan2(d[:, 1], d[:, 0])
    td = t + k1 * t ** 3 + k2 * t ** 5 + k3 * t ** 7 + k4 * t ** 9; return np.stack([f * td * np.cos(ph) + cx, f * td * np.sin(ph) + cy], 1)


def rot(w): return cv2.Rodrigues(np.asarray(w, float))[0]


def raw_to_ideal(p, par):
    d = unproject(p, par[:7]) @ rot(par[7:10]).T; return np.stack([K[0, 0] * d[:, 0] / d[:, 2] + K[0, 2], K[1, 1] * d[:, 1] / d[:, 2] + K[1, 2]], 1)


def ideal_to_raw(q, par):
    d = np.stack([(q[:, 0] - K[0, 2]) / K[0, 0], (q[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(q))], 1); d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return project_fisheye(d @ rot(par[7:10]), par[:7])


def matches(raw, stab, sift, scale=0.5):
    g1 = cv2.cvtColor(cv2.resize(raw, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(cv2.resize(stab, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, None); k2, d2 = sift.detectAndCompute(g2, None); bf = cv2.BFMatcher(cv2.NORM_L2); m12 = bf.knnMatch(d1, d2, k=2); m21 = bf.knnMatch(d2, d1, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in m21}; out = []
    for a, b in m12:
        if a.distance < 0.75 * b.distance and back.get(a.trainIdx) == a.queryIdx: out.append((np.array(k1[a.queryIdx].pt) / scale, np.array(k2[a.trainIdx].pt) / scale))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def fit_camera(cam, frames=(0, 5, 10, 15, 20, 25, 29), verbose=True):
    sift = cv2.SIFT_create(nfeatures=9000); P_raw, P_stab, F_id = [], [], []
    for fi in frames:
        a = grab(f"{MM}/DBECal/MMC17_DBECal_T{cam}.mov", fi); b = grab(f"{MM}/DBECal/post_processing/Gyroflow/MMC17_DBECal_T{cam}_stabilized.mp4", fi)
        if a is None or b is None: continue
        pr, ps = matches(a, b, sift); P_raw.append(pr); P_stab.append(ps); F_id.append(np.full(len(pr), fi))
        if verbose: print(f"  T{cam} frame {fi}: {len(pr)} matches", flush=True)
    P_raw = np.vstack(P_raw); P_stab = np.vstack(P_stab); F_id = np.concatenate(F_id)
    Q = cv2.undistortPoints(P_stab.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2)                         # stabilised pixel -> ideal pinhole pixel
    # initial scale from pairs near the centre
    c0 = np.array([W / 2, H / 2]); near = np.where(np.linalg.norm(P_raw - c0, axis=1) < 700)[0]; rng = np.random.default_rng(0); i, j = rng.choice(near, 4000), rng.choice(near, 4000); ok = i != j
    dr = np.linalg.norm(P_raw[i] - P_raw[j], axis=1)[ok]; dq = np.linalg.norm(Q[i] - Q[j], axis=1)[ok]; s = np.median((dq / np.maximum(dr, 1e-6))[dr > 150]); f0 = K[0, 0] / s
    th0 = np.array([f0, W / 2, H / 2, 0, 0, 0, 0, 0, 0, 0], float); sel = np.ones(len(P_raw), bool)
    for stage, (scale_px, ncut) in enumerate(((30.0, 60.0), (6.0, 12.0), (2.0, 5.0))):
        fun = lambda t: (raw_to_ideal(P_raw[sel], t) - Q[sel]).ravel()
        res = least_squares(fun, th0, loss="cauchy", f_scale=scale_px, x_scale=np.array([100, 50, 50, .05, .05, .05, .05, .01, .01, .01]), max_nfev=200); th0 = res.x
        r = np.linalg.norm(raw_to_ideal(P_raw, th0) - Q, axis=1); sel = r < ncut
        if verbose: print(f"  T{cam} stage {stage}: {sel.sum()} / {len(r)} inliers (< {ncut:g} px), median residual {np.median(r[sel]):.2f} px", flush=True)
    r = np.linalg.norm(raw_to_ideal(P_raw, th0) - Q, axis=1); inl = r < 5
    # held-out check: refit without the last frame and test on it
    tr = inl & (F_id != frames[-1]); te = inl & (F_id == frames[-1])
    res2 = least_squares(lambda t: (raw_to_ideal(P_raw[tr], t) - Q[tr]).ravel(), th0, loss="soft_l1", f_scale=1.5, x_scale=np.array([100, 50, 50, .05, .05, .05, .05, .01, .01, .01]), max_nfev=100)
    rt = np.linalg.norm(raw_to_ideal(P_raw[te], res2.x) - Q[te], axis=1); fin = least_squares(lambda t: (raw_to_ideal(P_raw[inl], t) - Q[inl]).ravel(), th0, loss="soft_l1", f_scale=1.5, x_scale=np.array([100, 50, 50, .05, .05, .05, .05, .01, .01, .01]), max_nfev=100).x
    r = np.linalg.norm(raw_to_ideal(P_raw[inl], fin) - Q[inl], axis=1); rad = np.linalg.norm(P_raw[inl] - c0, axis=1)
    info = dict(params=fin.tolist(), n=int(inl.sum()), median_px=float(np.median(r)), p90_px=float(np.percentile(r, 90)), heldout_frame=int(frames[-1]), heldout_median_px=float(np.median(rt)) if len(rt) else None,
                heldout_p90_px=float(np.percentile(rt, 90)) if len(rt) else None, median_by_radius={str(a): float(np.median(r[(rad >= a) & (rad < b)])) if ((rad >= a) & (rad < b)).sum() > 20 else None for a, b in ((0, 600), (600, 1000), (1000, 1400), (1400, 2300))})
    if verbose: print(f"  T{cam} final: {info['n']} matches, median {info['median_px']:.2f} px, p90 {info['p90_px']:.2f} px; held-out frame median {info['heldout_median_px']:.2f} px; by radius {info['median_by_radius']}", flush=True)
    return info


def align_camera(cam, frames=(150, 400, 700, 950), verbose=True):
    """Small constant rotation that registers the (lens-corrected) raw ADB background onto the stabilised TDB background of the same camera (cameras are on tripods and assumed not to have been
    translated between the trials; the extrinsics of tmp/mmc17_vicon_calib.json were estimated on the stabilised TDB video).  Updates the rotation in tmp/adb_geometry.json."""
    geo = json.load(open(GEO)); par = np.array(geo[f"T{cam}"]["params"]); Wp = Warper(cam); sift = cv2.SIFT_create(nfeatures=9000); A, B = [], []
    for fi in frames:
        a = grab(f"{MM}/ADB/MMC17_ADB_T{cam}.mov", fi); b = grab(f"{MM}/TDB/Gyroflow/MMC17_TDB_T{cam}_stabilized.mp4", fi)
        if a is None or b is None: continue
        pa, pb = matches(Wp(a), b, sift); r = np.linalg.norm(pb - pa, axis=1); k = r < 25; A.append(pa[k]); B.append(pb[k])
    A = np.vstack(A); B = np.vstack(B); qa = cv2.undistortPoints(A.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2); qb = cv2.undistortPoints(B.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2)
    ray = lambda q: np.stack([(q[:, 0] - K[0, 2]) / K[0, 0], (q[:, 1] - K[1, 1] * 0 - K[1, 2]) / K[1, 1], np.ones(len(q))], 1)
    def pr(d): return np.stack([K[0, 0] * d[:, 0] / d[:, 2] + K[0, 2], K[1, 1] * d[:, 1] / d[:, 2] + K[1, 2]], 1)
    sel = np.ones(len(qa), bool); w = np.zeros(3)
    for ncut in (20, 8, 4, 3):
        res = least_squares(lambda x: (pr(ray(qa[sel]) @ rot(x).T) - qb[sel]).ravel(), w, loss="soft_l1", f_scale=1.0); w = res.x; r = np.linalg.norm(pr(ray(qa) @ rot(w).T) - qb, axis=1); sel = r < ncut
    Rc = rot(w); Rm = Rc @ rot(par[7:10]); par[7:10] = cv2.Rodrigues(Rm)[0].ravel(); geo[f"T{cam}"]["params"] = par.tolist()
    geo[f"T{cam}"]["align"] = dict(extra_rotation_deg=np.degrees(w).tolist(), n=int(sel.sum()), median_after_px=float(np.median(r[sel])), median_before_px=float(np.median(np.linalg.norm(qa - qb, axis=1)[np.linalg.norm(qa - qb, axis=1) < 25])))
    json.dump(geo, open(GEO, "w"), indent=1)
    if verbose: print(f"  T{cam}: extra rotation {np.round(np.degrees(w), 4)} deg ({np.linalg.norm(w) * K[0, 0]:.1f} px), {sel.sum()} background matches, median offset {geo[f'T{cam}']['align']['median_before_px']:.2f} -> {np.median(r[sel]):.2f} px", flush=True)


class Warper:
    """Warps a raw ADB frame into the geometry of the stabilised videos (K, DIST of tmp/mmc17_vicon_calib.json)."""
    def __init__(self, cam, step=8):
        par = np.array(json.load(open(GEO))[f"T{cam}"]["params"]); gw, gh = W // step + 1, H // step + 1
        u, v = np.meshgrid(np.arange(gw) * step, np.arange(gh) * step); pts = np.stack([u.ravel(), v.ravel()], 1).astype(np.float64)                  # output (stabilised) pixel grid
        q = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, DIST, P=K).reshape(-1, 2); src = ideal_to_raw(q, par).reshape(gh, gw, 2)
        mx = cv2.resize(src[:, :, 0].astype(np.float32), (gw * step, gh * step), interpolation=cv2.INTER_CUBIC)[:H, :W]; my = cv2.resize(src[:, :, 1].astype(np.float32), (gw * step, gh * step), interpolation=cv2.INTER_CUBIC)[:H, :W]
        self.mx, self.my = np.ascontiguousarray(mx), np.ascontiguousarray(my)
    def __call__(self, raw): return cv2.remap(raw, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if mode == "fit":
        out = json.load(open(GEO)) if os.path.exists(GEO) else {}
        for cam in ([int(a) for a in sys.argv[2:]] or [1, 2, 3, 4, 5]):
            t0 = time.time(); out[f"T{cam}"] = fit_camera(cam); json.dump(out, open(GEO, "w"), indent=1); print(f"T{cam} done in {time.time() - t0:.0f}s", flush=True)
    elif mode == "align":
        for cam in ([int(a) for a in sys.argv[2:]] or [1, 2, 3, 4, 5]): align_camera(cam)
    elif mode == "check":
        cam = int(sys.argv[2]); fi = int(sys.argv[3]) if len(sys.argv) > 3 else 400; Wp = Warper(cam)
        a = Wp(grab(f"{MM}/ADB/MMC17_ADB_T{cam}.mov", fi)); b = grab(f"{MM}/TDB/Gyroflow/MMC17_TDB_T{cam}_stabilized.mp4", fi)
        sift = cv2.SIFT_create(nfeatures=9000); pa, pb = matches(a, b, sift); d = pb - pa; r = np.linalg.norm(d, axis=1)
        print(f"T{cam}: {len(pa)} matches between the warped ADB frame and the stabilised TDB frame; displacement median {np.median(r):.2f} px, p25 {np.percentile(r, 25):.2f}, p75 {np.percentile(r, 75):.2f}, p90 {np.percentile(r, 90):.2f}")
        bg = r < 12; print("  background-like matches (<12 px):", int(bg.sum()), "| median offset (dx, dy) = (%.2f, %.2f) px" % (np.median(d[bg, 0]), np.median(d[bg, 1])))
        cv2.imwrite(f"{TMP}/adb_check_T{cam}.jpg", cv2.resize(np.hstack([cv2.resize(a, (960, 540)), cv2.resize(b, (960, 540))]), None, fx=1, fy=1), [cv2.IMWRITE_JPEG_QUALITY, 85])
