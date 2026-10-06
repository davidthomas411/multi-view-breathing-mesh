"""How good is the marker-based Vicon calibration?  (a) in-sample reprojection of Vicon markers, (b) triangulation of the detected balls
from the cameras vs the Vicon coordinate (mm), including leave-one-camera-out, on frames not used to fit."""
import os
import sys, json, numpy as np, cv2
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import vicon_calib as V, mmc17_sam3d as M
K = V.K; C = json.load(open(M.OUT + "/mmc17_vicon_calib.json"))["cams"]
poses = {}
for i in range(5):
    c = C["T%d" % (i + 1)]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; poses[i] = (R, np.array(c["tvec"]))
def proj(i, X):                            # X metres (Vicon frame)
    R, t = poses[i]; Xc = X @ R.T + t; uv = Xc @ K.T; return uv[:, :2] / uv[:, 2:3], Xc[:, 2]
def P34(i): R, t = poses[i]; return K @ np.hstack([R, t[:, None]])
def tri(cams, uvs):
    A = []
    for i, uv in zip(cams, uvs): P = P34(i); A += [uv[0] * P[2] - P[0], uv[1] * P[2] - P[1]]
    X = np.linalg.svd(np.array(A))[2][-1]; return X[:3] / X[3]
fit_frames = list(range(260, 541, 40)); test_frames = [280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 700, 760]
err_all, err_loo, rep_hold, nm = [], [], [], []
for f in test_frames:
    Vf = V.vicon(f) * 1e-3; dets = V.detections(f); M_ = {}                  # marker idx -> {cam: 2D}
    for i in range(5):
        d = dets[i]
        if len(d) == 0: continue
        ok = np.where(~np.isnan(Vf).any(1))[0]; uv, z = proj(i, Vf[ok]); dist = np.linalg.norm(uv[:, None, :] - d[None, :, :2], axis=2)
        for a in range(len(ok)):
            j = dist[a].argmin()
            if z[a] > 0 and dist[a, j] < 6 and dist[:, j].argmin() == a: M_.setdefault(ok[a], {})[i] = d[j, :2]
    for m, obs in M_.items():
        cs = sorted(obs)
        if len(cs) >= 3:
            X = tri(cs, [obs[c] for c in cs]); err_all.append(np.linalg.norm(X - Vf[m]) * 1000)
        if len(cs) >= 4:
            for h in cs:
                r = [c for c in cs if c != h]; X = tri(r, [obs[c] for c in r]); err_loo.append(np.linalg.norm(X - Vf[m]) * 1000)
                uvh, _ = proj(h, X[None]); rep_hold.append(np.linalg.norm(uvh[0] - obs[h]))
    nm.append(len(M_))
print("test frames (not all used in fitting):", test_frames, "| matched labelled markers per frame:", nm)
print("markers triangulated from >=3 cameras: %d   error vs Vicon: median %.1f mm, p90 %.1f mm, mean %.1f mm" % (len(err_all), np.median(err_all), np.percentile(err_all, 90), np.mean(err_all)))
if err_loo: print("leave-one-camera-out (>=4 cams): %d triangulations, error vs Vicon median %.1f mm, p90 %.1f mm; held-out camera reprojection median %.1f px" % (len(err_loo), np.median(err_loo), np.percentile(err_loo, 90), np.median(rep_hold)))
