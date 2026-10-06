"""Recover camera T6 in the Vicon frame from the marker balls: search camera poses (6 DOF) around its board-derived hypotheses, scoring how many
projected Vicon markers land on T6's detected balls over several frames; then PnP.  Frame offsets (Vicon frame shifted by +-d image frames) are tested too,
because T6 was recorded at 24 fps (others 30)."""
import os
import sys, json, numpy as np, cv2, time
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import vicon_calib as V, mmc17_sam3d as M
K, DIST = V.K, V.DIST; p = np.load(M.OUT + "/vicon_to_board_params.npy"); s_, Rr = V.build(p); tb = np.array(p[3:6]); FLIP = V.FLIP
frames = list(range(260, 541, 40)); dets = {f: V.detections(f) for f in frames}
board = json.load(open(M.OUT + "/mmc17_calib_v3.json"))["cams"]["T6"]; Rc0 = np.array(board["R"]); tc0 = np.array(board["t"])
objp = np.zeros((35, 3), np.float32); objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2); idx = np.arange(35).reshape(5, 7)
sub = np.load(M.OUT + "/board_sub.npz")["sub"][5]; subu = cv2.undistortPoints(sub.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2)
def board_pose(opt):
    off, flip = opt // 2, opt % 2; sel = idx[:, off:off + 6].ravel(); sel = sel[::-1] if flip else sel
    ok, rv, tv = cv2.solvePnP(objp[sel], subu, K, None, flags=cv2.SOLVEPNP_ITERATIVE); return cv2.Rodrigues(rv)[0], tv.ravel()
def to_vicon_pose(Rc, tc):                     # board-frame camera -> Vicon metres frame (X_c = R V_m + t)
    S = s_ * 1000; R = Rc @ Rr @ FLIP; t = (Rc @ tb + tc) / S; return R * (S / S), t   # rotation part: unit scale after dividing
def score(R, t, shift, sigma):
    sc = 0
    for f in frames:
        Vf = V.vicon(min(max(f + shift, 0), 1124)) * 1e-3; d = dets[f][5]
        if len(d) == 0: continue
        Vk = Vf[~np.isnan(Vf).any(1)]; Xc = Vk @ R.T + t; z = Xc[:, 2]; uv = (Xc @ K.T); uv = uv[:, :2] / uv[:, 2:3]; m = z > 0
        if not m.any(): continue
        dd = ((uv[m][:, None, :] - d[None, :, :2]) ** 2).sum(2).min(1); sc += np.exp(-dd / (2 * sigma ** 2)).sum()
    return sc
def nmatch(R, t, shift, gate=6.0):
    n = 0
    for f in frames:
        Vf = V.vicon(min(max(f + shift, 0), 1124)) * 1e-3; d = dets[f][5]
        if len(d) == 0: continue
        idxm = np.where(~np.isnan(Vf).any(1))[0]; Xc = Vf[idxm] @ R.T + t; uv = Xc @ K.T; uv = uv[:, :2] / uv[:, 2:3]; dist = np.linalg.norm(uv[:, None, :] - d[None, :, :2], axis=2)
        for a in range(len(idxm)):
            j = dist[a].argmin()
            if Xc[a, 2] > 0 and dist[a, j] < gate and dist[:, j].argmin() == a: n += 1
    return n
rng = np.random.default_rng(1); best = []
print("detections in T6 per frame:", [len(dets[f][5]) for f in frames])
for opt in range(4):
    Rb, tb_ = board_pose(opt); R0, t0 = to_vicon_pose(Rb, tb_)
    print("hypothesis %d: camera centre in Vicon frame (m): %s" % (opt, np.round(-R0.T @ t0, 2)), " baseline matches:", nmatch(R0, t0, 0))
    for k in range(3500):
        dr = cv2.Rodrigues(rng.normal(0, np.radians(6), 3))[0]; R = dr @ R0; t = t0 + rng.normal(0, 0.12, 3)
        best.append((score(R, t, 0, 25.0), opt, R, t))
best.sort(key=lambda x: -x[0]); print("top random-search scores (sigma 25 px):", [(round(b[0], 1), b[1]) for b in best[:6]])
from scipy.optimize import minimize
res = []
for sc, opt, R, t in best[:5]:
    x0 = np.r_[cv2.Rodrigues(R)[0].ravel(), t]
    f = lambda x: -score(cv2.Rodrigues(x[:3])[0], x[3:], 0, 10.0)
    r = minimize(f, x0, method="Nelder-Mead", options=dict(maxiter=600, xatol=1e-4, fatol=1e-4)); Rn = cv2.Rodrigues(r.x[:3])[0]; tn = r.x[3:]
    res.append((nmatch(Rn, tn, 0), -r.fun, opt, Rn, tn))
res.sort(key=lambda x: (-x[0], -x[1])); print("refined candidates (matches within 6 px, score, hypothesis):", [(a, round(b, 1), c) for a, b, c, _, _ in res])
n, sc, opt, Rn, tn = res[0]
print("best: %d matches over %d frames, camera centre %s m" % (n, len(frames), np.round(-Rn.T @ tn, 3)))
shift = {d: nmatch(Rn, tn, d) for d in range(-6, 7)}; print("matches vs Vicon frame offset (image frames):", shift)
json.dump(dict(matches=int(n), score=float(sc), hypothesis=int(opt), R=Rn.tolist(), t=tn.tolist(), shift_matches=shift), open(M.OUT + "/t6_search.json", "w"), indent=1)
