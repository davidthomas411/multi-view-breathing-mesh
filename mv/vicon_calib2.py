"""Use the marker matches (anchored by T1-T3) to settle the board-alignment ambiguity of the other cameras: for each camera try every
sub-grid offset/flip hypothesis, project the Vicon markers with the Vicon->board transform, and count mutual-nearest matches to the
detected balls.  Then per-camera PnP in the Vicon frame for all cameras that match."""
import os
import sys, json, itertools, numpy as np, cv2
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import vicon_calib as V
import mmc17_sam3d as M
from mvcal import Camera
K, DIST = V.K, V.DIST
p = np.load(M.OUT + "/vicon_to_board_params.npy"); frames = list(range(260, 541, 40)); VV = {f: V.vicon(f) for f in frames}; dets = {f: V.detections(f) for f in frames}
objp = np.zeros((35, 3), np.float32); objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
sub = list(np.load(M.OUT + "/board_sub.npz")["sub"]); subu = [cv2.undistortPoints(s.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2) for s in sub]
idx = np.arange(35).reshape(5, 7)
def cam_from(i, opt):
    off, flip = opt // 2, opt % 2; sel = idx[:, off:off + 6].ravel(); sel = sel[::-1] if flip else sel
    ok, rv, tv = cv2.solvePnP(objp[sel], subu[i].astype(np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE); R = cv2.Rodrigues(rv)[0]
    return Camera("T%d" % (i + 1), K, R, tv.ravel(), 3840, 2160, DIST)
def count(cam, i):
    n = 0; s = 0.0
    for f in frames:
        Xb = V.to_board(p, VV[f]); idxm = np.where(~np.isnan(Xb).any(1))[0]; uv, z = V.project(cam, Xb[idxm]); d = dets[f][i]
        if len(d) == 0: continue
        dist = np.linalg.norm(uv[:, None, :] - d[None, :, :2], axis=2)
        for a in range(len(idxm)):
            j = dist[a].argmin()
            if z[a] > 0 and dist[a, j] < 6 and dist[:, j].argmin() == a: n += 1
        s += np.exp(-(dist.min(1) ** 2) / (2 * 15 ** 2)).sum()
    return n, s
print("matches within 6 px for each alignment hypothesis (option = offset*2 + flip), current choice marked *")
cur = json.load(open(M.OUT + "/mmc17_calib.json"))["combo"] if "combo" in json.load(open(M.OUT + "/mmc17_calib.json")) else None
best = {}
for i in range(6):
    res = [(count(cam_from(i, o), i), o) for o in range(4)]
    print("  T%d:" % (i + 1), "  ".join("opt%d: %3d matches (score %.1f)%s" % (o, n, s, "*" if cur and cur[i] == o else "") for ((n, s), o) in res))
    best[i] = max(res, key=lambda r: (r[0][0], r[0][1]))[1]
print("best hypothesis per camera:", {"T%d" % (i + 1): o for i, o in best.items()})
json.dump(dict(best=best), open(M.OUT + "/vicon_alignment_best.json", "w"))
