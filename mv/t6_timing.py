"""Does T6 drift in time against the Vicon/other cameras?  For frames across the trial, find the Vicon-frame shift (in image frames) that maximises marker matches, for T6 and a reference camera."""
import os
import sys, json, numpy as np, cv2
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import vicon_calib as V, mmc17_sam3d as M
K = V.K; C = json.load(open(M.OUT + "/mmc17_vicon_calib.json"))["cams"]
def pose(i): c = C["T%d" % (i + 1)]; return cv2.Rodrigues(np.array(c["rvec"]))[0], np.array(c["tvec"])
def matches(i, d, Vf, gate=6.0):
    R, t = pose(i); idxm = np.where(~np.isnan(Vf).any(1))[0]; Xc = Vf[idxm] @ R.T + t; uv = Xc @ K.T; uv = uv[:, :2] / uv[:, 2:3]
    if len(d) == 0: return 0
    dist = np.linalg.norm(uv[:, None, :] - d[None, :, :2], axis=2); n = 0
    for a in range(len(idxm)):
        j = dist[a].argmin()
        if Xc[a, 2] > 0 and dist[a, j] < gate and dist[:, j].argmin() == a: n += 1
    return n
frames = list(range(100, 1101, 100)); print("frame | T6: best shift (matches @0 / best) | T5 reference: best shift (matches @0 / best)")
rows = []
for f in frames:
    D = V.detections(f); out = []
    for i in (5, 4):
        res = {sh: matches(i, D[i], V.vicon(min(max(f + sh, 0), 1124)) * 1e-3) for sh in range(-8, 9)}
        best = max(res, key=lambda k: (res[k], -abs(k))); out.append((best, res[0], res[best]))
    rows.append((f, out)); print("%5d | T6: %+3d (%3d / %3d)            | T5: %+3d (%3d / %3d)" % (f, out[0][0], out[0][1], out[0][2], out[1][0], out[1][1], out[1][2]), flush=True)
json.dump([(f, o) for f, o in rows], open(M.OUT + "/t6_timing.json", "w"))
