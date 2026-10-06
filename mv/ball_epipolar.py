"""Do detected marker balls satisfy the calibrated epipolar geometry?  (hit = a ball in cam b within `tol` px of the epipolar line of a ball in cam a)"""
import os
import sys, glob, itertools, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M
K = M.K; Ki = np.linalg.inv(K)
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
def Fmat(cams, a, b):
    Rab = cams[b].R_wc @ cams[a].R_wc.T; tab = cams[b].t_wc - Rab @ cams[a].t_wc; return Ki.T @ (skew(tab) @ Rab) @ Ki
files = sorted(glob.glob(M.OUT + "/marker_dets/*.npz")); D = [list(np.load(f, allow_pickle=True)["d"]) for f in files]
print("frames:", len(D), " mean balls per camera:", np.round([np.mean([len(d[i]) for d in D]) for i in range(5)], 1))
def rate(cams, tol):
    hit = tot = 0; chance = 0
    for d in D:
        for a, b in itertools.permutations(range(5), 2):
            if len(d[a]) == 0 or len(d[b]) == 0: continue
            F = Fmat(cams, a, b); xa = np.c_[d[a][:, :2], np.ones(len(d[a]))]; xb = np.c_[d[b][:, :2], np.ones(len(d[b]))]
            l = xa @ F.T; dist = np.abs(l @ xb.T) / np.hypot(l[:, :1], l[:, 1:2]); hit += int((dist.min(1) < tol).sum()); tot += len(d[a])
            # chance level: same test with the other image's points mirrored vertically (breaks true correspondences, keeps density)
            xb2 = xb.copy(); xb2[:, 1] = 2160 - xb2[:, 1]; dist2 = np.abs(l @ xb2.T) / np.hypot(l[:, :1], l[:, 1:2]); chance += int((dist2.min(1) < tol).sum())
    return hit / tot, chance / tot
from mvcal import Camera
def load(name):
    C = json.load(open(M.OUT + "/" + name)); return [Camera("T%d" % i, np.array(C["K"]), np.array(C["cams"]["T%d" % i]["R"]), np.array(C["cams"]["T%d" % i]["t"]), 3840, 2160, np.array(C["dist"])) for i in range(1, 7)]
for name in ("mmc17_calib_v1_wrong.json", "mmc17_calib.json"):
    cams = load(name)
    for tol in (2.0, 4.0, 8.0, 16.0):
        h, c = rate(cams, tol); print("%-28s tol %4.0f px: hit rate %.2f   (chance with mirrored points %.2f)" % (name, tol, h, c))
