"""Benchmark SAM 3D Body against Vicon in the Vicon frame (mm): single view vs multi-view triangulation (T1-T5, and T1-T6 with the marker-recovered T6).
Landmarks with a sound anatomical correspondence only: knee centre (mid of lateral/medial epicondyle markers), ankle centre (mid malleoli),
wrist centre (mid RAD/ULN), elbow centre (mid ELB/MELB), head (mid R/LAHD vs mid of the eyes).   Frames >= 250 (after the sync leg tap)."""
import os
import sys, glob, json, numpy as np, cv2
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import vicon_calib as V, mmc17_sam3d as M
from mvcal import Camera, triangulate_joints
K = V.K; DIST = V.DIST; CJ = json.load(open(M.OUT + "/mmc17_vicon_calib.json"))["cams"]; LAB = json.load(open(M.OUT + "/mmc17_marker_labels.json"))
inv = {v["label"]: k for k, v in LAB.items()}; NAMES = V.NAMES
cams = []
for i in range(6):
    c = CJ["T%d" % (i + 1)]; cams.append(Camera("T%d" % (i + 1), K, cv2.Rodrigues(np.array(c["rvec"]))[0], np.array(c["tvec"]), 3840, 2160, DIST))
def mk(P, *labels):
    ids = [NAMES.index(inv.get(l, l)) for l in labels if inv.get(l, l) in NAMES]
    return np.nanmean([P[i] for i in ids], 0) if ids else np.full(3, np.nan)
# (name, SAM3D MHR70 index, Vicon marker labels)
LM = [("L knee", 11, ("LLKNE", "LMKNE")), ("R knee", 12, ("RLKNE", "RMKNE")), ("L ankle", 13, ("LLML", "LMML")), ("R ankle", 14, ("RLML", "RMML")),
      ("L wrist", 62, ("LRAD", "LULN")), ("R wrist", 41, ("RRAD", "RULN")), ("L elbow", 7, ("LELB", "LMELB")), ("R elbow", 8, ("RELB", "RMELB"))]
res = {m: {"single": [], "mv5": [], "mv6": []} for m, _, _ in LM}; res["head (eyes vs headband)"] = {"single": [], "mv5": [], "mv6": []}
for p in sorted(glob.glob(M.OUT + "/mmc17_sam3d/TDB_*.npz")):
    f = int(p[-8:-4])
    if f % 10 or f < 250: continue
    z = np.load(p); j3d, ct = z["j3d"], z["cam_t"]; Vf = V.vicon(f); j2d = np.full((6, 70, 2), np.nan)
    for i in range(6):
        if np.isnan(j3d[i]).any(): continue
        uv = (K @ (j3d[i] + ct[i]).T).T; j2d[i] = cams[i].undistort(uv[:, :2] / uv[:, 2:3])
    fused = {}
    for tag, use in (("mv5", [0, 1, 2, 3, 4]), ("mv6", [0, 1, 2, 3, 4, 5])):
        vs = [i for i in use if not np.isnan(j2d[i][M.S.BODY]).any()]
        if len(vs) >= 3:
            X, _, _ = triangulate_joints([cams[i] for i in vs], j2d[vs][:, M.S.BODY]); Xf = np.full((70, 3), np.nan); Xf[M.S.BODY] = X; fused[tag] = Xf * 1000
    single = []
    for i in range(5):                                          # single views T1-T5 -> Vicon frame (mm)
        if np.isnan(j3d[i]).any(): continue
        Xv = ((j3d[i] + ct[i] - cams[i].t_wc) @ cams[i].R_wc) * 1000; single.append(Xv)
    for name, j, labs in LM + [("head (eyes vs headband)", None, ("RAHD", "LAHD"))]:
        ref = mk(Vf, *labs)
        if np.isnan(ref).any(): continue
        for tag in ("mv5", "mv6"):
            if tag in fused:
                est = fused[tag][j] if j is not None else (fused[tag][1] + fused[tag][2]) / 2
                if not np.isnan(est).any(): res[name][tag].append(np.linalg.norm(est - ref))
        for Xv in single:
            est = Xv[j] if j is not None else (Xv[1] + Xv[2]) / 2
            res[name]["single"].append(np.linalg.norm(est - ref))
print("SAM 3D Body vs Vicon landmark centres, median / p90 error (mm), frames >= 250 every 10th")
print("%-26s %-18s %-18s %-18s" % ("landmark", "single view (T1-5)", "multi-view T1-T5", "multi-view T1-T6"))
out = {}
for n, d in res.items():
    f = lambda a: "%4.0f / %4.0f (n=%d)" % (np.median(a), np.percentile(a, 90), len(a)) if len(a) else "-"
    print("%-26s %-18s %-18s %-18s" % (n, f(d["single"]), f(d["mv5"]), f(d["mv6"]))); out[n] = {k: [float(np.median(v)), float(np.percentile(v, 90)), len(v)] if len(v) else None for k, v in d.items()}
allv = {k: np.concatenate([d[k] for d in res.values() if len(d[k])]) for k in ("single", "mv5", "mv6")}
print("ALL landmarks: single %.0f / %.0f   mv5 %.0f / %.0f   mv6 %.0f / %.0f" % tuple(x for k in ("single", "mv5", "mv6") for x in (np.median(allv[k]), np.percentile(allv[k], 90))))
out["ALL"] = {k: [float(np.median(v)), float(np.percentile(v, 90)), len(v)] for k, v in allv.items()}; json.dump(out, open(M.OUT + "/sam3d_mv_vs_vicon.json", "w"), indent=1)
