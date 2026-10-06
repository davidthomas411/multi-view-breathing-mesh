"""Like-for-like scoring (Vicon only here): every camera's single-view SAM 3D Body mesh T1..T5, the markerless multi-view mesh, and the marker-guided reference,
all scored as the distance from each Vicon marker to the nearest of the SAME ~6k subsampled vertices (so the numbers are comparable; they carry a small positive bias
vs a true surface distance, ~5-10 mm).   -> tmp/mv_fit/eval2.npz"""
import os, sys, json
import numpy as np, cv2
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
T = np.load(TMP + "/track/track.npz", allow_pickle=True); MV = np.load(TMP + "/mv_fit/mv_fit.npz"); names = [str(n) for n in T["names"]]
tf = {int(f): i for i, f in enumerate(T["frames"])}; CJ = json.load(open(TMP + "/mmc17_vicon_calib.json"))["cams"]
RT = [(cv2.Rodrigues(np.array(CJ["T%d" % i]["rvec"]))[0], np.array(CJ["T%d" % i]["tvec"])) for i in range(1, 6)]
D = {k: [] for k in ("sv1", "sv2", "sv3", "sv4", "sv5", "mv", "fit")}; frames = []
for n, f in enumerate(MV["frames"]):
    f = int(f); z = np.load(f"{TMP}/mv_sam3d/{f:04d}.npz"); vidx = z["vidx"]; i = tf[f]; mk = T["mk"][i].astype(np.float64); ok = ~np.isnan(mk).any(1); row = {}
    for v in range(5):
        if z["ok"][v]:
            R, t = RT[v]; Xv = ((z["verts"][v].astype(np.float64) + z["cam_t"][v]) - t) @ R; d = np.full(len(mk), np.nan); d[ok] = cKDTree(Xv).query(mk[ok])[0] * 1000; row["sv%d" % (v + 1)] = d
        else: row["sv%d" % (v + 1)] = np.full(len(mk), np.nan)
    for k, X in (("mv", MV["X"][n]), ("fit", T["vf"][i])):
        d = np.full(len(mk), np.nan); d[ok] = cKDTree(X[vidx].astype(np.float64)).query(mk[ok])[0] * 1000; row[k] = d
    frames.append(f)
    for k in D: D[k].append(row[k])
np.savez(TMP + "/mv_fit/eval2.npz", frames=np.array(frames), names=np.array(names), **{k: np.array(v) for k, v in D.items()})
print("frames scored:", len(frames))
sv = np.array([D["sv%d" % v] for v in range(1, 6)])                                # (5,F,M)
for v in range(5): print("  single view T%d : median %.1f mm  p90 %.1f" % (v + 1, np.nanmedian(sv[v]), np.nanpercentile(sv[v], 90)))
print("  single views, typical (mean of the 5 per-camera medians): %.1f mm | best camera by median %.1f mm" % (np.mean([np.nanmedian(sv[v]) for v in range(5)]), min(np.nanmedian(sv[v]) for v in range(5))))
for k, lab in (("mv", "multi-view (markerless)"), ("fit", "marker-guided (reference)")): print("  %-26s median %.1f mm  p90 %.1f" % (lab, np.nanmedian(D[k]), np.nanpercentile(D[k], 90)))
