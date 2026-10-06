"""Like-for-like scoring of the single-camera SAM 3D Body meshes (T1-T5) and the markerless multi-view mesh against the Vicon markers for ANY trial (TRIAL=ADB or TDB), without the marker-guided fit.
Each mesh: distance from every Vicon marker to the nearest of the same ~6k subsampled vertices (small positive bias, the same for all).   -> tmp/mv_fit[_TRIAL]/eval_trial.npz
usage:  TRIAL=ADB python mv_eval_trial.py"""
import csv, json, os, sys
import cv2, numpy as np
from scipy.spatial import cKDTree
import trial_io as TI
MVS = TI.tdir("mv_sam3d", make=False); MVF = TI.tdir("mv_fit", make=False)
CJ = json.load(open(TI.calib_path()))["cams"]; RT = [(cv2.Rodrigues(np.array(CJ["T%d" % i]["rvec"]))[0], np.array(CJ["T%d" % i]["tvec"])) for i in range(1, 6)]
rows = list(csv.reader(open(TI.vicon_csv()))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]; VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) * 1e-3 for r in rows[1:]}
MV = np.load(MVF + "/mv_fit.npz"); frames = [int(f) for f in MV["frames"]]; D = {k: [] for k in ("sv1", "sv2", "sv3", "sv4", "sv5", "mv")}
for n, f in enumerate(frames):
    z = np.load(f"{MVS}/{f:04d}.npz"); vidx = z["vidx"]; mk = VIC[f + 1]; ok = ~np.isnan(mk).any(1)
    for v in range(5):
        d = np.full(len(mk), np.nan)
        if z["ok"][v]: R, t = RT[v]; Xv = ((z["verts"][v].astype(np.float64) + z["cam_t"][v]) - t) @ R; d[ok] = cKDTree(Xv).query(mk[ok])[0] * 1000
        D["sv%d" % (v + 1)].append(d)
    d = np.full(len(mk), np.nan); d[ok] = cKDTree(MV["X"][n][vidx].astype(np.float64)).query(mk[ok])[0] * 1000; D["mv"].append(d)
np.savez(MVF + "/eval_trial.npz", frames=np.array(frames), names=np.array(names), **{k: np.array(v) for k, v in D.items()})
sv = np.array([D["sv%d" % v] for v in range(1, 6)]); print(TI.TRIAL, "frames scored:", len(frames))
for v in range(5): print("  single view T%d : median %.1f mm  p90 %.1f" % (v + 1, np.nanmedian(sv[v]), np.nanpercentile(sv[v], 90)))
print("  typical single view (mean of the 5 medians): %.1f | best %.1f | multi-view (markerless): median %.1f mm, p90 %.1f" % (np.mean([np.nanmedian(sv[v]) for v in range(5)]), min(np.nanmedian(sv[v]) for v in range(5)), np.nanmedian(D["mv"]), np.nanpercentile(D["mv"], 90)))
