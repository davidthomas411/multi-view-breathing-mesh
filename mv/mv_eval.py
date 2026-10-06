"""Score the three meshes against the Vicon markers (Vicon used ONLY here, not in the multi-view fit):
   red   = SAM 3D Body, single camera (T4)            (tmp/track/track.npz, 'vb')
   blue  = SAM 3D Body, markerless multi-view fit     (tmp/mv_fit/mv_fit.npz)
   green = marker-guided fit (reference)              (tmp/track/track.npz, 'vf')
Per marker: distance to the mesh surface (mm; ~8 mm = perfect).  Also the chest surface height under the chest markers (breathing).   -> tmp/mv_fit/eval.npz"""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
T = np.load(TMP + "/track/track.npz", allow_pickle=True); MV = np.load(TMP + "/mv_fit/mv_fit.npz"); faces = MV["faces"]
tf = {int(f): i for i, f in enumerate(T["frames"])}; names = [str(n) for n in T["names"]]
CHEST = [i for i, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]
def surface(v, n=80000, rng=np.random.default_rng(0)):
    tri = v[faces]; a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1); k = rng.choice(len(faces), n, p=a / a.sum())
    u, w = rng.random(n), rng.random(n); f = u + w > 1; u[f], w[f] = 1 - u[f], 1 - w[f]; return tri[k, 0] + u[:, None] * (tri[k, 1] - tri[k, 0]) + w[:, None] * (tri[k, 2] - tri[k, 0])
out = dict(frames=[], dist_sv=[], dist_mv=[], dist_fit=[], chest_mk=[], chest_sv=[], chest_mv=[], chest_fit=[])
for n, f in enumerate(MV["frames"]):
    f = int(f)
    if f not in tf: continue
    i = tf[f]; mk = T["mk"][i].astype(np.float64); ok = ~np.isnan(mk).any(1)
    meshes = {"sv": T["vb"][i].astype(np.float64), "mv": MV["X"][n].astype(np.float64), "fit": T["vf"][i].astype(np.float64)}
    row = {}
    for k, v in meshes.items():
        d = np.full(len(mk), np.nan)
        if not np.isnan(v).any():
            pts = surface(v); tree = cKDTree(pts); d[ok] = tree.query(mk[ok])[0] * 1000
            cz = [j for j in CHEST if ok[j]]; row["chest_" + k] = float(np.mean(pts[tree.query(mk[cz])[1], 2])) * 1000 if cz else np.nan
        else: row["chest_" + k] = np.nan
        row["dist_" + k] = d
    cz = [j for j in CHEST if ok[j]]; row["chest_mk"] = float(np.mean(mk[cz, 2])) * 1000 if cz else np.nan
    out["frames"].append(f)
    for k in ("dist_sv", "dist_mv", "dist_fit", "chest_mk", "chest_sv", "chest_mv", "chest_fit"): out[k].append(row[k])
    if n % 20 == 0: print("frame %d: median marker->surface  single-view %.0f | multi-view %.0f | marker-guided %.0f mm" % (f, np.nanmedian(row["dist_sv"]), np.nanmedian(row["dist_mv"]), np.nanmedian(row["dist_fit"])), flush=True)
np.savez(TMP + "/mv_fit/eval.npz", names=np.array(names), **{k: np.array(v) for k, v in out.items()})
D = {k: np.array(out["dist_" + k]) for k in ("sv", "mv", "fit")}
for k, lab in (("sv", "SAM 3D Body single view (T4)"), ("mv", "SAM 3D Body multi-view mesh (markerless)"), ("fit", "marker-guided fit (reference)")):
    pf = np.nanmedian(D[k], axis=1); print("%-42s per-frame median %.1f mm (p90 over frames %.1f) | all markers: median %.1f, p90 %.1f mm" % (lab, np.median(pf), np.percentile(pf, 90), np.nanmedian(D[k]), np.nanpercentile(D[k], 90)))
