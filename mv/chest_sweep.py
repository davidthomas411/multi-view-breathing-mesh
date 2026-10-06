"""Controls and matched comparisons for the chest-texture tracking feasibility test.  Writes tmp/chest_track/sweep.json.
 1. seed distance to the true (fitted) surface for every seed variant
 2. resolution sweep on a MATCHED point set (the points valid at every resolution), removing the confound that looser physical filters keep more points
 3. null controls: the tracked signal vs Vicon from OTHER trials of the same subject (DB, ADB) and vs time-shifted copies of the correct Vicon signal
"""
import os, json, csv
import numpy as np
from scipy.spatial import cKDTree
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; CT = TMP + "/chest_track"
T = np.load(TMP + "/track/track.npz", allow_pickle=True); P = np.load(CT + "/points.npz")
out = {}
# ---- 1. seed -> true-surface distance (mm), true surface = fitted mesh at the reference frame
def surface(v, faces, n=200000, rng=np.random.default_rng(0)):
    tri = v[faces]; a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1); k = rng.choice(len(faces), n, p=a / a.sum())
    u, w = rng.random(n), rng.random(n); f = u + w > 1; u[f], w[f] = 1 - u[f], 1 - w[f]; return tri[k, 0] + u[:, None] * (tri[k, 1] - tri[k, 0]) + w[:, None] * (tri[k, 2] - tri[k, 0])
tree = cKDTree(surface(T["vf"][0].astype(np.float64), T["faces"]))
seeds = {"fit": P["X"], "off10": P["X"] + [0, 0, .010], "off20": P["X"] + [0, 0, .020], "off30": P["X"] + [0, 0, .030], "off50": P["X"] + [0, 0, .050], "sam3d": T["vb"][0][P["idx"]].astype(np.float64)}
out["seed_dist_mm"] = {k: [float(np.median(tree.query(v)[0] * 1000)), float(np.percentile(tree.query(v)[0] * 1000, 90))] for k, v in seeds.items()}
# ---- 2. matched-point resolution sweep
def detr(x):
    t = np.arange(len(x)); ok = ~np.isnan(x); return x - np.polyval(np.polyfit(t[ok], x[ok], 2), t)
tags = {"4K": "s1.0", "1080p": "s0.5_fit", "960x540": "s0.25_fit", "480x270": "s0.125_fit"}
E = {}
for name, tag in tags.items():
    p = f"{CT}/eval_{tag}.npz"
    if not os.path.exists(p): p = f"{CT}/eval_s{tag.replace('s', '', 1).replace('_fit', '')}.npz"
    z = np.load(p, allow_pickle=True); E[name] = z
common = np.ones(306, bool)
for z in E.values(): common &= z["keep"]
out["matched_points"] = int(common.sum()); out["resolution"] = {}
for name, z in E.items():
    idx = np.where(z["keep"])[0]; cols = [int(np.where(idx == i)[0][0]) for i in np.where(common)[0]]
    s = np.nanmedian(z["D"][:, cols, 2], axis=1); a, b = detr(z["s_vic"]), detr(s); ok = ~np.isnan(a) & ~np.isnan(b)
    sl = np.polyfit(a[ok], b[ok], 1)[0]; out["resolution"][name] = dict(r=float(np.corrcoef(a[ok], b[ok])[0, 1]), slope=float(sl), mae=float(np.mean(np.abs(a[ok] - b[ok]))), p2p=float(np.nanpercentile(b, 95) - np.nanpercentile(b, 5)), n=int(common.sum()))
# ---- 3. null controls
z = E["4K"]; b = detr(z["s_trk"]); a = detr(z["s_vic"]); frames = z["frames"]
Dd = (os.environ.get("SGRT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/CUTrial/Trials_marker_positions_by_image_frame_all_cases")
CH = [f"{s}{r}{c}" for r in "1234" for s in "RL" for c in "12"]
def vicon_chest(man, fr):
    rows = list(csv.reader(open(f"{Dd}/MMC17_{man}_marker_positions_by_image_frame_wide.csv"))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]
    V = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}
    z0 = np.array([np.nanmean([V[int(f) + 1][names.index(c)][2] for c in CH if c in names]) if int(f) + 1 in V else np.nan for f in fr]); return z0 - z0[0]
null = {}
for man in ("DB", "ADB"):
    s = detr(vicon_chest(man, frames)); ok = ~np.isnan(s) & ~np.isnan(b); null[f"Vicon from the {man} trial (different breathing)"] = float(np.corrcoef(s[ok], b[ok])[0, 1])
sh = {}
for lag in (25, 50, 75, 100):                                                      # 2.5 - 10 s at 10 samples/s
    sh[lag] = float(np.corrcoef(np.roll(a, lag), b)[0, 1])
mae = lambda x, y: float(np.mean(np.abs(x - y)))
out["null_mae_mm"] = {"correct Vicon (lag 0)": mae(a, b)}
for lag in (25, 50, 75, 100): out["null_mae_mm"][f"correct Vicon shifted {lag/10:.1f} s"] = mae(np.roll(a, lag), b)
for man in ("DB", "ADB"):
    sv = detr(vicon_chest(man, frames)); okk = ~np.isnan(sv) & ~np.isnan(b); out["null_mae_mm"][f"Vicon from the {man} trial"] = float(np.mean(np.abs(sv[okk] - b[okk])))
out["null_other_trials"] = null; out["null_shifted_vicon"] = sh; out["true_r"] = float(np.corrcoef(a, b)[0, 1])
print(json.dumps(out, indent=1))
json.dump(out, open(CT + "/sweep.json", "w"), indent=1)
