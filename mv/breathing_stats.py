"""What does the breathing signal look like in the Vicon chest array (16 markers, 4x4), across all subjects and maneuvers?  amplitude, low-rank structure, pixel budget."""
import csv, glob, json, os, re
import numpy as np
D = (os.environ.get("SGRT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/CUTrial/Trials_marker_positions_by_image_frame_all_cases")
CHEST = [f"{s}{r}{c}" for r in "1234" for s in "RL" for c in "12"]
def load(p):
    rows = list(csv.reader(open(p))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]
    keep = [i for i, r in enumerate(rows[1:]) if r[9] == "True"]
    A = np.array([[float(v) if v else np.nan for v in rows[1 + i][10:]] for i in keep]).reshape(len(keep), -1, 3)
    return names, A
res = []
for p in sorted(glob.glob(D + "/MMC*_wide.csv")):
    sub, man = re.search(r"(MMC\d+)_(\w+?)_marker", os.path.basename(p)).groups(); names, A = load(p)
    if not all(c in names for c in CHEST): continue
    Z = np.array([A[:, names.index(c), 2] for c in CHEST]).T                    # (T,16) vertical = anterior when supine, mm
    ok = ~np.isnan(Z).any(1); Z = Z[ok]
    if len(Z) < 300: continue
    t = np.arange(len(Z)); Zd = Z - np.array([np.polyval(np.polyfit(t, Z[:, k], 2), t) for k in range(16)]).T            # remove slow drift (2nd order)
    k = 5; Zs = np.array([np.convolve(Zd[:, j], np.ones(k) / k, "same") for j in range(16)]).T                          # light smoothing (30 fps)
    p2p = np.percentile(Zs, 95, 0) - np.percentile(Zs, 5, 0)
    u, s, vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False); ev = s ** 2 / np.sum(s ** 2)
    rows_p2p = [np.mean(p2p[[j for j, c in enumerate(CHEST) if c[1] == r]]) for r in "1234"]
    res.append(dict(sub=sub, man=man, n=len(Z), p2p_mean=float(p2p.mean()), p2p_max=float(p2p.max()), ev=[float(x) for x in ev[:4]], rows=[float(x) for x in rows_p2p]))
print("trials analysed:", len(res), " subjects:", len({r['sub'] for r in res}))
for man in ("DB", "TDB", "ADB"):
    R = [r for r in res if r["man"] == man]
    print("%-3s n=%2d | peak-to-peak (mean over markers) median %.1f mm [IQR %.1f-%.1f], max marker median %.1f mm | PC1 %.0f%%  PC1+2 %.0f%%  PC1-3 %.0f%% of variance | by chest row 1..4 (mm): %s" % (
        man, len(R), np.median([r["p2p_mean"] for r in R]), np.percentile([r["p2p_mean"] for r in R], 25), np.percentile([r["p2p_mean"] for r in R], 75), np.median([r["p2p_max"] for r in R]),
        100 * np.median([r["ev"][0] for r in R]), 100 * np.median([sum(r["ev"][:2]) for r in R]), 100 * np.median([sum(r["ev"][:3]) for r in R]), np.round(np.median([r["rows"] for r in R], 0), 1)))
allr = res
print("ALL trials: PC1 median %.0f%% (min %.0f%%), first 3 modes median %.0f%% (min %.0f%%)" % (100 * np.median([r["ev"][0] for r in allr]), 100 * min(r["ev"][0] for r in allr), 100 * np.median([sum(r["ev"][:3]) for r in allr]), 100 * min(sum(r["ev"][:3]) for r in allr)))
json.dump(res, open((os.environ.get("SGRT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/tmp/breathing_stats.json"), "w"))
