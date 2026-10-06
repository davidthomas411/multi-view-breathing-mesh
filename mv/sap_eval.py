"""Score the Sapiens2 cold-start fits (sap_fit.py output) against the Vicon markers per region, next to the single-camera SAM 3D Body meshes, the markerless multi-view mesh and the marker-guided mesh
on the same frames (same metric: nearest of the same ~6k subsampled vertices, mm).   usage: python sap_eval.py [variant suffix: '' or _sil] [size 0.4b] [trial TDB]  ->  tmp/sap_fit/sap_eval<suffix>.json"""
import csv, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
import trial_io as TI
suffix = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else ""; SIZE = sys.argv[2] if len(sys.argv) > 2 else "0.4b"; TI.TRIAL = sys.argv[3] if len(sys.argv) > 3 else "TDB"
SF = np.load(f"{TI.tdir('sap_fit')}/sapfit_{SIZE}_rot{suffix}.npz"); E2 = np.load(f"{TI.tdir('mv_fit', make=False)}/" + ("eval2.npz" if TI.TRIAL == "TDB" else "eval_trial.npz"))
rows = list(csv.reader(open(TI.vicon_csv()))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]; VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) * 1e-3 for r in rows[1:]}
def grp_of(n): return "chest" if (len(n) == 3 and n[0] in "RL" and n[1:].isdigit()) else ("star" if n.startswith("*") else "low")
en = [str(n) for n in E2["names"]]; groups = {g: [j for j, n in enumerate(en) if (g == "all" or grp_of(n) == g)] for g in ("chest", "low", "star", "all")}                      # marker groups by NAME (the CSV and the eval files list markers in different orders)
vidx = np.load(f"{TI.tdir('mv_sam3d', make=False)}/{int(SF['frames'][0]):04d}.npz")["vidx"]; out = dict(frames=[int(f) for f in SF["frames"]], start=[int(s) for s in SF["start"]], variants={})
fr_e2 = [int(f) for f in E2["frames"]]
for key in ("sv1", "sv2", "sv3", "sv4", "sv5", "mv", "fit"):
    if key not in E2.files: continue
    out["variants"][key] = {g: [float(np.nanmedian(E2[key][fr_e2.index(f)][ix])) if f in fr_e2 else None for f in out["frames"]] for g, ix in groups.items()}
out["variants"]["sapiens_cold"] = {g: [] for g in groups}
for n, f in enumerate(out["frames"]):
    mk = VIC[f + 1]; ok = ~np.isnan(mk).any(1); d = np.full(len(mk), np.nan); d[ok] = cKDTree(SF["X"][n][vidx].astype(np.float64)).query(mk[ok])[0] * 1000; dn = {nm: d[j] for j, nm in enumerate(names)}
    for g, ix in groups.items(): out["variants"]["sapiens_cold"][g].append(float(np.nanmedian([dn.get(en[j], np.nan) for j in ix])))
json.dump(out, open(f"{TI.tdir('sap_fit')}/sap_eval{suffix}.json", "w"), indent=1)
print("frames", out["frames"], "start cameras", out["start"])
print("%-14s | %s" % ("mesh", "  ".join("%-18s" % g for g in groups)))
for k, v in out["variants"].items(): print("%-14s | %s" % (k, "  ".join("%-18s" % ("%.1f (per frame %s)" % (np.nanmedian([x for x in v[g] if x is not None]), [round(x) for x in v[g] if x is not None][:6])) if any(x is not None for x in v[g]) else "-" for g in groups)))
