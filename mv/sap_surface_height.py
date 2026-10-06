"""Height of each mesh's anterior surface above the Vicon chest-array balls: for every ball (rows 1-4), the highest mesh vertex within 3 cm in the horizontal plane minus the ball height (the ball centre sits ~8 mm above the skin).
Positive = the mesh surface floats above the skin.   usage: python sap_surface_height.py [suffix]"""
import sys
import numpy as np
import trial_io as TI
TMP = TI.TMP; suffix = sys.argv[1] if len(sys.argv) > 1 else ""
SF = np.load(f"{TMP}/sap_fit/sapfit_0.4b_rot{suffix}.npz"); T = np.load(TMP + "/track/track.npz", allow_pickle=True); tf = {int(f): i for i, f in enumerate(T["frames"])}; MV = np.load(TMP + "/mv_fit/mv_fit.npz"); mvf = [int(f) for f in MV["frames"]]
names = [str(n) for n in T["names"]]; chest = [j for j, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]; row = {j: int(names[j][1]) for j in chest}
res = {"cold start": {r: [] for r in (1, 2, 3, 4)}, "multi-view": {r: [] for r in (1, 2, 3, 4)}, "marker-guided": {r: [] for r in (1, 2, 3, 4)}}
for n, f in enumerate(SF["frames"]):
    f = int(f); mk = T["mk"][tf[f]].astype(np.float64)
    for nm, V in (("cold start", SF["X"][n]), ("multi-view", MV["X"][mvf.index(f)]), ("marker-guided", T["vf"][tf[f]])):
        V = V.astype(np.float64)
        for j in chest:
            if np.isnan(mk[j]).any(): continue
            near = np.linalg.norm(V[:, :2] - mk[j, :2], axis=1) < 0.03
            if near.sum() > 3: res[nm][row[j]].append((V[near, 2].max() - (mk[j, 2] - 0.008)) * 1000)
print("anterior surface height above the skin under the chest-array balls (mm, median over balls and frames; + = mesh floats above the skin)")
print("%-15s %8s %8s %8s %8s" % ("mesh", "row 1", "row 2", "row 3", "row 4"))
for nm in res: print("%-15s %s" % (nm, " ".join("%+8.0f" % np.median(res[nm][r]) for r in (1, 2, 3, 4))))
import json
json.dump({nm: {str(r): [float(x) for x in v] for r, v in d.items()} for nm, d in res.items()}, open(f"{TMP}/sap_fit/surface_height{suffix}.json", "w"))
