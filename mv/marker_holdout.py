"""How much of the marker-guided fit's advantage is real?  Fit the MHR body with only SOME of the Vicon markers and measure the distance of the markers it did NOT see to the fitted surface.

Schemes (each a complete 3-stage fit, same start: SAM 3D Body from camera T4 on the chosen frame):
  all           : every marker (the fit of reports 7-14; its distances are partly 'training error')
  no_chest      : the 16 chest-array markers held out
  no_lowbody    : pelvis / legs / feet markers held out
  no_armshead   : the 17 unlabelled arm / head markers held out
  fold1..fold5  : random 5-fold split, one fifth of all markers held out in turn
  joints12      : only 12 'joint-like' markers are used (ASIS x2, greater trochanter x2, lateral knee x2, lateral malleolus x2, toe x2, plus the two headband markers *56 and *69);
                  all other markers are held out  -> what a sparse set of 3D anchors, such as triangulated 2D keypoints, can pin down
  none          : SAM 3D Body (T4) as it comes (no markers at all)
Held-out distance = marker to the fitted surface (80k surface samples), mm.  usage (SAM3D env):  python marker_holdout.py [frame ...]     -> tmp/holdout/holdout_<frame>.json"""
import os, sys, json, time
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M, vicon_calib as V, marker_fit as MF
from sam3d_vs_vicon import surface
OUT = M.OUT + "/holdout"; os.makedirs(OUT, exist_ok=True); NAMES = list(V.NAMES)
chest = [n for n in NAMES if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]; star = [n for n in NAMES if n.startswith("*")]; low = [n for n in NAMES if n not in chest and n not in star]
JOINTS = ["LASIS", "RASIS", "LGRT", "RGRT", "LLKNE", "RLKNE", "LLML", "RLML", "LTOE", "RTOE", "*56", "*69"]
rng = np.random.default_rng(0); perm = rng.permutation(len(NAMES)); folds = [[NAMES[i] for i in perm[k::5]] for k in range(5)]
SCHEMES = {"all": [], "no_chest": chest, "no_lowbody": low, "no_armshead": star, **{f"fold{k + 1}": folds[k] for k in range(5)}, "joints12": [n for n in NAMES if n not in JOINTS]}


def main(frames):
    F = MF.Fitter(); res = {}
    for f in frames:
        Vfull = V.vicon(f) * 1e-3; ok_all = ~np.isnan(Vfull).any(1); o = MF.sam3d_raw(M.read("TDB", 4, f), 3); out = {}
        for name, held in SCHEMES.items():
            t0 = time.time(); Vf = Vfull.copy()
            for n in held: Vf[NAMES.index(n)] = np.nan
            if name == "all": hold_idx = []
            else: hold_idx = [NAMES.index(n) for n in held if ok_all[NAMES.index(n)]]
            info = F.fit(o, 3, Vf, verbose=False); verts = info["verts_after"]; tr = cKDTree(surface(verts, F.faces_np, 80000)); d = tr.query(Vfull[ok_all])[0] * 1000; idx = np.where(ok_all)[0]
            held_mask = np.isin(idx, hold_idx); seen = ~held_mask
            out[name] = dict(n_fitted=int((~np.isnan(Vf).any(1)).sum()), held_median=float(np.median(d[held_mask])) if held_mask.any() else None, held_p90=float(np.percentile(d[held_mask], 90)) if held_mask.any() else None, held_n=int(held_mask.sum()),
                             seen_median=float(np.median(d[seen])) if seen.any() else None, d=d.tolist(), idx=idx.tolist())
            print(f"frame {f} {name:12s}: fitted {out[name]['n_fitted']:2d} markers | held-out ({held_mask.sum():2d}): median {out[name]['held_median'] if out[name]['held_median'] is not None else float('nan'):6.1f} mm, p90 {out[name]['held_p90'] if out[name]['held_p90'] is not None else float('nan'):6.1f} | fitted-marker median {out[name]['seen_median'] if out[name]['seen_median'] is not None else float('nan'):5.1f} mm ({time.time() - t0:.0f}s)", flush=True)
            if name == "all":                                                                      # the starting mesh (SAM 3D Body, no markers)
                dn = cKDTree(surface(info["verts_before"], F.faces_np, 80000)).query(Vfull[ok_all])[0] * 1000
                out["none"] = dict(n_fitted=0, held_median=float(np.median(dn)), held_p90=float(np.percentile(dn, 90)), held_n=int(ok_all.sum()), seen_median=None, d=dn.tolist(), idx=idx.tolist())
                print(f"frame {f} none        : SAM 3D Body (T4) alone: all {ok_all.sum()} markers median {np.median(dn):.1f} mm, p90 {np.percentile(dn, 90):.1f}", flush=True)
        res[f] = out; json.dump(dict(names=NAMES, groups=dict(chest=chest, low=low, star=star), schemes={k: v for k, v in SCHEMES.items()}, results=res), open(f"{OUT}/holdout.json", "w"))
    return res


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [250])
