"""Orthographic side / front / top views of the three meshes (cold start yellow, multi-view blue, marker-guided green) and the Vicon markers (cyan) for one frame, to see by eye what is off.
usage: python sap_views_check.py [frame] [suffix]  -> tmp/sap_fit/views_<frame>.png"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trial_io as TI
TMP = TI.TMP; frame = int(sys.argv[1]) if len(sys.argv) > 1 else 250; suffix = sys.argv[2] if len(sys.argv) > 2 else ""
SF = np.load(f"{TMP}/sap_fit/sapfit_0.4b_rot{suffix}.npz"); T = np.load(TMP + "/track/track.npz", allow_pickle=True); tf = {int(f): i for i, f in enumerate(T["frames"])}; MV = np.load(TMP + "/mv_fit/mv_fit.npz"); mvf = [int(f) for f in MV["frames"]]
n = list(SF["frames"]).index(frame); X = SF["X"][n].astype(np.float64); R = T["vf"][tf[frame]].astype(np.float64); M = MV["X"][mvf.index(frame)].astype(np.float64); mk = T["mk"][tf[frame]].astype(np.float64); mk = mk[~np.isnan(mk).any(1)]
sub = slice(None, None, 6); fig, ax = plt.subplots(1, 3, figsize=(16, 5.4))
for a, (i, j, nm) in zip(ax, ((1, 2, "side view (Y along the body, Z up)"), (0, 2, "foot-end view (X across, Z up)"), (0, 1, "top view (X across, Y along the body)"))):
    for V, c, lab in ((M, "#5b8def", "multi-view (markerless)"), (R, "#3fb868", "marker-guided (reference)"), (X, "#e0a800", "Sapiens2 cold start")): a.scatter(V[sub, i], V[sub, j], s=1.6, c=c, alpha=.55, label=lab)
    a.scatter(mk[:, i], mk[:, j], s=26, facecolors="none", edgecolors="#06b6d4", linewidths=1.2, label="Vicon markers"); a.set_aspect("equal"); a.set_title(nm, fontsize=10)
    if i == 1: a.set_xlim(-0.35, 1.6); a.set_ylim(0.6, 1.25)
    if i == 0 and j == 2: a.set_xlim(-0.1, 0.55); a.set_ylim(0.75, 1.15)
    if j == 1: a.set_xlim(-0.1, 0.55); a.set_ylim(-0.35, 1.6)
ax[0].legend(markerscale=6, fontsize=8, loc="upper right"); plt.tight_layout(); os.makedirs(f"{TMP}/sap_fit", exist_ok=True); fig.savefig(f"{TMP}/sap_fit/views_{frame:04d}.png", dpi=110); print("saved")
