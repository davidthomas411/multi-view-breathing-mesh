"""Where does the markerless mesh error come from?  Per-marker error split into DEPTH (along the camera's viewing ray) and LATERAL (image plane), for every single-camera SAM 3D Body mesh,
the markerless multi-view mesh and the marker-guided mesh (TDB).

Correspondences: each Vicon marker is attached to the vertex of the marker-guided mesh (first frame) that is closest to it; the same vertex index on another mesh (same MHR topology) is the
'same anatomical place'.  For camera v:   e = V_cam(vertex) - M_cam(marker),  depth = e . r  (r = unit ray to the marker),  lateral = |e - depth r|.
(Only every 3rd vertex was saved per view, so the attachment is to the nearest saved vertex: <= ~1 cm discretisation.)
Writes tmp/mv_fit/err_decompose.npz and prints a table.   python err_decompose.py"""
import json, os
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
T = np.load(TMP + "/track/track.npz", allow_pickle=True); MV = np.load(TMP + "/mv_fit/mv_fit.npz"); names = [str(n) for n in T["names"]]; tf = {int(f): i for i, f in enumerate(T["frames"])}
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json"))["cams"]; RT = [(cv2.Rodrigues(np.array(CJ["T%d" % i]["rvec"]))[0], np.array(CJ["T%d" % i]["tvec"])) for i in range(1, 6)]
vidx = np.load(f"{TMP}/mv_sam3d/0250.npz")["vidx"]; mk0 = T["mk"][tf[250]].astype(np.float64); vf0 = T["vf"][tf[250]].astype(np.float64)[vidx]; okm0 = ~np.isnan(mk0).any(1)
att = np.array([int(np.argmin(np.linalg.norm(vf0 - mk0[m], axis=1))) if okm0[m] else -1 for m in range(len(names))]); print("attachment distance to the nearest saved vertex: median %.1f mm" % (np.median(np.linalg.norm(vf0[att[okm0]] - mk0[okm0], axis=1)) * 1000))
chest = [j for j, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]; star = [j for j, n in enumerate(names) if n.startswith("*")]; low = [j for j in range(len(names)) if j not in chest and j not in star]
groups = {"chest array": chest, "pelvis/legs/feet": low, "head/arms": star}
frames = [int(f) for f in MV["frames"]]; M = len(names)
E = {k: np.full((len(frames), M, 2), np.nan) for k in ["T1", "T2", "T3", "T4", "T5", "MV", "fit"]}                 # [..., 0] = |depth| along the ray of the camera in which it is evaluated, [..., 1] = lateral
S = {k: np.full((len(frames), M), np.nan) for k in E}                                                               # signed depth error (positive = mesh farther from the camera than the marker)
cam_for_mv = 3                                                                                                      # MV and fit are evaluated in the rays of T4 (the best single camera) and, separately, averaged over cameras below
EA = {k: np.full((len(frames), 5, M, 2), np.nan) for k in ("MV", "fit")}
for n, f in enumerate(frames):
    z = np.load(f"{TMP}/mv_sam3d/{f:04d}.npz"); i = tf[f]; mk = T["mk"][i].astype(np.float64); okm = ~np.isnan(mk).any(1) & (att >= 0)
    for v in range(5):
        R, t = RT[v]; Mc = mk[okm] @ R.T + t; r = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
        if z["ok"][v]:
            Vc = (z["verts"][v] + z["cam_t"][v])[att[okm]]; e = Vc - Mc; dz = np.sum(e * r, 1); lat = np.linalg.norm(e - dz[:, None] * r, axis=1)
            E["T%d" % (v + 1)][n, okm] = np.stack([np.abs(dz), lat], 1) * 1000; S["T%d" % (v + 1)][n, okm] = dz * 1000
        for key, X in (("MV", MV["X"][n]), ("fit", T["vf"][i])):
            Vc = X.astype(np.float64)[vidx][att[okm]] @ R.T + t; e = Vc - Mc; dz = np.sum(e * r, 1); lat = np.linalg.norm(e - dz[:, None] * r, axis=1); EA[key][n, v, okm] = np.stack([np.abs(dz), lat], 1) * 1000
            if v == 3: E[key][n, okm] = np.stack([np.abs(dz), lat], 1) * 1000; S[key][n, okm] = dz * 1000
for key in EA: E[key] = np.nanmedian(EA[key], axis=1)                                                              # MV/fit: median over the five viewing directions
print("%-17s %-8s | %s" % ("region", "mesh", "median |depth| (mm), median lateral (mm), 90th pct 3-D error (mm)"))
for g, ix in groups.items():
    for k in ["T1", "T2", "T3", "T4", "T5", "MV", "fit"]:
        a = E[k][:, ix, :]; d3 = np.sqrt(a[..., 0] ** 2 + a[..., 1] ** 2)
        print("%-17s %-8s | depth %6.1f | lateral %6.1f | p90 3-D %6.1f | signed depth bias %+6.1f" % (g, k, np.nanmedian(a[..., 0]), np.nanmedian(a[..., 1]), np.nanpercentile(d3, 90), np.nanmedian(S[k][:, ix]) if k in S else np.nan))
np.savez(TMP + "/mv_fit/err_decompose.npz", frames=np.array(frames), names=np.array(names), **{"E_" + k: v for k, v in E.items()}, **{"S_" + k: v for k, v in S.items()})
