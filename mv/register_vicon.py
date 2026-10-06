"""Vicon -> board-frame registration for MMC17 TDB (coarse, from SAM 3D Body multi-view hips/knees/ankles vs Vicon ASIS/knee/ankle markers).
Constraint: the checkerboard lies level on the couch, so Vicon Z (up) <-> board -z.  Unknowns: yaw, scale (mm per square), translation."""
import os
import sys, glob, csv, json, numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M
from mvcal import triangulate_joints
CSV = M.ROOT + "/CUTrial/Trials_marker_positions_by_image_frame_all_cases/MMC17_TDB_marker_positions_by_image_frame_wide.csv"
rows = list(csv.reader(open(CSV))); hdr = rows[0]; names = [c[:-2] for c in hdr[10:] if c.endswith("_X")]
data = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}
def mk(fr, *n):
    d = data[fr + 1]; return np.nanmean([d[names.index(x)] for x in n], 0)
# (MHR70 joint index, Vicon marker(s)): left/right hip~ASIS, knee~mid(lat,med knee markers), ankle~mid(malleoli)
PAIRS = [(9, ("LASIS",)), (10, ("RASIS",)), (11, ("LLKNE", "LMKNE")), (12, ("RLKNE", "RMKNE")), (13, ("LLML", "LMML")), (14, ("RLML", "RMML"))]
CAMS_USE = [0, 1, 2, 3, 4]       # T1-T5 (T6 inconsistent)
A, V, F = [], [], []
for p in sorted(glob.glob(M.OUT + "/mmc17_sam3d/TDB_*.npz")):
    fr = int(p[-8:-4])
    if fr < 250: continue                                            # skip the leg-tap sync movement at the start
    z = np.load(p); j2d = M.obs2d(z["j3d"], z["cam_t"]); vs = [i for i in CAMS_USE if not np.isnan(j2d[i][M.S.BODY]).any()]
    if len(vs) < 4: continue
    X, err, _ = triangulate_joints([M.CAMS17[i] for i in vs], j2d[vs][:, M.S.BODY]); Xfull = np.full((70, 3), np.nan); Xfull[M.S.BODY] = X
    pts_b = np.array([Xfull[j] for j, _ in PAIRS]); pts_v = np.array([mk(fr, *m) for _, m in PAIRS])
    if np.isnan(pts_b).any() or np.isnan(pts_v).any(): continue
    A.append(pts_b); V.append(pts_v); F.append(fr)
A = np.array(A); V = np.array(V); print("frames used:", len(F), F[:3], "...")
Ab = np.nanmedian(A, 0); Vb = np.nanmedian(V, 0)            # time-robust mean pose (steady supine)
def register(s_fixed=None):
    Q = Vb * np.array([1, -1, -1.0]); cq = Q[:, :2].mean(0); ca = Ab[:, :2].mean(0); qc = Q[:, :2] - cq; ac = Ab[:, :2] - ca
    zq = qc[:, 0] + 1j * qc[:, 1]; za = ac[:, 0] + 1j * ac[:, 1]; c = np.sum(np.conj(zq) * za) / np.sum(np.abs(zq) ** 2)
    s = abs(c) if s_fixed is None else s_fixed; th = np.angle(c)
    R2 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    xy = (Q[:, :2] - cq) @ R2.T * s + ca; tz = np.mean(Ab[:, 2] + s * Vb[:, 2]); pred = np.c_[xy, -s * Vb[:, 2] + tz]
    return dict(s=s, yaw=th, cq=cq, ca=ca, tz=tz, pred=pred, res_mm=np.linalg.norm(pred - Ab, axis=1) / s)
SQ = float(sys.argv[1]) if len(sys.argv) > 1 else 130.0
free = register(None); fixed = register(1.0 / SQ)
print("free scale: %.1f mm/square, RMS %.0f mm" % (1 / free["s"], np.sqrt(np.mean(free["res_mm"] ** 2))))
print("fixed %.0f mm squares: yaw %.1f deg, residual per landmark %s mm, RMS %.0f mm" % (SQ, np.degrees(fixed["yaw"]), np.round(fixed["res_mm"]), np.sqrt(np.mean(fixed["res_mm"] ** 2))))
dev = np.linalg.norm(A - Ab, axis=2) * SQ; print("frame-to-frame scatter of triangulated landmarks (130 mm squares): median %.0f mm" % np.median(dev))
np.savez(M.OUT + "/vicon_registration.npz", Ab=Ab, Vb=Vb, A=A, V=V, frames=F, sq_mm=SQ, free_s=free["s"], free_pred=free["pred"], free_res=free["res_mm"],
         fixed_s=fixed["s"], fixed_pred=fixed["pred"], fixed_res=fixed["res_mm"], fixed_yaw=fixed["yaw"], fixed_cq=fixed["cq"], fixed_ca=fixed["ca"], fixed_tz=fixed["tz"])
json.dump(dict(square_mm=SQ, yaw_rad=float(fixed["yaw"]), cq=fixed["cq"].tolist(), ca=fixed["ca"].tolist(), tz=float(fixed["tz"]), vicon_flip="diag(1,-1,-1)", residual_mm=fixed["res_mm"].tolist()), open(M.OUT + "/vicon_to_board.json", "w"), indent=1)
