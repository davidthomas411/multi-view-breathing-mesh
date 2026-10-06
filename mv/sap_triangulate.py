"""Triangulate Sapiens2 2D keypoints (sapiens2_run.py output) from the five calibrated cameras and compare lower-body landmarks with Vicon-derived joint proxies, next to SAM 3D Body's own
multi-view triangulation (j3d + cam_t projected, as in report 11).  Vicon proxies (mm): hip ~ greater trochanter, knee = midpoint of lateral/medial knee markers, ankle = midpoint of the malleoli,
big toe ~ 1st metatarsal head, small toe ~ 5th metatarsal head (joint centres and landmarks lie 2-6 cm from the skin markers, so absolute numbers are a few cm high for BOTH methods; the comparison is relative).
usage: python sap_triangulate.py <frame> [size] [rot]"""
import csv, json, os, sys
import cv2, numpy as np
import trial_io as TI
frame = int(sys.argv[1]) if len(sys.argv) > 1 else 250; SIZE = sys.argv[2] if len(sys.argv) > 2 else "0.4b"; ROT = sys.argv[3] if len(sys.argv) > 3 else "rot"
CJ = json.load(open(TI.calib_path())); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"]); R_, t_ = {}, {}
for c in range(1, 6): R_[c] = cv2.Rodrigues(np.array(CJ["cams"][f"T{c}"]["rvec"]))[0]; t_[c] = np.array(CJ["cams"][f"T{c}"]["tvec"])
P = {c: K @ np.hstack([R_[c], t_[c][:, None]]) for c in range(1, 6)}
def tri(obs, w, thr_px=25.0):
    """obs (5,K,2) ideal pixels (NaN = missing), w (5,K) weights -> X (K,3), per-keypoint reprojection residual (5,K)."""
    nk = obs.shape[1]; X = np.full((nk, 3), np.nan); res = np.full((5, nk), np.nan)
    for k in range(nk):
        use = [i for i in range(5) if not np.isnan(obs[i, k]).any() and w[i, k] > 0]
        while len(use) >= 2:
            A = []
            for i in use: c = i + 1; A += [w[i, k] * (obs[i, k, 0] * P[c][2] - P[c][0]), w[i, k] * (obs[i, k, 1] * P[c][2] - P[c][1])]
            Xh = np.linalg.svd(np.array(A))[2][-1]; Xk = Xh[:3] / Xh[3]; r = []
            for i in use: c = i + 1; Xc = R_[c] @ Xk + t_[c]; r.append(np.linalg.norm(K[:2, :2].diagonal() * Xc[:2] / Xc[2] + K[:2, 2] - obs[i, k]))
            if max(r) > thr_px and len(use) > 2: use.pop(int(np.argmax(r))); continue
            X[k] = Xk
            for i, rr in zip(use, r): res[i, k] = rr
            break
    return X, res
# ---- Sapiens2
und = np.full((5, 308, 2), np.nan); sc = np.zeros((5, 308))
for c in range(1, 6):
    p = f"{TI.tdir('sapiens2')}/{frame:04d}_T{c}_{SIZE}_{ROT}.npz"
    if os.path.exists(p): z = np.load(p); und[c - 1] = cv2.undistortPoints(z["kpts"].reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2); sc[c - 1] = z["scores"]
w = np.where(sc > 0.3, sc, 0); Xs, rs = tri(und[:, :70], w[:, :70])
# ---- SAM 3D Body (its own 3D skeleton + cam_t, projected through K and triangulated the same way)
zz = np.load(f"{TI.tdir('mv_sam3d')}/{frame:04d}.npz"); und2 = np.full((5, 70, 2), np.nan)
for v in range(5):
    if zz["ok"][v]: X = zz["j3d"][v] + zz["cam_t"][v]; uv = (K @ X.T).T; und2[v] = uv[:, :2] / uv[:, 2:3]
Xm, rm = tri(und2, np.ones((5, 70)))
# ---- Vicon proxies
rows = list(csv.reader(open(TI.vicon_csv()))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]; VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) * 1e-3 for r in rows[1:]}; mk = VIC[frame + 1]; g = lambda n: mk[names.index(n)]
prox = {"left_hip(9)": (9, g("LGRT")), "right_hip(10)": (10, g("RGRT")), "left_knee(11)": (11, 0.5 * (g("LLKNE") + g("LMKNE"))), "right_knee(12)": (12, 0.5 * (g("RLKNE") + g("RMKNE"))), "left_ankle(13)": (13, 0.5 * (g("LLML") + g("LMML"))),
        "right_ankle(14)": (14, 0.5 * (g("RLML") + g("RMML"))), "left_elbow(7)": (7, 0.5 * (g("*65") + g("*67"))), "right_elbow(8)": (8, 0.5 * (g("*64") + g("*63"))), "left_wrist(62)": (62, 0.5 * (g("*71") + g("*58"))), "right_wrist(41)": (41, 0.5 * (g("*66") + g("*60"))),
        "left_big_toe(15)": (15, g("LD1MT")), "left_small_toe(16)": (16, g("LD5MT")), "right_big_toe(18)": (18, g("RD1MT")), "right_small_toe(19)": (19, g("RD5MT"))}
head_ref = 0.5 * (g("*56") + g("*69")); es_head = np.linalg.norm(0.5 * (Xs[3] + Xs[4]) - head_ref) * 1000; em_head = np.linalg.norm(0.5 * (Xm[3] + Xm[4]) - head_ref) * 1000
print(f"frame {frame}  Sapiens2 {SIZE} ({ROT}): median reprojection residual of the triangulated body keypoints {np.nanmedian(rs[:, :21]):.1f} px; SAM3D skeleton {np.nanmedian(rm[:, :21]):.1f} px")
print("%-20s %12s %12s" % ("landmark", "Sapiens2 mm", "SAM3D mm")); es, em = [], []
for n, (k, ref) in prox.items():
    a = np.linalg.norm(Xs[k] - ref) * 1000 if not np.isnan(Xs[k]).any() else np.nan; b = np.linalg.norm(Xm[k] - ref) * 1000 if not np.isnan(Xm[k]).any() else np.nan; es.append(a); em.append(b); print("%-20s %12.0f %12.0f" % (n, a, b))
print("%-20s %12.0f %12.0f   (mean of the ears vs the headband markers)" % ("head centre", es_head, em_head)); es.append(es_head); em.append(em_head)
print("median over the landmarks: Sapiens2 %.0f mm | SAM3D triangulated %.0f mm" % (np.nanmedian(es), np.nanmedian(em)))
np.savez(f"{TI.tdir('sapiens2')}/tri_{frame:04d}_{SIZE}_{ROT}.npz", X=Xs, res=rs, Xsam=Xm)
lm = {}
for n, (k, ref) in prox.items():
    lm[n] = dict(sapiens=float(np.linalg.norm(Xs[k] - ref) * 1000) if not np.isnan(Xs[k]).any() else None, sam3d=float(np.linalg.norm(Xm[k] - ref) * 1000) if not np.isnan(Xm[k]).any() else None)
lm["head centre"] = dict(sapiens=float(es_head), sam3d=float(em_head))
json.dump(dict(frame=frame, size=SIZE, landmarks=lm, median_sapiens=float(np.nanmedian(es)), median_sam3d=float(np.nanmedian(em)), resid_sapiens_px=float(np.nanmedian(rs[:, :21])), resid_sam3d_px=float(np.nanmedian(rm[:, :21])),
           per_camera=[dict(cam=c + 1, body_score=float(sc[c, :21].mean()), hand_score=float(sc[c, 21:63].mean()), face_score=float(sc[c, 70:].mean()), resid_px=float(np.nanmedian(rs[c, :21]))) for c in range(5)]), open(f"{TI.tdir('sapiens2')}/tri_{frame:04d}_{SIZE}_{ROT}.json", "w"), indent=1)

for c in range(5):
    print("  T%d: mean score body kpts (0-20) %.2f, hands %.2f, face %.2f | median reprojection residual of body keypoints %.1f px" % (c + 1, sc[c, :21].mean(), sc[c, 21:63].mean(), sc[c, 70:].mean(), np.nanmedian(rs[c, :21])))
