"""Triangulate the per-camera chest-texture tracks (chest_lk.py) in the Vicon frame and compare the recovered breathing signal with the Vicon chest array.

usage: python chest_eval.py <scale>     (1.0 = 4K tracks, 0.5 = 1080p tracks)
Outputs tmp/chest_track/eval_s<scale>.json and .npz.
The reference for each point is its own 3D position at frame 250; the breathing signal is the vertical (Vicon Z = anterior, supine) displacement from that reference.
No marker position enters the tracks (the balls are only masked out), and Vicon is only used for the comparison."""
import os, sys, json, csv
import cv2, numpy as np
import trial_io as TI
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; CT = TI.tdir("chest_track")
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0; variant = sys.argv[2] if len(sys.argv) > 2 else "fit"
tag = f"s{scale}" + ("" if variant == "fit" and scale in (1.0, 0.5) and not os.path.exists(f"{CT}/lk_T1_s{scale}_fit.npz") else f"_{variant}")
MINF = float(os.environ.get("MINFRAC", 0.5)); MINCAM = int(os.environ.get("MINCAM", 3))                              # optional: fraction of frames a point must be triangulated in; minimum cameras per point
lktag = tag; tag += ("" if MINF == 0.5 else "_f%d" % round(MINF * 100)) + ("" if MINCAM == 3 else "_c%d" % MINCAM)          # lktag = the tracks read, tag = the evaluation written
CJ = json.load(open(TI.calib_path())); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"])
cams = {}
for i in range(1, 6):
    c = CJ["cams"]["T%d" % i]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; t = np.array(c["tvec"]); cams[i] = (R, t, K @ np.hstack([R, t[:, None]]))
L = {i: np.load(f"{CT}/lk_T{i}_{lktag}.npz") for i in range(1, 6)}
frames = L[1]["frames"]; F = len(frames); N = L[1]["pts"].shape[1]
U = np.zeros((5, F, N, 2)); OK = np.zeros((5, F, N), bool)
for k, i in enumerate(range(1, 6)):
    pts = L[i]["pts"].astype(np.float64); U[k] = cv2.undistortPoints(pts.reshape(-1, 1, 2), K, DIST, P=K).reshape(F, N, 2); OK[k] = L[i]["ok"] & L[i]["ok"][0][None, :]      # reference-frame validity (ball mask, inside image)
    OK[k] &= (L[i]["score"] > np.percentile(L[i]["score"], 10))[None, :]                                                                     # drop the lowest-texture tenth
Pm = np.array([cams[i][2] for i in range(1, 6)]); Rm = np.array([cams[i][0] for i in range(1, 6)]); tm = np.array([cams[i][1] for i in range(1, 6)])
def tri_batch(mask, uv):
    """Batched DLT. mask (5,n) bool = which cameras to use per point, uv (5,n,2) undistorted pixels -> X (n,3)."""
    n = mask.shape[1]; A = np.zeros((n, 10, 4))
    for k in range(5):
        m = mask[k].astype(float)[:, None]
        A[:, 2 * k] = m * (uv[k][:, 0:1] * Pm[k][2][None, :] - Pm[k][0][None, :]); A[:, 2 * k + 1] = m * (uv[k][:, 1:2] * Pm[k][2][None, :] - Pm[k][1][None, :])
    X = np.linalg.svd(A)[2][:, -1, :]; return X[:, :3] / X[:, 3:4]
def reproj(X, uv, mask):
    r = np.zeros((5, X.shape[0]))
    for k in range(5):
        Xc = X @ Rm[k].T + tm[k]; p = (Xc @ K.T)[:, :2] / Xc[:, 2:3]; r[k] = np.where(mask[k], np.linalg.norm(p - uv[k], axis=1), 0.0)
    return r
X3 = np.full((F, N, 3), np.nan); nview = np.zeros((F, N), int)
for fi in range(F):
    mask = OK[:, fi, :].copy(); good = mask.sum(0) >= MINCAM
    if not good.any(): continue
    m = mask[:, good]; uv = U[:, fi, good]; X = tri_batch(m, uv); rep = reproj(X, uv, m)
    drop = (rep.max(0) > 3.0) & (m.sum(0) > 3)                                     # drop the worst view once if its reprojection exceeds 3 px
    if drop.any():
        w = rep.argmax(0); m2 = m.copy(); m2[w[drop], np.where(drop)[0]] = False; X2 = tri_batch(m2, uv); X[drop] = X2[drop]; m = np.where(drop[None, :], m2, m)
    X3[fi, good] = X; nview[fi, good] = m.sum(0)
valid_frac = np.mean(~np.isnan(X3[:, :, 0]), axis=0); keep = (valid_frac > MINF) & ~np.isnan(X3[0, :, 0])
print('valid-frame fraction per candidate point: percentiles 10/25/50/75/90 =', np.round(np.percentile(valid_frac, [10, 25, 50, 75, 90]), 2), '| per-camera reference validity:', [int(OK[k, 0].sum()) for k in range(5)])
print("points with >= %d views in >%d%% of frames and at the reference: %d of %d (mean cameras per point-frame %.2f)" % (MINCAM, int(MINF * 100), keep.sum(), N, nview[:, keep][nview[:, keep] > 0].mean()))
D = (X3[:, keep, :] - X3[0:1, keep, :]) * 1000                                       # displacement from the reference, mm (Vicon frame)
# ---- Vicon chest array
D_ = TI.vicon_csv()
rows = list(csv.reader(open(D_))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]
VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}
chest = [n for n in names if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]
MK = np.array([[VIC[int(f) + 1][names.index(n)] for n in chest] for f in frames])               # (F,16,3) mm (the Vicon CSV is already in mm)
MKd = MK - MK[0:1]
s_vic = np.nanmean(MKd[:, :, 2], axis=1); s_trk = np.nanmedian(D[:, :, 2], axis=1)
def detr(x):
    t = np.arange(len(x)); ok = ~np.isnan(x); return x - np.polyval(np.polyfit(t[ok], x[ok], 2), t)
a, b = detr(s_vic), detr(s_trk); ok = ~np.isnan(a) & ~np.isnan(b)
r = np.corrcoef(a[ok], b[ok])[0, 1]; slope = np.polyfit(a[ok], b[ok], 1)[0]; p2p = lambda x: np.nanpercentile(x, 95) - np.nanpercentile(x, 5)
res = dict(scale=scale, variant=variant, n_points=int(keep.sum()), n_candidates=int(N), r=float(r), slope=float(slope), p2p_vicon=float(p2p(a)), p2p_track=float(p2p(b)),
           mae_mm=float(np.mean(np.abs(a[ok] - b[ok]))), rmse_mm=float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))), mae_scaled_mm=float(np.mean(np.abs(a[ok] - (b[ok] - np.mean(b[ok])) / slope))))
dd_ = (b[ok] - a[ok]); res["ba_bias_mm"] = float(np.mean(dd_)); res["ba_loa_mm"] = [float(np.mean(dd_) - 1.96 * np.std(dd_)), float(np.mean(dd_) + 1.96 * np.std(dd_))]; res["p90_abs_err_mm"] = float(np.percentile(np.abs(dd_), 90))
lags = {l: float(np.corrcoef(a[ok][max(0, l):len(a[ok]) + min(0, l)], b[ok][max(0, -l):len(b[ok]) + min(0, -l)])[0, 1]) for l in range(-6, 7)}; res["lag_corr"] = lags
# ---- regional check: tracked points near each chest marker vs that marker (xy distance < 45 mm), per marker
Xref = X3[0, keep, :] * 1000; per = []
for j, n in enumerate(chest):
    m = MKd[:, j, 2]
    if np.isnan(m).any(): continue
    near = np.where(np.linalg.norm(Xref[:, :2] - MK[0, j, :2], axis=1) < 45)[0]
    if len(near) < 2: continue
    tz = np.nanmedian(D[:, near, 2], axis=1); aa, bb = detr(m), detr(tz); o = ~np.isnan(bb)
    per.append(dict(marker=n, n_pts=int(len(near)), r=float(np.corrcoef(aa[o], bb[o])[0, 1]), mae_mm=float(np.mean(np.abs(aa[o] - bb[o]))), p2p_vicon=float(p2p(aa)), p2p_track=float(p2p(bb))))
def jitter(Dz):
    k = 5; out = []
    for n in range(Dz.shape[1]):
        z = Dz[:, n]; ok_ = ~np.isnan(z)
        if ok_.sum() < 40: continue
        zi = np.interp(np.arange(len(z)), np.where(ok_)[0], z[ok_]); sm = np.convolve(zi, np.ones(k) / k, "same"); out.append(np.sqrt(np.mean((zi - sm)[k:-k] ** 2)))
    return float(np.median(out)) if out else float("nan")
res["point_jitter_mm"] = jitter(D[:, :, 2]); res["per_marker"] = per
print("overall: r = %.3f | slope %.2f | peak-to-peak Vicon %.1f mm vs tracked %.1f mm | MAE %.2f mm (after best scale %.2f mm) | RMSE %.2f mm" % (r, slope, res["p2p_vicon"], res["p2p_track"], res["mae_mm"], res["mae_scaled_mm"], res["rmse_mm"]))
if per: print("per chest marker (n=%d): median r %.2f, median MAE %.2f mm, median amplitude ratio %.2f" % (len(per), np.median([p["r"] for p in per]), np.median([p["mae_mm"] for p in per]), np.median([p["p2p_track"] / p["p2p_vicon"] for p in per])))
print("Bland-Altman: bias %.2f mm, 95%% limits of agreement [%.2f, %.2f] mm; 90th percentile |error| %.2f mm" % (res["ba_bias_mm"], res["ba_loa_mm"][0], res["ba_loa_mm"][1], res["p90_abs_err_mm"]))
print("per-point jitter (RMS of vertical displacement about its 5-frame average, median over points): %.2f mm" % res["point_jitter_mm"])
print("lag with the best correlation (frames of 3 video frames each):", max(lags, key=lags.get), "->", round(max(lags.values()), 3))
json.dump(res, open(f"{CT}/eval_{tag}.json", "w"), indent=1)
np.savez(f"{CT}/eval_{tag}.npz", frames=frames, s_vic=s_vic, s_trk=s_trk, a=a, b=b, D=D, Xref=Xref, nview=nview, keep=keep)
