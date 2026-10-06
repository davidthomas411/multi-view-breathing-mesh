"""Thoracoabdominal regional bias B from the MARKERLESS dense skin tracks (reports 17/18), next to the Vicon chest array, for one MMC17 trial (TDB or ADB).

Virtual markers: at the reference-frame position (x, y) of each of the 16 Vicon chest-array markers, the tracked skin points within RADIUS mm (in x, y) are pooled (median of their vertical
displacement per frame) = one markerless 'marker' per ball position.  Amplitude = peak-to-peak of the (lightly) smoothed vertical displacement, as in the draft paper; B = mean(rows 1-2) / mean(rows 3-4).
The Vicon amplitudes are computed on exactly the same (every 3rd) video frames with the same smoothing, so the comparison is like-for-like.
usage: python thoracoabdominal_markerless.py <TRIAL> [tag] [radius_mm]     -> tmp/thoracoabdominal/markerless_<TRIAL>_<tag>.json"""
import csv, json, os, re, sys
import numpy as np
from scipy.signal import savgol_filter
import trial_io as TI
ROOT, TMP = TI.ROOT, TI.TMP; OUT = TMP + "/thoracoabdominal"; os.makedirs(OUT, exist_ok=True); CHEST = re.compile(r"([LR])(\d)(\d)")
trial = sys.argv[1] if len(sys.argv) > 1 else "TDB"; tag = sys.argv[2] if len(sys.argv) > 2 else "m41mv_e33_dense"; RADIUS = float(sys.argv[3]) if len(sys.argv) > 3 else 45.0
CT = TI.tdir("chest_track", trial, make=False); E = np.load(f"{CT}/eval_s0.5_{tag}.npz", allow_pickle=True); keep = E["keep"]; Xr = E["Xref"]; D = E["D"]; frames = E["frames"]
rows = list(csv.reader(open(TI.vicon_csv(trial)))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]
VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}
chest = [n for n in names if CHEST.fullmatch(n)]; MK = np.array([[VIC[int(f) + 1][names.index(n)] for n in chest] for f in frames])            # (F, 16, 3) mm
ref = MK[0]; row_of = {n: int(CHEST.fullmatch(n)[2]) for n in chest}


def p2p_smooth(z, win=3):
    ok = ~np.isnan(z)
    if ok.sum() < 20: return np.nan
    zi = np.interp(np.arange(len(z)), np.where(ok)[0], z[ok]); zs = savgol_filter(zi, win, 1) if win >= 3 else zi; return float(zs.max() - zs.min())


amp_v, amp_m, npts = {}, {}, {}
for j, n in enumerate(chest):
    amp_v[n] = p2p_smooth(MK[:, j, 2] - MK[0, j, 2])
    near = np.where(np.linalg.norm(Xr[:, :2] - ref[j, :2], axis=1) < RADIUS)[0]; npts[n] = int(len(near))
    amp_m[n] = p2p_smooth(np.nanmedian(D[:, near, 2], axis=1)) if len(near) >= 2 else np.nan


def regional(amp):
    thor = [a for n, a in amp.items() if row_of[n] in (1, 2) and a == a and a > 0]; abd = [a for n, a in amp.items() if row_of[n] in (3, 4) and a == a and a > 0]
    r = [float(np.mean([a for n, a in amp.items() if row_of[n] == k and a == a and a > 0])) if any(row_of[n] == k and a == a and a > 0 for n, a in amp.items()) else float("nan") for k in (1, 2, 3, 4)]
    return dict(B=float(np.mean(thor) / np.mean(abd)) if thor and abd else float("nan"), n_thoracic=len(thor), n_abdominal=len(abd), thoracic=float(np.mean(thor)) if thor else float("nan"), abdominal=float(np.mean(abd)) if abd else float("nan"), rows=r)


# the same comparison with every marker that has a markerless counterpart (so the two B values use identical markers)
common = {n: amp_v[n] for n in chest if amp_m[n] == amp_m[n]}; res = dict(trial=trial, tag=tag, radius_mm=RADIUS, n_points=int(keep.sum()), vicon=regional(amp_v), markerless=regional(amp_m), vicon_common=regional(common),
                                                                         amp_vicon=amp_v, amp_markerless=amp_m, n_points_near=npts, row_of=row_of)
# per-point amplitude map (for the regional profile figure): amplitude of every tracked point (5-95 percentile of its vertical displacement) and its reference position
res["points"] = dict(xy=Xr[:, :2].round(1).tolist(), amp=[float(np.nanpercentile(D[:, k, 2], 95) - np.nanpercentile(D[:, k, 2], 5)) for k in range(D.shape[1])])
ymarks = {k: float(np.mean([ref[j, 1] for j, n in enumerate(chest) if row_of[n] == k])) for k in (1, 2, 3, 4)}; res["row_y_mm"] = ymarks
# ---- second estimator: spatial cells (30 mm) over the two regions, not tied to the ball positions; bootstrap over cells for a confidence interval
CELL = 30.0; xlo, xhi = ref[:, 0].min() - 40, ref[:, 0].max() + 40; y1, y2, y3, y4 = (ymarks[k] for k in (1, 2, 3, 4)); ymid = 0.5 * (y2 + y3); ytop, ybot = y1 + 40, y4 - 40
pa = np.array(res["points"]["amp"]); sel = (Xr[:, 0] > xlo) & (Xr[:, 0] < xhi) & (Xr[:, 1] > ybot) & (Xr[:, 1] < ytop); cells = {}
for k in np.where(sel)[0]: cells.setdefault((int(Xr[k, 0] // CELL), int(Xr[k, 1] // CELL)), []).append(pa[k])
cam = np.array([[np.median(v), (cx_ + .5) * CELL, (cy_ + .5) * CELL] for (cx_, cy_), v in cells.items()]); thor_c = cam[cam[:, 2] >= ymid, 0]; abd_c = cam[cam[:, 2] < ymid, 0]
rng = np.random.default_rng(0); bs = [np.mean(rng.choice(thor_c, len(thor_c))) / np.mean(rng.choice(abd_c, len(abd_c))) for _ in range(500)] if len(thor_c) > 2 and len(abd_c) > 2 else []
res["cells"] = dict(B=float(thor_c.mean() / abd_c.mean()) if len(thor_c) and len(abd_c) else float("nan"), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None, n_thoracic=int(len(thor_c)), n_abdominal=int(len(abd_c)), thoracic=float(thor_c.mean()) if len(thor_c) else float("nan"), abdominal=float(abd_c.mean()) if len(abd_c) else float("nan"), cell_mm=CELL)
# per-marker time series (mm, relative to the reference frame): Vicon ball and the virtual marker (median vertical displacement of the tracked skin points around it)
ser_m = {}
for j, n in enumerate(chest):
    near = np.where(np.linalg.norm(Xr[:, :2] - ref[j, :2], axis=1) < RADIUS)[0]
    ser_m[n] = [None if v != v else float(v) for v in np.nanmedian(D[:, near, 2], axis=1)] if len(near) >= 2 else None
res["series"] = dict(frames=[int(f) for f in frames], vicon={n: [float(v) for v in (MK[:, j, 2] - MK[0, j, 2])] for j, n in enumerate(chest)}, markerless=ser_m)
json.dump(res, open(f"{OUT}/markerless_{trial}_{tag}.json", "w"))
print("  Cell-based : B = %.3f (95%% CI %s) from %d thoracic / %d abdominal cells of %.0f mm; mean amplitude thoracic %.1f mm, abdominal %.1f mm" % (res["cells"]["B"], np.round(res["cells"]["ci95"], 2).tolist() if res["cells"]["ci95"] else "n/a", res["cells"]["n_thoracic"], res["cells"]["n_abdominal"], CELL, res["cells"]["thoracic"], res["cells"]["abdominal"]))
print(f"{trial} [{tag}]: {int(keep.sum())} tracked points; virtual markers with >=2 points within {RADIUS:.0f} mm: {len(common)}/16 (points per marker: {[npts[n] for n in chest]})")
print("  Vicon      : B = %.3f  (rows 1-4: %s mm; n thoracic/abdominal = %d/%d)" % (res["vicon"]["B"], np.round(res["vicon"]["rows"], 1).tolist(), res["vicon"]["n_thoracic"], res["vicon"]["n_abdominal"]))
print("  Vicon, same markers as the markerless set: B = %.3f (rows: %s)" % (res["vicon_common"]["B"], np.round(res["vicon_common"]["rows"], 1).tolist()))
print("  Markerless : B = %.3f  (rows 1-4: %s mm; n thoracic/abdominal = %d/%d)" % (res["markerless"]["B"], np.round(res["markerless"]["rows"], 1).tolist(), res["markerless"]["n_thoracic"], res["markerless"]["n_abdominal"]))
