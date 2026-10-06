"""Skin-only chest tracking: every reflective ball is PAINTED OUT of every frame (inpainted) before tracking.

Ball positions come from the Vicon markers projected into the camera at each frame (missing markers interpolated in time); a 30 px (4K) disc around each is inpainted
(Telea) on a local crop.  Points whose tracking window would touch an inpainted disc are discarded (distance > 30 + half window + 3 px), so the tracker never sees a ball,
its dark base, or the inpainted patch.   usage: python chest_lk_masked.py <cam 1-5> <seed:scale:window4k> [...]      e.g.  fit:1.0:41 fit:0.5:41 fit:1.0:21 sam3d:0.5:41
window4k = tracking window in 4K pixels (scaled with the image, so 41 means the same physical patch at every resolution).
Output: tmp/chest_track/lk_T<cam>_s<scale>_m<window4k>[<seed if not fit>].npz   (same format as chest_lk.py)"""
import os, sys, json, csv, time
import cv2, numpy as np
import trial_io as TI
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; CT = TI.tdir("chest_track")
cam = int(sys.argv[1]); combos = []
for a in sys.argv[2:]:
    q = a.split(":"); combos.append((q[0], float(q[1]), int(q[2]), (float(q[3]) if len(q) > 3 and q[3] else None), (int(q[4]) if len(q) > 4 and q[4] else 1), (float(q[5]) if len(q) > 5 and q[5] else float(os.environ.get("MINEIG", 1e-4)))))     # seed:scale:win4k[:exclusion px[:inpaint 0/1[:minEigThreshold]]]
PTS = os.environ.get("PTS", "points"); OUTSUF = os.environ.get("OUTSUF", "")
R_MASK = 30
CJ = json.load(open(TI.calib_path())); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"]); c = CJ["cams"]["T%d" % cam]; rvec = np.array(c["rvec"]); tvec = np.array(c["tvec"])
P = np.load(f"{CT}/{PTS}.npz"); T = np.load(TMP + "/track/track.npz", allow_pickle=True)
def proj(Xv): return cv2.projectPoints(Xv.reshape(-1, 1, 3), rvec, tvec, K, DIST)[0].reshape(-1, 2)
def seed(v):
    if v == "fit": return P["X"]
    if v == "mv":                                                                                       # the markerless multi-view SAM 3D Body mesh (mv_fit.py), first frame: no Vicon in the seeds
        mvx = np.load(TI.tdir("mv_fit") + "/mv_fit.npz")["X"][0].astype(np.float64); return (P["bary"][:, :, None] * mvx[T["faces"][P["tri"]]]).sum(1)
    vb0 = T["vb"][0].astype(np.float64)
    if "tri" in P.files: return (P["bary"][:, :, None] * vb0[T["faces"][P["tri"]]]).sum(1)          # same barycentric seeds placed on SAM 3D Body's mesh
    return vb0[P["idx"]]
F0, FEND = TI.frame_range(); STEP = 3; frames = list(range(F0, FEND + 1, STEP))
# ---- Vicon marker positions per tracked frame (mm -> m), gaps interpolated
rows = list(csv.reader(open(TI.vicon_csv())))
VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}
MK = np.array([VIC[f + 1] for f in frames]) * 1e-3                                    # (F, M, 3) metres
for m in range(MK.shape[1]):
    ok = ~np.isnan(MK[:, m, 0])
    if 3 <= ok.sum() < len(ok):
        for a in range(3): MK[:, m, a] = np.interp(np.arange(len(ok)), np.where(ok)[0], MK[ok, m, a])
def inpaint_balls(g, centres):
    """Paint out each ball in place (Telea on a local crop)."""
    H, W = g.shape; h = R_MASK + 18
    for x, y in centres:
        xi, yi = int(round(x)), int(round(y))
        if xi < -h or xi > W + h or yi < -h or yi > H + h: continue
        x0, y0, x1, y1 = max(xi - h, 0), max(yi - h, 0), min(xi + h + 1, W), min(yi + h + 1, H)
        if x1 - x0 < 8 or y1 - y0 < 8: continue
        crop = np.ascontiguousarray(g[y0:y1, x0:x1]); m = np.zeros(crop.shape, np.uint8); cv2.circle(m, (xi - x0, yi - y0), R_MASK, 255, -1)
        g[y0:y1, x0:x1] = cv2.inpaint(crop, m, 5, cv2.INPAINT_TELEA)
state = {}
for seedv, scale, w4k, excl, inp, meig in combos:
    W = int(round(w4k * scale)) | 1; W = max(W, 7); p0 = proj(seed(seedv))
    state[(seedv, scale, w4k, excl, inp, meig)] = dict(p0=p0, W=W, excl=excl, inp=inp, ref=None, pts=[], ok=[], lk=dict(winSize=(W, W), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003), minEigThreshold=meig), thr=max(1.0 * scale, 0.35), half4k=(w4k / 2.0))
cap = TI.Capture(cam); f = 0; t0 = time.time(); n_done = 0
while f <= FEND:
    ok, im = cap.read()
    if not ok: break
    if f >= F0 and (f - F0) % STEP == 0:
        k = (f - F0) // STEP; g_raw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); bp = proj(MK[k][~np.isnan(MK[k]).any(1)]); g_inp = g_raw.copy()
        if any(S["inp"] for S in state.values()): inpaint_balls(g_inp, bp)
        cache = {}
        for key, S in state.items():
            scale = key[1]; ck = (scale, S["inp"])
            if ck not in cache: base = g_inp if S["inp"] else g_raw; cache[ck] = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else base
            g = cache[ck]
            if S["ref"] is None:
                S["ref"] = g; S["q0"] = (S["p0"] * scale).astype(np.float32).reshape(-1, 1, 2); S["prev"] = S["q0"].copy()
                e = cv2.cornerMinEigenVal(g, 5, ksize=3); r = int(round(15 * scale)); q = S["q0"][:, 0]
                S["score"] = np.array([e[max(int(y) - r, 0):int(y) + r + 1, max(int(x) - r, 0):int(x) + r + 1].mean() if 0 <= x < g.shape[1] and 0 <= y < g.shape[0] else 0 for x, y in q])
                S["ball_d"] = np.min(np.linalg.norm(S["p0"][:, None, :] - bp[None], axis=2), axis=1)
                inside = (S["p0"][:, 0] > 60) & (S["p0"][:, 0] < 3780) & (S["p0"][:, 1] > 60) & (S["p0"][:, 1] < 2100)
                S["exclusion"] = S["excl"] if S["excl"] is not None else (R_MASK + S["half4k"] + 3)
                S["pts"].append((S["q0"][:, 0] / scale).astype(np.float32)); S["ok"].append(inside & (S["ball_d"] > S["exclusion"]))
            else:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(S["ref"], g, S["q0"], S["prev"].copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **S["lk"])
                back, st2, _ = cv2.calcOpticalFlowPyrLK(g, S["ref"], nxt, S["q0"].copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **S["lk"])
                err = np.linalg.norm(back[:, 0] - S["q0"][:, 0], axis=1); good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (err < S["thr"])
                S["prev"] = np.where(good[:, None, None], nxt, S["prev"]); S["pts"].append((nxt[:, 0] / scale).astype(np.float32)); S["ok"].append(good)
        n_done += 1
        if n_done % 40 == 0: print("T%d frame %d (%d/%d) %.0fs" % (cam, f, n_done, len(frames), time.time() - t0), flush=True)
    f += 1
for (seedv, scale, w4k, excl, inp, meig), S in state.items():
    tag = f"s{scale}_m{w4k}" + ("" if seedv == "fit" else seedv) + ("" if excl is None else f"_e{int(excl)}") + ("" if inp else "_vis") + ("" if meig == 1e-4 else "_eig%d" % round(-np.log10(meig))) + OUTSUF
    np.savez(f"{CT}/lk_T{cam}_{tag}.npz", pts=np.array(S["pts"]), ok=np.array(S["ok"]), score=S["score"], ball_d=S["ball_d"], frames=np.array(frames[:len(S["pts"])]))
    print("T%d %s: reference-valid points %d of %d (exclusion %.0f px from any ball)" % (cam, tag, int(S["ok"][0].sum()), len(S["p0"]), S["exclusion"]))
print("T%d done: %d frames, %.0fs" % (cam, n_done, time.time() - t0))
