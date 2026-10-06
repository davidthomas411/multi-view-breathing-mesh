"""Chest-texture tracking, several (seed variant, resolution) combinations in ONE decode pass of a camera's video.

usage: python chest_lk_multi.py <cam 1-5> <variant:scale> [<variant:scale> ...]
variants:  fit      seed = fitted mesh vertices (the surface itself)
           offNN    fitted mesh points raised by NN mm along +Z (floating NN mm above the true surface)
           sam3d    the same vertices taken from SAM 3D Body's own mesh (what a deployed system would have)
scale:     1.0 = 4K, 0.5 = 1080p, 0.25 = 960x540, 0.125 = 480x270
Output per combo: tmp/chest_track/lk_T<cam>_s<scale>_<variant>.npz   (pts (F,N,2) in 4K px coords, ok (F,N), score (N), ball_d (N), frames)
Tracking is identical to chest_lk.py: pyramidal LK reference->frame seeded by the previous frame, forward-backward check, balls masked (32 px)."""
import os, sys, json, time
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
cam = int(sys.argv[1]); combos = [(a.split(":")[0], float(a.split(":")[1])) for a in sys.argv[2:]]
VID = f"{ROOT}/CUTrial/MM17/TDB/Gyroflow/MMC17_TDB_T{cam}_stabilized.mp4"
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"]); c = CJ["cams"]["T%d" % cam]
rvec = np.array(c["rvec"]); tvec = np.array(c["tvec"]); P = np.load(TMP + "/chest_track/points.npz"); balls = P["balls"]
T = np.load(TMP + "/track/track.npz", allow_pickle=True)
def seed(variant):
    if variant == "fit": return P["X"]
    if variant.startswith("off"): return P["X"] + np.array([0, 0, int(variant[3:]) / 1000.0])
    if variant == "sam3d": return T["vb"][0][P["idx"]].astype(np.float64)
    raise ValueError(variant)
def proj(Xv): return cv2.projectPoints(Xv.reshape(-1, 1, 3), rvec, tvec, K, DIST)[0].reshape(-1, 2)
bp = proj(balls); F0, FEND, STEP = 250, 1120, 3; frames = list(range(F0, FEND + 1, STEP))
state = {}
for variant, scale in combos:
    p0 = proj(seed(variant)); W = int(round(41 * max(scale, 0.5)))
    state[(variant, scale)] = dict(p0=p0, W=W, ref=None, q0=None, prev=None, pts=[], ok=[],
                                   lk=dict(winSize=(W, W), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003)))
cap = cv2.VideoCapture(VID); f = 0; t0 = time.time(); n_done = 0
while f <= FEND:
    ok, im = cap.read()
    if not ok: break
    if f >= F0 and (f - F0) % STEP == 0:
        g4 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY); cache = {1.0: g4}
        for (variant, scale), S in state.items():
            if scale not in cache: cache[scale] = cv2.resize(g4, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            g = cache[scale]
            if S["ref"] is None:
                S["ref"] = g; S["q0"] = (S["p0"] * scale).astype(np.float32).reshape(-1, 1, 2); S["prev"] = S["q0"].copy()
                e = cv2.cornerMinEigenVal(g, 5, ksize=3); r = int(round(15 * scale)); q = S["q0"][:, 0]
                S["score"] = np.array([e[max(int(y) - r, 0):int(y) + r + 1, max(int(x) - r, 0):int(x) + r + 1].mean() if 0 <= x < g.shape[1] and 0 <= y < g.shape[0] else 0 for x, y in q])
                S["ball_d"] = np.min(np.linalg.norm(S["p0"][:, None, :] - bp[None], axis=2), axis=1)
                inside = (S["p0"][:, 0] > 40) & (S["p0"][:, 0] < 3800) & (S["p0"][:, 1] > 40) & (S["p0"][:, 1] < 2120)
                S["pts"].append((S["q0"][:, 0] / scale).astype(np.float32)); S["ok"].append(inside & (S["ball_d"] > 32))
            else:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(S["ref"], g, S["q0"], S["prev"].copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **S["lk"])
                back, st2, _ = cv2.calcOpticalFlowPyrLK(g, S["ref"], nxt, S["q0"].copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **S["lk"])
                err = np.linalg.norm(back[:, 0] - S["q0"][:, 0], axis=1); good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (err < 1.0 * max(scale, 0.5))
                S["prev"] = np.where(good[:, None, None], nxt, S["prev"]); S["pts"].append((nxt[:, 0] / scale).astype(np.float32)); S["ok"].append(good)
        n_done += 1
        if n_done % 40 == 0: print("T%d frame %d (%d/%d) %.0fs" % (cam, f, n_done, len(frames), time.time() - t0), flush=True)
    f += 1
for (variant, scale), S in state.items():
    np.savez(f"{TMP}/chest_track/lk_T{cam}_s{scale}_{variant}.npz", pts=np.array(S["pts"]), ok=np.array(S["ok"]), score=S["score"], ball_d=S["ball_d"], frames=np.array(frames[:len(S["pts"])]))
print("T%d done: %d combos, %d frames, %.0fs" % (cam, len(state), n_done, time.time() - t0))
