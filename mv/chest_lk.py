"""Per-camera 2D tracking of chest-surface points (skin/fabric texture) through the MMC17 TDB video.

For one camera: decode the 4K video sequentially, take frame 250 as the reference, project the candidate 3D chest points into it, and track each point to every
3rd frame with pyramidal Lucas-Kanade (reference -> frame, seeded by the previous frame's result, forward-backward check).  The balls are masked: points
whose reference patch overlaps a Vicon-marker ball are dropped.   usage: python chest_lk.py <cam 1-5> <scale: 1.0 = 4K, 0.5 = 1080p>
Output: tmp/chest_track/lk_T<cam>_s<scale>.npz  (pts (F,N,2) in 4K pixel coords, ok (F,N), score (N), frames)"""
import os, sys, json, time
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
cam = int(sys.argv[1]); scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
VID = f"{ROOT}/CUTrial/MM17/TDB/Gyroflow/MMC17_TDB_T{cam}_stabilized.mp4"
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"]); c = CJ["cams"]["T%d" % cam]
rvec = np.array(c["rvec"]); tvec = np.array(c["tvec"]); P = np.load(TMP + "/chest_track/points.npz"); X = P["X"]; balls = P["balls"]
def proj(Xv): return cv2.projectPoints(Xv.reshape(-1, 1, 3), rvec, tvec, K, DIST)[0].reshape(-1, 2)            # distorted pixel coords (the frames are not undistorted)
p0 = proj(X); bp = proj(balls)
F0, FEND, STEP = 250, 1120, 3; frames = list(range(F0, FEND + 1, STEP)); W = int(round(41 * max(scale, 0.5)))
cap = cv2.VideoCapture(VID); f = 0; ref = None; prev = None; out_pts = []; out_ok = []; t0 = time.time()
lk = dict(winSize=(W, W), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))
while f <= FEND:
    ok, im = cap.read()
    if not ok: break
    if f >= F0 and (f - F0) % STEP == 0:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if scale != 1.0: g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if ref is None:
            ref = g; q0 = (p0 * scale).astype(np.float32).reshape(-1, 1, 2); prev = q0.copy()
            # texture score: smaller eigenvalue of the structure tensor in a 31 px (4K) window at each point
            e = cv2.cornerMinEigenVal(ref, 5, ksize=3); r = int(round(15 * scale))
            score = np.array([e[max(int(y) - r, 0):int(y) + r + 1, max(int(x) - r, 0):int(x) + r + 1].mean() if 0 <= x < ref.shape[1] and 0 <= y < ref.shape[0] else 0 for x, y in q0[:, 0]])
            ball_d = np.min(np.linalg.norm(p0[:, None, :] - bp[None], axis=2), axis=1)               # px (4K) to the nearest ball
            inside = (p0[:, 0] > 40) & (p0[:, 0] < 3800) & (p0[:, 1] > 40) & (p0[:, 1] < 2120)
            out_pts.append((q0[:, 0] / scale).astype(np.float32)); out_ok.append(inside & (ball_d > 32))
        else:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref, g, q0, prev.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(g, ref, nxt, q0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            err = np.linalg.norm(back[:, 0] - q0[:, 0], axis=1); good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (err < 1.0 * max(scale, 0.5))
            prev = np.where(good[:, None, None], nxt, prev); out_pts.append((nxt[:, 0] / scale).astype(np.float32)); out_ok.append(good)
        if len(out_pts) % 40 == 0: print("T%d s%.1f frame %d (%d/%d) %.0fs, tracked %.0f%%" % (cam, scale, f, len(out_pts), len(frames), time.time() - t0, 100 * np.mean(out_ok[-1])), flush=True)
    f += 1
np.savez(f"{TMP}/chest_track/lk_T{cam}_s{scale}.npz", pts=np.array(out_pts), ok=np.array(out_ok), score=score, ball_d=ball_d, frames=np.array(frames[:len(out_pts)]))
print("T%d s%.1f done: %d frames, %.0fs" % (cam, scale, len(out_pts), time.time() - t0))
