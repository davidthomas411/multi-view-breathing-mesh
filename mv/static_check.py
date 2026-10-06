"""Control: does the video itself shift (Gyroflow stabilisation, tripod flex, compression) by an amount that could fake or mask breathing?
Track static background corners (floor / room, outside the body and couch region) with the SAME LK settings and report their apparent displacement over the trial.
usage: python static_check.py <cam 1-5>     -> tmp/chest_track/static_T<cam>.json"""
import os, sys, json, time
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
cam = int(sys.argv[1]); VID = f"{ROOT}/CUTrial/MM17/TDB/Gyroflow/MMC17_TDB_T{cam}_stabilized.mp4"
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"]); c = CJ["cams"]["T%d" % cam]; rvec = np.array(c["rvec"]); tvec = np.array(c["tvec"])
T = np.load(TMP + "/track/track.npz", allow_pickle=True)
body = cv2.projectPoints(T["vf"][0].astype(np.float64).reshape(-1, 1, 3), rvec, tvec, K, DIST)[0].reshape(-1, 2)
x0, y0 = body.min(0) - 250; x1, y1 = body.max(0) + 250                              # exclude the body (and the couch around it) generously
cap = cv2.VideoCapture(VID); F0, FEND, STEP = 250, 1120, 3; f = 0; ref = None; disp = []; lk = dict(winSize=(41, 41), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))
while f <= FEND:
    ok, im = cap.read()
    if not ok: break
    if f >= F0 and (f - F0) % STEP == 0:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        if ref is None:
            ref = g; mask = np.full(g.shape, 255, np.uint8); mask[int(max(y0, 0)):int(y1), int(max(x0, 0)):int(x1)] = 0
            p0 = cv2.goodFeaturesToTrack(g, 400, 0.02, 40, mask=mask, blockSize=9).astype(np.float32); prev = p0.copy(); disp.append(np.zeros((len(p0), 2), np.float32)); good_all = np.ones(len(p0), bool)
        else:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(ref, g, p0, prev.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk); back, st2, _ = cv2.calcOpticalFlowPyrLK(g, ref, nxt, p0.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk)
            good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back[:, 0] - p0[:, 0], axis=1) < 1.0); good_all &= good; prev = np.where(good[:, None, None], nxt, prev); disp.append((nxt[:, 0] - p0[:, 0]).astype(np.float32))
    f += 1
D = np.array(disp)[:, good_all, :]; n = D.shape[1]
common = np.median(D, axis=1)                                                       # common-mode shift (what a global warp would add), px
res = dict(cam=cam, n_points=int(n), common_mode_rms_px=float(np.sqrt(np.mean(common ** 2))), common_mode_max_px=float(np.abs(common).max()), common_mode_p2p_px=[float(np.ptp(common[:, 0])), float(np.ptp(common[:, 1]))],
           per_point_rms_px=float(np.median(np.sqrt(np.mean((D - common[:, None, :]) ** 2, axis=(0, 2))))))
res["common_mode_rms_mm_at_0.77mm_per_px"] = res["common_mode_rms_px"] * 0.77
print(json.dumps(res)); json.dump(res, open(f"{TMP}/chest_track/static_T{cam}.json", "w"))
