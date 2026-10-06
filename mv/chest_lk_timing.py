"""Cost of the fine layer: LK tracking (forward + backward) for ~300 chest points on one camera, and DLT triangulation of a frame, excluding video decoding."""
import os, json, time
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
VID = f"{ROOT}/CUTrial/MM17/TDB/Gyroflow/MMC17_TDB_T3_stabilized.mp4"; cap = cv2.VideoCapture(VID); gs = []
for f in range(0, 262):
    ok, im = cap.read()
    if f in (250, 253): gs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"]); c = CJ["cams"]["T3"]
P = np.load(TMP + "/chest_track/points.npz"); p0 = cv2.projectPoints(P["X"].reshape(-1, 1, 3), np.array(c["rvec"]), np.array(c["tvec"]), K, DIST)[0].reshape(-1, 2)
for scale in (1.0, 0.5, 0.25, 0.125):
    a, b = (cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else g for g in gs); W = int(round(41 * max(scale, 0.5)))
    q = (p0 * scale).astype(np.float32).reshape(-1, 1, 2); lk = dict(winSize=(W, W), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003)); ts = []
    for _ in range(30):
        t0 = time.perf_counter(); n, st, _ = cv2.calcOpticalFlowPyrLK(a, b, q, q.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk); cv2.calcOpticalFlowPyrLK(b, a, n, q.copy(), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **lk); ts.append(time.perf_counter() - t0)
    t1 = time.perf_counter(); r = cv2.resize(gs[0], None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1 else gs[0]; tr = time.perf_counter() - t1
    print("scale %.3f (%4dx%4d): LK fwd+bwd, %d points: median %.1f ms (min %.1f)   [grayscale resize %.1f ms]" % (scale, a.shape[1], a.shape[0], len(q), 1000 * np.median(ts), 1000 * np.min(ts), 1000 * tr))
# triangulation of one frame: 120 points x 5 cameras
cams = []
for i in range(1, 6):
    cc = CJ["cams"]["T%d" % i]; R = cv2.Rodrigues(np.array(cc["rvec"]))[0]; cams.append(K @ np.hstack([R, np.array(cc["tvec"])[:, None]]))
rng = np.random.default_rng(0); pts = rng.uniform(1000, 2500, (120, 5, 2)); t0 = time.perf_counter()
for n in range(120):
    A = []
    for k in range(5): A += [pts[n, k, 0] * cams[k][2] - cams[k][0], pts[n, k, 1] * cams[k][2] - cams[k][1]]
    np.linalg.svd(np.array(A))
print("triangulation of 120 points from 5 cameras (python loop DLT): %.1f ms" % (1000 * (time.perf_counter() - t0)))
