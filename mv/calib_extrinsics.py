"""Extrinsics for the MMC17 tripod cameras (T1-T6) from the DBECal checkerboard clip (Gyroflow-stabilised, same as the trials).
Shared intrinsics (all HERO11, same mode) + 6 board poses solved together; world frame = checkerboard frame (units of squares
until the Vicon registration fixes the metric scale)."""
import cv2, numpy as np, json, sys, os
ROOT = (os.environ.get("SGRT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VID = ROOT + "/CUTrial/MM17/DBECal/post_processing/Gyroflow/MMC17_DBECal_T{i}_stabilized.mp4"
PAT = (7, 5)     # inner corners (cols, rows) of the full board; some views only resolve a 6x5 sub-grid
OUTJ = ROOT + "/tmp/mmc17_calib.json"

def _gray(i, fi=15):
    c = cv2.VideoCapture(VID.format(i=i)); c.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = c.read()
    return f, cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

def detect_view(i):
    """Return {'full': (35,2) 7x5 grid or None, 'sub': (30,2) 6x5 grid or None, 'img': frame} in full-res pixels."""
    f, g = _gray(i); out = dict(full=None, sub=None, img=f)
    for pat, key in (((7, 5), "full"), ((6, 5), "sub")):
        for s in (0.5, 0.35, 1.0):
            gg = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else g
            ok, cr = cv2.findChessboardCornersSB(gg, pat, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if ok:
                cr = (cr.reshape(-1, 2) / s).astype(np.float32)
                cr = cv2.cornerSubPix(g, cr.reshape(-1, 1, 2), (15, 15), (-1, -1), (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 1e-3)).reshape(-1, 2)
                out[key] = cr; break
    return out

def options(d):
    """All (corner index subset into the 7x5 board grid, image points) hypotheses for one view."""
    idx = np.arange(35).reshape(5, 7)
    sub = d["sub"]; r = []   # the 7x5 detections are unreliable (up to 67 px lattice error); only the clean 6x5 sub-grid is used
    for off in (0, 1):
        sel = idx[:, off:off + 6].ravel()
        r += [(sel, sub), (sel[::-1], sub)]
    return r

if __name__ == "__main__":
    objp = np.zeros((PAT[0] * PAT[1], 3), np.float32); objp[:, :2] = np.mgrid[0:PAT[0], 0:PAT[1]].T.reshape(-1, 2)   # 1 unit = 1 square
    import itertools
    objp_all = np.zeros((35, 3), np.float32); objp_all[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
    det = {i: detect_view(i) for i in range(1, 7)}
    for i, d in det.items(): print("T%d: 7x5 found=%s  6x5 found=%s" % (i, d["full"] is not None, d["sub"] is not None))
    ids = [i for i in range(1, 7) if det[i]["sub"] is not None]
    W, H = 3840, 2160
    opts = [options(det[i]) for i in ids]
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
    best = None
    for combo, f0 in itertools.product(itertools.product(*[range(len(o)) for o in opts]), (1100.0, 1500.0, 2000.0)):
        if combo[0] % 2: continue                       # fix the global 180-degree gauge on the first view
        K0 = np.array([[f0, 0, W / 2], [0, f0, H / 2], [0, 0, 1.0]])
        ob = [objp_all[opts[v][c][0]] for v, c in enumerate(combo)]; ip = [opts[v][c][1] for v, c in enumerate(combo)]
        try: rms = cv2.calibrateCamera(ob, ip, (W, H), K0.copy(), np.zeros(5), flags=flags)[0]
        except cv2.error: continue
        if best is None or rms < best[0]: best = (rms, combo, f0)
    print("best corner-assignment RMS %.2f px  (combo %s, f0 %s)" % best)
    combo = best[1]; K0 = np.array([[best[2], 0, W / 2], [0, best[2], H / 2], [0, 0, 1.0]])
    objp_l = [objp_all[opts[v][c][0]] for v, c in enumerate(combo)]; pts = {i: opts[v][c][1] for v, (i, c) in enumerate(zip(ids, combo))}
    imgs = {i: det[i]["img"] for i in ids}
    rms, K, dist, rv, tv = cv2.calibrateCamera(objp_l, [pts[i] for i in ids], (W, H), K0.copy(), np.zeros(5), flags=flags)
    objp = objp_all   # for the reprojection report below use each view's own subset
    print("shared-intrinsics RMS reprojection: %.3f px   f=%.0f cx=%.0f cy=%.0f dist=%s" % (rms, K[0, 0], K[0, 2], K[1, 2], np.round(dist.ravel(), 4)))
    per = []
    for j, i in enumerate(ids):
        pr, _ = cv2.projectPoints(objp_l[j], rv[j], tv[j], K, dist); per.append(np.sqrt(((pr.reshape(-1, 2) - pts[i]) ** 2).sum(1).mean()))
    print("per-view RMS px:", np.round(per, 2))
    cams = {}
    for j, i in enumerate(ids):
        R, _ = cv2.Rodrigues(rv[j]); cams["T%d" % i] = dict(R=R.tolist(), t=tv[j].ravel().tolist(), centre=(-R.T @ tv[j].ravel()).tolist())
        print("T%d camera centre in board frame (squares):" % i, np.round(-R.T @ tv[j].ravel(), 1))
    json.dump(dict(K=K.tolist(), dist=dist.ravel().tolist(), size=[W, H], cams=cams, rms=rms, pattern=PAT), open(OUTJ, "w"), indent=1)
    # visual check of corner ordering: draw corner index 0 (red) and the 6x5 grid
    tiles = []
    for i in ids:
        v = imgs[i].copy(); [cv2.circle(v, tuple(p.astype(int)), 10, (0, 255, 0), -1) for p in pts[i]]
        p0 = tuple(pts[i][0].astype(int)); cv2.circle(v, p0, 40, (0, 0, 255), 12); cv2.putText(v, "T%d" % i, (60, 160), 0, 5, (255, 255, 255), 10)
        tiles.append(cv2.resize(v, (640, 360)))
    cv2.imwrite(ROOT + "/tmp/ical/board_corners.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
