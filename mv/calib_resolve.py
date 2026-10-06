"""Resolve the per-view board alignment ambiguity (6x5 sub-grid offset 0/1 x 180-degree flip) using cross-view geometry:
for every option combination, per-view board poses -> pairwise epipolar geometry -> Sampson error on RANSAC-consistent
SIFT matches of the static room (collected from several trial frames). The correct combination should drive the error to ~1 px."""
import cv2, numpy as np, itertools, json, sys, os
sys.path.insert(0, "."); import mmc17_sam3d as M, calib_extrinsics as C
K, DIST = M.K, M.DIST; Ki = np.linalg.inv(K)
objp = np.zeros((35, 3), np.float32); objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

# 1. sub-grid detections (cached)
cache = M.OUT + "/board_sub.npz"
if os.path.exists(cache): sub = list(np.load(cache)["sub"])
else:
    sub = [C.detect_view(i)["sub"] for i in range(1, 7)]; np.savez(cache, sub=np.array(sub))
idx = np.arange(35).reshape(5, 7)
opts = []
for v in range(6):
    o = []
    for off in (0, 1):
        sel = idx[:, off:off + 6].ravel(); o += [(sel, sub[v]), (sel[::-1], sub[v])]
    opts.append(o)

# 2. correspondences between camera pairs: SIFT on trial frames, keep matches consistent with a free fundamental matrix
sift = cv2.SIFT_create(8000); bf = cv2.BFMatcher(); PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 3), (1, 4), (2, 5), (0, 2), (1, 3), (3, 5), (0, 4), (1, 5), (2, 4)]
corr = {p: [] for p in PAIRS}
for f in (60, 400, 800, 1000):
    gs = [cv2.cvtColor(M.read("TDB", i + 1, f), cv2.COLOR_RGB2GRAY) for i in range(6)]; feats = []
    for g in gs:
        k, d = sift.detectAndCompute(cv2.resize(g, None, fx=.5, fy=.5), None); feats.append((np.array([p.pt for p in k]) * 2, d))
    for a, b in PAIRS:
        m = [x for x, y in bf.knnMatch(feats[a][1], feats[b][1], k=2) if x.distance < 0.75 * y.distance]
        if len(m) < 15: continue
        pa = np.float32([feats[a][0][x.queryIdx] for x in m]); pb = np.float32([feats[b][0][x.trainIdx] for x in m])
        ua = cv2.undistortPoints(pa.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2); ub = cv2.undistortPoints(pb.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2)
        F, mask = cv2.findFundamentalMat(ua, ub, cv2.FM_RANSAC, 1.0, 0.9999)
        if mask is None: continue
        inl = mask.ravel() > 0; corr[(a, b)].append((ua[inl], ub[inl]))
corr = {p: (np.vstack([c[0] for c in v]), np.vstack([c[1] for c in v])) for p, v in corr.items() if v}
print("pairs with correspondences:", {f"T{a+1}-T{b+1}": len(v[0]) for (a, b), v in corr.items()})

def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
def poses(combo):
    out = []
    for v, c in enumerate(combo):
        sel, ip = opts[v][c]; ok, rv, tv = cv2.solvePnP(objp[sel], ip.astype(np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        out.append((cv2.Rodrigues(rv)[0], tv.ravel()))
    return out
def score(P):
    errs = []
    for (a, b), (xa, xb) in corr.items():
        Rab = P[b][0] @ P[a][0].T; tab = P[b][1] - Rab @ P[a][1]; F = Ki.T @ (skew(tab) @ Rab) @ Ki
        A = np.c_[xa, np.ones(len(xa))]; B = np.c_[xb, np.ones(len(xb))]; Fx = A @ F.T; Ftx = B @ F
        s = np.sqrt(np.sum(B * Fx, 1) ** 2 / (Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2)); errs.append(np.minimum(s, 30))
    return float(np.median(np.concatenate(errs)))

res = []
for combo in itertools.product(range(4), repeat=6):
    if combo[0] % 2: continue                        # global 180-degree gauge
    res.append((score(poses(combo)), combo))
res.sort(); print("best combos (median Sampson px, option index per camera [offset*2+flip]):")
for s, c in res[:5]: print("  %.2f px  %s" % (s, c))
print("median over all combos: %.1f px" % np.median([r[0] for r in res]))
best = res[0][1]; P = poses(best)
json.dump(dict(combo=list(best), score=res[0][0]), open(M.OUT + "/calib_resolved.json", "w"))
np.savez(M.OUT + "/calib_resolved_poses.npz", R=np.array([p[0] for p in P]), t=np.array([p[1] for p in P]))
