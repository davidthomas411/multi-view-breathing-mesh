"""Resolve the per-view 6x5 sub-grid alignment (offset 0/1 x 180-deg flip) with the marker balls as cross-view correspondences."""
import cv2, numpy as np, itertools, json, sys, os
sys.path.insert(0, "."); import mmc17_sam3d as M, marker_blobs as MB
K, DIST = M.K, M.DIST; Ki = np.linalg.inv(K)
objp = np.zeros((35, 3), np.float32); objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
sub = list(np.load(M.OUT + "/board_sub.npz")["sub"])
subu = [cv2.undistortPoints(s.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2) for s in sub]
idx = np.arange(35).reshape(5, 7); opts = []
for v in range(6):
    o = []
    for off in (0, 1):
        sel = idx[:, off:off + 6].ravel(); o += [(sel, subu[v]), (sel[::-1], subu[v])]
    opts.append(o)
FR = list(range(40, 1100, 60)); B = {}
for f in FR:
    B[f] = []
    for i in range(6):
        b = MB.blobs(cv2.cvtColor(M.read("TDB", i + 1, f), cv2.COLOR_RGB2GRAY))
        B[f].append(cv2.undistortPoints(b.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2) if len(b) else np.zeros((0, 2)))
print("blobs per view (mean over frames):", np.round([np.mean([len(B[f][i]) for f in FR]) for i in range(6)], 1))
PAIRS = list(itertools.combinations(range(6), 2))
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
def poses(combo):
    P = []
    for v, c in enumerate(combo):
        sel, ip = opts[v][c]; ok, rv, tv = cv2.solvePnP(objp[sel], ip, K, None, flags=cv2.SOLVEPNP_ITERATIVE); P.append((cv2.Rodrigues(rv)[0], tv.ravel()))
    return P
def score(P):
    hits = tot = 0
    for a, b in PAIRS:
        Rab = P[b][0] @ P[a][0].T; tab = P[b][1] - Rab @ P[a][1]; F = Ki.T @ (skew(tab) @ Rab) @ Ki
        for f in FR:
            pa, pb = B[f][a], B[f][b]
            if len(pa) == 0 or len(pb) == 0: continue
            l = np.c_[pa, np.ones(len(pa))] @ F.T; d = np.abs(l @ np.c_[pb, np.ones(len(pb))].T) / np.hypot(l[:, :1], l[:, 1:2])
            hits += int((d.min(1) < 2.5).sum()); tot += len(pa)
    return hits / max(tot, 1)
res = sorted(((score(poses(c)), c) for c in itertools.product(range(4), repeat=6) if c[0] % 2 == 0), reverse=True)
print("hit rate (fraction of balls with a partner within 2.5 px of the epipolar line) -- top 6 / median over all combos:")
for s, c in res[:6]: print("  %.3f  %s" % (s, c))
print("  median %.3f" % np.median([r[0] for r in res]))
json.dump(dict(combo=list(res[0][1]), score=res[0][0], runner_up=res[1][0]), open(M.OUT + "/calib_resolved.json", "w"))
