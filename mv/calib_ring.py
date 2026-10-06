"""Rank the 6x5 sub-grid alignment hypotheses with the physical layout prior from the paper (Fig. 2a): the six tripod cameras form a
ring around the couch, all aimed at the subject.  Cost = radius spread + aimed-at-centre angle + height spread (units: board squares)."""
import cv2, numpy as np, itertools, json, sys
sys.path.insert(0, "."); import mmc17_sam3d as M
K, DIST = M.K, M.DIST
objp = np.zeros((35, 3), np.float32); objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
sub = list(np.load(M.OUT + "/board_sub.npz")["sub"])
subu = [cv2.undistortPoints(s.reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2) for s in sub]
idx = np.arange(35).reshape(5, 7); opts = []
for v in range(6):
    o = []
    for off in (0, 1):
        sel = idx[:, off:off + 6].ravel(); o += [(sel, subu[v]), (sel[::-1], subu[v])]
    opts.append(o)
def poses(combo):
    P = []
    for v, c in enumerate(combo):
        sel, ip = opts[v][c]; ok, rv, tv = cv2.solvePnP(objp[sel], ip, K, None, flags=cv2.SOLVEPNP_ITERATIVE); R = cv2.Rodrigues(rv)[0]; P.append((R, tv.ravel(), -R.T @ tv.ravel(), R[2]))
    return P
def cost(P):
    C = np.array([p[2] for p in P]); ax = np.array([p[3] for p in P])                  # centres, optical axes in board frame
    c0 = C.mean(0); rel = C - c0; r = np.linalg.norm(rel[:, :2], axis=1); tgt = np.array([c0[0], c0[1], 0.0]); tov = tgt - C   # aim at the couch plane (board z=0) under the ring centre
    ang = np.degrees(np.arccos(np.clip(np.sum(ax * tov / np.linalg.norm(tov, axis=1, keepdims=True), 1), -1, 1)))   # axis vs direction to the ring centre
    return r.std() / r.mean() + ang.mean() / 30 + C[:, 2].std() / abs(C[:, 2].mean()), r, ang, C
if __name__ == '__main__':
    res = sorted(((cost(poses(c))[0], c) for c in itertools.product(range(4), repeat=6) if c[0] % 2 == 0))
    print("lowest ring-prior cost (radius spread + aim angle + height spread):")
    for s, c in res[:6]: print("  %.3f  %s" % (s, c))
    print("  median over all combos %.2f; 2nd-best/best ratio %.2f" % (np.median([x[0] for x in res]), res[1][0] / res[0][0]))
    s, r, ang, C = cost(poses(res[0][1]))
    print("best: ring radius per camera (squares):", np.round(r, 1), " aim angle (deg):", np.round(ang, 0))
    print("camera centres in board frame (squares):\n", np.round(C, 1))
    json.dump([dict(cost=a, combo=list(b)) for a, b in res[:20]], open(M.OUT + "/calib_ring_ranked.json", "w"))
