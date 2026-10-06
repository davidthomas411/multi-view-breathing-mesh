"""Estimate per-camera frame offsets: per-view 2D joints are independent of fusion, so compute them once for
every frame, then find the integer offsets that minimise multi-view reprojection error."""
import sys, itertools, cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
from mvhmr import MultiViewHMR, BODY
from mvcal import CAMS, FRAME_DIR, OUT, load_cameras, triangulate_joints

ranges = [(60, 200), (380, 500), (600, 706)]
frames = [f for a, b in ranges for f in range(a, b)]
cache = f"{OUT}/j2d_all.npz"
try:
    d = np.load(cache); J = d["J"]; have = d["have"]
except Exception:
    J = np.full((706, 6, 70, 2), np.nan, np.float32); have = np.zeros(706, bool)
if not have[frames].all():
    mv = MultiViewHMR(); pool = ThreadPoolExecutor(6)
    for f in frames:
        if have[f]: continue
        imgs = list(pool.map(lambda c: cv2.cvtColor(cv2.imread(f"{FRAME_DIR}/{c}/{f:04d}.jpg"), cv2.COLOR_BGR2RGB), CAMS))
        mv.prev_X = None
        r = mv.step(imgs, mode="detect")
        J[f] = r["j2d"]; have[f] = True
    np.savez(cache, J=J, have=have)
cams = [load_cameras()[c] for c in CAMS]

def score(off, fs):
    e = []
    for f in fs:
        idx = [f + o for o in off]
        if min(idx) < 0 or max(idx) >= 706: continue
        uv = np.stack([J[idx[v], v] for v in range(6)])[:, BODY]
        if np.isnan(uv).any(): continue
        _, err, _ = triangulate_joints(cams, uv)
        e.append(np.nanmedian(err))
    return np.median(e) if e else np.inf

fs = [f for f in frames][::4]
off = [0] * 6
print("zero offsets:", round(score(off, fs), 2), "px")
for it in range(2):
    for v in range(1, 6):
        best = min(range(-8, 9), key=lambda o: score(off[:v] + [o] + off[v + 1:], fs))
        off[v] = best
    print("iter", it, "offsets", off, "score", round(score(off, fs), 2))
np.save(f"{OUT}/offsets.npy", np.array(off))
