"""Per-segment camera offsets (frames) vs distance from the last frame. Uses cached per-view 2D joints (tmp/j2d_all.npz)."""
import numpy as np
from mvhmr import BODY
from mvcal import OUT, load_cameras, CAMS, triangulate_joints
d = np.load(f"{OUT}/j2d_all.npz"); J = d["J"]
cams = [load_cameras()[c] for c in CAMS]
def score(off, fs):
    e = []
    for f in fs:
        idx = [f + o for o in off]
        if min(idx) < 0 or max(idx) >= 706: continue
        uv = np.stack([J[idx[v], v] for v in range(6)])[:, BODY]
        if np.isnan(uv).any(): continue
        e.append(np.nanmedian(triangulate_joints(cams, uv)[1]))
    return np.median(e) if e else np.inf
res = {}
for name, (a, b) in {"60-200": (60, 200), "380-500": (380, 500), "600-700": (600, 700)}.items():
    fs = list(range(a, b, 3)); off = [0] * 6
    z = score(off, fs)
    for it in range(2):
        for v in range(1, 6):
            off[v] = min(range(-8, 9), key=lambda o: score(off[:v] + [o] + off[v + 1:], fs))
    print(f"frames {name}: offsets {off}  error {z:.1f} -> {score(off, fs):.1f} px", flush=True)
