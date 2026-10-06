"""DENSE seed points on the chest surface (tmp/chest_track/points_dense.npz): uniform area-weighted barycentric samples on the fitted mesh triangles that lie in the chest-array
footprint and face up.  'tri'+'bary' let the same seeds be placed on SAM 3D Body's mesh (same topology), i.e. the deployable seed."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000; MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04; OUTNAME = sys.argv[3] if len(sys.argv) > 3 else "points_dense"      # footprint margin around the chest array (m)
T = np.load(TMP + "/track/track.npz", allow_pickle=True); names = [str(n) for n in T["names"]]; faces = T["faces"]; v = T["vf"][0].astype(np.float64); mk = T["mk"][0].astype(np.float64)
chest = [i for i, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit() and not np.isnan(mk[i]).any()]; cm = mk[chest]
lo = cm[:, :2].min(0) - MARGIN; hi = cm[:, :2].max(0) + MARGIN; zc = np.median(cm[:, 2])
tri = v[faces]; cen = tri.mean(1); fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = np.linalg.norm(fn, axis=1) / 2; nz = fn[:, 2] / (np.linalg.norm(fn, axis=1) + 1e-12)
sel = np.where((cen[:, 0] > lo[0]) & (cen[:, 0] < hi[0]) & (cen[:, 1] > lo[1]) & (cen[:, 1] < hi[1]) & (nz > 0.6) & (np.abs(cen[:, 2] - zc) < 0.06))[0]
rng = np.random.default_rng(0); k = rng.choice(sel, N, p=area[sel] / area[sel].sum()); u, w = rng.random(N), rng.random(N); f = u + w > 1; u[f], w[f] = 1 - u[f], 1 - w[f]
bary = np.stack([1 - u - w, u, w], 1); X = (bary[:, :, None] * v[faces[k]]).sum(1)
print("dense seeds: %d points on %.0f cm2 of mesh surface (%.1f points/cm2, ~%.1f mm spacing)" % (N, area[sel].sum() * 1e4, N / (area[sel].sum() * 1e4), 10 / np.sqrt(N / (area[sel].sum() * 1e4))))
np.savez(TMP + f"/chest_track/{OUTNAME}.npz", X=X, tri=k, bary=bary, balls=cm, idx=np.arange(N))
