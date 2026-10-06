"""Candidate chest-surface points for the texture-tracking feasibility test (tmp/chest_track/points.npz).
Points are mesh vertices (fitted mesh at frame 250, Vicon frame, metres) inside the chest-array footprint, facing up.  The ORACLE here is the mesh/ROI:
in a deployed system the SAM 3D Body mesh would give the ROI.  Ball locations (from the Vicon markers) are only used to MASK the balls out of the tracked patches."""
import os, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; os.makedirs(TMP + "/chest_track", exist_ok=True)
T = np.load(TMP + "/track/track.npz", allow_pickle=True); names = [str(n) for n in T["names"]]; faces = T["faces"]
v = T["vf"][0].astype(np.float64); mk = T["mk"][0].astype(np.float64)
chest = [i for i, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit() and not np.isnan(mk[i]).any()]
cm = mk[chest]; lo = cm[:, :2].min(0) - 0.04; hi = cm[:, :2].max(0) + 0.04
fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]); vn = np.zeros_like(v)
for k in range(3): np.add.at(vn, faces[:, k], fn)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
zc = np.median(cm[:, 2]); sel = np.where((v[:, 0] > lo[0]) & (v[:, 0] < hi[0]) & (v[:, 1] > lo[1]) & (v[:, 1] < hi[1]) & (vn[:, 2] > 0.6) & (np.abs(v[:, 2] - zc) < 0.06))[0]
sel = sel[::2]                                                         # ~mesh spacing 1.7 cm -> every other vertex
print("chest footprint: %d vertices candidate, using %d; chest markers: %d" % (((v[:, 0] > lo[0]) & (v[:, 0] < hi[0]) & (v[:, 1] > lo[1]) & (v[:, 1] < hi[1])).sum(), len(sel), len(chest)))
np.savez(TMP + "/chest_track/points.npz", X=v[sel], N=vn[sel], balls=cm, idx=sel)
