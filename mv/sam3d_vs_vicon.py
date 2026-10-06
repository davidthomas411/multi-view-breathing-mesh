"""Benchmark SAM 3D Body meshes against the Vicon markers (real ground truth, mm).
Per camera: SAM3D mesh (camera frame) -> Vicon frame via the marker-based extrinsics -> distance of every Vicon marker to the mesh surface.
A perfect fit sits at ~8 mm (the reflective-ball radius).   usage: python sam3d_vs_vicon.py"""
import os
import sys, json, numpy as np, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M, vicon_calib as V, sam3d_mv as S
K = V.K; C = json.load(open(M.OUT + "/mmc17_vicon_calib.json"))["cams"]
NAMES = V.NAMES
def pose(i): c = C["T%d" % (i + 1)]; return cv2.Rodrigues(np.array(c["rvec"]))[0], np.array(c["tvec"])
REGION = {"torso (chest array R/L 11-42)": [n for n in NAMES if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()], "pelvis (ASIS/PSIS/GRT/ILCR)": [n for n in NAMES if n[1:] in ("ASIS", "GRT", "ILCR", "SPSK", "IPSK")],
          "thigh/shank (clusters, knee)": [n for n in NAMES if any(k in n for k in ("SATH", "IATH", "SPTH", "IPTH", "KNE", "SASK", "IASK", "TTUB"))], "ankle/foot": [n for n in NAMES if any(k in n for k in ("ML", "TOE", "D1MT", "D5MT", "TIP"))]}
def surface(verts, faces, n=150000, rng=np.random.default_rng(0)):
    tri = verts[faces]; a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1); k = rng.choice(len(faces), n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n); f = u + v > 1; u[f], v[f] = 1 - u[f], 1 - v[f]
    return tri[k, 0] + u[:, None] * (tri[k, 1] - tri[k, 0]) + v[:, None] * (tri[k, 2] - tri[k, 0])
if __name__ == "__main__":
    est, _ = S.models(); faces = np.asarray(est.faces)
    frames = [260, 340, 420, 500, 580]; per = {}; allm = []
    for f in frames:
        Vf = V.vicon(f) * 1e-3; ok = ~np.isnan(Vf).any(1)
        for i in range(5):
            img = M.read("TDB", i + 1, f); d = S.run_view(img, K, keep_mesh=True)
            if d is None: continue
            Xc = d["verts"] + d["cam_t"]; R, t = pose(i); Xv = (Xc - t) @ R       # camera -> Vicon frame (metres)
            dist = cKDTree(surface(Xv, faces))[0] if False else cKDTree(surface(Xv, faces)).query(Vf[ok])[0] * 1000
            for n, dd in zip(np.array(NAMES)[ok], dist): per.setdefault(n, []).append(dd)
            allm.append(np.median(dist))
        print("frame", f, "done", flush=True)
    print("\n== SAM 3D Body single-view mesh vs Vicon markers (distance marker->mesh surface, mm; ~8 mm = perfect) ==")
    print("all markers, all cameras/frames: median %.0f mm   p90 %.0f mm" % (np.median(np.concatenate([v for v in per.values()])), np.percentile(np.concatenate([v for v in per.values()]), 90)))
    for r, ns in REGION.items():
        v = np.concatenate([per[n] for n in ns if n in per]) if any(n in per for n in ns) else np.array([np.nan]); print("  %-34s median %4.0f mm   p90 %4.0f mm   (%d markers)" % (r, np.median(v), np.percentile(v, 90), sum(n in per for n in ns)))
    json.dump({n: float(np.median(v)) for n, v in per.items()}, open(M.OUT + "/sam3d_vs_vicon_per_marker.json", "w"))
