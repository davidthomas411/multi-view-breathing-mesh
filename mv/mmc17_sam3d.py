"""SAM 3D Body on the MMC17 TDB trial (6 tripod cameras, 3840x2160 Gyroflow videos) with the ECal-derived calibration.
World frame = checkerboard frame (units: squares) until registered to Vicon.   usage: python mmc17_sam3d.py [start stop step]"""
import os, sys, json, time, cv2, numpy as np
sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvcal import Camera, triangulate_joints, OUT
import sam3d_mv as S

ROOT = (os.environ.get("SGRT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VID = ROOT + "/CUTrial/MM17/{trial}/Gyroflow/MMC17_{trial}_T{i}_stabilized.mp4"
CAL = json.load(open(OUT + "/mmc17_calib.json"))
K = np.array(CAL["K"]); DIST = np.array(CAL["dist"])
CAMS17 = [Camera("T%d" % i, K, np.array(CAL["cams"]["T%d" % i]["R"]), np.array(CAL["cams"]["T%d" % i]["t"]), 3840, 2160, DIST) for i in range(1, 7)]
CACHE = OUT + "/mmc17_sam3d"; os.makedirs(CACHE, exist_ok=True)

def read(trial, i, f):
    import trial_io as TI
    c = TI.Capture(i, trial); c.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = c.read(); c.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None

def views(trial, f):
    p = f"{CACHE}/{trial}_{f:04d}.npz"
    if os.path.exists(p): z = np.load(p); return z["j3d"], z["cam_t"], z["box"]
    j3d = np.full((6, 70, 3), np.nan, np.float32); ct = np.full((6, 3), np.nan, np.float32); bx = np.full((6, 4), np.nan, np.float32)
    for i in range(6):
        img = read(trial, i + 1, f)
        d = S.run_view(img, K) if img is not None else None
        if d is not None: j3d[i], ct[i], bx[i] = d["j3d"], d["cam_t"], d["box"]
    np.savez(p, j3d=j3d, cam_t=ct, box=bx); return j3d, ct, bx

def obs2d(j3d, ct):
    out = np.full((6, 70, 2), np.nan, np.float32)
    for i in range(6):
        if np.isnan(j3d[i]).any(): continue
        uv = (K @ (j3d[i] + ct[i]).T).T; out[i] = CAMS17[i].undistort(uv[:, :2] / uv[:, 2:3])
    return out

if __name__ == "__main__":
    a = [int(x) for x in sys.argv[1:4]] if len(sys.argv) > 3 else [10, 1120, 25]
    rep, loo, nv = [], [], []
    for f in range(*a):
        t = time.time(); j3d, ct, bx = views("TDB", f); j2d = obs2d(j3d, ct)
        vs = [i for i in range(6) if not np.isnan(j2d[i][S.BODY]).any()]
        if len(vs) >= 3:
            X, err, _ = triangulate_joints([CAMS17[i] for i in vs], j2d[vs][:, S.BODY]); rep.append(np.nanmedian(err))
            l = []
            for h in vs:
                r = [v for v in vs if v != h]; Xl, _, _ = triangulate_joints([CAMS17[v] for v in r], j2d[r][:, S.BODY])
                l.append(np.linalg.norm(CAMS17[h].project(Xl) - j2d[h][S.BODY], axis=1))
            loo.append(np.nanmedian(np.concatenate(l)))
        nv.append(len(vs)); print(f"frame {f}: views {len(vs)}  reproj {rep[-1] if rep else float('nan'):.1f}px  loo {loo[-1] if loo else float('nan'):.1f}px  ({time.time()-t:.1f}s)", flush=True)
    print("== MMC17 TDB SAM3D multi-view == frames", len(nv), "median views", np.median(nv), "reproj %.1f px, loo %.1f px (at 3840x2160)" % (np.median(rep), np.median(loo)))
