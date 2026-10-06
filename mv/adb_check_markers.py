"""End-to-end check of the ADB geometry: Vicon markers of the ADB trial projected with the TDB-derived extrinsics into the WARPED raw ADB frames, against the balls found by the detector.
The same measurement on the stabilised TDB frames is the control.   python adb_check_markers.py [frames ...]"""
import csv, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adb_geometry as G, marker_detect as MD
ROOT, TMP, MM = G.ROOT, G.TMP, G.MM; CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K, DIST = G.K, G.DIST
CAM = {i: (np.array(CJ["cams"][f"T{i}"]["rvec"]), np.array(CJ["cams"][f"T{i}"]["tvec"])) for i in range(1, 6)}
def load_vicon(trial):
    rows = list(csv.reader(open(f"{ROOT}/CUTrial/Trials_marker_positions_by_image_frame_all_cases/MMC17_{trial}_marker_positions_by_image_frame_wide.csv")))
    names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]; return names, {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}
def measure(trial, frames, cams=(1, 2, 3, 4, 5)):
    names, V = load_vicon(trial); res = {c: [] for c in cams}
    for c in cams:
        Wp = G.Warper(c) if trial == "ADB" else None; cap = cv2.VideoCapture(f"{MM}/ADB/MMC17_ADB_T{c}.mov" if trial == "ADB" else f"{MM}/TDB/Gyroflow/MMC17_TDB_T{c}_stabilized.mp4")
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
            if not ok: continue
            if Wp is not None: im = Wp(im)
            mk = V[f + 1] * 1e-3; okm = ~np.isnan(mk).any(1); P = cv2.projectPoints(mk[okm].reshape(-1, 1, 3), CAM[c][0], CAM[c][1], K, DIST)[0].reshape(-1, 2)
            x0, y0 = P.min(0) - 80; x1, y1 = P.max(0) + 80; det = MD.detect(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), [x0, y0, x1, y1])
            if len(det) == 0: continue
            d = np.linalg.norm(P[:, None, :] - det[None, :, :2], axis=2); j = d.argmin(1); dmin = d[np.arange(len(P)), j]
            mutual = np.array([d[:, j[k]].argmin() == k for k in range(len(P))]); sel = (dmin < 25) & mutual
            if sel.sum() > 3: res[c].append((f, int(sel.sum()), float(np.median(dmin[sel])), float(np.mean(dmin[sel])), (P - det[j, :2])[sel].mean(0)))
        cap.release()
    return res
if __name__ == "__main__":
    frames = [int(a) for a in sys.argv[1:]] or [300, 500, 700, 900]
    for trial in ("TDB", "ADB"):
        r = measure(trial, frames); print(f"== {trial}")
        for c, v in r.items():
            print(f"  T{c}: " + " | ".join(f"f{f}: {n} matched, median {m:.1f} px, mean offset ({o[0]:+.1f},{o[1]:+.1f})" for f, n, m, mean, o in v))
        allm = [m for v in r.values() for _, _, m, _, _ in v]; print(f"  overall median of the per-frame median distances: {np.median(allm):.2f} px (n={len(allm)})")
