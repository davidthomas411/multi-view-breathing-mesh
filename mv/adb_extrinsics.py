"""Extrinsics of cameras T1-T5 for the ADB trial, by PnP on the Vicon marker balls found in the warped raw ADB frames (initialised from the TDB-derived extrinsics).
The warp (adb_geometry.py) maps the raw wide-angle video into the geometry of the stabilised videos; any small camera movement between the TDB and ADB trials, and the residual error of the warp,
are absorbed here.  Output tmp/mmc17_vicon_calib_ADB.json (same layout as mmc17_vicon_calib.json).   python adb_extrinsics.py [cams ...]"""
import json, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adb_geometry as G, marker_detect as MD
from adb_check_markers import load_vicon
TMP, MM, K, DIST = G.TMP, G.MM, G.K, G.DIST
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); names, V = load_vicon("ADB")
def collect(cam, frames, cur):
    Wp = G.Warper(cam); cap = cv2.VideoCapture(f"{MM}/ADB/MMC17_ADB_T{cam}.mov"); obj, img, fid = [], [], []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
        if not ok: continue
        im = Wp(im); mk = V[f + 1] * 1e-3; okm = ~np.isnan(mk).any(1); X = mk[okm]; P = cv2.projectPoints(X.reshape(-1, 1, 3), cur[0], cur[1], K, DIST)[0].reshape(-1, 2)
        x0, y0 = P.min(0) - 120; x1, y1 = P.max(0) + 120; det = MD.detect(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), [x0, y0, x1, y1])
        if len(det) < 4: continue
        d = np.linalg.norm(P[:, None, :] - det[None, :, :2], axis=2); j = d.argmin(1); dmin = d[np.arange(len(P)), j]; mutual = np.array([d[:, j[k]].argmin() == k for k in range(len(P))])
        off = np.median((det[j, :2] - P)[(dmin < 30) & mutual], axis=0) if ((dmin < 30) & mutual).sum() > 3 else np.zeros(2)                                   # remove the common offset, then match tightly
        d = np.linalg.norm((P + off)[:, None, :] - det[None, :, :2], axis=2); j = d.argmin(1); dmin = d[np.arange(len(P)), j]; mutual = np.array([d[:, j[k]].argmin() == k for k in range(len(P))]); sel = (dmin < 10) & mutual
        obj.append(X[sel]); img.append(det[j[sel], :2]); fid.append(np.full(sel.sum(), f))
    cap.release(); return np.vstack(obj), np.vstack(img), np.concatenate(fid)
if __name__ == "__main__":
    cams = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5]; out = json.load(open(TMP + "/mmc17_vicon_calib.json")); frames = list(range(260, 1010, 25)); report = {}
    for cam in cams:
        t0 = time.time(); c0 = CJ["cams"][f"T{cam}"]; cur = (np.array(c0["rvec"], float), np.array(c0["tvec"], float))
        for it in range(2):                                                                                                                                 # re-collect with the improved pose
            O, I, F = collect(cam, frames, cur)
            ok, rv, tv, inl = cv2.solvePnPRansac(O, I, K, DIST, cur[0].reshape(3, 1).copy(), cur[1].reshape(3, 1).copy(), useExtrinsicGuess=True, reprojectionError=4.0, iterationsCount=500, flags=cv2.SOLVEPNP_ITERATIVE)
            inl = inl.ravel(); rv, tv = cv2.solvePnPRefineLM(O[inl], I[inl], K, DIST, rv, tv); cur = (rv.ravel(), tv.ravel())
        P = cv2.projectPoints(O.reshape(-1, 1, 3), cur[0], cur[1], K, DIST)[0].reshape(-1, 2); r = np.linalg.norm(P - I, axis=1)
        R0 = cv2.Rodrigues(np.array(c0["rvec"]))[0]; R1 = cv2.Rodrigues(cur[0])[0]; dang = np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1)))
        C0 = -R0.T @ np.array(c0["tvec"]); C1 = -R1.T @ cur[1]; report[f"T{cam}"] = dict(n=int(len(O)), n_inliers=int(len(inl)), rms_px=float(np.sqrt(np.mean(r[inl] ** 2))), median_px=float(np.median(r)), rotation_change_deg=float(dang), centre_shift_mm=float(np.linalg.norm(C1 - C0) * 1000))
        out["cams"][f"T{cam}"] = dict(rvec=cur[0].tolist(), tvec=cur[1].tolist(), n=int(len(inl)), rms=float(np.sqrt(np.mean(r[inl] ** 2))))
        print(f"T{cam}: {len(O)} matched balls over {len(set(F.tolist()))} frames, {len(inl)} inliers, RMS {report[f'T{cam}']['rms_px']:.2f} px, median {report[f'T{cam}']['median_px']:.2f} px; vs the TDB extrinsics: rotation {dang:.3f} deg, camera centre moved {report[f'T{cam}']['centre_shift_mm']:.1f} mm ({time.time() - t0:.0f}s)", flush=True)
    out["adb_report"] = report; json.dump(out, open(TMP + "/mmc17_vicon_calib_ADB.json", "w"), indent=1); print("wrote tmp/mmc17_vicon_calib_ADB.json")
