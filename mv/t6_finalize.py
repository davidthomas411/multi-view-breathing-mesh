"""PnP for T6 in the Vicon frame from the marker matches found by t6_search.py; adds T6 to tmp/mmc17_vicon_calib.json (backup kept)."""
import os
import sys, json, shutil, numpy as np, cv2
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import vicon_calib as V, mmc17_sam3d as M
K = V.K; S = json.load(open(M.OUT + "/t6_search.json")); R0 = np.array(S["R"]); t0 = np.array(S["t"]); frames = list(range(260, 541, 40)); dets = {f: V.detections(f) for f in frames}
P3, P2, lab = [], [], []
for f in frames:
    Vf = V.vicon(f) * 1e-3; d = dets[f][5]; idxm = np.where(~np.isnan(Vf).any(1))[0]; Xc = Vf[idxm] @ R0.T + t0; uv = Xc @ K.T; uv = uv[:, :2] / uv[:, 2:3]; dist = np.linalg.norm(uv[:, None, :] - d[None, :, :2], axis=2)
    for a in range(len(idxm)):
        j = dist[a].argmin()
        if Xc[a, 2] > 0 and dist[a, j] < 8 and dist[:, j].argmin() == a: P3.append(Vf[idxm[a]]); P2.append(d[j, :2]); lab.append((f, V.NAMES[idxm[a]]))
P3 = np.array(P3, np.float64); P2 = np.array(P2, np.float64)
ok, rv, tv, inl = cv2.solvePnPRansac(P3, P2, K, None, rvec=cv2.Rodrigues(R0)[0], tvec=t0.reshape(3, 1), useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=4.0, iterationsCount=2000)
inl = inl.ravel(); rv, tv = cv2.solvePnPRefineLM(P3[inl], P2[inl], K, None, rv, tv)
pr = cv2.projectPoints(P3[inl], rv, tv, K, None)[0].reshape(-1, 2); rms = float(np.sqrt(np.mean(np.sum((pr - P2[inl]) ** 2, 1))))
print("T6: %d matches, %d RANSAC inliers, reprojection RMS %.2f px, camera centre %s m" % (len(P3), inl.size, rms, np.round(-cv2.Rodrigues(rv)[0].T @ tv.ravel(), 3)))
J = json.load(open(M.OUT + "/mmc17_vicon_calib.json")); shutil.copy(M.OUT + "/mmc17_vicon_calib.json", M.OUT + "/mmc17_vicon_calib_T1-T5.json")
J["cams"]["T6"] = dict(rvec=rv.ravel().tolist(), tvec=tv.ravel().tolist(), n=int(inl.size), rms=rms, note="found by marker pose search (t6_search.py)"); json.dump(J, open(M.OUT + "/mmc17_vicon_calib.json", "w"), indent=1)
