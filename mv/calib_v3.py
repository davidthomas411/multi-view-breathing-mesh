"""Calibration v3: board-alignment options fixed by the Vicon marker matches (T1-T5 = option 0; T6 unresolved, option 0 kept)."""
import os
import sys, json, numpy as np, cv2
sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M
combo = [0, 0, 0, 0, 0, 0]
objp = np.zeros((35, 3), np.float32); objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2); idx = np.arange(35).reshape(5, 7)
sub = [np.array(s, np.float32) for s in np.load(M.OUT + "/board_sub.npz")["sub"]]
ob = [objp[idx[:, :6].ravel()] for _ in range(6)]
K0 = np.array(M.CAL["K"]); flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
rms, K, dist, rv, tv = cv2.calibrateCamera(ob, sub, (3840, 2160), K0.copy(), np.zeros(5), flags=flags)
print("v3: RMS %.3f px  f=%.0f cx=%.0f cy=%.0f k1=%.4f" % (rms, K[0, 0], K[0, 2], K[1, 2], dist.ravel()[0]))
cams = {}
for j in range(6):
    R = cv2.Rodrigues(rv[j])[0]; cams["T%d" % (j + 1)] = dict(R=R.tolist(), t=tv[j].ravel().tolist(), centre=(-R.T @ tv[j].ravel()).tolist())
C = np.array([cams["T%d" % (j + 1)]["centre"] for j in range(6)]); print("camera centres (squares):\n", C.round(1))
json.dump(dict(K=K.tolist(), dist=dist.ravel().tolist(), size=[3840, 2160], cams=cams, rms=rms, combo=combo), open(M.OUT + "/mmc17_calib_v3.json", "w"), indent=1)
