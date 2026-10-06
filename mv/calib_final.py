"""Final extrinsics: per-view offset side from the image (calib_offsets), flip from the ring-layout prior (Fig. 2a), then a joint
refinement of shared intrinsics + 6 board poses with the board-frame correspondences fixed."""
import cv2, numpy as np, itertools, json, sys
sys.path.insert(0, "."); import mmc17_sam3d as M, calib_ring as R
extraL = json.load(open(M.OUT + "/calib_offsets.json"))      # 1 = extra column on the left in detection order
cands = []
for v in range(6):
    left = extraL[str(v + 1)]
    cands.append([1, 2] if left else [0, 3])                  # option = offset*2 + flip   (see calib_offsets reasoning)
rows = []
for combo in itertools.product(*cands):
    if combo[0] % 2 or combo[0] == 3: continue                # fix the global 180-degree gauge on T1
    rows.append((R.cost(R.poses(combo))[0], combo))
rows.sort(); print("ring-prior cost of the 32 image-consistent hypotheses (best first):")
for s, c in rows[:5]: print("  %.3f  %s" % (s, c))
print("  worst %.3f, median %.3f" % (rows[-1][0], np.median([r[0] for r in rows])))
best = rows[0][1]
# joint refinement with the chosen correspondences
W, H = 3840, 2160
ob = [R.objp[R.opts[v][c][0]] for v, c in enumerate(best)]; ip = [R.opts[v][c][1] for v, c in enumerate(best)]
raw = [np.array(s, np.float64) for s in R.sub]                # distorted detections for calibrateCamera
ip = [raw[v][::-1] if best[v] % 2 else raw[v] for v in range(6)]
K0 = np.array(M.CAL["K"]); flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3
rms, K, dist, rv, tv = cv2.calibrateCamera([o.astype(np.float32) for o in ob], [p.astype(np.float32) for p in ip], (W, H), K0.copy(), np.zeros(5), flags=flags)
print("refined: RMS %.3f px  f=%.0f cx=%.0f cy=%.0f k1=%.4f k2=%.4f" % (rms, K[0, 0], K[0, 2], K[1, 2], dist.ravel()[0], dist.ravel()[1]))
cams = {}
for j in range(6):
    Rm = cv2.Rodrigues(rv[j])[0]; cams["T%d" % (j + 1)] = dict(R=Rm.tolist(), t=tv[j].ravel().tolist(), centre=(-Rm.T @ tv[j].ravel()).tolist())
json.dump(dict(K=K.tolist(), dist=dist.ravel().tolist(), size=[W, H], cams=cams, rms=rms, combo=list(best)), open(M.OUT + "/mmc17_calib_v2.json", "w"), indent=1)
C = np.array([cams["T%d" % (j + 1)]["centre"] for j in range(6)]); print("camera centres (board squares):\n", C.round(1))
