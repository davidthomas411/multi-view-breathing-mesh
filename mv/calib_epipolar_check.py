"""Independent check of the MMC17 extrinsics: SIFT matches between camera pairs (static room features) vs the epipolar geometry
implied by the calibration. Median Sampson error in px should be ~1-2 px for a good calibration."""
import sys, cv2, numpy as np
sys.path.insert(0, "."); import mmc17_sam3d as M
f = 400; imgs = [cv2.cvtColor(M.read("TDB", i + 1, f), cv2.COLOR_RGB2GRAY) for i in range(6)]
sift = cv2.SIFT_create(6000); kp = []; 
for g in imgs:
    g2 = cv2.resize(g, None, fx=0.5, fy=0.5); k, d = sift.detectAndCompute(g2, None); kp.append((np.array([p.pt for p in k]) * 2, d))
bf = cv2.BFMatcher()
def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 3)]:
    m = bf.knnMatch(kp[a][1], kp[b][1], k=2); good = [x for x, y in m if x.distance < 0.7 * y.distance]
    pa = np.float32([kp[a][0][x.queryIdx] for x in good]); pb = np.float32([kp[b][0][x.trainIdx] for x in good])
    ua = cv2.undistortPoints(pa.reshape(-1, 1, 2), M.K, M.DIST, P=M.K).reshape(-1, 2); ub = cv2.undistortPoints(pb.reshape(-1, 1, 2), M.K, M.DIST, P=M.K).reshape(-1, 2)
    Ca, Cb = M.CAMS17[a], M.CAMS17[b]
    Rab = Cb.R_wc @ Ca.R_wc.T; tab = Cb.t_wc - Rab @ Ca.t_wc; E = skew(tab) @ Rab; Ki = np.linalg.inv(M.K); F = Ki.T @ E @ Ki
    xa = np.c_[ua, np.ones(len(ua))]; xb = np.c_[ub, np.ones(len(ub))]
    Fx = xa @ F.T; Ftx = xb @ F; num = np.sum(xb * Fx, 1) ** 2; den = Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2
    s_all = np.sqrt(num / den)
    # also the F estimated freely from the matches (RANSAC) for reference
    Fr, mask = cv2.findFundamentalMat(ua, ub, cv2.FM_RANSAC, 1.5, 0.999); inl = mask.ravel() > 0; s = s_all[inl]
    print("T%d-T%d: %d RANSAC-consistent matches (of %d); calibrated-geometry Sampson error on them: median %.1f px, %.0f%% under 3 px" % (a + 1, b + 1, inl.sum(), len(good), np.median(s), 100 * (s < 3).mean()))
