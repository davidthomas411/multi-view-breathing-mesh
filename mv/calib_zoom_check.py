"""Do the ECal and trial Gyroflow videos share the same virtual camera? Same tripod camera, ECal frame vs trial frame:
a rigid camera with identical intrinsics gives ~identity (scale 1.00, no shift) on the static background."""
import cv2, numpy as np, sys
sys.path.insert(0, "."); import mmc17_sam3d as M
ECAL = M.ROOT + "/CUTrial/MM17/DBECal/post_processing/Gyroflow/MMC17_DBECal_T{i}_stabilized.mp4"
def rd(p, f):
    c = cv2.VideoCapture(p); c.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = c.read(); return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create(8000); bf = cv2.BFMatcher()
for i in range(1, 7):
    a = rd(ECAL.format(i=i), 15); b = rd(M.VID.format(trial="TDB", i=i), 400)
    ka, da = sift.detectAndCompute(cv2.resize(a, None, fx=.5, fy=.5), None); kb, db = sift.detectAndCompute(cv2.resize(b, None, fx=.5, fy=.5), None)
    g = [x for x, y in bf.knnMatch(da, db, k=2) if x.distance < 0.75 * y.distance]
    pa = np.float32([ka[x.queryIdx].pt for x in g]) * 2; pb = np.float32([kb[x.trainIdx].pt for x in g]) * 2
    A, inl = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=4)
    s = np.hypot(A[0, 0], A[1, 0]); ang = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    d = np.linalg.norm(pb[inl.ravel() > 0] - pa[inl.ravel() > 0], axis=1)
    print("T%d: %d matches, %d inliers | scale %.4f  rotation %.2f deg  translation (%.0f, %.0f) px | median displacement of inliers %.1f px" % (i, len(g), inl.sum(), s, ang, A[0, 2], A[1, 2], np.median(d)))
