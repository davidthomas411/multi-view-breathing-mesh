"""Detect retroreflective marker balls (small bright round blobs) in the 4K trial frames, and test them against the calibrated epipolar geometry."""
import cv2, numpy as np, sys
sys.path.insert(0, "."); import mmc17_sam3d as M

def blobs(gray, bgr=None):
    g = cv2.GaussianBlur(gray, (0, 0), 2.0)
    bg = cv2.GaussianBlur(gray, (0, 0), 25.0)                     # local background
    top = cv2.subtract(g, bg)                                      # bright-on-local-background (white-hat-like)
    _, m = cv2.threshold(top, 28, 255, cv2.THRESH_BINARY); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        if 40 <= a <= 700 and 0.6 < w / max(h, 1) < 1.6 and a / (w * h) > 0.55 and gray[lab == k].mean() > 150: out.append(cen[k])
    return np.array(out)

def skew(t): return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
def fmat(a, b):
    Ca, Cb = M.CAMS17[a], M.CAMS17[b]; Rab = Cb.R_wc @ Ca.R_wc.T; tab = Cb.t_wc - Rab @ Ca.t_wc; Ki = np.linalg.inv(M.K)
    return Ki.T @ (skew(tab) @ Rab) @ Ki

if __name__ == "__main__":
    f = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    imgs = [M.read("TDB", i + 1, f) for i in range(6)]; B = []
    for i, im in enumerate(imgs):
        b = blobs(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)); B.append(cv2.undistortPoints(b.reshape(-1, 1, 2).astype(np.float64), M.K, M.DIST, P=M.K).reshape(-1, 2) if len(b) else b)
        print("T%d: %d blobs" % (i + 1, len(b)))
    # draw detections
    tiles = []
    for i, im in enumerate(imgs):
        v = cv2.cvtColor(im, cv2.COLOR_RGB2BGR); raw = blobs(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY))
        for p in raw: cv2.circle(v, (int(p[0]), int(p[1])), 22, (0, 0, 255), 5)
        tiles.append(cv2.resize(v, (640, 360)))
    cv2.imwrite(M.OUT + "/blobs_f%d.jpg" % f, np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    # epipolar test: fraction of blobs in view a with a partner within 2 px of its epipolar line in view b, vs a control with a wrong (shifted) geometry
    for a, b in [(0, 1), (1, 2), (3, 4), (4, 5), (0, 5), (2, 3)]:
        F = fmat(a, b); xa = np.c_[B[a], np.ones(len(B[a]))]; xb = np.c_[B[b], np.ones(len(B[b]))]
        l = xa @ F.T; d = np.abs(l @ xb.T) / np.hypot(l[:, :1], l[:, 1:2])           # (na, nb) point-to-epiline distance
        hit = (d.min(1) < 2.0).mean()
        Fr = fmat(a, (b + 1) % 6)                                                      # control: epilines of the wrong camera
        l2 = xa @ Fr.T; d2 = np.abs(l2 @ xb.T) / np.hypot(l2[:, :1], l2[:, 1:2]); ctrl = (d2.min(1) < 2.0).mean()
        print("T%d->T%d: %.0f%% of %d blobs have a partner within 2 px of the calibrated epipolar line  (control with wrong camera pair: %.0f%%)" % (a + 1, b + 1, 100 * hit, len(B[a]), 100 * ctrl))
