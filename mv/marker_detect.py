"""Detect reflective-marker spheres (grey balls with a dark base) in 4K frames.
Cue: low colour saturation + mid/high brightness + round + darker surround. Returns (N,3): x, y, radius_px."""
import os
import cv2, numpy as np

def detect(img_rgb, roi=None, rmin=7, rmax=24):
    H, W = img_rgb.shape[:2]
    x0, y0, x1, y1 = (0, 0, W, H) if roi is None else [int(v) for v in roi]
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
    sub = img_rgb[y0:y1, x0:x1]; hsv = cv2.cvtColor(sub, cv2.COLOR_RGB2HSV); g = cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY)
    S, V = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    gb = cv2.GaussianBlur(g, (0, 0), 1.5)
    loc = cv2.GaussianBlur(g, (0, 0), 3 * rmax)                                  # local background brightness
    m = ((S < 55) & (V > 105) & (gb > 0.0)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for k in range(1, n):
        x, y, w, h, a = st[k]
        r = np.sqrt(a / np.pi)
        if not (rmin <= r <= rmax) or not (0.7 < w / max(h, 1) < 1.4) or a / (w * h) < 0.62: continue
        cx, cy = cen[k]
        # dark surround: ring of radius 1.25r-1.9r should be darker than the ball on average (dark cup / shadow / black cloth)
        yy, xx = np.ogrid[:g.shape[0], :g.shape[1]]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring = (d2 > (1.25 * r) ** 2) & (d2 < (1.9 * r) ** 2); ball = lab == k
        if ring.sum() < 30: continue
        contrast = float(g[ball].mean() - np.percentile(g[ring], 25))
        if contrast < 25: continue
        out.append((cx + x0, cy + y0, r, contrast))
    return np.array(out) if out else np.zeros((0, 4))

if __name__ == "__main__":
    import sys; sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
    import mmc17_sam3d as M
    f = int(sys.argv[1]) if len(sys.argv) > 1 else 410
    z = np.load(M.OUT + f"/mmc17_sam3d/TDB_{f:04d}.npz"); tiles = []
    for i in range(6):
        im = M.read("TDB", i + 1, f); b = z["box"][i]; pad = 60
        d = detect(im, [b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad]); print("T%d: %d detections" % (i + 1, len(d)))
        v = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        for x, y, r, c in d: cv2.circle(v, (int(x), int(y)), int(r * 1.6), (0, 0, 255), 4)
        tiles.append(cv2.resize(v, (640, 360)))
    cv2.imwrite(M.OUT + f"/marker_detect_f{f}.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
