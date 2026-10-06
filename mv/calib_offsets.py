"""Per view: is the detected 6x5 sub-grid the left (offset 0) or right (offset 1) six columns of the board's 7x5 inner corners?
Extrapolate the lattice one corner beyond each end; the side that continues into more checkerboard (bright squares) is the
inside, the other side is the board edge / dark table."""
import cv2, numpy as np, sys, json
sys.path.insert(0, "."); import mmc17_sam3d as M, calib_extrinsics as C
sub = list(np.load(M.OUT + "/board_sub.npz")["sub"]); out = {}
for v in range(6):
    c = cv2.VideoCapture(C.VID.format(i=v + 1)); c.set(cv2.CAP_PROP_POS_FRAMES, 15); ok, fr = c.read(); g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (0, 0), 3)
    grid = np.mgrid[0:6, 0:5].T.reshape(-1, 2).astype(np.float32)          # (i,j) in detection order (6 per row)
    H, _ = cv2.findHomography(grid, sub[v].astype(np.float32), 0)
    def probe(i):                                                          # brightness along the 5 rows at lattice column i
        pts = np.array([[i, j] for j in np.linspace(-0.0, 4.0, 5)], np.float32).reshape(-1, 1, 2); p = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        return np.array([g[int(np.clip(y, 0, g.shape[0] - 1)), int(np.clip(x, 0, g.shape[1] - 1))] for x, y in p], float)
    left, right = probe(-1.5), probe(6.5)                                  # half a square beyond the end corner columns... (-1.5/6.5: one square + half)
    left2, right2 = probe(-0.5), probe(5.5)
    # board squares just outside the sub-grid: i=-0.5 / 5.5 are inside the board by construction; the discriminating squares are i=-1.5 / 6.5
    ml, mr = left.max(), right.max(); sl, sr = left.std(), right.std()
    side = "extra column on the LEFT (offset 1)" if (ml + sl) > (mr + sr) else "extra column on the RIGHT (offset 0)"
    out[v + 1] = int((ml + sl) > (mr + sr))
    print("T%d: left side max/std %.0f/%.0f   right side max/std %.0f/%.0f  ->  %s" % (v + 1, ml, sl, mr, sr, side))
json.dump(out, open(M.OUT + "/calib_offsets.json", "w"))
