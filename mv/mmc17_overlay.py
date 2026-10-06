import os
import sys, cv2, numpy as np
sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body")); sys.path.insert(0, ".")
import mmc17_sam3d as M
sys.path.insert(0, os.environ.get("INSTANTHMR_REPO", "C:/dev/InstantHMR"))
f = int(sys.argv[1]); j3d, ct, bx = M.views("TDB", f); j2d = M.obs2d(j3d, ct); tiles = []
for i in range(6):
    v = cv2.cvtColor(M.read("TDB", i + 1, f), cv2.COLOR_RGB2BGR)
    if not np.isnan(bx[i]).any():
        b = bx[i].astype(int); cv2.rectangle(v, tuple(b[:2]), tuple(b[2:]), (0, 255, 255), 8)
    for j in M.S.BODY:
        p = j2d[i, j]
        if not np.isnan(p).any(): cv2.circle(v, (int(p[0]), int(p[1])), 14, (0, 0, 255), -1)
    print("T%d" % (i + 1), "box", bx[i].round(0), "cam_t", ct[i].round(2))
    tiles.append(cv2.resize(v, (640, 360)))
cv2.imwrite(f"{M.OUT}/mmc17_overlay_f{f}.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
