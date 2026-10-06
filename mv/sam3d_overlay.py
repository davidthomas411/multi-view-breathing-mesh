import os
import sys, cv2, numpy as np
sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body")); sys.path.insert(0, ".")
from mvcal import load_cameras, CAMS, FRAME_DIR, OUT
from sam3d_mv import run_view
sys.path.insert(0, os.environ.get("INSTANTHMR_REPO", "C:/dev/InstantHMR"))
f = int(sys.argv[1]); cams = load_cameras(); tiles = []
for c in CAMS:
    img = cv2.cvtColor(cv2.imread(f"{FRAME_DIR}/{c}/{f:04d}.jpg"), cv2.COLOR_BGR2RGB)
    d = run_view(img, cams[c].K)
    v = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if d is not None:
        b = d["box"].astype(int); cv2.rectangle(v, tuple(b[:2]), tuple(b[2:]), (0, 255, 255), 3)
        for p in d["j2d"]: cv2.circle(v, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
        # same joints reprojected from the 3D prediction + cam_t using the calibrated K  (consistency check of the 2D output)
        X = d["j3d"] + d["cam_t"]; uv = (cams[c].K @ X.T).T; uv = uv[:, :2] / uv[:, 2:3]
        for p in uv: cv2.circle(v, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
        print(c, "box", b, "2D vs reproject(3D+cam_t) median px diff: %.1f" % np.median(np.linalg.norm(uv - d["j2d"], axis=1)))
    tiles.append(cv2.resize(v, (640, 360)))
cv2.imwrite(f"{OUT}/sam3d_overlay_f{f}.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
