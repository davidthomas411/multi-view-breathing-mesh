import os
import sys, cv2, numpy as np, torch
sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body")); sys.path.insert(0, ".")
from mvcal import load_cameras, CAMS, FRAME_DIR
import sam3d_mv as S
cams = load_cameras(); est, yolo = S.models()
MHR = [0, 5, 6, 9, 10, 11, 12, 13, 14]; COCO = [0, 5, 6, 11, 12, 13, 14, 15, 16]     # nose, shoulders, hips, knees, ankles
rows = []
for f in (100, 300, 500, 705):
    for c in CAMS:
        img = cv2.cvtColor(cv2.imread(f"{FRAME_DIR}/{c}/{f:04d}.jpg"), cv2.COLOR_BGR2RGB)
        r = yolo(img, verbose=False)[0]
        if len(r.boxes) == 0: continue
        k = r.keypoints.xy.cpu().numpy(); b = r.boxes.xyxy.cpu().numpy(); i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
        yk = k[i][COCO]
        d = S.run_view(img, cams[c].K)
        if d is None: continue
        uv3 = (cams[c].K @ (d["j3d"] + d["cam_t"]).T).T; uv3 = uv3[:, :2] / uv3[:, 2:3]
        e2 = np.linalg.norm(d["j2d"][MHR] - yk, axis=1); e3 = np.linalg.norm(uv3[MHR] - yk, axis=1)
        h = d["box"][3] - d["box"][1]
        rows.append((f, c, np.median(e2) / h * 100, np.median(e3) / h * 100))
print("median joint distance to YOLO-Pose, as %% of person height:")
for f in (100, 300, 500, 705):
    r = [x for x in rows if x[0] == f]
    if r: print(" frame", f, " SAM3D 2D output: %.1f%%   SAM3D 3D+cam_t reprojected: %.1f%%" % (np.median([x[2] for x in r]), np.median([x[3] for x in r])))
