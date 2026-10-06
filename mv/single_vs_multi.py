"""How far is a single view's own 3D skeleton from the multi-view fused one?

Root-relative (hip-centred) so it isolates pose, not placement; and a separate depth check:
distance of the person from each camera, from the model's cam_trans vs from triangulation.
The model assumes focal = image diagonal (pixel-conditioned, see InstantHMR README) whereas the real
cameras are wide-angle, so we also report depth after the focal correction f_real / f_assumed.
"""
import sys
import cv2
import numpy as np

from mvhmr import MultiViewHMR, BODY
from mvcal import CAMS, FRAME_DIR

mv = MultiViewHMR()
mv.det.warmup()
HIPS = [9, 10]
bodyj = [j for j in BODY if j > 0 or True]
root_err, depth_raw, depth_fix, tri_d, fig = [], [], [], [], []
for f in range(0, 706, 6):
    imgs = [cv2.cvtColor(cv2.imread(f"{FRAME_DIR}/{c}/{f:04d}.jpg"), cv2.COLOR_BGR2RGB) for c in CAMS]
    r = mv.step(imgs, mode="detect")
    if len(r["views"]) < 4:
        continue
    X = r["X"]
    Xrel = X - X[HIPS].mean(0)
    for i in r["views"]:
        cam = r["cams"][i]
        R_cw = cam.R_wc.T
        Jw = (r["j3d_local"][i] @ R_cw.T)              # rig-local camera axes -> world axes
        Jrel = Jw - Jw[HIPS].mean(0)
        d = np.linalg.norm((Jrel - Xrel)[BODY], axis=1)
        root_err.append(np.nanmean(d) * 1000)
        pelvis_cam = cam.R_wc @ X[HIPS].mean(0) + cam.t_wc
        t_model = r["cam_trans"][i] + r["j3d_local"][i][HIPS].mean(0)
        h, w = imgs[i].shape[:2]
        f_ratio = cam.K[0, 0] / np.hypot(h, w)
        tri_d.append(np.linalg.norm(pelvis_cam))
        depth_raw.append(np.linalg.norm(t_model))
        depth_fix.append(np.linalg.norm(t_model) * f_ratio)
root_err, depth_raw, depth_fix, tri_d = map(np.array, (root_err, depth_raw, depth_fix, tri_d))
print("samples (view-frames):", len(root_err))
print("single-view vs fused, root-relative joint error: median %.0f mm  (p90 %.0f mm)" % (np.median(root_err), np.percentile(root_err, 90)))
print("person distance from camera  triangulated median %.2f m" % np.median(tri_d))
print("  model cam_trans, raw        : median abs error %.2f m (%.0f%%)" % (np.median(np.abs(depth_raw - tri_d)), 100 * np.median(np.abs(depth_raw - tri_d) / tri_d)))
print("  model cam_trans, focal-fixed: median abs error %.2f m (%.0f%%)" % (np.median(np.abs(depth_fix - tri_d)), 100 * np.median(np.abs(depth_fix - tri_d) / tri_d)))
