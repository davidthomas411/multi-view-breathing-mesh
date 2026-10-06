import os
import sys, io, contextlib, cv2, numpy as np, torch
sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body")); sys.path.insert(0, ".")
import mmc17_sam3d as M, sam3d_mv as S
est, yolo = S.models()
COCO = [0, 5, 6, 11, 12, 13, 14, 15, 16]; MHR = [0, 5, 6, 9, 10, 11, 12, 13, 14]
def trial(img, K, tag):
    r = yolo(img, verbose=False)[0]; b = r.boxes.xyxy.cpu().numpy(); k = r.keypoints.xy.cpu().numpy()
    i = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
    for use_K in (True, False):
        with contextlib.redirect_stdout(io.StringIO()):
            o = est.process_one_image(img, bboxes=b[i:i+1].astype(np.float32), cam_int=torch.tensor(K[None], dtype=torch.float32) if use_K else None)[0]
        H, W = img.shape[:2]; f = float(o["focal_length"]); X = o["pred_keypoints_3d"] + o["pred_cam_t"]
        for name, (fx, cx, cy) in {"K": (K[0, 0], K[0, 2], K[1, 2]), "model focal, centre": (f, W / 2, H / 2)}.items():
            uv = np.stack([fx * X[:, 0] / X[:, 2] + cx, fx * X[:, 1] / X[:, 2] + cy], 1)
            e = np.median(np.linalg.norm(uv[MHR] - k[i][COCO], axis=1)) / (b[i][3] - b[i][1]) * 100
            print(f"{tag} cam_int given={use_K}: model focal_length={f:.0f} (K fx={K[0,0]:.0f}); project with {name}: {e:.1f}% of box height from YOLO")
        e2 = np.median(np.linalg.norm(o["pred_keypoints_2d"][MHR] - k[i][COCO], axis=1)) / (b[i][3] - b[i][1]) * 100
        print(f"{tag} cam_int given={use_K}: pred_keypoints_2d: {e2:.1f}%")
img = M.read("TDB", 1, 35); trial(img, M.K, "MMC17 T1 f35 (4K)")
cams = S.load_cameras(); img = cv2.cvtColor(cv2.imread(f"{S.FRAME_DIR}/01/0100.jpg"), cv2.COLOR_BGR2RGB); trial(img, cams["01"].K, "vault2 cam01 f100 (1080p)")
