"""Multi-view InstantHMR: 6 calibrated cameras -> one batched ONNX call -> triangulated 3D skeleton.

Per frame:
  1. boxes   : RF-DETR on every view (init / recovery) OR the previous 3D skeleton reprojected
               into each view (tracking mode, no detector at all)
  2. crops   : 224x224 square crop per view, all 6 stacked into one ONNX batch
  3. HMR     : one session.run on batch=6  -> 70 2D joints per view (+ per-view 3D / MHR params)
  4. fuse    : robust DLT triangulation of the 2D joints -> world-frame 3D joints
"""
from __future__ import annotations
import os

import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("INSTANTHMR_REPO", "C:/dev/InstantHMR"))
import torch  # noqa: F401  (must import before onnxruntime so both share torch's CUDA 12 DLLs)

from instanthmr.inference import InstantHMR  # noqa: E402

from mvcal import Camera, load_cameras, triangulate_joints, FRAME_DIR  # noqa: E402

# Body joints used for fusion (MHR70): face, shoulders, elbows, hips, knees, ankles, feet,
# wrists, olecranon/cubital/acromion, neck.  Fingers are excluded - far too noisy at this scale.
BODY = list(range(0, 21)) + [41, 62, 63, 64, 65, 66, 67, 68, 69]


class MultiViewHMR:
    def __init__(self, onnx=os.environ.get("INSTANTHMR_REPO", "C:/dev/InstantHMR") + "/models/instanthmr.onnx", cams: dict[str, Camera] | None = None,
                 detector: str | None = "medium", det_conf: float = 0.4):
        self.hmr = InstantHMR(onnx, device="cuda")
        self.cams = cams or load_cameras()
        self.names = list(self.cams)
        self.det = None
        if detector:
            from instanthmr.detector import RFDETRDetector
            self.det = RFDETRDetector(variant=detector, confidence=det_conf, max_persons=1, optimize_for_inference=False)
            self.det.warmup()
        n = len(self.names)
        self.hmr.warmup(max_batch_size=n)
        self.hmr.warmup(max_batch_size=n)
        self.prev_X: np.ndarray | None = None
        self.frame_no = 0
        self.rotate = bool(int(__import__('os').environ.get('ROTATE', 1)))
        self.recover_every = 3   # tracking mode: retry the detector on views that lost the person every N frames
        self.prev_boxes: list[np.ndarray | None] = [None] * n

    # ---------------- boxes ----------------
    def _boxes_detect(self, imgs):
        # NB: detector is run per view (RF-DETR wrapper is single-image); a batched call is the
        # obvious next optimisation if tracking mode is not used.
        return [(d[0]["bbox"] if (d := self.det.detect(im)) else None) for im in imgs]

    def margin(self):
        return float(__import__('os').environ.get('TRACK_MARGIN', 0.04))

    def _boxes_track(self, cams, imgs, margin=None):
        margin = float(__import__('os').environ.get('TRACK_MARGIN', 0.04)) if margin is None else margin
        out = []
        for c, im in zip(cams, imgs):
            h, w = im.shape[:2]
            if self.prev_X is None or np.isnan(self.prev_X[BODY]).any():
                out.append(None)
                continue
            if (c.depth(self.prev_X[BODY]) <= 0).any():
                out.append(None)
                continue
            uv = c.project(self.prev_X[BODY])
            x1, y1 = uv.min(0)
            x2, y2 = uv.max(0)
            bw, bh = x2 - x1, y2 - y1
            if x2 < 0 or y2 < 0 or x1 > w or y1 > h or bw < 8 or bh < 8 or bw > 1.5 * w or bh > 1.5 * h:
                out.append(None)
                continue
            out.append(np.array([x1 - bw * margin, y1 - bh * margin, x2 + bw * margin, y2 + bh * margin], np.float32))
        return out

    def _crop_rot(self, img, pts, margin):
        """Square crop whose vertical axis follows the body axis (neck->hips) so a supine person looks upright.
        pts: (N,2) projected body joints of the previous skeleton. Returns crop, cliff, inverse affine (2x3)."""
        from instanthmr.inference import CROP_EXPAND, IMAGENET_MEAN, IMAGENET_STD
        import cv2
        h, w = img.shape[:2]
        neck, hips = pts[BODY.index(69)], pts[[BODY.index(9), BODY.index(10)]].mean(0)
        d = neck - hips
        alpha = np.degrees(np.arctan2(-d[1], d[0]))          # visual ccw angle of the body axis
        phi = 90.0 - alpha                                    # rotate so it points up
        c = (pts.min(0) + pts.max(0)) / 2
        R = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), phi, 1.0)
        ap = pts @ R[:, :2].T + R[:, 2]                       # joints in the aligned frame
        bw, bh = np.ptp(ap[:, 0]), np.ptp(ap[:, 1])
        size = max(bw, bh) * (1 + 2 * margin) * CROP_EXPAND
        sc = self.hmr.input_size / size
        M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), phi, sc)
        M[0, 2] += self.hmr.input_size / 2 - c[0]
        M[1, 2] += self.hmr.input_size / 2 - c[1]
        crop = cv2.warpAffine(img, M, (self.hmr.input_size,) * 2, flags=cv2.INTER_LINEAR)
        crop = np.transpose((crop.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD, (2, 0, 1)).astype(np.float32)
        cliff = np.array([2 * c[0] / w - 1, 2 * c[1] / h - 1, max(bw, bh) * (1 + 2 * margin) / max(w, h)], np.float32)
        Minv = cv2.invertAffineTransform(M)
        return crop, cliff, Minv

    # ---------------- main step ----------------
    def step(self, imgs_rgb: list[np.ndarray], mode: str = "detect"):
        """imgs_rgb: one RGB frame per camera, in self.names order. mode: 'detect' | 'track'."""
        t = {}
        t0 = time.perf_counter()
        cams = [self.cams[n].scaled(im.shape[1], im.shape[0]) for n, im in zip(self.names, imgs_rgb)]

        ta = time.perf_counter()
        if mode == "track" and self.prev_X is not None:
            boxes = self._boxes_track(cams, imgs_rgb)
            # recover views that lost the person with the detector
            miss = [i for i, b in enumerate(boxes) if b is None]
            self.frame_no += 1
            if miss and self.det and self.frame_no % self.recover_every == 0:
                # recover ONE lost view per attempt (round-robin) so a re-acquire never stalls the frame
                i = miss[(self.frame_no // self.recover_every) % len(miss)]
                d = self.det.detect(imgs_rgb[i])
                boxes[i] = d[0]["bbox"] if d else None
        else:
            boxes = self._boxes_detect(imgs_rgb)
        t["boxes_ms"] = (time.perf_counter() - ta) * 1000

        # ---- preprocess + one batched HMR call ----
        tb = time.perf_counter()
        ok = [i for i, b in enumerate(boxes) if b is not None]
        n = len(imgs_rgb)
        S = self.hmr.input_size
        crops = np.zeros((n, 3, S, S), np.float32)
        cliffs = np.zeros((n, 3), np.float32)
        meta = {}
        use_rot = self.rotate and mode == "track" and self.prev_X is not None
        for i in ok:
            h, w = imgs_rgb[i].shape[:2]
            if use_rot and (cams[i].depth(self.prev_X[BODY]) > 0).all():
                pts = cams[i].project(self.prev_X[BODY])
                crops[i], cliffs[i], Minv = self._crop_rot(np.ascontiguousarray(imgs_rgb[i]), pts, self.margin())
                meta[i] = Minv
                continue
            crop, sx, sy, ss, cl = self.hmr._preprocess(np.ascontiguousarray(imgs_rgb[i]), boxes[i], h, w)
            crops[i], cliffs[i] = crop, cl
            meta[i] = (sx, sy, ss)
        t["pre_ms"] = (time.perf_counter() - tb) * 1000

        tc = time.perf_counter()
        outs = self.hmr.session.run(None, {self.hmr._in_image: crops, self.hmr._in_cliff: cliffs})
        t["hmr_ms"] = (time.perf_counter() - tc) * 1000

        # ---- fuse ----
        td = time.perf_counter()
        j2d = np.full((n, 70, 2), np.nan, np.float32)
        for i in ok:
            if isinstance(meta[i], np.ndarray):                  # rotated crop: invert the affine
                cp = (outs[3][i].astype(np.float32) + 1) * 0.5 * S
                j2d[i] = cp @ meta[i][:, :2].T + meta[i][:, 2]
                continue
            sx, sy, ss = meta[i]
            px = (outs[3][i].astype(np.float32) + 1) * 0.5 * S * (ss / S)
            j2d[i] = px + np.array([sx, sy], np.float32)
        for i in ok:   # lens distortion: move predictions onto the ideal pinhole model the triangulation assumes
            j2d[i] = cams[i].undistort(j2d[i])
        views = [i for i in ok]
        X = np.full((70, 3), np.nan)
        err = used = None
        if len(views) >= 2:
            Xb, eb, ub = triangulate_joints([cams[i] for i in views], j2d[views][:, BODY])
            X[BODY] = Xb
            err = np.full((n, 70), np.nan)
            used = np.zeros((n, 70), bool)
            err[np.ix_(views, BODY)] = eb
            used[np.ix_(views, BODY)] = ub
            ext = np.nanmax(Xb, 0) - np.nanmin(Xb, 0)
            # a fused skeleton larger than a body means a bad triangulation: don't seed tracking with it
            self.prev_X = X if (ext < 2.6).all() else None
        t["fuse_ms"] = (time.perf_counter() - td) * 1000
        t["total_ms"] = (time.perf_counter() - t0) * 1000
        return dict(X=X, j2d=j2d, boxes=boxes, err=err, used=used, views=views, timing=t,
                    cam_trans=outs[2].astype(np.float32), j3d_local=outs[4].astype(np.float32),
                    cams=cams)
