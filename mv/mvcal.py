"""Calibration + triangulation helpers for the 6-camera rig.

Calibration: cuvault2/vault2/videos/intri.yml + extri.yml (EasyMocap style, OpenCV conventions):
  K_xx / dist_xx at the native 5312x2988; Rot_xx (3x3) + T_xx = world->camera. Same cameras as scene.rrd.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = f"{ROOT}/cuvault2/vault2/videos"
OUT = f"{ROOT}/tmp"
FRAME_DIR = f"{ROOT}/tmp/frames"   # 1920x1080 cache made by cache_frames.py
CAMS = ["01", "02", "03", "04", "05", "06"]


@dataclass
class Camera:
    name: str
    K: np.ndarray        # (3,3) pixels at (W,H)
    R_wc: np.ndarray     # (3,3) world -> camera
    t_wc: np.ndarray     # (3,)
    W: int
    H: int
    dist: np.ndarray = None   # (5,) OpenCV distortion (resolution independent)

    @property
    def centre(self) -> np.ndarray:
        return -self.R_wc.T @ self.t_wc

    @property
    def P(self) -> np.ndarray:
        return self.K @ np.hstack([self.R_wc, self.t_wc[:, None]])

    def scaled(self, W: int, H: int) -> "Camera":
        """Same camera for a frame of a different resolution (same field of view)."""
        sx, sy = W / self.W, H / self.H
        K = self.K.copy()
        K[0] *= sx
        K[1] *= sy
        return Camera(self.name, K, self.R_wc, self.t_wc, W, H, self.dist)

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        """(N,2) distorted pixels -> ideal pinhole pixels (same K). NaNs pass through."""
        import cv2
        out = np.full(uv.shape, np.nan)
        ok = ~np.isnan(uv).any(1)
        if ok.any():
            out[ok] = cv2.undistortPoints(uv[ok].reshape(-1, 1, 2).astype(np.float64), self.K, self.dist, P=self.K).reshape(-1, 2)
        return out

    def project(self, X: np.ndarray) -> np.ndarray:
        """(N,3) world -> (N,2) pixels."""
        Xc = X @ self.R_wc.T + self.t_wc
        uv = Xc @ self.K.T
        return uv[:, :2] / uv[:, 2:3]

    def depth(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.R_wc.T + self.t_wc)[:, 2]


def load_cameras(native: bool = False, W: int = 1920, H: int = 1080) -> dict[str, Camera]:
    """Cameras from the yml files, scaled to the cached 1920x1080 frames (or native 5312x2988)."""
    import cv2
    ki = cv2.FileStorage(f"{VIDEO_DIR}/intri.yml", cv2.FILE_STORAGE_READ)
    ke = cv2.FileStorage(f"{VIDEO_DIR}/extri.yml", cv2.FILE_STORAGE_READ)
    cams = {}
    for name in CAMS:
        cam = Camera(name, ki.getNode(f"K_{name}").mat(), ke.getNode(f"Rot_{name}").mat(),
                     ke.getNode(f"T_{name}").mat().reshape(3), 5312, 2988, ki.getNode(f"dist_{name}").mat().reshape(-1))
        cams[name] = cam if native else cam.scaled(W, H)
    return cams


def triangulate(Ps: list[np.ndarray], uvs: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted linear (DLT) triangulation of one point. Ps: list of 3x4, uvs: (V,2)."""
    rows = []
    for i, (P, uv) in enumerate(zip(Ps, uvs)):
        wi = 1.0 if w is None else w[i]
        rows.append(wi * (uv[0] * P[2] - P[0]))
        rows.append(wi * (uv[1] * P[2] - P[1]))
    X = np.linalg.svd(np.array(rows))[2][-1]
    return X[:3] / X[3]


def triangulate_joints(cams: list[Camera], uv: np.ndarray, conf: np.ndarray | None = None,
                       reproj_thr: float = 12.0, min_views: int = 2):
    """Robust multi-view triangulation of J joints.

    uv: (V, J, 2) pixel predictions, one row per camera in `cams`.
    conf: (V, J) weights (or None).
    Iteratively drops the worst view of a joint while its reprojection error exceeds
    reproj_thr px and more than min_views remain.
    Returns X (J,3), reproj (V,J) error per view (nan if view unused), used (V,J) bool.
    """
    V, J, _ = uv.shape
    Ps = [c.P for c in cams]
    X = np.full((J, 3), np.nan)
    used = np.ones((V, J), bool)
    err = np.full((V, J), np.nan)
    for j in range(J):
        keep = list(range(V))
        while True:
            Xj = triangulate([Ps[i] for i in keep], uv[keep, j],
                             None if conf is None else conf[keep, j])
            e = np.array([np.linalg.norm(cams[i].project(Xj[None])[0] - uv[i, j]) for i in keep])
            z = np.array([cams[i].depth(Xj[None])[0] for i in keep])
            bad = (e > reproj_thr) | (z <= 0)
            if not bad.any() or len(keep) <= min_views:
                break
            keep.pop(int(np.argmax(np.where(bad, e + (z <= 0) * 1e6, -1))))
        X[j] = Xj
        used[:, j] = False
        used[keep, j] = True
        for i in range(V):
            err[i, j] = np.linalg.norm(cams[i].project(Xj[None])[0] - uv[i, j])
    return X, err, used
