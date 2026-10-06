"""Run multi-view HMR over the rrd frame sequence and report accuracy proxies + latency.

usage: python run_seq.py [start] [stop] [step] [detect|track] [out.npz]

Accuracy proxies (no ground truth available):
  reproj   : px error of the 2D predictions against the re-projected triangulated skeleton
  loo      : leave-one-view-out - triangulate from the other views, project into the held-out view
  jitter   : frame-to-frame joint acceleration of the fused 3D skeleton (mm), lower = steadier
"""
import sys
import cv2
import numpy as np

from mvhmr import MultiViewHMR, BODY
from mvcal import CAMS, FRAME_DIR, OUT, triangulate_joints

start, stop, step = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (0, 706, 5)
mode = sys.argv[4] if len(sys.argv) > 4 else "detect"
out = sys.argv[5] if len(sys.argv) > 5 else f"{OUT}/seq_{mode}.npz"

mv = MultiViewHMR()
rows, Xs, T, nviews, loo, idx = [], [], [], [], [], []
for f in range(start, stop, step):
    imgs = [cv2.cvtColor(cv2.imread(f"{FRAME_DIR}/{c}/{f:04d}.jpg"), cv2.COLOR_BGR2RGB) for c in CAMS]
    r = mv.step(imgs, mode=mode)
    idx.append(f)
    T.append([r["timing"][k] for k in ("boxes_ms", "pre_ms", "hmr_ms", "fuse_ms", "total_ms")])
    Xs.append(r["X"])
    nviews.append(len(r["views"]))
    rows.append(np.nanmedian(r["err"][:, BODY]) if r["err"] is not None else np.nan)
    # leave-one-view-out error (held-out view's own 2D vs projection from the others)
    if len(r["views"]) >= 3:
        e = []
        for hold in r["views"]:
            rest = [v for v in r["views"] if v != hold]
            Xl, _, _ = triangulate_joints([r["cams"][v] for v in rest], r["j2d"][rest][:, BODY])
            e.append(np.linalg.norm(r["cams"][hold].project(Xl) - r["j2d"][hold][BODY], axis=1))
        loo.append(np.nanmedian(np.concatenate(e)))
    else:
        loo.append(np.nan)
    if f % 50 == 0:
        print(f, "views", nviews[-1], "reproj %.1f" % rows[-1], "loo %.1f" % loo[-1], "total_ms %.0f" % T[-1][-1], flush=True)

T, Xs, nviews = np.array(T), np.array(Xs), np.array(nviews)
np.savez(out, idx=idx, X=Xs, T=T, nviews=nviews, reproj=rows, loo=loo)
print("\n== summary (%s, %d frames) ==" % (mode, len(idx)))
print("views per frame (mean): %.2f   frames with >=3 views: %d" % (nviews.mean(), (nviews >= 3).sum()))
print("median reproj px: %.2f   median leave-one-out px: %.2f" % (np.nanmedian(rows), np.nanmedian(loo)))
print("latency ms median / p90 : boxes %.1f/%.1f  hmr %.1f/%.1f  fuse %.1f/%.1f  total %.1f/%.1f" % tuple(
    v for c in range(4) for v in (np.median(T[:, [0, 2, 3, 4][c]]), np.percentile(T[:, [0, 2, 3, 4][c]], 90))))
