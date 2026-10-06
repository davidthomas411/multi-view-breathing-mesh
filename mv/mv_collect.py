"""Collect per-view SAM 3D Body meshes for the multi-view mesh fit (every 6th frame of the MMC17 TDB trial, cameras T1-T5).

Videos are decoded sequentially (5 threads); SAM 3D Body runs on every selected frame and view with the calibrated K.  Saved per frame:
tmp/mv_sam3d/<frame>.npz : verts (5, Vs, 3) camera-frame vertices (every 3rd vertex), vidx, cam_t (5,3), j3d (5,70,3), box (5,4), ok (5,)
No Vicon information is used here.   Run with the SAM3D env:  python mv_collect.py
"""
import os, sys, json, time
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M, sam3d_mv as S, trial_io as TI

OUTD = TI.tdir("mv_sam3d")
K = M.K; frames = list(range(TI.frame_range()[0], TI.frame_range()[1] + 1, 6)); want = set(frames)
caps = [TI.Capture(i) for i in range(1, 6)]
pool = ThreadPoolExecutor(5); vidx = np.arange(0, 18439, 3)
t0 = time.time(); done = 0
for f in range(0, frames[-1] + 1):
    got = list(pool.map(lambda c: c.read(), caps))                     # decode all five streams in parallel (we cannot skip frames in H.264)
    if f not in want: continue
    p = f"{OUTD}/{f:04d}.npz"
    if os.path.exists(p): continue
    verts = np.full((5, len(vidx), 3), np.nan, np.float32); ct = np.full((5, 3), np.nan, np.float32); j3 = np.full((5, 70, 3), np.nan, np.float32); bx = np.full((5, 4), np.nan, np.float32); ok = np.zeros(5, bool)
    for i, (r, im) in enumerate(got):
        if not r: continue
        d = S.run_view(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), K, keep_mesh=True)
        if d is None: continue
        verts[i] = d["verts"][vidx]; ct[i] = d["cam_t"]; j3[i] = d["j3d"]; bx[i] = d["box"]; ok[i] = True
    np.savez(p, verts=verts, vidx=vidx, cam_t=ct, j3d=j3, box=bx, ok=ok); done += 1
    if done % 10 == 0 or done == 1: print("frame %d (%d/%d) %.0fs, views ok %d/5" % (f, done, len(frames), time.time() - t0, ok.sum()), flush=True)
print("done: %d frames in %.0fs" % (done, time.time() - t0))
