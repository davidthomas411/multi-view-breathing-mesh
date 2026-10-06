"""Live multi-view fitting viewer (Rerun): 6 camera feeds with 2D fits, the fused 3D skeleton in the
room (TrueBeam mesh + camera frusta) and live latency / quality plots.

  python live_viewer.py                      # spawn the native Rerun viewer, track mode, real-time pacing
  python live_viewer.py --fast               # no pacing: as fast as the pipeline goes
  python live_viewer.py --mode detect        # run the person detector on every view every frame
  python live_viewer.py --save tmp/run.rrd   # also/instead write a recording (add --no-spawn for save only)
  python live_viewer.py --start 560 --stop 706

Latency is the real per-frame cost: load (decode 6 frames, threaded) + boxes + preprocess + HMR (one batch-6
ONNX call) + fuse.  'viz' (time spent logging to Rerun) is plotted separately and is NOT part of the budget.
"""
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from mvcal import CAMS, FRAME_DIR, OUT, load_cameras
from mvhmr import BODY, MultiViewHMR

import sys
sys.path.insert(0, os.environ.get("INSTANTHMR_REPO", "C:/dev/InstantHMR"))
from instanthmr.skeleton import SKELETON_EDGES  # noqa: E402

EDGES = [(a, b) for a, b in SKELETON_EDGES if a in BODY and b in BODY]
BUDGET_MS = 100.0
SHOW_W = 960  # width of the JPEG sent to the viewer (overlay coordinates stay in 1920x1080 space)


def blueprint():
    cams = [rrb.Spatial2DView(name=f"cam {c}", origin=f"world/cameras/{c}/image", contents=f"world/cameras/{c}/image/**") for c in CAMS]
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="room (world)", origin="world", contents=["world/**", "-world/cameras/*/image/**"]),
                rrb.TimeSeriesView(name="latency (ms)", origin="latency"),
                row_shares=[3, 2],
            ),
            rrb.Vertical(rrb.Grid(*cams, grid_columns=2),
                         rrb.TimeSeriesView(name="fit quality (px)", origin="quality"), row_shares=[4, 1]),
            column_shares=[2, 3],
        ),
        rrb.TimePanel(state="collapsed"),
    )


def log_static(cams):
    m = np.load(f"{OUT}/scene_meshes.npz")
    rr.log("world/linac", rr.Mesh3D(vertex_positions=m["linac_V"], triangle_indices=m["linac_F"],
                                    albedo_factor=[200, 200, 205, 90]), static=True)
    for c in CAMS:
        cam = cams[c]
        rr.log(f"world/cameras/{c}", rr.Transform3D(translation=cam.centre, mat3x3=cam.R_wc.T), static=True)
        rr.log(f"world/cameras/{c}", rr.Pinhole(image_from_camera=cam.K, resolution=[cam.W, cam.H], image_plane_distance=0.5), static=True)
    for name, ms in (("budget", BUDGET_MS),):
        rr.log(f"latency/{name}", rr.SeriesLines(colors=[255, 60, 60], names=name, widths=1), static=True)
    for n, col in (("total", [255, 255, 255]), ("load", [150, 150, 150]), ("boxes", [255, 170, 0]), ("hmr", [60, 160, 255]),
                   ("fuse", [80, 220, 120]), ("viz (not in total)", [120, 80, 160])):
        rr.log(f"latency/{n}", rr.SeriesLines(colors=col, names=n, widths=1.5), static=True)
    rr.log("quality/reproj_px", rr.SeriesLines(colors=[80, 220, 120], names="reprojection (median px @1080p)", widths=1.5), static=True)
    rr.log("quality/views", rr.SeriesLines(colors=[255, 170, 0], names="views used", widths=1.5), static=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="track", choices=["track", "detect"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=706)
    ap.add_argument("--fast", action="store_true", help="don't pace to 30 fps")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-spawn", action="store_true")
    a = ap.parse_args()

    rr.init("multiview_sgrt", default_blueprint=blueprint())
    if a.save:
        rr.save(a.save, default_blueprint=blueprint())
    if not a.no_spawn:
        rr.spawn(memory_limit="50%")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)   # world axes: cameras use OpenCV convention; world is as calibrated

    cams = load_cameras()
    log_static(cams)
    mv = MultiViewHMR(cams=cams)
    pool = ThreadPoolExecutor(6)

    def load(f):
        return list(pool.map(lambda c: cv2.imread(f"{FRAME_DIR}/{c}/{f:04d}.jpg"), CAMS))

    f = a.start
    nxt = pool.submit(load, f)
    t_start = time.perf_counter()
    n_done = 0
    while True:
        t0 = time.perf_counter()
        bgr = nxt.result()
        load_ms = (time.perf_counter() - t0) * 1000   # time the pipeline actually waited for frames (prefetched)
        if f + 1 < a.stop:
            nxt = pool.submit(load, f + 1)
        imgs = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bgr]
        r = mv.step(imgs, mode=a.mode)
        T = r["timing"]
        total = load_ms + T["total_ms"]

        tv = time.perf_counter()
        rr.set_time("frame", sequence=f)
        rr.set_time("video_time", duration=f / 29.97)
        for i, c in enumerate(CAMS):
            ok, buf = cv2.imencode(".jpg", cv2.resize(bgr[i], (SHOW_W, SHOW_W * 9 // 16)), [cv2.IMWRITE_JPEG_QUALITY, 80])
            rr.log(f"world/cameras/{c}/image", rr.EncodedImage(contents=buf.tobytes(), media_type="image/jpeg"))
            # overlays live under .../image so they share the (1920x1080) pixel space; the JPEG is scaled by Pinhole resolution
            pr = r["cams"][i].project(r["X"]) if not np.isnan(r["X"][BODY]).any() else None
            if pr is not None:
                strips = [[pr[x], pr[y]] for x, y in EDGES]
                rr.log(f"world/cameras/{c}/image/fused", rr.LineStrips2D(strips, colors=[60, 255, 90], radii=2.5))
            else:
                rr.log(f"world/cameras/{c}/image/fused", rr.Clear(recursive=False))
            if i in r["views"]:
                pts = r["j2d"][i][BODY]
                err = r["err"][i][BODY]
                col = np.where((err > 8)[:, None], [255, 70, 70], [255, 235, 60]).astype(np.uint8)
                rr.log(f"world/cameras/{c}/image/raw_2d", rr.Points2D(pts, colors=col, radii=4))
                b = r["boxes"][i]
                rr.log(f"world/cameras/{c}/image/box", rr.Boxes2D(array=b, array_format=rr.Box2DFormat.XYXY, colors=[0, 200, 255], radii=1.5))
            else:
                rr.log(f"world/cameras/{c}/image/raw_2d", rr.Clear(recursive=False))
                rr.log(f"world/cameras/{c}/image/box", rr.Clear(recursive=False))
        X = r["X"]
        if not np.isnan(X[BODY]).any():
            rr.log("world/body/joints", rr.Points3D(X[BODY], colors=[60, 255, 90], radii=0.015))
            rr.log("world/body/bones", rr.LineStrips3D([[X[x], X[y]] for x, y in EDGES], colors=[60, 255, 90], radii=0.006))
        rep = np.nanmedian(r["err"][:, BODY]) if r["err"] is not None else np.nan
        viz_ms = (time.perf_counter() - tv) * 1000
        for name, v in (("total", total), ("load", load_ms), ("boxes", T["boxes_ms"]), ("hmr", T["hmr_ms"] + T["pre_ms"]),
                        ("fuse", T["fuse_ms"]), ("viz (not in total)", viz_ms), ("budget", BUDGET_MS)):
            rr.log(f"latency/{name}", rr.Scalars(v))
        rr.log("quality/reproj_px", rr.Scalars(rep if not np.isnan(rep) else 0.0))
        rr.log("quality/views", rr.Scalars(len(r["views"])))

        n_done += 1
        if f % 30 == 0:
            print(f"frame {f}: total {total:.0f} ms (load {load_ms:.0f}, boxes {T['boxes_ms']:.0f}, hmr+pre {T['hmr_ms'] + T['pre_ms']:.0f}, "
                  f"fuse {T['fuse_ms']:.0f}, viz {viz_ms:.0f})  views {len(r['views'])}  reproj {rep:.1f}px", flush=True)
        f += 1
        if f >= a.stop:
            if not a.loop:
                break
            f = a.start
            mv.prev_X = None
            nxt = pool.submit(load, f)
        if not a.fast:
            target = t_start + n_done / 29.97
            time.sleep(max(0.0, target - time.perf_counter()))
    print(f"done: {n_done} frames in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
