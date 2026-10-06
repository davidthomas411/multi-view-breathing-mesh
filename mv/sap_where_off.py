"""Where is the Sapiens2 cold-start mesh off?  Compare it vertex by vertex (same MHR topology = same anatomical place) with the marker-guided mesh (the reference), by region along the body
axis, and compare the silhouettes of the cold-start mesh and of the multi-view mesh with the Sapiens2 person mask.   usage: python sap_where_off.py [suffix]"""
import json, os, sys
import cv2, numpy as np
import trial_io as TI
TMP = TI.TMP; suffix = sys.argv[1] if len(sys.argv) > 1 else ""
SF = np.load(f"{TMP}/sap_fit/sapfit_0.4b_rot{suffix}.npz"); T = np.load(TMP + "/track/track.npz", allow_pickle=True); tf = {int(f): i for i, f in enumerate(T["frames"])}; faces = T["faces"].astype(np.int32)
MV = np.load(TMP + "/mv_fit/mv_fit.npz"); mvf = [int(f) for f in MV["frames"]]; CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); D = np.array(CJ["dist"])
def sil(V, cam):
    c = CJ["cams"][f"T{cam}"]; uv = cv2.projectPoints(V.reshape(-1, 1, 3).astype(np.float64), np.array(c["rvec"]), np.array(c["tvec"]), K, D)[0].reshape(-1, 2) * 0.5; m = np.zeros((1080, 1920), np.uint8)
    for tri in np.round(uv[faces]).astype(np.int32): cv2.fillConvexPoly(m, tri, 1)
    return m
def person_mask(f, cam):
    seg = np.load(f"{TMP}/sapiens2/{f:04d}_T{cam}_0.4b_rot.npz")["seg"]; pm = (seg > 0).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(pm)
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) if n > 1 else pm
iou = lambda a, b: float((a.astype(bool) & b.astype(bool)).sum() / max((a.astype(bool) | b.astype(bool)).sum(), 1))
print("per-vertex distance of the cold-start mesh to the marker-guided mesh (mm), by position along the body (Vicon Y, towards the head):")
agg = {}
for n, f in enumerate(SF["frames"]):
    f = int(f); X = SF["X"][n].astype(np.float64); R = T["vf"][tf[f]].astype(np.float64); M = MV["X"][mvf.index(f)].astype(np.float64); d = np.linalg.norm(X - R, axis=1) * 1000; dm = np.linalg.norm(M - R, axis=1) * 1000
    dz = (X - R)[:, 2] * 1000; y = R[:, 1]; bins = [(0.0, 0.55), (0.55, 0.8), (0.8, 1.05), (1.05, 1.25), (1.25, 2.5)]; names = ["feet / lower legs", "knees / thighs", "pelvis / abdomen", "chest / shoulders", "head and arms above"]
    if n == 0: print("  mesh Y range %.2f-%.2f m" % (y.min(), y.max()))
    print(f"frame {f}: all vertices median {np.median(d):.0f} mm, p90 {np.percentile(d, 90):.0f}, p99 {np.percentile(d, 99):.0f}; multi-view mesh median {np.median(dm):.0f}, p90 {np.percentile(dm, 90):.0f}; mean vertical offset {dz.mean():+.0f} mm")
    for (lo, hi), nm in zip(bins, names):
        s = (y >= lo) & (y < hi)
        if s.sum() > 50: print(f"    {nm:20s} ({s.sum():5d} vertices): cold start median {np.median(d[s]):5.0f} p90 {np.percentile(d[s], 90):5.0f} mm | multi-view median {np.median(dm[s]):5.0f} | vertical offset of the cold start {np.mean(dz[s]):+5.0f} mm")
    agg[f] = []
    for c in range(1, 6):
        pm = person_mask(f, c); agg[f].append((c, iou(pm, sil(X, c)), iou(pm, sil(M, c)), iou(pm, sil(R, c))))
print("\nIoU of the Sapiens2 person mask with the mesh silhouette, per camera (cold start | multi-view | marker-guided), median over the frames:")
for c in range(1, 6): print("  T%d: %.2f | %.2f | %.2f" % ((c,) + tuple(np.median([[a[1], a[2], a[3]] for f in agg for a in agg[f] if a[0] == c], axis=0))))
print("  all cameras: %.2f | %.2f | %.2f" % tuple(np.median([[a[1], a[2], a[3]] for f in agg for a in agg[f]], axis=0)))
