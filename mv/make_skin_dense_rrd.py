"""Rerun recording of the DENSE skin-only chest tracking (4,000 seeds on the chest; every reflective ball painted out of the video; ~1,100 points triangulated from >=3 cameras).

Shows: the ball-free camera views with the tracked texture points; the triangulated 3D points coloured by how far they have risen/fallen (blue = down, red = up);
the Vicon chest markers in 3D (reference only - they are NOT in the video or the tracking); and the Vicon vs skin-tracked chest height.
  C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe make_skin_dense_rrd.py [tag]   ->  recordings/skin_tracking_dense.rrd      (tag default m41_e33_dense)"""
import json, os, sys
import cv2, numpy as np
import rerun as rr, rerun.blueprint as rrb
import matplotlib; matplotlib.use("Agg")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; CT = TMP + "/chest_track"
TAG = sys.argv[1] if len(sys.argv) > 1 else "m41_e33_dense"; OUT = ROOT + "/recordings/skin_tracking_dense.rrd"
E = np.load(f"{CT}/eval_s0.5_{TAG}.npz", allow_pickle=True); keep = np.where(E["keep"])[0]; frames = E["frames"]; D = E["D"]; Xref = E["Xref"]; a, b = E["a"], E["b"]
L = {i: np.load(f"{CT}/lk_T{i}_s0.5_{TAG}.npz") for i in range(1, 6)}
T = np.load(TMP + "/track/track.npz", allow_pickle=True); names = [str(n) for n in T["names"]]; tf = {int(f): i for i, f in enumerate(T["frames"])}
chest = [i for i, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]; faces = T["faces"].astype(np.uint32)
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); K2 = K.copy(); K2[:2] *= 1280 / 3840
cams = {}
for i in range(1, 6):
    c = CJ["cams"]["T%d" % i]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; t = np.array(c["tvec"]); cams[i] = (R, t, -R.T @ t)
cmap = matplotlib.colormaps["coolwarm"]; col = lambda dz: (cmap(0.5 + np.clip(np.asarray(dz, float), -10, 10) / 20.0)[:, :3] * 255).astype(np.uint8)
def normals(v):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]); vn = np.zeros_like(v)
    for k in range(3): np.add.at(vn, faces[:, k], fn)
    return (vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
os.makedirs(ROOT + "/recordings", exist_ok=True)
bp = rrb.Blueprint(rrb.Horizontal(
    rrb.Spatial3DView(name="3D: %d skin-tracked points (colour = rise/fall in mm), Vicon chest markers (cyan, reference only)" % len(keep), origin="world", contents=["world/skin_points", "world/vicon_chest", "world/mesh_context", "world/cameras/*"], eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)),
    rrb.Vertical(rrb.Grid(*[rrb.Spatial2DView(name="T%d (balls painted out)" % i, origin=f"world/cameras/T{i}/image") for i in range(1, 6)], grid_columns=2),
                 rrb.TimeSeriesView(name="chest height, mean removed (mm): Vicon vs skin tracking", origin="plots/chest"), row_shares=[6, 2]), column_shares=[5, 6]), rrb.TimePanel(state="expanded"))
rr.init("skin_tracking_dense", spawn=False); rr.save(OUT, default_blueprint=bp)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
for i, (R, t, c) in cams.items():
    rr.log(f"world/cameras/T{i}", rr.Transform3D(translation=c, mat3x3=R.T), static=True); rr.log(f"world/cameras/T{i}", rr.Pinhole(image_from_camera=K2, resolution=[1280, 720], image_plane_distance=0.3), static=True)
rr.log("plots/chest/vicon", rr.SeriesLines(colors=[0, 200, 255], names="Vicon chest markers", widths=2.5), static=True); rr.log("plots/chest/skin", rr.SeriesLines(colors=[255, 90, 90], names="skin tracking, %d points (balls painted out)" % len(keep), widths=2), static=True)
rr.log("world/mesh_context", rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=[200, 200, 200, 70]), static=True)
n_logged = 0
for n, f in enumerate(frames):
    if n % 2: continue
    f = int(f); rr.set_time("frame", sequence=f); rr.set_time("video_time", duration=f / 29.97)
    dz = D[n, :, 2]; P3 = (Xref + D[n]) / 1000.0; good = ~np.isnan(P3).any(1)
    rr.log("world/skin_points", rr.Points3D(P3[good], colors=col(dz[good]), radii=0.0035))
    if f in tf:
        mk = T["mk"][tf[f]][chest].astype(np.float64); okm = ~np.isnan(mk).any(1); rr.log("world/vicon_chest", rr.Points3D(mk[okm], colors=[0, 220, 255], radii=0.009))
        v = T["vf"][tf[f]].astype(np.float64); rr.log("world/mesh_context", rr.Mesh3D.from_fields(vertex_positions=v, vertex_normals=normals(v)))
    for i in range(1, 6):
        p = f"{TMP}/skin_frames/T{i}/{f:04d}.jpg"
        if not os.path.exists(p): continue
        rr.log(f"world/cameras/T{i}/image", rr.EncodedImage(contents=open(p, "rb").read(), media_type="image/jpeg"))
        okp = L[i]["ok"][n][keep] & good; uv = L[i]["pts"][n][keep][okp] / 3.0
        rr.log(f"world/cameras/T{i}/image/tracked", rr.Points2D(uv, colors=col(dz[okp]), radii=2.2))
    rr.log("plots/chest/vicon", rr.Scalars(float(a[n]))); rr.log("plots/chest/skin", rr.Scalars(float(b[n]))); n_logged += 1
print("wrote", OUT, ":", n_logged, "frames,", len(keep), "tracked points")
