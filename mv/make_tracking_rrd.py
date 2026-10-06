"""Rerun recording of the marker-guided tracking: fitted mesh vs SAM 3D Body's own mesh vs the Vicon markers, in the Vicon frame, with the five camera
views (mesh points + markers projected onto the video) and live plots of the marker-to-mesh error and the chest surface height (breathing).

  C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe make_tracking_rrd.py            -> tmp/track/tracking.rrd
  C:\\dev\\InstantHMR\\.venv\\Scripts\\rerun.exe tmp\\track\\tracking.rrd           (open it)
"""
import json, os, sys
import cv2, numpy as np
import rerun as rr
import rerun.blueprint as rrb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
T = np.load(TMP + "/track/track.npz", allow_pickle=True); frames = T["frames"]; faces = T["faces"].astype(np.uint32)
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); S = 1280 / 3840; K2 = K.copy(); K2[:2] *= S
cams = {}
for i in range(1, 6):
    c = CJ["cams"]["T%d" % i]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; t = np.array(c["tvec"]); cams[i] = (R, t, -R.T @ t)

def proj(i, X):
    R, t, _ = cams[i]; Xc = X @ R.T + t; uv = Xc @ K2.T; return uv[:, :2] / uv[:, 2:3], Xc[:, 2]

rr.init("marker_guided_tracking", spawn=False)
bp = rrb.Blueprint(rrb.Horizontal(
    rrb.Spatial3DView(name="Vicon frame: fitted mesh (green), SAM 3D Body (red), markers (cyan)", origin="world", contents=["world/mesh_fit", "world/mesh_sam3d", "world/markers", "world/cameras/*"], eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)),
    rrb.Vertical(rrb.Grid(*[rrb.Spatial2DView(name="T%d" % i, origin=f"world/cameras/T{i}/image") for i in range(1, 6)], grid_columns=2),
                 rrb.TimeSeriesView(name="marker-to-mesh distance (mm): lower is better", origin="plots/distance"),
                 rrb.TimeSeriesView(name="chest surface height under the chest markers (mm, mean removed) = breathing", origin="plots/chest"), row_shares=[5, 2, 2]),
    column_shares=[5, 6]), rrb.TimePanel(state="expanded"))
rr.save(TMP + "/track/tracking.rrd", default_blueprint=bp)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
for i, (R, t, c) in cams.items():
    rr.log(f"world/cameras/T{i}", rr.Transform3D(translation=c, mat3x3=R.T), static=True); rr.log(f"world/cameras/T{i}", rr.Pinhole(image_from_camera=K2, resolution=[1280, 720], image_plane_distance=0.3), static=True)
for name, col in (("fit", [60, 200, 90]), ("sam3d", [255, 90, 90]), ("chest_fit", [60, 200, 90]), ("chest_sam3d", [255, 90, 90]), ("chest_marker", [0, 200, 255])): pass
rr.log("plots/distance/fit", rr.SeriesLines(colors=[60, 200, 90], names="marker-guided fit", widths=2), static=True); rr.log("plots/distance/sam3d", rr.SeriesLines(colors=[255, 90, 90], names="SAM 3D Body alone", widths=2), static=True)
rr.log("plots/chest/vicon_marker", rr.SeriesLines(colors=[0, 200, 255], names="Vicon chest markers", widths=2), static=True); rr.log("plots/chest/fit", rr.SeriesLines(colors=[60, 200, 90], names="fitted mesh surface", widths=2), static=True)
rr.log("plots/chest/sam3d", rr.SeriesLines(colors=[255, 90, 90], names="SAM 3D Body mesh surface", widths=2), static=True)
cm = T["chest_mk"] * 1000; cf = T["chest_fit"] * 1000; cb = T["chest_base"] * 1000
cm0, cf0, cb0 = np.nanmean(cm), np.nanmean(cf), np.nanmean(cb)
def normals(v):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]); vn = np.zeros_like(v)
    for k in range(3): np.add.at(vn, faces[:, k], fn)
    return (vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
rr.log("world/mesh_fit", rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=[70, 200, 100, 255]), static=True)
rr.log("world/mesh_sam3d", rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=[255, 90, 90, 90]), static=True)
for n, f in enumerate(frames):
    if n % 2: continue
    rr.set_time("frame", sequence=int(f)); rr.set_time("video_time", duration=float(f) / 29.97)
    vf, vb, mk = T["vf"][n], T["vb"][n], T["mk"][n]; ok = ~np.isnan(mk).any(1)
    rr.log("world/mesh_fit", rr.Mesh3D.from_fields(vertex_positions=vf, vertex_normals=normals(vf.astype(np.float64))))
    if not np.isnan(vb).any(): rr.log("world/mesh_sam3d", rr.Mesh3D.from_fields(vertex_positions=vb, vertex_normals=normals(vb.astype(np.float64))))
    rr.log("world/markers", rr.Points3D(mk[ok], colors=[0, 220, 255], radii=0.009))
    for i in range(1, 6):
        p = f"{TMP}/mmc17_frames/T{i}/{int(f):04d}.jpg"
        if not os.path.exists(p): continue
        rr.log(f"world/cameras/T{i}/image", rr.EncodedImage(contents=open(p, "rb").read(), media_type="image/jpeg"))
        uv, z = proj(i, vf[::7]); m = z > 0; rr.log(f"world/cameras/T{i}/image/fit", rr.Points2D(uv[m], colors=[70, 255, 110], radii=1.2))
        if not np.isnan(vb).any(): uv, z = proj(i, vb[::7]); m = z > 0; rr.log(f"world/cameras/T{i}/image/sam3d", rr.Points2D(uv[m], colors=[255, 90, 90], radii=1.0))
        uv, z = proj(i, mk[ok]); rr.log(f"world/cameras/T{i}/image/markers", rr.Points2D(uv[z > 0], colors=[0, 220, 255], radii=5))
    rr.log("plots/distance/fit", rr.Scalars(float(T["d_fit"][n]))); rr.log("plots/distance/sam3d", rr.Scalars(float(T["d_base"][n])))
    rr.log("plots/chest/vicon_marker", rr.Scalars(float(cm[n] - cm0))); rr.log("plots/chest/fit", rr.Scalars(float(cf[n] - cf0)))
    if not np.isnan(cb[n]): rr.log("plots/chest/sam3d", rr.Scalars(float(cb[n] - cb0)))
print("wrote", TMP + "/track/tracking.rrd", len(frames), "frames")
