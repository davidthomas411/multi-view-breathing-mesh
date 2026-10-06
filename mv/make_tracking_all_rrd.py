"""Rerun recording with ALL THREE meshes in the Vicon frame, plus the five camera views and live plots:
   red   = SAM 3D Body, ONE camera (T4)                       - no markers
   blue  = SAM 3D Body, markerless MULTI-VIEW mesh fit (5 cameras)  - no markers (mv_fit.py)
   green = marker-guided fit                                  - uses the Vicon markers (the reference)
   cyan  = Vicon markers
  C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe make_tracking_all_rrd.py   ->  recordings/tracking_all_meshes.rrd
  C:\\dev\\InstantHMR\\.venv\\Scripts\\rerun.exe recordings\\tracking_all_meshes.rrd"""
import json, os
import cv2, numpy as np
import rerun as rr, rerun.blueprint as rrb
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"
T = np.load(TMP + "/track/track.npz", allow_pickle=True); faces = T["faces"].astype(np.uint32); tf = {int(f): i for i, f in enumerate(T["frames"])}
MV = np.load(TMP + "/mv_fit/mv_fit.npz"); EV = np.load(TMP + "/mv_fit/eval.npz"); E2 = np.load(TMP + "/mv_fit/eval2.npz")
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); K2 = K.copy(); K2[:2] *= 1280 / 3840; cams = {}
for i in range(1, 6):
    c = CJ["cams"]["T%d" % i]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; t = np.array(c["tvec"]); cams[i] = (R, t, -R.T @ t)
def proj(i, X): R, t, _ = cams[i]; Xc = X @ R.T + t; uv = Xc @ K2.T; return uv[:, :2] / uv[:, 2:3], Xc[:, 2]
def normals(v):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]); vn = np.zeros_like(v)
    for k in range(3): np.add.at(vn, faces[:, k], fn)
    return (vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
bp = rrb.Blueprint(rrb.Horizontal(
    rrb.Spatial3DView(name="Vicon frame: red = SAM 3D Body (1 camera), blue = SAM 3D Body multi-view (markerless), green = marker-guided (reference), cyan = Vicon markers", origin="world",
                      contents=["world/mesh_single", "world/mesh_multiview", "world/mesh_markerguided", "world/markers", "world/cameras/*"], eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)),
    rrb.Vertical(rrb.Grid(*[rrb.Spatial2DView(name="T%d" % i, origin=f"world/cameras/T{i}/image") for i in range(1, 6)], grid_columns=2),
                 rrb.TimeSeriesView(name="marker-to-mesh distance (mm): lower is better", origin="plots/distance"),
                 rrb.TimeSeriesView(name="chest surface height under the chest markers (mm, mean removed) = breathing", origin="plots/chest"), row_shares=[5, 2, 2]), column_shares=[5, 6]), rrb.TimePanel(state="expanded"))
os.makedirs(ROOT + "/recordings", exist_ok=True); rr.init("tracking_all_meshes", spawn=False); rr.save(ROOT + "/recordings/tracking_all_meshes.rrd", default_blueprint=bp); rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
for i, (R, t, c) in cams.items():
    rr.log(f"world/cameras/T{i}", rr.Transform3D(translation=c, mat3x3=R.T), static=True); rr.log(f"world/cameras/T{i}", rr.Pinhole(image_from_camera=K2, resolution=[1280, 720], image_plane_distance=0.3), static=True)
RED, BLUE, GREEN, GREY, CYAN = [255, 90, 90], [90, 150, 255], [70, 200, 100], [160, 160, 160], [0, 200, 255]
for name, col in (("single_view_T4", RED), ("multi_view", BLUE), ("marker_guided", GREEN), ("typical_single_view", GREY)): rr.log(f"plots/distance/{name}", rr.SeriesLines(colors=col, names=name.replace("_", " "), widths=2), static=True)
for name, col in (("vicon_markers", CYAN), ("single_view_T4", RED), ("multi_view", BLUE), ("marker_guided", GREEN)): rr.log(f"plots/chest/{name}", rr.SeriesLines(colors=col, names=name.replace("_", " "), widths=2), static=True)
for nm, col in (("mesh_single", RED + [90]), ("mesh_multiview", BLUE + [110]), ("mesh_markerguided", GREEN + [255])): rr.log(f"world/{nm}", rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=col), static=True)
mk0 = {k: np.nanmean(EV[k]) for k in ("chest_mk", "chest_sv", "chest_mv", "chest_fit")}
sv_typ = np.nanmean([np.nanmedian(E2["sv%d" % v], axis=1) for v in range(1, 6)], axis=0)
for n, f in enumerate(MV["frames"]):
    f = int(f); i = tf[f]; rr.set_time("frame", sequence=f); rr.set_time("video_time", duration=f / 29.97)
    vf, vb, vm, mk = T["vf"][i].astype(np.float64), T["vb"][i].astype(np.float64), MV["X"][n].astype(np.float64), T["mk"][i].astype(np.float64); ok = ~np.isnan(mk).any(1)
    rr.log("world/mesh_markerguided", rr.Mesh3D.from_fields(vertex_positions=vf, vertex_normals=normals(vf)))
    rr.log("world/mesh_multiview", rr.Mesh3D.from_fields(vertex_positions=vm, vertex_normals=normals(vm)))
    if not np.isnan(vb).any(): rr.log("world/mesh_single", rr.Mesh3D.from_fields(vertex_positions=vb, vertex_normals=normals(vb)))
    rr.log("world/markers", rr.Points3D(mk[ok], colors=CYAN, radii=0.009))
    for c in range(1, 6):
        p = f"{TMP}/mmc17_frames/T{c}/{f:04d}.jpg"
        if not os.path.exists(p): continue
        rr.log(f"world/cameras/T{c}/image", rr.EncodedImage(contents=open(p, "rb").read(), media_type="image/jpeg"))
        for nm, v, col, rad in (("markerguided", vf, [70, 255, 110], 1.1), ("multiview", vm, BLUE, 1.1), ("single", vb, RED, 1.0)):
            if np.isnan(v).any(): continue
            uv, z = proj(c, v[::7]); m = z > 0; rr.log(f"world/cameras/T{c}/image/{nm}", rr.Points2D(uv[m], colors=col, radii=rad))
        uv, z = proj(c, mk[ok]); rr.log(f"world/cameras/T{c}/image/markers", rr.Points2D(uv[z > 0], colors=CYAN, radii=5))
    e = int(np.where(EV["frames"] == f)[0][0]); e2 = int(np.where(E2["frames"] == f)[0][0])
    rr.log("plots/distance/single_view_T4", rr.Scalars(float(np.nanmedian(E2["sv4"][e2])))); rr.log("plots/distance/multi_view", rr.Scalars(float(np.nanmedian(E2["mv"][e2])))); rr.log("plots/distance/marker_guided", rr.Scalars(float(np.nanmedian(E2["fit"][e2])))); rr.log("plots/distance/typical_single_view", rr.Scalars(float(sv_typ[e2])))
    for nm, k in (("vicon_markers", "chest_mk"), ("single_view_T4", "chest_sv"), ("multi_view", "chest_mv"), ("marker_guided", "chest_fit")):
        v = EV[k][e] - mk0[k]
        if not np.isnan(v): rr.log(f"plots/chest/{nm}", rr.Scalars(float(v)))
print("wrote recordings/tracking_all_meshes.rrd:", len(MV["frames"]), "frames")
