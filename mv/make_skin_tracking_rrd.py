"""Rerun recording of the dense SKIN-ONLY tracking for one trial (TRIAL=TDB or ADB): ball-free camera views with the tracked points, the triangulated 3D skin points (colour = rise/fall in mm),
the markerless multi-view mesh as context, the Vicon chest balls (reference only: they are neither in the video nor in the tracking), and the thoracic / abdominal displacement, Vicon vs skin tracking.
  python make_skin_tracking_rrd.py [tag]      env TRIAL (default TDB); default tag m41_e33_dense (TDB) or m41mv_e33_wide_f25_c2 (ADB)
  needs: tmp/skin_frames[_TRIAL]/ (skin_frames.py), tmp/chest_track[_TRIAL]/ (chest_lk_masked.py, chest_eval.py), tmp/mv_fit[_TRIAL]/mv_fit.npz, tmp/thoracoabdominal/markerless_<TRIAL>_<tag>.json
  -> recordings/skin_tracking_<TRIAL>_<tag>.rrd"""
import csv, json, os, re, sys
import cv2, numpy as np
import rerun as rr, rerun.blueprint as rrb
import matplotlib; matplotlib.use("Agg")
import trial_io as TI
TRIAL = TI.TRIAL; TAG = sys.argv[1] if len(sys.argv) > 1 else ("m41_e33_dense" if TRIAL == "TDB" else "m41mv_e33_wide_f25_c2"); ROOT, TMP = TI.ROOT, TI.TMP
CT = TI.tdir("chest_track", make=False); SF = TI.tdir("skin_frames", make=False); LKTAG = re.sub(r"_(f\d+|c\d)", "", TAG)
E = np.load(f"{CT}/eval_s0.5_{TAG}.npz", allow_pickle=True); keep = np.where(E["keep"])[0]; frames = E["frames"]; D = E["D"]; Xref = E["Xref"]
L = {i: np.load(f"{CT}/lk_T{i}_s0.5_{LKTAG}.npz") for i in range(1, 6)}; MV = np.load(TI.tdir("mv_fit", make=False) + "/mv_fit.npz"); mvf = [int(f) for f in MV["frames"]]; faces = MV["faces"].astype(np.uint32)
M = json.load(open(f"{TMP}/thoracoabdominal/markerless_{TRIAL}_{TAG}.json")); rowof = M["row_of"]; ser = M["series"]
rows = list(csv.reader(open(TI.vicon_csv()))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]; VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) * 1e-3 for r in rows[1:]}; chest = [names.index(n) for n in rowof]
CJ = json.load(open(TI.calib_path())); K = np.array(CJ["K"]); K2 = K.copy(); K2[:2] *= 1280 / 3840; cams = {}
for i in range(1, 6): c = CJ["cams"]["T%d" % i]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; t = np.array(c["tvec"]); cams[i] = (R, t, -R.T @ t)
cmap = matplotlib.colormaps["coolwarm"]; col = lambda dz: (cmap(0.5 + np.clip(np.asarray(dz, float), -10, 10) / 20.0)[:, :3] * 255).astype(np.uint8)
def normals(v):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]); vn = np.zeros_like(v)
    for k in range(3): np.add.at(vn, faces[:, k], fn)
    return (vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
def detr(x):
    x = np.asarray(x, float); t = np.arange(len(x)); ok = ~np.isnan(x); return x - np.polyval(np.polyfit(t[ok], x[ok], 2), t)
reg = {}
for nm, rws in (("thoracic", (1, 2)), ("abdominal", (3, 4))):
    use = [n for n in rowof if rowof[n] in rws and ser["markerless"].get(n) is not None]
    reg[nm] = (np.nanmean([detr(ser["vicon"][n]) for n in use], axis=0), np.nanmean([detr([np.nan if x is None else x for x in ser["markerless"][n]]) for n in use], axis=0))
bp = rrb.Blueprint(rrb.Horizontal(
    rrb.Spatial3DView(name="%s: %d skin-tracked points (colour = rise/fall, mm); markerless multi-view mesh; Vicon balls (cyan, reference only)" % (TRIAL, len(keep)), origin="world", contents=["world/skin_points", "world/vicon_chest", "world/mesh_context", "world/cameras/*"], eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)),
    rrb.Vertical(rrb.Grid(*[rrb.Spatial2DView(name="T%d (balls painted out)" % i, origin=f"world/cameras/T{i}/image") for i in range(1, 6)], grid_columns=2),
                 rrb.TimeSeriesView(name="thoracic (rows 1-2) displacement, mm: Vicon vs skin tracking", origin="plots/thoracic"), rrb.TimeSeriesView(name="abdominal (rows 3-4) displacement, mm: Vicon vs skin tracking", origin="plots/abdominal"), row_shares=[6, 2, 2]), column_shares=[5, 6]), rrb.TimePanel(state="expanded"))
os.makedirs(ROOT + "/recordings", exist_ok=True); OUT = ROOT + f"/recordings/skin_tracking_{TRIAL}_{TAG}.rrd"; rr.init(f"skin_tracking_{TRIAL}", spawn=False); rr.save(OUT, default_blueprint=bp); rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
for i, (R, t, c) in cams.items(): rr.log(f"world/cameras/T{i}", rr.Transform3D(translation=c, mat3x3=R.T), static=True); rr.log(f"world/cameras/T{i}", rr.Pinhole(image_from_camera=K2, resolution=[1280, 720], image_plane_distance=0.3), static=True)
for nm in ("thoracic", "abdominal"): rr.log(f"plots/{nm}/vicon", rr.SeriesLines(colors=[0, 200, 255], names="Vicon balls", widths=2.5), static=True); rr.log(f"plots/{nm}/skin", rr.SeriesLines(colors=[255, 90, 90], names="skin tracking around the same positions", widths=2), static=True)
rr.log("world/mesh_context", rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=[200, 200, 200, 70]), static=True)
n_logged = 0
for n, f in enumerate(frames):
    f = int(f); rr.set_time("frame", sequence=f); rr.set_time("video_time", duration=f / 29.97)
    for nm in ("thoracic", "abdominal"): rr.log(f"plots/{nm}/vicon", rr.Scalars(float(reg[nm][0][n]))); rr.log(f"plots/{nm}/skin", rr.Scalars(float(reg[nm][1][n])))
    if n % 2: continue
    dz = D[n, :, 2]; P3 = (Xref + D[n]) / 1000.0; good = ~np.isnan(P3).any(1); rr.log("world/skin_points", rr.Points3D(P3[good], colors=col(dz[good]), radii=0.0035))
    mk = VIC[f + 1][chest]; okm = ~np.isnan(mk).any(1); rr.log("world/vicon_chest", rr.Points3D(mk[okm], colors=[0, 220, 255], radii=0.009))
    if f in mvf: v = MV["X"][mvf.index(f)].astype(np.float64); rr.log("world/mesh_context", rr.Mesh3D.from_fields(vertex_positions=v, vertex_normals=normals(v)))
    for i in range(1, 6):
        p = f"{SF}/T{i}/{f:04d}.jpg"
        if not os.path.exists(p): continue
        rr.log(f"world/cameras/T{i}/image", rr.EncodedImage(contents=open(p, "rb").read(), media_type="image/jpeg")); okp = L[i]["ok"][n][keep] & good; uv = L[i]["pts"][n][keep][okp] / 3.0
        rr.log(f"world/cameras/T{i}/image/tracked", rr.Points2D(uv, colors=col(dz[okp]), radii=2.2))
    n_logged += 1
print("wrote", OUT, ":", n_logged, "frames,", len(keep), "tracked points")
