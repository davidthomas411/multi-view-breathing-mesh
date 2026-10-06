"""Rerun recording of the Sapiens2 cold start: for each test frame, the five camera views with Sapiens2's 308 keypoints and body-part segmentation, the triangulated 3D keypoints, and four meshes in the
Vicon frame: Sapiens2 cold-start fit (yellow), markerless SAM 3D Body multi-view mesh (blue), marker-guided mesh (green, the reference) and the Vicon markers (cyan).
  C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe make_sapiens_rrd.py [suffix: '' or _sil]   ->  recordings/sapiens2_cold_start.rrd"""
import json, os, sys
import cv2, numpy as np
import rerun as rr, rerun.blueprint as rrb
import trial_io as TI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT, TMP = TI.ROOT, TI.TMP; suffix = sys.argv[1] if len(sys.argv) > 1 else ""; SIZE = "0.4b"
SF = np.load(f"{TMP}/sap_fit/sapfit_{SIZE}_rot{suffix}.npz"); T = np.load(TMP + "/track/track.npz", allow_pickle=True); faces = T["faces"].astype(np.uint32); tf = {int(f): i for i, f in enumerate(T["frames"])}; MV = np.load(TMP + "/mv_fit/mv_fit.npz"); mvf = list(MV["frames"])
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); K2 = K.copy(); K2[:2] *= 0.5; cams = {}
for i in range(1, 6): c = CJ["cams"]["T%d" % i]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; t = np.array(c["tvec"]); cams[i] = (R, t, -R.T @ t)
CLASSES = ["Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg", "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
           "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue"]
PAL = np.random.default_rng(3).integers(40, 255, size=(29, 3)).astype(np.uint8); PAL[0] = 0
def normals(v):
    fn = np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]); vn = np.zeros_like(v)
    for k in range(3): np.add.at(vn, faces[:, k], fn)
    return (vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
bp = rrb.Blueprint(rrb.Horizontal(
    rrb.Spatial3DView(name="Vicon frame: yellow = Sapiens2 cold start, blue = SAM 3D Body multi-view (markerless), green = marker-guided (reference), cyan = Vicon markers, white = triangulated Sapiens2 keypoints", origin="world",
                      contents=["world/mesh_sapiens", "world/mesh_multiview", "world/mesh_markerguided", "world/markers", "world/kpts3d", "world/cameras/*"], eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)),
    rrb.Grid(*[rrb.Spatial2DView(name="T%d: Sapiens2 keypoints + body parts" % i, origin=f"world/cameras/T{i}/image") for i in range(1, 6)], grid_columns=2), column_shares=[5, 6]), rrb.TimePanel(state="expanded"))
os.makedirs(ROOT + "/recordings", exist_ok=True); OUT = ROOT + f"/recordings/sapiens2_cold_start{suffix}.rrd"; rr.init("sapiens2_cold_start", spawn=False); rr.save(OUT, default_blueprint=bp); rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
rr.log("annotation", rr.AnnotationContext([rr.AnnotationInfo(id=i, label=n, color=tuple(int(x) for x in PAL[i])) for i, n in enumerate(CLASSES)]), static=True)
for i, (R, t, c) in cams.items():
    rr.log(f"world/cameras/T{i}", rr.Transform3D(translation=c, mat3x3=R.T), static=True); rr.log(f"world/cameras/T{i}", rr.Pinhole(image_from_camera=K2, resolution=[1920, 1080], image_plane_distance=0.3), static=True)
for nm, col in (("mesh_sapiens", [255, 200, 40, 200]), ("mesh_multiview", [90, 150, 255, 110]), ("mesh_markerguided", [70, 200, 100, 150])): rr.log(f"world/{nm}", rr.Mesh3D.from_fields(triangle_indices=faces, albedo_factor=col), static=True)
for n, f in enumerate(SF["frames"]):
    f = int(f); rr.set_time("frame", sequence=f); rr.set_time("video_time", duration=f / 29.97); i = tf[f]
    for nm, v in (("mesh_sapiens", SF["X"][n].astype(np.float64)), ("mesh_markerguided", T["vf"][i].astype(np.float64)), ("mesh_multiview", MV["X"][mvf.index(f)].astype(np.float64) if f in mvf else None)):
        if v is not None: rr.log(f"world/{nm}", rr.Mesh3D.from_fields(vertex_positions=v, vertex_normals=normals(v)))
    mk = T["mk"][i].astype(np.float64); ok = ~np.isnan(mk).any(1); rr.log("world/markers", rr.Points3D(mk[ok], colors=[0, 200, 255], radii=0.009))
    tri = f"{TMP}/sapiens2/tri_{f:04d}_{SIZE}_rot.npz"
    if os.path.exists(tri): X = np.load(tri)["X"]; g = ~np.isnan(X).any(1); rr.log("world/kpts3d", rr.Points3D(X[g], colors=[255, 255, 255], radii=0.012))
    for c in range(1, 6):
        p = f"{TMP}/sapiens2/{f:04d}_T{c}_{SIZE}_rot.npz"
        if not os.path.exists(p): continue
        z = np.load(p); cap = TI.Capture(c, "TDB"); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok_, im = cap.read(); cap.release()
        if not ok_: continue
        rr.log(f"world/cameras/T{c}/image", rr.EncodedImage(contents=cv2.imencode(".jpg", cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), media_type="image/jpeg"))
        rr.log(f"world/cameras/T{c}/image/parts", rr.SegmentationImage(z["seg"], opacity=0.45)); sc = z["scores"]; kp = z["kpts"] * 0.5
        rr.log(f"world/cameras/T{c}/image/kpts", rr.Points2D(kp[sc > 0.3][:70], colors=[255, 255, 0], radii=5)); rr.log(f"world/cameras/T{c}/image/face_kpts", rr.Points2D(kp[sc > 0.3][70:], colors=[255, 140, 0], radii=2))
print("wrote", OUT)
