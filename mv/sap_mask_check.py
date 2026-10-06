"""How good is the Sapiens2 person mask, including the dark lower-body clothing?  Compare it with the SILHOUETTES of two meshes that are close to the truth: the marker-guided mesh (16 mm to the
Vicon markers) and, for reference, the markerless multi-view mesh.  Per camera: IoU of the mask with the mesh silhouette, and the share of each body part (Sapiens2 labels: lower clothing,
upper clothing, torso, arms, legs/shoes ...) lying inside the marker-guided silhouette.   usage: python sap_mask_check.py [frame]"""
import json, os, sys
import cv2, numpy as np
import trial_io as TI
frame = int(sys.argv[1]) if len(sys.argv) > 1 else 250; SIZE = "0.4b"; TMP = TI.TMP
T = np.load(TMP + "/track/track.npz", allow_pickle=True); faces = T["faces"].astype(np.int32); tf = {int(f): i for i, f in enumerate(T["frames"])}; MV = np.load(TMP + "/mv_fit/mv_fit.npz"); mi = list(MV["frames"]).index(frame)
CJ = json.load(open(TI.calib_path())); K = np.array(CJ["K"]); D = np.array(CJ["dist"]); CLASSES = ["Background", "Apparel", "Eyeglass", "Face_Neck", "Hair", "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg", "Left_Shoe", "Left_Sock", "Left_Upper_Arm", "Left_Upper_Leg",
                                                                  "Lower_Clothing", "Right_Foot", "Right_Hand", "Right_Lower_Arm", "Right_Lower_Leg", "Right_Shoe", "Right_Sock", "Right_Upper_Arm", "Right_Upper_Leg", "Torso", "Upper_Clothing", "Lower_Lip", "Upper_Lip", "Lower_Teeth", "Upper_Teeth", "Tongue"]
def silhouette(V, cam, shape=(1080, 1920)):
    c = CJ["cams"][f"T{cam}"]; uv = cv2.projectPoints(V.reshape(-1, 1, 3).astype(np.float64), np.array(c["rvec"]), np.array(c["tvec"]), K, D)[0].reshape(-1, 2) * 0.5; m = np.zeros(shape, np.uint8)
    pts = np.round(uv[faces]).astype(np.int32)
    for tri in pts: cv2.fillConvexPoly(m, tri, 1)
    return m
rows = []
for cam in range(1, 6):
    p = f"{TI.tdir('sapiens2')}/{frame:04d}_T{cam}_{SIZE}_rot.npz"
    if not os.path.exists(p): continue
    z = np.load(p); seg = z["seg"]; pm = (seg > 0).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(pm); pm = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) if n > 1 else pm
    sg = silhouette(T["vf"][tf[frame]].astype(np.float64), cam); sm = silhouette(MV["X"][mi].astype(np.float64), cam)
    iou = lambda a, b: float((a & b).sum() / max((a | b).sum(), 1)); lower = np.isin(seg, [CLASSES.index(n) for n in CLASSES if "Lower_Clothing" in n or "Leg" in n or "Shoe" in n or "Sock" in n or "Foot" in n]); upper = np.isin(seg, [CLASSES.index(n) for n in ("Torso", "Upper_Clothing", "Face_Neck", "Hair") + tuple(c for c in CLASSES if "Arm" in c or "Hand" in c)])
    inside = lambda part: float((sg.astype(bool) & part).sum() / max(part.sum(), 1))
    rows.append((cam, iou(pm.astype(bool), sg.astype(bool)), iou(pm.astype(bool), sm.astype(bool)), inside(lower), inside(upper), int(pm.sum()), int(sg.sum())))
    vis = np.zeros(sg.shape + (3,), np.uint8); vis[..., 0] = sg * 255; vis[..., 1] = pm * 255; cv2.imwrite(f"{TI.tdir('sapiens2')}/vis/maskcheck_{frame:04d}_T{cam}.jpg", cv2.resize(vis, (960, 540)))
print("frame", frame, "  camera | IoU(Sapiens2 mask, marker-guided mesh) | IoU(mask, markerless multi-view mesh) | lower-body-class pixels inside the marker-guided silhouette | upper-body-class pixels inside")
for r in rows: print("  T%d | %.3f | %.3f | %.3f | %.3f   (mask %d px, silhouette %d px at half resolution)" % r)
print("median IoU: mask vs marker-guided %.3f, vs multi-view mesh %.3f" % (np.median([r[1] for r in rows]), np.median([r[2] for r in rows])))

json.dump(dict(frame=frame, rows=[dict(cam=r[0], iou_markerguided=r[1], iou_multiview=r[2], lower_inside=r[3], upper_inside=r[4], mask_px=r[5], sil_px=r[6]) for r in rows], median_iou_markerguided=float(np.median([r[1] for r in rows])), median_iou_multiview=float(np.median([r[2] for r in rows]))), open(f"{TI.tdir('sapiens2')}/maskcheck_{frame:04d}.json", "w"), indent=1)
