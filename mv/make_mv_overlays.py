"""Mosaics for report 16: five cameras, three meshes (red = SAM 3D Body one camera, blue = markerless multi-view mesh, green = marker-guided), cyan rings = Vicon markers."""
import json, os, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; OD = TMP + "/mv_fit/overlay"; os.makedirs(OD, exist_ok=True)
T = np.load(TMP + "/track/track.npz", allow_pickle=True); MV = np.load(TMP + "/mv_fit/mv_fit.npz"); E2 = np.load(TMP + "/mv_fit/eval2.npz"); tf = {int(f): i for i, f in enumerate(T["frames"])}
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); K2 = K.copy(); K2[:2] *= 1280 / 3840; cams = {}
for i in range(1, 6):
    c = CJ["cams"]["T%d" % i]; cams[i] = (cv2.Rodrigues(np.array(c["rvec"]))[0], np.array(c["tvec"]))
def proj(i, X): R, t = cams[i]; Xc = X @ R.T + t; uv = Xc @ K2.T; return uv[:, :2] / uv[:, 2:3], Xc[:, 2]
meta = []; sv_typ = np.nanmean([np.nanmedian(E2["sv%d" % v], axis=1) for v in range(1, 6)], axis=0)
for n, f in enumerate(MV["frames"]):
    f = int(f); i = tf[f]; vf, vb, vm, mk = T["vf"][i], T["vb"][i], MV["X"][n], T["mk"][i]; ok = ~np.isnan(mk).any(1); tiles = []
    for c in range(1, 6):
        im = cv2.imread(f"{TMP}/mmc17_frames/T{c}/{f:04d}.jpg")
        if im is None: im = np.zeros((720, 1280, 3), np.uint8)
        for v, col in ((vb, (80, 80, 255)), (vm, (255, 150, 90)), (vf, (90, 255, 110))):
            if np.isnan(v).any(): continue
            uv, z = proj(c, v[::9])
            for (x, y), zz in zip(uv, z):
                if zz > 0 and 0 <= x < 1280 and 0 <= y < 720: cv2.circle(im, (int(x), int(y)), 2, col, -1, cv2.LINE_AA)
        uv, z = proj(c, mk[ok].astype(np.float64))
        for (x, y), zz in zip(uv, z):
            if zz > 0: cv2.circle(im, (int(x), int(y)), 9, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(im, "T%d%s" % (c, "  (gated out of the fit)" if c == 3 else ""), (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3); tiles.append(cv2.resize(im, (640, 360)))
    e = int(np.where(E2["frames"] == f)[0][0]); dsv, dmv, dft = np.nanmedian(E2["sv4"][e]), np.nanmedian(E2["mv"][e]), np.nanmedian(E2["fit"][e])
    panel = np.zeros((360, 640, 3), np.uint8)
    for k, (txt, col) in enumerate(((f"frame {f}   t = {f / 29.97:.1f} s", (255, 255, 255)), (f"single camera (T4)      {dsv:.0f} mm", (80, 80, 255)), (f"multi-view, markerless  {dmv:.0f} mm", (255, 150, 90)), (f"marker-guided (reference) {dft:.0f} mm", (90, 255, 110)), ("cyan rings: Vicon markers", (255, 220, 0)), ("median marker-to-mesh distance", (200, 200, 200)))):
        cv2.putText(panel, txt, (20, 50 + 55 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2, cv2.LINE_AA)
    tiles.append(panel); cv2.imwrite(f"{OD}/{f:04d}.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]), [cv2.IMWRITE_JPEG_QUALITY, 68])
    meta.append(dict(frame=f, d_sv=float(dsv), d_mv=float(dmv), d_fit=float(dft), d_typ=float(sv_typ[e])))
json.dump(meta, open(OD + "/meta.json", "w")); print("mosaics:", len(meta))
