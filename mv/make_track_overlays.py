"""Per-frame 5-camera mosaics of the marker-guided tracking (HTML report 12): green = fitted mesh points, red = SAM 3D Body's own mesh, cyan rings = Vicon markers."""
import json, os, cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; OD = TMP + "/track/overlay"; os.makedirs(OD, exist_ok=True)
T = np.load(TMP + "/track/track.npz", allow_pickle=True); frames = T["frames"]
CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); K2 = K.copy(); K2[:2] *= 1280 / 3840
cams = {}
for i in range(1, 6):
    c = CJ["cams"]["T%d" % i]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; cams[i] = (R, np.array(c["tvec"]))
def proj(i, X):
    R, t = cams[i]; Xc = X @ R.T + t; uv = Xc @ K2.T; return uv[:, :2] / uv[:, 2:3], Xc[:, 2]
meta = []; cm = T["chest_mk"] * 1000; cf = T["chest_fit"] * 1000; cb = T["chest_base"] * 1000
for n, f in enumerate(frames):
    if n % 2: continue
    vf, vb, mk = T["vf"][n], T["vb"][n], T["mk"][n]; ok = ~np.isnan(mk).any(1); tiles = []
    for i in range(1, 6):
        im = cv2.imread(f"{TMP}/mmc17_frames/T{i}/{int(f):04d}.jpg")
        if im is None: im = np.zeros((720, 1280, 3), np.uint8)
        if not np.isnan(vb).any():
            uv, z = proj(i, vb[::9])
            for (x, y), zz in zip(uv, z):
                if zz > 0 and 0 <= x < 1280 and 0 <= y < 720: cv2.circle(im, (int(x), int(y)), 2, (80, 80, 255), -1, cv2.LINE_AA)
        uv, z = proj(i, vf[::9])
        for (x, y), zz in zip(uv, z):
            if zz > 0 and 0 <= x < 1280 and 0 <= y < 720: cv2.circle(im, (int(x), int(y)), 2, (90, 255, 110), -1, cv2.LINE_AA)
        uv, z = proj(i, mk[ok])
        for (x, y), zz in zip(uv, z):
            if zz > 0: cv2.circle(im, (int(x), int(y)), 9, (255, 220, 0), 2, cv2.LINE_AA)
        cv2.putText(im, "T%d" % i, (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3); tiles.append(cv2.resize(im, (640, 360)))
    panel = np.zeros((360, 640, 3), np.uint8)
    for k, (txt, col) in enumerate(((f"frame {int(f)}   t = {f / 29.97:.1f} s", (255, 255, 255)), (f"marker -> fitted mesh   {T['d_fit'][n]:.0f} mm", (90, 255, 110)), (f"marker -> SAM 3D Body    {T['d_base'][n]:.0f} mm", (80, 80, 255)),
                                    ("green: fitted mesh   red: SAM 3D Body", (200, 200, 200)), ("cyan rings: Vicon markers", (255, 220, 0)))): cv2.putText(panel, txt, (20, 60 + 55 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    tiles.append(panel); mos = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]); cv2.imwrite(f"{OD}/{int(f):04d}.jpg", mos, [cv2.IMWRITE_JPEG_QUALITY, 70])
    meta.append(dict(frame=int(f), d_fit=float(T["d_fit"][n]), d_base=float(T["d_base"][n]), chest_mk=float(cm[n]), chest_fit=float(cf[n]), chest_base=None if np.isnan(cb[n]) else float(cb[n])))
json.dump(meta, open(OD + "/meta.json", "w")); print("mosaics:", len(meta))
