"""Mosaics for report 20: the five cameras with the Sapiens2 cold-start mesh (yellow), the markerless multi-view mesh (blue), the marker-guided mesh (green) and the Vicon markers (cyan rings).
usage: python make_sapiens_overlays.py [suffix: '' or _sil]   -> tmp/sap_fit/overlay/<frame>.jpg"""
import json, os, sys
import cv2, numpy as np
import trial_io as TI
TMP = TI.TMP; suffix = sys.argv[1] if len(sys.argv) > 1 else ""; OD = TMP + "/sap_fit/overlay" + suffix; os.makedirs(OD, exist_ok=True)
SF = np.load(f"{TMP}/sap_fit/sapfit_0.4b_rot{suffix}.npz"); T = np.load(TMP + "/track/track.npz", allow_pickle=True); tf = {int(f): i for i, f in enumerate(T["frames"])}; MV = np.load(TMP + "/mv_fit/mv_fit.npz"); mvf = [int(f) for f in MV["frames"]]
E2 = np.load(TMP + "/mv_fit/eval2.npz"); fe = [int(f) for f in E2["frames"]]; CJ = json.load(open(TMP + "/mmc17_vicon_calib.json")); K = np.array(CJ["K"]); D = np.array(CJ["dist"]); cams = {i: CJ["cams"]["T%d" % i] for i in range(1, 6)}
sd = json.load(open(f"{TMP}/sap_fit/sap_eval{suffix}.json")) if os.path.exists(f"{TMP}/sap_fit/sap_eval{suffix}.json") else None; meta = []
for n, f in enumerate(SF["frames"]):
    f = int(f); tiles = []
    for c in range(1, 6):
        cap = TI.Capture(c, "TDB"); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); cap.release()
        if not ok: im = np.zeros((2160, 3840, 3), np.uint8)
        for V, colr in ((MV["X"][mvf.index(f)], (255, 150, 90)), (T["vf"][tf[f]], (90, 255, 110)), (SF["X"][n], (40, 220, 255))):
            P = cv2.projectPoints(V[::9].reshape(-1, 1, 3).astype(np.float64), np.array(cams[c]["rvec"]), np.array(cams[c]["tvec"]), K, D)[0].reshape(-1, 2)
            for x, y in P:
                if 0 <= x < 3840 and 0 <= y < 2160: cv2.circle(im, (int(x), int(y)), 3, colr, -1, cv2.LINE_AA)
        mk = T["mk"][tf[f]].astype(np.float64); okm = ~np.isnan(mk).any(1); P = cv2.projectPoints(mk[okm].reshape(-1, 1, 3), np.array(cams[c]["rvec"]), np.array(cams[c]["tvec"]), K, D)[0].reshape(-1, 2)
        for x, y in P: cv2.circle(im, (int(x), int(y)), 16, (255, 220, 0), 4, cv2.LINE_AA)
        cv2.putText(im, "T%d" % c, (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 4.2, (255, 255, 255), 9); tiles.append(cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA))
    panel = np.zeros_like(tiles[0]); i = fe.index(f) if f in fe else None
    lines = [("frame %d  t = %.1f s" % (f, f / 29.97), (255, 255, 255)), ("Sapiens2 cold start (no markers)", (40, 220, 255)), ("multi-view SAM 3D Body (no markers)", (255, 150, 90)), ("marker-guided (uses the markers)", (90, 255, 110)), ("cyan rings: Vicon markers", (255, 220, 0))]
    for k, (t_, col) in enumerate(lines): cv2.putText(panel, t_, (24, 70 + 62 * k), cv2.FONT_HERSHEY_SIMPLEX, 1.15, col, 2, cv2.LINE_AA)
    if sd and f in sd["frames"]:
        j = sd["frames"].index(f); vals = [("Sapiens2 cold start", sd["variants"]["sapiens_cold"]["all"][j]), ("multi-view", sd["variants"]["mv"]["all"][j]), ("marker-guided", sd["variants"]["fit"]["all"][j])]
        cv2.putText(panel, "median marker-to-mesh (mm):", (24, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(panel, "  ".join("%s %.0f" % (a, b) for a, b in vals), (24, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA); meta.append(dict(frame=f, cold=vals[0][1], mv=vals[1][1], fit=vals[2][1]))
    tiles.append(panel); cv2.imwrite(f"{OD}/{f:04d}.jpg", np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]), [cv2.IMWRITE_JPEG_QUALITY, 80])
json.dump(meta, open(OD + "/meta.json", "w")); print("wrote", len(SF["frames"]), "mosaics to", OD)
