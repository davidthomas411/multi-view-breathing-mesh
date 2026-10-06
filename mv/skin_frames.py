"""Ball-free colour frames for the skin-tracking visualisation: every reflective ball painted out (inpainted) exactly as in chest_lk_masked.py, then downscaled to 1280x720.
usage: python skin_frames.py <cam 1-5>     -> tmp/skin_frames/T<cam>/<frame>.jpg   (every 6th frame, 250..1120)"""
import os, sys, json, csv
import cv2, numpy as np
import trial_io as TI
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; cam = int(sys.argv[1]); R_MASK = 30
CJ = json.load(open(TI.calib_path())); K = np.array(CJ["K"]); DIST = np.array(CJ["dist"]); c = CJ["cams"]["T%d" % cam]; rvec = np.array(c["rvec"]); tvec = np.array(c["tvec"])
def proj(Xv): return cv2.projectPoints(Xv.reshape(-1, 1, 3), rvec, tvec, K, DIST)[0].reshape(-1, 2)
frames = list(range(250, TI.frame_range()[1] + 1, 3))
rows = list(csv.reader(open(TI.vicon_csv())))
VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}; MK = np.array([VIC[f + 1] for f in frames]) * 1e-3
for m in range(MK.shape[1]):
    ok = ~np.isnan(MK[:, m, 0])
    if 3 <= ok.sum() < len(ok):
        for a in range(3): MK[:, m, a] = np.interp(np.arange(len(ok)), np.where(ok)[0], MK[ok, m, a])
def inpaint_balls_color(im, centres):
    H, W = im.shape[:2]; h = R_MASK + 18
    for x, y in centres:
        xi, yi = int(round(x)), int(round(y))
        if xi < -h or xi > W + h or yi < -h or yi > H + h: continue
        x0, y0, x1, y1 = max(xi - h, 0), max(yi - h, 0), min(xi + h + 1, W), min(yi + h + 1, H)
        if x1 - x0 < 8 or y1 - y0 < 8: continue
        crop = np.ascontiguousarray(im[y0:y1, x0:x1]); m = np.zeros(crop.shape[:2], np.uint8); cv2.circle(m, (xi - x0, yi - y0), R_MASK, 255, -1)
        im[y0:y1, x0:x1] = cv2.inpaint(crop, m, 5, cv2.INPAINT_TELEA)
SF = TI.tdir("skin_frames"); os.makedirs(f"{SF}/T{cam}", exist_ok=True)
cap = TI.Capture(cam); f = 0
while f <= TI.frame_range()[1]:
    ok, im = cap.read()
    if not ok: break
    if f >= 250 and (f - 250) % 3 == 0 and ((f - 250) // 3) % 2 == 0:                       # every 6th video frame
        k = (f - 250) // 3; inpaint_balls_color(im, proj(MK[k][~np.isnan(MK[k]).any(1)]))
        cv2.imwrite(f"{SF}/T{cam}/{f:04d}.jpg", cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    f += 1
print("T%d done" % cam)
