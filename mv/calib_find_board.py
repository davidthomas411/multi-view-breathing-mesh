import os
import cv2, numpy as np, itertools, sys
ROOT=(os.environ.get("SGRT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/CUTrial/MM17/DBECal")
def frame(p, idx=15):
    c = cv2.VideoCapture(p); c.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, f = c.read(); return f
res = {}
for src, path in (("gyro", ROOT + "/post_processing/Gyroflow/MMC17_DBECal_T{i}_stabilized.mp4"),):
    for i in range(1, 7):
        g = cv2.cvtColor(frame(path.format(i=i)), cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        found = []
        for cols in range(4, 13):
            for rows in range(4, 13):
                if cols < rows: continue
                ok, c = cv2.findChessboardCornersSB(g, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
                if ok: found.append((cols * rows, cols, rows))
        print(src, "T%d" % i, "patterns found (inner corners):", sorted(found)[-3:])
