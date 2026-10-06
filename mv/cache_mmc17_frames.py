"""Sequentially decode MMC17 TDB videos once and cache every 3rd frame (250..1120) at 1280x720: tmp/mmc17_frames/T<i>/<frame>.jpg (one worker per camera)."""
import os, sys, cv2
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = ROOT + "/CUTrial/MM17/TDB/Gyroflow/MMC17_TDB_T{i}_stabilized.mp4"; OUT = ROOT + "/tmp/mmc17_frames"
def work(i):
    os.makedirs(f"{OUT}/T{i}", exist_ok=True); cap = cv2.VideoCapture(VID.format(i=i)); f = 0
    while True:
        ok, fr = cap.read()
        if not ok or f > 1120: break
        if f >= 250 and (f - 250) % 3 == 0: cv2.imwrite(f"{OUT}/T{i}/{f:04d}.jpg", cv2.resize(fr, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
        f += 1
    return i
if __name__ == "__main__":
    with Pool(5) as p:
        for i in p.imap_unordered(work, range(1, 6)): print("T%d done" % i, flush=True)
