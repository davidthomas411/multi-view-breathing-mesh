"""Decode each 5.3K video once and cache 1920x1080 JPEGs: tmp/frames/<cam>/<idx>.jpg (6 workers, one per camera)."""
import os, sys, cv2
from multiprocessing import Pool
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID = f"{ROOT}/cuvault2/vault2/videos"
OUT = f"{ROOT}/tmp/frames"
W, H = 1920, 1080

def work(cam):
    os.makedirs(f"{OUT}/{cam}", exist_ok=True)
    cap = cv2.VideoCapture(f"{VID}/{cam}.mp4")
    i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        cv2.imwrite(f"{OUT}/{cam}/{i:04d}.jpg", cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 93])
        i += 1
    return cam, i

if __name__ == "__main__":
    with Pool(6) as p:
        for cam, n in p.imap_unordered(work, [f"{i:02d}" for i in range(1, 7)]):
            print(cam, n, "frames", flush=True)
