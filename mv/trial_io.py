"""Which MMC17 trial a script works on (env TRIAL = TDB (default) or ADB) and where its files live.
TDB: Gyroflow-stabilised videos, extrinsics tmp/mmc17_vicon_calib.json, outputs in tmp/<name>/ (as before).
ADB: raw GoPro .mov, warped on the fly into the stabilised-video geometry (adb_geometry.py), extrinsics tmp/mmc17_vicon_calib_ADB.json (adb_extrinsics.py), outputs in tmp/<name>_ADB/."""
import os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; MM = ROOT + "/CUTrial/MM17"
TRIAL = os.environ.get("TRIAL", "TDB")


def suffix(trial=None): return "" if (trial or TRIAL) == "TDB" else "_" + (trial or TRIAL)
def tdir(name, trial=None, make=True):
    d = f"{TMP}/{name}{suffix(trial)}"
    if make: os.makedirs(d, exist_ok=True)
    return d
def calib_path(trial=None): return TMP + ("/mmc17_vicon_calib.json" if (trial or TRIAL) == "TDB" else f"/mmc17_vicon_calib_{trial or TRIAL}.json")
def vicon_csv(trial=None): return f"{ROOT}/CUTrial/Trials_marker_positions_by_image_frame_all_cases/MMC17_{trial or TRIAL}_marker_positions_by_image_frame_wide.csv"
def video_path(cam, trial=None):
    t = trial or TRIAL
    return f"{MM}/TDB/Gyroflow/MMC17_TDB_T{cam}_stabilized.mp4" if t == "TDB" else f"{MM}/{t}/MMC17_{t}_T{cam}.mov"
def frame_range(trial=None): return (250, 1120) if (trial or TRIAL) == "TDB" else (250, 1030)               # first / last video frame analysed


_WARP = {}


class Capture:
    """cv2.VideoCapture-like reader that returns BGR frames in the geometry of the stabilised videos (warped on the fly for the raw ADB videos)."""
    def __init__(self, cam, trial=None):
        self.trial = trial or TRIAL; self.cap = cv2.VideoCapture(video_path(cam, self.trial)); self.warp = None
        if self.trial != "TDB":
            import adb_geometry as G
            if cam not in _WARP: _WARP[cam] = G.Warper(cam)
            self.warp = _WARP[cam]
    def read(self):
        ok, im = self.cap.read()
        return (ok, self.warp(im) if (ok and self.warp is not None) else im)
    def set(self, prop, val): return self.cap.set(prop, val)
    def get(self, prop): return self.cap.get(prop)
    def release(self): self.cap.release()
    def isOpened(self): return self.cap.isOpened()
