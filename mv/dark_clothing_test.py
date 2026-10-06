"""Is SAM 3D Body hurt by the dark clothing on the lower limbs?  Direct test: run it on the original frame and on a brightened / contrast-enhanced copy (gamma 0.45 + CLAHE on the lightness channel,
which makes the black leggings readable), score each mesh against the Vicon markers and compare per body region.   If the lower-body error is about the same, the clothing is not the cause.
Also reports the share of each camera's SAM 3D Body pixels that are 'dark' (mean brightness of the lower-body marker neighbourhoods).
usage (SAM3D env):  python dark_clothing_test.py [frames...]      -> tmp/holdout/dark_clothing.json"""
import csv, json, os, sys, time
import cv2, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M, sam3d_mv as S, trial_io as TI
OUT = M.OUT + "/holdout"; os.makedirs(OUT, exist_ok=True); K = M.K
CJ = json.load(open(TI.calib_path()))["cams"]; RT = [(cv2.Rodrigues(np.array(CJ["T%d" % i]["rvec"]))[0], np.array(CJ["T%d" % i]["tvec"])) for i in range(1, 6)]
rows = list(csv.reader(open(TI.vicon_csv()))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]; VIC = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) * 1e-3 for r in rows[1:]}
chest = [j for j, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]; star = [j for j, n in enumerate(names) if n.startswith("*")]; low = [j for j in range(len(names)) if j not in chest and j not in star]
vidx = np.arange(0, 18439, 3)


def enhance(rgb):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB); l = lab[..., 0].astype(np.float32) / 255.0; l = np.power(l, 0.45); l = (l * 255).astype(np.uint8)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l); return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def score(d, v, f):
    R, t = RT[v]; mk = VIC[f + 1]; ok = ~np.isnan(mk).any(1)
    if d is None: return None
    Xv = ((d["verts"][vidx].astype(np.float64) + d["cam_t"]) - t) @ R; dist = np.full(len(mk), np.nan); dist[ok] = cKDTree(Xv).query(mk[ok])[0] * 1000; return dist


if __name__ == "__main__":
    frames = [int(a) for a in sys.argv[1:]] or [250, 400, 550, 700, 850, 1000]; res = dict(orig=[], enh=[]); t0 = time.time()
    for f in frames:
        for v in range(5):
            img = M.read(TI.TRIAL, v + 1, f)
            if img is None: continue
            mk = VIC[f + 1]; R, t = RT[v]; ok = ~np.isnan(mk[low]).any(1); P = cv2.projectPoints((mk[low][ok]).reshape(-1, 1, 3), cv2.Rodrigues(R)[0], t, K, M.DIST)[0].reshape(-1, 2)
            br = [float(img[int(max(y - 25, 0)):int(y + 25), int(max(x - 25, 0)):int(x + 25)].mean()) for x, y in P if 25 < x < 3815 and 25 < y < 2135]
            for cond, im in (("orig", img), ("enh", enhance(img))):
                d = S.run_view(im, K, keep_mesh=True); dist = score(d, v, f)
                res[cond].append(dict(frame=f, cam=v + 1, ok=dist is not None, chest=float(np.nanmedian(dist[chest])) if dist is not None else None, low=float(np.nanmedian(dist[low])) if dist is not None else None, star=float(np.nanmedian(dist[star])) if dist is not None else None,
                                      all=float(np.nanmedian(dist)) if dist is not None else None, brightness_low=float(np.median(br)) if br else None))
        print(f"frame {f} done ({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(OUT + "/dark_clothing.json", "w"))
    for cam in range(1, 6):
        for reg in ("low", "chest", "star", "all"):
            a = [r[reg] for r in res["orig"] if r["cam"] == cam and r[reg] is not None]; b = [r[reg] for r in res["enh"] if r["cam"] == cam and r[reg] is not None]
            if a and b: print("T%d %-5s : original %.1f mm -> enhanced %.1f mm (median over %d frames)" % (cam, reg, np.median(a), np.median(b), len(a)))
    print("lower-body marker neighbourhood brightness (0-255): ", {c: round(float(np.median([r['brightness_low'] for r in res['orig'] if r['cam'] == c and r['brightness_low'] is not None])), 0) for c in range(1, 6)})
