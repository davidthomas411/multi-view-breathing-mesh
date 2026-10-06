"""COLD-START fit of the MHR body to Sapiens2 2D keypoints in all calibrated cameras (no Vicon markers, no SAM 3D Body mesh in the objective).

Sapiens2 (pose, 308 keypoints) gives 2D keypoints per camera; MHR itself outputs the same 308 keypoints (mhr_forward(return_keypoints=True)), so one body (pose + rigid correction) is optimised so that
its keypoints project onto the Sapiens2 detections in all cameras at once (pseudo-Huber, confidence-weighted; body keypoints weigh most, face keypoints a little, hand keypoints very little).
The optimisation is started from each camera's SAM 3D Body pose in turn (multi-start, as in mv_fit.py), keeping the lowest loss.
Scoring against Vicon (never used in the fit): median marker-to-mesh distance, same metric as eval2 / report 16.
run with the SAM3D env:  python sap_fit.py [frames ...]  (needs tmp/sapiens2/<frame>_T<cam>_0.4b_rot.npz from sapiens2_run.py and tmp/mv_sam3d/<frame>.npz)      env: TRIAL, SAPSIZE (0.4b), SAPROT (rot)
"""
import os, sys, json, time
import cv2, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import torch.nn.functional as Fn
import mmc17_sam3d as M, marker_fit as MF, mv_fit as W, trial_io as TI

DEV = "cuda"; K = M.K; DIST = M.DIST; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
SIZE = os.environ.get("SAPSIZE", "0.4b"); ROT = os.environ.get("SAPROT", "rot"); SAPD = TI.tdir("sapiens2"); MVS = TI.tdir("mv_sam3d"); OUTD = TI.tdir("sap_fit")
# per-keypoint weights: body+feet (0-20) and the extra body points (63-69) = 1, hands (21-62) = 0.1, face and the rest (70-307) = 0.03
W_SIL = float(os.environ.get("W_SIL", 0.0)); W_MV = float(os.environ.get("W_MV", 0.0)); K1, K2 = float(DIST[0]), float(DIST[1]); VID = torch.arange(0, 18439, 3, device=DEV)         # silhouette term (weight W_SIL; 0 = keypoints only), radial distortion of the stabilised frames
CW = np.full(308, 0.03, np.float32); CW[:21] = 1.0; CW[63:70] = 1.0; CW[21:63] = 0.1; CWt = torch.tensor(CW, device=DEV)


def load_sapiens(f):
    """-> per camera (None or (und (308,2) ideal-pinhole px, score (308,)))."""
    out = []
    for c in range(1, 6):
        p = f"{SAPD}/{f:04d}_T{c}_{SIZE}_{ROT}.npz"
        if not os.path.exists(p): out.append(None); continue
        z = np.load(p); und = cv2.undistortPoints(z["kpts"].reshape(-1, 1, 2).astype(np.float64), K, DIST, P=K).reshape(-1, 2); out.append((torch.tensor(und, dtype=torch.float32, device=DEV), torch.tensor(z["scores"], dtype=torch.float32, device=DEV)))
    return out


def load_masks(f):
    """Sapiens2 person masks (half resolution): per camera (distance-to-mask map, sampled foreground pixels) as GPU tensors, or None."""
    out = []
    for c in range(1, 6):
        p = f"{SAPD}/{f:04d}_T{c}_{SIZE}_{ROT}.npz"
        if not os.path.exists(p): out.append(None); continue
        m = (np.load(p)["seg"] > 0).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)); n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 1: m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)                          # keep the largest blob (the person)
        dt = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3).astype(np.float32); ys, xs = np.nonzero(m); sel = np.random.default_rng(0).choice(len(xs), min(1500, len(xs)), replace=False)
        out.append((torch.tensor(dt, device=DEV), torch.tensor(np.stack([xs[sel], ys[sel]], 1), dtype=torch.float32, device=DEV)))
    return out


def project_dist(Xw, i):
    """Vicon-frame points -> pixel coordinates of the (slightly distorted) stabilised frame, half resolution."""
    Xc = Xw @ W.Rg[i].T + W.tg[i]; x = Xc[:, 0] / Xc[:, 2]; y = Xc[:, 1] / Xc[:, 2]; r2 = x * x + y * y; d = 1 + K1 * r2 + K2 * r2 * r2
    return torch.stack([(fx * x * d + cx) * 0.5, (fy * y * d + cy) * 0.5], 1)


def sil_loss(Xv, masks, sg=12.0):
    """vertices outside the person mask are pulled in; mask pixels far from every vertex pull the body out (half-resolution pixels, pseudo-Huber)."""
    tot = 0.0; n = 0
    for i in range(5):
        if masks[i] is None: continue
        dt, pix = masks[i]; u = project_dist(Xv, i); H_, W_ = dt.shape
        g = torch.stack([(2 * u[:, 0] + 1) / W_ - 1, (2 * u[:, 1] + 1) / H_ - 1], 1)[None, :, None, :]; dout = Fn.grid_sample(dt[None, None], g, mode="bilinear", padding_mode="border", align_corners=False)[0, 0, :, 0]
        dcov = torch.cdist(pix, u).min(1)[0]; rho = lambda r: (sg ** 2) * (torch.sqrt(1 + (r / sg) ** 2) - 1); tot = tot + rho(dout).mean() + rho(dcov).mean(); n += 1
    return tot / max(n, 1)


def kp_forward(Fh, p, body):
    out = Fh.head.mhr_forward(global_trans=torch.zeros(1, 3, device=DEV), global_rot=p["global_rot"], body_pose_params=body, hand_pose_params=p["hand"], scale_params=p["scale"], shape_params=p["shape"], expr_params=p["expr"], return_keypoints=True)
    return out[0][0], out[1][0]                                                   # vertices (18439,3), keypoints (308,3) in the model frame


def kp_loss(Xk, obs, thr=0.3, sg=25.0):
    tot = 0.0; wsum = 0.0; per = []
    for i in range(5):
        if obs[i] is None: continue
        u2, sc = obs[i]; Xc = Xk @ W.Rg[i].T + W.tg[i]; u = fx * Xc[:, 0] / Xc[:, 2] + cx; v = fy * Xc[:, 1] / Xc[:, 2] + cy
        r = torch.sqrt((u - u2[:, 0]) ** 2 + (v - u2[:, 1]) ** 2 + 1e-6); rho = (sg ** 2) * (torch.sqrt(1 + (r / sg) ** 2) - 1); w = CWt * sc * (sc > thr).float()
        tot = tot + (w * rho).sum(); wsum = wsum + w.sum()
        with torch.no_grad(): per.append(float((r[:21][sc[:21] > thr]).median()) if (sc[:21] > thr).any() else float("nan"))
    return tot / torch.clamp(wsum, min=1.0), per


def optimise(Fh, p, body, rc, tc, to_world, obs, body_ref, iters, lr_body, lr_rigid, lam, masks=None, mv=None):
    body = body.clone().requires_grad_(True); rc = rc.clone().requires_grad_(True); tc = tc.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [rc, tc], "lr": lr_rigid}] + ([{"params": [body], "lr": lr_body}] if lr_body > 0 else []))
    for it in range(iters):
        opt.zero_grad(); vl, kl = kp_forward(Fh, p, body); l, _ = kp_loss(to_world(kl, rc, tc), obs)
        if masks is not None and W_SIL > 0: l = l + W_SIL * sil_loss(to_world(vl[VID], rc, tc), masks)
        if mv is not None and W_MV > 0: l = l + W_MV * W.mv_loss(to_world(vl, rc, tc), mv["obs"], mv["okv"], mv["vidx"], mv["wts"])[0]
        (l + lam * ((body - body_ref) ** 2).sum()).backward(); opt.step()
    with torch.no_grad(): vl, kl = kp_forward(Fh, p, body); l, per = kp_loss(to_world(kl, rc, tc), obs); X = to_world(vl, rc, tc)
    return body.detach(), rc.detach(), tc.detach(), float(l), per, X.detach()


def vicon_markers(f):
    """Vicon markers (m) at video frame f for the current trial (CSV row f+1)."""
    import csv
    rows = list(csv.reader(open(TI.vicon_csv()))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]; V = {int(r[2]): np.array([float(v) if v else np.nan for v in r[10:]]).reshape(-1, 3) for r in rows[1:]}
    return names, V[f + 1] * 1e-3


if __name__ == "__main__":
    frames = [int(a) for a in sys.argv[1:]] or [250]; Fh = MF.Fitter(); res = dict(frames=[], X=[], loss=[], per=[], start=[]); names_all, _ = vicon_markers(250); t00 = time.time()
    for f in frames:
        obs = load_sapiens(f); masks = load_masks(f) if W_SIL > 0 else None; mvo = W.load_obs(f) if W_MV > 0 else None; vidx_t = torch.tensor(np.load(f"{MVS}/{f:04d}.npz")["vidx"], device=DEV); nv = sum(o is not None for o in obs); print(f"=== frame {f}: Sapiens2 keypoints in {nv} cameras", flush=True); best = None; t0 = time.time()
        for v in range(5):
            if obs[v] is None: continue
            o = MF.sam3d_raw(M.read(TI.TRIAL, v + 1, f), v)
            if o is None: continue
            p, b0 = Fh.init(o); to_world = W.make_world(v, np.asarray(o["pred_cam_t"])); rc = torch.zeros(3, device=DEV); tc = torch.zeros(3, device=DEV)
            body, rc, tc, l1, _, X1 = optimise(Fh, p, b0, rc, tc, to_world, obs, b0, 80, 0.0, 6e-3, 0.0, masks)
            mv = None
            if mvo is not None:                                                                                   # gate the SAM 3D Body views that disagree with this start (same rule as mv_fit.py), then add the consensus term
                obs_mv, okv, _ = mvo
                with torch.no_grad(): per0 = W.mv_loss(X1, obs_mv, okv, vidx_t, None)[1]
                vids = [i for i in range(5) if okv[i]]; med = float(np.median(per0)); wts = {i: 1.0 for i in range(5)}
                for i, r in zip(vids, per0):
                    if r > max(2.0 * med, 80.0): wts[i] = 0.0
                mv = dict(obs=obs_mv, okv=okv, vidx=vidx_t, wts=wts)
            body, rc, tc, l2, per, X = optimise(Fh, p, b0, rc, tc, to_world, obs, b0, 300, 1e-2, 3e-3, 0.02, masks, mv)
            print(f"   start from T{v + 1}: loss {l1:.1f} (rigid) -> {l2:.1f}; median keypoint residual per camera {np.round(per, 0)} px", flush=True)
            if best is None or l2 < best[0]: best = (l2, v, X, per)
        l, v, X, per = best; X = X.cpu().numpy().astype(np.float64); names, mk = vicon_markers(f); ok = ~np.isnan(mk).any(1); vidx = np.load(f"{MVS}/{f:04d}.npz")["vidx"]
        d = cKDTree(X[vidx]).query(mk[ok])[0] * 1000; res["frames"].append(f); res["X"].append(X.astype(np.float32)); res["loss"].append(l); res["per"].append(per); res["start"].append(v + 1)
        print(f"   chosen start T{v + 1} (loss {l:.1f}); marker->mesh median {np.median(d):.1f} mm, p90 {np.percentile(d, 90):.1f} mm  ({time.time() - t0:.0f}s)", flush=True)
        np.savez(f"{OUTD}/sapfit_{SIZE}_{ROT}{'_sil' if W_SIL > 0 else ''}{'_mv' if W_MV > 0 else ''}.npz", frames=np.array(res["frames"]), X=np.array(res["X"]), loss=np.array(res["loss"]), start=np.array(res["start"]))
    print("done in %.0fs" % (time.time() - t00))
