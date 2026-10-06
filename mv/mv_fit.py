"""Markerless multi-view SAM 3D Body mesh fit (no Vicon markers in the fit).

Per frame, every camera's SAM 3D Body mesh gives a 2D image location for each of ~6k vertices (accurate in the image plane, wrong in depth, differently in each camera).
ONE MHR body (pose + rigid correction, shape/scale from the best initialisation view) is optimised so that its vertices project onto those 2D locations in ALL cameras
at once (pseudo-Huber).  First frame: multi-start from each view's pose, keep the lowest loss.  Later frames: warm start + temporal prior.
Cameras: the marker-based extrinsics (camera geometry only).  Vicon is NOT used here (it is only used afterwards, in mv_eval.py, to score).

run with the SAM3D env:  python mv_fit.py [first_frame_index_in_list n_frames]        -> tmp/mv_fit/mv_fit.npz
"""
import os, sys, json, time
import cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M, sam3d_mv as S, marker_fit as MF, trial_io as TI

DEV = "cuda"; K = M.K; DIST = M.DIST; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
CJ = json.load(open(TI.calib_path()))["cams"]
RT = [(cv2.Rodrigues(np.array(CJ["T%d" % i]["rvec"]))[0], np.array(CJ["T%d" % i]["tvec"])) for i in range(1, 6)]
Rg = [torch.tensor(r, dtype=torch.float32, device=DEV) for r, _ in RT]; tg = [torch.tensor(t, dtype=torch.float32, device=DEV) for _, t in RT]
FRAMES = list(range(TI.frame_range()[0], TI.frame_range()[1] + 1, 6)); FLIP = torch.tensor([1.0, -1.0, -1.0], device=DEV); SG = 20.0
MVF = TI.tdir("mv_fit"); MVS = TI.tdir("mv_sam3d")

def load_obs(f):
    """2D targets (undistorted pinhole pixels) and validity for each camera at frame f."""
    z = np.load(f"{MVS}/{f:04d}.npz"); obs = []; okv = []
    for i in range(5):
        if not z["ok"][i]: obs.append(None); okv.append(False); continue
        Xc = z["verts"][i].astype(np.float64) + z["cam_t"][i]; uv = (K @ Xc.T).T; uv = uv[:, :2] / uv[:, 2:3]
        und = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, DIST, P=K).reshape(-1, 2)
        obs.append(torch.tensor(und, dtype=torch.float32, device=DEV)); okv.append(True)
    return obs, okv, z

def make_world(init_view, cam_t0):
    R0, t0 = Rg[init_view], tg[init_view]; ct = torch.tensor(cam_t0, dtype=torch.float32, device=DEV)
    def to_world(vl, rc, tc):
        vv = ((vl * FLIP + ct) - t0) @ R0                                  # model frame -> camera -> room (Vicon) frame
        Rcorr = torch.linalg.matrix_exp(torch.stack([torch.zeros_like(rc[0]), -rc[2], rc[1], rc[2], torch.zeros_like(rc[0]), -rc[0], -rc[1], rc[0], torch.zeros_like(rc[0])]).reshape(3, 3))
        return vv @ Rcorr.T + tc
    return to_world

def mv_loss(X, obs, okv, vidx, wts=None):
    tot = 0.0; n = 0; per = []
    for i in range(5):
        if not okv[i]: continue
        wi = 1.0 if wts is None else wts[i]
        Xc = X[vidx] @ Rg[i].T + tg[i]; u = fx * Xc[:, 0] / Xc[:, 2] + cx; w = fy * Xc[:, 1] / Xc[:, 2] + cy
        r = torch.sqrt((u - obs[i][:, 0]) ** 2 + (w - obs[i][:, 1]) ** 2 + 1e-6); rho = (SG ** 2) * (torch.sqrt(1 + (r / SG) ** 2) - 1); tot = tot + wi * rho.mean(); n += wi; per.append(float(r.median()))
    return tot / max(n, 1), per

def optimise(Fh, p, body, rc, tc, to_world, obs, okv, vidx, body_ref, iters, lr_body, lr_rigid, lam, wts=None):
    body = body.clone().requires_grad_(True); rc = rc.clone().requires_grad_(True); tc = tc.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [rc, tc], "lr": lr_rigid}] + ([{"params": [body], "lr": lr_body}] if lr_body > 0 else []))
    for it in range(iters):
        opt.zero_grad(); vl = Fh.forward(p, body)[0]; X = to_world(vl, rc, tc); l, _ = mv_loss(X, obs, okv, vidx, wts); loss = l + lam * ((body - body_ref) ** 2).sum(); loss.backward(); opt.step()
    with torch.no_grad(): vl = Fh.forward(p, body)[0]; X = to_world(vl, rc, tc); l, per = mv_loss(X, obs, okv, vidx, None)
    return body.detach(), rc.detach(), tc.detach(), float(l), per, X.detach()

if __name__ == "__main__":
    a0 = int(sys.argv[1]) if len(sys.argv) > 2 else 0; nf = int(sys.argv[2]) if len(sys.argv) > 2 else len(FRAMES)
    frames = FRAMES[a0:a0 + nf]; Fh = MF.Fitter(); vidx_np = np.load(f"{MVS}/{frames[0]:04d}.npz")["vidx"]; vidx = torch.tensor(vidx_np, device=DEV)
    obs, okv, z = load_obs(frames[0]); t0 = time.time(); best = None
    print("frame %d: multi-start over the initialisation views ..." % frames[0], flush=True)
    for v in range(5):
        if not okv[v]: continue
        o = MF.sam3d_raw(M.read(TI.TRIAL, v + 1, frames[0]), v)
        if o is None: continue
        p, b0 = Fh.init(o); to_world = make_world(v, np.asarray(o["pred_cam_t"]))
        rc = torch.zeros(3, device=DEV); tc = torch.zeros(3, device=DEV)
        body, rc, tc, l1, _, _ = optimise(Fh, p, b0, rc, tc, to_world, obs, okv, vidx, b0, 80, 0.0, 6e-3, 0.0)                       # rigid only
        body, rc, tc, l2, per, X = optimise(Fh, p, b0, rc, tc, to_world, obs, okv, vidx, b0, 250, 1e-2, 3e-3, 0.02)                  # + pose
        print("  init from camera T%d: loss rigid %.2f -> %.2f after pose, per-view median reprojection residual %s px" % (v + 1, l1, l2, np.round(per, 1)), flush=True)
        if best is None or l2 < best[0]: best = (l2, v, p, body, rc, tc, to_world, np.asarray(o["pred_cam_t"]))
    _, v0, p, body, rc, tc, to_world, ct0 = best; print("chosen initialisation: camera T%d (lowest multi-view loss)" % (v0 + 1), flush=True)
    out = dict(frames=[], X=[], loss=[], resid=[], dropped=[])
    for n, f in enumerate(frames):
        obs, okv, z = load_obs(f); prev = body.clone()
        with torch.no_grad(): _, per0 = mv_loss(to_world(Fh.forward(p, body)[0], rc, tc), obs, okv, vidx)       # residual of the warm-start state on the new frame
        vids = [i for i in range(5) if okv[i]]; med = float(np.median(per0)); wts = {i: 1.0 for i in range(5)}
        for i, r in zip(vids, per0):
            if r > max(2.0 * med, 80.0): wts[i] = 0.0                                                         # inconsistent view: gate it out for this frame
        its = 120 if n == 0 else 45
        body, rc, tc, l, per, X = optimise(Fh, p, body, rc, tc, to_world, obs, okv, vidx, prev, its, 4e-3 if n else 6e-3, 2e-3, 0.02, wts)
        out["frames"].append(f); out["X"].append(X.cpu().numpy().astype(np.float32)); out["loss"].append(l); out["resid"].append(float(np.median(per))); out["dropped"].append([i + 1 for i in range(5) if wts[i] == 0.0])
        if n % 10 == 0: print("frame %d (%d/%d): loss %.2f, median reprojection residual %.1f px, views %d, %.1fs/frame" % (f, n + 1, len(frames), l, np.median(per), sum(okv), (time.time() - t0) / (n + 1)), flush=True)
        if n % 20 == 0 or n == len(frames) - 1: np.savez(MVF + "/mv_fit.npz", frames=np.array(out["frames"]), X=np.array(out["X"]), loss=np.array(out["loss"]), resid=np.array(out["resid"]), dropped=np.array([sum(1 << (i - 1) for i in d) for d in out["dropped"]]), faces=Fh.faces_np, init_view=v0)
    print("done in %.0fs" % (time.time() - t0))
