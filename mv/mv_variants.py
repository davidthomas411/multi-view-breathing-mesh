"""Variants of the markerless multi-view mesh fit on two frames, to choose the loss/gating by an UNSUPERVISED criterion (final multi-view reprojection residual on the
views kept), and to show what each does against Vicon (Vicon only scores, it is not used in any variant).
  base   : pseudo-Huber(20 px), all views
  gate   : base, then drop views whose residual is > max(2x the median over views, 80 px) and refit
  cauchy : Cauchy(25 px) loss, all views
  gate+cauchy
usage (SAM3D env): python mv_variants.py [frame ...]"""
import os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mv_fit as W, mmc17_sam3d as M, marker_fit as MF
DEV = W.DEV
def loss2(X, obs, okv, vidx, kind, wts):
    tot = 0.0; ws = 0.0; per = {}
    for i in range(5):
        if not okv[i]: continue
        Xc = X[vidx] @ W.Rg[i].T + W.tg[i]; u = W.fx * Xc[:, 0] / Xc[:, 2] + W.cx; w = W.fy * Xc[:, 1] / Xc[:, 2] + W.cy
        r = torch.sqrt((u - obs[i][:, 0]) ** 2 + (w - obs[i][:, 1]) ** 2 + 1e-6)
        rho = (20.0 ** 2) * (torch.sqrt(1 + (r / 20.0) ** 2) - 1) if kind == "huber" else (25.0 ** 2) * torch.log(1 + (r / 25.0) ** 2)
        tot = tot + wts[i] * rho.mean(); ws += wts[i]; per[i] = float(r.median())
    return tot / ws, per
def opt2(Fh, p, body, rc, tc, to_world, obs, okv, vidx, iters, kind, wts, lr_body=1e-2, lr_rigid=3e-3, lam=0.02):
    b0 = body.clone(); body = body.clone().requires_grad_(True); rc = rc.clone().requires_grad_(True); tc = tc.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [rc, tc], "lr": lr_rigid}] + ([{"params": [body], "lr": lr_body}] if lr_body > 0 else []))
    for it in range(iters):
        opt.zero_grad(); X = to_world(Fh.forward(p, body)[0], rc, tc); l, _ = loss2(X, obs, okv, vidx, kind, wts); (l + lam * ((body - b0) ** 2).sum()).backward(); opt.step()
    with torch.no_grad(): X = to_world(Fh.forward(p, body)[0], rc, tc); l, per = loss2(X, obs, okv, vidx, kind, wts)
    return body.detach(), rc.detach(), tc.detach(), X.detach(), per
def surface(v, faces, n=80000, rng=np.random.default_rng(0)):
    tri = v[faces]; a = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1); k = rng.choice(len(faces), n, p=a / a.sum())
    u, w = rng.random(n), rng.random(n); f = u + w > 1; u[f], w[f] = 1 - u[f], 1 - w[f]; return tri[k, 0] + u[:, None] * (tri[k, 1] - tri[k, 0]) + w[:, None] * (tri[k, 2] - tri[k, 0])
if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[1:]] or [250, 700]; Fh = MF.Fitter(); T = np.load(M.OUT + "/track/track.npz", allow_pickle=True); tf = {int(f): i for i, f in enumerate(T["frames"])}
    vidx = torch.tensor(np.load(f"{M.OUT}/mv_sam3d/{frames[0]:04d}.npz")["vidx"], device=DEV)
    for f in frames:
        obs, okv, z = W.load_obs(f); mk = T["mk"][tf[f]].astype(np.float64); okm = ~np.isnan(mk).any(1); print("=== frame %d" % f, flush=True)
        o = MF.sam3d_raw(M.read("TDB", 2, f), 1); p, b0 = Fh.init(o); to_world = W.make_world(1, np.asarray(o["pred_cam_t"]))
        for name, kind, gate in (("base", "huber", False), ("gate", "huber", True), ("cauchy", "cauchy", False), ("gate+cauchy", "cauchy", True)):
            w = {i: 1.0 for i in range(5)}; rc = torch.zeros(3, device=DEV); tc = torch.zeros(3, device=DEV)
            body, rc, tc, X, per = opt2(Fh, p, b0, rc, tc, to_world, obs, okv, vidx, 80, kind, w, 0.0)
            body, rc, tc, X, per = opt2(Fh, p, body, rc, tc, to_world, obs, okv, vidx, 250, kind, w)
            if gate:
                med = np.median(list(per.values())); drop = [i for i, r in per.items() if r > max(2.0 * med, 80.0)]
                if drop:
                    w = {i: (0.0 if i in drop else 1.0) for i in range(5)}; body, rc, tc, X, per = opt2(Fh, p, body, rc, tc, to_world, obs, okv, vidx, 150, kind, w)
            kept = [r for i, r in per.items() if w[i] > 0]
            Xn = X.cpu().numpy().astype(np.float64); d = cKDTree(surface(Xn, Fh.faces_np)).query(mk[okm])[0] * 1000
            print("  %-12s dropped views %-8s internal: median residual on kept views %.1f px | VICON score: marker->surface median %.1f mm, p90 %.1f mm" % (name, [i + 1 for i, v in w.items() if v == 0], np.median(kept), np.median(d), np.percentile(d, 90)), flush=True)
