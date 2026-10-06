"""Marker-guided tracking over the trial: full fit on the first frame, then a short warm-started refinement per frame with the markers kept on the SAME
mesh points (fixed attachments), plus SAM 3D Body's own mesh (one camera) as the baseline.  Writes tmp/track/track.npz.

usage (SAM3D env):  python track_fit.py [start stop step]      default 250 1121 3
"""
import os, sys, time, json
import cv2, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M, vicon_calib as V, sam3d_mv as S
import marker_fit as MF
import marker_labels as ML
from sam3d_vs_vicon import surface

DEV = "cuda"; K = V.K
a = [int(x) for x in sys.argv[1:4]] if len(sys.argv) > 3 else [250, 1121, 3]
CHEST = [n for n in V.NAMES if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]

def main():
    F = MF.Fitter(); os.makedirs(M.OUT + "/track", exist_ok=True); frames = list(range(*a)); f0 = frames[0]
    # ---- first frame: choose the start camera, full 3-stage fit (as in marker_fit.py)
    Vf = V.vicon(f0) * 1e-3; best = None
    for i in range(5):
        o = MF.sam3d_raw(M.read("TDB", i + 1, f0), i)
        if o is None: continue
        p, b0 = F.init(o)
        with torch.no_grad(): vl = F.forward(p, b0)[0]
        R, t = MF.cam_pose(i); vc = (vl.cpu().numpy() * np.array([1, -1, -1]) + np.asarray(o["pred_cam_t"]) - t) @ R; okm = ~np.isnan(Vf).any(1)
        d = np.median(cKDTree(surface(vc, F.faces_np)).query(Vf[okm])[0])
        if best is None or d < best[0]: best = (d, i, o)
    _, cam_i, o0 = best; print("start camera T%d; full fit on frame %d ..." % (cam_i + 1, f0), flush=True)
    info = F.fit(o0, cam_i, Vf, verbose=True); st = info["state"]
    p = {k: v.clone() for k, v in st["p"].items()}; body = st["body"].clone(); rc = st["rc"].clone(); tc = st["tc"].clone(); tid, bw = st["tid"], st["bw"]; mnames = st["mnames"]
    okmask0 = st["ok"]; name_idx = {n: i for i, n in enumerate(V.NAMES)}; att_names = mnames                          # markers that have an attachment
    Rt, tt, cam_t, flip = st["Rt"], st["tt"], st["cam_t"], st["flip"]; wts = torch.tensor([ML.weight(MF.label_of(n)) for n in att_names], dtype=torch.float32, device=DEV)
    chest_att = [k for k, n in enumerate(att_names) if n in CHEST]
    def to_vicon(vl, rc_, tc_):
        vc = vl * flip + cam_t; vv = (vc - tt) @ Rt.T
        Rcorr = torch.linalg.matrix_exp(torch.stack([torch.zeros_like(rc_[0]), -rc_[2], rc_[1], rc_[2], torch.zeros_like(rc_[0]), -rc_[0], -rc_[1], rc_[0], torch.zeros_like(rc_[0])]).reshape(3, 3)); return vv @ Rcorr.T + tc_
    def attached_points(vv):
        nrm = MF.vertex_normals(vv, F.faces); fv = F.faces[tid]
        return (bw[:, :, None] * vv[fv]).sum(1) + MF.OFFSET * nrm[fv].mean(1).div(nrm[fv].mean(1).norm(dim=1, keepdim=True))
    R_c, t_c = MF.cam_pose(cam_i); out = dict(frames=[], vf=[], vb=[], mk=[], d_fit=[], d_base=[], chest_mk=[], chest_fit=[], chest_base=[], names=np.array(att_names))
    t0 = time.time()
    for n_f, f in enumerate(frames):
        Vf = V.vicon(f) * 1e-3; mk_all = np.array([Vf[name_idx[n]] for n in att_names]); valid = ~np.isnan(mk_all).any(1)
        mk = torch.tensor(np.nan_to_num(mk_all), dtype=torch.float32, device=DEV); vmask = torch.tensor(valid, device=DEV)
        body_prev = body.clone(); body.requires_grad_(True); rc.requires_grad_(True); tc.requires_grad_(True)
        if n_f > 0:                                                                  # warm-started short refinement
            opt = torch.optim.Adam([{"params": [rc, tc], "lr": 2e-3}, {"params": [body], "lr": 4e-3}])
            for it in range(60):
                opt.zero_grad(); vl = F.forward(p, body)[0]; vv = to_vicon(vl, rc, tc); pred = attached_points(vv); r = (pred - mk).norm(dim=1); sg = 0.02
                rho = (sg ** 2) * (torch.sqrt(1 + (r / sg) ** 2) - 1); loss = (wts * rho * vmask).sum() + 0.02 * ((body - body_prev) ** 2).sum(); loss.backward(); opt.step()
        with torch.no_grad(): vl = F.forward(p, body)[0]; vv = to_vicon(vl, rc, tc); pred = attached_points(vv)
        body = body.detach(); rc = rc.detach(); tc = tc.detach(); vfit = vv.cpu().numpy()
        # ---- baseline: SAM 3D Body (start camera) -> Vicon frame
        img = M.read("TDB", cam_i + 1, f); ob = MF.sam3d_raw(img, cam_i)
        vb = ((np.asarray(ob["pred_vertices"]) + np.asarray(ob["pred_cam_t"]) - t_c) @ R_c) if ob is not None else np.full_like(vfit, np.nan)
        mkv = mk_all[valid]; ptsf = surface(vfit, F.faces_np, 80000); tf = cKDTree(ptsf); d_fit = tf.query(mkv)[0] * 1000
        if ob is not None: ptsb = surface(vb, F.faces_np, 80000); tb = cKDTree(ptsb); d_base = tb.query(mkv)[0] * 1000
        else: d_base = np.full(len(mkv), np.nan)
        cz = np.array([k for k in chest_att if valid[k]])
        out["frames"].append(f); out["vf"].append(vfit.astype(np.float32)); out["vb"].append(vb.astype(np.float32)); out["mk"].append(mk_all.astype(np.float32))
        out["d_fit"].append(np.nanmedian(d_fit)); out["d_base"].append(np.nanmedian(d_base))
        # chest surface height under the chest-array markers: the mesh surface point nearest to each marker (fitted vs SAM3D)
        out["chest_mk"].append(float(np.mean(mk_all[cz, 2])) if len(cz) else np.nan)
        out["chest_fit"].append(float(ptsf[tf.query(mk_all[cz])[1], 2].mean()) if len(cz) else np.nan)
        out["chest_base"].append(float(ptsb[tb.query(mk_all[cz])[1], 2].mean()) if (len(cz) and ob is not None) else np.nan)
        if n_f % 10 == 0: print("frame %4d (%d/%d): marker->mesh fit %.0f mm | SAM3D baseline %.0f mm | %.1fs/frame" % (f, n_f + 1, len(frames), out["d_fit"][-1], out["d_base"][-1], (time.time() - t0) / (n_f + 1)), flush=True)
        if n_f % 25 == 0 or n_f == len(frames) - 1:
            np.savez(M.OUT + "/track/track.npz", frames=np.array(out["frames"]), vf=np.array(out["vf"]), vb=np.array(out["vb"]), mk=np.array(out["mk"]), d_fit=np.array(out["d_fit"]), d_base=np.array(out["d_base"]),
                     chest_mk=np.array(out["chest_mk"]), chest_fit=np.array(out["chest_fit"]), chest_base=np.array(out["chest_base"]), names=out["names"], faces=F.faces_np, cam_i=cam_i)
    print("done in %.0fs" % (time.time() - t0))

if __name__ == "__main__":
    main()
