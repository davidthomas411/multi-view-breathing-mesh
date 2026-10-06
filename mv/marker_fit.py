"""Marker-guided MHR fit (MoSh-style): pull the SAM 3D Body mesh onto the labelled Vicon markers.

Per frame:  SAM3D (best of T1-T5) -> MHR params -> mesh in the Vicon frame -> optimise {rigid correction, body pose} so that
(marker-attached vertex + 8 mm along its normal) matches each Vicon marker (robust loss), with a pose prior to the SAM3D estimate.
Marker->vertex assignment = nearest vertex, re-estimated each outer iteration (ICP style, annealed).  Reports marker->surface distance
before/after (mm) and writes a 5-camera overlay.   usage (SAM3D env): python marker_fit.py [frame ...]
"""
import os, sys, json, io, contextlib
import cv2, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.environ.get("SAM3D_REPO", "C:/dev/Fast-SAM-3D-Body"))
import mmc17_sam3d as M, vicon_calib as V, sam3d_mv as S
from sam3d_vs_vicon import surface

K = V.K; CAMJ = json.load(open(M.OUT + "/mmc17_vicon_calib.json"))["cams"]; NAMES = V.NAMES; DEV = "cuda"
OFFSET = 0.008          # marker sphere radius, metres

def cam_pose(i): c = CAMJ["T%d" % (i + 1)]; return cv2.Rodrigues(np.array(c["rvec"]))[0], np.array(c["tvec"])

def sam3d_raw(img, i):
    est, yolo = S.models(); r = yolo(img, verbose=False)[0]
    if len(r.boxes) == 0: return None
    b = r.boxes.xyxy.cpu().numpy(); a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]); b = b[int(np.argmax(a))][None].astype(np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        o = est.process_one_image(img, bboxes=b, cam_int=torch.tensor(K[None], dtype=torch.float32))
    return o[0] if o else None

def vertex_normals(v, faces):
    fn = torch.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]], dim=1); vn = torch.zeros_like(v)
    for k in range(3): vn.index_add_(0, faces[:, k], fn)
    return torch.nn.functional.normalize(vn, dim=1)

LABELS = json.load(open(M.OUT + "/mmc17_marker_labels.json")) if os.path.exists(M.OUT + "/mmc17_marker_labels.json") else {}
import marker_labels as ML

def label_of(n): return LABELS[n]["label"] if n in LABELS else n

def group_of(label):
    """Body part a (named) marker must attach to; None = unrestricted."""
    if label.endswith("AHD"): return "head"
    if label[:1] in "RL" and label[1:] in ("RAD", "ULN", "FIN"): return "wrist" + label[0]
    if label[:1] in "RL" and label[1:] in ("ELB", "MELB", "AUPA", "PUPA", "FRA"): return "arm" + label[0]
    return "arm?" if label.startswith("*") else None

def body_part_vertices(o):
    """Vertex index sets chosen on the initial mesh (identity then follows the vertices as the pose changes). Side-aware: MHR70 6/8/41 = right shoulder/elbow/wrist, 5/7/62 = left."""
    v = np.asarray(o["pred_vertices"]); kp = np.asarray(o["pred_keypoints_3d"])
    head = np.where(np.linalg.norm(v - kp[:5].mean(0), axis=1) < 0.14)[0]
    def seg(a, b):
        ab = b - a; t = np.clip(((v - a) @ ab) / (ab @ ab), 0, 1); return np.linalg.norm(v - (a + t[:, None] * ab), axis=1)
    out = dict(head=head)
    for side, (sh, el, wr) in (("R", (6, 8, 41)), ("L", (5, 7, 62))):
        out["wrist" + side] = np.where(np.linalg.norm(v - kp[wr], axis=1) < 0.16)[0]
        out["arm" + side] = np.where(np.minimum(seg(kp[sh], kp[el]), seg(kp[el], kp[wr])) < 0.12)[0]
    out["arm?"] = np.union1d(out["armR"], out["armL"])
    return out

def attach_groups(verts_np, faces_np, pts_np, names, parts):
    """Attach every marker to the closest surface point, restricted to its body part where the part is known."""
    import trimesh
    tid = np.zeros(len(pts_np), int); bary = np.zeros((len(pts_np), 3)); done = np.zeros(len(pts_np), bool)
    def sub(idx):
        keep = np.zeros(len(verts_np), bool); keep[idx] = True; return np.where(keep[faces_np].all(1))[0]
    gr = {}
    for i, n in enumerate(names):
        g = group_of(label_of(n))
        if g: gr.setdefault(g, []).append(i)
    for g, mi in gr.items():
        fi = sub(parts[g]); m = trimesh.Trimesh(verts_np, faces_np[fi], process=False)
        cp, d, t = trimesh.proximity.closest_point(m, pts_np[mi]); b = trimesh.triangles.points_to_barycentric(m.triangles[t], cp)
        tid[mi] = fi[t]; bary[mi] = np.clip(b, 0, 1) / np.clip(b, 0, 1).sum(1, keepdims=True); done[mi] = True
    rest = np.where(~done)[0]
    if len(rest):
        t, b = attach(verts_np, faces_np, pts_np[rest]); tid[rest] = t; bary[rest] = b
    return tid, bary

def attach(verts_np, faces_np, pts_np):
    """Closest surface point of each marker: triangle id + barycentric weights (continuous, not vertex-snapped)."""
    import trimesh
    m = trimesh.Trimesh(verts_np, faces_np, process=False); cp, d, tid = trimesh.proximity.closest_point(m, pts_np)
    bary = trimesh.triangles.points_to_barycentric(m.triangles[tid], cp); return tid, np.clip(bary, 0, 1) / np.clip(bary, 0, 1).sum(1, keepdims=True)

class Fitter:
    def __init__(self):
        self.est, _ = S.models(); self.head = self.est.model.head_pose; self.faces_np = np.asarray(self.est.faces); self.faces = torch.tensor(self.faces_np, dtype=torch.long, device=DEV)

    def forward(self, p, body):
        """MHR forward in the model frame (metres) from a dict of tensors; body = (1,133) pose."""
        out = self.head.mhr_forward(global_trans=torch.zeros(1, 3, device=DEV), global_rot=p["global_rot"], body_pose_params=body, hand_pose_params=p["hand"],
                                    scale_params=p["scale"], shape_params=p["shape"], expr_params=p["expr"])
        return out[0] if isinstance(out, tuple) else out

    def init(self, o):
        t = lambda k: torch.tensor(np.asarray(o[k], np.float32), device=DEV).reshape(1, -1)
        return dict(global_rot=t("global_rot"), hand=t("hand_pose_params"), scale=t("scale_params"), shape=t("shape_params"), expr=t("expr_params")), t("body_pose_params")

    def fit(self, o, i, Vf, iters=(150, 400, 500), verbose=True, fit_shape=True, use_parts=True, use_weights=True):
        p, body0 = self.init(o); R, t = cam_pose(i); shape0, scale0 = p['shape'].clone(), p['scale'].clone()
        Rt = torch.tensor(R.T, dtype=torch.float32, device=DEV); tt = torch.tensor(t, dtype=torch.float32, device=DEV); cam_t = torch.tensor(np.asarray(o["pred_cam_t"], np.float32), device=DEV)
        with torch.no_grad(): vloc = self.forward(p, body0)[0]
        # does mhr_forward reproduce the model's own camera-frame vertices (up to a y/z flip)?
        pv = torch.tensor(np.asarray(o["pred_vertices"], np.float32), device=DEV); flip = torch.tensor([1.0, -1.0, -1.0], device=DEV)
        e_id = (vloc - pv).abs().mean().item(); e_fl = (vloc * flip - pv).abs().mean().item(); self.flip = flip if e_fl < e_id else torch.ones(3, device=DEV)
        if verbose: print("  mhr_forward vs model vertices: mean abs diff %.4f m (no flip) / %.4f m (y,z flipped) -> %s" % (e_id, e_fl, "flip" if e_fl < e_id else "no flip"))
        ok = ~np.isnan(Vf).any(1); mk = torch.tensor(Vf[ok], dtype=torch.float32, device=DEV); mnames = [str(n) for n in np.array(NAMES)[ok]]; parts = body_part_vertices(o); wts = torch.tensor([ML.weight(label_of(n)) if use_weights else 1.0 for n in mnames], dtype=torch.float32, device=DEV)
        def to_vicon(vl, rc, tc):                       # model frame -> camera -> Vicon, then rigid correction
            vc = vl * self.flip + cam_t; vv = (vc - tt) @ Rt.T
            Rcorr = torch.linalg.matrix_exp(torch.stack([torch.zeros_like(rc[0]), -rc[2], rc[1], rc[2], torch.zeros_like(rc[0]), -rc[0], -rc[1], rc[0], torch.zeros_like(rc[0])]).reshape(3, 3))
            return vv @ Rcorr.T + tc
        rc = torch.zeros(3, device=DEV, requires_grad=True); tc = torch.zeros(3, device=DEV, requires_grad=True); body = body0.clone().requires_grad_(True)
        def dist_report(vl):
            with torch.no_grad(): vv = to_vicon(vl, rc, tc).cpu().numpy()
            return cKDTree(surface(vv, self.faces_np)).query(Vf[ok])[0] * 1000
        d0 = dist_report(vloc); info = dict(before=d0, names=mnames)
        for stage, (n_it, sigma, lam) in enumerate(zip(iters, (0.10, 0.05, 0.02), (0.05, 0.01, 0.005))):
            with torch.no_grad(): vl = self.forward(p, body)[0]; vv = to_vicon(vl, rc, tc)
            tid_np, bw_np = attach_groups(vv.cpu().numpy(), self.faces_np, mk.cpu().numpy(), mnames, parts) if use_parts else attach(vv.cpu().numpy(), self.faces_np, mk.cpu().numpy()); tid = torch.tensor(tid_np, device=DEV); bw = torch.tensor(bw_np, dtype=torch.float32, device=DEV)     # re-attach markers: closest surface point
            groups = [{"params": [rc, tc], "lr": 5e-3}] + ([{"params": [body], "lr": 1e-2}] if stage > 0 else [])
            if fit_shape and stage == 2:                                           # last stage: subject shape/scale too (regularised to the SAM3D estimate)
                for k in ("shape", "scale"): p[k] = p[k].detach().clone().requires_grad_(True)
                groups.append({"params": [p["shape"], p["scale"]], "lr": 1e-2})
            opt = torch.optim.Adam(groups)
            for it in range(n_it):
                opt.zero_grad(); vl = self.forward(p, body)[0]; vv = to_vicon(vl, rc, tc); nrm = vertex_normals(vv, self.faces)
                fv = self.faces[tid]; pred = (bw[:, :, None] * vv[fv]).sum(1) + OFFSET * nrm[fv].mean(1).div(nrm[fv].mean(1).norm(dim=1, keepdim=True)); r = (pred - mk).norm(dim=1); rho = (sigma ** 2) * (torch.sqrt(1 + (r / sigma) ** 2) - 1)       # pseudo-Huber
                loss = (wts * rho).sum() + lam * ((body - body0) ** 2).sum() + 0.003 * ((p['shape'] - shape0) ** 2).sum() + 0.003 * ((p['scale'] - scale0) ** 2).sum()
                loss.backward(); opt.step()
            if verbose: print("  stage %d: marker residual (assigned vertex+offset) median %.0f mm, loss %.4f" % (stage, 1000 * r.median().item(), loss.item()))
        with torch.no_grad(): vl = self.forward(p, body)[0]; vv_final = to_vicon(vl, rc, tc).cpu().numpy()
        info["after"] = dist_report(vl); info["verts_before"] = ((vloc * self.flip + cam_t - tt) @ Rt.T).cpu().numpy(); info["verts_after"] = vv_final
        info["state"] = dict(p={k: v.detach().clone() for k, v in p.items()}, body=body.detach().clone(), rc=rc.detach().clone(), tc=tc.detach().clone(), cam_t=cam_t, Rt=Rt, tt=tt, flip=self.flip, tid=tid, bw=bw, mnames=mnames, ok=ok); info["pose_change_rad"] = float((body - body0).abs().max().item()); info["shape_change"] = float((p["shape"] - shape0).abs().max().item()); return info

REG = {"torso": [n for n in NAMES if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()], "pelvis": [n for n in NAMES if n[1:] in ("ASIS", "GRT", "ILCR", "SPSK", "IPSK")],
       "thigh/shank": [n for n in NAMES if any(k in n for k in ("SATH", "IATH", "SPTH", "IPTH", "KNE", "SASK", "IASK", "TTUB"))], "ankle/foot": [n for n in NAMES if any(k in n for k in ("ML", "TOE", "D1MT", "D5MT", "TIP"))]}

def overlay(frame, info, Vf, fname):
    tiles = []
    for i in range(5):
        im = cv2.cvtColor(M.read("TDB", i + 1, frame), cv2.COLOR_RGB2BGR); R, t = cam_pose(i)
        for key, col in (("verts_before", (80, 80, 255)), ("verts_after", (60, 230, 60))):
            vv = info[key][::6]; Xc = vv @ R.T + t; uv = Xc @ K.T; uv = uv[:, :2] / uv[:, 2:3]; z = Xc[:, 2]
            for (x, y), zz in zip(uv, z):
                if zz > 0 and 0 <= x < 3840 and 0 <= y < 2160: cv2.circle(im, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
        ok = ~np.isnan(Vf).any(1); Xc = Vf[ok] @ R.T + t; uv = Xc @ K.T; uv = uv[:, :2] / uv[:, 2:3]
        for x, y in uv: cv2.circle(im, (int(x), int(y)), 16, (255, 220, 0), 4, cv2.LINE_AA)
        cv2.putText(im, "T%d" % (i + 1), (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8); tiles.append(cv2.resize(im, (640, 360)))
    tiles.append(np.zeros_like(tiles[0])); cv2.imwrite(fname, np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]), [cv2.IMWRITE_JPEG_QUALITY, 80])

if __name__ == "__main__":
    frames = [int(x) for x in sys.argv[1:]] or [420]; F = Fitter(); FITDIR = M.OUT + "/" + os.environ.get("FIT_DIR", "fit"); os.makedirs(FITDIR, exist_ok=True)
    for f in frames:
        Vf = V.vicon(f) * 1e-3; best = None
        for i in range(5):                                                         # pick the camera whose SAM3D mesh is already closest to the markers
            o = sam3d_raw(M.read("TDB", i + 1, f), i)
            if o is None: continue
            p, b0 = F.init(o)
            with torch.no_grad(): vl = F.forward(p, b0)[0]
            R, t = cam_pose(i); vc = (vl.cpu().numpy() * np.array([1, -1, -1]) + np.asarray(o["pred_cam_t"]) - t) @ R
            ok = ~np.isnan(Vf).any(1); d = cKDTree(surface(vc, F.faces_np)).query(Vf[ok])[0]
            if best is None or np.median(d) < best[0]: best = (np.median(d), i, o)
        print("frame %d: starting from camera T%d (median marker->mesh %.0f mm if flip assumed)" % (f, best[1] + 1, best[0] * 1000))
        info = F.fit(best[2], best[1], Vf); a, b = info["before"], info["after"]
        print("  marker->mesh surface distance:  before median %.0f mm (p90 %.0f)   ->   after median %.0f mm (p90 %.0f);  max pose param change %.2f rad" % (np.median(a), np.percentile(a, 90), np.median(b), np.percentile(b, 90), info["pose_change_rad"]))
        for r, ns in REG.items():
            m = np.isin(info["names"], ns)
            if m.any(): print("    %-12s before %4.0f mm -> after %4.0f mm  (%d markers)" % (r, np.median(a[m]), np.median(b[m]), m.sum()))
        np.savez(f"{FITDIR}/fit_{f:04d}.npz", verts_before=info["verts_before"], verts_after=info["verts_after"], before=a, after=b, names=info["names"])
        overlay(f, info, Vf, f"{FITDIR}/overlay_{f:04d}.jpg")
