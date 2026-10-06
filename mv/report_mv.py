"""Report 16: markerless multi-view SAM 3D Body mesh (one MHR body fitted to the five per-camera SAM 3D Body meshes), scored against Vicon."""
import base64, json
import cv2
import numpy as np


def _img(path, width=1300, q=60):
    im = cv2.imread(path)
    if im is None: return ""
    s = width / im.shape[1]; im = cv2.resize(im, None, fx=min(s, 1), fy=min(s, 1), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q]); return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


JS = """(function(){var I=%(imgs)s,L=%(info)s,S=%(series)s,r=document.getElementById("mv_rng"),m=document.getElementById("mv_img"),l=document.getElementById("mv_lab");
function plot(){var c=document.getElementById("mv_c1"),g=c.getContext("2d"),W=c.width,H=c.height,p=30,fg=getComputedStyle(document.body).color,ymin=0,ymax=200;g.clearRect(0,0,W,H);g.font="12px system-ui";g.fillStyle=fg;
 var n=S.frames.length;function X(k){return 46+(W-56)*k/(n-1)}function Y(v){return H-p-(H-p-10)*(v-ymin)/(ymax-ymin)}
 g.strokeStyle="rgba(128,128,128,.35)";g.lineWidth=1;for(var v=0;v<=ymax;v+=50){g.beginPath();g.moveTo(46,Y(v));g.lineTo(W-6,Y(v));g.stroke();g.fillText(v,12,Y(v)+4);}
 var sers=[S.typ,S.sv,S.mv,S.fit],cols=["#9ca3af","#ef4444","#3b82f6","#16a34a"],names=["typical single camera","best single camera (T4)","multi-view, markerless","marker-guided (reference)"];
 sers.forEach(function(s,i){g.strokeStyle=cols[i];g.lineWidth=i>1?2.4:1.6;g.beginPath();s.forEach(function(v,k){if(k==0)g.moveTo(X(k),Y(Math.min(v,ymax)));else g.lineTo(X(k),Y(Math.min(v,ymax)))});g.stroke();g.fillStyle=cols[i];g.fillText(names[i],56+i*210,14);});
 var k=+r.value;g.lineWidth=1.5;g.strokeStyle=fg;g.beginPath();g.moveTo(X(k),20);g.lineTo(X(k),H-p);g.stroke();g.fillStyle=fg;g.fillText("t = "+(S.frames[k]/29.97).toFixed(1)+" s",Math.min(X(k)+4,W-70),H-p+16);g.fillText("mm",6,28);}
r.addEventListener("input",function(){m.src=I[r.value];l.innerHTML=L[r.value];plot()});plot();})();"""


def build(ctx):
    TMP, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    MVD = f"{TMP}/mv_fit"; E2 = np.load(f"{MVD}/eval2.npz"); EV = np.load(f"{MVD}/eval.npz"); MV = np.load(f"{MVD}/mv_fit.npz"); meta = json.load(open(f"{MVD}/overlay/meta.json"))
    names = [str(n) for n in E2["names"]]; fr = E2["frames"]; F = len(fr); t = fr / 29.97
    sv = np.array([E2["sv%d" % v] for v in range(1, 6)]); mv, fit = E2["mv"], E2["fit"]; med = lambda x: float(np.nanmedian(x)); pc = lambda x, q: float(np.nanpercentile(x, q))
    cam_med = [med(sv[v]) for v in range(5)]; typ = float(np.mean(cam_med)); best_v = int(np.argmin(cam_med)); pf = lambda a: np.nanmedian(a, axis=1)
    mvf, fitf = pf(mv), pf(fit); svf = np.array([pf(sv[v]) for v in range(5)]); typf = np.nanmean(svf, axis=0)
    better_typ = 100 * np.mean(mvf < typf); better_best = 100 * np.mean(mvf < svf[best_v]); better_all = 100 * np.mean(mvf < np.nanmin(svf, axis=0))
    tail = lambda a, th=150: 100 * float(np.mean(a[~np.isnan(a)] > th))                                                       # share of marker-frames beyond 150 mm
    dropped = MV["dropped"]; drop_count = {c: int(np.sum((dropped >> (c - 1)) & 1)) for c in range(1, 6)}
    Pd = np.load(f"{TMP}/chest_track/points_dense.npz"); CJ = json.load(open(f"{TMP}/mmc17_vicon_calib.json")); Kc = np.array(CJ["K"]); mmpx = []
    for i in range(1, 6):
        c = CJ["cams"][f"T{i}"]; R_ = cv2.Rodrigues(np.array(c["rvec"]))[0]; mmpx.append(np.linalg.norm(Pd["X"].mean(0) * 1000 - (-R_.T @ np.array(c["tvec"]) * 1000)) / Kc[0, 0])
    mmpx = float(np.mean(mmpx))
    # ---- region groups
    chest = [j for j, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]; star = [j for j, n in enumerate(names) if n.startswith("*")]; low = [j for j in range(len(names)) if j not in chest and j not in star]
    groups = [("chest array (16)", chest), ("pelvis, legs, feet (%d)" % len(low), low), ("head and arms (%d, unlabelled in the files)" % len(star), star)]
    gval = {lab: dict(typ=float(np.mean([np.nanmedian(sv[v][:, ix]) for v in range(5)])), best=med(sv[best_v][:, ix]), mv=med(mv[:, ix]), fit=med(fit[:, ix])) for lab, ix in groups}
    # ---- ECDF
    f1, ax = plt.subplots(figsize=(7.4, 3.9)); lim = 300; shades = ["#cbd5e1", "#b6c2d3", "#94a3b8", "#ef4444", "#64748b"]
    for v in range(5):
        x = np.sort(sv[v][~np.isnan(sv[v])]); ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=shades[v], lw=2.0 if v == best_v else 1.1, label="single camera T%d (median %.0f mm)" % (v + 1, cam_med[v]))
    for arr, colr, lab in ((mv, C["blue"], "multi-view, markerless (median %.0f)" % med(mv)), (fit, C["green"], "marker-guided (median %.0f)" % med(fit))):
        x = np.sort(arr[~np.isnan(arr)]); ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=colr, lw=2.8, label=lab)
    ax.set_xlim(0, lim); ax.set_ylim(0, 1); ax.set_xlabel("distance from a Vicon marker to the nearest mesh vertex (mm)"); ax.set_ylabel("fraction of marker-frames"); ax.legend(fontsize=7.5, loc="lower right"); ecdf = fig_b64(f1)
    # ---- distance over time
    f2, ax = plt.subplots(figsize=(9, 3.2)); ax.plot(t, typf, color="#9ca3af", lw=1.5, label="typical single camera"); ax.plot(t, svf[best_v], color=C["red"], lw=1.5, label="best single camera (T%d, chosen with Vicon)" % (best_v + 1))
    ax.plot(t, mvf, color=C["blue"], lw=2.2, label="multi-view, markerless"); ax.plot(t, fitf, color=C["green"], lw=2.2, label="marker-guided (reference)"); ax.set_xlabel("time (s)"); ax.set_ylabel("median marker-to-mesh (mm)"); ax.set_ylim(0, 160); ax.legend(fontsize=8, ncol=2, loc="upper right"); overt = fig_b64(f2)
    # ---- regions
    f3, ax = plt.subplots(figsize=(7.6, 3.4)); w = 0.2; xs = np.arange(len(groups))
    for k, (key, colr, lab) in enumerate((("typ", "#9ca3af", "typical single camera"), ("best", C["red"], "best single camera"), ("mv", C["blue"], "multi-view, markerless"), ("fit", C["green"], "marker-guided"))):
        vals = [gval[g][key] for g, _ in groups]; ax.bar(xs + (k - 1.5) * w, vals, w, color=colr, label=lab)
        for x_, v_ in zip(xs + (k - 1.5) * w, vals): ax.text(x_, v_ + 1.5, "%.0f" % v_, ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([g for g, _ in groups], fontsize=8.5); ax.set_ylabel("median marker-to-mesh (mm)"); ax.legend(fontsize=8, ncol=2, loc="upper left"); regions = fig_b64(f3)
    # ---- chest breathing
    def detr(x):
        tt = np.arange(len(x)); ok = ~np.isnan(x); return x - np.polyval(np.polyfit(tt[ok], x[ok], 2), tt)
    p2p = lambda x: float(np.nanpercentile(x, 95) - np.nanpercentile(x, 5)); ref = detr(EV["chest_mk"]); ch = {}
    def jit(z, k=5): sm = np.convolve(z, np.ones(k) / k, "same"); return float(np.sqrt(np.mean((z - sm)[k:-k] ** 2)))
    for key in ("sv", "mv", "fit"):
        b = detr(EV["chest_" + key]); ok = ~np.isnan(ref) & ~np.isnan(b)
        ch[key] = dict(r=float(np.corrcoef(ref[ok], b[ok])[0, 1]), slope=float(np.polyfit(ref[ok], b[ok], 1)[0]), p2p=p2p(b), mae=float(np.mean(np.abs(ref[ok] - b[ok]))), rmse=float(np.sqrt(np.mean((ref[ok] - b[ok]) ** 2))), noise=jit(b), series=b)
    f4, ax = plt.subplots(figsize=(9, 3.2)); ax.plot(t, ref, color=C["blue"], lw=2.6, label="Vicon chest markers"); ax.plot(t, ch["sv"]["series"], color=C["red"], lw=1.2, label="SAM 3D Body, best single camera (T4)")
    ax.plot(t, ch["mv"]["series"], color="#7c3aed", lw=1.6, label="multi-view, markerless"); ax.plot(t, ch["fit"]["series"], color=C["green"], lw=1.2, ls="--", label="marker-guided"); ax.set_xlabel("time (s)"); ax.set_ylabel("chest height (mm, detrended)"); ax.legend(fontsize=8, ncol=2, loc="upper right"); breath = fig_b64(f4)
    # ---- scrubber (every 3rd frame)
    sel = list(range(0, len(meta), 3)); imgs = [_img(f"{MVD}/overlay/{meta[i]['frame']:04d}.jpg") for i in sel]
    info = [f"frame {meta[i]['frame']} (t = {meta[i]['frame'] / 29.97:.1f} s) &middot; marker&rarr;mesh: single T4 {meta[i]['d_sv']:.0f} mm, multi-view {meta[i]['d_mv']:.0f} mm, marker-guided {meta[i]['d_fit']:.0f} mm, typical single camera {meta[i]['d_typ']:.0f} mm" for i in sel]
    series = dict(frames=[meta[i]["frame"] for i in sel], typ=[round(meta[i]["d_typ"], 1) for i in sel], sv=[round(meta[i]["d_sv"], 1) for i in sel], mv=[round(meta[i]["d_mv"], 1) for i in sel], fit=[round(meta[i]["d_fit"], 1) for i in sel])
    js = JS % dict(imgs=json.dumps(imgs), info=json.dumps(info), series=json.dumps(series))
    # ---- tables
    rows = [["single camera T%d" % (v + 1), "%.1f" % cam_med[v], "%.1f" % pc(sv[v], 75), "%.1f" % pc(sv[v], 90), "gated out in all %d frames" % drop_count[v + 1] if drop_count[v + 1] == F else ("dropped in %d frames" % drop_count[v + 1] if drop_count[v + 1] else "kept")] for v in range(5)]
    rows += [["<b>typical single camera</b> (mean of the five medians)", "<b>%.1f</b>" % typ, "", "", "what you get if you do not know the best camera"],
             ["<b>multi-view, markerless</b>", "<b>%.1f</b>" % med(mv), "%.1f" % pc(mv, 75), "%.1f" % pc(mv, 90), "T3 dropped in %d/%d frames (T2 also in %d)" % (drop_count[3], F, drop_count[2])],
             ["marker-guided (uses the Vicon markers)", "%.1f" % med(fit), "%.1f" % pc(fit, 75), "%.1f" % pc(fit, 90), "the reference of reports 7&ndash;14"]]
    var = [["base: pseudo-Huber 20 px, all views", "46.1 / 44.3", "36.8 / 37.3", "93.2 / 98.3"], ["<b>gate: base, then drop inconsistent views</b> (used)", "41.6 / 40.3", "<b>36.8 / 37.7</b>", "90.3 / 86.7"], ["Cauchy 25 px, all views", "38.1 / 41.4", "51.2 / 40.6", "148.3 / 107.6"], ["gate + Cauchy", "35.0 / 37.3", "48.2 / 41.2", "144.8 / 108.8"]]
    chrows = [["Vicon chest markers", "&ndash;", "&ndash;", "%.1f" % p2p(ref), "&ndash;", "&ndash;", "&ndash;"]]
    for key, lab in (("sv", "SAM 3D Body, best single camera (T4)"), ("mv", "multi-view, markerless"), ("fit", "marker-guided (reference)")): c = ch[key]; chrows.append([lab, "%.3f" % c["r"], "%.2f" % c["slope"], "%.1f" % c["p2p"], "%.2f" % c["mae"], "%.2f" % c["rmse"], "%.2f" % c["noise"]])
    body = f"""
<div class="card"><b>Question.</b> Is there a multi-view SAM 3D Body that does not use the Vicon markers, and is it better than one camera? (Until now the green mesh in the recordings was marker-guided, and the red one a single camera.)</div>
{kpis([("%.0f mm" % med(mv), "multi-view, markerless: median marker-to-mesh distance (146 frames, 5 cameras)"), ("%.0f mm" % typ, "a typical single camera (mean of T1&ndash;T5 medians: %s)" % " / ".join("%.0f" % c for c in cam_med)), ("%.1f mm" % cam_med[best_v], "the best single camera, T%d (picked <i>with Vicon</i>; not known in advance)" % (best_v + 1)), ("%.0f mm" % med(fit), "marker-guided mesh (needs the markers): the reference")])}
<div class="card ok"><b>Answer.</b> Built and scored. The markerless multi-view mesh is <b>{typ / med(mv):.1f}&times; closer to the markers than a typical single camera</b> ({typ:.0f} &rarr; {med(mv):.0f} mm) and better than the average camera in {better_typ:.0f} % of frames; it finds and drops the worst camera (T3, {cam_med[2]:.0f} mm) by itself in every frame, which a single-camera system cannot do. <b>It does not beat the best single camera</b> (T{best_v + 1}, {cam_med[best_v]:.1f} mm; the multi-view mesh is better in {better_best:.0f} % of frames): its value is that you do not have to know which camera that is. And it does <b>not</b> improve the chest signal for breathing (below). The remaining ~{med(mv):.0f} mm is not noise that more cameras average away: the five SAM 3D Body meshes share their errors.</div>
<h2>What was done</h2>
<ol><li><b>One SAM 3D Body mesh per camera</b> (Fast-SAM-3D-Body / MHR, 18,439 vertices) on each of the five cameras, every 6th frame (146 frames, ~1 s per view), moved into the Vicon frame with the marker-based camera geometry. Each is good in its own image and wrong in depth, differently in each camera.</li>
<li><b>One body for all views.</b> A single MHR body (pose parameters, a rigid correction, the shape of the initialisation view) is optimised so that ~6,000 of its vertices project onto the <i>same vertices of each view's mesh</i>, in all five cameras at once (pseudo-Huber, 20 px). It is a consensus of the five predictions in the image plane; depth comes from the geometry of the camera ring.</li>
<li><b>Gating.</b> Per frame, a view whose residual after the warm start exceeds max(2&times; the median over views, 80 px) is dropped for that frame and the body refitted ({', '.join('T%d: %d' % (c, n) for c, n in drop_count.items() if n)} frames).</li>
<li><b>Tracking.</b> First frame: multi-start from each view's pose, keep the lowest loss (here T{int(MV['init_view']) + 1}). Later frames: warm start with a temporal prior, 45 optimiser steps.</li>
<li><b>Scoring.</b> The Vicon markers are used only afterwards: for each marker, the distance to the nearest of the same ~6,000 vertices of every mesh (single cameras, multi-view, marker-guided) so the numbers compare like with like. Nearest-vertex distance carries a small positive bias (~5&ndash;10 mm) for every method.</li></ol>
<h2>Results</h2>
{table(["Mesh", "median (mm)", "75th %", "90th %", "note"], rows, num=(1, 2, 3))}
{fig(ecdf, "Distance from every Vicon marker to each mesh over all 146 frames (cumulative). Left of a curve = closer. Beyond 150 mm: multi-view %.1f %% of marker-frames, T%d %.1f %%, T5 %.1f %%, T1 %.1f %%, T2 %.1f %%, T3 %.1f %%." % (tail(mv), best_v + 1, tail(sv[best_v]), tail(sv[4]), tail(sv[0]), tail(sv[1]), tail(sv[2])))}
{fig(overt, "Median marker-to-mesh distance per frame. Typical single camera %.0f&ndash;%.0f mm; best camera T%d %.0f&ndash;%.0f mm; multi-view %.0f&ndash;%.0f mm; marker-guided %.0f&ndash;%.0f mm." % (np.nanmin(typf), np.nanmax(typf), best_v + 1, np.nanmin(svf[best_v]), np.nanmax(svf[best_v]), np.nanmin(mvf), np.nanmax(mvf), np.nanmin(fitf), np.nanmax(fitf)))}
{fig(regions, "By region (median over markers and frames). Chest array: multi-view %.0f mm, best camera %.0f (typical %.0f). Pelvis, legs, feet: multi-view %.0f, best camera %.0f (typical %.0f). Head and arms: multi-view %.0f mm but camera T%d alone %.0f (typical %.0f, individual cameras %s): the one region where the consensus is clearly worse than a single camera." % (gval[groups[0][0]]["mv"], gval[groups[0][0]]["best"], gval[groups[0][0]]["typ"], gval[groups[1][0]]["mv"], gval[groups[1][0]]["best"], gval[groups[1][0]]["typ"], gval[groups[2][0]]["mv"], best_v + 1, gval[groups[2][0]]["best"], gval[groups[2][0]]["typ"], " / ".join("%.0f" % np.nanmedian(sv[v][:, star]) for v in range(5))))}
<h2>The recording</h2>
<figure><img id="mv_img" src="{imgs[0]}" style="width:100%">
<div style="display:flex;gap:10px;align-items:center;margin-top:6px"><input id="mv_rng" type="range" min="0" max="{len(sel) - 1}" value="0" style="flex:1"><span id="mv_lab" class="meta" style="min-width:420px;text-align:right">{info[0]}</span></div>
<canvas id="mv_c1" width="1100" height="170" style="width:100%;margin-top:8px;border:1px solid var(--line);border-radius:8px"></canvas>
<figcaption>Five cameras (T1&ndash;T5; T3 is gated out of the fit). <span style="color:#ff5a5a">&#9679;</span> SAM 3D Body, one camera (T4) &nbsp; <span style="color:#5a96ff">&#9679;</span> markerless multi-view mesh &nbsp; <span style="color:#46dc6e">&#9679;</span> marker-guided mesh &nbsp; <span style="color:#00c8ff">&#9711;</span> Vicon markers. Every 3rd of the 146 frames; drag the slider. The interactive 3D version with all three meshes is <code>recordings/tracking_all_meshes.rrd</code>.</figcaption></figure>
<script>{js}</script>
<h2>Does it help breathing?</h2>
{fig(breath, "Chest surface height under the chest markers, from each mesh, against the Vicon chest array (all detrended). The multi-view mesh follows the breathing (r = %.2f) but at about 43 %% of the amplitude." % ch["mv"]["r"])}
{table(["Chest signal from", "r", "slope vs Vicon", "peak-to-peak (mm)", "MAE (mm)", "RMSE (mm)", "frame-to-frame noise (mm)"], chrows, num=(1, 2, 3, 4, 5, 6))}
<p>The multi-view mesh does <b>not</b> improve the breathing signal: its amplitude is {ch['mv']['p2p']:.1f} mm against {p2p(ref):.1f} mm on the markers (slope {ch['mv']['slope']:.2f}; the temporal prior and the shape consensus damp the small chest motion that no single SAM 3D Body mesh predicts well), the correlation is about the same as one camera ({ch['mv']['r']:.2f} vs {ch['sv']['r']:.2f}), and the error is similar ({ch['mv']['mae']:.1f} vs {ch['sv']['mae']:.1f} mm). The single-camera series has the right amplitude but is noisier frame to frame. Only the marker-guided mesh (which is given the chest) and, independently, the skin tracking of reports 13&ndash;17 recover millimetres. This is the argument for the two-layer design of report 12: a coarse, low-rate multi-view mesh for the body, with high-rate skin tracking driving the chest.</p>
<p class="meta">&ldquo;frame-to-frame noise&rdquo; = RMS of the series about its own 5-sample average. (An earlier note of mine called RMSE against Vicon &ldquo;jitter&rdquo;: RMSE is the column to its left.)</p>
<h2>Why isn&rsquo;t it better, and what the loss variants say</h2>
{table(["Variant (frames 250 / 700)", "internal residual on kept views (px)", "marker&rarr;surface median vs Vicon (mm)", "90th percentile (mm)"], var, num=(1, 2, 3))}
<ul><li><b>The internal residual does not measure accuracy.</b> A Cauchy loss made the five views agree better with each other (38.1 vs 41.6 px on frame 250) and the mesh got <i>worse</i> against Vicon (51.2 vs 36.8 mm): the views are not independent noisy copies of the truth, they carry correlated, view-dependent errors; fitting them harder fits the errors. That is why the gate (drop a view that disagrees with the others) rather than a robust loss is used.</li>
<li><b>The consensus has no image evidence of its own.</b> It only sees SAM 3D Body's outputs. A body that all five meshes get wrong in the same way (the supine, arms-up pose, a systematic offset in the chest-to-bed depth) is reproduced faithfully. Median residual between the five predictions after fitting: {np.median(MV['resid']):.0f} px at 4K (~{np.median(MV['resid']) * mmpx / 10:.0f} cm at the chest).</li>
<li><b>The gate is per camera, not per body part.</b> Camera T{best_v + 1} sees the head and arms to {gval[groups[2][0]]['best']:.0f} mm while the multi-view mesh is at {gval[groups[2][0]]['mv']:.0f} mm there (the other four cameras are 97&ndash;283 mm on the arms-up pose and pull the consensus off). A per-body-part weighting of the views is the obvious next experiment; the catch is that, without ground truth, the only evidence for trusting a view on a part is its agreement with the others.</li>
<li><b>Next, in order of expected gain:</b> (a) add terms that look at the images: 2D keypoints and the person silhouette in every camera (so the fit is no longer circular); (b) drive the chest and abdomen with the dense skin tracks of report 17 (the high-rate layer); (c) fine-tune SAM 3D Body on the marker-guided meshes of all subjects (needs their videos).</li></ul>
<h2>Cost</h2>
<p>Per frame offline (RTX 4060, shared with other jobs, un-optimised PyTorch): <b>~11 s</b> to collect the five per-view meshes (1,612 s for 146 frames; SAM 3D Body itself is ~1 s per view, the rest is decoding 4K frames and saving meshes) plus <b>~4 s</b> for the multi-view fit (699 s in all, of which 129 s is the first-frame multi-start; ~3.9 s per later frame), ~15 s in total. Not real time; the fit is an optimisation that a trained multi-view head would replace in a deployed system.</p>
<h2>Limits</h2>
<ul><li class="warn">One subject, one trial, one pose (supine, thoracic breathing). Which camera is best is specific to this rig and pose; with other subjects the rank of the single cameras will change, and that is exactly the case the multi-view mesh protects against.</li>
<li class="warn">The &ldquo;typical single camera&rdquo; is the mean of the five camera medians; choosing the best one needs Vicon. Against the best camera the multi-view mesh is <i>slightly worse</i> ({med(mv):.1f} vs {cam_med[best_v]:.1f} mm).</li>
<li class="warn">Camera geometry comes from the marker calibration (Vicon-frame extrinsics); no markers enter the fit itself. T6 is excluded (unsynchronised).</li>
<li>Scores use nearest vertex of 6,000 subsampled vertices, not a true surface distance; the absolute values are a few millimetres high for every mesh.</li></ul>
<h2>Reproduce</h2><pre># SAM3D env:   five per-view meshes every 6th frame -> tmp/mv_sam3d/
python mv_collect.py          # (1,612 s)
python mv_fit.py              # SAM3D env: the multi-view fit -> tmp/mv_fit/mv_fit.npz   (699 s)
python mv_eval.py ; python mv_eval2.py ; python mv_variants.py 250 700
python make_mv_overlays.py ; python make_tracking_all_rrd.py        # mosaics + Rerun file</pre>"""
    page("16_multiview_sam3d_mesh", "16 &middot; Markerless multi-view SAM 3D Body mesh", "MMC17 TDB &middot; one body fitted to the five per-camera meshes, scored against Vicon", body)
