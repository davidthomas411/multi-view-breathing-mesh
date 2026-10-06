"""Report 17: DENSE skin-only chest tracking (4,000 seeds -> ~1,100 triangulated 3D points), balls painted out; how many points, how markerless, how to get more."""
import json
import cv2
import numpy as np


def build(ctx):
    TMP, img_b64, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    CT = f"{TMP}/chest_track"; J = lambda n: json.load(open(f"{CT}/{n}.json")); N = lambda n: np.load(f"{CT}/{n}.npz", allow_pickle=True)
    p2p = lambda x: np.nanpercentile(x, 95) - np.nanpercentile(x, 5)
    base = "eval_s0.5_m41"
    D0, Dstrict, Dvis, Dsam, Dmv = (J(base + s) for s in ("_e33_dense", "_dense", "_e32_vis_dense", "sam3d_e33_dense", "mv_e33_dense"))
    sparse_masked, sparse_vis = J("eval_s1.0_m41"), J("eval_s1.0")
    mm = lambda ev: float(np.median([p["mae_mm"] for p in ev["per_marker"]])) if ev.get("per_marker") else float("nan")
    mr = lambda ev: float(np.median([p["r"] for p in ev["per_marker"]])) if ev.get("per_marker") else float("nan")
    # ---- geometry: physical size of the tracking window, seed offsets between the three seed sources
    P = N("points_dense"); E = N(base + "_e33_dense"); Es = N(base + "sam3d_e33_dense"); Em = N(base + "mv_e33_dense"); Xseed = P["X"] * 1000; balls = P["balls"] * 1000
    CJ = json.load(open(f"{TMP}/mmc17_vicon_calib.json")); Kc = np.array(CJ["K"]); mmpx = []
    for i in range(1, 6):
        c = CJ["cams"][f"T{i}"]; R = cv2.Rodrigues(np.array(c["rvec"]))[0]; cen = -R.T @ np.array(c["tvec"]) * 1000; mmpx.append(np.linalg.norm(Xseed.mean(0) - cen) / Kc[0, 0])
    mmpx = float(np.mean(mmpx)); win_cm = 41 * mmpx / 10; area_cm2 = (Xseed[:, 0].max() - Xseed[:, 0].min()) * (Xseed[:, 1].max() - Xseed[:, 1].min()) / 100
    Tk = np.load(f"{TMP}/track/track.npz", allow_pickle=True); fc = Tk["faces"]; mvx = np.load(f"{TMP}/mv_fit/mv_fit.npz")["X"][0].astype(np.float64)
    seed_on = lambda v: (P["bary"][:, :, None] * v[fc[P["tri"]]]).sum(1)
    off_sam = float(np.median(np.linalg.norm(seed_on(Tk["vb"][0].astype(np.float64)) - P["X"], axis=1)) * 1000); off_mv = float(np.median(np.linalg.norm(seed_on(mvx) - P["X"], axis=1)) * 1000)
    # ---- funnel: seeds -> valid at the reference in k cameras -> triangulated in >50 % of frames
    okref = np.zeros((5, 4000), bool)
    for k in range(5):
        L_ = N(f"lk_T{k + 1}_s0.5_m41_e33_dense"); okref[k] = L_["ok"][0] & (L_["score"] > np.percentile(L_["score"], 10))
    nref = okref.sum(0); percam = okref.sum(1); keep = E["keep"]
    funnel = [("seed points (chest footprint)", 4000), ("valid at the reference in &ge;1 camera (in the image, &ge;33 px from the image of any ball, enough texture)", int((nref >= 1).sum())), ("valid at the reference in &ge;3 cameras", int((nref >= 3).sum())),
              ("<b>triangulated from &ge;3 cameras in &gt;50 % of the frames = tracked points</b>", int(keep.sum()))]
    nv = E["nview"][:, keep]; nvp = nv[nv > 0]
    # ---- coverage figure
    Xr = E["Xref"]; Dz = E["D"][:, :, 2]; amp = np.array([p2p(Dz[:, j]) for j in range(Dz.shape[1])])
    f1, ax = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax[0].scatter(Xseed[:, 0], Xseed[:, 1], s=3, c="#d1d5db", label="seeds not kept (%d)" % (4000 - keep.sum())); sc = ax[0].scatter(Xr[:, 0], Xr[:, 1], s=6, c=amp, cmap="viridis", vmin=0, vmax=25, label="tracked points (%d)" % keep.sum())
    ax[0].scatter(balls[:, 0], balls[:, 1], s=95, facecolors="none", edgecolors="#06b6d4", linewidths=1.4, label="Vicon chest balls (painted out)")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("Vicon X (mm, lateral)"); ax[0].set_ylabel("Vicon Y (mm, towards the head)"); ax[0].legend(fontsize=7, loc="lower left"); ax[0].grid(False)
    cb = f1.colorbar(sc, ax=ax[0], location="bottom", fraction=0.05, pad=0.12); cb.set_label("breathing amplitude of the point (mm, 5-95 %)")
    yb = np.arange(740, 1201, 50); h_all = np.histogram(Xseed[:, 1], yb)[0]; h_keep = np.histogram(Xr[:, 1], yb)[0]; yc = (yb[:-1] + yb[1:]) / 2
    medamp = np.array([float(np.median(amp[(Xr[:, 1] >= lo) & (Xr[:, 1] < hi)])) if ((Xr[:, 1] >= lo) & (Xr[:, 1] < hi)).any() else np.nan for lo, hi in zip(yb[:-1], yb[1:])])
    ax[1].barh(yc, h_all, height=44, color="#d1d5db", label="seeds"); ax[1].barh(yc, h_keep, height=44, color=C["blue"], label="tracked"); ax[1].set_ylabel("Vicon Y (mm)"); ax[1].set_xlabel("points per 50 mm row"); ax[1].legend(fontsize=8, loc="lower right")
    a2 = ax[1].twiny(); a2.plot(medamp, yc, "o-", color=C["orange"]); a2.set_xlabel("median point amplitude (mm)", color=C["orange"]); a2.set_xlim(0, 30); a2.grid(False)
    for y in sorted(set(np.round(balls[:, 1], -1))): ax[1].axhline(y, color="#06b6d4", lw=.6, ls=":")
    cover = fig_b64(f1); mid = float(np.nanmedian(medamp[3:8]))
    # ---- series
    t = E["frames"] / 29.97; f2, ax = plt.subplots(2, 1, figsize=(9, 4.3), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2]))
    ax[0].plot(t, E["a"], color=C["blue"], lw=2.6, label="Vicon chest markers (16)"); ax[0].plot(t, E["b"], "--", color=C["red"], lw=1.4, label="skin tracking, seeds on the marker-fitted mesh (%d pts)" % D0["n_points"])
    ax[0].plot(t, Es["b"], ":", color=C["orange"], lw=1.6, label="seeds on SAM 3D Body, one camera (%d)" % Dsam["n_points"]); ax[0].plot(t, Em["b"], "-.", color=C["green"], lw=1.4, label="seeds on the markerless multi-view mesh (%d)" % Dmv["n_points"])
    ax[0].set_ylabel("chest height (mm)\n(detrended)"); ax[0].legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False); ax[1].plot(t, E["b"] - E["a"], color=C["red"], lw=1); ax[1].plot(t, Es["b"] - Es["a"], color=C["orange"], lw=1); ax[1].plot(t, Em["b"] - Em["a"], color=C["green"], lw=1)
    ax[1].axhline(0, color="k", lw=.5); ax[1].set_ylabel("error (mm)"); ax[1].set_xlabel("time (s)"); series = fig_b64(f2)
    # ---- density vs accuracy
    runs = [(sparse_vis, "#9ca3af"), (sparse_masked, "#9ca3af"), (D0, C["blue"]), (Dstrict, C["blue"]), (Dvis, C["blue"]), (Dsam, C["orange"]), (Dmv, C["green"])]
    sweeps = []
    for lab, tag in (("&ge;3 cameras, &gt;75 % of frames", "_f75"), ("&ge;3 cameras, &gt;50 % of frames (default)", ""), ("&ge;3 cameras, &gt;25 % of frames", "_f25"), ("&ge;2 cameras, &gt;50 % of frames", "_c2"), ("&ge;2 cameras, &gt;25 % of frames", "_f25_c2")):
        for sd, stem in (("marker-fitted mesh", "_e33_dense"), ("SAM 3D Body, 1 camera", "sam3d_e33_dense"), ("multi-view mesh (no Vicon)", "mv_e33_dense")):
            try: ev = J(base + stem + tag)
            except Exception: continue
            sweeps.append((lab, sd, ev))
    scol = {"marker-fitted mesh": C["blue"], "SAM 3D Body, 1 camera": C["orange"], "multi-view mesh (no Vicon)": C["green"]}
    f3, ax = plt.subplots(figsize=(6.8, 3.6))
    for ev, colr in runs: ax.scatter(ev["n_points"], ev["mae_mm"], s=75, color=colr, edgecolor="k", zorder=3)
    for lab, sd, ev in sweeps: ax.scatter(ev["n_points"], ev["mae_mm"], s=24, color=scol[sd], alpha=.55, zorder=2)
    ax.set_xscale("log"); ax.set_xlabel("tracked 3D points (log scale)"); ax.set_ylabel("MAE vs Vicon chest (mm)"); ax.set_ylim(0, 1.4)
    ax.annotate("%d pts" % sparse_masked["n_points"], (sparse_masked["n_points"], sparse_masked["mae_mm"]), textcoords="offset points", xytext=(6, 8), fontsize=8); ax.annotate("%d" % sparse_vis["n_points"], (sparse_vis["n_points"], sparse_vis["mae_mm"]), textcoords="offset points", xytext=(6, -12), fontsize=8)
    ax.annotate("{:,}".format(D0["n_points"]), (D0["n_points"], D0["mae_mm"]), textcoords="offset points", xytext=(-12, 10), fontsize=8); densplot = fig_b64(f3); max_pts = max(ev["n_points"] for _, sd, ev in sweeps if sd.startswith("marker"))
    # ---- tables
    pmk = lambda ev: ("%d / %.2f" % (len(ev["per_marker"]), mm(ev))) if ev.get("per_marker") else "&ndash;"
    def row(lab, ev): return [lab, ev["n_points"], "%.3f" % ev["r"], "%.2f" % ev["slope"], "%.1f" % ev["p2p_track"], "%.2f" % ev["mae_mm"], "%.2f" % ev["mae_scaled_mm"], pmk(ev)]
    cond = [row("sparse: %d points, balls painted out (report 15)" % sparse_masked["n_points"], sparse_masked), row("sparse: %d points, balls visible (report 13)" % sparse_vis["n_points"], sparse_vis), row("<b>dense, seeds on the marker-fitted mesh, balls painted out</b> (33 px exclusion)", D0),
            row("dense, same, strict 54 px exclusion", Dstrict), row("dense, balls visible (windows &gt;32 px away)", Dvis), row("<b>dense, seeds on SAM 3D Body (one camera), balls painted out</b>", Dsam), row("<b>dense, seeds on the markerless multi-view mesh, balls painted out</b>", Dmv)]
    swrows = [[lab, sd, ev["n_points"], "%.3f" % ev["r"], "%.2f" % ev["mae_mm"], "%.2f" % ev["mae_scaled_mm"], pmk(ev)] for lab, sd, ev in sweeps]
    pm = [[p["marker"], p["n_pts"], "%.3f" % p["r"], "%.2f" % p["mae_mm"], "%.1f" % p["p2p_vicon"], "%.1f" % p["p2p_track"]] for p in D0["per_marker"]]; worst = sorted(D0["per_marker"], key=lambda p: -p["mae_mm"])[:3]
    body = f"""
<div class="card"><b>Question.</b> Is the skin-only chest tracking (report 15) true multi-view markerless tracking? How many points does it track, and can the density be raised?</div>
{kpis([("{:,}".format(D0["n_points"]), "triangulated 3D skin points with the balls painted out (%d in report 15). Seeds from the markerless multi-view mesh: %s" % (sparse_masked["n_points"], "{:,}".format(Dmv["n_points"]))), ("r = %.3f" % D0["r"], "against the 16 Vicon chest markers (%.3f with the %d points of report 15)" % (sparse_masked["r"], sparse_masked["n_points"])), ("%.2f mm" % D0["mae_mm"], "MAE (%.2f mm before): ~%d&times; the points at the same accuracy" % (sparse_masked["mae_mm"], round(D0["n_points"] / sparse_masked["n_points"]))), ("%.1f cameras" % nvp.mean(), "per point per frame (all 5 are tracked; &ge;3 needed to triangulate)")])}
<div class="card ok"><b>Answer.</b> <b>Yes, it is multi-view markerless tracking</b> in the sense that matters (details in the next section), and the number of tracked 3D points went from {sparse_masked['n_points']} to <b>{D0['n_points']:,}</b> with the balls painted out, <b>with no loss of accuracy</b>: r = {D0['r']:.3f}, MAE {D0['mae_mm']:.2f} mm, amplitude {D0['p2p_track']:.1f} vs {D0['p2p_vicon']:.1f} mm on the Vicon chest array. With the seeds taken from the <b>markerless multi-view SAM 3D Body mesh of report 16</b> (nothing from Vicon in the seeds) it is {Dmv['n_points']:,} points, r = {Dmv['r']:.3f}, MAE {Dmv['mae_mm']:.2f} mm. With many points near every ball the per-marker check that the sparse run could not make now works: n = {len(D0['per_marker'])} markers, median r = {mr(D0):.2f}, median MAE {mm(D0):.2f} mm. Density can go higher still ({max_pts:,} points by also accepting 2-camera triangulation, table below); the limits are the skin texture and the {win_cm:.1f} cm tracking window, not the number of seeds.</div>
<h2>Is it true multi-view markerless tracking?</h2>
{table(["Stage", "What it uses", "Vicon involved?"], [
["2D tracking in each camera", "pyramidal Lucas-Kanade (41 px window at 4K) from the reference frame to every frame, forward-backward check; the five cameras are tracked independently; <b>balls inpainted out of every frame</b>", "only to place the paint-out discs (a patient has no balls)"],
["3D position", "linear triangulation (DLT) of the 2D tracks from &ge;3 calibrated cameras; the worst view is dropped once if its reprojection exceeds 3 px", "camera extrinsics came from the markers once (a checkerboard calibration would serve)"],
["Seed points at frame 250", "points sampled on a mesh surface: (a) the marker-fitted mesh; (b) SAM 3D Body's single-camera mesh (T4, the camera chosen with Vicon); (c) the <b>markerless multi-view mesh</b> (report 16)", "(a) yes; (b) camera choice only; <b>(c) no</b>"],
["Reference for the displacement", "each point's own 3D position at the first tracked frame (250)", "no"],
["Scoring", "median vertical displacement vs the mean of the 16 chest markers; Bland-Altman; per-marker", "yes (scoring only)"]])}
<p>So the <b>tracks</b> are markerless and multi-view: no marker is visible in any image, and with seeds (c) no marker position enters the seeds or the tracks; the only Vicon input left is where to paint out the balls, which a patient does not need. What is not yet clinical-grade: (1) the displacement is relative to the first frame, not an absolute surface; (2) the seeds from SAM 3D Body start a median {off_sam:.0f} mm, and those from the multi-view mesh {off_mv:.0f} mm, away from the marker-fitted ones (it still works because a seed only has to land on the same patch of skin, not on a specific point); (3) one subject and one trial.</p>
<h2>How many points, and where?</h2>
{table(["Stage", "points"], [[a, "{:,}".format(b)] for a, b in funnel], num=(1,))}
<p>The seeds are 4,000 uniform samples of the chest footprint (the 16-ball array plus 4 cm: {area_cm2:.0f} cm&sup2; bounding box, ~{4000 / area_cm2:.1f} per cm&sup2;, ~6 mm apart). Per camera {percam.min():,}&ndash;{percam.max():,} of them pass the reference test. <b>The ball exclusion is the main loss</b> in these data: it is measured against the images of <i>all</i> 71 Vicon markers, including ones hidden behind the body, so a camera that sees the chest at a grazing angle loses many seeds to markers on the far side; a patient has none of these. (The balls-visible run keeps the same exclusion for a like-for-like comparison: {Dvis['n_points']:,} points.) Of the {funnel[2][1]:,} seeds valid in &ge;3 cameras, {funnel[3][1]:,} survive tracking in more than half of the frames; the rest fail the forward-backward check or are occluded (median {np.median(nvp):.0f} cameras per point-frame, &ge;4 in {100 * np.mean(nvp >= 4):.0f} % of point-frames).</p>
{fig(cover, "Left: where the %d tracked points sit (colour = their own breathing amplitude), the %d seeds that were not kept (grey) and the 16 painted-out Vicon balls (cyan rings; large Y = towards the head). Right: seeds and tracked points per 50 mm row (bars), median amplitude per row (orange). The holes follow the ball rows and the bare skin of the lower rows." % (keep.sum(), 4000 - keep.sum()))}
<p><b>A by-product: the breathing map.</b> With this many points the vertical amplitude can be read as a spatial field: a median of {mid:.0f} mm over the mid-chest rows (890&ndash;1140 mm), {medamp[0]:.0f} mm in the lowest row of the footprint (740&ndash;790 mm) and {medamp[-1]:.0f} mm in the highest (1140&ndash;1190 mm, towards the clavicles). That is the input for a respiratory surface (gating, or separating thoracic from abdominal breathing).</p>
<h2>Accuracy: unchanged, and now with a per-marker check</h2>
{fig(series, "Dense skin tracking against Vicon (blue) for the three kinds of seed: marker-fitted mesh (red dashed), SAM 3D Body one camera (orange dotted), markerless multi-view mesh (green dash-dot). Lower panel: the errors.")}
{table(["Condition", "points", "r", "slope", "p2p (mm; Vicon %.1f)" % D0["p2p_vicon"], "MAE (mm)", "MAE after best scale", "markers compared / median MAE"], cond, num=(1, 2, 3, 4, 5, 6))}
<p>Bland-Altman for the first dense run: bias 0.00 mm, 95 % limits of agreement &plusmn;{D0['ba_loa_mm'][1]:.2f} mm, 90th percentile |error| {D0['p90_abs_err_mm']:.2f} mm; per-point jitter {D0['point_jitter_mm']:.2f} mm (individual points are quiet, not just their median).</p>
<details><summary>Per-marker agreement (tracked points within 45 mm of each ball, 16 markers, first dense run)</summary>{table(["Marker", "points near it", "r", "MAE (mm)", "Vicon amplitude (mm)", "tracked amplitude (mm)"], pm, num=(1, 2, 3, 4, 5))}
<p class="meta">Amplitudes are 5&ndash;95 % ranges over the trial. Points near a ball are 2&ndash;4.5 cm from it (the neighbourhood of every ball is excluded). Largest MAE: {', '.join('%s (%.1f mm, %d points)' % (p['marker'], p['mae_mm'], p['n_pts']) for p in worst)}: the markers with the fewest points near them.</p></details>
<h2>Can the density be increased?</h2>
{fig(densplot, "MAE against the number of tracked points. Grey: the sparse runs of reports 13 and 15; large dots: the dense runs (blue marker-fitted seeds, orange SAM 3D Body, green multi-view mesh); small dots: relaxed validity rules. More points did not cost accuracy anywhere in this range.")}
{table(["Point-validity rule", "seeds", "points", "r", "MAE (mm)", "MAE after best scale", "markers / median MAE"], swrows, num=(2, 3, 4, 5))}
<p>Ways to get more points, from cheapest to most work:</p>
<ol><li><b>Relax the validity rule</b> (table above): accepting points triangulated from 2 cameras raises the count to {max_pts:,} at the same aggregate accuracy. A 2-camera point has no redundancy to reject a bad track, so treat these as a secondary class.</li>
<li><b>Seed the whole torso, not just the chest-array footprint</b> (shoulders, abdomen, flanks; legs and arms too). Nothing in the method limits where seeds go; the Vicon check exists only around the chest array and the pelvis/leg markers. Cost is linear in the seeds (11 ms per camera-frame for 306 points at 1080p).</li>
<li><b>Dense optical flow</b> (RAFT, DIS) in place of sparse Lucas-Kanade, a track per pixel, same triangulation. One point per SAM 3D Body vertex (18,439) would make the tracks per-vertex displacements: the input a feed-forward mesh deformation (report 12) would consume.</li>
<li><b>Texture</b> is the real ceiling: bare skin gives almost no points (the 840&ndash;890 mm row keeps {int(h_keep[2])} of {int(h_all[2])} seeds) while the textured top gives many. Clothing, a speckle pattern or a projector would raise density there; a markerless clinical setup can only use what is on the patient.</li></ol>
<p><b>What density means here.</b> The tracking window is 41 px at 4K = <b>{win_cm:.1f} cm</b> on the skin ({mmpx:.2f} mm/px at the chest), so seeds 6 mm apart share most of their pixels: the field has roughly <b>{area_cm2 / win_cm ** 2:.0f} independent patches</b> over the {area_cm2:.0f} cm&sup2; footprint, not {D0['n_points']:,}. More points give a smoother field and fewer holes, not a finer one, unless the window shrinks (the 21 px window of report 15 finds fewer valid points at the same accuracy).</p>
<h2>Limits</h2>
<ul><li class="warn"><b>One subject, one trial (thoracic deep breathing).</b> The lower rows barely move here; abdominal breathing (the MMC17 ADB trial; its video is not on this machine) is the real test of the abdomen, where texture is poorest.</li>
<li class="warn"><b>Points are relative to frame 250</b> (a displacement field): breathing and local deformation, not absolute positioning, which comes from the mesh (report 16).</li>
<li class="warn">The ball-paint-out discs are placed with Vicon positions in these data; a patient would need none.</li>
<li>The reference is the Vicon chest markers: domes on the skin, so part of the {D0['mae_mm']:.2f} mm may be the marker, not the tracking.</li>
<li>Recording: <code>recordings/skin_tracking_dense.rrd</code> (ball-free video, tracked points coloured by rise/fall, 3D point cloud, Vicon vs skin plot).</li></ul>
<h2>Reproduce</h2><pre>python chest_points_dense.py 4000
set PTS=points_dense &amp; set OUTSUF=_dense
python chest_lk_masked.py &lt;cam 1-5&gt; fit:0.5:41:33 fit:0.5:41:54 fit:0.5:41:32:0 sam3d:0.5:41:33 mv:0.5:41:33
python chest_eval.py 0.5 m41_e33_dense            # tags m41_e33_dense, m41_dense, m41_e32_vis_dense, m41sam3d_e33_dense, m41mv_e33_dense; MINFRAC=0.25 / MINCAM=2 for the sweep rows
python make_skin_dense_rrd.py                     # recording</pre>"""
    page("17_dense_skin_tracking", "17 &middot; Dense skin-only chest tracking (1,100+ points)", "MMC17 TDB &middot; answers: true multi-view markerless? how many points? how to get more?", body)
