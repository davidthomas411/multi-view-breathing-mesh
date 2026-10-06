"""Report 13: markerless chest-surface tracking vs Vicon - feasibility test with controls (MMC17 TDB)."""
import glob, json, os
import numpy as np


def build(ctx):
    TMP, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    CT = f"{TMP}/chest_track"; J = lambda n: json.load(open(f"{CT}/{n}.json"))
    e4k, e1080 = J("eval_s1.0"), J("eval_s0.5"); S = J("sweep"); Z = np.load(f"{CT}/eval_s1.0.npz", allow_pickle=True)
    st = [json.load(open(p)) for p in sorted(glob.glob(f"{CT}/static_T*.json"))]
    # ---- fig 1: time series + error
    fr = Z["frames"]; a, b = Z["a"], Z["b"]; t = fr / 29.97
    f, ax = plt.subplots(2, 1, figsize=(9, 4.3), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2]))
    ax[0].plot(t, a, color=C["blue"], lw=2.2, label="Vicon chest markers (mean of 16)"); ax[0].plot(t, b, "--", color=C["red"], lw=1.6, label="markerless: triangulated skin/fabric texture (%d points)" % e4k["n_points"])
    ax[0].set_ylabel("chest height (mm)\n(detrended)"); ax[0].legend(fontsize=8, loc="upper right"); ax[1].plot(t, b - a, color=C["grey"], lw=1); ax[1].axhline(0, color="k", lw=.5); ax[1].set_ylabel("error (mm)"); ax[1].set_xlabel("time (s)"); series = fig_b64(f)
    # ---- fig 2: scatter + Bland-Altman
    f, ax = plt.subplots(1, 2, figsize=(9, 3.4)); ax[0].scatter(a, b, s=8, color=C["blue"]); lim = [min(a.min(), b.min()) - 1, max(a.max(), b.max()) + 1]; ax[0].plot(lim, lim, "k--", lw=.8)
    ax[0].set_xlabel("Vicon (mm)"); ax[0].set_ylabel("markerless (mm)"); ax[0].set_title("agreement (identity dashed)", fontsize=9)
    m, d = (a + b) / 2, b - a; ax[1].scatter(m, d, s=8, color=C["blue"]); bi = e4k["ba_bias_mm"]; lo, hi = e4k["ba_loa_mm"]
    for y, ls in ((bi, "-"), (lo, "--"), (hi, "--")): ax[1].axhline(y, color=C["red"], ls=ls, lw=1)
    ax[1].set_xlabel("mean of the two (mm)"); ax[1].set_ylabel("markerless - Vicon (mm)"); ax[1].set_title("Bland-Altman: bias %.2f mm, 95%% LoA [%.2f, %.2f]" % (bi, lo, hi), fontsize=9); ba = fig_b64(f)
    # ---- fig 3: seed tolerance
    seeds = [("fit", "tracked surface"), ("off10", "+10 mm"), ("off20", "+20 mm"), ("off30", "+30 mm"), ("off50", "+50 mm"), ("sam3d", "SAM 3D Body mesh")]
    ev = {k: (e1080 if k == "fit" else J(f"eval_s0.5_{k}")) for k, _ in seeds}; dist = {k: S["seed_dist_mm"][k][0] for k, _ in seeds}
    f, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    for k, lab in seeds:
        col = C["green"] if k == "sam3d" else C["blue"]; ax[0].plot(dist[k], ev[k]["slope"], "o", color=col); ax[0].annotate(lab, (dist[k], ev[k]["slope"]), textcoords="offset points", xytext=(4, 5), fontsize=7)
        ax[1].plot(dist[k], ev[k]["mae_mm"], "o", color=col); ax[1].annotate(lab, (dist[k], ev[k]["mae_mm"]), textcoords="offset points", xytext=(4, 5), fontsize=7)
    ax[0].axhline(1, color="k", ls="--", lw=.7); ax[0].set_xlabel("seed distance to the true surface (mm, median)"); ax[0].set_ylabel("amplitude ratio (tracked / Vicon)"); ax[1].set_xlabel("seed distance to the true surface (mm, median)"); ax[1].set_ylabel("MAE vs Vicon (mm)"); seedfig = fig_b64(f)
    # ---- tables
    pm = e4k["per_marker"]
    pm_rows = [[p["marker"], p["n_pts"], "%.3f" % p["r"], "%.2f" % p["mae_mm"], "%.1f / %.1f" % (p["p2p_vicon"], p["p2p_track"])] for p in sorted(pm, key=lambda p: p["marker"][1:] + p["marker"][0])]
    res_rows = [[k, "%.3f" % v["r"], "%.2f" % v["slope"], "%.2f" % v["mae"], "%.1f" % v["p2p"]] for k, v in S["resolution"].items()]
    nm = S["null_mae_mm"]; nr = {**S["null_other_trials"], **{f"correct Vicon shifted {k.replace('shifted ', '')}": v for k, v in []}}
    null_rows = [["correct Vicon, correct time", "%.2f" % nm["correct Vicon (lag 0)"], "%.3f" % S["true_r"]]]
    for lag, lab in (("25", "2.5 s"), ("50", "5.0 s"), ("75", "7.5 s"), ("100", "10.0 s")): null_rows.append([f"correct Vicon, shifted {lab}", "%.2f" % nm[f"correct Vicon shifted {lab}"], "%+.2f" % S["null_shifted_vicon"][lag]])
    for man in ("DB", "ADB"): null_rows.append([f"Vicon from the {man} trial of the same subject", "%.2f" % nm[f"Vicon from the {man} trial"], "%+.2f" % S["null_other_trials"][f"Vicon from the {man} trial (different breathing)"]])
    seed_rows = [[lab, "%.1f / %.1f" % tuple(S["seed_dist_mm"][k]), ev[k]["n_points"], "%.3f" % ev[k]["r"], "%.2f" % ev[k]["slope"], "%.1f" % ev[k]["p2p_track"], "%.2f" % ev[k]["mae_mm"], "%.2f" % np.median([p["p2p_track"] / p["p2p_vicon"] for p in ev[k]["per_marker"]])] for k, lab in seeds]
    static_rows = [["T%d" % s["cam"], s["n_points"], "%.3f" % s["common_mode_rms_px"], "%.2f" % s["common_mode_max_px"], "%.2f" % s["per_point_rms_px"], "%.2f" % s["common_mode_rms_mm_at_0.77mm_per_px"]] for s in st]
    body = f"""
<div class="card"><b>Question.</b> Report 12 proposed a learned high-frequency head for breathing. Before building one, can plain multi-view tracking of chest skin/fabric texture recover the breathing signal at millimetre level on the calibrated rig &mdash; and how sensitive is it to resolution and to where the tracked points start?</div>
{kpis([("r = %.3f" % e4k["r"], "correlation with Vicon chest height (MMC17 TDB, 5 cameras at 4K)"), ("%.2f mm" % e4k["mae_mm"], "mean absolute error (lab paper's depth-network method: 1.93 mm)"), ("%.1f vs %.1f mm" % (e4k["p2p_vicon"], e4k["p2p_track"]), "peak-to-peak amplitude: Vicon vs markerless"), ("%.1f mm" % (e4k["ba_loa_mm"][1]), "Bland-Altman 95%% limit of agreement (paper: 2.05); bias %.2f mm" % e4k["ba_bias_mm"])])}
<div class="card warn"><b>Read this first.</b> One subject, one trial (MMC17 TDB). The tracked points and the region of interest come from the Vicon-fitted mesh (an oracle; a SAM 3D Body seed is tested below), and the marker balls are masked using the Vicon marker positions. Nothing here shows the method works on other subjects, clothing or drapes.</div>
<h2>Method</h2>
<ol><li><b>Points.</b> {e4k['n_candidates']} chest-surface mesh vertices (fitted mesh at frame 250, inside the chest-array footprint, facing up), projected into the five calibrated 4K cameras (T1&ndash;T5, Vicon-frame extrinsics from report 5).</li>
<li><b>Texture only.</b> The marker balls are masked out (32 px) so only skin and fabric texture is tracked; the lowest-texture 10% of points are dropped.</li>
<li><b>Tracking.</b> Pyramidal Lucas-Kanade from the reference frame to every 3rd frame (10 fps) with 41 px windows at 4K (the same <i>physical</i> patch at every resolution), seeded by the previous frame, with a forward-backward consistency check. Each frame is solved against the reference, so errors do not accumulate.</li>
<li><b>Triangulation.</b> Per point and frame from &ge;3 cameras (the worst view is dropped once if its reprojection exceeds 3 px). {e4k['n_points']} of {e4k['n_candidates']} points have &ge;3 views in more than half the frames.</li>
<li><b>Signal.</b> Median over points of the vertical (Vicon Z, anterior for a supine subject) displacement from each point's own reference position, compared with the mean of the 16 Vicon chest markers; both detrended with a 2nd-order polynomial.</li></ol>
<h2>Result</h2>
{fig(series, "Breathing (thoracic deep breathing, 5 cycles). Blue: Vicon. Red dashed: markerless. Lower panel: the difference.")}
{fig(ba, "Left: agreement with the identity line. Right: Bland-Altman plot (bias and 95 % limits of agreement).")}
{table(["Metric", "this test (4K)", "this test (1080p)", "lab paper (DA3 point cloud)"], [["correlation r", "%.3f" % e4k["r"], "%.3f" % e1080["r"], "0.896 (288 marker-level comparisons)"], ["amplitude ratio (slope)", "%.2f" % e4k["slope"], "%.2f" % e1080["slope"], "&ndash;"], ["MAE (mm)", "%.2f" % e4k["mae_mm"], "%.2f" % e1080["mae_mm"], "1.93 &plusmn; 0.99"], ["Bland-Altman bias / 95 % LoA (mm)", "%.2f / [%.2f, %.2f]" % (e4k["ba_bias_mm"], *e4k["ba_loa_mm"]), "%.2f / [%.2f, %.2f]" % (e1080["ba_bias_mm"], *e1080["ba_loa_mm"]), "&minus;0.01 / [&minus;2.06, +2.05]"], ["90th percentile |error| (mm)", "%.2f" % e4k["p90_abs_err_mm"], "%.2f" % e1080["p90_abs_err_mm"], "&ndash;"]], num=(1, 2))}
<p class="meta">The comparison with the paper is indicative only: the paper evaluates 6 markers &times; 16 subjects &times; 3 maneuvers; this is an aggregate signal plus {len(pm)} markers on one trial.</p>
<h3>Per chest marker (tracked points within 45 mm of each marker)</h3>
{table(["Marker", "points", "r", "MAE (mm)", "peak-to-peak Vicon / markerless (mm)"], pm_rows, num=(1, 2, 3))}
<h2>Controls</h2>
<h3>1. Is a high correlation meaningful for a periodic signal? (null controls)</h3>
{table(["Compared with", "MAE (mm)", "correlation r"], null_rows, num=(1, 2))}
<p>Breathing is nearly periodic, so correlation alone is a weak test: a 7.5 s-shifted copy of the <i>correct</i> Vicon trace still gives r = +0.78. The millimetre error separates them clearly: <b>{nm['correct Vicon (lag 0)']:.2f} mm</b> for the correct signal against 3&ndash;9 mm for every wrong one.</p>
<h3>2. Does the video itself move? (static background)</h3>
{table(["Camera", "static points", "common-mode shift RMS (px)", "max (px)", "per-point jitter (px)", "common-mode (mm at 0.77 mm/px)"], static_rows, num=(1, 2, 3, 4, 5))}
<p>Static room features track to ~0.03&ndash;0.05 px common-mode and ~0.1 px per point, so Gyroflow stabilisation, tripod flex or compression cannot account for a 17 mm breathing signal (they are &lt; 0.05 mm).</p>
<h3>3. Does resolution matter? (same {S['matched_points']} points at every resolution)</h3>
{table(["Resolution", "r", "slope", "MAE (mm)", "peak-to-peak (mm)"], res_rows, num=(1, 2, 3, 4))}
<ul><li>No loss down to 480&times;270 (the apparent <i>improvement</i> at 480p, 0.71 mm, is within what point choice and chance can do; it is not evidence that low resolution is better).</li>
<li class="warn"><b>This corrects report 12,</b> which said resolution matters a great deal. That argument holds for tracking a <i>single</i> point; this signal is a median over ~{S['matched_points']} points each seen by ~3 cameras with sub-pixel Lucas-Kanade, so pixel noise averages away (per-point jitter 0.13&ndash;0.17 mm at every scale). The limit is elsewhere (seed accuracy, texture, point selection).</li></ul>
<h3>4. How accurate must the starting points be? (seed tolerance)</h3>
{fig(seedfig, "Points are raised vertically above the true surface by a fixed amount (blue) or taken from SAM 3D Body's own mesh (green). Left: recovered amplitude relative to Vicon. Right: MAE.")}
{table(["Seed", "distance to the true surface, median / p90 (mm)", "points kept", "r", "slope", "peak-to-peak (mm)", "MAE (mm)", "per-marker amplitude ratio (median)"], seed_rows, num=(2, 3, 4, 5, 6, 7))}
<ul><li><b>Tolerance is about 10&ndash;20 mm.</b> Points floating 9 mm above the surface behave like points on it; at 26 mm the amplitude falls by ~28 %, at 43 mm by ~79 %. Cause: when a point is off the surface, the cameras track different physical locations, and the triangulated point no longer rides the surface.</li>
<li><b>SAM 3D Body's own mesh is a usable seed on this frame</b> (median 12.8 mm from the true surface, MAE 0.97 mm), but its tail is long (p90 34.8 mm, and single frames can be 100+ mm off, report 12), so a deployed system needs a robust choice of reference frame and a seed refinement (e.g. choose the depth along each ray that maximises multi-view patch agreement).</li></ul>
<h2>Cost (measured, CPU, excludes video decoding)</h2>
{table(["Resolution", "LK forward+backward, 306 points, one camera", "five cameras sequential"], [["4K", "43 ms", "~215 ms"], ["1080p", "11 ms", "~57 ms"], ["960&times;540", "4.9 ms", "~25 ms"], ["480&times;270", "1.5 ms", "~7.5 ms"]], num=(1, 2))}
<p>Plus ~3.5 ms to triangulate 120 points from five cameras. Cameras are independent, so they parallelise. The fine layer is therefore cheap compared with SAM 3D Body (about 1 s per view).</p>
<h2>What this means for the two-layer plan</h2>
<ul><li class="ok">The <b>fine end can be measured without a learned model</b> on this trial: r = {e4k['r']:.2f} and ~1 mm MAE from classical multi-view tracking, with the signal being almost one-dimensional (report 12).</li>
<li>That changes what a learned head would need to justify. It is not needed for basic accuracy on this subject; it would be for <b>robustness</b> (clothing, drapes, low texture, lighting, occlusion), for <b>starting points</b> without an oracle mesh, and for <b>speed</b>.</li>
<li>The coarse layer matters through the <b>seed</b>: its surface must be within ~1 cm over the chest. That is a concrete accuracy target for fine-tuning SAM 3D Body on the marker-fitted meshes.</li></ul>
<h2>Limits</h2>
<ul><li>One subject and one trial (thoracic deep breathing); the other 15 subjects' videos are not available locally.</li>
<li><b>Coverage:</b> the tracked points are concentrated in the upper chest rows (58 / 35 / 15 / 6 of 114 across rows 1&ndash;4), where thoracic breathing is strongest; abdominal breathing is untested (see <a href="15_skin_tracking_balls_painted_out.html">report 15</a>).</li>
<li><b>Balls visible in the video:</b> this test only dropped points near the balls; <a href="15_skin_tracking_balls_painted_out.html">report 15</a> repeats it with the balls painted out and finds the same result.</li>
<li>Oracle inputs: mesh-derived points/ROI and Vicon-based ball masking (the SAM 3D Body seed removes the first; patients have no balls).</li>
<li>Reference-relative: displacement from a reference frame, not absolute position. Absolute positioning needs the coarse layer.</li>
<li>The subject wears a textured top with seams plus bare skin; hospital gowns or drapes may offer far less texture.</li>
<li>Per-marker comparisons cover only the {len(pm)} markers that have at least two tracked points within 45 mm.</li></ul>
<h2>Reproduce</h2><pre>python chest_points.py                         # seed points from the fitted mesh
python chest_lk.py &lt;cam&gt; 1.0                   # per-camera tracking at 4K (also 0.5)
python chest_lk_multi.py &lt;cam&gt; fit:0.25 fit:0.125 off10:0.5 off20:0.5 off30:0.5 off50:0.5 sam3d:0.5
python chest_eval.py 1.0 [variant]             # triangulate + compare with Vicon
python chest_sweep.py ; python static_check.py &lt;cam&gt;     # controls</pre>"""
    page("13_chest_tracking_feasibility", "13 &middot; Markerless chest tracking vs Vicon: feasibility and controls", "multi-view texture tracking &middot; MMC17 TDB &middot; experiment 1 of report 12", body)
