"""Report 15: skin-only chest tracking with every reflective ball painted out of the video (answering: is the markerless result real?)."""
import glob, json, os
import numpy as np


def build(ctx):
    TMP, img_b64, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    CT = f"{TMP}/chest_track"; J = lambda n: json.load(open(f"{CT}/{n}.json")); N = lambda n: np.load(f"{CT}/{n}.npz", allow_pickle=True)
    m4k, m1080, m960, m21, msam, un4k = J("eval_s1.0_m41"), J("eval_s0.5_m41"), J("eval_s0.25_m41"), J("eval_s1.0_m21"), J("eval_s0.5_m41sam3d"), J("eval_s1.0")
    U, M = N("eval_s1.0"), N("eval_s1.0_m41")
    def detr(x):
        t = np.arange(len(x)); ok = ~np.isnan(x); return x - np.polyval(np.polyfit(t[ok], x[ok], 2), t)
    ku, km = np.where(U["keep"])[0], np.where(M["keep"])[0]; common = np.intersect1d(ku, km); cu = [int(np.where(ku == i)[0][0]) for i in common]; cm_ = [int(np.where(km == i)[0][0]) for i in common]
    a = detr(U["s_vic"]); same = []
    for name, D, cols in (("balls visible (windows kept &gt;32 px away)", U["D"], cu), ("balls painted out of the video", M["D"], cm_)):
        b = detr(np.nanmedian(D[:, cols, 2], axis=1)); ok = ~np.isnan(a) & ~np.isnan(b)
        same.append([name, len(common), "%.3f" % np.corrcoef(a[ok], b[ok])[0, 1], "%.2f" % np.polyfit(a[ok], b[ok], 1)[0], "%.1f" % (np.nanpercentile(b, 95) - np.nanpercentile(b, 5)), "%.2f" % np.mean(np.abs(a[ok] - b[ok]))])
    dz = U["D"][:, cu, 2] - M["D"][:, cm_, 2]; rms = np.sqrt(np.nanmean(dz ** 2, axis=0))
    t = M["frames"] / 29.97; f, ax = plt.subplots(2, 1, figsize=(9, 4.0), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.2]))
    ax[0].plot(t, M["a"], color=C["blue"], lw=2.2, label="Vicon chest markers"); ax[0].plot(t, M["b"], "--", color=C["red"], lw=1.6, label="skin only, balls painted out (%d points)" % m4k["n_points"])
    ax[0].set_ylabel("chest height (mm)\n(detrended)"); ax[0].legend(fontsize=8, loc="upper right"); ax[1].plot(t, M["b"] - M["a"], color=C["grey"], lw=1); ax[1].axhline(0, color="k", lw=.5); ax[1].set_ylabel("error (mm)"); ax[1].set_xlabel("time (s)"); series = fig_b64(f)
    rows = [[lab, ev["n_points"], "%.3f" % ev["r"], "%.2f" % ev["slope"], "%.1f" % ev["p2p_track"], "%.2f" % ev["mae_mm"], "%.2f" % ev["point_jitter_mm"]] for lab, ev in (
        ("4K, 41 px window (balls painted out)", m4k), ("1080p, same physical window", m1080), ("960&times;540, same physical window", m960), ("4K, smaller 21 px window", m21), ("1080p, seeds from SAM 3D Body's mesh", msam))]
    body = f"""
<div class="card"><b>Question.</b> You pointed out that the skin-tracking result must be tested with the markers masked out. In report 13 I had only <i>dropped tracked points near the balls</i>; the balls were still in the video, and the tracking windows (41 px at 4K, physically larger at low resolution) could overlap a ball's edge. So was the &ldquo;markerless&rdquo; result really markerless?</div>
{kpis([("r = %.3f" % m4k["r"], "balls painted out of the video, 5 cameras at 4K, MMC17 TDB"), ("%.2f mm" % m4k["mae_mm"], "MAE vs Vicon with the balls removed (balls visible: %.2f mm)" % un4k["mae_mm"]), ("0.34 mm", "median RMS difference between a point's track with balls visible vs painted out (same 20 points)"), ("%d" % m4k["n_points"], "points that survive the stricter masking (114 before)")])}
<div class="card ok"><b>Answer.</b> Yes. With every ball inpainted out of every frame the result is unchanged: r {m4k['r']:.3f}, MAE {m4k['mae_mm']:.2f} mm, amplitude {m4k['p2p_track']:.1f} vs {m4k['p2p_vicon']:.1f} mm. The earlier numbers were not driven by the balls. The price is far fewer points, and a limit on what can be compared (below).</div>
<h2>What was done</h2>
<ol><li><b>Painting the balls out.</b> In every frame and camera, a 30 px disc (the ball is ~15 px in radius; the dark base is covered too) around each projected Vicon marker (71 balls per frame, missing markers interpolated in time) is inpainted (Telea) on a local crop; it costs ~0.3 s per 4K frame. Check on the chest region of one frame: my ball detector finds <b>18</b> balls before and <b>0</b> after.</li>
<li><b>Windows never touch a painted region.</b> A tracked point is kept only if it is farther than 30 + half-window + 3 px from every ball (54 px for the 41 px window), so the tracker never sees a ball, its base, or the smooth inpainted patch (which would otherwise move with the ball).</li>
<li>Everything else is as in report 13 (seed points, Lucas-Kanade from the reference frame, triangulation from &ge;3 cameras, median vertical displacement vs the 16 Vicon chest markers). The forward-backward threshold is now physically constant across resolutions.</li></ol>
{fig(img_b64(f"{CT}/inpaint_check.jpg", 1500), "A chest crop (T3, frame 403): original on the left, balls painted out on the right. Faint smudges remain where the dark bases were, which is why windows must stay &ge;54 px away.")}
<h2>Results</h2>
{fig(series, "Skin-only tracking with the balls painted out (red dashed) against Vicon (blue). Lower panel: the difference.")}
{table(["Condition", "points", "r", "slope", "peak-to-peak (mm; Vicon 16.9)", "MAE (mm)", "per-point jitter (mm)"], rows, num=(1, 2, 3, 4, 5, 6))}
<h3>Same points, balls visible vs painted out</h3>
{table(["Condition", "points", "r", "slope", "peak-to-peak (mm)", "MAE (mm)"], same, num=(1, 2, 3, 4, 5))}
<p>On the {len(common)} points valid in both runs, the median RMS difference between a point's vertical track with the balls visible and with them painted out is <b>{np.median(rms):.2f} mm</b> (90th percentile {np.percentile(rms, 90):.2f} mm). The balls were not contributing.</p>
{fig(img_b64(f"{CT}/skin_stills.jpg", 1500), "Ball-free video with the tracked skin points, coloured by how far they have moved from the reference (blue = lower, red = higher; scale &plusmn;10 mm): exhale (top, frame 922, Vicon chest &minus;8.4 mm) and inhale (bottom, frame 1006, +10.1 mm), cameras T2, T3, T4. No balls are visible on the chest.")}
<h2>What this does and does not show</h2>
<ul><li class="ok"><b>Does:</b> breathing is recovered from skin and fabric texture alone, to ~1 mm MAE on this trial, at 4K, 1080p and 960&times;540 alike, and also when the starting points come from SAM 3D Body's mesh (r = {msam['r']:.3f}, MAE {msam['mae_mm']:.2f} mm, slope {msam['slope']:.2f}).</li>
<li class="warn"><b>Only {m4k['n_points']} points</b> pass the masking: the balls are 6&ndash;8 cm apart, so a 4 cm exclusion around each leaves few places for a 41 px window. In a patient with no markers the area around where the balls were is ordinary skin, so this is a conservative test.</li>
<li class="warn"><b>The points cover the upper chest, not the abdomen.</b> By nearest chest-array row (1 = upper &hellip; 4 = lower) the {m4k['n_points']} ball-free points fall 11 / 8 / 3 / 1 (58 / 35 / 15 / 6 of the 114 with balls visible): the textured top gives points, the bare abdomen almost none. Thoracic deep breathing moves the upper rows most (report 12), so this trial favours the method. Abdominal breathing (lower rows move most) is <b>untested and probably harder</b>.</li>
<li class="warn"><b>No per-marker comparison.</b> Points within 45 mm of a marker are exactly the ones excluded, so only the aggregate (median over points vs mean of the 16 markers) is available here; the per-marker spatial check of report 13 is not repeated.</li>
<li>One subject, one trial; Vicon marker positions were used to place the masks (patients would have no balls to mask); the seed points still come from a mesh (the oracle fitted mesh, or SAM 3D Body's own in the last row).</li>
<li>The skin-tracking <b>recording</b> (ball-free video, tracked points and the 3D point cloud) is in <code>recordings/skin_tracking.rrd</code>.</li></ul>
<h2>Reproduce</h2><pre>python chest_lk_masked.py &lt;cam&gt; fit:1.0:41 fit:0.5:41 fit:0.25:41 fit:1.0:21 sam3d:0.5:41
python chest_eval.py 1.0 m41     # etc.  (tags m41, m21, m41sam3d)
python skin_frames.py &lt;cam&gt; ; python make_skin_rrd.py     # recording</pre>"""
    page("15_skin_tracking_balls_painted_out", "15 &middot; Skin-only chest tracking with the balls painted out", "MMC17 TDB &middot; answers: is the markerless chest result real?", body)
