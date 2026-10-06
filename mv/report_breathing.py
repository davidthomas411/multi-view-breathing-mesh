"""Report 12: breathing - project review, signal statistics, approach comparison (MoSE3 etc.) and a proposed design."""
import json
import numpy as np


def build(ctx):
    TMP, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    R = json.load(open(f"{TMP}/breathing_stats.json")); mans = ["DB", "TDB", "ADB"]; T = np.load(f"{TMP}/track/track.npz", allow_pickle=True)
    # --- figure 1: low-rank structure + amplitude by chest row
    f, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for k, m in enumerate(mans):
        rr = [r for r in R if r["man"] == m]; ax[0].boxplot([[100 * r["ev"][0] for r in rr]], positions=[k], widths=.5, showfliers=False)
        ax[0].scatter(np.full(len(rr), k) + np.random.default_rng(k).uniform(-.12, .12, len(rr)), [100 * r["ev"][0] for r in rr], s=10, color=C["blue"], zorder=3)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(mans); ax[0].set_ylabel("variance explained by mode 1 (%)"); ax[0].set_ylim(70, 101); ax[0].set_title("16 chest markers, one point per trial", fontsize=9)
    x = np.arange(4)
    for k, (m, col) in enumerate(zip(mans, (C["grey"], C["orange"], C["blue"]))):
        rr = [r for r in R if r["man"] == m]; ax[1].bar(x + (k - 1) * .27, np.median([r["rows"] for r in rr], 0), .27, color=col, label=m)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["row 1\n(upper)", "row 2", "row 3", "row 4\n(lower)"]); ax[1].set_ylabel("peak-to-peak (mm), median"); ax[1].legend(fontsize=8); ax[1].set_title("amplitude by chest row", fontsize=9); stat = fig_b64(f)
    # --- figure 2: chest height series: Vicon vs fitted mesh vs SAM 3D Body mesh
    fr = T["frames"]; mk = T["chest_mk"] * 1000; ft = T["chest_fit"] * 1000; bs = T["chest_base"] * 1000
    def detr(v):
        t = np.arange(len(v)); ok = ~np.isnan(v); return v - np.polyval(np.polyfit(t[ok], v[ok], 2), t)
    mk_d, ft_d, bs_d = detr(mk), detr(ft), detr(bs); ok = ~np.isnan(bs_d)
    f, ax = plt.subplots(figsize=(8.8, 3.0)); tt = fr / 29.97
    ax.plot(tt, bs_d, color=C["red"], lw=1, label="SAM 3D Body mesh surface"); ax.plot(tt, mk_d, color=C["blue"], lw=2, label="Vicon chest markers"); ax.plot(tt, ft_d, "--", color=C["green"], lw=1.5, label="marker-fitted mesh (uses the markers)")
    ax.set_ylim(-25, 25); ax.set_xlabel("time (s)"); ax.set_ylabel("chest height, mean removed (mm)"); ax.legend(fontsize=8, ncol=3, loc="upper center")
    for xo in tt[np.where(np.abs(bs_d) > 25)[0]]: ax.annotate("SAM 3D Body outlier (off scale)", (xo, -24), ha="center", va="bottom", fontsize=7, color=C["red"], arrowprops=dict(arrowstyle="->", color=C["red"]), xytext=(xo, -12))
    series = fig_b64(f)
    rng = lambda v: np.nanpercentile(v, 95) - np.nanpercentile(v, 5)
    e = bs_d - mk_d; mad = 1.4826 * np.nanmedian(np.abs(e - np.nanmedian(e))); outl = np.abs(e - np.nanmedian(e)) > 5 * mad; keep = ok & ~outl
    r_fit = np.corrcoef(mk_d, ft_d)[0, 1]; r_b = np.corrcoef(mk_d[keep], bs_d[keep])[0, 1]; rms_b = mad; rms_f = np.sqrt(np.mean((ft_d - mk_d) ** 2)); n_out = int(np.nansum(outl)); worst = float(np.nanmax(np.abs(e)))
    g = lambda m, k: np.median([r[k] for r in R if r["man"] == m])
    body = f"""
<div class="card"><b>Question.</b> The aim is respiration monitoring. Review what this project knows about the breathing signal, review other approaches (including MoSE3), and judge the idea of adding a feed-forward mesh-deformation component to Fast-SAM-3D-Body so that the coarse (global pose/shape) and fine (mm surface motion) ends are both covered.</div>
{kpis([("94 %", "of chest-array motion explained by ONE mode (median, 45 trials, 16 subjects; min 82 %)"), ("&asymp; 15 mm", "peak-to-peak chest amplitude (mean of the 16 markers, median over trials)"), ("%.1f mm" % rms_b, "SAM 3D Body chest-surface jitter (robust SD vs Vicon): ~2&times; the 1.9 mm bar, ~1/4 of the signal; plus rare gross outliers"), ("1.9 mm", "MAE of the published DA3 point-cloud method (the bar to beat)")])}
<h2>1. What the breathing signal looks like (new analysis, all 16 subjects)</h2>
{fig(stat, "Left: share of the chest-array motion carried by the first mode (an SVD of the 16 vertical marker traces, after removing slow drift). Right: median peak-to-peak amplitude per chest row for each maneuver &mdash; thoracic deep breathing is largest in the upper rows, abdominal in the lower rows.")}
{table(["Maneuver", "trials", "mean peak-to-peak (mm)", "largest marker (mm)", "mode 1", "modes 1&ndash;3"], [[m, sum(1 for r in R if r["man"] == m), "%.1f" % g(m, "p2p_mean"), "%.1f" % g(m, "p2p_max"), "%.0f %%" % (100 * np.median([r["ev"][0] for r in R if r["man"] == m])), "%.1f %%" % (100 * np.median([sum(r["ev"][:3]) for r in R if r["man"] == m]))] for m in mans], num=(1, 2, 3, 4, 5))}
<ul><li><b>The deformation field is almost one-dimensional.</b> One mode carries ~94 % of the chest motion in every maneuver; three modes carry &ge; 98 % in all 45 trials. A model does not need per-vertex freedom: it needs a few amplitudes plus a subject/maneuver-specific mode shape. This agrees with the RGB-D literature, which describes respiration with PCA models (e.g. 0.53 mm error against laser scanning).</li>
<li><b>Pixel budget (revised after report 13).</b> A 4K pixel is ~0.8 mm and chest motion is mostly along the surface normal, so a 15 mm excursion is ~10&ndash;15 px in an oblique 4K view but ~3 px at 1080p. That limits tracking of a <i>single</i> point, but the aggregate chest signal did <b>not</b> degrade down to 480&times;270 in the feasibility test, because ~90 points seen by ~3 cameras are averaged with sub-pixel Lucas-Kanade tracking.</li></ul>
<h2>2. How the coarse end performs (SAM 3D Body at the chest)</h2>
{fig(series, f"Chest surface height under the chest-array markers over {fr[0] / 29.97:.0f}&ndash;{fr[-1] / 29.97:.0f} s. SAM 3D Body (red) is dominated by frame-to-frame jitter; the Vicon markers (blue) show the breathing; the marker-fitted mesh (green) tracks it because the markers are its input.")}
{table(["Signal", "peak-to-peak (mm)", "correlation with Vicon", "error vs Vicon (mm)"], [["Vicon chest markers", "%.1f" % rng(mk_d), "&ndash;", "&ndash;"], ["Marker-fitted mesh", "%.1f" % rng(ft_d), "%.3f" % r_fit, "%.2f" % rms_f], ["SAM 3D Body mesh (single camera; outliers excluded)", "%.1f" % rng(bs_d[keep]), "%.2f" % r_b, "%.1f (robust SD)" % rms_b]], num=(1, 2, 3))}
<ul><li class="warn"><b>The coarse feed-forward end sees the breathing, noisily.</b> After excluding {n_out} gross outlier frame(s) (the worst is {worst:.0f} mm off), SAM 3D Body's chest surface follows the true signal with r = {r_b:.2f} and a robust jitter of {rms_b:.1f} mm &mdash; about a quarter of the ~{rng(mk_d):.0f} mm signal and twice the 1.9 mm accuracy of the published method. A deformation component must cut that jitter at least ~3&times; and remove the outliers.</li>
<li class="warn">The marker-fitted mesh is <b>ground truth, not a result</b>: it is built from these very markers, so r = {r_fit:.3f} is by construction. It is the supervision target for a markerless model.</li></ul>
<h2>3. Other approaches</h2>
{table(["Approach", "what it gives", "evidence / numbers", "fit for breathing"], [
 ["DA3 feed-forward multi-view depth + SAM masks (the lab's paper)", "dense surface point cloud per frame, 6 cameras", "MAE 1.93 mm, r = 0.896 over 288 marker-level comparisons; ~0.05 s per 6-camera set on an A100", "<b>The baseline.</b> No body model; ROI chosen by marker position"],
 ["RGB-D + PCA breathing models (literature)", "low-rank motion model per patient", "0.53 mm vs laser scanning; r = 0.97 vs spirometer (<a href='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5579577/'>PMC5579577</a>)", "Supports the low-rank view; needs a depth sensor"],
 ["MVTracker (multi-view 3D point tracking, ICCV 2025) <a href='https://arxiv.org/abs/2508.21060'>arXiv</a>", "feed-forward tracks of arbitrary points from ~4 cameras", "median trajectory error 3.1 cm (Panoptic) / 2.0 cm (DexYCB); trained on synthetic Kubric", "Right geometry, wrong scale: 10&times; too coarse for mm without fine-tuning on chest data"],
 ["D4RT (feed-forward 4D reconstruction) <a href='https://arxiv.org/abs/2512.08924'>arXiv</a>", "depth, tracks and cameras from one video via a query decoder", "state of the art on 4D benchmarks (cm-level scenes)", "Candidate backbone; mm accuracy unproven; monocular depth ambiguity along the breathing direction"],
 ["MoSE3 (<a href='https://mose3-tracker.github.io/'>project page</a>); builds on <a href='https://arxiv.org/abs/2012.00726'>RAFT-3D</a> / <a href='https://arxiv.org/abs/2503.07739'>SIRE</a>", "dense per-pixel SE(3) motion + rigid clusters from <b>monocular</b> RGB", "HO3D: 2.06 cm ADD, 17.96&deg; rotation error; ~5&ndash;6 s for all pixels; trained on 5k synthetic scenes of moving rigid bodies; fails on stretched/folded surfaces (per its page)", "<b>Not for the mm signal</b> (see below); relevant for separating rigid motion from deformation"],
 ["Per-view point tracking (CoTracker3 / TAPIR) triangulated with the calibrated rig", "sub-pixel 2D tracks of skin texture, triangulated in 3D", "exists in the lab's earlier scene.rrd (CoTracker3D, MVTracker experiments); no accuracy numbers yet", "Most promising <i>measurement</i>; 4K chest crops give ~0.8 mm per pixel"],
 ["Marker-fitted MHR mesh (this project, reports 7/9/12)", "body mesh in the Vicon frame, per frame", "11&ndash;13 mm median marker-to-surface, chest series r = 0.998", "<b>Ground truth / supervision</b>, needs Vicon"]])}
<h2>4. Assessment of the proposal: MoSE3 + feed-forward deformation on Fast-SAM-3D-Body</h2>
<div class="card ok"><b>Update after the feasibility test (report 13).</b> Classical multi-view texture tracking already recovers the breathing signal at ~1 mm MAE on the one trial available, so the fine end does not <i>need</i> a learned model for accuracy; a learned head would have to earn its place on robustness (clothing, drapes, low texture), on seeding without an oracle mesh, or on speed. The coarse layer matters through the seed: it must put the chest surface within ~1 cm.</div>
<ul><li><b>The premise is right.</b> Two scales are needed: a coarse global layer (pose, shape: tens of mm accuracy is fine) and a fine layer for the few-mm surface motion. The data support a very compact fine layer (1&ndash;3 amplitudes).</li>
<li><b>MoSE3 as the fine layer is a poor match, from what its page says</b> (I have read the project page and search summaries, not the paper or code): (i) it is monocular, and breathing is motion along the camera ray where monocular depth is weakest; (ii) its reported accuracy on real data is ~2 cm, 20&times; the 1 mm we need; (iii) it is trained on rigid-body motion and its page lists stretched surfaces as a failure case, while the chest is soft, smooth and slowly deforming; (iv) ~5&ndash;6 s per clip.</li>
<li><b>What from MoSE3 is useful:</b> the rigidity idea. Patient setup shifts are rigid per body part; breathing is a smooth non-rigid residual. Separating the two (an SE(3) per body part from the MHR skeleton, then a deformation on top) is exactly what SGRT positioning + respiration needs, and MHR already gives the rigid layer, so MoSE3 is not required for it.</li>
<li><b>Suggested design (both ends):</b> (a) coarse layer = Fast-SAM-3D-Body fine-tuned on supine data with the marker-fitted meshes as labels (fixes the 100&ndash;200 mm error and the jitter); (b) fine layer = a small head that predicts <b>mode amplitudes</b> {{a<sub>k</sub>(t)}} (k = 1&ndash;3) from 4K multi-view crops of the chest, with vertex displacement &Delta;v = &sum; a<sub>k</sub> B<sub>k</sub> along the surface normal, where the basis B<sub>k</sub> is learned from the Vicon chest arrays across subjects (conditioned on shape/maneuver); (c) temporal filter on a<sub>k</sub> (breathing is &lt; 1 Hz; 30 fps gives a lot of averaging).</li></ul>
<h2>5. Proposed experiments, in order, with success criteria</h2>
{table(["#", "Experiment", "Success if", "Cost"], [
 ["1", "<b>Measurement feasibility: DONE (<a href='13_chest_tracking_feasibility.html'>report 13</a>).</b> Multi-view tracking of chest skin/fabric texture, triangulated with the Vicon-frame cameras, compared with Vicon mode 1", "<b>Met on one trial:</b> r = 0.983, amplitude 17.9 vs 16.9 mm, MAE 0.92 mm (null controls 3&ndash;9 mm); seeds must lie within ~10&ndash;20 mm of the surface", "done; other subjects not available locally"],
 ["2", "<b>Oracle ceiling:</b> train the mode-amplitude head on frozen features with ground-truth chest crops (one subject held out)", "held-out subject r &ge; 0.9", "2&ndash;3 days"],
 ["3", "<b>Coarse layer:</b> fine-tune SAM 3D Body on marker-fitted meshes; report marker-to-surface and chest jitter", "chest jitter (robust SD) &lt; 1.5 mm and no gross outliers (now ~4 mm, one frame in ~170 off by &gt; 150 mm)", "needs whole-trial fits for all subjects"],
 ["4", "<b>MoSE3 only if wanted:</b> run it on a 4K chest crop and measure the mm error against Vicon", "mm-level or it is dropped", "0.5 day"]])}
<h2>6. Caveats</h2>
<ul><li>Amplitudes here are vertical (Vicon Z); the paper's amplitudes are anterior-posterior surface motion, the same direction for a supine subject.</li>
<li>The one-mode result is for the chest array only; abdomen and sides may carry more modes.</li>
<li>All numbers are healthy volunteers on a couch; patients (clothing, drapes, immobilisation) will be harder.</li>
<li>The tracking run behind the chest series is a single trial (MMC17 TDB, frames {fr[0]}&ndash;{fr[-1]}).</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe breathing_stats.py      # signal statistics, all subjects
C:\\dev\\sam3d-venv\\Scripts\\python.exe track_fit.py                   # tracking run (chest series)</pre>"""
    page("12_breathing_review", "12 &middot; Breathing: project review, approaches, and the MoSE3 idea", "signal statistics from all 16 subjects &middot; approach comparison &middot; proposed design", body)
