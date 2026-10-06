"""Reports 5 (Vicon <-> GoPro calibration from the marker balls) and 7 (marker-guided SAM 3D Body mesh fit).  Called from make_reports.py."""
import glob, json, os
import numpy as np


def _regions(names):
    names = [str(n) for n in names]
    return {
        "upper chest (rows 1-2)": [n for n in names if len(n) == 3 and n[0] in "RL" and n[1] in "12" and n[2] in "12"],
        "lower chest / abdomen (rows 3-4)": [n for n in names if len(n) == 3 and n[0] in "RL" and n[1] in "34" and n[2] in "12"],
        "pelvis": [n for n in names if n[1:] in ("ASIS", "GRT", "ILCR", "SPSK", "IPSK")],
        "thigh / shank": [n for n in names if any(k in n for k in ("SATH", "IATH", "SPTH", "IPTH", "KNE", "SASK", "IASK", "TTUB"))],
        "ankle / foot": [n for n in names if any(k in n for k in ("ML", "TOE", "D1MT", "D5MT", "TIP"))],
        "head (headband *56, *69)": [n for n in names if n in ("*56", "*69")],
        "wrists / hands (*58 *59 *60 *62 *66)": [n for n in names if n in ("*58", "*59", "*60", "*62", "*66")],
        "other arm markers (*)": [n for n in names if n.startswith("*") and n not in ("*56", "*69", "*58", "*59", "*60", "*62", "*66")],
    }


def build_r5(ctx):
    TMP, img_b64, kpis, table, fig, page = ctx["TMP"], ctx["img_b64"], ctx["kpis"], ctx["table"], ctx["fig"], ctx["page"]
    cal = json.load(open(f"{TMP}/mmc17_vicon_calib.json"))
    rows = [[k, v["n"], "%.2f" % v["rms"]] for k, v in cal["cams"].items()]
    body = f"""
<div class="card"><b>Question.</b> Where is the Vicon room frame relative to each GoPro? The aim is a real ground truth, so the registration must come from the markers themselves, not from a pose model. This repeats the idea of your <code>pnp_gopro_vicon_code</code> (labelled 2D&ndash;3D matches &rarr; <code>solvePnPRansac</code> + LM per camera), but the 2D marker positions are found automatically.</div>
{kpis([("465", "labelled markers matched to detected balls (8 frames, T1&ndash;T5)"), ("0.98&ndash;1.73 px", "per-camera PnP reprojection RMS"), ("120.7 mm", "board square size implied by the Vicon scale (SOP says 13 cm; 120 mm is used)"), ("0.5 mm", "median triangulated-vs-Vicon distance, 36 markers (consistency check)")])}
<h2>Method</h2>
<ol><li><b>Find the balls.</b> The markers are grey spheres (about 25&ndash;35 px across at 4K) sitting on a dark base. Brightness is the wrong cue (they are <i>less</i> saturated than skin or the teal top, not brighter), so the detector uses low colour saturation + roundness + a darker surround: 7&ndash;29 balls per camera per frame (recall is partial, mostly leg and lower-torso markers).</li>
<li><b>Vicon&rarr;room transform.</b> A search over yaw, tilt, translation and scale maximises how many projected Vicon markers land on detections, in all cameras and 8 frames at once (random search with a wide capture radius, then simplex, then robust least squares with the gate annealed 60 &rarr; 6 px). The transform is shared by all cameras.</li>
<li><b>Settle the board alignment with the markers.</b> For each camera, every sub-grid offset/flip hypothesis (<a href="04_mmc17_calibration.html">report 4</a>) is scored by how many Vicon markers it matches. This is a far stronger cue than the layout prior I used before.</li>
<li><b>Per-camera PnP</b> in the Vicon frame with fixed intrinsics, exactly as in your script.</li></ol>
<h2>Results</h2>
<h3>The markers resolved an alignment error that nothing else could</h3>
{table(["Camera", "option 0", "option 1", "option 2", "option 3", "alignment used before (ring prior)"], [["T1", "<b>58</b>", "0", "0", "0", "0 &#10003;"], ["T2", "<b>40</b>", "0", "1", "0", "0 &#10003;"], ["T3", "<b>227</b>", "0", "0", "0", "0 &#10003;"], ["T4", "<b>39</b>", "0", "0", "0", "<span class='bad'>2 &#10007;</span>"], ["T5", "<b>17</b>", "0", "0", "0", "<span class='bad'>2 &#10007;</span>"], ["T6", "0", "0", "0", "0", "&ndash; (matches no option)"]], num=(1, 2, 3, 4))}
<p class="meta">Marker matches within 6 px per board-alignment hypothesis (option = offset&times;2 + flip). The ring-layout prior in report 4 picked the wrong alignment for T4 and T5; only the markers could tell, because reprojection error of the board is blind to it.</p>
<h3>Per-camera calibration in the Vicon frame</h3>
{table(["Camera", "RANSAC inliers", "reprojection RMS (px)"], rows + [["T6", "&ndash;", "not solved: unsynchronized (23 fps)"]], num=(1, 2))}
{fig(img_b64(f"{TMP}/marker_detect_f410.jpg", 1500), "Detected marker balls (red circles) at frame 410, T1&ndash;T6. Detections cluster on the legs and lower torso where the dark clothing gives contrast.")}
<h2>Consistency check against Vicon</h2>
<ul><li>Markers seen by &ge;3 cameras (36, over 12 frames out to frame 760, most not used for fitting) triangulate to <b>0.5 mm median, 1.4 mm p90</b> from their Vicon coordinates. The calibration therefore holds well into the trial (it was fitted on frames 260&ndash;540).</li>
<li class="warn">This is a consistency check, not an independent accuracy number: the labels come from matching each detection to the nearest projected Vicon marker. An independent number needs held-out cameras with larger baselines.</li></ul>
<h2>What this replaces</h2>
<p>The earlier registration (<code>register_vicon.py</code>) used SAM 3D Body hips/knees/ankles and gave ~98 mm RMS and a 169 mm square. It is superseded: the marker-based registration is ~100&times; tighter and independent of any pose model.</p>
<h2>Open items</h2>
<ul><li><b>T6</b> was recorded at 24 fps (all others 30 fps, per the lab's notes). It matches no board-alignment option here; its recovery is attempted in report 10.</li>
<li><b>Square size:</b> the SOP notes say 13 cm; the Vicon scale gives 120.7 mm; 120 mm is used from now on. The Vicon-frame extrinsics are metric by construction (PnP on Vicon metres) and do not depend on this number.</li>
<li>Marker recall is limited on the torso and arms; a learned marker detector would raise it.</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\sam3d-venv\\Scripts\\python.exe vicon_calib.py        # detect, register, per-camera PnP  -> tmp\\mmc17_vicon_calib.json
C:\\dev\\sam3d-venv\\Scripts\\python.exe vicon_calib2.py       # alignment hypotheses per camera
C:\\dev\\sam3d-venv\\Scripts\\python.exe vicon_validate.py     # triangulation vs Vicon</pre>"""
    page("05_vicon_calibration", "5 &middot; Vicon &harr; GoPro calibration from the marker balls", "MMC17 TDB &middot; automatic marker detection + PnP (cf. pnp_gopro_vicon_code)", body)


def build_r7(ctx):
    TMP, img_b64, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    fits = sorted(glob.glob(f"{TMP}/fit/fit_*.npz"))
    if not fits:
        return
    data = []
    for p in fits:
        z = np.load(p, allow_pickle=True); data.append((int(os.path.basename(p)[4:8]), z["names"], z["before"], z["after"]))
    regs = _regions(data[0][1])
    agg = {}
    for r, ns in regs.items():
        b = np.concatenate([d[2][np.isin(d[1], ns)] for d in data]); a = np.concatenate([d[3][np.isin(d[1], ns)] for d in data])
        if len(b): agg[r] = (np.median(b), np.percentile(b, 90), np.median(a), np.percentile(a, 90), len(b) // len(data))
    ballb = np.concatenate([d[2] for d in data]); balla = np.concatenate([d[3] for d in data])
    f, ax = plt.subplots(figsize=(8.2, 3.2)); x = np.arange(len(agg)); labs = list(agg)
    ax.bar(x - .2, [agg[k][0] for k in labs], .4, color=C["grey"], label="SAM 3D Body (best of 5 views)"); ax.bar(x + .2, [agg[k][2] for k in labs], .4, color=C["green"], label="after marker-guided fit")
    ax.axhline(8, color=C["red"], ls="--", lw=1, label="8 mm = ball radius (perfect)")
    ax.set_xticks(x); ax.set_xticklabels([k.replace(" / ", "/\n").replace(" (", "\n(") for k in labs], fontsize=7.5); ax.set_ylabel("median marker &rarr; mesh surface (mm)".replace("&rarr;", "->")); ax.legend(fontsize=8); chart = fig_b64(f)
    per = [[f"frame {d[0]}", "%.0f / %.0f" % (np.median(d[2]), np.percentile(d[2], 90)), "%.0f / %.0f" % (np.median(d[3]), np.percentile(d[3], 90))] for d in data]
    imgs = "".join(fig(img_b64(f"{TMP}/fit/overlay_{d[0]:04d}.jpg", 1500), f"Frame {d[0]}: cyan = Vicon markers projected through the marker-based calibration (they land on the real balls); red = SAM 3D Body mesh before, green = after the marker-guided fit.") for d in data)
    base = json.load(open(f"{TMP}/sam3d_vs_vicon_per_marker.json")) if os.path.exists(f"{TMP}/sam3d_vs_vicon_per_marker.json") else {}
    body = f"""
<div class="card"><b>Question.</b> The SAM 3D Body fit on MMC17 is poor. With the Vicon markers now calibrated into the camera frame, can they guide the mesh fit &mdash; and by how much does that improve it?</div>
{kpis([("%.0f &rarr; %.0f mm" % (np.median(ballb), np.median(balla)), "median marker-to-mesh distance, before &rarr; after (%d frames)" % len(data)), ("%.0f &rarr; %.0f mm" % (np.percentile(ballb, 90), np.percentile(balla, 90)), "90th percentile"), ("103 mm", "SAM 3D Body single view, averaged over all five cameras (p90 286 mm)"), ("&asymp; 8 mm", "the floor: marker sphere radius")])}
<h2>Answer to the question</h2>
<ul><li><b>Yes, large improvement &mdash; but because the markers are used as labelled 3D points, not as unlabelled 2D detections.</b> Unlabelled 2D balls only say &ldquo;the surface is near here&rdquo;, which any mesh covering the silhouette already satisfies; the gain comes from knowing <i>which</i> body location each ball is, in metric 3D.</li>
<li>This is MoSh-style fitting: the standard way mocap markers are turned into a body mesh. Done per frame here, with SAM 3D Body as initialisation and pose prior.</li></ul>
<h2>Method</h2>
<ol><li>SAM 3D Body on the five calibrated cameras; start from the camera whose mesh is closest to the markers.</li>
<li>Move that mesh into the Vicon frame with the marker-based extrinsics (<a href="05_vicon_calibration.html">report 5</a>). The MHR layer (<code>mhr_forward</code>) is differentiable and reproduces the model's own vertices exactly (up to a y/z flip).</li>
<li>Optimise, in three stages of increasing strictness: a rigid correction; then body pose with a weak prior to the SAM 3D Body pose; then subject shape and scale too (regularised). Loss = pseudo-Huber distance between each Vicon marker and its attached surface point + 8 mm along the normal.</li>
<li>Each marker is attached to the <i>closest point on the mesh surface</i> (triangle + barycentric weights), re-estimated at every stage (ICP style).</li></ol>
<h3>Baseline: SAM 3D Body single view vs Vicon (all five cameras, five frames)</h3>
{table(["Region", "median (mm)", "p90 (mm)", "markers"], [["all markers", "103", "286", "&ndash;"], ["torso (chest array)", "87", "263", "16"], ["pelvis", "58", "250", "8"], ["thigh / shank", "91", "260", "18"], ["ankle / foot", "135", "247", "12"]], num=(1, 2, 3))}
<p class="meta">Mesh moved into the Vicon frame with the marker-based extrinsics; distance from each Vicon marker to the mesh surface (sampled at 150k points). A perfect fit would sit at ~8 mm.</p>
<h2>Results</h2>
{fig(chart, "Median distance from each Vicon marker to the mesh surface, pooled over the fitted frames. Dashed line: the ideal distance (the ball radius).")}
{table(["Region", "markers", "before median / p90 (mm)", "after median / p90 (mm)"], [[k, v[4], "%.0f / %.0f" % (v[0], v[1]), "%.0f / %.0f" % (v[2], v[3])] for k, v in agg.items()], num=(1, 2, 3))}
{table(["Frame", "before median / p90 (mm)", "after median / p90 (mm)"], per, num=(1, 2))}
<p class="meta">&ldquo;Before&rdquo; is the <i>best</i> of the five cameras, chosen using the Vicon markers, so it flatters SAM 3D Body: averaged over all five cameras its error is 103 mm median (table below).</p>
<h2>Overlays</h2>{imgs}
<h2>The unlabelled markers are the arms and head</h2>
{fig(img_b64(f"{TMP}/unlabelled_markers_f420.jpg", 1500), "The 19 unlabelled <code>*NN</code> markers projected into T1, T3, T4, T5 (frame 420). <span style='color:#dc2626'>Red</span> = wrist/hand markers (*58 *59 *60 *62 *66); the two <code>*56</code> and <code>*69</code> sit on the headband; the rest are on the upper arms and elbows.")}
<ul><li>You told me there are two markers on a headband; the projections identify them as <code>*56</code> and <code>*69</code> (both ~210 mm beyond the chest along the body axis, symmetric). The wrist markers sit on the hands. The Vicon CSV does not name them, so I identified them from the images.</li>
<li>Each is attached only to its own body part in the fit (head, wrist/hand, arm), so a wrist marker cannot latch onto the wrong limb.</li>
<li>SAM 3D Body's <b>head is already good</b> (headband markers 2&ndash;8 mm from its surface before fitting); the <b>wrists/hands were the worst</b> (124&ndash;208 mm) and improve substantially but not fully (see table).</li></ul>
<h2>What is and isn't constrained</h2>
<ul><li>There are no named head or upper-limb markers, so the arms (raised overhead here) are constrained only by the unlabelled markers and SAM 3D Body. The optimisation converges to the same loss with more iterations, so the remaining wrist error is a model limit (arm length/hand articulation), not under-optimisation.</li>
<li>The fit is per frame and not temporally smoothed; the pose prior is deliberately weak, so there is no joint-limit model and an implausible pose is possible where markers are sparse.</li></ul>
<h2>Limits and next steps</h2>
<ul><li class="warn"><b>Needs Vicon.</b> This produces ground-truth meshes for the 16&ndash;17 subjects (the training data for a supine tracker), not a deployable marker-free fit.</li>
<li>Next: fit the whole trial with temporal smoothing and one shared shape per subject; add the detected 2D balls as an extra image term; then extract the respiratory signal from the chest-array surface and compare with the Vicon chest markers.</li>
<li>Seen on the cameras but not yet used: arms/head need extra markers or a different cue.</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\sam3d-venv\\Scripts\\python.exe sam3d_vs_vicon.py          # single-view benchmark vs Vicon
C:\\dev\\sam3d-venv\\Scripts\\python.exe marker_fit.py 260 340 420 500 580 700   # fits + overlays in tmp\\fit</pre>"""
    page("07_marker_guided_mesh_fit", "7 &middot; Marker-guided SAM 3D Body mesh fit", "MMC17 TDB &middot; MHR mesh fitted to the labelled Vicon markers", body)
