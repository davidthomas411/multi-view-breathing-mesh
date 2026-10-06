"""Report 19: why is the marker-guided fit so much better than SAM 3D Body (depth vs lateral error, hold-out markers), and is the dark clothing a problem?"""
import json, os
import numpy as np


def build(ctx):
    TMP, img_b64, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    # ---------------------------------------------------------------- 1. depth vs lateral
    Z = np.load(f"{TMP}/mv_fit/err_decompose.npz", allow_pickle=True); names = [str(n) for n in Z["names"]]
    chest = [j for j, n in enumerate(names) if len(n) == 3 and n[0] in "RL" and n[1:].isdigit()]; star = [j for j, n in enumerate(names) if n.startswith("*")]; low = [j for j in range(len(names)) if j not in chest and j not in star]
    groups = [("chest array", chest), ("pelvis, legs, feet", low), ("head and arms", star)]; meshes = ["T1", "T2", "T3", "T4", "T5", "MV", "fit"]; lab = {"MV": "multi-\nview", "fit": "marker-\nguided"}
    med = lambda a: float(np.nanmedian(a))
    dep = {(g, m): med(Z["E_" + m][:, ix, 0]) for g, ix in groups for m in meshes}; lat = {(g, m): med(Z["E_" + m][:, ix, 1]) for g, ix in groups for m in meshes}; bias = {(g, m): med(Z["S_" + m][:, ix]) for g, ix in groups for m in meshes if "S_" + m in Z.files}
    f1, axs = plt.subplots(1, 2, figsize=(10.4, 3.7), sharey=False); w = 0.26; xs = np.arange(len(meshes)); colg = [C["blue"], C["orange"], C["purple"]]
    for a_, d_, ttl in ((axs[0], dep, "depth error along the viewing ray (median |.|, mm)"), (axs[1], lat, "lateral (image-plane) error (median, mm)")):
        for k, (g, _) in enumerate(groups): a_.bar(xs + (k - 1) * w, [d_[(g, m)] for m in meshes], w, color=colg[k], label=g)
        a_.set_xticks(xs); a_.set_xticklabels([lab.get(m, m) for m in meshes], fontsize=8); a_.set_title(ttl, fontsize=9)
    axs[0].legend(fontsize=8); decomp = fig_b64(f1)
    rows = [[{"MV": "multi-view (markerless)", "fit": "marker-guided"}.get(m, "single camera " + m)] + sum(([("%.0f" % dep[(g, m)]), ("%.0f" % lat[(g, m)]), ("%+.0f" % bias[(g, m)]) if (g, m) in bias else "&ndash;"] for g, _ in groups), []) for m in meshes]
    # ---------------------------------------------------------------- 2. hold-out markers
    hp = f"{TMP}/holdout/holdout.json"; hold = json.load(open(hp)) if os.path.exists(hp) else None; hold_html = ""; hold_kpi = None
    if hold:
        frames = sorted(hold["results"], key=int); f0 = frames[0]; R = hold["results"][f0]
        order = [("none", "SAM 3D Body alone (no markers)"), ("joints12", "12 joint-like markers only"), ("no_lowbody", "all but the lower body"), ("no_armshead", "all but the arms/head"), ("no_chest", "all but the chest array"), ("folds", "random 5-fold (20 % held out each)"), ("all", "all markers (training error)")]
        def held(r): return r["held_median"]
        folds = [R[f"fold{k}"] for k in range(1, 6)]; vals = []
        for key, lbl in order:
            if key == "folds": vals.append((lbl, float(np.mean([held(r) for r in folds])), float(np.mean([r["seen_median"] for r in folds])), float(np.mean([r["held_p90"] for r in folds]))))
            elif key == "all": vals.append((lbl, None, R["all"]["seen_median"], None))
            elif key == "none": vals.append((lbl, R["none"]["held_median"], None, R["none"]["held_p90"]))
            else: vals.append((lbl, held(R[key]), R[key]["seen_median"], R[key]["held_p90"]))
        f2, ax = plt.subplots(figsize=(8.6, 3.6)); x = np.arange(len(vals))
        ax.bar(x - 0.2, [v[1] if v[1] is not None else 0 for v in vals], 0.38, color=C["red"], label="markers NOT used in the fit (held out)"); ax.bar(x + 0.2, [v[2] if v[2] is not None else 0 for v in vals], 0.38, color=C["green"], label="markers used in the fit")
        for xi, v in zip(x, vals):
            if v[1] is not None: ax.text(xi - 0.2, v[1] + 1, "%.0f" % v[1], ha="center", fontsize=8)
            if v[2] is not None: ax.text(xi + 0.2, v[2] + 1, "%.0f" % v[2], ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([v[0].replace(" (", "\n(").replace(" only", "\nonly").replace("all but the ", "all but\nthe ") for v in vals], fontsize=7); ax.set_ylabel("marker to fitted surface, median (mm)"); ax.legend(fontsize=8); holdfig = fig_b64(f2)
        htab = [[v[0], "%.1f" % v[1] if v[1] is not None else "&ndash;", "%.1f" % v[3] if v[3] is not None else "&ndash;", "%.1f" % v[2] if v[2] is not None else "&ndash;"] for v in vals]
        rand = vals[5]; sam = vals[0]; allv = vals[6]; j12 = vals[1]
        hold_kpi = ("%.0f mm" % rand[1], "held-out markers, random 5-fold marker-guided fit (SAM 3D Body alone %.0f mm; the fitted markers: %.0f mm)" % (sam[1], allv[2]))
        hold_html = f"""
<h2>2 &middot; The honest number: markers the fit never saw</h2>
<p>The marker-guided distance (~16 mm, report 14) is measured at the very markers the fit pulls the mesh onto, so part of it is training error. To separate that, the fit was repeated with markers held out (frame {f0}, same start and settings, full three-stage fit each time) and the distance of the held-out markers to the fitted surface was measured.</p>
{fig(holdfig, "Median marker-to-surface distance of the held-out markers (red) and of the markers that were fitted (green), by scheme. Frame %s." % f0)}
{table(["Scheme", "held-out median (mm)", "held-out 90th percentile", "fitted-marker median (mm)"], htab, num=(1, 2, 3))}
<ul><li><b>Random 5-fold</b> (a fifth of the markers hidden each time): the hidden markers are {rand[1]:.0f} mm from the surface, against {sam[1]:.0f} mm for SAM 3D Body alone and {allv[2]:.0f} mm for markers that were used. So the fit does not just memorise its markers: a body constrained by a dense set of 3D anchors is right <i>between</i> the anchors too.</li>
<li><b>Hiding a whole region</b> (chest array, lower body, arms/head) is the harder test: the held-out region has no anchors nearby and falls back towards SAM 3D Body's shape for that region; the numbers in the table say how much of the gain is local and how much transfers.</li>
<li><b>Only 12 joint-like markers</b> (ASIS, trochanters, knees, malleoli, toes, headband): all other markers are held out at {j12[1]:.0f} mm. This is the relevant number for a markerless replacement for the markers: a sparse set of accurately triangulated joints (what 2D keypoints can supply) pins down a large part of the body already.</li></ul>"""
    # ---------------------------------------------------------------- 3. dark clothing
    dp = f"{TMP}/holdout/dark_clothing.json"; dark = json.load(open(dp)) if os.path.exists(dp) else None; dark_html = ""
    if dark:
        cams = sorted({r["cam"] for r in dark["orig"]}); reg = [("low", "pelvis, legs, feet"), ("chest", "chest array"), ("star", "head and arms")]
        def agg(cond, cam, key): v = [r[key] for r in dark[cond] if r["cam"] == cam and r[key] is not None]; return float(np.median(v)) if v else float("nan")
        f3, axs = plt.subplots(1, 3, figsize=(10.4, 3.2), sharey=False)
        for a_, (k, ttl) in zip(axs, reg):
            xs_ = np.arange(len(cams)); a_.bar(xs_ - 0.2, [agg("orig", c, k) for c in cams], 0.38, color=C["grey"], label="original frames"); a_.bar(xs_ + 0.2, [agg("enh", c, k) for c in cams], 0.38, color=C["orange"], label="brightened + CLAHE")
            a_.set_xticks(xs_); a_.set_xticklabels(["T%d" % c for c in cams]); a_.set_title(ttl, fontsize=9)
        axs[0].set_ylabel("single-camera SAM 3D Body\nmarker-to-mesh, median (mm)"); axs[0].legend(fontsize=8); darkfig = fig_b64(f3)
        low_o = np.nanmedian([agg("orig", c, "low") for c in cams]); low_e = np.nanmedian([agg("enh", c, "low") for c in cams]); ch_o = np.nanmedian([agg("orig", c, "chest") for c in cams]); ch_e = np.nanmedian([agg("enh", c, "chest") for c in cams])
        br = {c: np.median([r["brightness_low"] for r in dark["orig"] if r["cam"] == c and r["brightness_low"] is not None]) for c in cams}
        dark_html = f"""
<h2>3 &middot; Is SAM 3D Body hurt by the dark clothing on the lower limbs?</h2>
<p>Direct test: every frame was run twice, as recorded and with the lightness gamma-lifted (0.45) and contrast-equalised (CLAHE), which makes the black leggings readable to a person; the meshes were scored against the Vicon markers per region ({len(set(r['frame'] for r in dark['orig']))} frames, five cameras).</p>
{fig(darkfig, "Median marker-to-mesh distance of the single-camera SAM 3D Body mesh, as recorded (grey) and after brightening (orange), per camera and region.")}
<ul><li>Across cameras the lower-body error moves from {low_o:.0f} to {low_e:.0f} mm and the chest error (control: pale top) from {ch_o:.0f} to {ch_e:.0f} mm. <b>{'No consistent improvement on the legs: dark clothing is not what limits SAM 3D Body here.' if low_e > 0.85 * low_o else 'A clear improvement on the legs: contrast does matter.'}</b></li>
<li>Mean brightness (0&ndash;255) around the lower-body markers per camera: {', '.join('T%d %.0f' % (c, br[c]) for c in cams)}.</li>
<li>Independent of this: the Sapiens2 segmentation covers the black leggings (report 20), and the leg <i>lateral</i> error of the best camera is small (section 1): the legs are found in the image; their depth is what is wrong.</li></ul>"""
    body = f"""
<div class="card"><b>Questions.</b> (1) How is the marker-guided fit so much better than SAM 3D Body? (2) Is that real, or just the metric (the fit is built to minimise exactly that distance)? (3) Is SAM 3D Body suffering from the dark clothing on the lower limbs?</div>
{kpis([("%.0f cm" % (abs(bias[('chest array', 'T2')]) / 10), "typical depth error of a single camera's SAM 3D Body mesh on the chest (T2: mesh farther than the markers); T4 is the one camera within a centimetre"), ("%.0f mm" % dep[('chest array', 'MV')], "depth error left after the five-camera consensus (chest)"), ("%.0f mm" % dep[('chest array', 'fit')], "depth error of the marker-guided mesh (chest)")] + ([hold_kpi] if hold_kpi else []))}
<div class="card ok"><b>Answer in one paragraph.</b> SAM 3D Body is a <i>monocular</i> estimator: from one camera it places the body well in the image but not in depth (along the viewing ray), and on this supine, arms-up pose the depth error is huge (tens of centimetres in four of five cameras, with the body placed too far away). The Vicon markers are 3D points: they fix exactly the quantity SAM 3D Body cannot see. The fit is then right at the markers and, per the hold-out test below, between them too; but the headline 16 mm is partly training error. Dark clothing is not the cause: see section 3.</div>
<h2>1 &middot; Where the error is: depth, not the image plane</h2>
<p>Each Vicon marker is attached to the vertex of the marker-guided mesh nearest to it; the same vertex index on another mesh is the same anatomical place. For each camera the vector from that vertex to the marker is split into <b>depth</b> (along the camera's ray to the marker) and <b>lateral</b> (perpendicular to it, i.e. the image-plane error). Median over the 146 frames.</p>
{fig(decomp, "Depth (left) and lateral (right) error per mesh and region. Single-camera SAM 3D Body (T1&ndash;T5): depth dominates and varies enormously between cameras; the multi-view consensus and the marker-guided mesh remove it.")}
{table(["Mesh"] + sum(([g + " depth", g + " lateral", g + " signed depth"] for g, _ in groups), []), rows, num=tuple(range(1, 10)))}
<ul><li>Signed depth is positive when the mesh is <b>farther from the camera than the markers</b>. Four cameras place the person 13&ndash;31 cm too far away on the chest; camera T4 is accurate to ~1 cm. This is not a clothing effect: the chest is covered by a pale top and shows the same error as the legs.</li>
<li>Lateral errors (partly the mismatch between SAM 3D Body's body shape and the subject's, partly localisation) are 25&ndash;60 mm for the best camera and larger for the grazing views.</li>
<li>The five-camera consensus (report 16) brings the depth error down to {dep[('chest array', 'MV')]:.0f} mm (chest) but keeps a lateral error of {lat[('chest array', 'MV')]:.0f} mm: the cameras agree about the image plane only to ~3 cm.</li></ul>
{hold_html}
{dark_html}
<h2>Limits</h2>
<ul><li class="warn">One subject, one trial (TDB), one or two frames for the hold-out fits (each fit ~1 min); the 12-marker and region-hold-out results are for the first frame only.</li>
<li class="warn">The lateral/depth split uses correspondences defined by the marker-guided mesh (nearest saved vertex, median {17:.0f} mm discretisation), so lateral numbers include shape mismatch.</li>
<li>Held-out markers are scored against the full fitted surface (80,000 samples), not the 6,000 subsampled vertices used in report 16, so the numbers are not interchangeable with report 16's.</li></ul>
<h2>Reproduce</h2><pre>python err_decompose.py                      # InstantHMR env, uses tmp/mv_sam3d, mv_fit, track
python marker_holdout.py 250 700             # SAM3D env, ~1 min per fit
python dark_clothing_test.py                 # SAM3D env</pre>"""
    page("19_why_marker_guided_is_better", "19 &middot; Why the marker-guided fit is so much better (and the dark-clothing question)", "MMC17 TDB &middot; depth vs lateral error, hold-out markers, brightened frames", body)
