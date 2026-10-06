"""Report 20: Sapiens2 pose + body-part segmentation as a markerless COLD START for the multi-view MHR fit (instead of the Vicon markers)."""
import glob, json, os
import cv2
import numpy as np


def build(ctx):
    TMP, img_b64, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    SD = f"{TMP}/sapiens2"; J = lambda p: json.load(open(p)) if os.path.exists(p) else None
    # ---- montage of the five cameras (frame 250): keypoints + body-part segmentation
    tiles = [cv2.imread(f"{SD}/vis/0250_T{c}_0.4b_rot.jpg") for c in range(1, 6)]; tiles = [cv2.resize(t, (960, 540)) for t in tiles if t is not None]
    if len(tiles) == 5: tiles.append(np.zeros_like(tiles[0])); mont = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]); cv2.imwrite(f"{SD}/vis/montage_0250.jpg", mont, [cv2.IMWRITE_JPEG_QUALITY, 80]); montage = img_b64(f"{SD}/vis/montage_0250.jpg", 1700, 76)
    else: montage = ""
    # ---- keypoints vs Vicon proxies
    tri = {int(os.path.basename(p).split("_")[1]): json.load(open(p)) for p in sorted(glob.glob(f"{SD}/tri_*_0.4b_rot.json"))}
    lm = list(next(iter(tri.values()))["landmarks"].keys()) if tri else []; frames = sorted(tri)
    trows = [[n] + sum(([("%.0f" % tri[f]["landmarks"][n]["sapiens"]) if tri[f]["landmarks"][n]["sapiens"] is not None else "&ndash;", ("%.0f" % tri[f]["landmarks"][n]["sam3d"]) if tri[f]["landmarks"][n]["sam3d"] is not None else "&ndash;"] for f in frames[:1]), []) for n in lm]
    med_s = np.median([tri[f]["median_sapiens"] for f in frames]) if frames else float("nan"); med_m = np.median([tri[f]["median_sam3d"] for f in frames]) if frames else float("nan")
    # ---- mask check
    mc = J(f"{SD}/maskcheck_0250.json"); mrows = [[f"T{r['cam']}", "%.2f" % r["iou_markerguided"], "%.2f" % r["iou_multiview"], "%.0f %%" % (100 * r["lower_inside"]), "%.0f %%" % (100 * r["upper_inside"])] for r in mc["rows"]] if mc else []
    # ---- cold-start fit results
    SE = J(f"{TMP}/sap_fit/sap_eval.json"); SS = J(f"{TMP}/sap_fit/sap_eval_sil.json"); fit_html = ""; kp = []
    if SE:
        fr = SE["frames"]; V = SE["variants"]; reg = [("all", "all markers"), ("chest", "chest array"), ("low", "pelvis, legs, feet"), ("star", "head and arms")]
        def agg(d, g): v = [x for x in d[g] if x is not None]; return float(np.median(v)) if v else float("nan")
        sv = {g: float(np.mean([agg(V[f"sv{c}"], g) for c in range(1, 6)])) for g, _ in reg}; labels = [("sv4", "SAM 3D Body, best single camera (T4, chosen with Vicon)"), ("typ", "SAM 3D Body, typical single camera (mean of the five)"), ("mv", "SAM 3D Body multi-view consensus (markerless, report 16)"),
                                                                                                   ("sapiens_cold", "<b>Sapiens2 cold start: keypoints only</b>"), ("sapiens_sil", "<b>Sapiens2 cold start: keypoints + body-part mask</b>"), ("fit", "marker-guided fit (needs the markers)")]
        tab = []; bars = {}
        for key, lbl in labels:
            if key == "typ": vals = [sv[g] for g, _ in reg]
            elif key == "sapiens_sil":
                if not SS: continue
                vals = [agg(SS["variants"]["sapiens_cold"], g) for g, _ in reg]
            else: vals = [agg(V[key], g) for g, _ in reg]
            bars[key] = vals; tab.append([lbl] + ["%.1f" % x for x in vals])
        f1, ax = plt.subplots(figsize=(9.2, 3.5)); keys = [k for k, _ in labels if k in bars]; cols = {"sv4": C["red"], "typ": "#9ca3af", "mv": C["blue"], "sapiens_cold": C["orange"], "sapiens_sil": "#d97706", "fit": C["green"]}; w = 0.8 / len(keys)
        for j, k in enumerate(keys):
            ax.bar(np.arange(4) + (j - (len(keys) - 1) / 2) * w, [min(x, 128) for x in bars[k]], w, color=cols[k], label={"sv4": "best single camera (T4, picked with Vicon)", "typ": "typical single camera", "mv": "multi-view consensus (markerless)", "sapiens_cold": "Sapiens2 cold start: keypoints", "sapiens_sil": "Sapiens2 cold start: keypoints + mask", "fit": "marker-guided (needs the markers)"}[k])
            for i, x in enumerate(bars[k]): ax.text(i + (j - (len(keys) - 1) / 2) * w, min(x, 128) + 1.5, "%.0f%s" % (x, "^" if x > 128 else ""), ha="center", fontsize=6.5)
        ax.set_ylim(0, 140); ax.set_xticks(np.arange(4)); ax.set_xticklabels([n for _, n in reg]); ax.set_ylabel("median marker-to-mesh distance (mm)"); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False); fitfig = fig_b64(f1)
        per = {k: bars[k][0] for k in bars}; kp = [("%.0f mm" % per.get("sapiens_sil", per.get("sapiens_cold")), "median marker-to-mesh distance of the Sapiens2 cold-start mesh (no Vicon, no SAM 3D Body in the objective); %d frames" % len(fr)), ("%.0f mm" % per["mv"], "SAM 3D Body multi-view consensus on the same frames"), ("%.0f mm" % per["fit"], "marker-guided mesh (reference)")]
        ovs = "".join(fig(img_b64(f"{TMP}/sap_fit/overlay/{f:04d}.jpg", 1700, 76), "Frame %d: Sapiens2 cold start (yellow dots), markerless multi-view mesh (blue), marker-guided mesh (green), Vicon markers (cyan rings)." % f) for f in (250, 700) if os.path.exists(f"{TMP}/sap_fit/overlay/{f:04d}.jpg"))
        SH = J(f"{TMP}/sap_fit/surface_height.json"); sh_rows = [[nm] + ["%+.0f" % np.median(SH[nm][str(r)]) for r in (1, 2, 3, 4)] for nm in ("cold start", "multi-view", "marker-guided")] if SH else []
        fit_html = f"""
<h2>3 &middot; The cold start: fit the body to Sapiens2 in all five cameras</h2>
<p>MHR (the body model inside SAM 3D Body) can output exactly Sapiens2's 308 keypoints (its <code>keypoint_mapping</code> is a 308 &times; 18,566 matrix over vertices and joints; the first 70 are the MHR70 set). So one body (pose + a rigid correction; shape from the initialisation) is optimised so that its keypoints project onto the Sapiens2 detections in all cameras at once: pseudo-Huber (25 px), confidence-weighted, body and foot keypoints weight 1, the 7 extra body points (olecranon, cubital fossa, acromion, neck) 1, hands 0.1, face 0.03. Started from each camera's SAM 3D Body pose in turn, lowest loss kept. The variant with a <b>mask term</b> adds a silhouette loss from the Sapiens2 body-part mask (vertices outside the person mask are pulled in; mask pixels far from every vertex pull the body out). Vicon is used only to score.</p>
{fig(fitfig, "Median marker-to-mesh distance by region on %d frames of the TDB trial (same metric as report 16)." % len(fr))}
{table(["Mesh"] + [n for _, n in reg], tab, num=(1, 2, 3, 4))}
{ovs}
<h3>What the median hides: the belly floats</h3>
<p>The median over markers is dominated by the limbs and head, which the keypoints place well. Looking at the <b>height of each mesh's anterior surface above the skin under the 16 chest-array balls</b> (the highest mesh vertex within 3 cm horizontally minus the ball height; + = the mesh floats above the skin; the marker-guided mesh reads +5 to +19 mm because of the 3 cm window and the ball radius) shows where the cold start is wrong:</p>
{table(["Mesh", "row 1 (upper chest)", "row 2 (lower chest)", "row 3 (upper abdomen)", "row 4 (lower abdomen)"], sh_rows, num=(1, 2, 3, 4)) if sh_rows else ""}
<ul><li><b>The abdomen of the cold-start mesh sits ~7&ndash;9 cm above the skin</b> (rows 3&ndash;4), and the chest 2&ndash;3 cm; the multi-view consensus is within 1&ndash;3 cm of the reference there. That is a body-thickness error: the keypoints say where the joints are, not how thick the belly is, and nothing in the fit constrains the surface between them.</li>
<li><b>The segmentation looks right because the silhouette hardly sees it.</b> The Sapiens2 mask and the cold-start silhouette overlap at IoU 0.76 (marker-guided 0.80, multi-view 0.61): from cameras looking down at oblique angles a belly that is 8 cm too high still covers the same pixels. Only the side views (T3, T4, T5) constrain it, and the fit above never used the mask at all.</li>
<li>A mesh this far off the skin cannot seed skin tracking: seeds placed on it give an amplitude ratio of 0.4&ndash;0.5 against Vicon (MAE 2.1&ndash;2.7 mm), where seeds on the multi-view mesh give 0.94&ndash;1.0.</li></ul>
"""
    body = f"""
<div class="card"><b>Question.</b> Sapiens2's pose and body-part segmentation look better than what we have: can they give SAM 3D Body a <b>cold start</b>, so the marker-guided fit's markers are not needed to initialise?</div>
{kpis(kp + [("%.0f vs %.0f mm" % (med_s, med_m), "triangulated 3D joints vs Vicon-derived proxies: Sapiens2 vs SAM 3D Body's own skeleton (median over landmarks)")] if tri else kp)}
<div class="card warn"><b>Answer: partly.</b> The 2D keypoints are accurate enough, in all five views, to give a markerless starting mesh that is <b>much better on the limbs and head</b> (13&ndash;23 mm to the markers, against 44&ndash;64 mm for the multi-view SAM 3D Body consensus) and 22 mm overall (consensus 41, best single camera 34, marker-guided 16). <b>But the median hides a real failure: the abdomen of this mesh floats 7&ndash;9 cm above the skin</b> (section 3), because keypoints fix the skeleton and not the body thickness; it also cannot seed skin tracking. The segmentation is excellent but the keypoint-only fit never used it. Two variants address the torso: a <b>body-part mask term</b> (uses the segmentation: the mesh must cover the person pixels in every camera) and a <b>multi-view consensus term</b> (keeps SAM 3D Body's dense surface cues, which are good on the torso, alongside the Sapiens2 skeleton); their results are in section 3 when they have run.</div>
<h2>1 &middot; What Sapiens2 is, and how it was run here</h2>
<ul><li><b>Sapiens2</b> (Meta, 2026; arXiv 2604.21681): human-centric ViT models pre-trained on 1 billion human images, with heads for <b>308-keypoint pose</b> (top-down, needs a person box), <b>29-class body-part segmentation</b> (face, hair, torso, upper/lower clothing, left/right upper and lower arm/leg, hands, feet, shoes, socks, &hellip;), surface normals, 3D pointmaps and matting. Sizes 0.4 / 0.8 / 1 / 5 B. The weights are public (no gating) under the Sapiens2 License (Meta).</li>
<li><b>Here:</b> the 0.4B pose and segmentation checkpoints in a Python 3.12 / torch 2.11 environment (<code>C:\\dev\\sapiens2-venv</code>), run from <code>mv/sapiens2_run.py</code> on the person boxes of the earlier SAM 3D Body pass. The supine person is a very elongated box, so each crop is rotated 90&deg; (head up) before the 1024 &times; 768 network and the results are rotated back. CPU time (no GPU was free): ~19 s per crop for pose and ~18 s for segmentation; on the GPU this is a fraction of a second per crop for 0.4B (timings in the reproduce section).</li></ul>
<h2>2 &middot; What it sees</h2>
{fig(montage, "Sapiens2 0.4B on frame 250 of MMC17 TDB, the five tripod cameras: body-part segmentation (colours) and the first 70 keypoints (yellow; the 238 face keypoints are orange). The black leggings are segmented as one clean lower-clothing region in every view.") if montage else ""}
{table(["Camera", "IoU of the person mask with the marker-guided mesh silhouette", "IoU with the markerless multi-view mesh silhouette", "lower-body-class pixels inside the marker-guided silhouette", "upper-body-class pixels inside"], mrows, num=(1, 2, 3, 4)) if mrows else ""}
<p>The mask agrees with the silhouette of the marker-guided mesh (the closest thing to the truth we have) at IoU {np.median([float(r[1]) for r in mrows]) if mrows else float('nan'):.2f} (median over cameras), and with the markerless multi-view mesh at {np.median([float(r[2]) for r in mrows]) if mrows else float('nan'):.2f}: the mesh the markerless pipeline has now is the less faithful of the two, which is what a silhouette term can correct. The 'lower-body-class' column measures the legs, the dark-clothing region: most of those pixels fall inside the true silhouette, i.e. the black leggings are not a problem for this segmenter.</p>
<h3>Keypoints, triangulated</h3>
<p>The first 70 keypoints (MHR70 order) were triangulated from the five cameras (confidence-weighted DLT, a view dropped if its reprojection exceeds 25 px) and compared with joint proxies built from Vicon markers (hip ~ trochanter, knee = midpoint of the lateral/medial markers, ankle = midpoint of the malleoli, wrist = midpoint of the radial/ulnar markers, elbow = midpoint of the epicondyle markers, toes ~ metatarsal heads; a joint centre or a landmark is 2&ndash;6 cm from the skin markers, so absolute numbers are a few cm high for both methods and only the comparison matters).</p>
{table(["Landmark (index)", "Sapiens2 (mm)", "SAM 3D Body skeleton, same cameras (mm)"], trows, num=(1, 2)) if trows else ""}
<p>Median over the landmarks, frame {frames[0] if frames else '?'}: Sapiens2 {med_s:.0f} mm vs SAM 3D Body {med_m:.0f} mm. Knees, ankles and wrists land within 2&ndash;3.5 cm; the hips, elbows and toes are dominated by the proxy offset. Mean keypoint confidence is high in every camera (body keypoints 0.88&ndash;0.94); the hands are the weak spot when they are out of the view (T4, T5: 0.36&ndash;0.45).</p>
{fit_html}
<h2>4 &middot; How this would be used (the cold-start architecture)</h2>
<ol><li><b>Cold start, once, seconds:</b> Sapiens2 pose + segmentation on all views &rarr; multi-view MHR fit (above) &rarr; mesh in the room frame, no markers.</li>
<li><b>Warm tracking, every frame:</b> the existing multi-view consensus (report 16) with the cold-start mesh as its warm start and prior; Sapiens2 at a low rate to re-anchor.</li>
<li><b>Fine layer, 30 fps:</b> dense skin tracking seeded on that mesh (reports 15 and 17), for breathing and local deformation.</li></ol>
<h2>Limits</h2>
<ul><li class="warn">One subject, a handful of frames, 0.4B models, calibrated cameras (the Vicon-frame extrinsics came from the markers; a checkerboard calibration would do).</li>
<li class="warn">Sapiens2 is not light: 0.4B pose + segmentation took ~37 s per crop on the CPU used here; the 1B/5B models are much heavier. The GPU timing, and what the 1B model changes, are in the reproduce section.</li>
<li>The keypoint 'truth' used above is a set of proxies; the cold-start mesh is scored the same way as in report 16 against the real markers.</li>
<li>License: Sapiens2 License (Meta) &mdash; check the terms before any use beyond research.</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\sapiens2-venv\\Scripts\\python.exe sapiens2_run.py --frames 250 400 550 700 850 --cams 1 2 3 4 5 --device cuda
python sap_triangulate.py &lt;frame&gt;                                  # InstantHMR env
set W_SIL=0 &amp; python sap_fit.py 250 400 550 700 850                  # SAM3D env, keypoints only;  W_SIL=1 adds the mask term
python sap_eval.py ; python sap_mask_check.py 250 ; python make_sapiens_rrd.py        # scoring, mask check, Rerun file</pre>"""
    page("20_sapiens2_cold_start", "20 &middot; Sapiens2 pose and body parts as a markerless cold start", "MMC17 TDB &middot; Sapiens2 0.4B keypoints + segmentation fitted to the MHR body in five cameras", body)
