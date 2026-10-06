"""Report 9: marker-guided fit v2 (named arm/head markers attached to their own limb + priority weights) vs the previous fit; and
Report 11: SAM 3D Body multi-view benchmarked against Vicon in the Vicon frame."""
import glob, json, os
import numpy as np
from report_vicon import _regions


def _load(d):
    out = {}
    for p in sorted(glob.glob(f"{d}/fit_*.npz")):
        z = np.load(p, allow_pickle=True); out[int(os.path.basename(p)[4:8])] = (z["names"], z["before"], z["after"])
    return out


def build(ctx):
    TMP, img_b64, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    A = _load(f"{TMP}/fit"); B = _load(f"{TMP}/fit_v2"); common = sorted(set(A) & set(B))
    if not common:
        return
    regs = _regions(A[common[0]][0]); pool = lambda D, k: np.concatenate([D[f][k] for f in common])
    def med(D, k, ns):
        m = np.concatenate([np.isin(D[f][0], ns) for f in common]); return np.median(pool(D, k)[m]), np.percentile(pool(D, k)[m], 90), int(m.sum() // len(common))
    rows = []; labs = []; va = []; vb = []
    for r, ns in regs.items():
        if not any(n in A[common[0]][0] for n in ns): continue
        a = med(A, 2, ns); b = med(B, 2, ns); rows.append([r, a[2], "%.0f / %.0f" % a[:2], "%.0f / %.0f" % b[:2]]); labs.append(r); va.append(a[0]); vb.append(b[0])
    allA = pool(A, 2); allB = pool(B, 2); bef = pool(A, 1)
    f, ax = plt.subplots(figsize=(8.4, 3.2)); x = np.arange(len(labs))
    ax.bar(x - .27, [med(A, 1, regs[k])[0] for k in labs], .27, color=C["grey"], label="SAM 3D Body (best view)"); ax.bar(x, va, .27, color=C["orange"], label="fit v1 (unnamed arm markers)"); ax.bar(x + .27, vb, .27, color=C["green"], label="fit v2 (named + weighted)")
    ax.axhline(8, color=C["red"], ls="--", lw=1, label="8 mm (ball radius)"); ax.set_xticks(x); ax.set_xticklabels([k.replace(" / ", "/\n").replace(" (", "\n(") for k in labs], fontsize=7); ax.set_ylabel("median marker -> mesh surface (mm)"); ax.legend(fontsize=7); chart = fig_b64(f)
    per = [[f"frame {fr}", "%.0f / %.0f" % (np.median(A[fr][1]), np.percentile(A[fr][1], 90)), "%.0f / %.0f" % (np.median(A[fr][2]), np.percentile(A[fr][2], 90)), "%.0f / %.0f" % (np.median(B[fr][2]), np.percentile(B[fr][2], 90))] for fr in common]
    ws = ["*58", "*59", "*60", "*62", "*66"]
    def wr(D): return np.median(np.concatenate([[D[fr][2][list(map(str, D[fr][0])).index(k)] for k in ws if k in list(map(str, D[fr][0]))] for fr in common]))
    imgs = "".join(fig(img_b64(f"{TMP}/fit_v2/overlay_{fr:04d}.jpg", 1400), f"Frame {fr}, fit v2: cyan = Vicon markers, red = SAM 3D Body before, green = after.") for fr in common[:3])
    body = f"""
<div class="card"><b>Question.</b> Using the named head/arm markers (report 8) and the paper's marker priorities, does the marker-guided mesh fit (report 7) get better &mdash; especially at the wrists and hands, where it was worst?</div>
{kpis([("%.0f mm" % np.median(allB), "median marker-to-mesh, fit v2 (%d frames)" % len(common)), ("%.0f mm" % np.median(allA), "same frames, fit v1"), ("%.0f mm" % np.median(bef), "SAM 3D Body (best view) before fitting"), ("%.0f &rarr; %.0f mm" % (wr(A), wr(B)), "median wrist/hand marker error, v1 &rarr; v2")])}
<h2>What changed from the previous fit</h2>
<ul><li><b>Named markers:</b> the arm/head markers carry their report-8 names. Each is attached only to the surface of <i>its own side and limb</i> (head, right/left wrist-hand, right/left arm), where v1 only split head / wrist / arm.</li>
<li><b>Priority weights</b> from the paper figure: red 1.0, green 0.4, blue 0.15, so soft-tissue-prone cluster markers and rarely visible medial markers pull less.</li>
<li>Everything else is identical (3 stages, pseudo-Huber, barycentric attachment, shape/scale in the last stage, SAM 3D Body prior).</li></ul>
<h2>Results</h2>
{fig(chart, "Median distance from each Vicon marker to the fitted mesh, pooled over the common frames. The metric is unweighted, so down-weighted markers may look slightly worse even when the fit is better.")}
{table(["Region", "markers", "v1 median / p90 (mm)", "v2 median / p90 (mm)"], rows, num=(1, 2, 3))}
{table(["Frame", "SAM 3D Body median / p90 (mm)", "fit v1", "fit v2"], per, num=(1, 2, 3))}
<h2>Overlays</h2>{imgs}
<h2>Reading the result</h2>
<ul><li class="warn"><b>Result: no meaningful change.</b> Naming the arm/head markers, restricting them to their own limb and weighting by priority moves the median by about 1 mm (e.g. frame 420: 13 &rarr; 12 mm) and leaves the wrist/hand markers where they were (frame 420: 52, 74, 38, 38, 6 mm &rarr; 47, 74, 39, 38, 5 mm). The earlier fit was not limited by labelling.</li>
<li>This is a same-data comparison on the same frames and the same starting camera, so differences come only from the labelling, restriction and weights.</li>
<li>A weighted loss lowers the pull of green/blue markers, so improvements at red-priority regions are the meaningful ones.</li>
<li class="warn">The remaining wrist/hand error is a model limit (the optimisation converges to the same loss with more iterations), not a labelling problem.</li></ul>
<h2>Reproduce</h2><pre>$env:FIT_DIR = "fit_v2"
C:\\dev\\sam3d-venv\\Scripts\\python.exe marker_fit.py 260 340 420 500 580 700</pre>"""
    page("09_marker_fit_v2", "9 &middot; Marker-guided mesh fit v2: named arm/head markers + priority weights", "MMC17 TDB &middot; same frames as report 7", body)


def build_bench(ctx):
    TMP, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    J = json.load(open(f"{TMP}/sam3d_mv_vs_vicon.json")); keys = [k for k in J if k != "ALL"]
    f, ax = plt.subplots(figsize=(8.6, 3.3)); x = np.arange(len(keys)); w = .27
    for k, (tag, lab, col) in enumerate((("single", "single view", C["grey"]), ("mv5", "multi-view T1&ndash;T5".replace("&ndash;", "-"), C["blue"]), ("mv6", "multi-view T1&ndash;T6".replace("&ndash;", "-"), C["green"]))):
        ax.bar(x + (k - 1) * w, [J[n][tag][0] if J[n][tag] else 0 for n in keys], w, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([n.replace(" (", "\n(") for n in keys], fontsize=7.5); ax.set_ylabel("median error vs Vicon (mm)"); ax.legend(fontsize=8); chart = fig_b64(f)
    fm = lambda v: "%.0f / %.0f (n=%d)" % tuple(v) if v else "&ndash;"
    rows = [[n, fm(J[n]["single"]), fm(J[n]["mv5"]), fm(J[n]["mv6"])] for n in keys]
    A = J["ALL"]
    body = f"""
<div class="card"><b>Question.</b> With all cameras calibrated into the Vicon frame (reports 5 and 10), how far is SAM 3D Body from the Vicon ground truth &mdash; single view and triangulated over several views? (This is the benchmark requested at the start of the MMC17 work.)</div>
{kpis([("%.0f mm" % A["single"][0], "single view (T1&ndash;T5), median over landmarks"), ("%.0f mm" % A["mv5"][0], "multi-view triangulation T1&ndash;T5"), ("%.0f mm" % A["mv6"][0], "multi-view T1&ndash;T6"), ("&asymp; 10 mm", "what the marker-guided fit reaches at the markers (reports 7, 9)")])}
<h2>Method</h2>
<ul><li>Per camera: SAM 3D Body skeleton + <code>cam_t</code>, projected through the calibrated K; the 2D points are triangulated with the Vicon-frame extrinsics, so the fused skeleton is in Vicon coordinates (metres).</li>
<li>Compared with <b>landmark centres with a sound anatomical correspondence</b>: knee (mid of the two epicondyle markers), ankle (mid malleoli), wrist (mid RAD/ULN), elbow (mid ELB/MELB), head (mid of the eyes vs mid of the two headband markers). Pelvis/hip is left out: the hip joint centre is not where the ASIS markers are.</li>
<li>Frames from 250 on (after the coached leg tap), every 10th frame; the multi-view numbers use frames where at least three cameras see him.</li></ul>
<h2>Results</h2>
{fig(chart, "Median error of SAM 3D Body landmark centres against Vicon (mm), by landmark.")}
{table(["Landmark", "single view: median / p90 (mm)", "multi-view T1&ndash;T5", "multi-view T1&ndash;T6"], rows)}
<p class="meta">Marker-to-joint offsets of 1&ndash;3 cm are inherent to this comparison (markers sit on the skin, not at joint centres), so errors of this size are within the metric's own noise; errors of 15&ndash;24 cm are not.</p>
<h2>Takeaways</h2>
<ul><li class="bad"><b>SAM 3D Body's skeleton is far from Vicon on this supine, arms-up pose</b>: about 17 cm median even when triangulated across cameras (21 cm single view). Triangulation helps (211 &rarr; 170 mm) but does not fix it, so the error is a systematic pose error in each view, not noise that averaging removes.</li>
<li class='warn'><b>Adding T6 does not help and hurts the right arm</b> (right wrist 177 &rarr; 239 mm, right elbow 57 &rarr; 98 mm); the other landmarks are unchanged. Its 24 fps clock (up to ~1 frame of misalignment while the arm moves) or a small residual in its pose are the likely causes; the markers cannot resolve which. Until its timestamps are known, T1&ndash;T5 is the safer set for fusion.</li>
<li>This confirms why guiding the fit with the markers (reports 7 and 9) is needed, and gives the number a fine-tuned or marker-supervised model must beat.</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\sam3d-venv\\Scripts\\python.exe sam3d_mv_vs_vicon.py</pre>"""
    page("11_sam3d_multiview_vs_vicon", "11 &middot; SAM 3D Body multi-view vs Vicon (MMC17)", "benchmark in the Vicon frame &middot; knees, ankles, wrists, elbows, head", body)
