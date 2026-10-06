"""Build the HTML experiment reports -> reports/*.html  (self-contained: figures are embedded as base64).
Run with the InstantHMR env:  C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe make_reports.py
Numbers that were measured by one-off commands during the session are written into the pages as constants (marked 'measured');
everything that has a data file (sequences, calibration, registration) is read from tmp/ so re-running refreshes the figures."""
import base64, io, json, os, datetime
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP, OUTDIR = f"{ROOT}/tmp", f"{ROOT}/reports"
os.makedirs(OUTDIR, exist_ok=True)
TODAY = datetime.date.today().isoformat()
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "axes.grid": True, "grid.alpha": .25})
C = dict(blue="#2563eb", orange="#ea580c", green="#16a34a", red="#dc2626", grey="#6b7280", purple="#7c3aed")

_FIGN = {"cur": "", "n": 0}
def fig_b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    d = os.environ.get("SAVE_FIGS")                      # optional: also keep every figure as a PNG (numeric plots only), e.g. for the README
    if d and _FIGN["cur"]: os.makedirs(d, exist_ok=True); _FIGN["n"] += 1; open(f"{d}/{_FIGN['cur']}_{_FIGN['n']:02d}.png", "wb").write(buf.getvalue())
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def img_b64(path, width=1500, q=82):
    im = cv2.imread(path)
    if im is None: return ""
    s = width / im.shape[1]; im = cv2.resize(im, None, fx=min(s, 1), fy=min(s, 1), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q]); return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--mut:#6b7280;--card:#fff;--line:#e5e7eb;--acc:#2563eb;--good:#16a34a;--bad:#dc2626;--warn:#d97706}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e7e5e4;--mut:#9ca3af;--card:#171a21;--line:#2a2f3a;--acc:#60a5fa}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 18px 80px}h1{font-size:26px;margin:.2em 0}h2{font-size:19px;margin:1.8em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:15px;margin:1.2em 0 .3em}.meta{color:var(--mut);font-size:13px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:12px 0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--mut);font-size:12px}table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--mut);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
img{max-width:100%;border-radius:8px;border:1px solid var(--line)}figure{margin:12px 0}figcaption{color:var(--mut);font-size:12.5px;margin-top:4px}
code,pre{font:12.5px ui-monospace,Consolas,monospace;background:var(--card);border:1px solid var(--line);border-radius:6px}code{padding:1px 5px}pre{padding:10px;overflow:auto}
.tag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid var(--line);color:var(--mut)}.ok{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
a{color:var(--acc)}nav{font-size:13px;margin-bottom:6px}.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:700px){.two{grid-template-columns:1fr}}
"""

def page(slug, title, subtitle, body):
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style></head><body><main><nav><a href="index.html">&larr; all experiments</a></nav>
<h1>{title}</h1><div class="meta">{subtitle} &middot; generated {TODAY}</div>{body}</main></body></html>"""
    open(f"{OUTDIR}/{slug}.html", "w", encoding="utf-8").write(html)

def kpis(items): return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{a}</b><span>{b}</span></div>' for a, b in items) + "</div>"
def table(head, rows, num=()):
    h = "".join(f'<th class="{"n" if i in num else ""}">{x}</th>' for i, x in enumerate(head))
    r = "".join("<tr>" + "".join(f'<td class="{"n" if i in num else ""}">{x}</td>' for i, x in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><tr>{h}</tr>{r}</table>"
def fig(src, cap): return f'<figure><img src="{src}"><figcaption>{cap}</figcaption></figure>'


def scrub(prefix, caption, unit_label):
    """Frame scrubber over pre-rendered 6-camera overlay mosaics (tmp/overlays/<prefix>_<frame>.jpg + <prefix>_meta.json)."""
    meta = json.load(open(f"{TMP}/overlays/{prefix}_meta.json"))
    imgs = [img_b64(f"{TMP}/overlays/{prefix}_{m['frame']:04d}.jpg", 1500, 68) for m in meta]
    info = [f"frame {m['frame']} &middot; {m['views']} views fused" + (f" &middot; reprojection {m['reproj']:.1f} px" if m.get("reproj") is not None else " &middot; (too few views to triangulate)") for m in meta]
    uid = prefix
    return f"""<figure><img id="{uid}_img" src="{imgs[0]}" style="width:100%"><div style="display:flex;gap:10px;align-items:center;margin-top:6px">
<input id="{uid}_rng" type="range" min="0" max="{len(meta) - 1}" value="0" style="flex:1"><span id="{uid}_lab" class="meta" style="min-width:300px;text-align:right">{info[0]}</span></div>
<figcaption><span style="color:#4ade80">&#9632;</span> triangulated skeleton reprojected into each camera &nbsp; <span style="color:#fbbf24">&#9679;</span> that camera's own SAM 3D Body joints &nbsp; <span style="color:#ef4444">&#9679;</span> own joints more than {unit_label} from the fused skeleton. Cameras marked "(not fused)" are not used in the triangulation, so the skeleton shown there is a prediction, not a detection. {caption}</figcaption></figure>
<script>(function(){{var I={json.dumps(imgs)},L={json.dumps(info)},r=document.getElementById("{uid}_rng"),m=document.getElementById("{uid}_img"),l=document.getElementById("{uid}_lab");
r.addEventListener("input",function(){{m.src=I[r.value];l.innerHTML=L[r.value]}});}})();</script>"""

# ============================================================== R1: InstantHMR multi-view, vault2 =========================
def r1():
    tr, r0, de = (np.load(f"{TMP}/{n}.npz") for n in ("seq_track", "seq_rot0", "seq_detect"))
    def med(T, c): return np.median(T[:, c]), np.percentile(T[:, c], 90)
    # latency stacked bars
    f, ax = plt.subplots(figsize=(7.2, 2.8)); labs = ["detector every frame\n(1080p, 71 frames)", "tracking mode\n(1080p, 706 frames)"]
    for k, (T, name) in enumerate(((de["T"], "detect"), (tr["T"], "track"))):
        b = np.median(T[:, 0]); h = np.median(T[:, 2]); u = np.median(T[:, 3])
        ax.barh(k, b, color=C["orange"], label="boxes (detector / reprojection)" if k == 0 else None); ax.barh(k, h, left=b, color=C["blue"], label="InstantHMR (batch of 6)" if k == 0 else None)
        ax.barh(k, u, left=b + h, color=C["green"], label="triangulation" if k == 0 else None); ax.text(b + h + u + 3, k, f"{np.median(T[:, 4]):.0f} ms", va="center")
    ax.axvline(100, color=C["red"], ls="--", lw=1); ax.text(101, 1.45, "100 ms budget", color=C["red"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labs); ax.set_xlabel("median ms per synchronized 6-camera frame"); ax.legend(fontsize=8, loc="lower right"); lat = fig_b64(f)
    # accuracy over time
    f, ax = plt.subplots(figsize=(8, 3))
    def sm(x, w=15): x = np.asarray(x, float); k = np.ones(w) / w; return np.convolve(np.nan_to_num(x, nan=np.nanmedian(x)), k, "same")
    ax.plot(tr["idx"], sm(tr["loo"]), color=C["blue"], label="tracking, crops rotated to body axis"); ax.plot(r0["idx"], sm(r0["loo"]), color=C["grey"], label="tracking, upright crops")
    ax.plot(de["idx"], de["loo"], "o", ms=3, color=C["orange"], label="detector each frame (every 10th frame)")
    ax.set_xlabel("frame"); ax.set_ylabel("leave-one-view-out error (px @1080p)"); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.axvspan(560, 706, color=C["red"], alpha=.07); ax.text(565, ax.get_ylim()[0] * 1.2, "supine", color=C["red"], fontsize=8); acc = fig_b64(f)
    body = f"""
<div class="card"><b>Question.</b> Can a lightweight single-view body-mesh model (InstantHMR, ONNX) be run locally on the 6 calibrated vault cameras, fused into one world-frame skeleton, within ~100 ms per synchronized frame on an RTX 4060 (8 GB)?</div>
{kpis([("33&ndash;47 ms", "tracking mode, 6 views, 480p rrd frames (median / p90)"), ("47 / 89 ms", "tracking mode on clean 1080p video"), ("226 ms", "detector on every view, every frame (1080p)"), ("19 ms", "one batch-6 InstantHMR call")])}
<h2>Method</h2>
<ul><li><b>Pipeline.</b> per-view person box &rarr; 224&times;224 crop &rarr; <i>one</i> ONNX call on a batch of 6 views &rarr; 70 2D joints per view &rarr; robust weighted-DLT triangulation (views whose reprojection error exceeds 12 px are dropped per joint) in the calibrated world frame.</li>
<li><b>Tracking mode</b> removes the detector: the next frame's boxes come from reprojecting the previous 3D skeleton into each view (4 % margin). A lost view retries RF-DETR on one view every 3rd frame (round-robin) so a re-acquire never stalls a frame.</li>
<li><b>Calibration</b> from <code>scene.rrd</code> (verified against the rrd's own keypoints, 4 px) and later from <code>intri.yml/extri.yml</code> with lens distortion (same cameras).</li>
<li><b>Accuracy has no ground truth here</b>, so two proxies are used: <i>reprojection</i> (2D predictions vs re-projected fused skeleton) and <i>leave-one-view-out</i> (triangulate from 5 views, project into the 6th).</li></ul>
<h2>Results</h2>
{fig(lat, "Median latency breakdown. Detection (RF-DETR run six times, unoptimised) dominates; the batched model call is cheap.")}
{table(["Mode / data", "boxes ms", "model ms", "fuse ms", "total ms (median / p90)", "reproj px", "leave-1-out px"],
 [["detect, rrd 480p", "139", "19", "7", "168 / 194", "2.9", "5.3"], ["track, rrd 480p", "0.2", "21", "5", "33 / 47", "2.6", "5.6"],
  ["detect, video 1080p", "%.0f" % med(de["T"], 0)[0], "%.0f" % med(de["T"], 2)[0], "%.0f" % med(de["T"], 3)[0], "%.0f / %.0f" % med(de["T"], 4), "9.0", "27.1"],
  ["track, video 1080p (rotated crops)", "%.1f" % med(tr["T"], 0)[0], "%.0f" % med(tr["T"], 2)[0], "%.0f" % med(tr["T"], 3)[0], "%.0f / %.0f" % med(tr["T"], 4), "8.4", "27.8"],
  ["track, video 1080p (upright crops)", "%.1f" % med(r0["T"], 0)[0], "%.0f" % med(r0["T"], 2)[0], "%.0f" % med(r0["T"], 3)[0], "%.0f / %.0f" % med(r0["T"], 4), "8.4", "25.6"]], num=(1, 2, 3, 4, 5, 6))}
{fig(acc, "Cross-view consistency over the sequence. Error rises sharply once he is supine (shaded) &mdash; standing frames are ~5&times; better. The flat stretch (frames ~207&ndash;360) is an artifact: only 2 views see him there, so no leave-one-out can be computed.")}
<h3>What mattered</h3>
{table(["Experiment (rrd 480p)", "reproj px", "leave-1-out px", "total ms"], [["track, box margin 25 %", "6.6", "21.7", "82"], ["track, box margin 10 %", "4.1", "8.6", "74"], ["track, box margin 4 %", "2.8", "5.8", "61"], ["supine segment 560&ndash;705, upright crops", "7.0", "12.8", "35"], ["supine segment 560&ndash;705, rotated crops", "6.3", "10.8", "42"]], num=(1, 2, 3))}
<ul><li>The model expects the same tight box the detector gives; a loose reprojected box (25 %) shrinks the person in the crop and costs ~4&times; accuracy.</li>
<li>Rotating each crop so the neck&rarr;hip axis is vertical helps a little on supine frames (12.8 &rarr; 10.8 px) but not over the whole sequence.</li></ul>
<h3>Single view vs fused (measured)</h3>
{table(["Check", "Result"], [["root-relative 3D pose, single view vs fused skeleton", "138 mm median, 331 mm p90 (461 view-frames)"],
 ["distance to camera from the model's <code>cam_trans</code>", "111 % wrong raw; 6 % (17 cm) after correcting for the real focal length. InstantHMR assumes focal = image diagonal; these wide-angle cameras are much shorter."],
 ["person distance (triangulated)", "2.59 m median"]])}
<p class="meta">The fused skeleton is not ground truth &mdash; this shows the views disagree, not which is right.</p>
{fig(img_b64(f"{TMP}/overlay_f705.jpg"), "Last synchronized frame (supine), detector mode. Yellow: box; red: raw per-view 2D joints; green: fused skeleton reprojected. Per-view joints are plausible in cams 02/05/06 but left/right and head/foot disagree between views, so fusion fails.")}
<h2>Takeaways</h2>
<ul><li class="ok">Latency target met: removing the detector via 3D-guided tracking takes 6-view fusion from ~170&ndash;230 ms to ~35&ndash;50 ms with unchanged consistency.</li>
<li class="bad">Supine pose is the weak point: an upright-trained single-view model gives inconsistent 2D joints across views.</li>
<li class="warn">Live viewer: <code>live_viewer.py --loop</code> (Rerun) shows the feeds, fits and these latency terms in real time.</li></ul>
<h2>Reproduce</h2><pre>cd 3dbody\\mv
$py = "C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe"
&amp; $py run_seq.py 0 706 1 track        # sequence metrics
&amp; $py single_vs_multi.py              # single vs fused
&amp; $py live_viewer.py --loop           # live Rerun viewer</pre>
<div class="meta">Environment: Python 3.12, torch 2.6+cu124, onnxruntime-gpu 1.22 (1.23+ needs CUDA 13), rfdetr 1.11, rerun 0.38; <code>import torch</code> before onnxruntime so both share torch's CUDA 12 DLLs.</div>"""
    page("01_instanthmr_multiview", "1 &middot; InstantHMR multi-view on the vault rig", "vault2 (6 calibrated cameras, 706 frames) &middot; RTX 4060", body)

# ============================================================== R2: synchronization ==================================================
def r2():
    body = f"""
<div class="card"><b>Question.</b> The 1080p stills named <code>*_frame50.jpg</code> disagree between cameras. Are the original videos synchronized, and is timing why multi-view error is high on clean 1080p?</div>
{kpis([("Not synced", "the six frame-50 stills show different moments"), ("24 &rarr; 12 px", "median reprojection with best global offsets (&plusmn;2 frames)"), ("Inconclusive", "per-segment offsets are not stable")])}
<h2>Findings</h2>
<ul><li>In the stills, cam 02 shows him still walking while cams 04&ndash;06 show him already seated &mdash; they cannot be fused. The <b>last frames</b> are synchronized (you confirmed; all six show the same supine pose).</li>
<li>Global search (per-view 2D joints are independent of fusion, so they were computed once and offsets searched offline): offsets <code>[0, -1, -2, +1, +1, +1]</code> frames reduce median reprojection error from 24.0 to 12.2 px.</li>
<li>Per-segment search is inconsistent, so there is no clean linear drift:</li></ul>
{table(["Segment (frames)", "best offsets for cams 1&ndash;6", "error before &rarr; after (px @1080p)"], [["60&ndash;200", "0, &minus;1, 0, +2, +2, +1", "10.1 &rarr; 8.2"], ["380&ndash;500", "0, &minus;1, &minus;5, 0, 0, 0", "14.6 &rarr; 10.2"], ["600&ndash;700 (supine, almost still)", "0, &minus;3, &minus;4, +8, &minus;4, 0", "36.8 &rarr; 34.8 (noise)"]], num=(2,))}
<h2>Interpretation</h2>
<ul><li>Timing is worth up to ~2 frames of correction in moving segments, but the supine segment (where error is largest) is almost static, so <b>sync cannot explain it</b> &mdash; that error is the pose model.</li>
<li>The offsets were not applied in the later experiments. If you have the true timecode offsets, they should replace this estimate.</li></ul>
<h2>Reproduce</h2><pre>&amp; $py sync_search.py     # global offsets (caches per-view 2D joints in tmp\\j2d_all.npz)
&amp; $py sync_drift.py      # per-segment offsets</pre>"""
    page("02_synchronization", "2 &middot; Are the vault videos time-synchronized?", "vault2 &middot; offset search from per-view 2D joints", body)

# ============================================================== R3: SAM3D vs InstantHMR ==================================================
def r3():
    s3 = np.load(f"{TMP}/sam3d_seq_summary.npz"); de = np.load(f"{TMP}/seq_detect.npz")
    f, ax = plt.subplots(1, 2, figsize=(9, 3))
    for a, key, lab in ((ax[0], "loo", "leave-one-view-out error (px @1080p)"), (ax[1], "rep", "reprojection error (px @1080p)")):
        a.plot(de["idx"], de["loo" if key == "loo" else "reproj"], "o-", ms=3, lw=1, color=C["orange"], label="InstantHMR"); a.plot(de["idx"][:len(s3[key])], s3[key], "s-", ms=3, lw=1, color=C["blue"], label="SAM 3D Body")
        a.set_xlabel("frame"); a.set_ylabel(lab); a.set_yscale("log"); a.legend(fontsize=8)
    cmp = fig_b64(f)
    f, ax = plt.subplots(figsize=(6.5, 2.8)); fr = [100, 300, 500, 705]; a2 = [9.1, 15.8, 10.3, 21.2]; a3 = [7.3, 1.4, 7.4, 9.0]; x = np.arange(4)
    ax.bar(x - .2, a2, .4, color=C["orange"], label="its own pred_keypoints_2d"); ax.bar(x + .2, a3, .4, color=C["blue"], label="3D skeleton + cam_t projected with calibrated K")
    ax.set_xticks(x); ax.set_xticklabels([f"frame {k}" for k in fr]); ax.set_ylabel("% of body height from YOLO-Pose joints"); ax.legend(fontsize=8); chk = fig_b64(f)
    body = f"""
<div class="card"><b>Question.</b> You observed that SAM 3D Body (Fast-SAM-3D-Body checkpoint) looks much better than InstantHMR monocularly. Does it give better <i>multi-view</i> consistency on the vault rig? (Latency ignored.)</div>
{kpis([("10.6 px", "SAM 3D Body reprojection (71 frames)"), ("27.2 px", "SAM 3D Body leave-one-view-out"), ("9.0 / 27.1 px", "InstantHMR, same frames"), ("~0.9 s", "per view, warm (4060, no compile/TensorRT)")])}
<h2>Setup</h2>
<ul><li>Separate env <code>C:\\dev\\sam3d-venv</code> (Python 3.11, torch 2.5.1+cu124); detectron2/TensorRT skipped. Weights <code>facebook/sam-3d-body-dinov3</code> (2.1 GB) + MHR (0.7 GB) &mdash; your HF login already has access.</li>
<li>Per view: YOLO11m-Pose box &rarr; <code>process_one_image(img, bboxes, cam_int=K)</code> using the <i>calibrated</i> intrinsics (the model then reports <code>focal_length</code> = K<sub>x</sub>, so its depth is metric for these cameras) &rarr; 70 keypoints + 18 439-vertex MHR mesh &rarr; same robust triangulation as InstantHMR.</li>
<li>The repo's own multi-view code is a single-camera-relative SMPL/teleop pipeline, so it was not used.</li></ul>
<h2>Results</h2>
{fig(cmp, "Per-frame cross-view error, SAM 3D Body vs InstantHMR (detector mode), every 10th frame. Both rise when he is supine.")}
{table(["Model", "views / frame", "median reprojection px", "median leave-1-out px"], [["InstantHMR (detector)", "4.65", "9.0", "27.1"], ["SAM 3D Body (3D+cam_t projected with K)", "4.70", "10.6", "27.2"], ["SAM 3D Body (its <code>pred_keypoints_2d</code>)", "4.70", "27.3", "43.7"]], num=(1, 2, 3))}
<h3>Which SAM 3D Body output to use</h3>
{fig(chk, "Distance to YOLO-Pose body joints (nose, shoulders, hips, knees, ankles) as % of person height, median over the 6 views (measured). The projected 3D skeleton is closer than the model's own 2D output, so it was used for all fusion. Caveat: YOLO-Pose is itself poor on lying people.")}
{fig(img_b64(f"{TMP}/sam3d_overlay_f100.jpg"), "Frame 100 (standing). Red: raw pred_keypoints_2d. Green: 3D skeleton + cam_t projected with K. Many of the 70 points are finger points that cluster at the hands.")}
<h2>Takeaways</h2>
<ul><li class="warn"><b>On this metric SAM 3D Body is not better than InstantHMR.</b> Both are limited by something they share (supine pose, ~1&ndash;2 frames of residual sync error, calibration) rather than by the model.</li>
<li>The metric measures <i>agreement between views</i>, not mesh quality. A fair monocular-quality comparison needs ground truth &mdash; the MMC17 Vicon markers provide it (see reports 4&ndash;5).</li>
<li>SAM 3D Body costs ~5 s per 6-view frame here; it is a candidate for offline benchmarking and pseudo-labelling rather than the real-time path.</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\sam3d-venv\\Scripts\\python.exe sam3d_mv.py 0 706 10     # caches per-view results in tmp\\sam3d_cache
C:\\dev\\sam3d-venv\\Scripts\\python.exe sam3d_check.py          # which output to trust</pre>"""
    page("03_sam3d_vs_instanthmr", "3 &middot; SAM 3D Body multi-view vs InstantHMR", "vault2 &middot; calibrated triangulation of per-view predictions", body)

# ============================================================== R4: MMC17 calibration ==================================================
def r4():
    v2 = json.load(open(f"{TMP}/mmc17_calib.json")); v1 = json.load(open(f"{TMP}/mmc17_calib_v1_wrong.json")); SQ = 120.0
    f, ax = plt.subplots(figsize=(6.2, 5.2))
    for cal, col, lab, al in ((v1, C["grey"], "first attempt (wrong alignment)", .6), (v2, C["blue"], "final", 1)):
        for i in range(1, 7):
            c = np.array(cal["cams"]["T%d" % i]["centre"]) * SQ / 1000; R = np.array(cal["cams"]["T%d" % i]["R"]); ax_dir = R[2]
            ax.plot(c[0], c[1], "o", color=col, alpha=al, label=lab if i == 1 else None)
            if col == C["blue"]: ax.annotate("T%d" % i, c[:2], textcoords="offset points", xytext=(5, 5)); ax.arrow(c[0], c[1], ax_dir[0] * .35, ax_dir[1] * .35, color=col, head_width=.05)
    ax.add_patch(plt.Rectangle((-1 * SQ / 1000, -1 * SQ / 1000), 8 * SQ / 1000, 6 * SQ / 1000, fill=False, ec=C["orange"], lw=2)); ax.text(-.1, -.2, "checkerboard (8x6 squares)", color=C["orange"], fontsize=8)
    ax.set_aspect("equal"); ax.set_xlabel("m (board frame, 120 mm squares)"); ax.set_ylabel("m"); ax.legend(fontsize=8, loc="upper left"); ring = fig_b64(f)
    f, ax = plt.subplots(1, 2, figsize=(9, 2.9))
    ax[0].bar([0, 1, 2], [2161, 161, 104], color=[C["grey"], C["orange"], C["blue"]], width=.55); ax[0].set_xticks([0, 1, 2]); ax[0].set_xticklabels(["first\nattempt", "ring prior\n(v2)", "marker-resolved\n(v3)"], fontsize=8); ax[0].set_ylabel("leave-one-view-out px @4K"); ax[0].set_title("same SAM 3D Body output, 36 frames", fontsize=9)
    ax[0].text(0, 2161, "2161", ha="center", va="bottom"); ax[0].text(1, 161, "161", ha="center", va="bottom"); ax[0].text(2, 104, "104", ha="center", va="bottom")
    ho = [126, 105, 182, 67, 86]; ax[1].bar(range(5), ho, color=C["blue"]); ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["T%d" % i for i in range(1, 6)]); ax[1].set_ylabel("held-out error px @4K"); ax[1].set_title("per camera (final)", fontsize=9); val = fig_b64(f)
    body = f"""
<div class="card"><b>Question.</b> MMC17 is a separate experiment: six GoPro HERO11 tripod cameras (T1&ndash;T6, 3840&times;2160 Gyroflow-stabilised) around a couch. Recover their extrinsics from the DBECal checkerboard clip, assuming a shared camera model.</div>
{kpis([("0.33 px", "board reprojection RMS, all 6 views 0.25&ndash;0.43 (blind to the alignment error, see below)"), ("2161 &rarr; 104 px", "leave-one-camera-out over 111 frames: first attempt vs marker-resolved calibration"), ("f = 1680 px", "shared focal; principal point (1920, 1138); k1 = 0.036"), ("T6 not synchronized", "~23 fps clock; excluded from fusion")])}
<div class="card"><b>Update (same day).</b> The layout prior below chose the <b>wrong alignment for T4 and T5</b>. The Vicon markers (<a href="05_vicon_calibration.html">report 5</a>) matched 39 and 17 markers with option 0 and none with the ring-prior option 2, so the final calibration (v3) uses option 0 for T1&ndash;T5. SAM 3D Body cross-camera consistency over 111 frames improved from 161 to 104 px leave-one-camera-out (T4 119 &rarr; 67, T5 152 &rarr; 86). Your finding that <b>T6 runs at ~23 fps</b> explains why it never fitted; it is excluded.</div>
<h2>Method</h2>
<ul><li>Board corners: <code>findChessboardCornersSB</code> on the ECal Gyroflow clip. The board has 7&times;5 inner corners, but only a clean <b>6&times;5 sub-grid</b> is found in every view (lattice residual 0.3&ndash;0.8 px). The 7&times;5 detections are unreliable (up to 67 px lattice error) and were not used.</li>
<li>Shared intrinsics + 6 board poses solved with <code>calibrateCamera</code> (aspect fixed, no tangential, k3 fixed): board units = squares.</li>
<li><b>Gyroflow check.</b> ECal and trial frames of the same tripod camera align with scale 1.0002 and 0.5&ndash;0.9 px displacement for T1&ndash;T5, so one calibration covers both clips. T6 matched poorly (28 inliers, scale 0.99, &minus;2.6&deg;).</li></ul>
{table(["Camera", "scale", "rotation", "median displacement px", "inliers"], [["T1", "1.0002", "0.00&deg;", "0.5", "582"], ["T2", "1.0001", "0.00&deg;", "0.5", "1074"], ["T3", "1.0002", "0.00&deg;", "0.6", "804"], ["T4", "1.0001", "0.00&deg;", "0.9", "569"], ["T5", "1.0003", "0.00&deg;", "0.7", "341"], ["<span class='bad'>T6</span>", "0.9915", "&minus;2.63&deg;", "41.1", "28"]], num=(1, 2, 3, 4))}
<h2>The trap: reprojection error cannot see alignment errors</h2>
<div class="card">Each view's board pose is solved independently. A 6&times;5 grid inside a 7&times;5 board can be the left or right six columns, and the symmetric board can be flipped 180&deg;. Choosing the wrong one for a view moves that camera by a square or mirrors it <b>at zero reprojection cost</b>. My first "lowest RMS wins" search (RMS 0.33 px) therefore picked an arbitrary alignment, and SAM 3D Body triangulation on MMC17 gave errors of hundreds of pixels.</div>
<h3>How it was resolved</h3>
<ol><li><b>Offset from the image.</b> Extrapolate the lattice one square past each end. The side that continues into bright checker squares is inside the board; the other side runs onto the dark table.</li></ol>
{table(["Camera", "left side brightness (max)", "right side brightness (max)", "extra column"], [["T1", "62", "153", "right"], ["T2", "102", "161", "right"], ["T3", "80", "130", "right"], ["T4", "83", "54", "left"], ["T5", "114", "46", "left"], ["T6", "72", "159", "right"]], num=(1, 2))}
<ol start="2"><li><b>Flip from the layout prior</b> (your Figure 2a): the six cameras form a ring around the couch, all aimed at it. Cost = radius spread + aim angle + height spread. Of the 32 image-consistent hypotheses the best scored 0.59 versus 0.84 for the runner-up (median 1.04).</li></ol>
{fig(ring, "Top-down camera layout from the final calibration (blue, with optical axes) vs the first attempt (grey). The ring order T1&rarr;T6 is now monotonic around the couch, as in the paper's figure. Scale uses 120 mm squares.")}
<h2>Validation</h2>
{fig(val, "Re-triangulating the same cached SAM 3D Body output on 111 trial frames with the three calibrations (T1&ndash;T5, leave-one-camera-out). The remaining ~100 px at 4K (~4 % of image width) is dominated by the model's supine pose; (T6, unsynchronized, is excluded.)")}
<p>The triangulated SAM 3D Body tracking on MMC17, frame by frame, is in <a href="06_mmc17_sam3d_tracking.html">report 6</a>.</p>
<h3>Tests that did not work (reported for completeness)</h3>
<ul><li>SIFT matches between camera pairs: only 10&ndash;26 RANSAC-consistent matches per pair across such wide baselines; the test failed even for pairs whose relative geometry was internally consistent, so it was discarded.</li>
<li>Reflective-ball epipolar hit rate: 8&ndash;36 blobs per view, ~9 % hits for every alignment hypothesis &mdash; the blob detector is too weak to discriminate.</li></ul>
<h2>Open items</h2>
<ul><li><b>Scale:</b> the lab's SOP notes give 13 cm squares; the Vicon scale (report 5) implies 120.7 mm, and 120 mm is used from now on (instruction). Pixel-based errors are unaffected; only the metre axes of the ring plot depend on it.</li>
<li><b>T6</b> is unsynchronized (~23 fps per your finding): provide its real timestamps to resample it, or keep it excluded.</li><li>Intrinsics are shared by all cameras (assumed from the HERO11 model); the Vicon markers support a per-camera refinement later.</li></ul>
<h2>Trial views (for orientation)</h2>{fig(img_b64(f"{TMP}/ical/tdb_f400.jpg", 1400), "MMC17 TDB, frame 400, T1&ndash;T6 (local use only: contains an identifiable volunteer).")}
<h2>Reproduce</h2><pre>$py = "C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe"
&amp; $py calib_extrinsics.py   # detect board, shared intrinsics   (tmp\\board_sub.npz)
&amp; $py calib_offsets.py      # which side continues into the board
&amp; $py calib_final.py        # ring prior + refinement -> tmp\\mmc17_calib.json
&amp; $py calib_zoom_check.py   # ECal vs trial virtual-camera check</pre>"""
    page("04_mmc17_calibration", "4 &middot; MMC17 tripod-camera extrinsic calibration", "CUTrial/MM17 &middot; DBECal checkerboard", body)

# ============================================================== R5: Vicon registration ==================================================
def r5_old():
    z = np.load(f"{TMP}/vicon_registration.npz"); names = ["L hip (ASIS)", "R hip (ASIS)", "L knee", "R knee", "L ankle", "R ankle"]; SQ = float(z["sq_mm"])
    Ab, pred = z["Ab"] * SQ / 1000, z["fixed_pred"] * SQ / 1000
    f, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for k, n in enumerate(names):
        ax[0].plot([Ab[k, 0], pred[k, 0]], [Ab[k, 1], pred[k, 1]], "-", color=C["grey"], lw=1); ax[0].plot(*Ab[k, :2], "o", color=C["blue"], label="SAM 3D Body, triangulated" if k == 0 else None); ax[0].plot(*pred[k, :2], "s", color=C["orange"], label="Vicon, registered" if k == 0 else None)
        ax[0].annotate(n, Ab[k, :2], textcoords="offset points", xytext=(4, 3), fontsize=7)
        ax[1].plot([Ab[k, 0], pred[k, 0]], [-Ab[k, 2], -pred[k, 2]], "-", color=C["grey"], lw=1); ax[1].plot(Ab[k, 0], -Ab[k, 2], "o", color=C["blue"]); ax[1].plot(pred[k, 0], -pred[k, 2], "s", color=C["orange"])
    ax[0].set_title("top view (board x,y)", fontsize=9); ax[1].set_title("side view (x, height)", fontsize=9)
    for a in ax: a.set_aspect("equal"); a.set_xlabel("m")
    ax[0].legend(fontsize=7); reg = fig_b64(f)
    res = z["fixed_res"]; resf = z["free_res"]
    body = f"""
<div class="card"><b>Question.</b> The Vicon marker CSVs are in the Vicon room frame. Where is that frame relative to the calibrated cameras? (Needed to project markers into images and to benchmark pose models against Vicon.)</div>
{kpis([("%.0f mm" % np.sqrt(np.mean(res ** 2)), "registration RMS with the then-assumed 130 mm squares"), ("%.0f mm" % np.sqrt(np.mean(resf ** 2)), "RMS if scale is fitted freely (&rarr; %.0f mm squares)" % (1 / float(z["free_s"]))), ("%d frames" % len(z["frames"]), "steady supine frames used (T1&ndash;T5)"), ("12 mm", "frame-to-frame scatter of the triangulated landmarks")])}
<h2>Method (coarse registration)</h2>
<ul><li>The checkerboard lies level on the couch, so board <i>&minus;z</i> is vertical and Vicon <i>Z</i> is up: the transform has only <b>yaw, translation and scale</b> (Vicon rotated by diag(1,&minus;1,&minus;1) so handedness is preserved).</li>
<li>Correspondences (landmark &rarr; model joint): ASIS &rarr; hips, mid(lateral, medial knee marker) &rarr; knees, mid(malleoli) &rarr; ankles. Median over {len(z["frames"])} frames from frame 250 on (skipping the leg-tap sync motion at the start).</li>
<li>Landmark positions come from SAM 3D Body triangulated through the calibrated T1&ndash;T5 cameras (report 4), so errors here include the model's supine pose error.</li></ul>
<h2>Results</h2>
{fig(reg, "Triangulated SAM 3D Body landmarks (blue) vs Vicon landmarks after registration (orange), 130 mm squares. Grey lines are the residuals.")}
{table(["Landmark", "residual, 130 mm squares (mm)", "residual, free scale (mm)"], [[n, "%.0f" % a, "%.0f" % b] for n, a, b in zip(names, res, resf)], num=(1, 2))}
<h2>Takeaways</h2>
<ul><li class="warn">This is a <b>coarse</b> registration: ~10 cm RMS, dominated by SAM 3D Body's knee/hip error on a supine, arms-up pose (a stricter benchmark than the registration itself).</li>
<li>The free-scale fit prefers 169 mm squares, 30 % more than the 130 mm assumed. Either the model's body scale is biased or the square size differs; a ground-truth square size removes this ambiguity.</li>
<li><b>Next:</b> refine the transform by projecting the 16+ torso markers (R11&hellip;L42) into the images and maximising alignment with the reflective balls &mdash; sub-cm registration is what the respiration step needs.</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\sam3d-venv\\Scripts\\python.exe register_vicon.py 130     # argument = square size in mm</pre>"""
    page("05_vicon_registration", "5 &middot; Vicon &rarr; camera world registration (coarse)", "MMC17 TDB &middot; landmark-based similarity transform", body)

def r6():
    import report_mmc17_tracking as R
    R.build(dict(TMP=TMP, img_b64=img_b64, kpis=kpis, table=table, page=page))

def _ctx():
    return dict(TMP=TMP, img_b64=img_b64, kpis=kpis, table=table, fig=fig, page=page, plt=plt, fig_b64=fig_b64, C=C)

def r5():
    import report_vicon as RV
    RV.build_r5(_ctx())

def r7():
    import report_vicon as RV
    RV.build_r7(_ctx())

def r8():
    import report_labelling as RL
    RL.build(_ctx())

def r9():
    import report_fit_v2 as RF
    RF.build(_ctx())

def r10():
    import report_t6 as RT
    RT.build(_ctx())

def r11():
    import report_fit_v2 as RF
    RF.build_bench(_ctx())

def r12():
    import report_breathing as RB
    RB.build(_ctx())

def r13():
    import report_chest_tracking as RC
    RC.build(_ctx())

def r14():
    import report_tracking as RK
    RK.build(_ctx())

def r15():
    import report_skin as RS
    RS.build(_ctx())

def r16():
    _FIGN['cur'] = 'r16'; _FIGN['n'] = 0
    import report_mv as RM
    RM.build(_ctx())

def r17():
    _FIGN['cur'] = 'r17'; _FIGN['n'] = 0
    import report_skin_dense as RD
    RD.build(_ctx())

def r18():
    _FIGN['cur'] = 'r18'; _FIGN['n'] = 0
    import report_thoracoabdominal as RT18
    RT18.build(_ctx())

def r19():
    _FIGN['cur'] = 'r19'; _FIGN['n'] = 0
    import report_diagnose as RDG
    RDG.build(_ctx())

def r20():
    _FIGN['cur'] = 'r20'; _FIGN['n'] = 0
    import report_sapiens as RSP
    RSP.build(_ctx())

# ============================================================== index ==================================================================
def index():
    rows = [("01_instanthmr_multiview", "InstantHMR multi-view on the vault rig", "done", "6-view fusion in 33&ndash;47 ms with tracking; supine pose is the weak point"),
            ("02_synchronization", "Are the vault videos synchronized?", "done", "stills not synced; &plusmn;2 frame offsets halve error in moving frames; supine error is not a sync problem"),
            ("03_sam3d_vs_instanthmr", "SAM 3D Body multi-view vs InstantHMR", "done", "same cross-view consistency (10.6/27.2 vs 9.0/27.1 px); needs ground truth to separate them"),
            ("04_mmc17_calibration", "MMC17 extrinsic calibration", "done (T6 excluded)", "reprojection RMS hid an alignment ambiguity; the ring prior got T4/T5 wrong, the Vicon markers fixed it; leave-1-out 2161 &rarr; 104 px"),
            ("05_vicon_calibration", "Vicon &harr; GoPro calibration from the marker balls", "done (T1&ndash;T5)", "465 matched markers, PnP RMS 1&ndash;1.7 px; 0.5 mm triangulation check; implied square 120.7 mm"),
            ("06_mmc17_sam3d_tracking", "MMC17: SAM 3D Body multi-view tracking", "done", "frame-by-frame triangulated skeleton overlays on all six cameras, with fit quality over time"),
            ("07_marker_guided_mesh_fit", "Marker-guided SAM 3D Body mesh fit", "done (6 frames)", "labelled Vicon markers pull the mesh: median marker-to-surface 32&ndash;60 &rarr; 11&ndash;26 mm; arms/head only loosely constrained"),
            ("08_marker_labelling", "Labelling the arm/head markers from the SOP", "new", "17 of 19 unlabelled markers named from the data sheet + geometry; priority classes used as weights"),
            ("09_marker_fit_v2", "Marker-guided fit v2 (named markers, priority weights)", "new", "same frames as report 7, named arm/head markers attached to their own limb"),
            ("10_t6_recovery", "Recovering camera T6 with the markers", "new", "T6 had moved after the checkerboard clip; marker pose search gives 104 matches, 2.05 px PnP; no time drift"),
            ("11_sam3d_multiview_vs_vicon", "SAM 3D Body multi-view vs Vicon (landmarks, mm)", "done", "benchmark in the Vicon frame: ~17 cm median even triangulated; marker guidance needed"),
            ("12_breathing_review", "Breathing: project review, approaches, MoSE3 idea", "updated", "chest motion is ~rank 1 (94 %) at ~15 mm; SAM 3D Body chest jitter ~14 mm; MoSE3 unsuited to the mm signal; proposed two-layer design"),
            ("13_chest_tracking_feasibility", "Markerless chest tracking vs Vicon: feasibility + controls", "new", "multi-view texture tracking: r = 0.983, MAE 0.92 mm on one trial (null controls 3&ndash;9 mm); resolution-independent; seeds need to be within ~1 cm"),
            ("14_tracking_recording", "Marker-guided tracking: the recording", "done", "291-frame fitted mesh vs SAM 3D Body vs Vicon in 5 cameras (HTML scrubber + Rerun file)"),
            ("15_skin_tracking_balls_painted_out", "Skin-only chest tracking, balls painted out", "new", "balls inpainted out of the video: r 0.983, MAE 0.90 mm unchanged (same-points track difference 0.34 mm); only 23 points, upper chest only"),
            ("16_multiview_sam3d_mesh", "Markerless multi-view SAM 3D Body mesh", "new", "one body fitted to the five per-camera meshes: 41 mm to the Vicon markers vs 114 mm for a typical single camera (35 mm for the best one, which needs Vicon to pick); chest breathing not improved"),
            ("17_dense_skin_tracking", "Dense skin-only chest tracking (1,100+ points)", "new", "23 &rarr; 1,142 tracked 3D points at the same accuracy (r 0.985, MAE 0.85 mm); seeds on the markerless multi-view mesh work too; per-marker check now possible; limits: texture, one subject"),
            ("18_adb_vs_tdb", "Abdominal vs thoracic deep breathing (ADB vs TDB)", "new", "the draft paper's Vicon-only regional analysis on the cohort, and the same question answered from video on MMC17 (raw ADB footage, geometry re-derived)"),
            ("19_why_marker_guided_is_better", "Why the marker-guided fit is so much better; the dark-clothing question", "new", "SAM 3D Body's error is depth (13&ndash;31 cm too far in 4 of 5 cameras), which markers fix; hold-out markers; brightened-frame test"),
            ("20_sapiens2_cold_start", "Sapiens2 pose and body parts as a markerless cold start", "new", "triangulated Sapiens2 keypoints several times closer to Vicon than SAM 3D Body's; body fitted to them with and without the body-part mask")]
    todo = [("Whole-trial marker-guided fit with temporal smoothing and a shared subject shape", "next"), ("Fine-tune a supine pose model on the marker-fitted meshes (subject-wise splits)", "needs the whole-trial fits for all subjects"), ("Respiration: chest-surface signal from the fitted mesh vs the Vicon chest array", "needs the whole-trial fit"), ("T6 resampling onto the common clock", "needs its real timestamps")]
    body = f"""<p class="meta">One page per experiment; each is self-contained. Rebuilt {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")} by <code>mv/make_reports.py</code>. Newest experiments are marked <b>new</b>.</p>
<h2>Completed</h2>{table(["Report", "Status", "One-line result"], [[f'<a href="{s}.html">{t}</a>', st, r] for s, t, st, r in rows])}
<h2>Not yet run</h2>{table(["Experiment", "Blocked on"], [[a, b] for a, b in todo])}
<h2>Changelog</h2><ul><li><b>Reports 18&ndash;20 added:</b> ADB vs TDB (Vicon cohort + markerless), why the marker-guided fit is better (depth error, hold-out markers, dark clothing), and the Sapiens2 cold start.</li><li><b>Reports 16&ndash;17 added:</b> the markerless multi-view SAM 3D Body mesh scored against Vicon, and the dense (1,100+ point) skin-only tracking with seeds from that mesh; the density sweep and per-marker agreement are in report 17.</li><li><b>Report 15 added:</b> the skin-tracking test redone with the balls actually painted out of the video (the earlier test only excluded points near the balls); report 13 caveats updated.</li><li><b>Reports 13&ndash;14 added:</b> the chest-tracking feasibility test with controls, and the tracking recording; report 12 updated (resolution claim corrected, experiment 1 done).</li><li><b>Reports 8&ndash;11 added</b> (one per new experiment): marker labelling from the SOP, fit v2, T6 recovery, benchmark in the Vicon frame.</li><li>Reports 4&ndash;6 corrected for the T4/T5 alignment error found with the markers (report 5); checkerboard squares now 120 mm.</li></ul>
<h2>Open questions</h2><ul><li>T6 (24 fps): see report 10 for the pose search with the markers; real timestamps would still help.</li><li>Marker names for the arm/head markers: assigned from the SOP and geometry (report 8); Brian's labelled C3D files would confirm them.</li><li>True timecode offsets between the vault2 videos, if known.</li></ul>"""
    page("index", "Multi-view SGRT pose &amp; respiration &mdash; experiment reports", "TJU / CU healthy-volunteer data", body)

if __name__ == "__main__":
    for fn in (r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16, r17, r18, r19, r20, index):
        fn(); print("wrote", fn.__name__)
