"""Report 10: recovering camera T6 (24 fps) in the Vicon frame from the marker balls."""
import json
import numpy as np


def build(ctx):
    TMP, img_b64, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    S = json.load(open(f"{TMP}/t6_search.json")); J = json.load(open(f"{TMP}/mmc17_vicon_calib.json"))["cams"]; T = json.load(open(f"{TMP}/t6_timing.json"))
    f, ax = plt.subplots(figsize=(5.8, 5.0))
    for k, v in J.items():
        R = np.array([[0, 0, 0]])
        import cv2
        Rm = cv2.Rodrigues(np.array(v["rvec"]))[0]; c = -Rm.T @ np.array(v["tvec"]); d = Rm[2]
        ax.plot(c[0], c[1], "o", color=C["red"] if k == "T6" else C["blue"]); ax.annotate(k, c[:2], textcoords="offset points", xytext=(5, 5)); ax.arrow(c[0], c[1], d[0] * .3, d[1] * .3, color=C["red"] if k == "T6" else C["blue"], head_width=.04)
    ax.set_aspect("equal"); ax.set_xlabel("Vicon x (m)"); ax.set_ylabel("Vicon y (m)"); ax.set_title("camera positions in the Vicon frame (top view)", fontsize=9); ring = fig_b64(f)
    shifts = S["shift_matches"]; f, ax = plt.subplots(figsize=(5.6, 2.6)); xs = sorted(int(k) for k in shifts); ax.plot(xs, [shifts[str(k)] for k in xs], "o-", color=C["red"])
    ax.set_xlabel("Vicon frame shift (image frames)"); ax.set_ylabel("matches (8 frames)"); ax.set_ylim(0, max(shifts.values()) * 1.15); shiftfig = fig_b64(f)
    trows = [[a, "%+d" % o[0][0], "%d / %d" % (o[0][1], o[0][2]), "%+d" % o[1][0], "%d / %d" % (o[1][1], o[1][2])] for a, o in T]
    body = f"""
<div class="card"><b>Question.</b> The lab's notes say camera T6 was recorded at <b>24 fps</b> (the others at 30). Its extrinsics from the checkerboard never matched anything. Is T6 usable, and was it the frame rate or something else?</div>
{kpis([("104 matches", "T6 markers matched after a pose search (all four board hypotheses: 0)"), ("2.05 px", "T6 PnP reprojection RMS (81 inliers); the others 0.98&ndash;1.73 px"), ("no drift", "match count flat at 10&ndash;12 over the whole trial at zero time shift"), ("camera moved", "ECal vs trial: 2.6&deg; rotation, 109 px shift")])}
<h2>Finding</h2>
<ul><li><b>The problem was the camera moving, not the frame rate.</b> The ECal-vs-trial comparison (report 4) already showed T6's view changed between the checkerboard clip and the trial (scale 0.9915, rotation &minus;2.6&deg;, 109 px shift; only 28 feature inliers, while T1&ndash;T5 matched with sub-pixel agreement). A board-based calibration cannot see that; the markers can.</li>
<li>A pose search in the Vicon frame around T6's board-derived hypotheses finds a placement with <b>93 matched markers over 8 frames</b> (then 104 matches / 81 inliers in the final PnP, 2.05 px RMS). Camera centre in the Vicon frame: (1.02, 0.82, 1.46) m.</li></ul>
{fig(ring, "All six cameras in the Vicon frame. T6 (red) is on the same ring as the others, consistent with the layout in the paper's Figure 2a.")}
{fig(img_b64(f"{TMP}/t6_overlay_f420.jpg", 1500), "Frame 420: projected Vicon markers (cyan rings) and detected balls (red dots), T6 with the recovered pose (left) vs T5 (right).")}
<h2>Timing (24 fps)</h2>
{table(["Frame", "T6 best shift", "T6 matches @0 / best", "T5 best shift", "T5 matches @0 / best"], trows, num=(1, 2, 3, 4))}
{fig(shiftfig, "Marker matches for T6 as the Vicon frame is shifted (frames 260&ndash;540): flat between &minus;6 and +6, so these steady frames cannot resolve a shift smaller than ~6 frames.")}
<ul><li>At zero shift T6 matches as many markers as T5 at every frame from 100 to 1100: there is <b>no growing offset</b> over the 37 s trial. A 24 fps stream conformed to 29.97 fps (the file has 1125 frames like the others) would give at most ~1 frame of misalignment, consistent with this.</li>
<li class="warn"><b>Limit:</b> the subject is nearly still, so this test cannot detect an offset of a few frames. For slow breathing that is harmless; for fast motion T6 would need its real timestamps.</li></ul>
<h2>Consequence</h2>
<p>T6 is added to <code>mmc17_vicon_calib.json</code> (the previous T1&ndash;T5 file is kept as <code>mmc17_vicon_calib_T1-T5.json</code>). Cameras now live in the Vicon frame in metres, so report 11 can benchmark multi-view SAM 3D Body directly against Vicon using up to six cameras.</p>
<h2>Reproduce</h2><pre>C:\\dev\\sam3d-venv\\Scripts\\python.exe t6_search.py      # pose search
C:\\dev\\sam3d-venv\\Scripts\\python.exe t6_finalize.py    # PnP, adds T6 to the calibration
C:\\dev\\sam3d-venv\\Scripts\\python.exe t6_timing.py      # drift test</pre>"""
    page("10_t6_recovery", "10 &middot; Recovering camera T6 with the markers", "MMC17 &middot; pose search + PnP in the Vicon frame", body)
