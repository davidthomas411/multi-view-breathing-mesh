"""Report 8: labelling the unlabelled arm/head Vicon markers from the MMC data collection sheet + the lab Q&A (PDF)."""
import json
import numpy as np


def build(ctx):
    TMP, img_b64, kpis, table, fig, page = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page"))
    L = json.load(open(f"{TMP}/mmc17_marker_labels.json"))
    order = {"head (headband)": 0, "upper arm": 1, "elbow": 2, "forearm": 3, "wrist": 4, "hand/finger": 5}
    rows = [[k, v["label"], "right" if v["side"] == "R" else "left", v["segment"], v["confidence"]] for k, v in sorted(L.items(), key=lambda kv: (kv[1]["side"], order.get(kv[1]["segment"], 9), kv[0]))]
    body = f"""
<div class="card"><b>Question.</b> Vicon names the pelvis, leg, foot and chest-array markers but leaves the head and arm markers as unlabelled trajectories (<code>*55</code>&hellip;<code>*73</code>). Can the lab's data collection sheet (v3) and the Q&amp;A document name them, and how should the marker colour priorities in the paper figure be used?</div>
{kpis([("17 of 19", "unlabelled markers named (the other two, *72 and *73, are missing at the frames used)"), ("2 + 7 + 8", "head pair + left-arm + right-arm markers"), ("3 classes", "tracking priority: red 1.0, green 0.4, blue 0.15 (fit weights)"), ("120 mm", "checkerboard square size used from now on")])}
<h2>What the documents say (relevant facts)</h2>
{table(["Source", "Fact", "Used for"], [
 ["Data sheet", "72 markers: 56 on bases + 4 clusters &times; 4 markers. Full names for pelvis, thigh/shank clusters (SATH, IATH, SPTH, IPTH, SASK, IASK, SPSK, IPSK), feet, head (RAHD LAHD RPHD LPHD), acromion (RAC LAC), CLAV, arms (AUPA PUPA ELB MELB FRA RAD ULN FIN).", "the candidate names"],
 ["Data sheet", "Supine setup: <b>remove</b> LPSIS RPSIS BPV T10 C7 RBAK and the sternal notch; add a <b>4&times;4 chest array</b>, top row ~4 cm below the clavicular notch, bottom row level with the navel.", "which names can exist in these trials"],
 ["Q&amp;A (PDF)", "<b>T6 recorded at 24 fps</b>; all other cameras 30 fps. Sync by the knee stomp/clap; C3D starts 1 s before the sync action.", "report 10"],
 ["Q&amp;A (PDF)", "Checkerboard squares are <b>13 cm</b>. Intrinsics calibrated once (same GoPros); extrinsics per session. <b>No GoPro&harr;Vicon frame alignment was done by the lab</b> (this project does it, reports 5/10).", "scale; context"],
 ["Q&amp;A (PDF)", "Vicon origin: bottom-right corner of the rightmost black tile, viewed from camera T2 (MMC02 SUP DBEcal).", "sanity check of the Vicon frame"],
 ["Q&amp;A (PDF)", "Marker priority from the figure: <b>red</b> = high tracking priority / stable; <b>green</b> = tracked less (soft-tissue artefact, orientation info); <b>blue</b> = low priority (not visible to cameras). <b>Treat markers as independent points, not segments.</b>", "fit weights; no rigid-segment constraints"],
 ["Existing C3D", "The supine <code>CLIN_SUP.c3d</code> (MMC06) also leaves <code>*55</code>&ndash;<code>*69</code> unnamed &mdash; there is no labelled ground truth for these markers in the files we have.", "labels must be inferred"]])}
<h2>Method</h2>
<ol><li><b>Body-fixed frame</b> from named markers: lateral axis from the ASIS pair (subject right &rarr; left), superior axis from the pelvis to the chest array, anterior = up (supine).</li>
<li><b>Head:</b> the unlabelled markers at head level (more than 12 cm beyond the chest array) close to the midline. There are exactly two (<code>*69</code> on the subject's right, <code>*56</code> on the left), matching your statement that two markers sit on a headband, so they are named RAHD / LAHD.</li>
<li><b>Each arm</b> (side from the sign of the lateral coordinate): the <i>wrist pair</i> is the closest pair among the four markers farthest from the shoulder (5&ndash;7 cm apart); the <i>finger</i> marker lies beyond it; the <i>forearm</i> marker is the one ~13 cm from the wrist centre; the two markers nearest the forearm marker are the <i>elbow</i> pair; the rest are the <i>upper-arm</i> pair.</li>
<li>Within each pair the assignment (radial/ulnar, lateral/medial, anterior/posterior) uses only weak geometric cues and is flagged <b>low confidence</b>. Segment and side are reliable; the pair order is not.</li></ol>
<h2>Result</h2>
{table(["Vicon trajectory", "Assigned name", "Side", "Segment", "Confidence"], rows)}
{fig(img_b64(f"{TMP}/labelled_markers_f420.jpg", 1500), "Assigned names projected into T1, T2, T3, T5 at frame 420 (red = high-priority class, blue = low-priority class). Head names sit on the headband, finger/wrist names on the hands, forearm names on the forearms, on the correct sides across views.")}
<h2>Priority classes used as fit weights</h2>
{table(["Class", "Weight", "Markers"], [["red (high)", "1.0", "ASIS, iliac crest, chest array, lateral knee/ankle, 1st/5th metatarsal, toe tip, head, forearm, wrist, finger, elbow/upper arm"], ["green (tracked less)", "0.4", "trochanter, thigh and shank clusters (SATH IATH SPTH IPTH SASK IASK SPSK IPSK), tibial tuberosity, toe (2nd&ndash;3rd metatarsal), calcaneus"], ["blue (rarely visible)", "0.15", "medial elbow, medial knee, medial malleolus"]], num=(1,))}
<h2>Limits</h2>
<ul><li>RAC/LAC (acromion), CLAV and the posterior head markers (RPHD/LPHD) do not appear in the unlabelled set; they were either untracked or hidden by the table, so the shoulders are unconstrained.</li>
<li>Pair-order ambiguity (above) matters little for body-part attachment but would matter for per-marker anatomy. Brian's labelled C3D files (promised by the lab) would settle it.</li>
<li class="warn">The routine is validated only visually on MMC17; it should be run on the other subjects and spot-checked the same way.</li></ul>
<h2>Reproduce</h2><pre>python marker_labels.py   # label_unlabelled(frames, names) -&gt; tmp\\mmc17_marker_labels.json</pre>"""
    page("08_marker_labelling", "8 &middot; Labelling the arm and head markers from the SOP", "MMC data collection sheet v3 + lab Q&amp;A &middot; MMC17", body)
