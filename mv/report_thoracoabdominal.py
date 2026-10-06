"""Report 18: abdominal- vs thoracic-directed deep breathing (ADB vs TDB) measured by MARKERLESS SKIN TRACKING (video only) against the Vicon chest array, plus the draft paper's Vicon-only cohort analysis."""
import json, os
import numpy as np
from scipy import stats


def build(ctx):
    TMP, img_b64, kpis, table, fig, page, plt, fig_b64, C = (ctx[k] for k in ("TMP", "img_b64", "kpis", "table", "fig", "page", "plt", "fig_b64", "C"))
    TA = f"{TMP}/thoracoabdominal"; S = json.load(open(f"{TA}/vicon_summary.json")); subs = S["subjects"]; B = {t: np.array(S["B"][t]) for t in ("ADB", "DB", "TDB")}; ROWS = {t: np.array(S["rows"][t]) for t in ("ADB", "DB", "TDB")}; i17 = subs.index("MMC17")
    # ---- markerless results per trial and validity rule; the primary rule is the one that supports EVERY ball position with >= 60 tracked points in both trials
    RULES = [("&ge;3 cameras, &gt;50 % of frames", "m41mv_e33_wide", "m41mv_e33_dense"), ("&ge;3 cameras, &gt;25 % of frames", "m41mv_e33_wide_f25", "m41mv_e33_dense_f25"), ("<b>&ge;2 cameras, &gt;25 % of frames (primary)</b>", "m41mv_e33_wide_f25_c2", "m41mv_e33_dense_f25_c2")]
    def get(trial, wide, dense):
        for tag in (wide, dense):
            p = f"{TA}/markerless_{trial}_{tag}.json"
            if os.path.exists(p): return tag, json.load(open(p))
        return None, None
    R = {(trial, k): get(trial, w, d) for trial in ("TDB", "ADB") for k, (_, w, d) in enumerate(RULES)}
    tagT, mT = R[("TDB", 2)]; tagA, mA = R[("ADB", 2)]; have = mT is not None and mA is not None
    oracle = json.load(open(f"{TA}/markerless_TDB_m41_e33_dense_f25_c2.json")) if os.path.exists(f"{TA}/markerless_TDB_m41_e33_dense_f25_c2.json") else None
    figs = {}; rowsB = []; senrows = []; per_rows = []; kp = []; answer = ""
    if have:
        names = list(mT["amp_vicon"].keys()); rowof = mT["row_of"]
        # ---- per-marker agreement (amplitude) and time-series agreement
        def detr(x):
            x = np.asarray(x, float); t = np.arange(len(x)); ok = ~np.isnan(x); return x - np.polyval(np.polyfit(t[ok], x[ok], 2), t)
        agree = {}
        for trial, m in (("TDB", mT), ("ADB", mA)):
            sv, sm = m["series"]["vicon"], m["series"]["markerless"]; out = {}
            for n in names:
                if sm.get(n) is None: continue
                a = detr(sv[n]); b = detr([np.nan if v is None else v for v in sm[n]]); ok = ~np.isnan(b)
                out[n] = dict(r=float(np.corrcoef(a[ok], b[ok])[0, 1]), mae=float(np.mean(np.abs(a[ok] - b[ok]))), av=m["amp_vicon"][n], am=m["amp_markerless"][n], npts=m["n_points_near"][n])
            agree[trial] = out
        # ---- figure A: amplitude per ball, skin tracking vs Vicon, both trials
        f1, ax = plt.subplots(1, 2, figsize=(10.4, 4.0), gridspec_kw=dict(width_ratios=[1, 1.15]))
        for trial, col, mk_ in (("TDB", C["blue"], "o"), ("ADB", C["orange"], "s")):
            xs = [agree[trial][n]["av"] for n in agree[trial]]; ys = [agree[trial][n]["am"] for n in agree[trial]]; ax[0].scatter(xs, ys, s=44, color=col, marker=mk_, label="%s (%d balls)" % (trial, len(xs)), edgecolor="k", linewidth=.4)
        allx = np.array([agree[t][n]["av"] for t in agree for n in agree[t]]); ally = np.array([agree[t][n]["am"] for t in agree for n in agree[t]]); lim = max(allx.max(), ally.max()) * 1.05
        ax[0].plot([0, lim], [0, lim], "k--", lw=.8); ax[0].set_xlim(0, lim); ax[0].set_ylim(0, lim); ax[0].set_aspect("equal"); ax[0].set_xlabel("Vicon ball: peak-to-peak vertical motion (mm)"); ax[0].set_ylabel("skin tracking around the ball (mm)")
        r_all = float(np.corrcoef(allx, ally)[0, 1]); mae_all = float(np.mean(np.abs(allx - ally))); ax[0].legend(fontsize=8, loc="upper left"); ax[0].set_title("amplitude per ball: r = %.2f, MAE = %.1f mm (n = %d)" % (r_all, mae_all, len(allx)), fontsize=9)
        xr = np.arange(1, 5); wbar = 0.2
        for j, (trial, m, col) in enumerate((("ADB", mA, C["orange"]), ("TDB", mT, C["blue"]))):
            ax[1].bar(xr + (j - 0.5) * 2 * wbar - wbar / 2 * 0, m["vicon"]["rows"], wbar, color=col, alpha=.45, label="%s Vicon" % trial); ax[1].bar(xr + (j - 0.5) * 2 * wbar + wbar, m["markerless"]["rows"], wbar, color=col, label="%s skin tracking" % trial)
        ax[1].set_xticks(xr + 0.0); ax[1].set_xticklabels(["row 1\nupper chest", "row 2\nlower chest", "row 3\nupper abdomen", "row 4\nlower abdomen"], fontsize=8); ax[1].set_ylabel("mean peak-to-peak vertical motion (mm)"); ax[1].legend(fontsize=7, ncol=2); ax[1].set_title("amplitude by marker row", fontsize=9)
        figs["agree"] = fig_b64(f1)
        # ---- figure B: maps
        f2, axs = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
        for a_, trial, m in zip(axs, ("TDB", "ADB"), (mT, mA)):
            pts = np.array(m["points"]["xy"]); amp = np.array(m["points"]["amp"]); ry = {int(k): v for k, v in m["row_y_mm"].items()}
            sc = a_.scatter(pts[:, 0], pts[:, 1], c=amp, s=6, cmap="viridis", vmin=0, vmax=32)
            for k, y in ry.items(): a_.axhline(y, color="#06b6d4", lw=.7, ls=":"); a_.text(pts[:, 0].max() + 5, y, "row %d" % k, fontsize=7, va="center", color="#0e7490")
            a_.axhline(0.5 * (ry[2] + ry[3]), color="#dc2626", lw=1.1); a_.set_aspect("equal"); a_.set_xlabel("Vicon X (mm)"); a_.set_title("%s: %d tracked skin points\nB (skin) = %.2f, B (Vicon) = %.2f" % (trial, m["n_points"], m["markerless"]["B"], m["vicon"]["B"]), fontsize=9); a_.grid(False)
        axs[0].set_ylabel("Vicon Y (mm, towards the head)"); f2.colorbar(sc, ax=axs, fraction=0.025, label="breathing amplitude of each tracked point (mm, 5-95 %)"); figs["maps"] = fig_b64(f2)
        # ---- figure C: thoracic vs abdominal displacement over time
        f3, axs = plt.subplots(2, 2, figsize=(10.4, 5.0), sharey="row")
        for col_, (trial, m) in enumerate((("TDB", mT), ("ADB", mA))):
            fr = np.array(m["series"]["frames"]) / 29.97
            for row_, (nm, rws) in enumerate((("thoracic (rows 1-2)", (1, 2)), ("abdominal (rows 3-4)", (3, 4)))):
                use = [n for n in names if rowof[n] in rws and m["series"]["markerless"].get(n) is not None]
                v = np.nanmean([detr(m["series"]["vicon"][n]) for n in use], axis=0); s_ = np.nanmean([detr([np.nan if x is None else x for x in m["series"]["markerless"][n]]) for n in use], axis=0)
                a_ = axs[row_, col_]; a_.plot(fr, v, color=C["blue"], lw=2.2, label="Vicon balls (mean of %d)" % len(use)); a_.plot(fr, s_, "--", color=C["red"], lw=1.4, label="skin tracking around the same balls")
                a_.set_title("%s &middot; %s".replace("&middot;", "-") % (trial, nm), fontsize=9)
                if row_ == 1: a_.set_xlabel("time (s)")
                if col_ == 0: a_.set_ylabel("vertical displacement (mm)")
        h_, l_ = axs[0, 0].get_legend_handles_labels(); f3.legend(h_, ["Vicon balls (mean of the balls in the rows)", "skin tracking around the same ball positions"], loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.03)); f3.subplots_adjust(hspace=0.42, wspace=0.08, top=0.9); figs["series"] = fig_b64(f3)
        # ---- tables
        for trial, m, tag in (("TDB", mT, tagT), ("ADB", mA, tagA)):
            rowsB.append([trial, "%.3f" % m["vicon"]["B"], "%.3f" % m["markerless"]["B"], " / ".join("%.1f" % x for x in m["vicon"]["rows"]), " / ".join("%.1f" % x for x in m["markerless"]["rows"]), "%d" % m["n_points"], "%d / %d" % (m["markerless"]["n_thoracic"], m["markerless"]["n_abdominal"])])
        for k, (lab, w, d) in enumerate(RULES):
            cells = [lab]
            for trial in ("TDB", "ADB"):
                tag, m = R[(trial, k)]
                cells += ["%.2f" % m["markerless"]["B"], "%d" % m["n_points"], "%d / %d" % (m["markerless"]["n_thoracic"], m["markerless"]["n_abdominal"]), "%d" % min(m["n_points_near"].values())] if m else ["&ndash;"] * 4
            senrows.append(cells)
        if oracle: senrows.append(["TDB with seeds on the marker-fitted mesh (oracle), &ge;2 cameras, &gt;25 %", "%.2f" % oracle["markerless"]["B"], "%d" % oracle["n_points"], "%d / %d" % (oracle["markerless"]["n_thoracic"], oracle["markerless"]["n_abdominal"]), "%d" % min(oracle["n_points_near"].values()), "&ndash;", "&ndash;", "&ndash;", "&ndash;"])
        for n in sorted(names, key=lambda x: (rowof[x], x)):
            ra, rt = agree["ADB"].get(n), agree["TDB"].get(n)
            f = lambda d_: ["%.1f" % d_["av"], "%.1f" % d_["am"], "%.3f" % d_["r"], "%.2f" % d_["mae"], "%d" % d_["npts"]] if d_ else ["&ndash;"] * 5
            per_rows.append([n, rowof[n]] + f(rt) + f(ra))
        dB_v = mT["vicon"]["B"] - mA["vicon"]["B"]; dB_m = mT["markerless"]["B"] - mA["markerless"]["B"]
        kp = [("%.2f vs %.2f" % (mA["markerless"]["B"], mT["markerless"]["B"]), "B from skin tracking, ADB vs TDB (video only; Vicon: %.2f vs %.2f)" % (mA["vicon"]["B"], mT["vicon"]["B"])), ("r = %.2f" % r_all, "skin vs Vicon amplitude over the %d ball positions of both trials (MAE %.1f mm)" % (len(allx), mae_all)),
              ("%.2f mm" % np.median([agree[t][n]["mae"] for t in agree for n in agree[t]]), "median error of the skin-tracked vertical displacement vs its Vicon ball over time (%d ball positions)" % len(allx)), ("%d / %d" % (mA["n_points"], mT["n_points"]), "tracked 3D skin points, ADB / TDB (balls painted out)")]
        answer = (f"<b>Yes, and the video-only measurement sees it.</b> With the balls painted out of the video and the seeds taken from the markerless multi-view mesh (no Vicon in the tracking), the skin tracking gives a regional bias B of <b>{mA['markerless']['B']:.2f} for ADB and {mT['markerless']['B']:.2f} for TDB</b> "
                  f"(Vicon on the same trials: {mA['vicon']['B']:.2f} and {mT['vicon']['B']:.2f}). ADB is abdomen-dominant (rows 3&ndash;4 move {mA['markerless']['rows'][2]:.0f} and {mA['markerless']['rows'][3]:.0f} mm, rows 1&ndash;2 only {mA['markerless']['rows'][0]:.0f} and {mA['markerless']['rows'][1]:.0f} mm), TDB is chest-dominant (rows 1&ndash;2 {mT['markerless']['rows'][0]:.0f} and {mT['markerless']['rows'][1]:.0f} mm). "
                  f"Skin tracking and the Vicon balls agree on the amplitude at the {len(allx)} ball positions (r = {r_all:.2f}, MAE {mae_all:.1f} mm) and on the motion over time (median error {np.median([agree[t][n]['mae'] for t in agree for n in agree[t]]):.2f} mm). "
                  f"The markerless TDB&minus;ADB contrast ({dB_m:+.2f}) is a little larger than Vicon's ({dB_v:+.2f}) because the poorly textured abdomen is under-sampled in TDB (see limits).")
    vs = S["S_test"]; fr_ = S["friedman"]; pw = {d["pair"]: d["p_holm"] for d in S["pairwise"]}
    paper = dict(B=dict(ADB=(0.601, 0.215), DB=(0.925, 0.278), TDB=(1.137, 0.272)), rows=dict(ADB=[13.8, 14.1, 19.4, 28.4], DB=[20.5, 21.2, 22.7, 24.3], TDB=[27.6, 29.1, 27.9, 23.9]))
    # ---- cohort figure (Vicon only)
    f4, ax = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw=dict(width_ratios=[1, 1.15])); xs = np.arange(3); cols = {"ADB": C["orange"], "DB": C["grey"], "TDB": C["blue"]}
    for k in range(len(subs)): ax[0].plot(xs, [B["ADB"][k], B["DB"][k], B["TDB"][k]], color="#cbd5e1", lw=1, zorder=1)
    ax[0].plot(xs, [B["ADB"][i17], B["DB"][i17], B["TDB"][i17]], "o-", color="k", lw=2.4, ms=7, zorder=4, label="MMC17")
    for j, t in enumerate(("ADB", "DB", "TDB")):
        m_ = B[t].mean(); ci = stats.t.ppf(0.975, len(B[t]) - 1) * B[t].std(ddof=1) / np.sqrt(len(B[t])); ax[0].errorbar(j, m_, yerr=ci, fmt="D", color=cols[t], ms=9, capsize=5, lw=2, zorder=3, label="mean, 95 % CI" if j == 0 else None); ax[0].plot(j + 0.17, paper["B"][t][0], "x", color="#111827", ms=8, mew=2, zorder=5, label="draft paper, 17 participants" if j == 0 else None)
    ax[0].axhline(1, color="k", ls="--", lw=.8); ax[0].set_xticks(xs); ax[0].set_xticklabels(["ADB", "DB", "TDB"]); ax[0].set_ylabel("B = thoracic / abdominal amplitude"); ax[0].legend(fontsize=8, loc="upper left"); ax[0].set_title("Vicon, n = %d participants" % len(subs), fontsize=10)
    xr = np.arange(1, 5)
    for t in ("ADB", "DB", "TDB"):
        m_ = ROWS[t].mean(0); se = ROWS[t].std(0, ddof=1) / np.sqrt(len(subs)) * stats.t.ppf(0.975, len(subs) - 1); ax[1].plot(xr, m_, "o-", color=cols[t], lw=2, label=t + " (cohort mean)"); ax[1].fill_between(xr, m_ - se, m_ + se, color=cols[t], alpha=.15); ax[1].plot(xr, ROWS[t][i17], "s--", color=cols[t], lw=1.4, ms=6, mfc="none", label=t + " MMC17")
    ax[1].set_xticks(xr); ax[1].set_xticklabels(["row 1", "row 2", "row 3", "row 4"], fontsize=8); ax[1].set_ylabel("peak-to-peak vertical motion (mm)"); ax[1].legend(fontsize=7, ncol=2, loc="upper left"); ax[1].set_title("regional profile", fontsize=10); cohort = fig_b64(f4)
    rowsT = [[t, "%.3f &plusmn; %.3f" % (B[t].mean(), B[t].std(ddof=1)), "%.3f &plusmn; %.3f" % paper["B"][t], " / ".join("%.1f" % x for x in ROWS[t].mean(0)), " / ".join("%.1f" % x for x in paper["rows"][t]), "%.3f" % B[t][i17]] for t in ("ADB", "DB", "TDB")]
    G = json.load(open(f"{TMP}/mmc17_vicon_calib_ADB.json")).get("adb_report", {}); AG = json.load(open(f"{TMP}/adb_geometry.json")); geo = [[c, "%d" % v["n_inliers"], "%.2f" % v["rms_px"], "%.2f" % AG[c]["median_px"], "%.2f" % v["rotation_change_deg"], "%.0f" % v["centre_shift_mm"]] for c, v in G.items()]
    body = f"""
<div class="card"><b>Question.</b> You copied the ADB video and asked whether abdominal-directed (ADB) and thoracic-directed (TDB) deep breathing differ, as in the draft paper (<i>Maneuver-dependent thoracoabdominal surface motion during deep breathing</i>), whose example analysis uses the Vicon markers only. Does <b>skin tracking from the cameras</b> show the same thing, and how does it compare with the Vicon data on the same trials?</div>
{kpis(kp) if kp else ""}
<div class="card ok"><b>Answer.</b> {answer if answer else "pending"}</div>
<h2>The analysis (from the draft)</h2>
<ul><li>Sixteen anterior markers in a 4 &times; 4 array: rows 1&ndash;2 (upper / lower chest) = thoracic, rows 3&ndash;4 (upper / lower abdomen) = abdominal.</li>
<li>Amplitude of a marker = peak-to-peak of its (lightly smoothed) vertical (anterior-posterior) coordinate. <b>B = mean thoracic amplitude / mean abdominal amplitude</b>: B &lt; 1 abdominal-dominant, B &gt; 1 thoracic-dominant.</li></ul>
<h2>1 &middot; MMC17: skin tracking (video only) against the Vicon chest array</h2>
<p><b>What was done.</b> Same pipeline as reports 15&ndash;17, run on both trials: seeds placed on the <b>markerless multi-view SAM 3D Body mesh</b> of the first frame (no Vicon in the seeds; 8,000 seeds over the chest, abdomen and flanks), the reflective balls <b>painted out of every frame</b>, Lucas-Kanade from the reference frame in each of the five cameras, triangulation from the cameras that track a point. To compare with the 16-ball array, the tracked skin points within 45 mm of each ball position are pooled into one <i>virtual marker</i> (median vertical displacement); amplitude and B are then computed from rows 1&ndash;2 vs 3&ndash;4 exactly as for the balls. Vicon values use the same frames (every 3rd video frame) and smoothing. Primary rule: a point counts if it is triangulated in &gt;25 % of the frames from at least two cameras (the only rule that puts &ge; 60 tracked points around <i>every</i> ball in both trials; the others are in the sensitivity table).</p>
{table(["Trial", "B (Vicon, 16 balls)", "B (skin tracking)", "Vicon rows 1-4 (mm)", "skin tracking rows 1-4 (mm)", "tracked points", "virtual markers thoracic / abdominal"], rowsB, num=(1, 2, 5)) if rowsB else ""}
{fig(figs["agree"], "Left: peak-to-peak vertical motion at each of the 16 ball positions, skin tracking against the Vicon ball, TDB and ADB. Right: the same by marker row. The skin tracks reproduce both the large and the small amplitudes.") if figs else ""}
{fig(figs["maps"], "Top-down maps of the tracked skin points (colour = each point's own breathing amplitude), the four marker rows (dotted) and the thoracic/abdominal boundary (red). TDB: the chest moves; ADB: the chest is nearly still and the lower rows carry the motion.") if figs else ""}
{fig(figs["series"], "Mean vertical displacement of the thoracic (rows 1-2) and abdominal (rows 3-4) positions over the trial: Vicon balls (blue) against the skin tracking around the same balls (red dashed), both trials.") if figs else ""}
<h3>Per ball position</h3>
<details open><summary>Amplitude, error over time and tracked points for each ball position (skin tracking vs Vicon)</summary>
{table(["ball", "row", "TDB Vicon (mm)", "TDB skin (mm)", "r over time", "MAE (mm)", "points near", "ADB Vicon (mm)", "ADB skin (mm)", "r over time", "MAE (mm)", "points near"], per_rows, num=tuple(range(1, 12))) if per_rows else ""}
<p class="meta">Amplitude = peak-to-peak of the smoothed vertical displacement; r and MAE compare the displacement time series after removing a quadratic drift (same as report 13); &ldquo;points near&rdquo; = tracked skin points within 45 mm of the ball position. Skin points are 2&ndash;4.5 cm from the ball itself (its neighbourhood is painted out and excluded).</p></details>
<h3>How sensitive is B to which points are accepted?</h3>
{table(["Point-validity rule", "TDB B (skin)", "TDB points", "TDB virtual markers thoracic / abdominal", "TDB min points near a ball", "ADB B (skin)", "ADB points", "ADB virtual markers thoracic / abdominal", "ADB min points near a ball"], senrows, num=(1, 2, 4, 5, 6, 8)) if senrows else ""}
<p>Vicon: B = {mT['vicon']['B'] if have else float('nan'):.2f} (TDB) and {mA['vicon']['B'] if have else float('nan'):.2f} (ADB). With the markerless seeds the skin tracking puts TDB above 1 under every rule and ADB below 0.5 (with the seeds on the marker-fitted mesh, TDB comes out at 0.79, Vicon 0.86): the separation of the maneuvers does not depend on the rule. How close B comes to Vicon's does: the stricter rules throw away the abdominal points (bare skin, few cameras see them) and push B up, which is why a rule that supports every ball is the primary one.</p>
<h2>2 &middot; The same question on the whole cohort, Vicon only (the draft's analysis)</h2>
<p>For orientation: the CSVs in <code>CUTrial/Trials_marker_positions_by_image_frame_all_cases</code> contain ADB, DB and TDB for {len(subs)} participants with all three maneuvers (the draft has 17 and also free breathing, which is not in the export). Same definitions, 30 fps image-frame timeline.</p>
{fig(cohort, "Left: B per participant (thin grey lines), cohort mean with 95 % CI, the draft's means (x) and MMC17 (black). Right: peak-to-peak amplitude by marker row; solid = cohort mean &plusmn; 95 % CI, dashed = MMC17.")}
{table(["Maneuver", "B (this export)", "B (draft)", "rows 1-4 mean, mm (this export)", "rows 1-4 mean, mm (draft)", "MMC17 B"], rowsT, num=(1, 2, 5))}
<ul><li>Friedman &chi;&sup2;(2) = {fr_['chi2']:.2f}, p = {fr_['p']:.1g} (draft: 24.82, 4e-6); Holm-adjusted Wilcoxon: ADB vs DB p = {pw['ADB vs DB']:.1g}, ADB vs TDB p = {pw['ADB vs TDB']:.1g}, DB vs TDB p = {pw['DB vs TDB']:.1g}. ADB &lt; DB &lt; TDB holds for the cohort and no participant has B<sub>TDB</sub> &lt; B<sub>ADB</sub> (the draft has one). MMC17 sits at the abdominal end (B<sub>ADB</sub> = {B['ADB'][i17]:.2f}, cohort {B['ADB'].mean():.2f}).</li>
<li>Natural deep breathing: S = |B<sub>DB</sub> &minus; B<sub>ADB</sub>| &minus; |B<sub>DB</sub> &minus; B<sub>TDB</sub>| = {vs['mean']:.2f} &plusmn; {vs['sd']:.2f}, {vs['n_tdb']}/{len(subs)} closer to TDB (t = {vs['t']:.2f}, p = {vs['p']:.2g}); the draft had no significant preference at n = 17 (S = 0.109, p = 0.14), so do not read more into it than the draft does.</li>
<li>Absolute amplitudes in this export are ~25&ndash;30 % smaller than the draft's Table 1 (e.g. TDB row 1: {ROWS['TDB'].mean(0)[0]:.1f} vs {paper['rows']['TDB'][0]} mm). The smoothing and the analysis window were varied (5- or 9-sample filter, whole trial or the saved window) and change B by &lt; 0.04, so I read the difference as the smaller cohort and the 30 fps resampling of the export; only the direction and significance are reproduced.</li></ul>
<h2>3 &middot; Limits</h2>
<ul><li class="warn"><b>One participant, two trials</b> for the skin tracking; the cohort statistics are Vicon-only.</li>
<li class="warn"><b>The abdomen is bare skin with little texture.</b> Few tracked points lie near rows 3&ndash;4 (the 8,000 seeds give {mT['n_points'] if have else '?'} / {mA['n_points'] if have else '?'} tracked points in total, concentrated on the chest), and the amplitude of the upper-abdomen row is under-estimated (row 3: skin {mT['markerless']['rows'][2] if have else float('nan'):.1f} vs Vicon {mT['vicon']['rows'][2] if have else float('nan'):.1f} mm in TDB, {mA['markerless']['rows'][2] if have else float('nan'):.1f} vs {mA['vicon']['rows'][2] if have else float('nan'):.1f} mm in ADB). That is why markerless B is a little high in TDB. The fix is more texture or more cameras on the abdomen (or a model that interpolates it), not a different tracker.</li>
<li class="warn">The ball positions are the reference grid for the virtual markers and the balls are painted out using their Vicon positions; a patient has no balls, and the equivalent would be a grid defined on the mesh.</li>
<li>Free breathing (FB) is not in the exported CSVs and DB has no video here, so S for FB/DB is Vicon-only.</li></ul>
<h2>4 &middot; ADB geometry (raw video)</h2>
<p>The ADB videos exist only as raw wide-angle footage, whereas the calibration was made on Gyroflow-stabilised video. A per-camera lens-correction warp into the stabilised geometry was fitted from the ECal raw/stabilised pairs (fisheye radial model + rotation on SIFT matches, static in time to &lt; 0.1 px), registered to the TDB background (a common ~4 px rotation), and each camera's extrinsics were then re-estimated by PnP on the ADB Vicon balls (RMS ~2 px). Running Gyroflow on the ADB files with the TDB settings would remove the approximation.</p>
{table(["Camera", "balls used", "PnP RMS (px)", "warp fit residual (px)", "rotation vs TDB extrinsics (deg)", "camera centre moved vs TDB (mm)"], geo, num=(1, 2, 3, 4, 5))}
<p class="meta">Cameras were not necessarily moved between trials: part of the change is the common offset of the Vicon-to-camera registration; the ADB extrinsics are used for ADB only. Marker projections on the warped frames land on the balls in all five cameras (checked by eye).</p>
<h2>Reproduce</h2><pre>python thoracoabdominal_vicon.py                                          # cohort, Vicon only
python adb_geometry.py fit ; python adb_geometry.py align ; python adb_extrinsics.py     # ADB geometry
set TRIAL=ADB &amp; python mv_collect.py ; python mv_fit.py                              # SAM3D env: first-frame mesh used for the seeds
python chest_points_dense.py 8000 0.10 points_dense_wide ; run_lk_trial.sh ADB points_dense_wide _wide ; run_lk_trial.sh TDB points_dense_wide _wide
set TRIAL=ADB &amp; python chest_eval.py 0.5 m41mv_e33_wide     # (MINFRAC=0.25 MINCAM=2 for the primary rule)
python thoracoabdominal_markerless.py ADB m41mv_e33_wide_f25_c2 ; python thoracoabdominal_markerless.py TDB m41mv_e33_wide_f25_c2</pre>"""
    page("18_adb_vs_tdb", "18 &middot; Abdominal vs thoracic deep breathing (ADB vs TDB)", "MMC17 &middot; skin tracking (video only) vs the Vicon chest array, and the draft's Vicon-only cohort analysis", body)
