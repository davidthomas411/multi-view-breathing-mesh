"""Vicon-only thoracoabdominal analysis after the draft paper (ASTRO_markers_paper_draft): peak-to-peak anterior-posterior (Vicon Z) amplitude of the 16 anterior markers (4 rows x 4),
thoracic = rows 1-2, abdominal = rows 3-4, regional bias B = mean(thoracic amplitudes) / mean(abdominal amplitudes), per participant and maneuver (ADB, DB, TDB; the exported CSVs have no FB).
Signed reference-distance score for DB:  S = |B_DB - B_ADB| - |B_DB - B_TDB|  (S > 0: closer to TDB).
Writes tmp/thoracoabdominal/vicon_summary.json and vicon_amplitudes.npz.   python thoracoabdominal_vicon.py"""
import csv, glob, json, os, re
import numpy as np
from scipy.signal import savgol_filter
from scipy import stats
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); TMP = ROOT + "/tmp"; OUT = TMP + "/thoracoabdominal"; os.makedirs(OUT, exist_ok=True)
D = ROOT + "/CUTrial/Trials_marker_positions_by_image_frame_all_cases"; CHEST = re.compile(r"([LR])(\d)(\d)")


def load(subject, trial, window_only=True):
    p = f"{D}/{subject}_{trial}_marker_positions_by_image_frame_wide.csv"
    if not os.path.exists(p): return None
    rows = list(csv.reader(open(p))); names = [c[:-2] for c in rows[0][10:] if c.endswith("_X")]; body = rows[1:]
    if window_only: body = [r for r in body if r[9] == "True"]
    X = np.array([[float(v) if v else np.nan for v in r[10:]] for r in body]).reshape(len(body), -1, 3); frames = np.array([int(r[2]) for r in body]); return names, X, frames


def amplitudes(subject, trial, smooth=(5, 2)):
    """per-marker peak-to-peak of the smoothed anterior-posterior (Vicon Z) coordinate; returns {(row, col_label): mm} and the 4x? marker names."""
    d = load(subject, trial)
    if d is None: return None
    names, X, frames = d; amp = {}
    for j, n in enumerate(names):
        m = CHEST.fullmatch(n)
        if not m: continue
        z = X[:, j, 2]; ok = ~np.isnan(z)
        if ok.sum() < 30: continue
        zi = np.interp(np.arange(len(z)), np.where(ok)[0], z[ok]); zs = savgol_filter(zi, smooth[0], smooth[1]); amp[n] = (int(m[2]), float(zs.max() - zs.min()))
    return amp


def regional(amp):
    thor = [a for r, a in amp.values() if r in (1, 2) and a > 0]; abd = [a for r, a in amp.values() if r in (3, 4) and a > 0]
    if len(thor) < 7 or len(abd) < 7: return None
    rows = [np.mean([a for r, a in amp.values() if r == k and a > 0]) for k in (1, 2, 3, 4)]
    return dict(B=float(np.mean(thor) / np.mean(abd)), thoracic=float(np.mean(thor)), abdominal=float(np.mean(abd)), rows=[float(x) for x in rows], mean_amp=float(np.mean([a for _, a in amp.values()])))


def holm(p):
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(o): run = max(run, (m - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


if __name__ == "__main__":
    subjects = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(D + "/MMC*_*_wide.csv")}); R = {}
    for s in subjects:
        r = {t: (regional(a) if (a := amplitudes(s, t)) else None) for t in ("ADB", "DB", "TDB")}
        if all(r[t] for t in r): R[s] = r
    print("participants with ADB, DB and TDB:", len(R), sorted(R))
    B = {t: np.array([R[s][t]["B"] for s in R]) for t in ("ADB", "DB", "TDB")}; rows = {t: np.array([R[s][t]["rows"] for s in R]) for t in B}
    for t in B: print(f"  B {t}: {B[t].mean():.3f} +- {B[t].std(ddof=1):.3f}   regional means (rows 1-4): {np.round(rows[t].mean(0), 1)}   overall mean amplitude {np.mean([R[s][t]['mean_amp'] for s in R]):.1f} mm")
    fr = stats.friedmanchisquare(B["ADB"], B["DB"], B["TDB"]); pw = [("ADB vs DB", stats.wilcoxon(B["ADB"], B["DB"]).pvalue), ("ADB vs TDB", stats.wilcoxon(B["ADB"], B["TDB"]).pvalue), ("DB vs TDB", stats.wilcoxon(B["DB"], B["TDB"]).pvalue)]
    adj = holm(np.array([p for _, p in pw])); print(f"  Friedman chi2(2) = {fr.statistic:.2f}, p = {fr.pvalue:.2g}; Holm-adjusted Wilcoxon: " + ", ".join(f"{n} {a:.3g}" for (n, _), a in zip(pw, adj)))
    S = np.abs(B["DB"] - B["ADB"]) - np.abs(B["DB"] - B["TDB"]); t1 = stats.ttest_1samp(S, 0)
    print(f"  DB score S: mean {S.mean():.3f} +- {S.std(ddof=1):.3f}; one-sample t({len(S) - 1}) = {t1.statistic:.2f}, p = {t1.pvalue:.3g}; {int((S > 0).sum())}/{len(S)} closer to TDB")
    order = int(np.sum(B["TDB"] < B["ADB"])); print(f"  ADB/TDB order reversed (B_TDB < B_ADB) in {order} participants")
    m = R.get("MMC17"); print("  MMC17:", {t: dict(B=round(m[t]['B'], 3), rows=np.round(m[t]['rows'], 1).tolist()) for t in m} if m else None)
    json.dump(dict(subjects=list(R), B={t: B[t].tolist() for t in B}, rows={t: rows[t].tolist() for t in rows}, S=S.tolist(), friedman=dict(chi2=float(fr.statistic), p=float(fr.pvalue)), pairwise=[dict(pair=n, p=float(p), p_holm=float(a)) for (n, p), a in zip(pw, adj)],
                   S_test=dict(mean=float(S.mean()), sd=float(S.std(ddof=1)), t=float(t1.statistic), p=float(t1.pvalue), n_tdb=int((S > 0).sum())), reversed_order=order, per_subject=R), open(OUT + "/vicon_summary.json", "w"), indent=1)
