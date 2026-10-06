"""Marker labelling for the supine MMC trials, from the MMC data collection sheet (v3) + the paper's marker figure.

Supine set (sheet): posterior markers (LPSIS RPSIS BPV T10 C7 RBAK) and the sternal notch are REMOVED; a 4x4 chest array is added.
Vicon names most pelvis/leg/foot/chest-array markers but leaves the head/arm markers as unlabelled '*NN' trajectories.  This module names those
from geometry: body-fixed frame (lateral R->L from the ASIS pair, superior pelvis->chest, anterior = up when supine), then
  head   : the two unlabelled markers at head level on either side of the midline            -> RAHD / LAHD (headband)
  per arm: wrist pair = closest pair far from the shoulder; finger = beyond the wrist pair; forearm = nearest to the wrist pair;
           elbow pair = the two nearest the forearm marker among the rest; remaining = upper-arm pair.
Within-pair assignments (radial/ulnar, lateral/medial, anterior/posterior) are weak and flagged as such.

Priority classes from the paper figure / SOP notes: red = high tracking priority, green = tracked less (soft-tissue artefact), blue = rarely visible.
Per the SOP, markers are treated as independent points (not segments)."""
import numpy as np

# ---- priority classes (paper Fig. 2b colours) -> fit weights
BLUE = {"RMELB", "LMELB", "RMKNE", "LMKNE", "RMML", "LMML"}                                     # medial markers, rarely visible
GREEN = {"RGRT", "LGRT", "RTTUB", "LTTUB", "RTOE", "LTOE", "RICAL", "LICAL"} | {f"{s}{c}" for s in "RL" for c in ("SATH", "IATH", "SPTH", "IPTH", "SASK", "IASK", "SPSK", "IPSK")}
WEIGHT = {"red": 1.0, "green": 0.4, "blue": 0.15}
def priority(name):
    n = name.split(":")[0]
    return "blue" if n in BLUE else "green" if n in GREEN else "red"
def weight(name): return WEIGHT[priority(name)]


def body_frame(P, names):
    g = lambda n: P[names.index(n)]
    pel = (g("RASIS") + g("LASIS")) / 2; chest = np.nanmean([g(n) for n in ("R11", "L11", "R12", "L12")], 0)
    s = chest - pel; s /= np.linalg.norm(s); l = g("LASIS") - g("RASIS"); l -= l @ s * s; l /= np.linalg.norm(l); a = np.cross(l, s)
    if a[2] < 0: a = -a
    return pel, chest, np.array([l, s, a])           # rows: lateral (R->L), superior, anterior


def label_unlabelled(frames_P, names):
    """frames_P: list of (n_markers,3) arrays (mm, Vicon frame) from several steady frames.  Returns {'*NN': dict(label, side, segment, confidence)}."""
    star = [i for i, n in enumerate(names) if n.startswith("*")]
    Pm = np.nanmedian(np.array(frames_P), axis=0); pel, chest, B = body_frame(Pm, names)
    rel = {names[i]: B @ (Pm[i] - pel) for i in star if not np.isnan(Pm[i]).any()}           # (lateral, superior, anterior)
    out = {}; chest_s = (B @ (chest - pel))[1]
    # ---- head: markers beyond the chest by > 150 mm and within 120 mm of the midline
    head = {k: v for k, v in rel.items() if v[1] > chest_s + 120 and abs(v[0]) < 120 and v[1] < chest_s + 400}
    for k, v in head.items():
        side = "R" if v[0] < 0 else "L"
        out[k] = dict(label=f"{side}AHD", side=side, segment="head (headband)", confidence="medium" if len(head) == 2 else "low")
    arm_ms = [k for k in rel if k not in head]
    for side, sign in (("R", -1), ("L", 1)):
        ms = [k for k in arm_ms if np.sign(rel[k][0]) == sign]
        if len(ms) < 3: continue
        pos = {k: Pm[names.index(k)] for k in ms}; D = lambda a, b: np.linalg.norm(pos[a] - pos[b])
        shoulder = Pm[names.index(("R" if side == "R" else "L") + "12")]
        far = sorted(ms, key=lambda k: -np.linalg.norm(pos[k] - shoulder))
        # wrist pair: closest pair among the farthest 4
        cand = far[:4]; pair = min(((a, b) for i, a in enumerate(cand) for b in cand[i + 1:]), key=lambda ab: D(*ab))
        rest = [k for k in ms if k not in pair]
        finger = max(far[:4], key=lambda k: np.linalg.norm(pos[k] - shoulder)) if far[0] not in pair else far[0]
        if finger in pair: finger = next(k for k in far if k not in pair)
        rest = [k for k in rest if k != finger]
        wc = (pos[pair[0]] + pos[pair[1]]) / 2
        forearm = min(rest, key=lambda k: abs(np.linalg.norm(pos[k] - wc) - 130)) if rest else None
        rest = [k for k in rest if k != forearm]
        by_fore = sorted(rest, key=lambda k: np.linalg.norm(pos[k] - pos[forearm])) if forearm else rest
        elbow, upper = by_fore[:2], by_fore[2:]
        out[finger] = dict(label=f"{side}FIN", side=side, segment="hand/finger", confidence="medium")
        for k, nm in zip(sorted(pair, key=lambda k: rel[k][0] * sign), ("RAD", "ULN")): out[k] = dict(label=f"{side}{nm}", side=side, segment="wrist", confidence="low (radial/ulnar swap possible)")
        if forearm: out[forearm] = dict(label=f"{side}FRA", side=side, segment="forearm", confidence="medium")
        for k, nm in zip(sorted(elbow, key=lambda k: -abs(rel[k][0])), ("ELB", "MELB")): out[k] = dict(label=f"{side}{nm}", side=side, segment="elbow", confidence="low (lateral/medial swap possible)")
        for k, nm in zip(sorted(upper, key=lambda k: -rel[k][2]), ("AUPA", "PUPA")): out[k] = dict(label=f"{side}{nm}", side=side, segment="upper arm", confidence="low (anterior/posterior swap possible)")
    return out
