"""Report 6: MMC17 SAM 3D Body multi-view tracking (frame scrubber + fit-quality plot).  Called from make_reports.py."""
import json
import numpy as np

SCRUB_JS = """(function(){var I=%(imgs)s,L=%(info)s,S=%(series)s,r=document.getElementById("m17_rng"),m=document.getElementById("m17_img"),l=document.getElementById("m17_lab"),c=document.getElementById("m17_cv"),g=c.getContext("2d");
function draw(){var W=c.width,H=c.height,p=34,mx=1,i;for(i=0;i<S.rep.length;i++){if(S.rep[i]!=null)mx=Math.max(mx,S.rep[i]);if(S.t6[i]!=null)mx=Math.max(mx,S.t6[i]);}mx=Math.ceil(mx/50)*50;
var fg=getComputedStyle(document.body).color;g.clearRect(0,0,W,H);g.font="12px system-ui";g.fillStyle=fg;
function X(k){return 40+(W-50)*k/(S.frames.length-1)}function Y(v){return H-p-(H-p-10)*v/mx}
g.strokeStyle="rgba(128,128,128,.35)";g.lineWidth=1;for(i=0;i<=mx;i+=50){var y=Y(i);g.beginPath();g.moveTo(40,y);g.lineTo(W-6,y);g.stroke();g.fillText(i,6,y+4);}
[["rep","#16a34a"],["t6","#ea580c"]].forEach(function(s){g.strokeStyle=s[1];g.lineWidth=2;g.beginPath();var st=false;S[s[0]].forEach(function(v,k){if(v==null){st=false;return}if(!st){g.moveTo(X(k),Y(v));st=true}else g.lineTo(X(k),Y(v))});g.stroke();});
var k=+r.value;g.lineWidth=1.5;g.strokeStyle=fg;g.beginPath();g.moveTo(X(k),6);g.lineTo(X(k),H-p);g.stroke();g.fillStyle=fg;g.fillText("frame "+S.frames[k],Math.min(X(k)+4,W-90),H-p+16);
g.fillStyle="#16a34a";g.fillText("fused fit",W-190,16);g.fillStyle="#ea580c";g.fillText("T6 held out",W-120,16);g.fillStyle=fg;g.fillText("px @4K",6,12);}
r.addEventListener("input",function(){m.src=I[r.value];l.innerHTML=L[r.value];draw()});draw();})();"""


def build(ctx):
    TMP, img_b64, kpis, table, page = ctx["TMP"], ctx["img_b64"], ctx["kpis"], ctx["table"], ctx["page"]
    meta = json.load(open(f"{TMP}/overlays/mmc17_meta.json")); n = len(meta); step = meta[1]["frame"] - meta[0]["frame"]
    imgs = [img_b64(f"{TMP}/overlays/mmc17_{m['frame']:04d}.jpg", 1700, 70) for m in meta]
    rep = [m["reproj"] for m in meta]; t6 = [m["cam_err"][5] for m in meta]
    ce = np.array([[np.nan if v is None else v for v in m["cam_err"]] for m in meta])
    ok = [r for r in rep if r is not None]
    info = [f"frame {m['frame']} (t = {m['frame'] / 29.97:.1f} s) &middot; {m['views']} cameras fused"
            + (f" &middot; fit {m['reproj']:.0f} px" if m["reproj"] is not None else "")
            + (f" &middot; T6 (held out) {m['cam_err'][5]:.0f} px" if m["cam_err"][5] is not None else "") for m in meta]
    series = dict(frames=[m["frame"] for m in meta], rep=rep, t6=t6)
    pcam = {f"T{i + 1}": (float(np.nanmedian(ce[:, i])) if not np.all(np.isnan(ce[:, i])) else None) for i in range(6)}
    js = SCRUB_JS % dict(imgs=json.dumps(imgs), info=json.dumps(info), series=json.dumps(series))
    all5 = sum(1 for m in meta if m["views"] == 5)
    body = f"""
<div class="card"><b>Question.</b> What does SAM 3D Body multi-view tracking of the MMC17 volunteer (supine, arms up, TDB breathing trial) look like once the six tripod cameras are calibrated? Every {step}th frame ({step / 29.97:.2f} s) is shown: {n} frames over {meta[-1]['frame'] / 29.97:.0f} s.</div>
{kpis([("%.0f px" % np.median(ok), "median fit error: fused skeleton vs each camera's own joints (4K px)"), ("%.0f px" % np.nanmedian(t6), "camera T6 (unsynchronized, not fused): shown for reference"), ("%d / %d" % (all5, n), "frames fused from all of T1&ndash;T5"), ("&asymp; 0.8 mm / px", "at ~1.3 m subject distance (f = 1680 px): 150 px &asymp; 12 cm")])}
<h2>Tracking, frame by frame</h2>
<figure><img id="m17_img" src="{imgs[0]}" style="width:100%">
<div style="display:flex;gap:10px;align-items:center;margin-top:6px"><input id="m17_rng" type="range" min="0" max="{n - 1}" value="0" style="flex:1"><span id="m17_lab" class="meta" style="min-width:380px;text-align:right">{info[0]}</span></div>
<canvas id="m17_cv" width="1100" height="170" style="width:100%;margin-top:8px;border:1px solid var(--line);border-radius:8px"></canvas>
<figcaption><span style="color:#4ade80">&#9632;</span> triangulated skeleton from T1&ndash;T5, reprojected into every camera &nbsp; <span style="color:#fbbf24">&#9679;</span> that camera's own SAM 3D Body joints &nbsp; <span style="color:#ef4444">&#9679;</span> own joints more than 120 px from the fused skeleton. T6 is <b>not</b> fused (it is unsynchronized, ~23 fps): its skeleton is a prediction from the other cameras, shown for reference only. Plot: fit error of the fused skeleton (green) and T6's independent error (orange) per frame, cursor at the frame shown. Drag the slider or use the arrow keys.</figcaption></figure>
<script>{js}</script>
<h2>Setup</h2>
<ul><li><b>Cameras.</b> MMC17 tripod cameras T1&ndash;T5 with the marker-resolved extrinsics v3 (<a href="04_mmc17_calibration.html">report 4</a>), shared intrinsics (f = 1680 px, 3840&times;2160, Gyroflow); world frame = checkerboard, assumed 130 mm squares.</li>
<li><b>Per camera.</b> YOLO11m-Pose box &rarr; SAM 3D Body (Fast-SAM-3D-Body checkpoint, <code>cam_int</code> = calibrated K) &rarr; 3D skeleton + <code>cam_t</code>, projected through K to 2D.</li>
<li><b>Fusion.</b> Robust per-joint DLT triangulation of the 30 body joints (fingers excluded); a camera's observation of a joint is dropped when it disagrees by more than 12 px.</li>
<li>No Vicon information is used here; this is vision-only.</li></ul>
<h2>Fit by camera</h2>
{table(["Camera", "median own-joint distance to the fused skeleton (px @4K)", "role"], [[k, ("%.0f" % v) if v is not None else "&ndash;", "fused (in-sample)" if k != "T6" else "<b>held out</b> (independent)"] for k, v in pcam.items()], num=(1,))}
<p class="meta">Leave-one-camera-out over all frames with the v3 calibration: T1 126, T2 105, T3 182, T4 67, T5 86 px (v2 ring-prior alignment: 172, 154, 196, 119, 152). T6 runs at ~23 fps (unsynchronized) and is excluded.</p>
<h2>What to look at</h2>
<ul><li>The start of the trial contains the coached leg tap used for Vicon/camera synchronization, so the pose there is not a resting supine pose.</li>
<li>Whether the skeleton stays on the legs in views where they are foreshortened (T1, T2 at the head end), and whether the arms-up pose is recovered.</li></ul>
<h2>Limits</h2>
<ul><li class="warn">Fit error measures agreement between cameras, not accuracy against Vicon. The Vicon benchmark is in report 7 (the marker-based registration is report 5) (<a href="05_vicon_calibration.html">report 5</a>).</li>
<li>The pixel-to-mm scale is approximate (assumes ~1.3 m subject distance); metric conversions use 120 mm checkerboard squares (SOP: 13 cm; Vicon scale: 120.7 mm).</li>
<li>SAM 3D Body takes ~14 s per 6-camera frame on the 4060 without compilation, so this is offline.</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\sam3d-venv\\Scripts\\python.exe mmc17_sam3d.py 10 1120 10     # per-view SAM 3D Body, cached in tmp\\mmc17_sam3d
C:\\dev\\sam3d-venv\\Scripts\\python.exe make_overlays.py mmc17          # mosaics + per-frame metrics
C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe make_reports.py</pre>"""
    page("06_mmc17_sam3d_tracking", "6 &middot; MMC17: SAM 3D Body multi-view tracking", "TDB trial &middot; 6 tripod cameras (T1&ndash;T5 fused, T6 held out)", body)
