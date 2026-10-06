"""Report 14: the marker-guided tracking recording (Rerun file + HTML scrubber) and a plain-language explanation of what the fit is and what it gives."""
import json
import numpy as np
import cv2
import base64


def _img(path, width=1400, q=62):
    im = cv2.imread(path)
    if im is None: return ""
    s = width / im.shape[1]; im = cv2.resize(im, None, fx=min(s, 1), fy=min(s, 1), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q]); return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


JS = """(function(){var I=%(imgs)s,L=%(info)s,S=%(series)s,r=document.getElementById("tk_rng"),m=document.getElementById("tk_img"),l=document.getElementById("tk_lab");
function plot(id,series,cols,names,unit,ymin,ymax){var c=document.getElementById(id),g=c.getContext("2d"),W=c.width,H=c.height,p=30,fg=getComputedStyle(document.body).color;g.clearRect(0,0,W,H);g.font="12px system-ui";g.fillStyle=fg;
 var n=S.frames.length;function X(k){return 46+(W-56)*k/(n-1)}function Y(v){return H-p-(H-p-10)*(v-ymin)/(ymax-ymin)}
 g.strokeStyle="rgba(128,128,128,.35)";g.lineWidth=1;var st=Math.ceil((ymax-ymin)/4/5)*5||1;for(var v=Math.ceil(ymin/st)*st;v<=ymax;v+=st){g.beginPath();g.moveTo(46,Y(v));g.lineTo(W-6,Y(v));g.stroke();g.fillText(v,6,Y(v)+4);}
 series.forEach(function(s,i){g.strokeStyle=cols[i];g.lineWidth=2;g.beginPath();var on=false;s.forEach(function(v,k){if(v==null||v<ymin||v>ymax){on=false;return}if(!on){g.moveTo(X(k),Y(v));on=true}else g.lineTo(X(k),Y(v))});g.stroke();g.fillStyle=cols[i];g.fillText(names[i],56+i*200,16);});
 var k=+r.value;g.lineWidth=1.5;g.strokeStyle=fg;g.beginPath();g.moveTo(X(k),6);g.lineTo(X(k),H-p);g.stroke();g.fillStyle=fg;g.fillText(unit,6,12);g.fillText("t = "+(S.frames[k]/29.97).toFixed(1)+" s",Math.min(X(k)+4,W-70),H-p+16);}
function draw(){plot("tk_c1",[S.d_fit,S.d_base],["#16a34a","#ef4444"],["marker-guided fit","SAM 3D Body alone"],"mm",0,100);plot("tk_c2",[S.chest_mk,S.chest_fit,S.chest_base],["#0ea5e9","#16a34a","#ef4444"],["Vicon chest markers","fitted mesh","SAM 3D Body mesh"],"mm",-25,25);}
r.addEventListener("input",function(){m.src=I[r.value];l.innerHTML=L[r.value];draw()});draw();})();"""


def build(ctx):
    TMP, kpis, table, fig, page = (ctx[k] for k in ("TMP", "kpis", "table", "fig", "page"))
    meta = json.load(open(f"{TMP}/track/overlay/meta.json")); n = len(meta)
    T = np.load(f"{TMP}/track/track.npz", allow_pickle=True)
    def detr(v):
        v = np.asarray(v, float); t = np.arange(len(v)); ok = ~np.isnan(v); return v - np.polyval(np.polyfit(t[ok], v[ok], 2), t)
    cm, cf, cb = (detr(np.array([m[k] if m[k] is not None else np.nan for m in meta]) ) for k in ("chest_mk", "chest_fit", "chest_base"))
    clean = lambda v: [None if np.isnan(x) else round(float(x) * 1000, 2) for x in v]
    series = dict(frames=[m["frame"] for m in meta], d_fit=[round(m["d_fit"], 1) for m in meta], d_base=[None if np.isnan(m["d_base"]) else round(m["d_base"], 1) for m in meta], chest_mk=clean(cm), chest_fit=clean(cf), chest_base=clean(cb))
    imgs = [_img(f"{TMP}/track/overlay/{m['frame']:04d}.jpg") for m in meta]
    info = [f"frame {m['frame']} (t = {m['frame'] / 29.97:.1f} s) &middot; marker&rarr;mesh: fitted {m['d_fit']:.0f} mm, SAM 3D Body {m['d_base']:.0f} mm" for m in meta]
    d_fit, d_base = T["d_fit"], T["d_base"]
    js = JS % dict(imgs=json.dumps(imgs), info=json.dumps(info), series=json.dumps(series))
    body = f"""
<div class="card"><b>Question.</b> What does the marker-guided fit actually look like over time, and how has it helped?</div>
{kpis([("%.1f mm" % np.median(d_fit), "median marker-to-mesh distance, fitted mesh (%d frames, range %.0f&ndash;%.0f mm)" % (len(d_fit), d_fit.min(), d_fit.max())), ("%.1f mm" % np.nanmedian(d_base), "SAM 3D Body alone (range %.0f&ndash;%.0f mm)" % (np.nanmin(d_base), np.nanmax(d_base))), ("291 frames", "37 s of the thoracic deep-breathing trial, every 3rd frame"), ("~5.5 s / frame", "offline cost of the fit (needs the Vicon markers)")])}
<h2>What the fit is, in plain terms</h2>
<ul><li><b>The problem.</b> A markerless model such as SAM 3D Body proposes a body mesh from the video, but on this supine, arms-up pose it is typically 10&ndash;30 cm from where the body really is, and its chest surface jitters by ~4 mm from frame to frame (with the odd frame 100+ mm off).</li>
<li><b>The anchor.</b> The Vicon markers are millimetre-accurate 3D points on the skin, with known identity. Here they are the ground truth.</li>
<li><b>The fit.</b> Per frame, the mesh is bent and moved so that the surface point attached to each marker sits on that marker (with a weak pull towards SAM 3D Body's shape so it stays a plausible body). The attachments are fixed across frames, so the same mesh point follows the same marker through the whole trial.</li>
<li><b>What it gives.</b> A full body mesh in the Vicon frame, correct to about 1 cm at the markers (the floor is 8 mm, the ball radius), for every frame. That is a <b>reference</b> for judging a markerless method and <b>training labels</b> for a better one, and a clean chest surface for breathing studies.</li>
<li><b>What it does not do.</b> It does not make SAM 3D Body better by itself: without Vicon you do not get this. And it is only constrained where there are markers (torso, pelvis, legs, feet, plus the unlabelled arm/head markers): the hands and arms are the loosest part.</li></ul>
<h2>Recording</h2>
<figure><img id="tk_img" src="{imgs[0]}" style="width:100%">
<div style="display:flex;gap:10px;align-items:center;margin-top:6px"><input id="tk_rng" type="range" min="0" max="{n - 1}" value="0" style="flex:1"><span id="tk_lab" class="meta" style="min-width:420px;text-align:right">{info[0]}</span></div>
<canvas id="tk_c1" width="1100" height="150" style="width:100%;margin-top:8px;border:1px solid var(--line);border-radius:8px"></canvas>
<canvas id="tk_c2" width="1100" height="170" style="width:100%;margin-top:6px;border:1px solid var(--line);border-radius:8px"></canvas>
<figcaption>Five cameras (T1&ndash;T5; T6 excluded, unsynchronised), <span style="color:#4ade80">&#9679;</span> fitted mesh points, <span style="color:#ef4444">&#9679;</span> SAM 3D Body's own mesh (one camera), <span style="color:#22d3ee">&#9711;</span> Vicon markers. Top plot: distance from the markers to each mesh (lower is better). Bottom plot: chest surface height under the chest markers, detrended (the breathing). The fitted mesh follows it because the chest markers are its input &mdash; it is the reference, not a markerless result (report 13 is the markerless measurement). Drag the slider.</figcaption></figure>
<script>{js}</script>
<h2>The interactive 3D recording (Rerun)</h2>
<p>The same run as a Rerun file with a 3D view (fitted mesh, SAM 3D Body's mesh, markers, camera frusta in the Vicon frame), the five camera views with overlays, and live plots. Open it with:</p>
<pre>C:\\dev\\InstantHMR\\.venv\\Scripts\\rerun.exe recordings\\marker_guided_tracking.rrd</pre>
<p class="meta">264 MB, 146 frames (every 6th video frame), Rerun 0.38.</p>
<h2>How it was made</h2>
<ol><li>First frame (250): SAM 3D Body on the five cameras, start from the camera closest to the markers, full three-stage fit (rigid correction, then pose, then shape/scale) with the named, weighted markers (reports 7&ndash;9).</li>
<li>Every later frame: warm start from the previous frame, 60 optimiser steps, markers kept on the same mesh points; pose change penalised for smoothness.</li>
<li>Baseline: SAM 3D Body on the start camera each frame, moved into the Vicon frame with the marker-based extrinsics.</li></ol>
<h2>Limits</h2>
<ul><li>Hands and arms: the green points bunch at the hands in some views; the arm markers are unlabelled in the files and only loosely constrain them.</li>
<li>Vertex count and mesh resolution are those of Fast-SAM-3D-Body (18,439 vertices).</li>
<li>Single trial; offline (~5.5 s per frame on the RTX 4060, shared with other jobs).</li></ul>
<h2>Reproduce</h2><pre>C:\\dev\\sam3d-venv\\Scripts\\python.exe track_fit.py                 # the tracking run (tmp\\track\\track.npz)
C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe make_tracking_rrd.py     # Rerun recording
C:\\dev\\InstantHMR\\.venv\\Scripts\\python.exe make_track_overlays.py   # mosaics for this page</pre>"""
    page("14_tracking_recording", "14 &middot; Marker-guided tracking: the recording", "MMC17 TDB &middot; fitted mesh vs SAM 3D Body vs Vicon, 5 cameras", body)
