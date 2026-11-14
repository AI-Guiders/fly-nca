"""Interactive 3D WebGL (Three.js) Colony & Connectome Viewer for Fly-NCA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import torch

from fly_nca.genome.spawner import FlySpawner, MultiFlySocialArena3D


def export_colony_3d_viewer_html(
    output_path: Path | str = "colony_3d_viewer.html",
    *,
    n_founders: int = 4,
    frames: int = 16,
    seed: int = 108,
    device: str | None = None,
) -> dict[str, Any]:
    """Run a multi-generational 3D colony simulation and export a self-contained Three.js 3D HTML viewer."""
    spawner = FlySpawner(seed=seed, device=device)
    flies = spawner.spawn_population(n_flies=n_founders)
    # Place founders close to the central yeast food cup so courtship & oviposition begin promptly
    for idx, f in enumerate(flies):
        ang = 2.0 * np.pi * idx / max(1, len(flies))
        f.arena.pos_xyz[0] = float(0.0022 * np.cos(ang))
        f.arena.pos_xyz[1] = float(0.0022 * np.sin(ang))

    colony = MultiFlySocialArena3D(flies=flies, spawner=spawner)
    history_frames: list[list[dict[str, Any]]] = []
    for _ in range(max(1, int(frames))):
        snaps = colony.step_population(frames=1, enable_ontogeny_and_reproduction=True)
        history_frames.append(snaps)

    # Sample 4,000 representative 3D soma coordinates [X, Y, Z] in micrometers from the 189,766-cell substrate
    sub = flies[0].full_connectome.substrate
    soma_um = sub.soma_xyz_um.cpu().numpy()
    organ_ids = sub.organ_id.cpu().numpy()
    valid_mask = np.linalg.norm(soma_um, axis=1) > 1.0
    valid_indices = np.flatnonzero(valid_mask)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_indices, size=min(3500, len(valid_indices)), replace=False)
    pts = soma_um[chosen]
    center = pts.mean(axis=0)
    pts_centered = (pts - center) * 0.01  # scaled for 3D scene
    soma_points = [
        {
            "x": round(float(pts_centered[k, 0]), 3),
            "y": round(float(pts_centered[k, 2]), 3),
            "z": round(float(pts_centered[k, 1]), 3),
            "organ": int(organ_ids[chosen[k]]),
        }
        for k in range(len(chosen))
    ]

    data_json = json.dumps(
        {
            "n_total_cells": sub.n_total,
            "n_total_edges": sub.n_edges,
            "tau_fast_ms": 1.711,
            "tau_frame_ms": 4.038,
            "frames": history_frames,
            "events": colony.colony_events,
            "soma_cloud": soma_points,
        },
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Fly-NCA 3D Colony & 189,766-Cell Connectome Viewer</title>
<style>
  body {{ margin: 0; background: #0b0f17; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; overflow: hidden; }}
  #hud {{ position: absolute; top: 14px; left: 14px; width: 420px; background: rgba(15, 23, 42, 0.88); border: 1px solid #334155; border-radius: 10px; padding: 14px; z-index: 10; box-shadow: 0 8px 24px rgba(0,0,0,0.5); }}
  #events {{ position: absolute; top: 14px; right: 14px; width: 380px; max-height: 85vh; overflow-y: auto; background: rgba(15, 23, 42, 0.88); border: 1px solid #334155; border-radius: 10px; padding: 14px; z-index: 10; }}
  h1 {{ font-size: 16px; margin: 0 0 6px 0; color: #38bdf8; }}
  .sub {{ font-size: 12px; color: #94a3b8; margin-bottom: 10px; }}
  .row {{ display: flex; justify-content: space-between; font-size: 12px; padding: 4px 0; border-bottom: 1px solid #1e293b; }}
  .badge {{ padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px; }}
  .controls {{ margin-top: 10px; display: flex; gap: 8px; align-items: center; }}
  button {{ background: #0284c7; color: white; border: none; padding: 6px 12px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
  button:hover {{ background: #0369a1; }}
  input[type=range] {{ flex: 1; }}
  .ev-item {{ font-size: 12px; padding: 6px; margin-bottom: 6px; background: #1e293b; border-left: 3px solid #38bdf8; border-radius: 4px; }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<div id="hud">
  <h1>Fly-NCA: 3D Colony & Ontogeny</h1>
  <div class="sub">189,766 Cells · 6,853,423 5-Port Synapses · &tau;<sub>fast</sub>=1.711 ms · &tau;<sub>frame</sub>=4.038 ms</div>
  <div id="frame-info" class="sub">Frame 1</div>
  <div id="fly-list"></div>
  <div class="controls">
    <button id="play-btn">Pause</button>
    <input id="frame-slider" type="range" min="0" max="{len(history_frames) - 1}" value="0"/>
  </div>
</div>
<div id="events">
  <h1>Ontogenetic & Social Events</h1>
  <div class="sub">Egg (Syncytium) &rarr; Larva (Wave 1) &rarr; Pupa (20E Wave 2) &rarr; Adult</div>
  <div id="ev-list"></div>
</div>
<script>
const DATA = {data_json};
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0f17);
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(0, 12, 16);
camera.lookAt(0, 1, 0);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

const amb = new THREE.AmbientLight(0xffffff, 0.7);
scene.add(amb);
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(10, 20, 10);
scene.add(dir);

// 3D Arena floor + Yeast Food Cup at center (0, 0, 0)
const arenaGeo = new THREE.CylinderGeometry(9, 9, 0.2, 48);
const arenaMat = new THREE.MeshStandardMaterial({{ color: 0x1e293b, roughness: 0.8 }});
const arenaMesh = new THREE.Mesh(arenaGeo, arenaMat);
arenaMesh.position.y = -0.1;
scene.add(arenaMesh);

const cupGeo = new THREE.CylinderGeometry(2.2, 2.2, 0.25, 32);
const cupMat = new THREE.MeshStandardMaterial({{ color: 0xf59e0b, transparent: true, opacity: 0.65 }});
const cupMesh = new THREE.Mesh(cupGeo, cupMat);
cupMesh.position.y = 0.05;
scene.add(cupMesh);

// Floating 3D Soma Point Cloud of the 189,766-Cell Connectome above the arena
const somaGeo = new THREE.BufferGeometry();
const posArr = new Float32Array(DATA.soma_cloud.length * 3);
const colArr = new Float32Array(DATA.soma_cloud.length * 3);
const organPalette = [
  [0.22, 0.74, 0.97], [0.65, 0.54, 0.98], [0.96, 0.45, 0.71],
  [0.98, 0.75, 0.14], [0.20, 0.83, 0.60], [0.97, 0.44, 0.44],
  [0.38, 0.65, 0.98], [0.99, 0.57, 0.24], [0.58, 0.64, 0.72]
];
DATA.soma_cloud.forEach((p, i) => {{
  posArr[i*3] = p.x * 0.8;
  posArr[i*3+1] = 5.5 + p.y * 0.8;
  posArr[i*3+2] = p.z * 0.8;
  const c = organPalette[p.organ % organPalette.length];
  colArr[i*3] = c[0]; colArr[i*3+1] = c[1]; colArr[i*3+2] = c[2];
}});
somaGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
somaGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
const somaMat = new THREE.PointsMaterial({{ size: 0.12, vertexColors: true, transparent: true, opacity: 0.75 }});
const somaCloud = new THREE.Points(somaGeo, somaMat);
scene.add(somaCloud);

// Populate event log
const evList = document.getElementById('ev-list');
DATA.events.forEach(ev => {{
  const d = document.createElement('div');
  d.className = 'ev-item';
  d.innerHTML = `<b>[${{ev.time_ms}} ms] ${{ev.event}}</b><br/>${{JSON.stringify(ev)}}`;
  evList.appendChild(d);
}});

const flyMeshes = {{}};
function updateFrame(fIdx) {{
  const snaps = DATA.frames[fIdx] || [];
  document.getElementById('frame-info').innerText = `Frame ${{fIdx + 1}} / ${{DATA.frames.length}} (t = ${{((fIdx+1)*DATA.tau_frame_ms).toFixed(2)}} ms = ${{fIdx+1}} wingbeats)`;
  const listEl = document.getElementById('fly-list');
  listEl.innerHTML = '';

  snaps.forEach(s => {{
    let m = flyMeshes[s.fly_id];
    if (!m) {{
      const g = new THREE.SphereGeometry(0.45, 20, 20);
      const mat = new THREE.MeshStandardMaterial({{ color: 0x38bdf8 }});
      m = new THREE.Mesh(g, mat);
      scene.add(m);
      flyMeshes[s.fly_id] = m;
    }}
    // Convert meters to arena scene units (1 mm = 1 unit)
    m.position.set(s.pos_xyz[0] * 1000.0, 0.35 + s.pos_xyz[2] * 1000.0, s.pos_xyz[1] * 1000.0);
    if (s.life_stage === 'egg') {{
      m.scale.set(0.4, 0.25, 0.25);
      m.material.color.setHex(0xfef08a);
    }} else if (s.life_stage === 'larva') {{
      m.scale.set(0.8, 0.35, 0.4);
      m.material.color.setHex(0xa3e635);
    }} else if (s.life_stage === 'pupa') {{
      m.scale.set(0.7, 0.5, 0.45);
      m.material.color.setHex(0xd97706);
    }} else {{
      m.scale.set(1.0, 0.8, 1.3);
      m.material.color.setHex(s.sex === 'F' ? 0xf472b6 : 0x38bdf8);
    }}

    const row = document.createElement('div');
    row.className = 'row';
    const col = s.sex === 'F' ? '#f472b6' : '#38bdf8';
    row.innerHTML = `<span style="color:${{col}};font-weight:600">${{s.fly_id}}</span>
      <span>${{s.life_stage.toUpperCase()}} (${{(s.stage_progress*100).toFixed(0)}}%)</span>
      <span class="badge" style="background:#334155">${{s.behavior}}</span>`;
    listEl.appendChild(row);
  }});
}}

let curFrame = 0;
let playing = true;
const slider = document.getElementById('frame-slider');
const playBtn = document.getElementById('play-btn');
slider.addEventListener('input', e => {{ curFrame = parseInt(e.target.value); updateFrame(curFrame); }});
playBtn.addEventListener('click', () => {{ playing = !playing; playBtn.innerText = playing ? 'Pause' : 'Play'; }});

let tick = 0;
function animate() {{
  requestAnimationFrame(animate);
  somaCloud.rotation.y += 0.004;
  tick++;
  if (playing && tick % 25 === 0) {{
    curFrame = (curFrame + 1) % DATA.frames.length;
    slider.value = curFrame;
    updateFrame(curFrame);
  }}
  renderer.render(scene, camera);
}}
updateFrame(0);
animate();
</script>
</body>
</html>
"""
    out_p = Path(output_path)
    out_p.write_text(html, encoding="utf-8")
    return {
        "html_path": str(out_p.resolve()),
        "n_founders": n_founders,
        "n_final_colony": len(colony.flies),
        "n_events": len(colony.colony_events),
        "events": colony.colony_events,
    }
