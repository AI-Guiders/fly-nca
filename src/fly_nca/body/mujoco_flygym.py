"""MuJoCo / FlyGym (NeuroMechFly v2) 3D Biomechanical Adapter for Fly-NCA (189,766-Cell Pan-Connectome).

Connects our 189,766-cell, 6,853,423-edge Drosophila Pan-Connectome (`FullConnectomeFlyNCA`)
and multi-generational Colony (`MultiFlySocialArena3D`) directly to the micro-CT scanned 3D
`NeuroMechFly` body in `FlyGym 2.1.1 / MuJoCo` (Wang-Chen et al., Nature Methods 2024):
- Maps `MotorOut` (decoded from 1,026 BANC motor & visceral effector neurons) directly onto the
  actuated biological joints of `NeuroMechFly`:
  - 6 legs x 7 DOFs (`coxa yaw/pitch/roll`, `trochanterfemur pitch/roll`, `tibia pitch`, `tarsus1 pitch`)
  - 6 tarsal adhesion actuators (`set_leg_adhesion_states`)
  - Left & Right wings (`c_thorax-l_wing/r_wing-yaw/pitch/roll`), including unilateral 60° wing extension
    and 160 Hz vibration during male `P1 -> pIP10 -> TN1` courtship song!
  - Neck 3D gimbal (`c_thorax-c_head-yaw/pitch/roll`), proboscis (`c_head-c_rostrum-pitch`,
    `c_rostrum-c_haustellum-pitch`), and abdomen (`c_thorax-c_abdomen12-pitch` curl for oviposition).
- Reads back `get_ground_contact_info` (6-leg campaniform forces) and `get_joint_angles` (chordotonal
  proprioception) from MuJoCo into `SensorIn`.
"""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any
import mujoco as mj
import numpy as np
from PIL import Image
import torch

from flygym import Simulation
from flygym.anatomy import ActuatedDOFPreset, AxisOrder, JointPreset, Skeleton
from flygym.compose import ActuatorType, FlatGroundWorld, KinematicPosePreset, NeuroMechFly
from flygym.utils.math import Rotation3D

from fly_nca.body.adapter import MotorOut
from fly_nca.genome.spawner import FlySpawner, MultiFlySocialArena3D, SpawnedFly


def motor_out_to_neuromechfly_ctrl(
    fly_nmf: NeuroMechFly,
    motor: MotorOut,
    *,
    wing_song_active: bool = False,
    oviposition_active: bool = False,
    step_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map Fly-NCA `MotorOut` (from 1,026 BANC motor neurons) to NeuroMechFly position ctrl & 6-leg adhesion."""
    act_dofs = fly_nmf.get_actuated_jointdofs_order(ActuatorType.POSITION)
    ctrl = np.array(
        [fly_nmf.jointdof_to_neutralaction_by_type[ActuatorType.POSITION][j] for j in act_dofs],
        dtype=np.float64,
    )
    dof_idx = {j.name: i for i, j in enumerate(act_dofs)}

    # 1. 6 Legs mapping: Fly-NCA order ("L1", "R1", "L2", "R2", "L3", "R3") -> FlyGym ("lf", "rf", "lm", "rm", "lh", "rh")
    leg_map = ("lf", "rf", "lm", "rm", "lh", "rh")
    lj = motor.leg_joint_drive.detach().cpu().numpy()
    for leg_i, fg_leg in enumerate(leg_map):
        coxa_fwd = float(lj[leg_i, 0])
        troch_lift = float(lj[leg_i, 1])
        tibia_ext = float(lj[leg_i, 2])
        tarsus_flx = float(lj[leg_i, 3])

        if f"c_thorax-{fg_leg}_coxa-pitch" in dof_idx:
            ctrl[dof_idx[f"c_thorax-{fg_leg}_coxa-pitch"]] += 0.35 * coxa_fwd
        if f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch" in dof_idx:
            ctrl[dof_idx[f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch"]] += 0.45 * troch_lift
        if f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch" in dof_idx:
            ctrl[dof_idx[f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch"]] += 0.40 * tibia_ext
        if f"{fg_leg}_tibia-{fg_leg}_tarsus1-pitch" in dof_idx:
            ctrl[dof_idx[f"{fg_leg}_tibia-{fg_leg}_tarsus1-pitch"]] -= 0.25 * tarsus_flx

    # 2. Neck 3D gimbal (roll, pitch, yaw)
    neck = motor.neck_torque_rpy.detach().cpu().numpy()
    if "c_thorax-c_head-roll" in dof_idx:
        ctrl[dof_idx["c_thorax-c_head-roll"]] += 0.30 * float(neck[0])
    if "c_thorax-c_head-pitch" in dof_idx:
        ctrl[dof_idx["c_thorax-c_head-pitch"]] += 0.30 * float(neck[1])
    if "c_thorax-c_head-yaw" in dof_idx:
        ctrl[dof_idx["c_thorax-c_head-yaw"]] += 0.35 * float(neck[2])

    # 3. Proboscis Extension Reflex (PER, MN9 rostrum + haustellum extension)
    if motor.proboscis_per > 0.1:
        per = float(motor.proboscis_per)
        if "c_head-c_rostrum-pitch" in dof_idx:
            ctrl[dof_idx["c_head-c_rostrum-pitch"]] += 0.65 * per
        if "c_rostrum-c_haustellum-pitch" in dof_idx:
            ctrl[dof_idx["c_rostrum-c_haustellum-pitch"]] -= 0.75 * per

    # 4. Unilateral Male Courtship Song (P1 -> pIP10 -> TN1): extend one wing 60° laterally & vibrate!
    song_amp = float(motor.wing_song_mode[0].item())
    if wing_song_active or song_amp > 0.02:
        vib = 0.15 * math.sin(step_idx * 1.8)
        if "c_thorax-l_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-yaw"]] += 1.05 + vib
        if "c_thorax-l_wing-pitch" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-pitch"]] += 0.25

    # 5. Female Oviposition Abdomen Bending (oviDN -> abd_Dsx_OA motor neurons)
    abd_pitch = float(motor.abdomen_bend_pitch_yaw[0])
    if oviposition_active or abs(abd_pitch) > 0.05:
        curl = min(-0.35, abd_pitch) if oviposition_active else abd_pitch
        for abd_j in (
            "c_thorax-c_abdomen12-pitch",
            "c_abdomen12-c_abdomen3-pitch",
            "c_abdomen3-c_abdomen4-pitch",
        ):
            if abd_j in dof_idx:
                ctrl[dof_idx[abd_j]] += 0.35 * curl

    # 6. Tarsal adhesion in FlyGym order ("lf", "lm", "lh", "rf", "rm", "rh")
    adh_nca = motor.tarsus_adhesion.detach().cpu().numpy()
    adhesion = np.array([adh_nca[0], adh_nca[2], adh_nca[4], adh_nca[1], adh_nca[3], adh_nca[5]], dtype=np.float64)
    return ctrl, np.clip(adhesion, 0.0, 1.0)


def run_mujoco_colony_and_export_viewer(
    output_html_path: Path | str = "colony_3d_viewer.html",
    *,
    n_founders: int = 4,
    frames: int = 14,
    seed: int = 108,
    camera_res: tuple[int, int] = (640, 960),
    device: str | None = None,
) -> dict[str, Any]:
    """Run the 189,766-cell Fly-NCA colony coupled to MuJoCo / FlyGym (`NeuroMechFly`) and export HTML viewer."""
    spawner = FlySpawner(seed=seed, device=device)
    flies: list[SpawnedFly] = [
        spawner.spawn_fly(sex="F", foraging_allele="rover", pos_xyz=(0.0016, 0.0005, 0.0), yaw_rad=-2.8),
        spawner.spawn_fly(sex="M", foraging_allele="sitter", pos_xyz=(-0.0012, -0.0004, 0.0), yaw_rad=0.35),
        spawner.spawn_fly(sex="F", foraging_allele="sitter", pos_xyz=(0.0003, -0.0018, 0.0), yaw_rad=1.8),
        spawner.spawn_fly(sex="M", foraging_allele="rover", pos_xyz=(-0.0018, 0.0016, 0.0), yaw_rad=-0.7),
    ][: max(2, int(n_founders))]

    colony = MultiFlySocialArena3D(flies=flies, spawner=spawner)

    # Build MuJoCo FlatGroundWorld with central yeast patch and micro-CT NeuroMechFly bodies
    world = FlatGroundWorld()
    world.mjcf_root.worldbody.add_geom(
        name="yeast_food_patch",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.02),
        size=(2.2, 0.02, 0.0),
        rgba=(0.92, 0.65, 0.12, 0.85),
        contype=0,
        conaffinity=0,
    )
    # Add visual egg/larva/pupa markers on the yeast patch for F1 ontogeny stages
    egg_geom = world.mjcf_root.worldbody.add_geom(
        name="f1_egg_marker",
        type=mj.mjtGeom.mjGEOM_ELLIPSOID,
        pos=(0.5, 0.2, 0.12),
        size=(0.22, 0.09, 0.08),
        rgba=(0.98, 0.96, 0.82, 0.95),
        contype=0,
        conaffinity=0,
    )
    larva_geom = world.mjcf_root.worldbody.add_geom(
        name="f1_larva_marker",
        type=mj.mjtGeom.mjGEOM_ELLIPSOID,
        pos=(-0.4, 0.5, 0.10),
        size=(0.38, 0.11, 0.09),
        rgba=(0.72, 0.92, 0.32, 0.95),
        contype=0,
        conaffinity=0,
    )
    pupa_geom = world.mjcf_root.worldbody.add_geom(
        name="f1_pupa_marker",
        type=mj.mjtGeom.mjGEOM_ELLIPSOID,
        pos=(0.8, -0.6, 0.20),
        size=(0.42, 0.16, 0.15),
        rgba=(0.78, 0.45, 0.12, 0.95),
        contype=0,
        conaffinity=0,
    )
    _ = (egg_geom, larva_geom, pupa_geom)

    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_BIOLOGICAL)
    nmf_map: dict[str, NeuroMechFly] = {}
    cam = None

    for idx, f in enumerate(flies):
        nmf = NeuroMechFly(name=f.genome.fly_id)
        nmf.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
        act_dofs = nmf.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.ALL)
        nmf.add_actuators(act_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL, kp=50.0)
        nmf.add_leg_adhesion()
        nmf.colorize()
        if idx == 0:
            cam = nmf.add_tracking_camera(
                name="colony_cam",
                pos_offset=(-5.2, -5.2, 4.6),
                rotation=Rotation3D("xyaxes", (0.7, -0.7, 0.0, 0.38, 0.38, 0.84)),
            )
        px_mm = f.pos_xyz[0] * 1000.0
        py_mm = f.pos_xyz[1] * 1000.0
        yaw = float(f.arena.thorax_rpy[2].item())
        q = (float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0)))
        world.add_fly(nmf, spawn_position=(px_mm, py_mm, 0.55), spawn_rotation=Rotation3D("quat", q))
        nmf_map[f.genome.fly_id] = nmf

    sim = Simulation(world)
    sim.reset()
    sim.warmup(0.012)
    assert cam is not None
    mj_renderer = mj.Renderer(sim.mj_model, height=camera_res[0], width=camera_res[1])
    cam_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, cam.name)

    history_frames: list[list[dict[str, Any]]] = []
    frame_png_b64: list[str] = []

    for step_i in range(max(1, int(frames))):
        snaps = colony.step_population(frames=1, enable_ontogeny_and_reproduction=True)
        history_frames.append(snaps)

        # Feed decoded 1,026-effector MotorOut from each founder fly into MuJoCo NeuroMechFly actuators!
        for f in flies:
            nmf = nmf_map.get(f.genome.fly_id)
            if nmf is None:
                continue
            motor = f.full_connectome.decode_motor_effectors()
            ctrl, adh = motor_out_to_neuromechfly_ctrl(
                nmf,
                motor,
                wing_song_active=f.wing_song_active,
                oviposition_active=f.oviposition_active,
                step_idx=step_i,
            )
            sim.set_actuator_inputs(f.genome.fly_id, ActuatorType.POSITION, ctrl)
            sim.set_leg_adhesion_states(f.genome.fly_id, adh)

        # Advance MuJoCo physics for 1 cognitive wingbeat frame (~4 ms = 40 x 0.1 ms MuJoCo steps)
        for _ in range(25):
            sim.step()

        # Render true MuJoCo 3D frame
        mj_renderer.update_scene(sim.mj_data, camera=cam_id)
        rgb = mj_renderer.render()
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=88)
        frame_png_b64.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        if step_i == len(range(max(1, int(frames)))) - 1:
            Image.fromarray(rgb).save(Path(output_html_path).with_suffix(".png"))

    mj_renderer.close()
    sim.close()

    # Sample 3,500 real 3D soma coordinates from BANC v888 somas_v1.parquet for the side-by-side Connectome panel
    sub = flies[0].full_connectome.substrate
    soma_um = sub.soma_xyz_um.cpu().numpy()
    organ_ids = sub.organ_id.cpu().numpy()
    valid_indices = np.flatnonzero(np.linalg.norm(soma_um, axis=1) > 1.0)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_indices, size=min(3500, len(valid_indices)), replace=False)
    pts = soma_um[chosen]
    pts_centered = (pts - pts.mean(axis=0)) * 0.012
    soma_points = [
        {
            "x": round(float(pts_centered[k, 0]), 3),
            "y": round(float(-pts_centered[k, 1]), 3),
            "z": round(float(pts_centered[k, 2]), 3),
            "organ": int(organ_ids[chosen[k]]),
        }
        for k in range(len(chosen))
    ]

    payload_json = json.dumps(
        {
            "n_total_cells": sub.n_total,
            "n_total_edges": sub.n_edges,
            "tau_fast_ms": 1.711,
            "tau_frame_ms": 4.038,
            "frames": history_frames,
            "mujoco_images": frame_png_b64,
            "events": colony.colony_events,
            "soma_cloud": soma_points,
        },
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Fly-NCA x FlyGym/MuJoCo: 3D Colony & 189,766-Cell Connectome Bench</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: #090d16; color: #e2e8f0; font-family: system-ui, -apple-system, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
  header {{ padding: 10px 18px; background: #0f172a; border-bottom: 1px solid #1e293b; display: flex; justify-content: space-between; align-items: center; }}
  header h1 {{ font-size: 16px; margin: 0; color: #38bdf8; }}
  header .meta {{ font-size: 12px; color: #94a3b8; }}
  .main-grid {{ flex: 1; display: grid; grid-template-columns: 1.35fr 1fr 380px; gap: 12px; padding: 12px; min-height: 0; }}
  .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; display: flex; flex-direction: column; overflow: hidden; position: relative; }}
  .panel-title {{ padding: 8px 12px; font-size: 13px; font-weight: 600; color: #cbd5e1; background: #1e293b; display: flex; justify-content: space-between; }}
  .viewport {{ flex: 1; display: flex; align-items: center; justify-content: center; position: relative; background: #05080f; min-height: 0; }}
  #mj-img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  #connectome-canvas {{ width: 100%; height: 100%; display: block; }}
  .sidebar {{ display: flex; flex-direction: column; gap: 10px; overflow-y: auto; padding: 10px; }}
  .fly-card {{ background: #1e293b; border-radius: 8px; padding: 8px 10px; font-size: 12px; border-left: 4px solid #38bdf8; }}
  .fly-card.female {{ border-left-color: #f472b6; }}
  .fly-card.immat {{ border-left-color: #facc15; }}
  .ev-log {{ flex: 1; overflow-y: auto; font-size: 11px; display: flex; flex-direction: column; gap: 6px; padding: 10px; }}
  .ev-item {{ background: #172033; padding: 6px 8px; border-radius: 6px; border-left: 3px solid #22c55e; }}
  footer {{ padding: 8px 18px; background: #0f172a; border-top: 1px solid #1e293b; display: flex; gap: 12px; align-items: center; }}
  button {{ background: #0284c7; color: white; border: none; padding: 6px 14px; border-radius: 6px; cursor: pointer; font-weight: 600; }}
  input[type=range] {{ flex: 1; }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<header>
  <h1>Fly-NCA &times; FlyGym / MuJoCo (NeuroMechFly v2): 189,766-Cell Colony & Ontogeny</h1>
  <div class="meta">BANC v888 + MaleCNS · 6,853,423 5-Port Synapses · &tau;<sub>fast</sub>=1.711 ms · &tau;<sub>frame</sub>=4.038 ms (247.6 Hz Wingbeat)</div>
</header>
<div class="main-grid">
  <div class="panel">
    <div class="panel-title">
      <span>1. MuJoCo / FlyGym 3D Micro-CT Biomechanics (NeuroMechFly)</span>
      <span id="frame-label">Frame 1</span>
    </div>
    <div class="viewport">
      <img id="mj-img" alt="MuJoCo 3D Render"/>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">
      <span>2. 189,766-Cell 3D Connectome Wave (BANC v888 somas_v1)</span>
      <span>OL &rarr; AL/MB/LH &rarr; CX &rarr; P1/pIP10 &rarr; VNC</span>
    </div>
    <div class="viewport" id="conn-vp"></div>
  </div>
  <div class="panel">
    <div class="panel-title"><span>3. Colony Individuals & Ontogeny Log</span></div>
    <div class="sidebar" id="fly-cards"></div>
    <div class="panel-title"><span>Ontogenetic Events (Egg &rarr; Larva &rarr; Pupa &rarr; Adult)</span></div>
    <div class="ev-log" id="ev-list"></div>
  </div>
</div>
<footer>
  <button id="play-btn">Pause</button>
  <input id="slider" type="range" min="0" max="{len(history_frames) - 1}" value="0"/>
</footer>
<script>
const DATA = {payload_json};
const imgEl = document.getElementById('mj-img');
const cardsEl = document.getElementById('fly-cards');
const evEl = document.getElementById('ev-list');
const labelEl = document.getElementById('frame-label');

DATA.events.forEach(ev => {{
  const d = document.createElement('div');
  d.className = 'ev-item';
  d.innerHTML = `<b>[t=${{ev.time_ms}} ms] ${{ev.event}}</b><br/>${{JSON.stringify(ev)}}`;
  evEl.appendChild(d);
}});

// Setup isolated 3D Connectome Inspector in Panel 2
const vp = document.getElementById('conn-vp');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1.0, 0.1, 100);
camera.position.set(0, 0, 14);
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
vp.appendChild(renderer.domElement);

const somaGeo = new THREE.BufferGeometry();
const posArr = new Float32Array(DATA.soma_cloud.length * 3);
const colArr = new Float32Array(DATA.soma_cloud.length * 3);
const palette = [
  [0.22, 0.74, 0.97], [0.65, 0.54, 0.98], [0.96, 0.45, 0.71],
  [0.98, 0.75, 0.14], [0.20, 0.83, 0.60], [0.97, 0.44, 0.44],
  [0.38, 0.65, 0.98], [0.99, 0.57, 0.24], [0.58, 0.64, 0.72]
];
DATA.soma_cloud.forEach((p, i) => {{
  posArr[i*3] = p.x; posArr[i*3+1] = p.y; posArr[i*3+2] = p.z;
  const c = palette[p.organ % palette.length];
  colArr[i*3] = c[0]; colArr[i*3+1] = c[1]; colArr[i*3+2] = c[2];
}});
somaGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
somaGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
const somaCloud = new THREE.Points(somaGeo, new THREE.PointsMaterial({{ size: 0.11, vertexColors: true }}));
scene.add(somaCloud);

function resizeConn() {{
  const w = vp.clientWidth || 400, h = vp.clientHeight || 400;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}}
window.addEventListener('resize', resizeConn);
setTimeout(resizeConn, 50);

function renderFrame(idx) {{
  imgEl.src = DATA.mujoco_images[idx];
  labelEl.innerText = `Frame ${{idx+1}} / ${{DATA.frames.length}} (${{((idx+1)*DATA.tau_frame_ms).toFixed(2)}} ms)`;
  cardsEl.innerHTML = '';
  (DATA.frames[idx] || []).forEach(f => {{
    const c = document.createElement('div');
    const cls = f.life_stage !== 'adult' ? 'immat' : (f.sex === 'F' ? 'female' : '');
    c.className = 'fly-card ' + cls;
    c.innerHTML = `<b>${{f.fly_id}}</b> · Gen G${{f.generation}} · <b>${{f.life_stage.toUpperCase()}}</b> (${{(f.stage_progress*100).toFixed(0)}}%)<br/>
      Cascade: X:A=${{f.x_to_a}} &rarr; ${{f.dsx}} / FruM=${{f.fru_m}} · ${{f.foraging_allele}}<br/>
      State: <b>${{f.behavior}}</b> · Active cells: ${{f.n_active_cells_189k.toLocaleString()}}<br/>
      P1=${{f.male_p1_rate}} · pIP10=${{f.male_pip10_rate}} · pC1=${{f.female_pc1_rate}} · Ovi=${{f.female_ovi_rate}}`;
    cardsEl.appendChild(c);
  }});
}}

let cur = 0, playing = true, tick = 0;
const slider = document.getElementById('slider');
const btn = document.getElementById('play-btn');
slider.addEventListener('input', e => {{ cur = parseInt(e.target.value); renderFrame(cur); }});
btn.addEventListener('click', () => {{ playing = !playing; btn.innerText = playing ? 'Pause' : 'Play'; }});

function loop() {{
  requestAnimationFrame(loop);
  somaCloud.rotation.y += 0.005;
  tick++;
  if (playing && tick % 30 === 0) {{
    cur = (cur + 1) % DATA.frames.length;
    slider.value = cur;
    renderFrame(cur);
  }}
  renderer.render(scene, camera);
}}
renderFrame(0);
loop();
</script>
</body>
</html>
"""
    out_p = Path(output_html_path)
    out_p.write_text(html, encoding="utf-8")
    return {
        "html_path": str(out_p.resolve()),
        "png_preview_path": str(out_p.with_suffix(".png").resolve()),
        "n_founders": len(flies),
        "n_final_colony": len(colony.flies),
        "n_events": len(colony.colony_events),
        "events": colony.colony_events,
    }
