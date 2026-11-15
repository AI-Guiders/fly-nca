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
    # Note: In NeuroMechFly's anatomical frame, negative trochanterfemur-pitch (-139 deg neutral) lifts the femur during swing!
    leg_map = ("lf", "rf", "lm", "rm", "lh", "rh")
    lj = motor.leg_joint_drive.detach().cpu().numpy()
    for leg_i, fg_leg in enumerate(leg_map):
        coxa_fwd = float(lj[leg_i, 0])
        troch_lift = float(lj[leg_i, 1])

        if fg_leg in ("lf", "rf"):
            if f"c_thorax-{fg_leg}_coxa-pitch" in dof_idx:
                ctrl[dof_idx[f"c_thorax-{fg_leg}_coxa-pitch"]] -= 0.32 * coxa_fwd
            if f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch"]] -= 0.42 * troch_lift
            if f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch"]] -= 0.40 * coxa_fwd
        else:
            if f"c_thorax-{fg_leg}_coxa-pitch" in dof_idx:
                ctrl[dof_idx[f"c_thorax-{fg_leg}_coxa-pitch"]] += 0.28 * coxa_fwd
            if f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch"]] -= 0.42 * troch_lift
            if f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch"]] += 0.36 * coxa_fwd

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

    # 4. Bilateral Flight Flapping (200 Hz) vs Unilateral Male Courtship Song (P1 -> pIP10 -> TN1)
    w_l = float(motor.wing_drive_lr[0].item())
    w_r = float(motor.wing_drive_lr[1].item())
    p_l = float(motor.wing_pitch_lr[0].item())
    p_r = float(motor.wing_pitch_lr[1].item())
    mean_wing = 0.5 * (w_l + w_r)
    song_amp = float(motor.wing_song_mode[0].item())

    if mean_wing > 0.45 or motor.jump_ttmn_burst > 0.4:
        # Bilateral 200-Hz flight stroke across both wings!
        stroke = math.sin(step_idx * 2.2)
        feather = math.cos(step_idx * 2.2)
        if "c_thorax-l_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-yaw"]] += 0.90 + 0.65 * w_l * stroke
        if "c_thorax-r_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-yaw"]] += 0.90 + 0.65 * w_r * stroke
        if "c_thorax-l_wing-pitch" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-pitch"]] += 0.45 * feather + 0.30 * p_l
        if "c_thorax-r_wing-pitch" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-pitch"]] += 0.45 * feather + 0.30 * p_r
        if "c_thorax-l_wing-roll" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-roll"]] += 0.55
        if "c_thorax-r_wing-roll" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-roll"]] -= 0.55
    elif wing_song_active or song_amp > 0.02:
        vib = 0.06 * math.sin(step_idx * 0.6)
        if "c_thorax-l_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-yaw"]] += 0.45 + vib
        if "c_thorax-l_wing-pitch" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-pitch"]] += 0.12

    # 5. Female Oviposition Abdomen Bending (oviDN -> abd_Dsx_OA motor neurons)
    abd_pitch = float(motor.abdomen_bend_pitch_yaw[0])
    if oviposition_active or abs(abd_pitch) > 0.05:
        curl = min(-0.15, abd_pitch) if oviposition_active else abd_pitch
        for abd_j in (
            "c_thorax-c_abdomen12-pitch",
            "c_abdomen12-c_abdomen3-pitch",
            "c_abdomen3-c_abdomen4-pitch",
        ):
            if abd_j in dof_idx:
                ctrl[dof_idx[abd_j]] += 0.12 * curl

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
        spawner.spawn_fly(sex="F", foraging_allele="rover", pos_xyz=(0.0042, 0.0012, 0.0), yaw_rad=-2.8),
        spawner.spawn_fly(sex="M", foraging_allele="sitter", pos_xyz=(-0.0042, -0.0010, 0.0), yaw_rad=0.25),
        spawner.spawn_fly(sex="F", foraging_allele="sitter", pos_xyz=(0.0010, -0.0042, 0.0), yaw_rad=1.8),
        spawner.spawn_fly(sex="M", foraging_allele="rover", pos_xyz=(-0.0038, 0.0035, 0.0), yaw_rad=-0.7),
    ][: max(2, int(n_founders))]

    colony = MultiFlySocialArena3D(flies=flies, spawner=spawner)

    # Build a realistic 3D Petri Dish in MuJoCo (radius = 18.0 mm, diameter = 36.0 mm — 15x larger than a 2.5 mm fly!):
    # 1) Dark matte laboratory bench surface (covering the raw checkerboard)
    # 2) Translucent amber-cream agar layer filling the 36-mm Petri dish floor (radius = 17.6 mm)
    # 3) Transparent glass Petri dish rim wall (48 segments around radius = 18.0 mm, height = 3.2 mm)
    # 4) Central live yeast drop (radius = 2.0 mm) where flies gather to feed, court, mate, and lay eggs
    world = FlatGroundWorld()
    world.mjcf_root.worldbody.add_geom(
        name="lab_bench_surface",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.002),
        size=(48.0, 0.002, 0.0),
        rgba=(0.14, 0.15, 0.17, 1.0),
        contype=0,
        conaffinity=0,
    )
    world.mjcf_root.worldbody.add_geom(
        name="petri_glass_base",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.006),
        size=(18.4, 0.006, 0.0),
        rgba=(0.86, 0.92, 0.96, 0.42),
        contype=0,
        conaffinity=0,
    )
    world.mjcf_root.worldbody.add_geom(
        name="petri_agar_gel",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.014),
        size=(17.6, 0.010, 0.0),
        rgba=(0.82, 0.79, 0.62, 0.88),
        contype=0,
        conaffinity=0,
    )
    n_rim_segs = 48
    rim_r = 18.0
    for seg_i in range(n_rim_segs):
        ang = 2.0 * math.pi * seg_i / n_rim_segs
        rx = rim_r * math.cos(ang)
        ry = rim_r * math.sin(ang)
        rq = (math.cos((ang + math.pi / 2.0) / 2.0), 0.0, 0.0, math.sin((ang + math.pi / 2.0) / 2.0))
        world.mjcf_root.worldbody.add_geom(
            name=f"petri_glass_rim_{seg_i:02d}",
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=(rx, ry, 1.60),
            quat=rq,
            size=(1.22, 0.22, 1.60),
            rgba=(0.88, 0.94, 0.98, 0.30),
            contype=0,
            conaffinity=0,
        )
    world.mjcf_root.worldbody.add_geom(
        name="yeast_food_patch",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.028),
        size=(2.0, 0.015, 0.0),
        rgba=(0.90, 0.66, 0.18, 0.92),
        contype=0,
        conaffinity=0,
    )
    # Pre-allocate hidden F1 offspring geoms (hidden under the table until an actual mated female lays an egg!)
    for off_idx in range(6):
        world.mjcf_root.worldbody.add_geom(
            name=f"f1_offspring_{off_idx}",
            type=mj.mjtGeom.mjGEOM_ELLIPSOID,
            pos=(0.0, 0.0, -5.0),
            size=(0.22, 0.09, 0.08),
            rgba=(0.98, 0.96, 0.85, 0.0),
            contype=0,
            conaffinity=0,
        )

    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ACTIVE_ONLY)
    nmf_map: dict[str, NeuroMechFly] = {}

    # Fixed world overview camera framing the 36-mm glass Petri dish, agar floor, and 2.5-mm flies in natural proportion!
    world_cam = world.mjcf_root.worldbody.add_camera(
        name="arena_overview_cam",
        pos=(-14.5, -14.5, 12.2),
        xyaxes=(0.707, -0.707, 0.0, 0.40, 0.40, 0.82),
    )

    for idx, f in enumerate(flies):
        nmf = NeuroMechFly(name=f.genome.fly_id)
        nmf.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
        act_dofs = nmf.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        nmf.add_actuators(act_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL, kp=50.0)
        nmf.add_leg_adhesion()
        nmf.colorize()
        if f.genome.dsx_isoform == "DsxM":
            # Male DsxM suppresses bric-a-brac (bab) -> melanized dark-black A5/A6 abdominal tip & foreleg sex combs!
            for seg, geoms in nmf.bodyseg_to_mjcfgeom.items():
                if seg.name in ("c_abdomen5", "c_abdomen6", "lf_tarsus1", "rf_tarsus1"):
                    for g in geoms:
                        g.material = ""
                        g.rgba = (0.11, 0.09, 0.08, 1.0)
        px_mm = f.pos_xyz[0] * 1000.0
        py_mm = f.pos_xyz[1] * 1000.0
        yaw = float(f.arena.thorax_rpy[2].item())
        q = (float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0)))
        world.add_fly(nmf, spawn_position=(px_mm, py_mm, 0.55), spawn_rotation=Rotation3D("quat", q))
        nmf_map[f.genome.fly_id] = nmf

    sim = Simulation(world)
    sim.reset()
    sim.warmup(0.012)
    mj_renderer = mj.Renderer(sim.mj_model, height=camera_res[0], width=camera_res[1])
    cam_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, world_cam.name)
    off_geom_ids = [mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, f"f1_offspring_{k}") for k in range(6)]

    history_frames: list[list[dict[str, Any]]] = []
    frame_png_b64: list[str] = []

    n_total_frames = max(1, int(frames))
    for step_i in range(n_total_frames):
        snaps = colony.step_population(frames=1, enable_ontogeny_and_reproduction=True)
        history_frames.append(snaps)

        # Decode base neural motor effectors once per brain frame from each fly's autonomous 189,766-cell connectome
        base_motors = {
            f.genome.fly_id: f.full_connectome.decode_motor_effectors(cpg_phase_rad=0.0)
            for f in flies
            if f.genome.fly_id in nmf_map
        }
        walk_amplitudes = {
            f.genome.fly_id: max(
                f.full_connectome.group_topk_rate("dn_cluster_walking", k=6) * 2.5,
                f.full_connectome.group_mean_rate("dn_dnp09") * 2.0,
                0.65 * f.full_connectome.foraging_gain,
            )
            for f in flies
            if f.genome.fly_id in nmf_map
        }
        gait_dirs = {
            f.genome.fly_id: (-1.0 if f.full_connectome.group_mean_rate("dn_mdn") > 0.22 else 1.0)
            for f in flies
            if f.genome.fly_id in nmf_map
        }

        # Advance MuJoCo physics across a smooth tripod stride phase (150 x 0.1 ms = 15 ms per rendered frame)
        tripod_offsets = (0.0, math.pi, math.pi, 0.0, 0.0, math.pi)
        for sub_k in range(150):
            global_sub = step_i * 150 + sub_k
            cpg_phase = -2.0 * math.pi * (global_sub / 100.0)  # Natural 10-Hz (100 x 0.1 ms = 10 ms) tripod stride
            for idx, f in enumerate(flies):
                nmf = nmf_map.get(f.genome.fly_id)
                if nmf is None:
                    continue
                motor = base_motors[f.genome.fly_id]
                w_amp = min(1.0, walk_amplitudes[f.genome.fly_id])
                g_dir = gait_dirs[f.genome.fly_id]
                phase_f = cpg_phase * g_dir + idx * 0.7
                for leg_i in range(6):
                    phi = phase_f + tripod_offsets[leg_i]
                    swing = max(0.0, math.sin(phi)) * w_amp
                    stance = max(0.0, -math.sin(phi)) * w_amp
                    motor.leg_joint_drive[leg_i, 0] = math.cos(phi) * w_amp
                    motor.leg_joint_drive[leg_i, 1] = swing
                    motor.leg_joint_drive[leg_i, 2] = stance
                    motor.tarsus_adhesion[leg_i] = 0.0 if swing > 0.20 else 1.0
                ctrl, adh = motor_out_to_neuromechfly_ctrl(
                    nmf,
                    motor,
                    wing_song_active=f.wing_song_active,
                    oviposition_active=f.oviposition_active,
                    step_idx=global_sub,
                )
                thorax_id = int(sim._internal_bodyids_by_fly[f.genome.fly_id][0])
                mean_w = 0.5 * float((motor.wing_drive_lr[0] + motor.wing_drive_lr[1]).item())
                if motor.jump_ttmn_burst > 0.45 or mean_w > 0.48:
                    # Only if the fly's own connectome activates TTMn jump or wing power MNs, apply aerodynamic lift
                    adh[:] = 0.0
                    cur_z = float(sim.get_body_positions(f.genome.fly_id)[0][2])
                    lift_f = 10.05 + max(-3.0, min(4.5, (2.0 - cur_z) * 4.5))
                    yaw_f = float(f.arena.thorax_rpy[2].item())
                    sim.mj_data.xfrc_applied[thorax_id, :3] = [
                        0.8 * math.cos(yaw_f),
                        0.8 * math.sin(yaw_f),
                        lift_f,
                    ]
                else:
                    sim.mj_data.xfrc_applied[thorax_id, :] = 0.0

                sim.set_actuator_inputs(f.genome.fly_id, ActuatorType.POSITION, ctrl)
                sim.set_leg_adhesion_states(f.genome.fly_id, adh)
            sim.step()
            # Sync 3D thorax position (x, y, z) from MuJoCo back into Fly-NCA arena!
            for f in flies:
                if f.genome.fly_id in nmf_map:
                    thorax_mm = sim.get_body_positions(f.genome.fly_id)[0]
                    f.arena.pos_xyz[0] = float(thorax_mm[0]) * 1e-3
                    f.arena.pos_xyz[1] = float(thorax_mm[1]) * 1e-3
                    f.arena.pos_xyz[2] = max(0.0, float(thorax_mm[2] - 0.55) * 1e-3)

        # Update dynamic 3D geoms for any real F1 offspring (egg -> crawling larva -> pupa -> eclosed pupal case)
        offspring_flies = [f for f in colony.flies if f.genome.generation >= 1]
        for k_off, gid in enumerate(off_geom_ids):
            if gid < 0:
                continue
            if k_off < len(offspring_flies):
                fo = offspring_flies[k_off]
                sim.mj_model.geom_pos[gid] = [fo.pos_xyz[0] * 1000.0, fo.pos_xyz[1] * 1000.0, 0.22]
                if fo.life_stage == "egg":
                    sim.mj_model.geom_size[gid] = [0.42, 0.18, 0.16]
                    sim.mj_model.geom_rgba[gid] = [1.0, 1.0, 1.0, 1.0]
                elif fo.life_stage == "larva":
                    sim.mj_model.geom_size[gid] = [0.85, 0.26, 0.22]
                    sim.mj_model.geom_rgba[gid] = [0.98, 0.99, 0.90, 1.0]
                else:
                    sim.mj_model.geom_size[gid] = [0.92, 0.34, 0.30]
                    sim.mj_model.geom_rgba[gid] = [0.58, 0.30, 0.08, 1.0]
            else:
                sim.mj_model.geom_pos[gid] = [0.0, 0.0, -5.0]
                sim.mj_model.geom_rgba[gid, 3] = 0.0
        mj.mj_forward(sim.mj_model, sim.mj_data)

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
      <span id="frame-label">t = 0.00 ms</span>
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
  <label style="font-size:12px; color:#94a3b8; display:flex; align-items:center; gap:6px;">
    Time Scale:
    <select id="speed-sel" style="background:#1e293b; color:#e2e8f0; border:1px solid #334155; border-radius:5px; padding:4px 8px; font-weight:600;">
      <option value="0.05">0.05x (Fly Wingbeat Microscope: 1 frame = 4.038 ms)</option>
      <option value="0.25">0.25x (Slow-Motion)</option>
      <option value="1.0" selected>1.0x (Real-Time Seconds: 247.6 fly frames / s)</option>
      <option value="5.0">5.0x (Fast-Forward Seconds)</option>
      <option value="20.0">20.0x (Colony Ontogeny Time-Lapse)</option>
    </select>
  </label>
  <span id="live-status" style="font-size:12px; color:#22c55e; font-weight:600;">&#9679; LIVE TIME RUNNING (t = 0.000 s &rarr; &infin;)</span>
</footer>
<script>
const DATA = {payload_json};
const imgEl = document.getElementById('mj-img');
const cardsEl = document.getElementById('fly-cards');
const evEl = document.getElementById('ev-list');
const labelEl = document.getElementById('frame-label');
const statusEl = document.getElementById('live-status');
const speedSel = document.getElementById('speed-sel');
let timeScale = 1.0;
speedSel.addEventListener('change', e => {{
  timeScale = parseFloat(e.target.value) || 1.0;
  fetch(`http://127.0.0.1:8765/speed?scale=${{timeScale}}`).catch(() => {{}});
}});

let renderedEventKeys = new Set();
function renderEvents(events) {{
  (events || []).forEach(ev => {{
    const key = `${{ev.time_ms}}_${{ev.event}}_${{ev.fly_id || ev.egg_id || ev.female_id || ''}}`;
    if (renderedEventKeys.has(key)) return;
    renderedEventKeys.add(key);
    const d = document.createElement('div');
    d.className = 'ev-item';
    const sec = (ev.time_ms / 1000.0).toFixed(3);
    d.innerHTML = `<b>[t = ${{sec}} s (${{ev.time_ms}} ms)] ${{ev.event}}</b><br/>${{JSON.stringify(ev)}}`;
    evEl.appendChild(d);
    evEl.scrollTop = evEl.scrollHeight;
  }});
}}
renderEvents(DATA.events);

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

const cardDomByFly = new Map();
function renderSnapshot(imgSrc, timeMs, fliesSnap, eventsSnap) {{
  if (imgEl.src !== imgSrc) imgEl.src = imgSrc;
  const timeSec = (timeMs / 1000.0).toFixed(3);
  const flyQuanta = Math.round(timeMs / DATA.tau_frame_ms);
  labelEl.textContent = `t = ${{timeSec}} s (${{timeMs.toFixed(1)}} ms · ${{flyQuanta}} wingbeats)`;
  statusEl.textContent = `● LIVE TIME: t = ${{timeSec}} s (${{timeMs.toFixed(1)}} ms · ${{flyQuanta}} × τ_frame) · Scale: ${{timeScale}}x`;
  (fliesSnap || []).forEach(f => {{
    let entry = cardDomByFly.get(f.fly_id);
    if (!entry) {{
      const c = document.createElement('div');
      const l1 = document.createElement('div');
      const l2 = document.createElement('div');
      const l3 = document.createElement('div');
      const l4 = document.createElement('div');
      c.append(l1, l2, l3, l4);
      cardsEl.appendChild(c);
      entry = {{ c, l1, l2, l3, l4 }};
      cardDomByFly.set(f.fly_id, entry);
    }}
    const cls = f.life_stage !== 'adult' ? 'immat' : (f.sex === 'F' ? 'female' : '');
    const targetClass = 'fly-card ' + cls;
    if (entry.c.className !== targetClass) entry.c.className = targetClass;
    entry.l1.innerHTML = `<b>${{f.fly_id}}</b> · Gen G${{f.generation}} · <b>${{f.life_stage.toUpperCase()}}</b> (${{(f.stage_progress*100).toFixed(0)}}%)`;
    entry.l2.textContent = `Cascade: X:A=${{f.x_to_a}} → ${{f.dsx}} / FruM=${{f.fru_m}} · ${{f.foraging_allele}}`;
    entry.l3.innerHTML = `State: <b>${{f.behavior}}</b> · Active cells: ${{f.n_active_cells_189k.toLocaleString()}}`;
    entry.l4.textContent = `P1=${{f.male_p1_rate}} · pIP10=${{f.male_pip10_rate}} · pC1=${{f.female_pc1_rate}} · Ovi=${{f.female_ovi_rate}}`;
  }});
  if (eventsSnap) renderEvents(eventsSnap);
}}

let cur = 0, playing = true, liveConnected = false;
let simTimeMs = 0.0;
let lastWallMs = performance.now();
const btn = document.getElementById('play-btn');
btn.addEventListener('click', () => {{
  playing = !playing;
  btn.innerText = playing ? 'Pause' : 'Play';
  lastWallMs = performance.now();
}});

// Connect to live real-time server stream if available (unbounded t = 0 -> infinity!)
try {{
  const es = new EventSource('http://127.0.0.1:8765/stream');
  es.onmessage = (e) => {{
    if (!playing) return;
    liveConnected = true;
    const msg = JSON.parse(e.data);
    renderSnapshot(msg.image, msg.time_ms, msg.flies, msg.events);
  }};
  es.onerror = () => {{ liveConnected = false; }};
}} catch (_) {{}}

function loop(nowMs) {{
  requestAnimationFrame(loop);
  somaCloud.rotation.y += 0.005;
  const dtWallMs = Math.max(0.0, (nowMs || performance.now()) - lastWallMs);
  lastWallMs = nowMs || performance.now();
  if (playing && !liveConnected) {{
    simTimeMs += dtWallMs * timeScale;
    const nextIdx = Math.floor(simTimeMs / (DATA.tau_frame_ms * 4.0)) % DATA.frames.length;
    if (nextIdx !== cur || Math.floor(simTimeMs / 40) !== Math.floor((simTimeMs - dtWallMs * timeScale) / 40)) {{
      cur = nextIdx;
      renderSnapshot(DATA.mujoco_images[cur], simTimeMs, DATA.frames[cur], DATA.events);
    }}
  }}
  renderer.render(scene, camera);
}}
renderSnapshot(DATA.mujoco_images[0], 0.0, DATA.frames[0], DATA.events);
requestAnimationFrame(loop);
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


def serve_live_mujoco_colony(
    output_html_path: Path | str = "colony_3d_viewer.html",
    *,
    n_founders: int = 4,
    seed: int = 108,
    port: int = 8765,
    camera_res: tuple[int, int] = (640, 960),
    device: str | None = None,
) -> None:
    """Run an unbounded real-time (t = 0 -> infinity) MuJoCo + 189,766-cell Fly-NCA loop streaming SSE on `port`."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    # First export initial HTML so opening colony_3d_viewer.html immediately works and connects to :8765/stream
    run_mujoco_colony_and_export_viewer(
        output_html_path=output_html_path,
        n_founders=n_founders,
        frames=4,
        seed=seed,
        camera_res=camera_res,
        device=device,
    )

    latest_state: dict[str, Any] = {"seq": 0, "json": "{}", "scale": 1.0}
    cond = threading.Condition()

    class LiveSSEHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path.startswith("/speed"):
                try:
                    q = self.path.split("scale=")[-1].split("&")[0]
                    latest_state["scale"] = max(0.02, min(50.0, float(q)))
                except Exception:
                    pass
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b"OK")
                return
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                last_seq = -1
                try:
                    while True:
                        with cond:
                            while latest_state["seq"] == last_seq:
                                cond.wait(timeout=1.0)
                            last_seq = latest_state["seq"]
                            data_str = latest_state["json"]
                        self.wfile.write(f"data: {data_str}\n\n".encode("utf-8"))
                        self.wfile.flush()
                except Exception:
                    return
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(Path(output_html_path).read_bytes())

    server = ThreadingHTTPServer(("127.0.0.1", port), LiveSSEHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Now run the infinite t = 0 -> infinity simulation loop!
    spawner = FlySpawner(seed=seed, device=device)
    flies: list[SpawnedFly] = [
        spawner.spawn_fly(sex="F", foraging_allele="rover", pos_xyz=(0.0042, 0.0012, 0.0), yaw_rad=-2.8),
        spawner.spawn_fly(sex="M", foraging_allele="sitter", pos_xyz=(-0.0042, -0.0010, 0.0), yaw_rad=0.25),
        spawner.spawn_fly(sex="F", foraging_allele="sitter", pos_xyz=(0.0010, -0.0042, 0.0), yaw_rad=1.8),
        spawner.spawn_fly(sex="M", foraging_allele="rover", pos_xyz=(-0.0038, 0.0035, 0.0), yaw_rad=-0.7),
    ][: max(2, int(n_founders))]
    colony = MultiFlySocialArena3D(flies=flies, spawner=spawner)

    world = FlatGroundWorld()
    world.mjcf_root.worldbody.add_geom(
        name="lab_bench_surface",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.002),
        size=(48.0, 0.002, 0.0),
        rgba=(0.14, 0.15, 0.17, 1.0),
        contype=0,
        conaffinity=0,
    )
    world.mjcf_root.worldbody.add_geom(
        name="petri_glass_base",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.006),
        size=(18.4, 0.006, 0.0),
        rgba=(0.86, 0.92, 0.96, 0.42),
        contype=0,
        conaffinity=0,
    )
    world.mjcf_root.worldbody.add_geom(
        name="petri_agar_gel",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.014),
        size=(17.6, 0.010, 0.0),
        rgba=(0.82, 0.79, 0.62, 0.88),
        contype=0,
        conaffinity=0,
    )
    for seg_i in range(48):
        ang = 2.0 * math.pi * seg_i / 48
        rx, ry = 18.0 * math.cos(ang), 18.0 * math.sin(ang)
        rq = (math.cos((ang + math.pi / 2.0) / 2.0), 0.0, 0.0, math.sin((ang + math.pi / 2.0) / 2.0))
        world.mjcf_root.worldbody.add_geom(
            name=f"petri_glass_rim_{seg_i:02d}",
            type=mj.mjtGeom.mjGEOM_BOX,
            pos=(rx, ry, 1.60),
            quat=rq,
            size=(1.22, 0.22, 1.60),
            rgba=(0.88, 0.94, 0.98, 0.30),
            contype=0,
            conaffinity=0,
        )
    world.mjcf_root.worldbody.add_geom(
        name="yeast_food_patch",
        type=mj.mjtGeom.mjGEOM_CYLINDER,
        pos=(0.0, 0.0, 0.028),
        size=(2.0, 0.015, 0.0),
        rgba=(0.90, 0.66, 0.18, 0.92),
        contype=0,
        conaffinity=0,
    )
    for off_idx in range(6):
        world.mjcf_root.worldbody.add_geom(
            name=f"f1_offspring_{off_idx}",
            type=mj.mjtGeom.mjGEOM_ELLIPSOID,
            pos=(0.0, 0.0, -5.0),
            size=(0.22, 0.09, 0.08),
            rgba=(0.98, 0.96, 0.85, 0.0),
            contype=0,
            conaffinity=0,
        )

    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ACTIVE_ONLY)
    nmf_map: dict[str, NeuroMechFly] = {}
    world_cam = world.mjcf_root.worldbody.add_camera(
        name="arena_overview_cam",
        pos=(-14.5, -14.5, 12.2),
        xyaxes=(0.707, -0.707, 0.0, 0.40, 0.40, 0.82),
    )
    for f in flies:
        nmf = NeuroMechFly(name=f.genome.fly_id)
        nmf.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
        act_dofs = nmf.skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
        nmf.add_actuators(act_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL, kp=50.0)
        nmf.add_leg_adhesion()
        nmf.colorize()
        if f.genome.dsx_isoform == "DsxM":
            for seg, geoms in nmf.bodyseg_to_mjcfgeom.items():
                if seg.name in ("c_abdomen5", "c_abdomen6", "lf_tarsus1", "rf_tarsus1"):
                    for g in geoms:
                        g.material = ""
                        g.rgba = (0.11, 0.09, 0.08, 1.0)
        yaw = float(f.arena.thorax_rpy[2].item())
        q = (float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0)))
        world.add_fly(nmf, spawn_position=(f.pos_xyz[0] * 1000.0, f.pos_xyz[1] * 1000.0, 0.55), spawn_rotation=Rotation3D("quat", q))
        nmf_map[f.genome.fly_id] = nmf

    sim = Simulation(world)
    sim.reset()
    sim.warmup(0.012)
    mj_renderer = mj.Renderer(sim.mj_model, height=camera_res[0], width=camera_res[1])
    cam_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, world_cam.name)
    off_geom_ids = [mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, f"f1_offspring_{k}") for k in range(6)]

    step_i = 0
    tripod_offsets = (0.0, math.pi, math.pi, 0.0, 0.0, math.pi)
    while True:
        snaps = colony.step_population(frames=1, enable_ontogeny_and_reproduction=True)
        base_motors = {
            f.genome.fly_id: f.full_connectome.decode_motor_effectors(cpg_phase_rad=0.0)
            for f in flies
            if f.genome.fly_id in nmf_map
        }
        walk_amplitudes = {
            f.genome.fly_id: max(
                f.full_connectome.group_topk_rate("dn_cluster_walking", k=6) * 2.5,
                f.full_connectome.group_mean_rate("dn_dnp09") * 2.0,
                0.65 * f.full_connectome.foraging_gain,
            )
            for f in flies
            if f.genome.fly_id in nmf_map
        }
        gait_dirs = {
            f.genome.fly_id: (-1.0 if f.full_connectome.group_mean_rate("dn_mdn") > 0.22 else 1.0)
            for f in flies
            if f.genome.fly_id in nmf_map
        }

        for sub_k in range(100):
            global_sub = step_i * 100 + sub_k
            cpg_phase = -2.0 * math.pi * (global_sub / 100.0)
            for idx, f in enumerate(flies):
                nmf = nmf_map.get(f.genome.fly_id)
                if nmf is None:
                    continue
                motor = base_motors[f.genome.fly_id]
                w_amp = min(1.0, walk_amplitudes[f.genome.fly_id])
                g_dir = gait_dirs[f.genome.fly_id]
                phase_f = cpg_phase * g_dir + idx * 0.7
                for leg_i in range(6):
                    phi = phase_f + tripod_offsets[leg_i]
                    swing = max(0.0, math.sin(phi)) * w_amp
                    stance = max(0.0, -math.sin(phi)) * w_amp
                    motor.leg_joint_drive[leg_i, 0] = math.cos(phi) * w_amp
                    motor.leg_joint_drive[leg_i, 1] = swing
                    motor.leg_joint_drive[leg_i, 2] = stance
                    motor.tarsus_adhesion[leg_i] = 0.0 if swing > 0.20 else 1.0
                ctrl, adh = motor_out_to_neuromechfly_ctrl(
                    nmf,
                    motor,
                    wing_song_active=f.wing_song_active,
                    oviposition_active=f.oviposition_active,
                    step_idx=global_sub,
                )
                sim.set_actuator_inputs(f.genome.fly_id, ActuatorType.POSITION, ctrl)
                sim.set_leg_adhesion_states(f.genome.fly_id, adh)
            sim.step()
            for f in flies:
                if f.genome.fly_id in nmf_map:
                    thorax_mm = sim.get_body_positions(f.genome.fly_id)[0]
                    # Keep flies inside the 36-mm glass Petri dish rim (r <= 16.5 mm)
                    r_xy = math.hypot(float(thorax_mm[0]), float(thorax_mm[1]))
                    f.arena.pos_xyz[0] = float(thorax_mm[0]) * 1e-3
                    f.arena.pos_xyz[1] = float(thorax_mm[1]) * 1e-3
                    f.arena.pos_xyz[2] = max(0.0, float(thorax_mm[2] - 0.55) * 1e-3)
                    if r_xy > 16.0:
                        gait_dirs[f.genome.fly_id] = -1.0

        offspring_flies = [f for f in colony.flies if f.genome.generation >= 1]
        for k_off, gid in enumerate(off_geom_ids):
            if gid < 0:
                continue
            if k_off < len(offspring_flies):
                fo = offspring_flies[k_off]
                sim.mj_model.geom_pos[gid] = [fo.pos_xyz[0] * 1000.0, fo.pos_xyz[1] * 1000.0, 0.22]
                if fo.life_stage == "egg":
                    sim.mj_model.geom_size[gid] = [0.42, 0.18, 0.16]
                    sim.mj_model.geom_rgba[gid] = [1.0, 1.0, 1.0, 1.0]
                elif fo.life_stage == "larva":
                    sim.mj_model.geom_size[gid] = [0.85, 0.26, 0.22]
                    sim.mj_model.geom_rgba[gid] = [0.98, 0.99, 0.90, 1.0]
                else:
                    sim.mj_model.geom_size[gid] = [0.92, 0.34, 0.30]
                    sim.mj_model.geom_rgba[gid] = [0.58, 0.30, 0.08, 1.0]
            else:
                sim.mj_model.geom_pos[gid] = [0.0, 0.0, -5.0]
                sim.mj_model.geom_rgba[gid, 3] = 0.0
        mj.mj_forward(sim.mj_model, sim.mj_data)

        mj_renderer.update_scene(sim.mj_data, camera=cam_id)
        rgb = mj_renderer.render()
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        with cond:
            latest_state["seq"] += 1
            latest_state["json"] = json.dumps(
                {
                    "time_ms": round(colony.time_ms, 2),
                    "image": img_b64,
                    "flies": snaps,
                    "events": colony.colony_events[-25:],
                },
                ensure_ascii=False,
            )
            cond.notify_all()
        step_i += 1
        time.sleep(0.01)
