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

from fly_nca.body.adapter import FlyBodyAdapter, MotorOut, SensorIn
from fly_nca.body.arena_3d import OdorPlume3D
from fly_nca.core.cell_state import TAU_FAST_MS, TAU_FRAME_MS
from fly_nca.genome.spawner import FlySpawner, MultiFlySocialArena3D, SpawnedFly


class MuJoCoFlyBodyAdapter:
    """External 3D MuJoCo / FlyGym (NeuroMechFly) body adapter implementing `FlyBodyAdapter`.

    Usage: `MuJoCoFlyBodyAdapter(sim, nmf).connect(fly)`
    - Connects a living `SpawnedFly` / `FullConnectomeFlyNCA` directly to a 3D `NeuroMechFly` mecha in `MuJoCo`.
    - `step_physics(motor, dt_ms)` applies `motor` to the MuJoCo actuators and returns `SensorIn` read
      directly from the MuJoCo simulation (`get_ground_contact_info` campaniform forces, `get_joint_angles`
      FeCO proprioception, thorax pose & horizon).
    """

    def __init__(
        self,
        sim: Simulation,
        nmf: NeuroMechFly,
        *,
        device: torch.device | str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        dev = torch.device(device)
        self.device = dev
        self.dtype = dtype
        self.sim = sim
        self.nmf = nmf
        self.fly_ref: SpawnedFly | None = None

        self.pos_xyz = torch.zeros(3, device=dev, dtype=dtype)
        self.vel_xyz = torch.zeros(3, device=dev, dtype=dtype)
        self.thorax_rpy = torch.zeros(3, device=dev, dtype=dtype)
        self.omega_rpy = torch.zeros(3, device=dev, dtype=dtype)
        self.neck_rpy = torch.zeros(3, device=dev, dtype=dtype)
        self.leg_angles = torch.zeros((6, 4), device=dev, dtype=dtype)
        self.leg_load = torch.full((6,), 0.5, device=dev, dtype=dtype)
        self.bristle_touch = torch.zeros(6, device=dev, dtype=dtype)
        self.time_ms: float = 0.0

        self.sun_azimuth_world_rad: float = 0.0
        self.wind_world_xyz: tuple[float, float, float] = (0.15, 0.0, 0.0)
        self.turbulence_torque_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.looming_rate: float = 0.0
        self.small_target_pos_xyz: tuple[float, float, float] | None = None
        self.temperature_c: float = 25.0
        self.ambient_temperature_c: float = 25.0
        self.surface_texture_freq: float = 1.0
        self.crop_stretch: float = 0.2
        self.plumes: list[OdorPlume3D] = []
        self.last_motor: MotorOut = MotorOut.zeros(device=dev, dtype=dtype)
        self.smoothed_leg_drive = torch.zeros((6, 4), device=dev, dtype=dtype)

    @property
    def head_world_rpy(self) -> torch.Tensor:
        return self.thorax_rpy + self.neck_rpy

    @property
    def time_fast_quanta(self) -> float:
        return self.time_ms / TAU_FAST_MS

    @property
    def time_frames(self) -> float:
        return self.time_ms / TAU_FRAME_MS

    def connect(self, fly: SpawnedFly) -> MuJoCoFlyBodyAdapter:
        """Bind a living Fly-NCA connectome/organism to this MuJoCo NeuroMechFly peripheral adapter."""
        prev = getattr(fly, "arena", None)
        if prev is not None:
            self.device = prev.pos_xyz.device
            self.dtype = prev.pos_xyz.dtype
            self.pos_xyz = prev.pos_xyz.clone()
            self.vel_xyz = prev.vel_xyz.clone()
            self.thorax_rpy = prev.thorax_rpy.clone()
            self.omega_rpy = prev.omega_rpy.clone()
            self.neck_rpy = prev.neck_rpy.clone()
            self.leg_angles = prev.leg_angles.clone()
            self.leg_load = prev.leg_load.clone()
            self.bristle_touch = prev.bristle_touch.clone()
            self.plumes = list(getattr(prev, "plumes", []))
            self.wind_world_xyz = getattr(prev, "wind_world_xyz", (0.15, 0.0, 0.0))
            self.temperature_c = getattr(prev, "temperature_c", 25.0)
            self.surface_texture_freq = getattr(prev, "surface_texture_freq", 1.0)
            self.crop_stretch = getattr(prev, "crop_stretch", 0.2)
            self.last_motor = MotorOut.zeros(device=self.device, dtype=self.dtype)
        self.fly_ref = fly
        fly.arena = self  # type: ignore[assignment]
        self.sync_from_mujoco(full_proprioception=True)
        return self

    def sync_from_mujoco(self, *, full_proprioception: bool = True) -> None:
        """Read thorax pose, ground contact forces, and joint angles directly from MuJoCo."""
        thorax_mm = self.sim.get_body_positions(self.nmf.name)[0]
        self.pos_xyz[0] = float(thorax_mm[0]) * 1e-3
        self.pos_xyz[1] = float(thorax_mm[1]) * 1e-3
        self.pos_xyz[2] = max(0.0, float(thorax_mm[2] - 0.55) * 1e-3)
        th_id = int(self.sim._internal_bodyids_by_fly[self.nmf.name][0])
        R_th = self.sim.mj_data.xmat[th_id].reshape(3, 3)
        self.thorax_rpy[2] = float(math.atan2(float(R_th[1, 0]), float(R_th[0, 0])))
        if full_proprioception:
            c_found, c_forces, _, _, _, _ = self.sim.get_ground_contact_info(self.nmf.name)
            fg_load = np.clip(np.linalg.norm(c_forces, axis=1) / 4.0, 0.0, 1.0) * (c_found > 0)
            nca_load = [fg_load[0], fg_load[3], fg_load[1], fg_load[4], fg_load[2], fg_load[5]]
            self.leg_load = torch.tensor(nca_load, device=self.pos_xyz.device, dtype=torch.float32)

            j_ang = self.sim.get_joint_angles(self.nmf.name)
            act_order = self.nmf.get_actuated_jointdofs_order(ActuatorType.POSITION)
            neut_map = self.nmf.jointdof_to_neutralaction_by_type[ActuatorType.POSITION]
            d_idx = {j.name: i for i, j in enumerate(act_order)}
            feco_mat = np.zeros((6, 4), dtype=np.float32)
            for li, fg_l in enumerate(("lf", "rf", "lm", "rm", "lh", "rh")):
                for ji, jn in enumerate((
                    f"c_thorax-{fg_l}_coxa-pitch",
                    f"{fg_l}_coxa-{fg_l}_trochanterfemur-pitch",
                    f"{fg_l}_trochanterfemur-{fg_l}_tibia-pitch",
                    f"{fg_l}_tibia-{fg_l}_tarsus1-pitch",
                )):
                    if jn in d_idx:
                        k_d = d_idx[jn]
                        feco_mat[li, ji] = float(np.clip(j_ang[k_d] - neut_map[act_order[k_d]], -1.0, 1.0))
            self.leg_angles = torch.tensor(feco_mat, device=self.pos_xyz.device, dtype=torch.float32)

    def apply_motor_to_mujoco(self, motor: MotorOut | None = None, step_idx: int = 0) -> None:
        """Apply MotorOut command to the NeuroMechFly position & adhesion actuators in MuJoCo."""
        m = motor if motor is not None else self.last_motor
        # Hill-type muscle activation-contraction low-pass filter (tau_muscle ~ 3 ms = 30 x 0.1 ms MuJoCo steps)
        self.smoothed_leg_drive = 0.92 * self.smoothed_leg_drive + 0.08 * m.leg_joint_drive.to(self.device, dtype=self.dtype)
        m_smooth = MotorOut(
            leg_joint_drive=self.smoothed_leg_drive,
            tarsus_adhesion=m.tarsus_adhesion,
            wing_drive_lr=m.wing_drive_lr,
            wing_pitch_lr=m.wing_pitch_lr,
            wing_song_mode=m.wing_song_mode,
            haltere_drive=m.haltere_drive,
            neck_torque_rpy=m.neck_torque_rpy,
            proboscis_per=m.proboscis_per,
            jump_ttmn_burst=m.jump_ttmn_burst,
            abdomen_bend_pitch_yaw=m.abdomen_bend_pitch_yaw,
        )
        w_song = self.fly_ref.wing_song_active if self.fly_ref is not None else False
        ovi_act = self.fly_ref.oviposition_active if self.fly_ref is not None else False
        ctrl, adh = motor_out_to_neuromechfly_ctrl(
            self.nmf,
            m_smooth,
            wing_song_active=w_song,
            oviposition_active=ovi_act,
            step_idx=step_idx,
        )
        thorax_id = int(self.sim._internal_bodyids_by_fly[self.nmf.name][0])
        # Zero external force or torque on the thorax: 100% supported by the 6 legs in MuJoCo!
        self.sim.mj_data.xfrc_applied[thorax_id, :] = 0.0
        self.sim.set_actuator_inputs(self.nmf.name, ActuatorType.POSITION, ctrl)
        self.sim.set_leg_adhesion_states(self.nmf.name, adh)

    def read_sensors(self, proboscis_per: float = 0.0, dt_ms: float = 0.0) -> SensorIn:
        """Encode current MuJoCo body state and environmental fields into a SensorIn afferent packet."""
        dev = self.pos_xyz.device
        head_yaw = float(self.head_world_rpy[2].item())
        sun_ego = math.atan2(
            math.sin(self.sun_azimuth_world_rad - head_yaw),
            math.cos(self.sun_azimuth_world_rad - head_yaw),
        )
        wx, wy, wz = self.wind_world_xyz
        cos_y = math.cos(-head_yaw)
        sin_y = math.sin(-head_yaw)
        wind_ego = torch.tensor(
            [wx * cos_y - wy * sin_y, wx * sin_y + wy * cos_y, wz],
            device=dev,
            dtype=self.dtype,
        )
        odor_lr = torch.zeros((2, 50), device=dev, dtype=self.dtype)
        taste = torch.zeros(3, device=dev, dtype=self.dtype)
        ant_offset = 0.00015
        left_ant_xy = (
            float(self.pos_xyz[0].item()) - ant_offset * math.sin(head_yaw),
            float(self.pos_xyz[1].item()) + ant_offset * math.cos(head_yaw),
        )
        right_ant_xy = (
            float(self.pos_xyz[0].item()) + ant_offset * math.sin(head_yaw),
            float(self.pos_xyz[1].item()) - ant_offset * math.cos(head_yaw),
        )
        fz = float(self.pos_xyz[2].item())
        for plume in self.plumes:
            px, py, pz = plume.pos_xyz
            for side_idx, (ax, ay) in enumerate((left_ant_xy, right_ant_xy)):
                d2 = (ax - px) ** 2 + (ay - py) ** 2 + (fz - pz) ** 2
                conc = plume.intensity * math.exp(-d2 / (2.0 * plume.sigma_m**2))
                g_idx = plume.glomerulus_id % 50
                odor_lr[side_idx, g_idx] = min(1.0, float(odor_lr[side_idx, g_idx].item()) + conc)
            dist_body = math.sqrt((float(self.pos_xyz[0]) - px) ** 2 + (float(self.pos_xyz[1]) - py) ** 2 + (fz - pz) ** 2)
            if dist_body < 0.008:
                if plume.is_sweet_food:
                    taste[0] = 1.0
                    if proboscis_per > 0.5 and dt_ms > 0.0:
                        self.crop_stretch = min(1.0, self.crop_stretch + 0.01 * dt_ms)
                if plume.is_bitter_shock:
                    taste[1] = 1.0
                if plume.is_yeast_protein:
                    taste[2] = 1.0
        small_sal = 0.0
        small_az = 0.0
        if self.small_target_pos_xyz is not None:
            tx, ty, _ = self.small_target_pos_xyz
            dx = tx - float(self.pos_xyz[0].item())
            dy = ty - float(self.pos_xyz[1].item())
            small_az = math.atan2(dy, dx) - head_yaw
            small_az = math.atan2(math.sin(small_az), math.cos(small_az))
            small_sal = 1.0 / (1.0 + 10.0 * (dx * dx + dy * dy))
        angles = torch.linspace(-math.pi, math.pi, 892, device=dev, dtype=self.dtype)
        ommatidia = 0.5 * (1.0 + torch.cos(angles - sun_ego))
        return SensorIn(
            ommatidia_light=ommatidia,
            sun_azimuth_rad=sun_ego,
            looming_rate=self.looming_rate,
            small_target_salience=small_sal,
            small_target_azimuth_rad=small_az,
            odor_glomeruli_lr=odor_lr,
            wind_vector_ego=wind_ego,
            sound_vibration=0.0,
            leg_joint_angles=self.leg_angles.clone(),
            leg_ground_load=self.leg_load.clone(),
            bristle_touch=self.bristle_touch.clone(),
            haltere_coriolis_rpy=self.omega_rpy.clone(),
            neck_angles_rpy=self.neck_rpy.clone(),
            head_horizon_rpy=self.head_world_rpy.clone(),
            taste_sweet_bitter_amino=taste,
            temperature_c=self.temperature_c,
            surface_texture_freq=self.surface_texture_freq,
            crop_stretch=self.crop_stretch,
        )

    def step_physics(self, motor: MotorOut, dt_ms: float = 1.711) -> SensorIn:
        """Execute FlyBodyAdapter contract directly against the MuJoCo NeuroMechFly body."""
        if float(dt_ms) > 0.0:
            self.time_ms += float(dt_ms)
            self.last_motor = motor
            self.apply_motor_to_mujoco(motor)
        return self.read_sensors(proboscis_per=float(motor.proboscis_per), dt_ms=float(dt_ms))

    def inspect_physics_state(self) -> dict[str, Any]:
        """AI-Friendly X-Ray Inspector: structured JSON telemetry of MuJoCo contacts, forces, joint limits, and self-collisions."""
        m = self.sim.mj_model
        d = self.sim.mj_data
        prefix = f"{self.nmf.name}/"

        thorax_mm = self.sim.get_body_positions(self.nmf.name)[0]
        th_id = int(self.sim._internal_bodyids_by_fly[self.nmf.name][0])
        R_th = d.xmat[th_id].reshape(3, 3)
        yaw = float(math.atan2(float(R_th[1, 0]), float(R_th[0, 0])))
        pitch = float(math.asin(max(-1.0, min(1.0, float(-R_th[2, 0])))))
        roll = float(math.atan2(float(R_th[2, 1]), float(R_th[2, 2])))

        # 1. Per-leg ground contact forces in world & body frame
        c_found, c_forces, _, _, _, _ = self.sim.get_ground_contact_info(self.nmf.name)
        leg_names_fg = ("lf", "lm", "lh", "rf", "rm", "rh")
        leg_names_nca = ("L1_front_left", "L2_mid_left", "L3_hind_left", "R1_front_right", "R2_mid_right", "R3_hind_right")
        leg_contacts: dict[str, Any] = {}
        for i, (fg_n, nca_n) in enumerate(zip(leg_names_fg, leg_names_nca)):
            f_vec = [round(float(x), 4) for x in c_forces[i]]
            f_norm = round(float(np.linalg.norm(c_forces[i])), 4)
            leg_contacts[nca_n] = {
                "flygym_leg": fg_n,
                "touching_ground": bool(c_found[i] > 0),
                "force_xyz_mN": f_vec,
                "force_norm_mN": f_norm,
            }

        # 2. Detailed raw MuJoCo contact list (ground vs self-collisions)
        self_collisions: list[dict[str, Any]] = []
        ground_contacts_raw: list[dict[str, Any]] = []
        f6 = np.zeros(6, dtype=np.float64)
        for ci in range(int(d.ncon)):
            con = d.contact[ci]
            g1 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(con.geom1)) or f"geom_{con.geom1}"
            g2 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(con.geom2)) or f"geom_{con.geom2}"
            if not (g1.startswith(prefix) or g2.startswith(prefix)):
                continue
            mj.mj_contactForce(m, d, ci, f6)
            entry = {
                "geom1": g1,
                "geom2": g2,
                "dist_mm": round(float(con.dist), 5),
                "normal_force_mN": round(float(abs(f6[0])), 4),
                "friction_force_mN": round(float(np.linalg.norm(f6[1:3])), 4),
            }
            if g1.startswith(prefix) and g2.startswith(prefix):
                self_collisions.append(entry)
            else:
                ground_contacts_raw.append(entry)

        # 3. Joint limit hits across all actuated DOFs of this fly
        joint_limit_hits: list[dict[str, Any]] = []
        for jid in range(int(m.njnt)):
            jname = mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, jid) or ""
            if not jname.startswith(prefix) or int(m.jnt_limited[jid]) == 0:
                continue
            qadr = int(m.jnt_qposadr[jid])
            qval = float(d.qpos[qadr])
            jmin, jmax = float(m.jnt_range[jid, 0]), float(m.jnt_range[jid, 1])
            margin = min(abs(qval - jmin), abs(jmax - qval))
            if margin < 0.03:
                joint_limit_hits.append({
                    "joint": jname[len(prefix) :],
                    "angle_rad": round(qval, 4),
                    "range_rad": [round(jmin, 4), round(jmax, 4)],
                    "hit_bound": "min" if abs(qval - jmin) <= abs(jmax - qval) else "max",
                })

        return {
            "fly_name": self.nmf.name,
            "sim_time_ms": round(float(self.time_ms), 3),
            "thorax_pos_mm": [round(float(x), 4) for x in thorax_mm],
            "thorax_rpy_rad": [round(roll, 4), round(pitch, 4), round(yaw, 4)],
            "upright_z_cosine": round(float(R_th[2, 2]), 4),
            "leg_ground_contacts": leg_contacts,
            "n_legs_on_ground": int(sum(1 for v in leg_contacts.values() if v["touching_ground"])),
            "n_self_collisions": len(self_collisions),
            "self_collisions": self_collisions[:10],
            "n_joint_limit_hits": len(joint_limit_hits),
            "joint_limit_hits": joint_limit_hits[:12],
            "ground_contacts_detail": ground_contacts_raw[:12],
        }

    def render_diagnostic_views(
        self,
        output_path: Path | str,
        *,
        width: int = 480,
        height: int = 360,
        show_contact_forces: bool = True,
    ) -> Path:
        """Render a 3-panel orthographic diagnostic sheet (Side, Top, Front-ISO) with MuJoCo contact forces."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        m = self.sim.mj_model
        d = self.sim.mj_data
        thorax_mm = self.sim.get_body_positions(self.nmf.name)[0]

        renderer = mj.Renderer(m, height=height, width=width)
        vopt = mj.MjvOption()
        if show_contact_forces:
            vopt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            vopt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        views = [
            ("SIDE", 0.0, -15.0, 5.2),
            ("TOP", 90.0, -89.0, 6.0),
            ("ISO_3D", 135.0, -25.0, 5.0),
        ]
        panels: list[np.ndarray] = []
        for _, az, el, dist in views:
            cam = mj.MjvCamera()
            mj.mjv_defaultFreeCamera(m, cam)
            cam.lookat[:] = thorax_mm
            cam.azimuth = az
            cam.elevation = el
            cam.distance = dist
            renderer.update_scene(d, camera=cam, scene_option=vopt)
            panels.append(renderer.render().copy())
        renderer.close()

        combined = np.concatenate(panels, axis=1)
        Image.fromarray(combined).save(out)
        return out


def _set_fly_geoms_visibility(sim: Simulation, nmf_obj: NeuroMechFly, visible: bool) -> None:
    """Toggle visibility of a pre-allocated NeuroMechFly body without mutating qpos/qvel."""
    prefix = f"{nmf_obj.name}/"
    for gid in range(sim.mj_model.ngeom):
        gname = mj.mj_id2name(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, gid)
        if gname and gname.startswith(prefix):
            sim.mj_model.geom_rgba[gid, 3] = 1.0 if visible else 0.0


def _eclose_pupa_into_mujoco_mecha(
    fo: SpawnedFly,
    r_name: str,
    nmf_new: NeuroMechFly,
    sim: Simulation,
    nmf_map: dict[str, NeuroMechFly],
    free_qpos_by_body: dict[int, int],
    free_qvel_by_body: dict[int, int],
) -> None:
    """Reveal a newly eclosed G1/G2 adult NeuroMechFly mecha on the agar dish and connect its 189,766-cell brain."""
    _ = (r_name, free_qpos_by_body, free_qvel_by_body)
    nmf_map[fo.genome.fly_id] = nmf_new
    if fo.genome.dsx_isoform == "DsxM":
        for seg, geoms in nmf_new.bodyseg_to_mjcfgeom.items():
            if seg.name in ("c_abdomen5", "c_abdomen6"):
                for g in geoms:
                    g.material = ""
                    g.rgba = (0.12, 0.10, 0.09, 1.0)
                    gid = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, f"{nmf_new.name}/{g.name}")
                    if gid >= 0:
                        sim.mj_model.geom_rgba[gid] = [0.12, 0.10, 0.09, 1.0]
    _set_fly_geoms_visibility(sim, nmf_new, visible=True)
    MuJoCoFlyBodyAdapter(sim, nmf_new).connect(fo)


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
    # Every BANC leg motor pool maps directly to its anatomical NeuroMechFly joint!
    leg_map = ("lf", "rf", "lm", "rm", "lh", "rh")
    lj = motor.leg_joint_drive.detach().cpu().numpy()
    for leg_i, fg_leg in enumerate(leg_map):
        coxa_fwd = float(lj[leg_i, 0])
        troch_lift = float(lj[leg_i, 1])
        tibia_ext = float(lj[leg_i, 2])
        tarsus_flx = float(lj[leg_i, 3])

        # Direct 1:1 anatomical joint mapping from VNC motor output (zero artificial offsets):
        if fg_leg in ("lf", "rf"):
            if f"c_thorax-{fg_leg}_coxa-pitch" in dof_idx:
                ctrl[dof_idx[f"c_thorax-{fg_leg}_coxa-pitch"]] += coxa_fwd
            if f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch"]] -= troch_lift
            if f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch"]] += tibia_ext
            if f"{fg_leg}_tibia-{fg_leg}_tarsus1-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_tibia-{fg_leg}_tarsus1-pitch"]] -= tarsus_flx
        else:
            if f"c_thorax-{fg_leg}_coxa-roll" in dof_idx:
                ctrl[dof_idx[f"c_thorax-{fg_leg}_coxa-roll"]] += coxa_fwd
            if f"c_thorax-{fg_leg}_coxa-pitch" in dof_idx:
                ctrl[dof_idx[f"c_thorax-{fg_leg}_coxa-pitch"]] += coxa_fwd
            if f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_coxa-{fg_leg}_trochanterfemur-pitch"]] -= troch_lift - float(motor.jump_ttmn_burst)
            if f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_trochanterfemur-{fg_leg}_tibia-pitch"]] += tibia_ext
            if f"{fg_leg}_tibia-{fg_leg}_tarsus1-pitch" in dof_idx:
                ctrl[dof_idx[f"{fg_leg}_tibia-{fg_leg}_tarsus1-pitch"]] -= tarsus_flx

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

    # 4. Wing Motor Readout: Bilateral Flight Power/Pitch vs Unilateral Courtship Song (P1 -> pIP10 -> TN1)
    w_l = float(motor.wing_drive_lr[0].item())
    w_r = float(motor.wing_drive_lr[1].item())
    p_l = float(motor.wing_pitch_lr[0].item())
    p_r = float(motor.wing_pitch_lr[1].item())
    mean_wing = 0.5 * (w_l + w_r)
    song_amp = float(motor.wing_song_mode[0].item())
    dlm_dvm_diff = float(motor.wing_song_mode[1].item())

    if mean_wing > 0.45 or motor.jump_ttmn_burst > 0.4:
        if "c_thorax-l_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-yaw"]] += 0.95 + 0.85 * w_l + 0.65 * p_l
        if "c_thorax-r_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-yaw"]] += 0.95 + 0.85 * w_r + 0.65 * p_r
        if "c_thorax-l_wing-pitch" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-pitch"]] += 0.45 * p_l
        if "c_thorax-r_wing-pitch" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-pitch"]] += 0.45 * p_r
        if "c_thorax-l_wing-roll" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-roll"]] += 0.65 * w_l
        if "c_thorax-r_wing-roll" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-roll"]] -= 0.65 * w_r
    elif wing_song_active or song_amp > 0.02:
        # Unilateral male courtship song: extend one wing laterally to ~60 deg (+0.95 rad yaw)
        # and vibrate it with a fine, realistic small-amplitude tremor (+/- 0.09 rad ~ +/- 5 deg), keeping the other wing folded!
        ext_base = max(0.65, song_amp)
        if "c_thorax-l_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-yaw"]] += 0.95 * ext_base + 0.10 * dlm_dvm_diff
        if "c_thorax-l_wing-pitch" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-pitch"]] += 0.30 * ext_base + 0.12 * dlm_dvm_diff
        if "c_thorax-l_wing-roll" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-roll"]] += 0.35 * ext_base + 0.08 * dlm_dvm_diff
        if "c_thorax-r_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-yaw"]] -= 0.12
        if "c_thorax-r_wing-roll" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-roll"]] += 0.18
    else:
        if "c_thorax-l_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-yaw"]] -= 0.12
        if "c_thorax-r_wing-yaw" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-yaw"]] -= 0.12
        if "c_thorax-l_wing-roll" in dof_idx:
            ctrl[dof_idx["c_thorax-l_wing-roll"]] -= 0.18
        if "c_thorax-r_wing-roll" in dof_idx:
            ctrl[dof_idx["c_thorax-r_wing-roll"]] += 0.18

    # 5. Haltere motor drive (l_haltere / r_haltere pitch oscillation) & Female Oviposition Abdomen Bending
    h_l = float(motor.haltere_drive[0].item())
    h_r = float(motor.haltere_drive[1].item())
    if "c_thorax-l_haltere-pitch" in dof_idx:
        ctrl[dof_idx["c_thorax-l_haltere-pitch"]] += 0.35 * (h_l - 0.5)
    if "c_thorax-r_haltere-pitch" in dof_idx:
        ctrl[dof_idx["c_thorax-r_haltere-pitch"]] += 0.35 * (h_r - 0.5)

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
        spawner.spawn_fly(sex="F", foraging_allele="rover", pos_xyz=(0.0038, 0.0010, 0.0), yaw_rad=-2.88),
        spawner.spawn_fly(sex="M", foraging_allele="sitter", pos_xyz=(-0.0038, -0.0008, 0.0), yaw_rad=0.21),
        spawner.spawn_fly(sex="F", foraging_allele="sitter", pos_xyz=(0.0010, -0.0040, 0.0), yaw_rad=1.81),
        spawner.spawn_fly(sex="M", foraging_allele="rover", pos_xyz=(-0.0036, 0.0034, 0.0), yaw_rad=-0.75),
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
            size=(1.22, 0.28, 1.60),
            rgba=(0.88, 0.94, 0.98, 0.30),
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
    # Sucrose crystal droplets on the agar surface (environmental sugar food sources)
    for sc_i, (sx, sy) in enumerate(((1.2, 0.8), (-1.1, 1.0), (0.6, -1.3))):
        world.mjcf_root.worldbody.add_geom(
            name=f"sucrose_crystal_{sc_i}",
            type=mj.mjtGeom.mjGEOM_ELLIPSOID,
            pos=(sx, sy, 0.06),
            size=(0.24, 0.22, 0.08),
            rgba=(0.96, 0.98, 1.0, 0.85),
            contype=0,
            conaffinity=0,
        )
    # Pre-allocate 6 segmented offspring slots: each slot has 7 segments (6 metameric body rings + 1 anterior mouth-hook/spiracle cap)
    # so a crawling Drosophila larva looks like a real segmented white maggot with black mouth hooks, not a brown blob!
    for off_idx in range(6):
        for seg_m in range(7):
            world.mjcf_root.worldbody.add_geom(
                name=f"f1_offspring_{off_idx}_seg_{seg_m}",
                type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                pos=(0.0, 0.0, -5.0),
                size=(0.14, 0.12, 0.11),
                rgba=(0.98, 0.99, 0.94, 0.0),
                contype=0,
                conaffinity=0,
            )

    # Build clean Mecha skeleton using FlyGym's native LEGS_ONLY (includes passive tarsal joints!) + active LEGS_ACTIVE_ONLY actuators
    s_legs_full = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
    s_legs_act = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ACTIVE_ONLY)
    s_all = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_BIOLOGICAL)
    leg_full_set = set(s_legs_full.anatomical_joints)
    leg_act_set = set(s_legs_act.anatomical_joints)
    extra_body_names = ("l_wing", "r_wing", "l_haltere", "r_haltere", "c_head", "c_rostrum", "c_haustellum", "c_abdomen12")
    clean_anat_joints = [
        aj
        for aj in s_all.anatomical_joints
        if (aj in leg_full_set) or (aj.child.name in extra_body_names)
    ]
    act_anat_joints = [
        aj
        for aj in s_all.anatomical_joints
        if (aj in leg_act_set) or (aj.child.name in extra_body_names)
    ]
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=clean_anat_joints)
    act_skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=act_anat_joints)
    act_dofs = list(act_skeleton.iter_jointdofs())
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
        nmf.add_actuators(act_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL)
        nmf.add_leg_adhesion()
        nmf.colorize()
        if f.genome.dsx_isoform == "DsxM":
            # Male DsxM suppresses bric-a-brac (bab) -> melanized dark-black A5/A6 abdominal tip!
            for seg, geoms in nmf.bodyseg_to_mjcfgeom.items():
                if seg.name in ("c_abdomen5", "c_abdomen6"):
                    for g in geoms:
                        g.material = ""
                        g.rgba = (0.12, 0.10, 0.09, 1.0)
        px_mm = f.pos_xyz[0] * 1000.0
        py_mm = f.pos_xyz[1] * 1000.0
        yaw = float(f.arena.thorax_rpy[2].item())
        q = (float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0)))
        world.add_fly(nmf, spawn_position=(px_mm, py_mm, 0.0), spawn_rotation=Rotation3D("quat", q))
        nmf_map[f.genome.fly_id] = nmf

    reserve_nmf_names = ["g1_eclosed_adult_0", "g1_eclosed_adult_1", "g1_eclosed_adult_2"]
    reserve_spots = ((-4.8, 4.2, 0.0), (4.8, -4.2, 0.0), (4.2, 4.6, 0.0))
    reserve_nmf_objs: dict[str, NeuroMechFly] = {}
    for r_idx, r_name in enumerate(reserve_nmf_names):
        r_nmf = NeuroMechFly(name=r_name)
        r_nmf.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
        r_nmf.add_actuators(act_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL)
        r_nmf.add_leg_adhesion()
        r_nmf.colorize()
        world.add_fly(
            r_nmf,
            spawn_position=reserve_spots[r_idx],
            spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        )
        reserve_nmf_objs[r_name] = r_nmf

    all_fly_objs = list(nmf_map.values()) + list(reserve_nmf_objs.values())
    sim = Simulation(world)
    sim.reset()
    for any_nmf in all_fly_objs:
        ad_order = any_nmf.get_actuated_jointdofs_order(ActuatorType.POSITION)
        n_c = np.array(
            [any_nmf.jointdof_to_neutralaction_by_type[ActuatorType.POSITION][j] for j in ad_order],
            dtype=np.float64,
        )
        sim.set_actuator_inputs(any_nmf.name, ActuatorType.POSITION, n_c)
        sim.set_leg_adhesion_states(any_nmf.name, np.ones(6, dtype=np.float64))
    sim.warmup(0.012)
    for r_nmf in reserve_nmf_objs.values():
        _set_fly_geoms_visibility(sim, r_nmf, visible=False)
    for f in colony.flies:
        nmf_init = nmf_map.get(f.genome.fly_id)
        if nmf_init is not None:
            MuJoCoFlyBodyAdapter(sim, nmf_init).connect(f)

    mj_renderer = mj.Renderer(sim.mj_model, height=camera_res[0], width=camera_res[1])
    cam_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, world_cam.name)
    off_geom_ids = [
        [mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, f"f1_offspring_{k}_seg_{m}") for m in range(7)]
        for k in range(6)
    ]
    free_qpos_by_body: dict[int, int] = {}
    free_qvel_by_body: dict[int, int] = {}
    for jid in range(sim.mj_model.njnt):
        if int(sim.mj_model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE):
            bid = int(sim.mj_model.jnt_bodyid[jid])
            free_qpos_by_body[bid] = int(sim.mj_model.jnt_qposadr[jid])
            free_qvel_by_body[bid] = int(sim.mj_model.jnt_dofadr[jid])

    history_frames: list[list[dict[str, Any]]] = []
    frame_png_b64: list[str] = []

    n_total_frames = max(1, int(frames))
    for step_i in range(n_total_frames):
        snaps = colony.step_population(frames=1, enable_ontogeny_and_reproduction=True)
        history_frames.append(snaps)

        # Synchronize MuJoCo world time with Fly-NCA cognitive frame time (1 cognitive frame = 4.038 ms = 40 x 0.1 ms MuJoCo steps!)
        for sub_k in range(40):
            global_sub = step_i * 40 + sub_k
            for f in colony.flies:
                if isinstance(f.arena, MuJoCoFlyBodyAdapter):
                    f.arena.apply_motor_to_mujoco(step_idx=global_sub)
            sim.step()
            for f in colony.flies:
                if isinstance(f.arena, MuJoCoFlyBodyAdapter):
                    f.arena.sync_from_mujoco(full_proprioception=(sub_k == 39))

        # Update dynamic 3D segmented geoms for any real F1 offspring (egg -> segmented white maggot larva -> amber pupa)
        # When a G1/G2 pupa ecloses into an adult, bind its 189,766-cell brain to a real 6-legged NeuroMechFly mecha at the pupa's position!
        for fo in colony.flies:
            if fo.life_stage == "adult" and fo.genome.fly_id not in nmf_map and reserve_nmf_names:
                r_name = reserve_nmf_names.pop(0)
                nmf_new = reserve_nmf_objs[r_name]
                _eclose_pupa_into_mujoco_mecha(fo, r_name, nmf_new, sim, nmf_map, free_qpos_by_body, free_qvel_by_body)

        offspring_flies = [f for f in colony.flies if f.genome.generation >= 1]
        for k_off, seg_gids in enumerate(off_geom_ids):
            if k_off < len(offspring_flies):
                fo = offspring_flies[k_off]
                bx_mm = fo.pos_xyz[0] * 1000.0
                by_mm = fo.pos_xyz[1] * 1000.0
                dir_ang = float(fo.arena.thorax_rpy[2].item())
                ca, sa = math.cos(dir_ang), math.sin(dir_ang)
                if fo.life_stage == "egg":
                    # Chorion-white ovoid egg with 2 dorsal respiratory filaments
                    sim.mj_model.geom_pos[seg_gids[0]] = [bx_mm, by_mm, 0.16]
                    sim.mj_model.geom_size[seg_gids[0]] = [0.36, 0.15, 0.14]
                    sim.mj_model.geom_rgba[seg_gids[0]] = [0.99, 0.99, 0.97, 1.0]
                    for m_i in range(1, 7):
                        sim.mj_model.geom_pos[seg_gids[m_i]] = [0.0, 0.0, -5.0]
                        sim.mj_model.geom_rgba[seg_gids[m_i], 3] = 0.0
                elif fo.life_stage == "larva":
                    # Segmented pearly-white Drosophila maggot (6 tapered metameric rings + black anterior mouth hooks!)
                    # Each larval segment m_i=0..5 is driven directly by the larva's own Wave 1 connectome `leg_joint_drive[m_i]`!
                    lm_out = fo.full_connectome.decode_motor_effectors()
                    seg_drives = lm_out.leg_joint_drive.detach().cpu().numpy()
                    head_cast = float(lm_out.neck_torque_rpy[2].item())
                    radii = (0.14, 0.21, 0.25, 0.26, 0.23, 0.17)
                    for m_i in range(6):
                        wave = 0.10 * float(seg_drives[m_i, 0])
                        along = (2.5 - m_i) * 0.24 + wave
                        lat = 0.06 * float(seg_drives[m_i, 1] - 0.5) + (0.08 * head_cast if m_i >= 4 else 0.0)
                        px_s = bx_mm + along * ca - lat * sa
                        py_s = by_mm + along * sa + lat * ca
                        r_s = radii[m_i]
                        sim.mj_model.geom_pos[seg_gids[m_i]] = [px_s, py_s, r_s * 0.85]
                        sim.mj_model.geom_size[seg_gids[m_i]] = [r_s * 1.15, r_s, r_s * 0.85]
                        c_shade = 1.0 if (m_i % 2 == 0) else 0.90
                        sim.mj_model.geom_rgba[seg_gids[m_i]] = [c_shade, c_shade, 0.98 * c_shade, 1.0]
                    # Black chitinous mouth hooks at the anterior tip (segment 6), driven by proboscis_per & head_cast!
                    mh_along = 3.25 * 0.24 + 0.08 * float(lm_out.proboscis_per)
                    mh_lat = 0.10 * head_cast
                    mh_x = bx_mm + mh_along * ca - mh_lat * sa
                    mh_y = by_mm + mh_along * sa + mh_lat * ca
                    sim.mj_model.geom_pos[seg_gids[6]] = [mh_x, mh_y, 0.10]
                    sim.mj_model.geom_size[seg_gids[6]] = [0.09, 0.07, 0.07]
                    sim.mj_model.geom_rgba[seg_gids[6]] = [0.08, 0.07, 0.06, 1.0]
                elif fo.life_stage == "pupa":
                    # Golden-amber segmented puparium with anterior spiracle horns
                    radii_p = (0.16, 0.23, 0.27, 0.27, 0.23, 0.17)
                    for m_i in range(6):
                        along = (2.5 - m_i) * 0.23
                        px_s = bx_mm + along * ca
                        py_s = by_mm + along * sa
                        r_s = radii_p[m_i]
                        sim.mj_model.geom_pos[seg_gids[m_i]] = [px_s, py_s, r_s * 0.9]
                        sim.mj_model.geom_size[seg_gids[m_i]] = [r_s * 1.15, r_s, r_s * 0.9]
                        sim.mj_model.geom_rgba[seg_gids[m_i]] = [0.82, 0.58, 0.24, 0.96]
                    sim.mj_model.geom_pos[seg_gids[6]] = [bx_mm + 3.2 * 0.23 * ca, by_mm + 3.2 * 0.23 * sa, 0.18]
                    sim.mj_model.geom_size[seg_gids[6]] = [0.09, 0.12, 0.06]
                    sim.mj_model.geom_rgba[seg_gids[6]] = [0.45, 0.28, 0.10, 1.0]
                else:
                    # Eclosed adult: hide the temporary developmental geoms since the adult now drives its own 6-legged NeuroMechFly!
                    for gid in seg_gids:
                        if gid >= 0:
                            sim.mj_model.geom_pos[gid] = [0.0, 0.0, -5.0]
                            sim.mj_model.geom_rgba[gid, 3] = 0.0
            else:
                for gid in seg_gids:
                    if gid >= 0:
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
        spawner.spawn_fly(sex="F", foraging_allele="rover", pos_xyz=(0.0038, 0.0010, 0.0), yaw_rad=-2.88),
        spawner.spawn_fly(sex="M", foraging_allele="sitter", pos_xyz=(-0.0038, -0.0008, 0.0), yaw_rad=0.21),
        spawner.spawn_fly(sex="F", foraging_allele="sitter", pos_xyz=(0.0010, -0.0040, 0.0), yaw_rad=1.81),
        spawner.spawn_fly(sex="M", foraging_allele="rover", pos_xyz=(-0.0036, 0.0034, 0.0), yaw_rad=-0.75),
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
            size=(1.22, 0.28, 1.60),
            rgba=(0.88, 0.94, 0.98, 0.30),
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
    # Sucrose crystal droplets on the agar surface (environmental sugar food sources)
    for sc_i, (sx, sy) in enumerate(((1.2, 0.8), (-1.1, 1.0), (0.6, -1.3))):
        world.mjcf_root.worldbody.add_geom(
            name=f"sucrose_crystal_{sc_i}",
            type=mj.mjtGeom.mjGEOM_ELLIPSOID,
            pos=(sx, sy, 0.06),
            size=(0.24, 0.22, 0.08),
            rgba=(0.96, 0.98, 1.0, 0.85),
            contype=0,
            conaffinity=0,
        )
    for off_idx in range(6):
        for seg_m in range(7):
            world.mjcf_root.worldbody.add_geom(
                name=f"f1_offspring_{off_idx}_seg_{seg_m}",
                type=mj.mjtGeom.mjGEOM_ELLIPSOID,
                pos=(0.0, 0.0, -5.0),
                size=(0.14, 0.12, 0.11),
                rgba=(0.98, 0.99, 0.94, 0.0),
                contype=0,
                conaffinity=0,
            )

    s_legs_full = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ONLY)
    s_legs_act = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.LEGS_ACTIVE_ONLY)
    s_all = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_BIOLOGICAL)
    leg_full_set = set(s_legs_full.anatomical_joints)
    leg_act_set = set(s_legs_act.anatomical_joints)
    extra_body_names = ("l_wing", "r_wing", "l_haltere", "r_haltere", "c_head", "c_rostrum", "c_haustellum", "c_abdomen12")
    clean_anat_joints = [
        aj
        for aj in s_all.anatomical_joints
        if (aj in leg_full_set) or (aj.child.name in extra_body_names)
    ]
    act_anat_joints = [
        aj
        for aj in s_all.anatomical_joints
        if (aj in leg_act_set) or (aj.child.name in extra_body_names)
    ]
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=clean_anat_joints)
    act_skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL, anatomical_joints=act_anat_joints)
    act_dofs = list(act_skeleton.iter_jointdofs())
    nmf_map: dict[str, NeuroMechFly] = {}
    world_cam = world.mjcf_root.worldbody.add_camera(
        name="arena_overview_cam",
        pos=(-14.5, -14.5, 12.2),
        xyaxes=(0.707, -0.707, 0.0, 0.40, 0.40, 0.82),
    )
    for f in flies:
        nmf = NeuroMechFly(name=f.genome.fly_id)
        nmf.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
        nmf.add_actuators(act_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL)
        nmf.add_leg_adhesion()
        nmf.colorize()
        if f.genome.dsx_isoform == "DsxM":
            for seg, geoms in nmf.bodyseg_to_mjcfgeom.items():
                if seg.name in ("c_abdomen5", "c_abdomen6"):
                    for g in geoms:
                        g.material = ""
                        g.rgba = (0.12, 0.10, 0.09, 1.0)
        yaw = float(f.arena.thorax_rpy[2].item())
        q = (float(math.cos(yaw / 2.0)), 0.0, 0.0, float(math.sin(yaw / 2.0)))
        world.add_fly(nmf, spawn_position=(f.pos_xyz[0] * 1000.0, f.pos_xyz[1] * 1000.0, 0.0), spawn_rotation=Rotation3D("quat", q))
        nmf_map[f.genome.fly_id] = nmf

    reserve_nmf_names = ["g1_eclosed_adult_0", "g1_eclosed_adult_1", "g1_eclosed_adult_2"]
    reserve_spots = ((-4.8, 4.2, 0.0), (4.8, -4.2, 0.0), (4.2, 4.6, 0.0))
    reserve_nmf_objs: dict[str, NeuroMechFly] = {}
    for r_idx, r_name in enumerate(reserve_nmf_names):
        r_nmf = NeuroMechFly(name=r_name)
        r_nmf.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
        r_nmf.add_actuators(act_dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL)
        r_nmf.add_leg_adhesion()
        r_nmf.colorize()
        world.add_fly(
            r_nmf,
            spawn_position=reserve_spots[r_idx],
            spawn_rotation=Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)),
        )
        reserve_nmf_objs[r_name] = r_nmf

    all_fly_objs = list(nmf_map.values()) + list(reserve_nmf_objs.values())
    sim = Simulation(world)
    sim.reset()
    for any_nmf in all_fly_objs:
        ad_order = any_nmf.get_actuated_jointdofs_order(ActuatorType.POSITION)
        n_c = np.array(
            [any_nmf.jointdof_to_neutralaction_by_type[ActuatorType.POSITION][j] for j in ad_order],
            dtype=np.float64,
        )
        sim.set_actuator_inputs(any_nmf.name, ActuatorType.POSITION, n_c)
        sim.set_leg_adhesion_states(any_nmf.name, np.ones(6, dtype=np.float64))
    sim.warmup(0.012)
    for r_nmf in reserve_nmf_objs.values():
        _set_fly_geoms_visibility(sim, r_nmf, visible=False)
    for f in colony.flies:
        nmf_init = nmf_map.get(f.genome.fly_id)
        if nmf_init is not None:
            MuJoCoFlyBodyAdapter(sim, nmf_init).connect(f)

    mj_renderer = mj.Renderer(sim.mj_model, height=camera_res[0], width=camera_res[1])
    cam_id = mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_CAMERA, world_cam.name)
    off_geom_ids = [
        [mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, f"f1_offspring_{k}_seg_{m}") for m in range(7)]
        for k in range(6)
    ]
    free_qpos_by_body: dict[int, int] = {}
    free_qvel_by_body: dict[int, int] = {}
    for jid in range(sim.mj_model.njnt):
        if int(sim.mj_model.jnt_type[jid]) == int(mj.mjtJoint.mjJNT_FREE):
            bid = int(sim.mj_model.jnt_bodyid[jid])
            free_qpos_by_body[bid] = int(sim.mj_model.jnt_qposadr[jid])
            free_qvel_by_body[bid] = int(sim.mj_model.jnt_dofadr[jid])

    step_i = 0
    # Launch the real native interactive MuJoCo 3D Desktop GUI (mujoco.viewer.launch_passive)
    # so the user can freely rotate/pan/zoom the 3D camera, inspect joints/forces, and watch in 3D alongside SSE!
    import mujoco.viewer as mj_viewer

    passive_viewer = None
    try:
        passive_viewer = mj_viewer.launch_passive(sim.mj_model, sim.mj_data)
        passive_viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        passive_viewer.cam.distance = 22.0
        passive_viewer.cam.elevation = -32.0
        passive_viewer.cam.azimuth = 45.0
    except Exception:
        passive_viewer = None

    while True:
        if passive_viewer is not None and not passive_viewer.is_running():
            passive_viewer = None
        scale_val = max(0.05, float(latest_state.get("scale", 1.0)))
        n_brain_frames = max(1, min(8, int(round(scale_val))))
        snaps = []
        for _ in range(n_brain_frames):
            snaps = colony.step_population(frames=1, enable_ontogeny_and_reproduction=True)
            # Unified Clock: 1 cognitive frame (4.038 ms) = 40 MuJoCo physics steps (40 x 0.1 ms = 4.0 ms)!
            for sub_k in range(40):
                global_sub = step_i * 40 + sub_k
                for f in colony.flies:
                    if isinstance(f.arena, MuJoCoFlyBodyAdapter):
                        f.arena.apply_motor_to_mujoco(step_idx=global_sub)
                sim.step()
                if passive_viewer is not None and sub_k % 20 == 0:
                    passive_viewer.sync()
                for f in colony.flies:
                    if isinstance(f.arena, MuJoCoFlyBodyAdapter):
                        f.arena.sync_from_mujoco(full_proprioception=(sub_k == 39))
            step_i += 1
        if passive_viewer is not None:
            passive_viewer.sync()

        # When a G1/G2 pupa ecloses into an adult, bind its 189,766-cell brain to a real 6-legged NeuroMechFly mecha at the pupa's position!
        for fo in colony.flies:
            if fo.life_stage == "adult" and fo.genome.fly_id not in nmf_map and reserve_nmf_names:
                r_name = reserve_nmf_names.pop(0)
                nmf_new = reserve_nmf_objs[r_name]
                _eclose_pupa_into_mujoco_mecha(fo, r_name, nmf_new, sim, nmf_map, free_qpos_by_body, free_qvel_by_body)

        offspring_flies = [f for f in colony.flies if f.genome.generation >= 1]
        for k_off, seg_gids in enumerate(off_geom_ids):
            if k_off < len(offspring_flies):
                fo = offspring_flies[k_off]
                bx_mm = fo.pos_xyz[0] * 1000.0
                by_mm = fo.pos_xyz[1] * 1000.0
                dir_ang = float(fo.arena.thorax_rpy[2].item())
                ca, sa = math.cos(dir_ang), math.sin(dir_ang)
                if fo.life_stage == "egg":
                    sim.mj_model.geom_pos[seg_gids[0]] = [bx_mm, by_mm, 0.16]
                    sim.mj_model.geom_size[seg_gids[0]] = [0.36, 0.15, 0.14]
                    sim.mj_model.geom_rgba[seg_gids[0]] = [0.99, 0.99, 0.97, 1.0]
                    for m_i in range(1, 7):
                        sim.mj_model.geom_pos[seg_gids[m_i]] = [0.0, 0.0, -5.0]
                        sim.mj_model.geom_rgba[seg_gids[m_i], 3] = 0.0
                elif fo.life_stage == "larva":
                    lm_out = fo.full_connectome.decode_motor_effectors()
                    seg_drives = lm_out.leg_joint_drive.detach().cpu().numpy()
                    head_cast = float(lm_out.neck_torque_rpy[2].item())
                    radii = (0.14, 0.21, 0.25, 0.26, 0.23, 0.17)
                    for m_i in range(6):
                        wave = 0.10 * float(seg_drives[m_i, 0])
                        along = (2.5 - m_i) * 0.24 + wave
                        lat = 0.06 * float(seg_drives[m_i, 1] - 0.5) + (0.08 * head_cast if m_i >= 4 else 0.0)
                        px_s = bx_mm + along * ca - lat * sa
                        py_s = by_mm + along * sa + lat * ca
                        r_s = radii[m_i]
                        sim.mj_model.geom_pos[seg_gids[m_i]] = [px_s, py_s, r_s * 0.85]
                        sim.mj_model.geom_size[seg_gids[m_i]] = [r_s * 1.15, r_s, r_s * 0.85]
                        c_shade = 1.0 if (m_i % 2 == 0) else 0.90
                        sim.mj_model.geom_rgba[seg_gids[m_i]] = [c_shade, c_shade, 0.98 * c_shade, 1.0]
                    mh_along = 3.25 * 0.24 + 0.08 * float(lm_out.proboscis_per)
                    mh_lat = 0.10 * head_cast
                    mh_x = bx_mm + mh_along * ca - mh_lat * sa
                    mh_y = by_mm + mh_along * sa + mh_lat * ca
                    sim.mj_model.geom_pos[seg_gids[6]] = [mh_x, mh_y, 0.10]
                    sim.mj_model.geom_size[seg_gids[6]] = [0.09, 0.07, 0.07]
                    sim.mj_model.geom_rgba[seg_gids[6]] = [0.08, 0.07, 0.06, 1.0]
                elif fo.life_stage == "pupa":
                    radii_p = (0.16, 0.23, 0.27, 0.27, 0.23, 0.17)
                    for m_i in range(6):
                        along = (2.5 - m_i) * 0.23
                        px_s = bx_mm + along * ca
                        py_s = by_mm + along * sa
                        r_s = radii_p[m_i]
                        sim.mj_model.geom_pos[seg_gids[m_i]] = [px_s, py_s, r_s * 0.9]
                        sim.mj_model.geom_size[seg_gids[m_i]] = [r_s * 1.15, r_s, r_s * 0.9]
                        sim.mj_model.geom_rgba[seg_gids[m_i]] = [0.82, 0.58, 0.24, 0.96]
                    sim.mj_model.geom_pos[seg_gids[6]] = [bx_mm + 3.2 * 0.23 * ca, by_mm + 3.2 * 0.23 * sa, 0.18]
                    sim.mj_model.geom_size[seg_gids[6]] = [0.09, 0.12, 0.06]
                    sim.mj_model.geom_rgba[seg_gids[6]] = [0.45, 0.28, 0.10, 1.0]
                else:
                    for gid in seg_gids:
                        if gid >= 0:
                            sim.mj_model.geom_pos[gid] = [0.0, 0.0, -5.0]
                            sim.mj_model.geom_rgba[gid, 3] = 0.0
            else:
                for gid in seg_gids:
                    if gid >= 0:
                        sim.mj_model.geom_pos[gid] = [0.0, 0.0, -5.0]
                        sim.mj_model.geom_rgba[gid, 3] = 0.0
        mj.mj_forward(sim.mj_model, sim.mj_data)
        if passive_viewer is not None:
            passive_viewer.sync()

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
        time.sleep(0.01 if scale_val >= 1.0 else max(0.02, 0.01 / scale_val))
