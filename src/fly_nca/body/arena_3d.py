"""FastTensorArena3D: Analytical 3D Physics & Sensory Arena for Fly-NCA."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch

from fly_nca.body.adapter import MotorOut, SensorIn
from fly_nca.core.cell_state import TAU_FAST_MS, TAU_FRAME_MS


@dataclass
class OdorPlume3D:
    """3D odor source in the arena emitting a wind-advected plume."""

    pos_xyz: tuple[float, float, float]
    glomerulus_id: int
    intensity: float = 1.0
    sigma_m: float = 0.08
    is_sweet_food: bool = False
    is_bitter_shock: bool = False
    is_yeast_protein: bool = False


@dataclass
class FastTensorArena3D:
    """3D physics arena (x, y, z) with 6x4 leg kinematics, 200-Hz flight aerodynamics, and 3D plumes."""

    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float32

    # 3D Spatial Coordinates: (x, y, z) in meters, simulation clock time_ms in milliseconds
    pos_xyz: torch.Tensor = field(init=False)
    vel_xyz: torch.Tensor = field(init=False)
    thorax_rpy: torch.Tensor = field(init=False)       # [3] (roll, pitch, yaw) in radians
    omega_rpy: torch.Tensor = field(init=False)        # [3] angular velocity (rad/s)
    neck_rpy: torch.Tensor = field(init=False)         # [3] neck joint angles relative to thorax
    leg_angles: torch.Tensor = field(init=False)       # [6, 4] joint angles in [-1, +1]
    leg_load: torch.Tensor = field(init=False)         # [6] campaniform ground load in [0, 1]
    bristle_touch: torch.Tensor = field(init=False)    # [6] tactile dust/touch on legs
    time_ms: float = 0.0

    # Environmental fields
    sun_azimuth_world_rad: float = 0.0
    wind_world_xyz: tuple[float, float, float] = (0.15, 0.0, 0.0)
    turbulence_torque_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    looming_rate: float = 0.0
    small_target_pos_xyz: tuple[float, float, float] | None = None
    temperature_c: float = 25.0
    surface_texture_freq: float = 1.0
    crop_stretch: float = 0.2
    plumes: list[OdorPlume3D] = field(default_factory=list)

    def __post_init__(self) -> None:
        dev = torch.device(self.device)
        self.pos_xyz = torch.zeros(3, device=dev, dtype=self.dtype)
        self.vel_xyz = torch.zeros(3, device=dev, dtype=self.dtype)
        self.thorax_rpy = torch.zeros(3, device=dev, dtype=self.dtype)
        self.omega_rpy = torch.zeros(3, device=dev, dtype=self.dtype)
        self.neck_rpy = torch.zeros(3, device=dev, dtype=self.dtype)
        self.leg_angles = torch.zeros((6, 4), device=dev, dtype=self.dtype)
        self.leg_load = torch.full((6,), 0.5, device=dev, dtype=self.dtype)
        self.bristle_touch = torch.zeros(6, device=dev, dtype=self.dtype)

    @property
    def head_world_rpy(self) -> torch.Tensor:
        """World-frame head orientation = thorax orientation + neck gimbal angles."""
        return self.thorax_rpy + self.neck_rpy

    @property
    def time_fast_quanta(self) -> float:
        """Elapsed organism time in natural fast reflex quanta (tau_fast = 1.711 ms)."""
        return self.time_ms / TAU_FAST_MS

    @property
    def time_frames(self) -> float:
        """Elapsed organism time in natural cognitive wingbeat frames (tau_frame = 4.038 ms)."""
        return self.time_ms / TAU_FRAME_MS

    def step_physics(self, motor: MotorOut, dt_ms: float = TAU_FAST_MS) -> SensorIn:
        """Advance 3D body kinematics, aerodynamics, and sensory fields by dt_ms (default 1 tau_fast = 1.711 ms)."""
        dt_s = float(dt_ms) * 1e-3
        self.time_ms += float(dt_ms)
        dev = self.pos_xyz.device

        # 1. Update 6 legs x 4 joints (1 cognitive wingbeat frame = TAU_FRAME_MS = 4.038 ms)
        alpha_joint = min(1.0, dt_ms / TAU_FRAME_MS)
        prev_coxa = self.leg_angles[:, 0].clone()
        self.leg_angles = ((1.0 - alpha_joint) * self.leg_angles + alpha_joint * motor.leg_joint_drive).clamp(-1.0, 1.0)
        d_coxa = self.leg_angles[:, 0] - prev_coxa

        # Ground contact & campaniform load: leg bears load when on surface (z <= 0.001) and tibia is flexed (> 0)
        on_ground = float(self.pos_xyz[2].item()) <= 0.001
        if on_ground:
            stance_mask = (self.leg_angles[:, 2] >= 0.0).to(self.dtype) * motor.tarsus_adhesion.clamp(0.0, 1.0)
            self.leg_load = (0.15 + 0.85 * stance_mask).clamp(0.0, 1.0)

            # Ground propulsion from stance legs:
            # Left legs = indices (0, 2, 4), Right legs = indices (1, 3, 5)
            # Positive coxa drive during stance pushes body forward; L - R difference turns yaw
            coxa_rate = motor.leg_joint_drive[:, 0] * stance_mask
            left_push = coxa_rate[[0, 2, 4]].mean()
            right_push = coxa_rate[[1, 3, 5]].mean()
            fwd_speed = 0.025 * float((left_push + right_push) * 0.5)  # m/s (~25 mm/s walking speed)
            yaw_rate_walk = 3.5 * float(left_push - right_push)       # rad/s
        else:
            self.leg_load.zero_()
            fwd_speed = 0.0
            yaw_rate_walk = 0.0
            _ = d_coxa

        # 2. Giant Fiber -> TTMn jump burst launches the fly into 3D flight within 1 ms!
        if motor.jump_ttmn_burst > 0.5 and on_ground:
            self.vel_xyz[2] = 0.35 * float(motor.jump_ttmn_burst)  # Vertical takeoff velocity (m/s)
            self.pos_xyz[2] = self.pos_xyz[2] + self.vel_xyz[2] * dt_s + 0.002
            on_ground = False

        # 3. 3D Flight Aerodynamics & Haltere / Wing Torques when airborne (or wing-assisted)
        w_l = float(motor.wing_drive_lr[0].item())
        w_r = float(motor.wing_drive_lr[1].item())
        p_l = float(motor.wing_pitch_lr[0].item())
        p_r = float(motor.wing_pitch_lr[1].item())
        mean_wing = 0.5 * (w_l + w_r)

        if not on_ground or mean_wing > 0.30:
            # Roll torque proportional to (w_l - w_r), Pitch torque to mean wing pitch, Yaw to differential pitch/drive
            tau_roll = 45.0 * (w_l - w_r) + self.turbulence_torque_rpy[0]
            tau_pitch = 45.0 * 0.5 * (p_l + p_r) + self.turbulence_torque_rpy[1]
            tau_yaw = 35.0 * ((w_l - w_r) * 0.6 + (p_l - p_r) * 0.4) + self.turbulence_torque_rpy[2]

            # Rotational damping in air + angular acceleration
            damping = min(0.95, 18.0 * dt_s)
            self.omega_rpy[0] = (1.0 - damping) * self.omega_rpy[0] + tau_roll * dt_s
            self.omega_rpy[1] = (1.0 - damping) * self.omega_rpy[1] + tau_pitch * dt_s
            self.omega_rpy[2] = (1.0 - damping) * self.omega_rpy[2] + tau_yaw * dt_s
            self.thorax_rpy = self.thorax_rpy + self.omega_rpy * dt_s

            # Lift & forward thrust in 3D space (hover equilibrium at mean_wing=0.45, up to ~2G climb)
            lift_acc = 9.81 * (mean_wing / 0.45) - 9.81 + 8.0 * 0.5 * (p_l + p_r)
            self.vel_xyz[2] = (1.0 - 2.0 * dt_s) * self.vel_xyz[2] + lift_acc * dt_s
            air_fwd = 0.25 * mean_wing
            yaw = float(self.thorax_rpy[2].item())
            self.vel_xyz[0] = air_fwd * math.cos(yaw)
            self.vel_xyz[1] = air_fwd * math.sin(yaw)
            self.pos_xyz = self.pos_xyz + self.vel_xyz * dt_s
            if float(self.pos_xyz[2].item()) < 0.0:
                self.pos_xyz[2] = 0.0
                self.vel_xyz[2] = 0.0
        else:
            # Ground walking kinematics
            self.omega_rpy[0] = 0.0
            self.omega_rpy[1] = 0.0
            self.omega_rpy[2] = yaw_rate_walk
            self.thorax_rpy[2] = self.thorax_rpy[2] + yaw_rate_walk * dt_s
            yaw = float(self.thorax_rpy[2].item())
            self.vel_xyz[0] = fwd_speed * math.cos(yaw)
            self.vel_xyz[1] = fwd_speed * math.sin(yaw)
            self.vel_xyz[2] = 0.0
            self.pos_xyz = self.pos_xyz + self.vel_xyz * dt_s

        # 4. Neck 3D Gimbal update (49 neck motor neurons integrate counter-rotation torque)
        self.neck_rpy = (self.neck_rpy + 25.0 * dt_s * motor.neck_torque_rpy - 2.0 * dt_s * self.neck_rpy).clamp(-0.8, 0.8)

        # Grooming clears dust from bristles when front/hind legs rub while standing
        if on_ground and abs(fwd_speed) < 0.002 and float(motor.leg_joint_drive[:, 3].abs().mean()) > 0.2:
            self.bristle_touch = (self.bristle_touch - 0.15 * dt_ms).clamp(0.0, 1.0)

        # 5. Compute Sensory Periphery (SensorIn)
        head_yaw = float(self.head_world_rpy[2].item())
        sun_ego = math.atan2(
            math.sin(self.sun_azimuth_world_rad - head_yaw),
            math.cos(self.sun_azimuth_world_rad - head_yaw),
        )

        # Wind vector rotated into head ego frame
        wx, wy, wz = self.wind_world_xyz
        cos_y = math.cos(-head_yaw)
        sin_y = math.sin(-head_yaw)
        wind_ego = torch.tensor(
            [wx * cos_y - wy * sin_y, wx * sin_y + wy * cos_y, wz],
            device=dev,
            dtype=self.dtype,
        )

        # Stereo antennal odor sampling (2 antennae separated by 0.3 mm left/right of head)
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

            # Contact taste if within 5 mm of source
            dist_body = math.sqrt((float(self.pos_xyz[0]) - px) ** 2 + (float(self.pos_xyz[1]) - py) ** 2 + (fz - pz) ** 2)
            if dist_body < 0.008:
                if plume.is_sweet_food:
                    taste[0] = 1.0
                    if motor.proboscis_per > 0.5:
                        self.crop_stretch = min(1.0, self.crop_stretch + 0.01 * dt_ms)
                if plume.is_bitter_shock:
                    taste[1] = 1.0
                if plume.is_yeast_protein:
                    taste[2] = 1.0

        # Small visual target bearing & salience
        small_sal = 0.0
        small_az = 0.0
        if self.small_target_pos_xyz is not None:
            tx, ty, _ = self.small_target_pos_xyz
            dx = tx - float(self.pos_xyz[0].item())
            dy = ty - float(self.pos_xyz[1].item())
            small_az = math.atan2(dy, dx) - head_yaw
            small_az = math.atan2(math.sin(small_az), math.cos(small_az))
            small_sal = 1.0 / (1.0 + 10.0 * (dx * dx + dy * dy))

        # 892 ommatidia luminance profile (peaked around sun_ego)
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
