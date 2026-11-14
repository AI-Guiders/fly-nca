"""L4 Hardware Body Contract: FlyBodyAdapter, MotorOut (805 BANC channels), and SensorIn (16,563 channels)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import torch

# 6 legs: L1 (0), R1 (1), L2 (2), R2 (3), L3 (4), R3 (5)
# 4 joints per leg: body-coxa (0), coxa-trochanter (1), femur-tibia (2), tibia-tarsus (3)
LEG_NAMES: tuple[str, ...] = ("L1", "R1", "L2", "R2", "L3", "R3")
JOINT_NAMES: tuple[str, ...] = ("body_coxa", "coxa_trochanter", "femur_tibia", "tibia_tarsus")


@dataclass
class MotorOut:
    """Typed motor output intent emitted by VNC-NCA & SEZ-NCA (805 BANC motor neurons mapped to effectors)."""

    leg_joint_drive: torch.Tensor    # [6, 4] Target joint torque/extension in [-1, +1] (24 DOF)
    tarsus_adhesion: torch.Tensor    # [6] Foot surface grip in [0, 1] (floor/wall/ceiling climbing)
    wing_drive_lr: torch.Tensor      # [2] Left/Right wing stroke amplitude in [0, 1] (64 wing MNs)
    wing_pitch_lr: torch.Tensor      # [2] Left/Right wing angle of attack offset in [-1, +1]
    wing_song_mode: torch.Tensor     # [2] Unilateral male courtship vibration (pulse/sine song, fru+)
    haltere_drive: torch.Tensor      # [2] Active haltere oscillation drive in [0, 1] (27 haltere MNs)
    neck_torque_rpy: torch.Tensor    # [3] Neck 3D gimbal torque (roll, pitch, yaw) in [-1, +1] (49 neck MNs)
    proboscis_per: float = 0.0       # Proboscis extension reflex (PER) & pharyngeal pumping (35 MNs)
    jump_ttmn_burst: float = 0.0     # Giant Fiber -> TTMn tergotrochanteral jump muscle burst in [0, 1]
    abdomen_bend_pitch_yaw: tuple[float, float] = (0.0, 0.0)  # Abdomen posture (168 MNs)

    @staticmethod
    def zeros(device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> MotorOut:
        return MotorOut(
            leg_joint_drive=torch.zeros((6, 4), device=device, dtype=dtype),
            tarsus_adhesion=torch.ones((6,), device=device, dtype=dtype),
            wing_drive_lr=torch.zeros((2,), device=device, dtype=dtype),
            wing_pitch_lr=torch.zeros((2,), device=device, dtype=dtype),
            wing_song_mode=torch.zeros((2,), device=device, dtype=dtype),
            haltere_drive=torch.ones((2,), device=device, dtype=dtype),
            neck_torque_rpy=torch.zeros((3,), device=device, dtype=dtype),
            proboscis_per=0.0,
            jump_ttmn_burst=0.0,
            abdomen_bend_pitch_yaw=(0.0, 0.0),
        )


@dataclass
class SensorIn:
    """Typed sensory periphery input from 3D body to OL / AL / AMMC / SEZ / VNC (16,563 BANC sensors)."""

    # Vision (892 hex columns x 2 eyes or canonical spherical projection)
    ommatidia_light: torch.Tensor      # [892] Hex ommatidia luminance/contrast
    sun_azimuth_rad: float             # Polarized skylight / sun direction in body frame
    looming_rate: float                # Radial expansion speed (0..1) on retina
    small_target_salience: float       # Small moving target contrast (0..1)
    small_target_azimuth_rad: float    # Bearing to small target

    # Antennae: Stereo Olfaction (50 glomeruli x L/R) & Johnston's Organ (3D wind/sound)
    odor_glomeruli_lr: torch.Tensor    # [2, 50] Left/Right antennal odor concentrations
    wind_vector_ego: torch.Tensor      # [3] 3D wind vector (vx, vy, vz) in fly head frame (JO)
    sound_vibration: float             # Acoustic particle velocity at JO (courtship song / threat)

    # Legs, Wings, Halteres & Cuticle Proprioception / Mechanoreception
    leg_joint_angles: torch.Tensor     # [6, 4] Current joint angles in [-1, +1] (chordotonal)
    leg_ground_load: torch.Tensor      # [6] Cuticular strain / weight load on each leg (campaniform)
    bristle_touch: torch.Tensor        # [6] Tactile dust/obstacle contact on legs/head (triggers grooming)
    haltere_coriolis_rpy: torch.Tensor # [3] 3D angular velocity (roll, pitch, yaw) sensed by halteres
    neck_angles_rpy: torch.Tensor      # [3] Current head-to-thorax neck angles (prosternal organ)
    head_horizon_rpy: torch.Tensor     # [3] Head tilt relative to gravity/horizon (JO gravity + ocelli)

    # Gustatory, Thermal, Tactile Context & Visceral Interoception
    taste_sweet_bitter_amino: torch.Tensor  # [3] (Gr5a sweet, Gr66a bitter, Ir76b yeast/amino)
    temperature_c: float                    # Ambient / surface temperature (C)
    surface_texture_freq: float             # Tactile grid roughness sensed via AMMC/legs (for cLTM gate)
    crop_stretch: float                     # Visceral crop fullness in [0, 1]


class FlyBodyAdapter(Protocol):
    """Client-agnostic hardware contract between Fly-NCA and any 3D physics engine."""

    def step_physics(self, motor: MotorOut, dt_ms: float = 1.0) -> SensorIn:
        ...
