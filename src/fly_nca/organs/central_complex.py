"""L3 CX-NCA: Pure Topological Vector Calculus on 16-Wedge EB Ring, PB (±45°), 9x9 FB Fan (hDelta 180°), dFB, & LAL.

Zero Python trigonometry (`atan2`) in the control loop:
1. Coordinate rotation (`copy-and-shift` from egocentric wind to geocentric FB goal) is computed purely by
   topological wire shifts of the EPG heading bump: 0 (0°), -4 (+90°), +8 (180° hDelta jump), +4 (-90°),
   gated by the 4 WPN wind channels (Front, Back, Left, Right) via Port 1 & Port 3!
2. Steering error `sin(goal - heading)` in LAL (`16` PFL3 / CPU1 cells) is computed purely by the anatomical
   PB phase shift of ±2 wedges (±45°) between EB and FB:
       sum_m FB_goal[m] * EB[m - 2]  -  sum_m FB_goal[m] * EB[m + 2]  ~  sqrt(2) * sin(theta_goal - theta_head)!
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal
import torch
import torch.nn as nn

from fly_nca.body.adapter import SensorIn
from fly_nca.core.cell_state import (
    CH_A,
    CH_PERSIST,
    CH_V,
    CellPhysiologyParams,
    UniversalFlyCellUpdater,
    init_cell_state,
)
from fly_nca.core.ports import port1_2_dendritic_integration, port3_axo_axonic_presynaptic_gate
from fly_nca.genome.banc_biophysics import BANC_EMPIRICAL_BIOPHYSICS
from fly_nca.organs.optic_lobe import VisualProjectionDigest
from fly_nca.organs.vnc_autopilot import DescendingJoystickDN

# Cell layout inside CentralComplexNCA (66 8-channel 5-port NCA cells):
# 0..15:  EB EPG compass ring cells (16 wedges)
# 16..31: FB hDeltaK21 persistent goal cells (16 columns, layer 6)
# 32..47: LAL PFL3_Left steering cells (16 columns, receive FB[m] and EB[m+2])
# 48..63: LAL PFL3_Right steering cells (16 columns, receive FB[m] and EB[m-2])
# 64..65: dFB (R23E10) sleep switch cell (64) & ER5 slow-wave sleep tracker (65)
N_CX_CELLS: int = 66


@dataclass(frozen=True, slots=True)
class CXNavigationReadout:
    """Diagnostic readout of Central Complex NCA state."""

    heading_angle_rad: float                         # Decoded EPG compass bump angle (for telemetry only)
    goal_angle_rad: float                            # Decoded FB goal bump angle (for telemetry only)
    goal_bump_amplitude: float                       # Peak persistent current p_c^{goal} in FB
    nav_mode: Literal["BASELINE", "GOAL", "SEARCH"]  # 3-state machine emerging from FB bump amplitude
    is_sleeping_dfb: bool                            # dFB (R23E10) sleep cell state
    sleep_phase: Literal["WAKE", "SLOW_WAVE", "ACTIVE_REM"]
    dn_command: DescendingJoystickDN                 # 20 AN/DN steering & locomotion output


class CentralComplexNCA(nn.Module):
    """Pure Topological Central Complex NCA (66 cell pools running UniversalFlyCellUpdater)."""

    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.updater = UniversalFlyCellUpdater(slope=3.0)

        self.params = CellPhysiologyParams.default(N_CX_CELLS, device=self.device, dtype=dtype)
        cx_in_bio = BANC_EMPIRICAL_BIOPHYSICS["central_complex_intrinsic_neuron"]
        cx_out_bio = BANC_EMPIRICAL_BIOPHYSICS["central_complex_output_neuron"]

        # EB EPG ring cells (0..15): tau_V from BANC central_complex_intrinsic_neuron (3.10 ms)
        self.params.tau_v_ms[:16] = cx_in_bio.tau_v_ms
        self.params.theta_base[:16] = 0.05
        self.params.lambda_adapt[:16] = 0.0

        # FB hDeltaK21 goal cells (16..31): 5-10s (7500 ms) persistent working-memory current
        self.params.tau_v_ms[16:32] = cx_in_bio.tau_v_ms * 2.0
        self.params.theta_base[16:32] = 0.08
        self.params.tau_persist_ms[16:32] = 7500.0
        self.params.gamma_persist[16:32] = 0.55
        self.params.lambda_adapt[16:32] = 0.0

        # LAL PFL3 Left (32..47) & Right (48..63) output cells: tau_V from BANC central_complex_output_neuron (4.14 ms!)
        # Multiplicative coincidence (mu_conj = 0.85) between FB goal branch (b=0) and shifted EB heading branch (b=1)!
        self.params.tau_v_ms[32:64] = cx_out_bio.tau_v_ms
        self.params.theta_base[32:64] = 0.02
        self.params.mu_conj[32:64] = 0.85
        self.params.lambda_adapt[32:64] = 0.0

        self.register_buffer("state", init_cell_state(N_CX_CELLS, device=self.device, dtype=dtype))
        self.register_buffer(
            "wedge_angles",
            torch.linspace(-math.pi, math.pi, 17, device=self.device, dtype=dtype)[:-1],
        )
        diff = self.wedge_angles.unsqueeze(0) - self.wedge_angles.unsqueeze(1)
        self.register_buffer("cos_kernel", torch.cos(diff))
        self.sleep_timer_ms: float = 0.0

    def seed_heading(self, angle_rad: float) -> None:
        """Initialize EPG bump in EB at angle_rad."""
        profile = torch.relu(torch.cos(self.wedge_angles - float(angle_rad)))
        self.state[:16, CH_V] = profile * 0.6
        self.state[:16, CH_A] = profile

    def decode_circular_bump(self, weights_16: torch.Tensor) -> tuple[float, float]:
        """Telemetry decoder: circular mean angle (-pi..pi) and peak amplitude from 16 wedges."""
        x = (weights_16 * torch.cos(self.wedge_angles)).sum()
        y = (weights_16 * torch.sin(self.wedge_angles)).sum()
        amp = float(weights_16.max().item())
        if float(x.abs() + y.abs()) < 1e-6:
            return 0.0, 0.0
        return float(torch.atan2(y, x).item()), amp

    def hdelta_180_invert(self, fb_vec_16: torch.Tensor) -> torch.Tensor:
        """Topological hDelta half-width jump (+8 wedges = 180 deg vector inversion in FB)."""
        return torch.roll(fb_vec_16, shifts=8, dims=0)

    def copy_and_shift_wind_to_geocentric_nca(
        self,
        eb_act_16: torch.Tensor,
        wind_ego_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """Pure Topological Copy-and-Shift (PFN 4x90° basis + hDelta 180° jump, ZERO atan2!).

        Decomposes egocentric wind vector (wx, wy) into 4 orthogonal WPN receptor channels:
            w_front = relu(wx), w_back = relu(-wx), w_left = relu(wy), w_right = relu(-wy)
        and gates 4 topologically shifted copies of the EB compass bump via Port 3 (a-a presynaptic gain):
            - shift  0 (  0°): Front PFN
            - shift +8 (180°): Back hDelta jump
            - shift +4 (+90°): Left PFN
            - shift -4 (-90°): Right PFN
        By trigonometric identity:
            wx * cos(phi - theta_head) + wy * sin(phi - theta_head) == ||w|| * cos(phi - (theta_head + theta_wind_ego))!
        """
        wx = float(wind_ego_xyz[0].item())
        wy = float(wind_ego_xyz[1].item())
        mag = max(1e-6, (wx * wx + wy * wy) ** 0.5)
        ux, uy = wx / mag, wy / mag

        # 4 topological wire shifts on the 16-column ring/fan (0, +8 hDelta, +4 PFN_L, -4 PFN_R)
        pfn_front = eb_act_16
        pfn_back = self.hdelta_180_invert(eb_act_16)
        pfn_left = torch.roll(eb_act_16, shifts=4, dims=0)
        pfn_right = torch.roll(eb_act_16, shifts=-4, dims=0)

        geo_bump = (
            max(0.0, ux) * pfn_front
            + max(0.0, -ux) * pfn_back
            + max(0.0, uy) * pfn_left
            + max(0.0, -uy) * pfn_right
        )
        peak = float(geo_bump.max().item())
        if peak > 1e-6:
            geo_bump = geo_bump / peak
        return geo_bump

    def step_cx(
        self,
        sensor: SensorIn,
        vis: VisualProjectionDigest,
        valence_gate_signal: float = 0.0,
        target_altitude_z: float = 0.0,
        current_z: float = 0.0,
        sleep_need_alg: float = 0.0,
        ppl1_wake_da: float = 0.2,
        visual_darkness: bool = False,
        gtacr1_inhibit_fb_goal: bool = False,
        dt_ms: float = 4.0,
    ) -> CXNavigationReadout:
        """Advance all 66 CX NCA cells using purely local/topological 5-port updates (no atan2 steering!)."""
        dev = self.device
        dt = float(dt_ms)

        exc_b = torch.zeros((N_CX_CELLS, 2), device=dev, dtype=self.dtype)
        shunt_b = torch.zeros((N_CX_CELLS, 2), device=dev, dtype=self.dtype)
        i_reset = torch.zeros(N_CX_CELLS, device=dev, dtype=self.dtype)

        # --- 1. EB Ring Attractor (cells 0..15): EPG + PEN (±1 wedge angular velocity shift) + Delta7 cosine ---
        eb_act = self.state[:16, CH_A]
        omega_yaw = float(sensor.haltere_coriolis_rpy[2].item())
        shift_step = omega_yaw * (dt * 1e-3) / (2.0 * math.pi / 16.0)
        pen_shift = (
            max(0.0, shift_step) * torch.roll(eb_act, shifts=1, dims=0)
            + max(0.0, -shift_step) * torch.roll(eb_act, shifts=-1, dims=0)
        )
        rec_cos = torch.mv(self.cos_kernel, eb_act) / max(1.0, float(eb_act.sum().item()))
        vis_tether = torch.zeros_like(eb_act) if visual_darkness else (0.35 * vis.bu_compass_ring_16)

        exc_b[:16, 0] = (0.50 * eb_act + 1.2 * pen_shift + 0.65 * torch.relu(rec_cos) + vis_tether).clamp(min=0.0)
        shunt_b[:16, 0] = 0.85 * eb_act.mean() + 2.0 * torch.relu(-rec_cos)

        # --- 2. Pure Topological Copy-and-Shift & FB Goal Cells (cells 16..31) ---
        upwind_geo_profile = self.copy_and_shift_wind_to_geocentric_nca(eb_act, sensor.wind_vector_ego)
        pos_evidence = max(0.0, float(valence_gate_signal))
        fb_persist = self.state[16:32, CH_PERSIST]

        if gtacr1_inhibit_fb_goal:
            shunt_b[16:32, 0] = 20.0
            i_reset[16:32] = 1.0
        else:
            exc_b[16:32, 0] = pos_evidence * upwind_geo_profile * 1.2 + 0.55 * fb_persist
            shunt_b[16:32, 0] = 0.25 * fb_persist.sum()

        # Strong aversive signal triggers topological hDelta 180° jump (+8 columns) -> opposite goal!
        if valence_gate_signal < -0.3:
            avoid_profile = self.hdelta_180_invert(upwind_geo_profile)
            exc_b[16:32, 0] = abs(valence_gate_signal) * avoid_profile * 1.4

        # --- 3. LAL Steering Cells: PFL3_Left (32..47) and PFL3_Right (48..63) ---
        # Each PFL3 cell receives Branch 0 = FB goal column m, and Branch 1 = EB heading shifted by ±2 wedges (±45°)!
        fb_out = (self.state[16:32, CH_PERSIST] + 0.3 * self.state[16:32, CH_A]).clamp(0.0, 1.0)
        if pos_evidence > 0.2 and float(fb_out.max().item()) < 0.2:
            fb_out = upwind_geo_profile

        eb_shift_left = torch.roll(eb_act, shifts=2, dims=0)   # -45° anatomical shift in PB for left turn
        eb_shift_right = torch.roll(eb_act, shifts=-2, dims=0) # +45° anatomical shift in PB for right turn

        exc_b[32:48, 0] = fb_out
        exc_b[32:48, 1] = eb_shift_left
        exc_b[48:64, 0] = fb_out
        exc_b[48:64, 1] = eb_shift_right

        # --- 4. dFB (R23E10, cell 64) & ER5 (cell 65) Homeostatic Sleep Switch ---
        exc_b[64, 0] = 1.5 * sleep_need_alg
        shunt_b[64, 0] = 3.0 * ppl1_wake_da
        exc_b[65, 0] = 1.2 * sleep_need_alg

        # Run Universal 5-Port NCA Cell Update across all 66 CX cells!
        i_dend = port1_2_dendritic_integration(exc_b, shunt_b, self.params)
        i_gap = torch.zeros(N_CX_CELLS, device=dev, dtype=self.dtype)
        self.state = self.updater(self.state, self.params, i_dend, i_gap, i_persist_reset=i_reset, dt_ms=dt)

        # Normalize EB ring attractor via ring-neuron global pool
        new_eb = self.state[:16, CH_A]
        if float(new_eb.max().item()) > 1e-4:
            self.state[:16, CH_A] = new_eb / float(new_eb.max().item())

        if gtacr1_inhibit_fb_goal:
            self.state[16:32, CH_A] = 0.0
            self.state[16:32, CH_PERSIST] = 0.0
            goal_rad, goal_amp = 0.0, 0.0
        else:
            fb_goal_vec = self.state[16:32, CH_PERSIST]
            goal_rad, goal_amp = self.decode_circular_bump(fb_goal_vec + 0.2 * self.state[16:32, CH_A])

        heading_rad, _ = self.decode_circular_bump(self.state[:16, CH_A])

        if goal_amp > 0.35 or pos_evidence > 0.25:
            nav_mode: Literal["BASELINE", "GOAL", "SEARCH"] = "GOAL"
        elif goal_amp > 0.08:
            nav_mode = "SEARCH"
        else:
            nav_mode = "BASELINE"

        is_sleeping = float(self.state[64, CH_A].item()) > 0.45
        if is_sleeping:
            self.sleep_timer_ms += dt
            sleep_phase: Literal["WAKE", "SLOW_WAVE", "ACTIVE_REM"] = (
                "ACTIVE_REM" if (int(self.sleep_timer_ms // 40) % 4 == 3) else "SLOW_WAVE"
            )
            dn_cmd = DescendingJoystickDN(freeze_dn=1.0)
        else:
            self.sleep_timer_ms = 0.0
            sleep_phase = "WAKE"

            # Pure NCA readout from PFL3_Left (32..47) vs PFL3_Right (48..63) cells (ZERO atan2!):
            pfl3_left_sum = float(self.state[32:48, CH_A].sum().item())
            pfl3_right_sum = float(self.state[48:64, CH_A].sum().item())
            steer_diff = pfl3_left_sum - pfl3_right_sum

            turn_l = max(0.0, steer_diff * 0.45)
            turn_r = max(0.0, -steer_diff * 0.45)

            dz = float(target_altitude_z - current_z)
            in_flight = target_altitude_z > 0.005 or current_z > 0.005
            fwd_drive = 0.80 if nav_mode == "GOAL" else (0.35 if nav_mode == "SEARCH" else 0.0)

            dn_cmd = DescendingJoystickDN(
                forward_walk_dnp09=(0.0 if in_flight else fwd_drive),
                turn_left_dna02=min(1.0, turn_l),
                turn_right_dna02=min(1.0, turn_r),
                flight_power_dng02=(0.95 if in_flight else 0.0),
                flight_lift_dn=max(-1.0, min(1.0, dz * 12.0)) if in_flight else 0.0,
            )

        return CXNavigationReadout(
            heading_angle_rad=heading_rad,
            goal_angle_rad=goal_rad,
            goal_bump_amplitude=goal_amp,
            nav_mode=nav_mode,
            is_sleeping_dfb=is_sleeping,
            sleep_phase=sleep_phase,
            dn_command=dn_cmd,
        )
