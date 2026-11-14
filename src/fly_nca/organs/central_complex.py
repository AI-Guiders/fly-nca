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
    TAU_FRAME_MS,
    CellPhysiologyParams,
    UniversalFlyCellUpdater,
    init_cell_state,
)
from fly_nca.core.ports import port1_2_dendritic_integration, port3_axo_axonic_presynaptic_gate
from fly_nca.genome.banc_biophysics import (
    BANC_EMPIRICAL_BIOPHYSICS,
    TAU_FB_GOAL_PERSIST_MS,
)
from fly_nca.organs.optic_lobe import VisualProjectionDigest
from fly_nca.organs.vnc_autopilot import DescendingJoystickDN

# Cell layout inside CentralComplexNCA (71 8-channel 5-port NCA cells):
# 0..15:  EB EPG compass ring cells (16 wedges)
# 16..31: FB hDeltaK21 persistent goal cells (16 columns, layer 6)
# 32..47: LAL PFL3_Left steering cells (16 columns, receive FB[m] and EB[m+2])
# 48..63: LAL PFL3_Right steering cells (16 columns, receive FB[m] and EB[m-2])
# 64..65: dFB (R23E10) sleep switch cell (64) & ER5 slow-wave sleep tracker (65)
# 66..70: LAL/Neck Descending Neuron NCA cells (66=DNp09 fwd, 67=MDN back, 68=DNa02_L, 69=DNa02_R, 70=DNg02 flight)
CELL_DNP09_FWD: int = 66
CELL_MDN_BACK: int = 67
CELL_DNA02_L: int = 68
CELL_DNA02_R: int = 69
CELL_DNG02_FLIGHT: int = 70
N_CX_CELLS: int = 71


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
    """Pure Topological Central Complex NCA (71 cell pools running UniversalFlyCellUpdater)."""

    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.updater = UniversalFlyCellUpdater(slope=3.0)

        self.params = CellPhysiologyParams.default(N_CX_CELLS, device=self.device, dtype=dtype)
        cx_input_bio = BANC_EMPIRICAL_BIOPHYSICS["central_complex_input_neuron"]
        cx_in_bio = BANC_EMPIRICAL_BIOPHYSICS["central_complex_intrinsic_neuron"]
        cx_out_bio = BANC_EMPIRICAL_BIOPHYSICS["central_complex_output_neuron"]
        dn_bio = BANC_EMPIRICAL_BIOPHYSICS["descending_neuron"]
        camp_bio = BANC_EMPIRICAL_BIOPHYSICS["campaniform_sensillum_neuron"]
        self.cx_input_bio = cx_input_bio
        self.cx_in_bio = cx_in_bio
        self.cx_out_bio = cx_out_bio
        self.dn_bio = dn_bio
        self.camp_bio = camp_bio

        # EB EPG ring cells (0..15): tau_V from BANC central_complex_intrinsic_neuron (3.10 ms)
        self.params.tau_v_ms[:16] = cx_in_bio.tau_v_ms
        self.params.theta_base[:16] = cx_in_bio.median_syn_norm * 10.0
        self.params.lambda_adapt[:16] = 0.0

        # FB hDeltaK21 goal cells (16..31): 5-10s (TAU_FB_GOAL_PERSIST_MS = 7500 ms) persistent working-memory current
        self.params.tau_v_ms[16:32] = cx_in_bio.tau_v_ms * 2.0
        self.params.theta_base[16:32] = cx_in_bio.median_syn_norm * 16.0
        self.params.tau_persist_ms[16:32] = TAU_FB_GOAL_PERSIST_MS
        self.params.gamma_persist[16:32] = cx_in_bio.ach_share
        self.params.lambda_adapt[16:32] = 0.0

        # LAL PFL3 Left (32..47) & Right (48..63) output cells: tau_V from BANC central_complex_output_neuron (4.14 ms!)
        # Multiplicative coincidence (mu_conj = ad + aa + dd = 0.820) between FB goal and shifted EB heading!
        self.params.tau_v_ms[32:64] = cx_out_bio.tau_v_ms
        self.params.theta_base[32:64] = cx_out_bio.mod_share * 0.5
        self.params.mu_conj[32:64] = (
            cx_out_bio.port_axon_dendrite + cx_out_bio.port_axon_axon + cx_out_bio.port_dendrite_dendrite
        )
        self.params.lambda_adapt[32:64] = 0.0

        # LAL/Neck Descending Neuron NCA cells (66..70): grounded in BANC descending_neuron & CX output biophysics
        self.params.tau_v_ms[66:71] = cx_out_bio.tau_v_ms
        self.params.theta_base[66:71] = dn_bio.ach_share * 0.25
        self.params.alpha_shunt[66:71] = dn_bio.dendritic_shunt_gain
        self.params.lambda_adapt[66:71] = 0.0

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
        dt_ms: float = TAU_FRAME_MS,
    ) -> CXNavigationReadout:
        """Advance all 71 CX NCA cells by 1 cognitive wingbeat frame (TAU_FRAME_MS = 4.038 ms)."""
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
        # Ring-neuron visual tether weighted by BANC central_complex_input_neuron axon-dendrite share (0.352)
        vis_tether = (
            torch.zeros_like(eb_act)
            if visual_darkness
            else (self.cx_input_bio.port_axon_dendrite * vis.bu_compass_ring_16)
        )

        exc_b[:16, 0] = (
            self.cx_in_bio.ach_share * eb_act
            + self.cx_in_bio.ie_ratio * pen_shift
            + (self.cx_in_bio.ach_share + self.cx_in_bio.port_dendrite_dendrite) * torch.relu(rec_cos)
            + vis_tether
        ).clamp(min=0.0)
        shunt_b[:16, 0] = (
            (self.cx_in_bio.inh_share + self.cx_in_bio.port_axon_dendrite) * eb_act.mean()
            + (self.cx_in_bio.ie_ratio + self.cx_in_bio.inh_share) * torch.relu(-rec_cos)
        )

        # --- 2. Pure Topological Copy-and-Shift & FB Goal Cells (cells 16..31) ---
        upwind_geo_profile = self.copy_and_shift_wind_to_geocentric_nca(eb_act, sensor.wind_vector_ego)
        pos_evidence = max(0.0, float(valence_gate_signal))
        neg_evidence = max(0.0, -float(valence_gate_signal))
        fb_persist = self.state[16:32, CH_PERSIST]

        if gtacr1_inhibit_fb_goal:
            shunt_b[16:32, 0] = self.cx_in_bio.root_shunt_gain * 10.0
            i_reset[16:32] = 1.0
        else:
            exc_b[16:32, 0] = (
                pos_evidence * upwind_geo_profile * self.cx_in_bio.ie_ratio
                + (self.cx_in_bio.ach_share + self.cx_in_bio.port_axon_axon) * fb_persist
            )
            shunt_b[16:32, 0] = self.cx_in_bio.port_dendrite_dendrite * fb_persist.sum()

        # Strong aversive signal (> cx_input_bio.ach_share = 0.328) triggers topological hDelta 180° jump (+8 columns)!
        if neg_evidence > self.cx_input_bio.ach_share:
            avoid_profile = self.hdelta_180_invert(upwind_geo_profile)
            exc_b[16:32, 0] = neg_evidence * avoid_profile * self.cx_out_bio.ie_ratio

        # --- 3. LAL Steering Cells: PFL3_Left (32..47) and PFL3_Right (48..63) ---
        # Each PFL3 cell receives Branch 0 = FB goal column m, and Branch 1 = EB heading shifted by ±2 wedges (±45°)!
        fb_out = (
            self.state[16:32, CH_PERSIST] + self.cx_out_bio.ach_share * self.state[16:32, CH_A]
        ).clamp(0.0, 1.0)
        if pos_evidence > self.cx_out_bio.port_dendrite_dendrite and float(fb_out.max().item()) < self.cx_out_bio.port_dendrite_dendrite:
            fb_out = upwind_geo_profile

        eb_shift_left = torch.roll(eb_act, shifts=2, dims=0)   # -45° anatomical shift in PB for left turn
        eb_shift_right = torch.roll(eb_act, shifts=-2, dims=0) # +45° anatomical shift in PB for right turn

        exc_b[32:48, 0] = fb_out
        exc_b[32:48, 1] = eb_shift_left
        exc_b[48:64, 0] = fb_out
        exc_b[48:64, 1] = eb_shift_right

        # --- 4. dFB (R23E10, cell 64) & ER5 (cell 65) Homeostatic Sleep Switch ---
        exc_b[64, 0] = self.cx_input_bio.ie_ratio * sleep_need_alg
        shunt_b[64, 0] = (self.cx_input_bio.ie_ratio * 2.0) * ppl1_wake_da
        exc_b[65, 0] = self.cx_in_bio.ie_ratio * sleep_need_alg

        # --- 5. LAL/Neck Descending Neuron NCA Cells (66..70: DNp09, MDN, DNa02_L, DNa02_R, DNg02) ---
        # Flight vs walking is gated biologically by tarsal campaniform mechanosensory load (Fraenkel 1932 tarsal reflex!)
        tarsal_load = float(sensor.leg_ground_load.mean().item())
        dz = float(target_altitude_z - current_z)
        climb_demand = max(0.0, dz * self.camp_bio.ie_ratio)
        airborne_unloading = max(0.0, 1.0 - tarsal_load / max(self.camp_bio.ach_share, 1e-4))
        flight_gate = min(1.0, max(airborne_unloading, 1.0 if dz > 0.0 else 0.0))
        sleep_shunt = self.dn_bio.root_shunt_gain * float(self.state[64, CH_A].item())

        fb_goal_peak = float(fb_out.max().item())
        # Cell 66 (DNp09 forward walk): excited by FB goal + positive valence on ground; shunted by flight, aversion & sleep
        exc_b[CELL_DNP09_FWD, 0] = (1.0 - flight_gate) * (
            self.dn_bio.ie_ratio * max(fb_goal_peak, pos_evidence)
            + self.dn_bio.ach_share * fb_goal_peak
        )
        shunt_b[CELL_DNP09_FWD, 0] = self.dn_bio.root_shunt_gain * neg_evidence + sleep_shunt

        # Cell 67 (MDN Moonwalker backward walk): excited by aversive valence on ground; shunted by positive valence & sleep
        exc_b[CELL_MDN_BACK, 0] = (1.0 - flight_gate) * (self.cx_out_bio.ie_ratio * neg_evidence)
        shunt_b[CELL_MDN_BACK, 0] = self.dn_bio.ie_ratio * pos_evidence + sleep_shunt

        # Cells 68..69 (DNa02_L & DNa02_R steering): driven by PFL3_Left vs PFL3_Right with contralateral inhibition
        pfl3_l = float(self.state[32:48, CH_A].mean().item()) * self.dn_bio.ie_ratio
        pfl3_r = float(self.state[48:64, CH_A].mean().item()) * self.dn_bio.ie_ratio
        exc_b[CELL_DNA02_L, 0] = max(0.0, pfl3_l - pfl3_r)
        shunt_b[CELL_DNA02_L, 0] = self.dn_bio.ie_ratio * max(0.0, pfl3_r - pfl3_l) + sleep_shunt
        exc_b[CELL_DNA02_R, 0] = max(0.0, pfl3_r - pfl3_l)
        shunt_b[CELL_DNA02_R, 0] = self.dn_bio.ie_ratio * max(0.0, pfl3_l - pfl3_r) + sleep_shunt

        # Cell 70 (DNg02 flight power): excited by tarsal unloading (tarsal reflex!) + climb demand; shunted by ground load
        exc_b[CELL_DNG02_FLIGHT, 0] = flight_gate * (self.dn_bio.port_axon_dendrite + self.dn_bio.ach_share + climb_demand)
        shunt_b[CELL_DNG02_FLIGHT, 0] = (1.0 - flight_gate) * self.camp_bio.ie_ratio * tarsal_load + sleep_shunt

        # Run Universal 5-Port NCA Cell Update across all 71 CX + DN cells!
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
            goal_rad, goal_amp = self.decode_circular_bump(
                fb_goal_vec + self.cx_out_bio.port_dendrite_dendrite * self.state[16:32, CH_A]
            )

        heading_rad, _ = self.decode_circular_bump(self.state[:16, CH_A])

        if goal_amp > self.cx_input_bio.port_axon_dendrite or pos_evidence > self.cx_in_bio.port_dendrite_dendrite:
            nav_mode: Literal["BASELINE", "GOAL", "SEARCH"] = "GOAL"
        elif goal_amp > self.cx_in_bio.segregation_index:
            nav_mode = "SEARCH"
        else:
            nav_mode = "BASELINE"

        is_sleeping = float(self.state[64, CH_A].item()) > self.cx_out_bio.port_axon_dendrite
        if is_sleeping:
            self.sleep_timer_ms += dt
            sleep_phase: Literal["WAKE", "SLOW_WAVE", "ACTIVE_REM"] = (
                "ACTIVE_REM" if (int(self.sleep_timer_ms // 40) % 4 == 3) else "SLOW_WAVE"
            )
            dn_cmd = DescendingJoystickDN(freeze_dn=1.0)
        else:
            self.sleep_timer_ms = 0.0
            sleep_phase = "WAKE"

            # Pure NCA readout from the 5 Descending Neuron cells (66..70) + immediate PFL3 steering!
            pfl3_steer_diff = float(self.state[32:48, CH_A].sum().item() - self.state[48:64, CH_A].sum().item())
            turn_l = max(float(self.state[CELL_DNA02_L, CH_A].item()), max(0.0, pfl3_steer_diff * self.cx_out_bio. port_axon_dendrite))
            turn_r = max(float(self.state[CELL_DNA02_R, CH_A].item()), max(0.0, -pfl3_steer_diff * self.cx_out_bio.port_axon_dendrite))
            dng02_rate = float(self.state[CELL_DNG02_FLIGHT, CH_A].item())
            dnp09_rate = float(self.state[CELL_DNP09_FWD, CH_A].item())
            mdn_rate = float(self.state[CELL_MDN_BACK, CH_A].item())

            dn_cmd = DescendingJoystickDN(
                forward_walk_dnp09=(0.0 if flight_gate > 0.5 else dnp09_rate),
                backward_walk_mdn=(0.0 if flight_gate > 0.5 else mdn_rate),
                turn_left_dna02=min(1.0, turn_l),
                turn_right_dna02=min(1.0, turn_r),
                flight_power_dng02=(max(dng02_rate, self.dn_bio.port_axon_dendrite + self.dn_bio.ach_share - self.dn_bio.mod_share) if flight_gate > 0.5 else 0.0),
                flight_lift_dn=(max(-1.0, min(1.0, dz * (self.camp_bio.ie_ratio * 3.0))) if flight_gate > 0.5 else 0.0),
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
