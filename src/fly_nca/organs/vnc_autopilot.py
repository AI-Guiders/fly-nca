"""L3 VNC-NCA: 200-Hz Flight ESP, 3D Neck Gimbal, 6 Serial Leg Loops (LBL40/LUL130), and GF Escape."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from fly_nca.body.adapter import MotorOut, SensorIn
from fly_nca.core.cell_state import (
    CH_A,
    CH_ADAPT,
    CH_AUTO,
    CH_V,
    TAU_FAST_MS,
    CellPhysiologyParams,
    UniversalFlyCellUpdater,
    init_cell_state,
)
from fly_nca.core.ports import (
    port1_2_dendritic_integration,
    port3_axo_axonic_presynaptic_gate,
    port4_electrical_gap_junction,
)
from fly_nca.genome.banc_biophysics import (
    BANC_EMPIRICAL_BIOPHYSICS,
    BANC_GJ_FRA_LOF_FRACTION,
    BANC_GJ_WT_FRACTION,
)


@dataclass
class DescendingJoystickDN:
    """Descending neuron (DN) command bundle from Central Brain to VNC-NCA (1,316 DNs in BANC)."""

    forward_walk_dnp09: float = 0.0   # Forward walking drive
    backward_walk_mdn: float = 0.0    # Moonwalker (MDN) backward walking command
    turn_left_dna02: float = 0.0      # Left steering command (walking & flight)
    turn_right_dna02: float = 0.0     # Right steering command (walking & flight)
    flight_power_dng02: float = 0.0   # Wing power drive (flight mode)
    flight_lift_dn: float = 0.0       # Vertical pitch/lift command in 3D flight
    gaze_pitch_dn: float = 0.0        # Head-orienting gaze pitch offset
    gf_looming_spike: float = 0.0     # Giant Fiber (GF / DNp01) emergency escape trigger
    freeze_dn: float = 0.0            # Freezing command
    proboscis_fdg: float = 0.0        # Feeding command (SEZ Fdg)


# Cell index layout inside VNCAutopilotNCA (42 canonical NCA cell pools):
# 0..5:   Haltere mechanoreceptors [Roll+, Roll-, Pitch+, Pitch-, Yaw+, Yaw-]
# 6..9:   Flight steering & power MNs [Wing_L, Wing_R, Wing_Pitch_Up, Wing_Pitch_Down]
# 10..13: Neck 3D Gimbal MNs [Neck_Roll_L, Neck_Roll_R, Neck_Pitch_Up, Neck_Pitch_Down]
# 14..37: 6 Serial Leg Loops (legs 0..5 x 4 cells: [LBL40_Stance, LUL130_Swing, Coxa_Fwd, Coxa_Bwd])
# 38..41: Giant Fiber Escape & Grooming [GF_Soma, TTMn_Jump, Bristle_Sensor, Groom_Motor]
N_VNC_CELLS: int = 42


class VNCAutopilotNCA(nn.Module):
    """All 3 VNC Autopilots + GF Catapult implemented strictly as 8-channel 5-port NCA cells."""

    def __init__(
        self,
        frazzled_state: str = "wt",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.updater = UniversalFlyCellUpdater(slope=3.5)

        self.params = CellPhysiologyParams.default(N_VNC_CELLS, device=self.device, dtype=dtype)
        # Biophysical constants from BANC_EMPIRICAL_BIOPHYSICS (banc_888_metrics.feather & edgelist_split_v3):
        # Sensory halteres (0..5): tau_V = TAU_FAST_MS (1.711 ms), segregation_index = 0.010 -> tonic
        # Wing/Neck MNs (6..13): segregation_index = 0.005 -> tonic motor integration (no adaptation drop!)
        wing_bio = BANC_EMPIRICAL_BIOPHYSICS["wing_motor_neuron"]
        leg_in_bio = BANC_EMPIRICAL_BIOPHYSICS["single_leg_neuromere"]
        leg_mn_bio = BANC_EMPIRICAL_BIOPHYSICS["leg_motor_neuron"]
        self.wing_bio = wing_bio
        self.leg_in_bio = leg_in_bio
        self.leg_mn_bio = leg_mn_bio

        self.params.tau_v_ms[:6] = BANC_EMPIRICAL_BIOPHYSICS["sensory"].tau_v_ms
        self.params.tau_v_ms[6:14] = 1.0
        self.params.lambda_adapt[:14] = wing_bio.segregation_index
        self.params.tau_v_ms[38:40] = 0.8
        self.params.theta_base[:14] = 3.0 * wing_bio.port_primary_root  # 0.051 from BANC wing_motor_neuron!
        self.params.theta_base[38] = BANC_EMPIRICAL_BIOPHYSICS["lobula_columnar"].port_axon_axon  # GF threshold (0.142)
        self.params.theta_base[39] = leg_mn_bio.ach_share  # High-threshold giant jump motor neuron TTMn (0.499, requires full shak-B GJ!)

        # Leg CPG cells (14..37): pacemaker adaptation from BANC single_leg_neuromere (LBL40/LUL130 class!)
        self.params.tau_v_ms[14:38] = 2.0
        self.params.theta_base[14:38] = leg_in_bio.segregation_index
        self.params.tau_adapt_ms[14:38] = 2.0 * leg_in_bio.tau_v_ms  # 22.04 ms from BANC single_leg_neuromere l2_cable_length_um!
        self.params.beta_adapt[14:38] = leg_in_bio.inh_share          # 0.472 from BANC single_leg_neuromere!
        self.params.lambda_adapt[14:38] = leg_in_bio.ie_ratio         # 0.900 from BANC single_leg_neuromere!

        self.register_buffer("state", init_cell_state(N_VNC_CELLS, device=self.device, dtype=dtype))

        # Ablation mask [N_VNC_CELLS] (1.0 = intact, 0.0 = genetically silenced e.g. LBL40 in T3)
        self.register_buffer("ablation_mask", torch.ones(N_VNC_CELLS, device=self.device, dtype=dtype))

        # Gap junction (Port 4) wiring:
        # 1) Haltere Roll/Pitch -> Wing & Neck MNs (wing_bio.port_axon_dendrite = 0.643, BANC_GJ_WT_FRACTION = 0.0904)
        # 2) Giant Fiber (cell 38) -> TTMn Jump Motor (cell 39), regulated by Frazzled ICD (eNeuro 2025)
        fra_scale = 1.0
        if frazzled_state == "lof":
            fra_scale = (BANC_GJ_FRA_LOF_FRACTION / BANC_GJ_WT_FRACTION) * wing_bio.port_dendrite_dendrite
        elif frazzled_state == "icd_rescue":
            fra_scale = 1.0 + BANC_GJ_WT_FRACTION

        w_halt_wing = wing_bio.port_axon_dendrite
        w_halt_neck = BANC_GJ_WT_FRACTION
        w_gf_ttmn = 10.0 * BANC_GJ_WT_FRACTION * fra_scale

        self.register_buffer(
            "gap_src",
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 38], device=self.device, dtype=torch.long),
        )
        self.register_buffer(
            "gap_dst",
            torch.tensor([7, 6, 9, 8, 11, 10, 13, 12, 39], device=self.device, dtype=torch.long),
        )
        self.register_buffer(
            "gap_g",
            torch.tensor(
                [w_halt_wing, w_halt_wing, w_halt_wing, w_halt_wing, w_halt_neck, w_halt_neck, w_halt_neck, w_halt_neck, w_gf_ttmn],
                device=self.device,
                dtype=dtype,
            ),
        )

    def silence_lbl40_t3(self) -> None:
        """Silencing LBL40 in hind legs T3 (legs 4 and 5) as in Nat Commun 2020 epistasis experiment."""
        for leg_t3 in (4, 5):
            self.ablation_mask[14 + leg_t3 * 4 + 0] = 0.0

    def step_autopilot(
        self,
        sensor: SensorIn,
        dn: DescendingJoystickDN | None = None,
        dt_ms: float = TAU_FAST_MS,
    ) -> tuple[MotorOut, torch.Tensor]:
        """Run 1 NCA step (default 1 tau_fast = 1.711 ms) across all 42 VNC cell pools and emit MotorOut + Ascending AN copy."""
        dn = dn or DescendingJoystickDN()
        dev = self.device
        dt = float(dt_ms)

        exc_b = torch.zeros((N_VNC_CELLS, 2), device=dev, dtype=self.dtype)
        shunt_b = torch.zeros((N_VNC_CELLS, 2), device=dev, dtype=self.dtype)
        root_shunt = torch.zeros(N_VNC_CELLS, device=dev, dtype=self.dtype)

        # Port 3 axo-axonic presynaptic modulation on current cell outputs
        aa_exc = torch.zeros(N_VNC_CELLS, device=dev, dtype=self.dtype)
        aa_inh = torch.zeros(N_VNC_CELLS, device=dev, dtype=self.dtype)
        # DNa04 steering modulates haltere->wing presynaptic terminals (Port 3)
        aa_exc[0] = dn.turn_left_dna02
        aa_exc[1] = dn.turn_right_dna02

        presyn_out = port3_axo_axonic_presynaptic_gate(
            self.state[:, CH_A] * self.ablation_mask,
            self.state[:, CH_AUTO],
            self.params.kappa_auto,
            aa_exc=aa_exc,
            aa_inh=aa_inh,
        )

        # --- 1. Haltere Mechanoreceptor Cells (0..5) driven by 3D Coriolis angular velocity ---
        om_r, om_p, om_y = (
            float(sensor.haltere_coriolis_rpy[0].item()),
            float(sensor.haltere_coriolis_rpy[1].item()),
            float(sensor.haltere_coriolis_rpy[2].item()),
        )
        exc_b[0, 0] = max(0.0, om_r * 3.0)
        exc_b[1, 0] = max(0.0, -om_r * 3.0)
        exc_b[2, 0] = max(0.0, om_p * 3.0)
        exc_b[3, 0] = max(0.0, -om_p * 3.0)
        exc_b[4, 0] = max(0.0, om_y * 3.0)
        exc_b[5, 0] = max(0.0, -om_y * 3.0)

        # --- 2. Flight Autopilot Cells (6..9): Wing_L (6), Wing_R (7), Pitch_Up (8), Pitch_Down (9) ---
        # Synaptic port shares from BANC wing_motor_neuron: axon-dendrite = 0.643, dendrite-dendrite = 0.334
        w_ad = self.wing_bio.port_axon_dendrite
        w_dd = self.wing_bio.port_dendrite_dendrite
        base_flight = (w_ad * self.wing_bio.ach_share) * dn.flight_power_dng02
        exc_b[6, 0] = base_flight + w_dd * dn.turn_right_dna02 + w_ad * float(presyn_out[1].item())
        exc_b[7, 0] = base_flight + w_dd * dn.turn_left_dna02 + w_ad * float(presyn_out[0].item())
        shunt_b[6, 0] = 4.0 * float(presyn_out[0].item()) + (1.0 + self.wing_bio.ach_share) * dn.turn_left_dna02
        shunt_b[7, 0] = 4.0 * float(presyn_out[1].item()) + (1.0 + self.wing_bio.ach_share) * dn.turn_right_dna02

        exc_b[8, 0] = w_dd * max(0.0, dn.flight_lift_dn) + w_ad * float(presyn_out[3].item())
        exc_b[9, 0] = w_dd * max(0.0, -dn.flight_lift_dn) + w_ad * float(presyn_out[2].item())
        shunt_b[8, 0] = 4.0 * float(presyn_out[2].item())
        shunt_b[9, 0] = 4.0 * float(presyn_out[3].item())

        # --- 3. Neck 3D Gimbal Cells (10..13): Roll_L (10), Roll_R (11), Pitch_Up (12), Pitch_Down (13) ---
        # Driven by head horizon error (JO gravity / ocelli) + haltere rate, shunted by prosternal organ
        head_err_r = float(sensor.head_horizon_rpy[0].item())
        head_err_p = float(sensor.head_horizon_rpy[1].item()) - dn.gaze_pitch_dn
        neck_gain = 1.0 + BANC_EMPIRICAL_BIOPHYSICS["neck_motor_neuron"].ach_share
        exc_b[10, 0] = max(0.0, -head_err_r * neck_gain)
        exc_b[11, 0] = max(0.0, head_err_r * neck_gain)
        shunt_b[10, 0] = 4.0 * max(0.0, head_err_r)
        shunt_b[11, 0] = 4.0 * max(0.0, -head_err_r)

        exc_b[12, 0] = max(0.0, -head_err_p * neck_gain)
        exc_b[13, 0] = max(0.0, head_err_p * neck_gain)
        shunt_b[12, 0] = 4.0 * max(0.0, head_err_p)
        shunt_b[13, 0] = 4.0 * max(0.0, -head_err_p)

        # --- 4. Six Serial Leg Loops (14..37): LBL40 (stance) <-> LUL130 (swing) + Tripod Coupling ---
        walk_drive = max(dn.forward_walk_dnp09, dn.backward_walk_mdn)
        standing_posture = w_dd if walk_drive < self.leg_in_bio.segregation_index else 0.0

        # Tripod neighbors for each leg: L1(0)<->{R1(1),L2(2)}, R1(1)<->{L1(0),R2(3)}, etc.
        tripod_competitors = {
            0: (1, 2),
            1: (0, 3),
            2: (0, 3, 4),
            3: (1, 2, 5),
            4: (2, 5),
            5: (3, 4),
        }

        w_leg_exc = self.leg_in_bio.port_axon_dendrite + self.leg_in_bio.port_axon_axon  # 0.650 from BANC single_leg_neuromere!
        w_leg_load = self.leg_in_bio.port_dendrite_dendrite                               # 0.246 from BANC single_leg_neuromere!
        w_leg_shunt = self.leg_in_bio.dendritic_shunt_gain                                # 4.818 from BANC single_leg_neuromere!

        for leg in range(6):
            base = 14 + leg * 4
            c_lbl40 = base + 0
            c_lul130 = base + 1
            c_fwd = base + 2
            c_bwd = base + 3

            is_left = (leg % 2) == 0
            side_gain = (
                1.0 + self.leg_mn_bio.ach_share * dn.turn_right_dna02 - self.leg_mn_bio.ach_share * dn.turn_left_dna02
                if is_left
                else 1.0 + self.leg_mn_bio.ach_share * dn.turn_left_dna02 - self.leg_mn_bio.ach_share * dn.turn_right_dna02
            )
            leg_drive = max(0.0, walk_drive * side_gain)

            # Mutual half-center inhibition between LBL40 and LUL130 via Port 2 (GABA/GluCla shunt)
            lbl_act = float(presyn_out[c_lbl40].item())
            lul_act = float(presyn_out[c_lul130].item())
            load_val = float(sensor.leg_ground_load[leg].item())

            exc_b[c_lbl40, 0] = standing_posture + w_leg_exc * leg_drive + w_leg_load * load_val
            shunt_b[c_lbl40, 0] = w_leg_shunt * lul_act

            # Cross-leg campaniform load shunts neighbor LUL130 swing initiation -> spontaneous Tripod Gait!
            comp_swing_inh = sum(float(presyn_out[14 + nb * 4 + 1].item()) for nb in tripod_competitors[leg])
            exc_b[c_lul130, 0] = w_leg_exc * leg_drive + (w_leg_load + self.leg_in_bio.segregation_index) * float(self.state[c_lbl40, CH_ADAPT].item())
            shunt_b[c_lul130, 0] = (
                w_leg_shunt * lbl_act
                + 2.0 * w_leg_load * load_val
                + (2.0 * self.leg_in_bio.ie_ratio) * comp_swing_inh
            )

            # Forward Coxa vs Moonwalker (MDN) Backward Coxa gated by Port 2 shunt!
            exc_b[c_fwd, 0] = lbl_act * dn.forward_walk_dnp09 * side_gain
            shunt_b[c_fwd, 0] = w_leg_shunt * dn.backward_walk_mdn  # MDN shunts forward protraction!

            exc_b[c_bwd, 0] = lbl_act * dn.backward_walk_mdn * side_gain
            shunt_b[c_bwd, 0] = w_leg_shunt * dn.forward_walk_dnp09

            if dn.freeze_dn > self.leg_mn_bio.ach_share:
                root_shunt[c_lul130] = 2.0 * w_leg_shunt
                root_shunt[c_fwd] = 2.0 * w_leg_shunt
                root_shunt[c_bwd] = 2.0 * w_leg_shunt

        # --- 5. Giant Fiber Escape (38 -> 39) & Decapitated Bristle Grooming Reflex (40 -> 41) ---
        lc_bio = BANC_EMPIRICAL_BIOPHYSICS["lobula_columnar"]
        looming_thresh = lc_bio.ach_share + lc_bio.port_axon_axon  # 0.743 from BANC lobula_columnar!
        exc_b[38, 0] = dn.gf_looming_spike + (2.0 * lc_bio.ach_share if sensor.looming_rate > looming_thresh else 0.0)
        exc_b[39, 0] = lc_bio.port_axon_axon * float(presyn_out[38].item())  # Chemical component (slow); fast component is Port 4 GapJ!

        touch_sum = float(sensor.bristle_touch.sum().item())
        exc_b[40, 0] = touch_sum
        exc_b[41, 0] = self.leg_in_bio.ie_ratio * float(presyn_out[40].item())
        shunt_b[41, 0] = w_leg_shunt * walk_drive  # Locomotion suppresses grooming (subsumption!)

        # Compute integrated dendritic currents (Port 1 & 2) and electrical gap currents (Port 4)
        i_dend = port1_2_dendritic_integration(exc_b, shunt_b, self.params, root_shunt=root_shunt)
        i_gap = port4_electrical_gap_junction(self.state[:, CH_V], self.gap_src, self.gap_dst, self.gap_g)

        # Advance all 42 VNC NCA cells by dt_ms
        self.state = self.updater(self.state, self.params, i_dend, i_gap, dt_ms=dt)
        self.state[:, CH_A] = self.state[:, CH_A] * self.ablation_mask

        # Read out MotorOut from VNC motor cells
        a_out = self.state[:, CH_A]
        leg_drive_out = torch.zeros((6, 4), device=dev, dtype=self.dtype)
        adhesion_out = torch.zeros(6, device=dev, dtype=self.dtype)

        for leg in range(6):
            base = 14 + leg * 4
            stance_a = a_out[base + 0]
            swing_a = a_out[base + 1]
            fwd_a = a_out[base + 2]
            bwd_a = a_out[base + 3]

            # Joint 0 (body-coxa): +fwd_a - bwd_a
            leg_drive_out[leg, 0] = (fwd_a - bwd_a).clamp(-1.0, 1.0)
            # Joint 1 (coxa-trochanter lift): swing_a - stance_a
            leg_drive_out[leg, 1] = (swing_a - stance_a).clamp(-1.0, 1.0)
            # Joint 2 (femur-tibia flexion, LBL40): +stance_a - swing_a
            leg_drive_out[leg, 2] = (stance_a - swing_a).clamp(-1.0, 1.0)
            # Joint 3 (tibio-tarsal grooming / grip): grooming cell 41 or stance
            leg_drive_out[leg, 3] = a_out[41] if float(a_out[41].item()) > 0.2 else stance_a
            adhesion_out[leg] = (stance_a >= swing_a).to(self.dtype)

        wing_lr = torch.stack([a_out[6], a_out[7]])
        wing_pitch = torch.stack([a_out[8] - a_out[9], a_out[8] - a_out[9]])
        neck_rpy = torch.stack([a_out[10] - a_out[11], a_out[12] - a_out[13], torch.tensor(0.0, device=dev, dtype=self.dtype)])

        motor = MotorOut(
            leg_joint_drive=leg_drive_out,
            tarsus_adhesion=adhesion_out,
            wing_drive_lr=wing_lr,
            wing_pitch_lr=wing_pitch,
            wing_song_mode=torch.zeros(2, device=dev, dtype=self.dtype),
            haltere_drive=torch.ones(2, device=dev, dtype=self.dtype),
            neck_torque_rpy=neck_rpy,
            proboscis_per=float(dn.proboscis_fdg),
            jump_ttmn_burst=float(a_out[39].item()),
            abdomen_bend_pitch_yaw=(0.0, 0.0),
        )

        # Ascending Efference Copy (AN19B001 hub, 1,849 ANs in BANC):
        # Reports [actual_fwd_step, actual_bwd_step, actual_turn_rate, flight_active, groom_active]
        an_efference_copy = torch.stack([
            a_out[14 + 2:38:4].mean(),
            a_out[14 + 3:38:4].mean(),
            (a_out[6] - a_out[7]) + (a_out[14 + 2] - a_out[18 + 2]),
            0.5 * (a_out[6] + a_out[7]),
            a_out[41],
        ])
        return motor, an_efference_copy
