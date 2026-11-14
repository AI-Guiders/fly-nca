"""Full Closed-Loop Composition of NCA-Fly: Genome -> 6 Topological NCA Organs + Glia + 4D Body Arena."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from fly_nca.body import FastTensorArena4D, MotorOut, SensorIn
from fly_nca.core import GlialSubstrateNCA, HemolymphState
from fly_nca.genome.banc_biophysics import (
    BANC_EMPIRICAL_BIOPHYSICS,
    BANC_ORGAN_INPUT_MATRIX,
)
from fly_nca.genome.fly_genome import FlyGenome
from fly_nca.genome.morphogenesis import develop_fly_from_genome
from fly_nca.nervous import (
    BiologicalPort,
    ImpulseChannel,
    ImpulseOutcome,
    ImpulseRecorder,
    impulse_recorder,
)
from fly_nca.organs import (
    AntennalSEZNCA,
    CentralComplexNCA,
    ChemosensoryDigest,
    CXNavigationReadout,
    DescendingJoystickDN,
    LHSpinGateNCA,
    LHSpinOutput,
    MBStepOutput,
    MushroomBodyNCA,
    OpticLobeNCA,
    VisualProjectionDigest,
    VNCAutopilotNCA,
)


@dataclass(frozen=True, slots=True)
class FlyFrameTelemetry:
    """Complete telemetry of 1 cognitive frame (~4 ms = 4 fast 1-ms VNC autopilot ticks)."""

    pos_xyz: tuple[float, float, float]
    time_ms: float
    vis: VisualProjectionDigest
    chem: ChemosensoryDigest
    mb: MBStepOutput
    lh_spin: LHSpinOutput
    cx: CXNavigationReadout
    motor: MotorOut


class FlyNCAComposition(nn.Module):
    """Unified compositional NCA-Fly running at dual physical time quanta (1 ms VNC / 4 ms OL+CB)."""

    def __init__(
        self,
        genome: FlyGenome | None = None,
        sex: str = "F",
        frazzled_state: str = "wt",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.genome = genome or FlyGenome()
        self.developed = develop_fly_from_genome(
            self.genome,
            stage="adult",
            sex="M" if sex == "M" else "F",
            scale=0.05,
            frazzled_state=frazzled_state,
            device=self.device,
        )

        # Instantiate the 6 natural topological NCA organs + Glia + Hemolymph
        self.glia = GlialSubstrateNCA(n_compartments=15, device=self.device, dtype=dtype)
        self.hemolymph = HemolymphState()
        self.optic_lobe = OpticLobeNCA(device=self.device, dtype=dtype)
        self.antennal_sez = AntennalSEZNCA(device=self.device, dtype=dtype)
        self.lh_spin = LHSpinGateNCA(device=self.device, dtype=dtype)
        self.mushroom_body = MushroomBodyNCA(device=self.device, dtype=dtype)
        self.central_complex = CentralComplexNCA(device=self.device, dtype=dtype)
        self.vnc_autopilot = VNCAutopilotNCA(frazzled_state=frazzled_state, device=self.device, dtype=dtype)

        self.last_an_efference = torch.zeros(5, device=self.device, dtype=dtype)
        self.frame_tick_id: int = 0

    def step_cognitive_frame(
        self,
        arena: FastTensorArena4D,
        target_altitude_z: float = 0.0,
        sleep_need_override: float | None = None,
        creb_intact: bool = True,
        visual_darkness: bool = False,
        gtacr1_inhibit_fb_goal: bool = False,
        recorder: ImpulseRecorder | None = None,
    ) -> FlyFrameTelemetry:
        """Execute 1 cognitive frame (4.0 ms = 1 wingbeat) composed of 1 CB/OL update + 4 x 1.0 ms VNC ticks."""
        self.frame_tick_id += 1
        rec = recorder if recorder is not None else impulse_recorder()
        rec.begin_tick(self.frame_tick_id, time_ms=arena.time_ms)

        with rec.channel(ImpulseChannel.COGNITIVE_FRAME) as imp:
            sensor: SensorIn = arena.step_physics(MotorOut.zeros(device=self.device, dtype=self.dtype), dt_ms=0.0)
            lc_bio = BANC_EMPIRICAL_BIOPHYSICS["lobula_columnar"]
            dan_bio = BANC_EMPIRICAL_BIOPHYSICS["mushroom_body_dopaminergic_neuron"]
            looming_thresh = lc_bio.ach_share + lc_bio.port_axon_axon  # 0.743 from BANC lobula_columnar

            # 1. Autonomous Optic Lobe Radar (OL-NCA, 892 hex columns -> AOTU/BU + VPN)
            vis = self.optic_lobe.step_optic_lobe(sensor, dt_ms=4.0)
            imp.activate(
                "optic_lobe.vpn_digest",
                organ="OL",
                port=BiologicalPort.PORT1_ACH,
                inputs={
                    "looming_lc4": round(float(vis.lc4_lplc2_looming), 4),
                    "bar_salience_lc11": round(float(vis.lc11_small_target), 4),
                },
            )
            if vis.lc4_lplc2_looming > looming_thresh:
                imp.threshold(
                    "optic_lobe.lc4_to_gf_escape",
                    organ="OL",
                    port=BiologicalPort.PORT4_SHAKB_GAP,
                    reason=f"LC4 looming {vis.lc4_lplc2_looming:.2f} > {looming_thresh:.3f} -> Giant Fiber shak-B electrical spike",
                    inputs={"lc4_looming": round(float(vis.lc4_lplc2_looming), 4)},
                )

            # 2. Antennal Lobe (50 glomeruli divisive normalization) & SEZ Gustatory/Feeding NCA
            chem = self.antennal_sez.step_chemosensory(sensor, self.hemolymph)
            top_glom_val, top_glom_idx = torch.topk(chem.pn_glomeruli_50, k=3)
            imp.activate(
                "antennal_lobe.pn_glomeruli_50",
                organ="AL_SEZ",
                port=BiologicalPort.PORT1_ACH,
                inputs={
                    "sweet_us": round(float(chem.sweet_reward_us), 4),
                    "bitter_us": round(float(chem.bitter_punish_us), 4),
                    "fdg_per_cmd": round(float(chem.fdg_proboscis_cmd), 4),
                },
            )
            imp.select(
                "antennal_lobe.divisive_norm_winners",
                organ="AL_SEZ",
                port=BiologicalPort.PORT2_GLUCLA_GABA,
                selected=[f"glom_{int(idx.item())}" for idx in top_glom_idx if float(chem.pn_glomeruli_50[idx].item()) > 0.01],
                potentials=[
                    {"id": f"glom_{int(idx.item())}", "score": round(float(val.item()), 4)}
                    for val, idx in zip(top_glom_val, top_glom_idx)
                ],
            )

            # 3. Mushroom Body (Calyx APL sparse KC -> 15 compartments + DN->DAN efference copy weighted by CAM tract)
            hunger = max(0.0, 1.0 - self.hemolymph.sugar_level)
            w_al_to_mb = self.developed.tract_weight("AL", "MB") / BANC_ORGAN_INPUT_MATRIX[1][3]
            w_vnc_to_mb = self.developed.tract_weight("VNC", "MB") / BANC_ORGAN_INPUT_MATRIX[6][3]
            mb_read = self.mushroom_body.step_mb(
                pn_glomeruli_50=chem.pn_glomeruli_50 * w_al_to_mb,
                sweet_reward_us=chem.sweet_reward_us,
                bitter_shock_us=chem.bitter_punish_us,
                dn_efference_copy=self.last_an_efference * w_vnc_to_mb,
                hunger_mp1_gate=hunger,
            )
            active_kcs = int(round(mb_read.kc_active_fraction * self.mushroom_body.n_kc))
            suppressed_kcs = self.mushroom_body.n_kc - active_kcs
            imp.suppress(
                "mushroom_body.calyx_apl_gaba",
                organ="MB",
                port=BiologicalPort.PORT2_GLUCLA_GABA,
                reason=f"APL feedback inhibited {suppressed_kcs}/{self.mushroom_body.n_kc} KCs (sparsity={mb_read.kc_active_fraction:.3f})",
                rejected_above_floor=suppressed_kcs,
                inputs={"kc_sparsity": round(float(mb_read.kc_active_fraction), 4)},
            )
            pos_mask = self.mushroom_body.mbon_valence_sign > 0
            neg_mask = self.mushroom_body.mbon_valence_sign < 0
            approach_drive = float(mb_read.mbon_rates[pos_mask].mean().item())
            avoid_drive = float(mb_read.mbon_rates[neg_mask].mean().item())
            weighted_approach = avoid_drive + float(mb_read.net_learned_valence)
            imp.score(
                "mushroom_body.mbon_15_compartments",
                organ="MB",
                port=BiologicalPort.PORT3_AXO_AXONIC,
                potentials=[
                    {"id": "approach_mbon", "score": round(approach_drive, 4)},
                    {"id": "weighted_approach", "score": round(weighted_approach, 4)},
                    {"id": "avoid_mbon", "score": round(avoid_drive, 4)},
                    {"id": "net_learned_valence", "score": round(float(mb_read.net_learned_valence), 4)},
                ],
                inputs={"hunger_mp1": round(float(hunger), 4)},
            )

            # 4. Lateral Horn & SPIN Multimodal Context Gate (GluCla AND-gate + soft priors)
            if (chem.sweet_reward_us > 0.1 or chem.bitter_punish_us > 0.1) and float(chem.pn_glomeruli_50.max().item()) > 0.1:
                us_sign = 1.0 if chem.sweet_reward_us >= chem.bitter_punish_us else -1.0
                self.lh_spin.record_cltm_engram(
                    odor_glomerulus=int(top_glom_idx[0].item()),
                    sensor=sensor,
                    valence_sign=us_sign,
                )
            lh_out = self.lh_spin.step_lh_spin(
                pn_glomeruli_50=chem.pn_glomeruli_50,
                sensor=sensor,
                hemolymph=self.hemolymph,
                mbon_learned_valence=mb_read.net_learned_valence,
            )
            if lh_out.cltm_gate_open < 0.5:
                imp.suppress(
                    "lh_spin.glucla_context_shunt",
                    organ="LH_SPIN",
                    port=BiologicalPort.PORT2_GLUCLA_GABA,
                    reason=f"GluCla dendritic shunt gated cLTM (gate_open={lh_out.cltm_gate_open:.2f})",
                    inputs={"cltm_gate_open": round(float(lh_out.cltm_gate_open), 4)},
                )
            imp.select(
                "lh_spin.valence_arbiter",
                organ="LH_SPIN",
                port=BiologicalPort.PORT1_ACH,
                selected=["approach_valence" if lh_out.net_valence_to_cx_fbt >= 0.0 else "avoid_valence"],
                potentials=[
                    {"id": "innate_prior", "score": round(float(lh_out.innate_prior_valence), 4)},
                    {"id": "cltm_retrieved", "score": round(float(lh_out.cltm_retrieved_valence), 4)},
                    {"id": "net_to_cx_fbt", "score": round(float(lh_out.net_valence_to_cx_fbt), 4)},
                ],
            )

            # 5. Central Complex Navigator & dFB Sleep Switch (weighted by CAM-assembled LH_SPIN -> CX tract!)
            w_lh_to_cx = self.developed.tract_weight("LH_SPIN", "CX") / BANC_ORGAN_INPUT_MATRIX[2][4]
            sleep_need = (
                float(sleep_need_override)
                if sleep_need_override is not None
                else float(self.glia.alg_sleep_need.mean().item())
            )
            cx_out = self.central_complex.step_cx(
                sensor=sensor,
                vis=vis,
                valence_gate_signal=lh_out.net_valence_to_cx_fbt * w_lh_to_cx,
                target_altitude_z=target_altitude_z,
                current_z=float(arena.pos_xyz[2].item()),
                sleep_need_alg=sleep_need,
                visual_darkness=visual_darkness,
                gtacr1_inhibit_fb_goal=gtacr1_inhibit_fb_goal,
                dt_ms=4.0,
            )
            eb_rates = self.central_complex.state[0:16, 1]
            fb_persist = self.central_complex.state[16:32, 3]
            eb_wedge = int(torch.argmax(eb_rates).item())
            fb_wedge = int(torch.argmax(fb_persist).item())
            imp.select(
                "central_complex.ring_attractor_16",
                organ="CX",
                port=BiologicalPort.PORT1_ACH,
                selected=[f"eb_wedge_{eb_wedge}", f"fb_goal_wedge_{fb_wedge}", f"phase_{cx_out.sleep_phase}"],
                potentials=[
                    {"id": f"eb_wedge_{eb_wedge}", "score": round(float(eb_rates[eb_wedge].item()), 4)},
                    {"id": f"fb_goal_wedge_{fb_wedge}", "score": round(float(fb_persist[fb_wedge].item()), 4)},
                    {"id": "pfl3_left", "score": round(float(cx_out.dn_command.turn_left_dna02), 4)},
                    {"id": "pfl3_right", "score": round(float(cx_out.dn_command.turn_right_dna02), 4)},
                ],
            )

            # If in Active Sleep (REM-like phase of dFB), consolidate MB gamma traces into alpha/beta via CREB!
            if cx_out.sleep_phase == "ACTIVE_REM":
                self.mushroom_body.consolidate_sleep_creb(creb_intact=creb_intact, transfer_rate=dan_bio.mod_share)
                imp.threshold(
                    "central_complex.dfb_active_sleep_creb",
                    organ="CX",
                    port=BiologicalPort.PORT5_GLIA,
                    reason=f"ALG sleep_need={sleep_need:.2f} triggered dFB ACTIVE_REM -> MB CREB consolidation (creb_intact={creb_intact})",
                    inputs={"sleep_need": round(float(sleep_need), 4)},
                )

            # Combine CX steering with SEZ feeding and LC4 looming -> GF escape on the descending neck bundle
            dn_cmd = DescendingJoystickDN(
                forward_walk_dnp09=cx_out.dn_command.forward_walk_dnp09,
                backward_walk_mdn=cx_out.dn_command.backward_walk_mdn,
                turn_left_dna02=cx_out.dn_command.turn_left_dna02,
                turn_right_dna02=cx_out.dn_command.turn_right_dna02,
                flight_power_dng02=cx_out.dn_command.flight_power_dng02,
                flight_lift_dn=cx_out.dn_command.flight_lift_dn,
                gf_looming_spike=(2.0 * lc_bio.ach_share if vis.lc4_lplc2_looming > looming_thresh else 0.0),
                freeze_dn=cx_out.dn_command.freeze_dn,
                proboscis_fdg=chem.fdg_proboscis_cmd,
            )

            # 6. Run 4 fast ticks (4 x 1.0 ms = 4.0 ms) of VNC 200-Hz Autopilot & 4D Arena physics
            motor = MotorOut.zeros(device=self.device, dtype=self.dtype)
            for _ in range(4):
                motor, self.last_an_efference = self.vnc_autopilot.step_autopilot(sensor, dn=dn_cmd, dt_ms=1.0)
                sensor = arena.step_physics(motor, dt_ms=1.0)

            active_routes: list[str] = []
            route_port = BiologicalPort.PORT1_ACH
            if dn_cmd.gf_looming_spike > 0.5:
                active_routes.append("GF->TTMn_escape_jump")
                outcome = ImpulseOutcome.ESCAPE_JUMP
                route_port = BiologicalPort.PORT4_SHAKB_GAP
            elif cx_out.sleep_phase == "ACTIVE_REM":
                active_routes.append("dFB->MB_CREB_consolidation")
                outcome = ImpulseOutcome.ACTIVE_SLEEP_CREB
                route_port = BiologicalPort.PORT5_GLIA
            elif cx_out.sleep_phase == "SLOW_WAVE":
                active_routes.append("dFB->VNC_quiet_sleep_gate")
                outcome = ImpulseOutcome.QUIET_SLEEP
                route_port = BiologicalPort.PORT5_GLIA
            elif dn_cmd.proboscis_fdg > 0.5:
                active_routes.append("FDG->MN9_proboscis_extension")
                outcome = ImpulseOutcome.FEED_PER
            elif dn_cmd.backward_walk_mdn > 0.25:
                active_routes.append("MDN->LBL40_LUL130_backward_walk")
                outcome = ImpulseOutcome.BACKWARD_MDN
            elif abs(dn_cmd.turn_left_dna02 - dn_cmd.turn_right_dna02) > 0.08:
                if dn_cmd.turn_left_dna02 > dn_cmd.turn_right_dna02:
                    active_routes.append("DNa02_L->VNC_turn_left")
                    outcome = ImpulseOutcome.TURN_LEFT
                else:
                    active_routes.append("DNa02_R->VNC_turn_right")
                    outcome = ImpulseOutcome.TURN_RIGHT
            elif dn_cmd.forward_walk_dnp09 > 0.1:
                active_routes.append("DNp09->VNC_tripod_forward")
                outcome = ImpulseOutcome.FORWARD_RUN
            else:
                active_routes.append("VNC_postural_stabilize")
                outcome = ImpulseOutcome.HOVER_STABILIZE

            imp.route(
                "vnc_autopilot.descending_bundle",
                organ="VNC",
                port=route_port,
                selected=active_routes,
                potentials=[
                    {"id": "GF_shakB", "score": round(float(dn_cmd.gf_looming_spike), 4)},
                    {"id": "DNp09_fwd", "score": round(float(dn_cmd.forward_walk_dnp09), 4)},
                    {"id": "MDN_back", "score": round(float(dn_cmd.backward_walk_mdn), 4)},
                    {"id": "DNa02_steer", "score": round(float(dn_cmd.turn_left_dna02 - dn_cmd.turn_right_dna02), 4)},
                ],
            )
            imp.finish(outcome)

        return FlyFrameTelemetry(
            pos_xyz=(
                float(arena.pos_xyz[0].item()),
                float(arena.pos_xyz[1].item()),
                float(arena.pos_xyz[2].item()),
            ),
            time_ms=arena.time_ms,
            vis=vis,
            chem=chem,
            mb=mb_read,
            lh_spin=lh_out,
            cx=cx_out,
            motor=motor,
        )
