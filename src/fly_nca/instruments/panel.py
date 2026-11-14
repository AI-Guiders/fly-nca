"""Laboratory Instrument Panel — samples all Cellular, Glial, Nervous-Impulse, and Body instruments at once."""

from __future__ import annotations

from typing import Any
import torch

from fly_nca.body.arena_3d import FastTensorArena3D
from fly_nca.composition import FlyFrameTelemetry, FlyNCAComposition
from fly_nca.core.cell_state import CH_A, CH_PERSIST, CH_V
from fly_nca.instruments.catalog import InstrumentId, REGISTRY
from fly_nca.nervous.channels import BiologicalPort
from fly_nca.nervous.impulse import ImpulseRecorder
from fly_nca.organs.mushroom_body import COMP_GAMMA1, COMP_GAMMA5

PANEL_SCHEMA: int = 1


def sample_panel(
    fly: FlyNCAComposition,
    arena: FastTensorArena3D,
    *,
    last_telem: FlyFrameTelemetry | None = None,
    recorder: ImpulseRecorder | None = None,
    pd2a1_conc: float = 0.02,
    pd2a1_time_s: float = 80.0,
) -> dict[str, Any]:
    """Sample all registered instruments across C_cell, G_glia, N_impulse, and B_body layers."""
    if last_telem is None:
        last_telem = fly.step_cognitive_frame(arena, recorder=recorder)

    # Tripod A (L1=0, R2=3, L3=4) vs Tripod B (R1=1, L2=2, R3=5) stance LBL40 cells in VNC (14 + leg*4)
    vnc_a = fly.vnc_autopilot.state[:, CH_A]
    tripod_a = float((vnc_a[14 + 0 * 4] + vnc_a[14 + 3 * 4] + vnc_a[14 + 4 * 4]).item() / 3.0)
    tripod_b = float((vnc_a[14 + 1 * 4] + vnc_a[14 + 2 * 4] + vnc_a[14 + 5 * 4]).item() / 3.0)
    tripod_coh = abs(tripod_a - tripod_b) / max(0.05, tripod_a + tripod_b)

    # Consolidated alpha/beta LTM mean weight across compartments 9..14
    alpha_ltm_w = float(fly.mushroom_body.w_mb[:, 9:15].mean().item())

    # PD2a1/b1 soft-prior probe under current hemolymph state
    pd2_val = fly.lh_spin.evaluate_pd2a1_soft_prior(
        aversive_odor_conc=pd2a1_conc,
        exposure_time_s=pd2a1_time_s,
        hemolymph=fly.hemolymph,
    )

    # Nervous impulse layer summary
    imp_outcome = "NONE"
    imp_count = 0
    port4_active = False
    if recorder is not None:
        impulses = recorder.last_impulses(limit=1)
        if impulses:
            last_imp = impulses[-1]
            imp_outcome = str(last_imp.summary.get("outcome", "NONE"))
            imp_count = len(last_imp.events)
            port4_active = any(ev.port == BiologicalPort.PORT4_SHAKB_GAP.value for ev in last_imp.events)

    dn = last_telem.cx.dn_command
    row: dict[str, Any] = {
        "schema": PANEL_SCHEMA,
        "frame_tick_id": fly.frame_tick_id,
        "time_ms": round(float(last_telem.time_ms), 3),
        # C_cell
        InstrumentId.KC_SPARSITY.value: round(float(last_telem.mb.kc_active_fraction), 4),
        InstrumentId.MBON_G1_RATE.value: round(float(last_telem.mb.mbon_rates[COMP_GAMMA1].item()), 4),
        InstrumentId.MBON_G5_RATE.value: round(float(last_telem.mb.mbon_rates[COMP_GAMMA5].item()), 4),
        InstrumentId.MB_NET_VALENCE.value: round(float(last_telem.mb.net_learned_valence), 4),
        InstrumentId.MB_ALPHA_LTM_W.value: round(alpha_ltm_w, 4),
        InstrumentId.VPM4_BOXING_DRIVE.value: round(float(last_telem.mb.aggression_boxing_drive), 4),
        InstrumentId.CLTM_GATE_OPEN.value: round(float(last_telem.lh_spin.cltm_gate_open), 4),
        InstrumentId.CLTM_VALENCE.value: round(float(last_telem.lh_spin.cltm_retrieved_valence), 4),
        InstrumentId.PD2A1_PRIOR.value: round(float(pd2_val), 4),
        InstrumentId.EB_BUMP_PEAK.value: round(float(fly.central_complex.state[:16, CH_A].max().item()), 4),
        InstrumentId.EB_HEADING_RAD.value: round(float(last_telem.cx.heading_angle_rad), 4),
        InstrumentId.FB_GOAL_PERSIST.value: round(float(fly.central_complex.state[16:32, CH_PERSIST].max().item()), 4),
        InstrumentId.CX_NAV_MODE.value: last_telem.cx.nav_mode,
        InstrumentId.DFB_SLEEP_PHASE.value: last_telem.cx.sleep_phase,
        InstrumentId.DN_DNP09_FWD.value: round(float(dn.forward_walk_dnp09), 4),
        InstrumentId.DN_MDN_BACK.value: round(float(dn.backward_walk_mdn), 4),
        InstrumentId.DN_DNA02_STEER.value: round(float(dn.turn_left_dna02 - dn.turn_right_dna02), 4),
        InstrumentId.DN_DNG02_FLIGHT.value: round(float(dn.flight_power_dng02), 4),
        InstrumentId.GF_TTMN_V.value: round(float(fly.vnc_autopilot.state[39, CH_V].item()), 4),
        # G_glia
        InstrumentId.HEMO_SUGAR.value: round(float(fly.hemolymph.sugar_level), 4),
        InstrumentId.HEMO_AMINO.value: round(float(fly.hemolymph.amino_acid_level), 4),
        InstrumentId.ALG_SLEEP_NEED.value: round(float(fly.glia.alg_sleep_need.mean().item()), 4),
        # N_impulse
        InstrumentId.IMPULSE_OUTCOME.value: imp_outcome,
        InstrumentId.IMPULSE_EVENT_COUNT.value: imp_count,
        InstrumentId.PORT4_SHAKB_ACTIVE.value: port4_active,
        # B_body
        InstrumentId.POS_X_M.value: round(float(last_telem.pos_xyz[0]), 5),
        InstrumentId.POS_Y_M.value: round(float(last_telem.pos_xyz[1]), 5),
        InstrumentId.POS_Z_M.value: round(float(last_telem.pos_xyz[2]), 5),
        InstrumentId.WING_STROKE_AMP.value: round(float(last_telem.motor.wing_drive_lr.mean().item()), 4),
        InstrumentId.TARSAL_LOAD_MEAN.value: round(float(arena.leg_load.mean().item()), 4),
        InstrumentId.TRIPOD_COHERENCE.value: round(float(tripod_coh), 4),
        InstrumentId.SWEET_US.value: round(float(last_telem.chem.sweet_reward_us), 4),
        InstrumentId.BITTER_US.value: round(float(last_telem.chem.bitter_punish_us), 4),
        InstrumentId.YEAST_DAWED_DRIVE.value: round(float(last_telem.chem.yeast_protein_drive), 4),
        InstrumentId.PROBOSCIS_PER.value: round(float(last_telem.motor.proboscis_per), 4),
        InstrumentId.JUMP_TTMN_BURST.value: round(float(last_telem.motor.jump_ttmn_burst), 4),
    }
    return row


__all__ = ["InstrumentId", "PANEL_SCHEMA", "REGISTRY", "sample_panel"]
