"""Milestone M2 Verification: 4D Body Physics & VNC-NCA 200-Hz Autopilots (Flight ESP, Neck Gimbal, 6 Leg Loops, GF Escape)."""

from __future__ import annotations

import torch

from fly_nca.body import FastTensorArena4D, MotorOut
from fly_nca.organs import DescendingJoystickDN, VNCAutopilotNCA


def test_decapitated_fly_standing_and_bristle_grooming_reflex() -> None:
    """1. Decapitated fly (DN = 0): stands on 6 legs and grooms dust off bristles autonomously via VNC loops."""
    arena = FastTensorArena4D()
    vnc = VNCAutopilotNCA()

    # Deposit dust on bristles while brain descending commands are zero (decapitated state)
    arena.bristle_touch = torch.ones(6)
    sensor = arena.step_physics(MotorOut.zeros(), dt_ms=1.0)

    max_groom_copy = 0.0
    for _ in range(35):
        motor, an_copy = vnc.step_autopilot(sensor, dn=DescendingJoystickDN(), dt_ms=1.0)
        max_groom_copy = max(max_groom_copy, float(an_copy[4].item()))
        sensor = arena.step_physics(motor, dt_ms=1.0)

    # Fly remains standing on ground (z == 0, ground load > 0), grooming copy fired during cleaning, and dust is gone!
    assert float(arena.pos_xyz[2].item()) == 0.0
    assert float(sensor.leg_ground_load.mean().item()) > 0.40
    assert max_groom_copy > 0.30
    assert float(arena.bristle_touch.mean().item()) < 0.25


def test_autopilot_1_and_2_flight_esp_and_neck_3d_gimbal_stabilization() -> None:
    """2. Airborne fly under strong roll/pitch wind gust stabilizes attitude via haltere GapJ ESP & Neck Gimbal."""
    arena_with_esp = FastTensorArena4D()
    arena_no_esp = FastTensorArena4D()
    vnc = VNCAutopilotNCA()

    # Place fly in mid-air (z = 0.05 m) with an initial 0.25 rad bank angle + lateral roll turbulence
    arena_with_esp.pos_xyz[2] = 0.05
    arena_no_esp.pos_xyz[2] = 0.05
    arena_with_esp.thorax_rpy[0] = 0.25
    arena_no_esp.thorax_rpy[0] = 0.25
    arena_with_esp.turbulence_torque_rpy = (8.0, 5.0, 0.0)
    arena_no_esp.turbulence_torque_rpy = (8.0, 5.0, 0.0)

    dn_hover = DescendingJoystickDN(flight_power_dng02=0.65)
    sensor = arena_with_esp.step_physics(MotorOut.zeros(), dt_ms=1.0)

    uncontrolled_motor = MotorOut.zeros()
    uncontrolled_motor.wing_drive_lr = torch.tensor([0.55, 0.55])

    for _ in range(40):
        motor, _ = vnc.step_autopilot(sensor, dn=dn_hover, dt_ms=1.0)
        sensor = arena_with_esp.step_physics(motor, dt_ms=1.0)
        arena_no_esp.step_physics(uncontrolled_motor, dt_ms=1.0)

    # Autopilot 1 (Haltere -> Wing ESP): roll rate and roll angle are strongly damped compared to uncontrolled fly
    assert abs(float(arena_with_esp.omega_rpy[0])) < abs(float(arena_no_esp.omega_rpy[0])) * 0.45

    # Autopilot 2 (Haltere -> 49 Neck MNs Gimbal): neck counter-rotates opposite to thorax roll/pitch
    # so head roll in world frame is closer to horizon than thorax roll!
    assert float(arena_with_esp.neck_rpy[0]) * float(arena_with_esp.thorax_rpy[0]) < 0.0
    assert abs(float(arena_with_esp.head_world_rpy[0])) < abs(float(arena_with_esp.thorax_rpy[0]))


def test_autopilot_3_forward_walk_moonwalker_mdn_reverse_and_lbl40_ablation() -> None:
    """3. Forward walking (DNp09), Moonwalker (MDN) backward walking, and LBL40 T3 ablation (Nat Commun 2020)."""
    # A. Forward walking via DNp09
    arena_fwd = FastTensorArena4D()
    vnc_fwd = VNCAutopilotNCA()
    s_fwd = arena_fwd.step_physics(MotorOut.zeros(), dt_ms=1.0)
    for _ in range(80):
        m_fwd, _ = vnc_fwd.step_autopilot(s_fwd, dn=DescendingJoystickDN(forward_walk_dnp09=0.85), dt_ms=1.0)
        s_fwd = arena_fwd.step_physics(m_fwd, dt_ms=1.0)
    assert float(arena_fwd.pos_xyz[0].item()) > 0.0002  # Moved forward (+x)

    # B. Moonwalker (MDN) backward walking via Port 2 shunt inversion
    arena_bwd = FastTensorArena4D()
    vnc_bwd = VNCAutopilotNCA()
    s_bwd = arena_bwd.step_physics(MotorOut.zeros(), dt_ms=1.0)
    for _ in range(80):
        m_bwd, _ = vnc_bwd.step_autopilot(s_bwd, dn=DescendingJoystickDN(backward_walk_mdn=0.85), dt_ms=1.0)
        s_bwd = arena_bwd.step_physics(m_bwd, dt_ms=1.0)
    assert float(arena_bwd.pos_xyz[0].item()) < -0.0002  # Moved backward (-x)!

    # C. Nat Commun 2020 epistasis: silencing LBL40 in T3 abolishes hind-leg tibia flexion (stance)
    vnc_ablated = VNCAutopilotNCA()
    vnc_ablated.silence_lbl40_t3()
    m_abl, _ = vnc_ablated.step_autopilot(s_fwd, dn=DescendingJoystickDN(forward_walk_dnp09=0.85), dt_ms=1.0)
    # T3 hind legs (indices 4 and 5) lose positive stance flexion on joint 2 while T1 front legs retain it
    assert float(m_abl.leg_joint_drive[4, 2]) <= 0.0
    assert float(m_abl.leg_joint_drive[5, 2]) <= 0.0
    assert float(m_abl.leg_joint_drive[0, 2]) > 0.15


def test_giant_fiber_gap_junction_escape_latency_and_frazzled_rescue() -> None:
    """4. eNeuro 2025: GF -> TTMn electrical synapse triggers takeoff in <=2 ms (wt/ICD rescue) vs delayed in frazzled LOF."""
    arena_wt = FastTensorArena4D()
    arena_lof = FastTensorArena4D()
    arena_icd = FastTensorArena4D()

    vnc_wt = VNCAutopilotNCA(frazzled_state="wt")
    vnc_lof = VNCAutopilotNCA(frazzled_state="lof")
    vnc_icd = VNCAutopilotNCA(frazzled_state="icd_rescue")

    arena_wt.looming_rate = 0.95
    arena_lof.looming_rate = 0.95
    arena_icd.looming_rate = 0.95

    s_wt = arena_wt.step_physics(MotorOut.zeros(), dt_ms=1.0)
    s_lof = arena_lof.step_physics(MotorOut.zeros(), dt_ms=1.0)
    s_icd = arena_icd.step_physics(MotorOut.zeros(), dt_ms=1.0)

    # Step 1 (t = 1 ms): GF depolarizes
    m_wt1, _ = vnc_wt.step_autopilot(s_wt, dt_ms=1.0)
    m_lof1, _ = vnc_lof.step_autopilot(s_lof, dt_ms=1.0)
    m_icd1, _ = vnc_icd.step_autopilot(s_icd, dt_ms=1.0)

    # Step 2 (t = 2 ms): Port 4 shak-B gap junction transfers current to TTMn jump motor neuron!
    m_wt2, _ = vnc_wt.step_autopilot(s_wt, dt_ms=1.0)
    m_lof2, _ = vnc_lof.step_autopilot(s_lof, dt_ms=1.0)
    m_icd2, _ = vnc_icd.step_autopilot(s_icd, dt_ms=1.0)

    assert m_wt2.jump_ttmn_burst > 0.50   # Wild-type jumps within 2 ms (~1.14 ms in vivo!)
    assert m_icd2.jump_ttmn_burst >= m_wt2.jump_ttmn_burst  # Frazzled ICD rescue restores full fast response
    assert m_lof2.jump_ttmn_burst < 0.25  # Frazzled LOF (5.31% GJ volume) fails 2-ms threshold!

    # Apply jump burst to 4D arena -> verify immediate vertical liftoff into 3D air (z > 0)
    arena_wt.step_physics(m_wt2, dt_ms=1.0)
    assert float(arena_wt.pos_xyz[2].item()) > 0.001
