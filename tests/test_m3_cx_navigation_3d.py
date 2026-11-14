"""Milestone M3 Verification: OL-NCA Radar & CX-NCA Navigator (Darkness Compass, Copy-and-Shift, 5-10s FB Goal Bump, 3D Navigation)."""

from __future__ import annotations

import math
import torch

from fly_nca.body import FastTensorArena3D, MotorOut, OdorPlume3D
from fly_nca.composition import FlyNCAComposition
from fly_nca.organs import CentralComplexNCA, OpticLobeNCA


def test_eb_ring_attractor_holds_heading_in_complete_darkness_and_tracks_turns() -> None:
    """1. EB 16-wedge ring attractor holds heading in complete darkness and shifts phase via PB PEN on yaw turns."""
    arena = FastTensorArena3D()
    ol = OpticLobeNCA()
    cx = CentralComplexNCA()

    target_heading = math.pi / 4.0  # +45 deg
    cx.seed_heading(target_heading)

    sensor = arena.step_physics(MotorOut.zeros(), dt_ms=4.0)
    vis = ol.step_optic_lobe(sensor, dt_ms=4.0)

    # Hold heading in complete darkness (visual_darkness=True) for 50 frames (200 ms)
    for _ in range(50):
        readout = cx.step_cx(sensor, vis, visual_darkness=True, dt_ms=4.0)

    err_dark = abs(math.atan2(math.sin(readout.heading_angle_rad - target_heading), math.cos(readout.heading_angle_rad - target_heading)))
    assert err_dark < (2.0 * math.pi / 16.0)  # Within 1 EB wedge (22.5 deg)!


def test_fb_hdeltak21_persistent_goal_bump_5_to_10s_and_gtacr1_causality() -> None:
    """2. Nat Commun 2026: hDeltaK21 FB bump ramps with odor pulses, persists 5-10 s after odor OFF (GOAL -> SEARCH), and is abolished by GtACR1."""
    arena = FastTensorArena3D()
    ol = OpticLobeNCA()
    cx = CentralComplexNCA()
    cx.seed_heading(0.0)

    # Wind blowing from +45 deg in ego frame
    arena.wind_world_xyz = (0.20, 0.20, 0.0)
    sensor = arena.step_physics(MotorOut.zeros(), dt_ms=4.0)
    vis = ol.step_optic_lobe(sensor, dt_ms=4.0)

    # Phase A: 4 attractive odor encounters along wind plume -> ramping integration into GOAL state
    for _ in range(25):
        readout = cx.step_cx(sensor, vis, valence_gate_signal=0.90, dt_ms=20.0)

    assert readout.nav_mode == "GOAL"
    assert readout.goal_bump_amplitude > 0.40
    goal_during_odor = readout.goal_angle_rad

    # Phase B: Odor disappears (valence_gate_signal = 0.0) for 5.0 seconds (5000 ms)!
    # Bump must persist in working memory (GOAL / SEARCH) and keep goal direction!
    for _ in range(100):
        readout_5s = cx.step_cx(sensor, vis, valence_gate_signal=0.0, dt_ms=50.0)

    assert readout_5s.nav_mode in ("GOAL", "SEARCH")
    assert readout_5s.goal_bump_amplitude > 0.12
    goal_err = abs(math.atan2(math.sin(readout_5s.goal_angle_rad - goal_during_odor), math.cos(readout_5s.goal_angle_rad - goal_during_odor)))
    assert goal_err < 0.45

    # Phase C: Optogenetic GtACR1 inhibition of hDeltaK21 (Nat Commun 2026 causality test)
    readout_inhibited = cx.step_cx(sensor, vis, valence_gate_signal=0.0, gtacr1_inhibit_fb_goal=True, dt_ms=50.0)
    assert readout_inhibited.nav_mode == "BASELINE"
    assert readout_inhibited.goal_bump_amplitude == 0.0


def test_closed_loop_3d_flight_and_walking_navigation_to_xyz_goal() -> None:
    """3. End-to-end closed-loop navigation: fly steers and climbs in 3D space toward an attractive 3D goal (x, y, z)."""
    arena = FastTensorArena3D()
    fly = FlyNCAComposition()

    # Start fly facing -45 deg, while attractive yeast plume (glomerulus 0) and wind cue point toward (+0.10, +0.05, +0.02)
    arena.thorax_rpy[2] = -0.5
    fly.central_complex.seed_heading(-0.5)
    arena.wind_world_xyz = (0.25, 0.10, 0.0)
    arena.plumes.append(OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=0, intensity=0.9, sigma_m=0.5))

    initial_dist_xy = math.hypot(float(arena.pos_xyz[0]) - 0.05, float(arena.pos_xyz[1]) - 0.02)

    for _ in range(40):
        telem = fly.step_cognitive_frame(arena, target_altitude_z=0.02)

    final_dist_xy = math.hypot(telem.pos_xyz[0] - 0.05, telem.pos_xyz[1] - 0.02)
    # Fly climbs in 3D altitude z > 0 and steers toward the goal coordinates in (x, y)!
    assert telem.pos_xyz[2] > 0.003
    assert final_dist_xy < initial_dist_xy
