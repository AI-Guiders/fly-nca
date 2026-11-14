"""Milestone M4 Verification: MB-NCA [4552 x 15] Compartments, gamma1<->gamma5 Balance, Shibire Blockade, & LH-SPIN cLTM Gate."""

from __future__ import annotations

import pytest
import torch

from fly_nca.body import FastTensorArena3D, MotorOut
from fly_nca.core import HemolymphState
from fly_nca.organs import COMP_GAMMA1, COMP_GAMMA5, AntennalSEZNCA, LHSpinGateNCA, MushroomBodyNCA


def test_mb_5_percent_sparse_coding_and_t_maze_aversive_ltd_learning() -> None:
    """1. Verify 5% APL sparse KC code, 1-shot aversive LTD in gamma1, and Sci Rep 2022 shibire blockade of gamma1 / gamma5."""
    mb = MushroomBodyNCA(n_kc=4552)

    # Odor CS+ (glomeruli 5, 6) vs Odor CS- (glomeruli 20, 21)
    cs_plus = torch.zeros(50)
    cs_plus[5] = 0.9
    cs_plus[6] = 0.8

    cs_minus = torch.zeros(50)
    cs_minus[20] = 0.9
    cs_minus[21] = 0.8

    # Before conditioning: both odors have ~5% KC sparsity and near-neutral / equal valence
    pre_plus = mb.step_mb(cs_plus)
    pre_minus = mb.step_mb(cs_minus)
    assert 0.04 <= pre_plus.kc_active_fraction <= 0.06
    assert abs(pre_plus.net_learned_valence - pre_minus.net_learned_valence) < 0.05

    # Pair CS+ with electric shock / bitter punishment (bitter_shock_us = 1.0 -> PPL1-gamma1pedc LTD!)
    for _ in range(3):
        mb.step_mb(cs_plus, bitter_shock_us=1.0)

    # Test retrieval after conditioning (no shock):
    post_plus = mb.step_mb(cs_plus)
    post_minus = mb.step_mb(cs_minus)

    # 1. Synaptic depression is localized to MBON-gamma1pedc (compartment 0) specifically for CS+, NOT CS-!
    assert float(post_plus.mbon_rates[COMP_GAMMA1]) < float(pre_plus.mbon_rates[COMP_GAMMA1]) * 0.55
    assert float(post_minus.mbon_rates[COMP_GAMMA1]) > float(pre_minus.mbon_rates[COMP_GAMMA1]) * 0.90

    # 2. Depression of MBON-gamma1pedc disinhibits avoidance MBON-gamma5 (compartment 4) -> strongly negative valence for CS+!
    assert float(post_plus.mbon_rates[COMP_GAMMA5]) > float(pre_plus.mbon_rates[COMP_GAMMA5]) * 1.3
    assert post_plus.net_learned_valence < -0.35
    assert post_plus.net_learned_valence < post_minus.net_learned_valence - 0.35

    # 3. Sci Rep 2022 shibire^ts1 blockade: blocking EITHER gamma1 OR gamma5 abolishes CS+/CS- discrimination!
    mb.set_shibire_blockade([COMP_GAMMA1], blocked=True)
    blocked_g1 = mb.step_mb(cs_plus)
    assert blocked_g1.net_learned_valence == 0.0
    mb.set_shibire_blockade([COMP_GAMMA1], blocked=False)

    mb.set_shibire_blockade([COMP_GAMMA5], blocked=True)
    blocked_g5 = mb.step_mb(cs_plus)
    assert blocked_g5.net_learned_valence == 0.0
    mb.set_shibire_blockade([COMP_GAMMA5], blocked=False)


def test_lh_spin_multimodal_glucla_and_gate_for_cltm() -> None:
    """2. Nat Commun 2019 (PMC6779902): cLTM requires ALL context modalities (odor + light + temp + texture) to match; any mismatch -> PI=0!"""
    arena = FastTensorArena3D()
    al = AntennalSEZNCA()
    lh_spin = LHSpinGateNCA()
    hemo = HemolymphState(sugar_level=0.8)

    # Training context: Odor glomerulus 12, temperature = 25.0 C, texture = 1.0, bright light
    arena.temperature_c = 25.0
    arena.surface_texture_freq = 1.0
    sensor_train = arena.step_physics(MotorOut.zeros(), dt_ms=1.0)
    sensor_train.odor_glomeruli_lr[:, 12] = 0.9
    chem_train = al.step_chemosensory(sensor_train, hemo)

    # Record immediate DA-dependent cLTM engram (aversive valence -1.0)
    lh_spin.record_cltm_engram(odor_glomerulus=12, sensor=sensor_train, valence_sign=-1.0, da_intact=True)

    # A. Retrieval in the EXACT same multimodal context -> AND-gate opens (~1.0) and retrieves full aversive cLTM!
    out_match = lh_spin.step_lh_spin(chem_train.pn_glomeruli_50, sensor_train, hemo)
    assert out_match.cltm_gate_open > 0.75
    assert out_match.cltm_retrieved_valence < -0.75

    # B. Change ONLY temperature (25 C -> 31 C) while keeping odor, light, and grid texture identical -> PI = 0!
    arena.temperature_c = 31.0
    sensor_temp_mismatch = arena.step_physics(MotorOut.zeros(), dt_ms=1.0)
    sensor_temp_mismatch.odor_glomeruli_lr[:, 12] = 0.9
    out_temp_mismatch = lh_spin.step_lh_spin(chem_train.pn_glomeruli_50, sensor_temp_mismatch, hemo)
    assert out_temp_mismatch.cltm_gate_open == 0.0
    assert out_temp_mismatch.cltm_retrieved_valence == 0.0

    # C. Restore temperature (25 C), but change tactile floor grid texture (1.0 -> 2.5, AMMC mismatch) -> PI = 0!
    arena.temperature_c = 25.0
    arena.surface_texture_freq = 2.5
    sensor_tex_mismatch = arena.step_physics(MotorOut.zeros(), dt_ms=1.0)
    sensor_tex_mismatch.odor_glomeruli_lr[:, 12] = 0.9
    out_tex_mismatch = lh_spin.step_lh_spin(chem_train.pn_glomeruli_50, sensor_tex_mismatch, hemo)
    assert out_tex_mismatch.cltm_gate_open == 0.0
    assert out_tex_mismatch.cltm_retrieved_valence == 0.0

    # D. Restore texture (1.0), but turn off ambient light (visual mismatch) -> PI = 0!
    arena.surface_texture_freq = 1.0
    sensor_light_mismatch = arena.step_physics(MotorOut.zeros(), dt_ms=1.0)
    sensor_light_mismatch.odor_glomeruli_lr[:, 12] = 0.9
    sensor_light_mismatch.ommatidia_light.zero_()
    out_light_mismatch = lh_spin.step_lh_spin(chem_train.pn_glomeruli_50, sensor_light_mismatch, hemo)
    assert out_light_mismatch.cltm_gate_open == 0.0
    assert out_light_mismatch.cltm_retrieved_valence == 0.0
