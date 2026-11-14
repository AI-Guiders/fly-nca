"""Milestone M5 Verification: Hunger Soft-Prior Gating (PD2a1/b1), DA-WED Protein Switch, VPM4 Aggression, & Active Sleep Consolidation."""

from __future__ import annotations

import torch

from fly_nca.body import FastTensorArena4D, MotorOut
from fly_nca.core import HemolymphState
from fly_nca.organs import (
    COMP_GAMMA1,
    AntennalSEZNCA,
    CentralComplexNCA,
    LHSpinGateNCA,
    MushroomBodyNCA,
    OpticLobeNCA,
)


def test_pd2a1_b1_naive_avoidance_is_soft_prior_gated_by_concentration_time_and_hunger() -> None:
    """1. eLife 2020 (PMC7145822): PD2a1/b1 avoidance depends on concentration, exposure time, and vanishes under starvation."""
    lh = LHSpinGateNCA()
    fed_hemo = HemolymphState(sugar_level=0.90)
    starved_hemo = HemolymphState(sugar_level=0.10)

    # A. Fed fly at high concentration (10^-2 = 0.02) and 80 s exposure -> strong avoidance (< -0.2)
    val_fed_80s = lh.evaluate_pd2a1_soft_prior(aversive_odor_conc=0.02, exposure_time_s=80.0, hemolymph=fed_hemo)
    assert val_fed_80s < -0.20

    # B. Low concentration (10^-3 = 0.001) -> no naive avoidance (0.0)
    val_low_conc = lh.evaluate_pd2a1_soft_prior(aversive_odor_conc=0.001, exposure_time_s=80.0, hemolymph=fed_hemo)
    assert val_low_conc == 0.0

    # C. Short exposure time (40 s) -> no naive avoidance (0.0)
    val_short_time = lh.evaluate_pd2a1_soft_prior(aversive_odor_conc=0.02, exposure_time_s=40.0, hemolymph=fed_hemo)
    assert val_short_time == 0.0

    # D. Starved fly (sugar_level = 0.10 -> high NPF/sNPF) -> naive avoidance is completely suppressed (0.0)!
    val_starved = lh.evaluate_pd2a1_soft_prior(aversive_odor_conc=0.02, exposure_time_s=80.0, hemolymph=starved_hemo)
    assert val_starved == 0.0


def test_da_wed_amino_acid_hunger_switch_and_crop_stretch_stop_loop() -> None:
    """2. Open Biol 2019 & BANC: protein deprivation switches preference to yeast (DA-WED), and crop stretch stops feeding."""
    arena = FastTensorArena4D()
    al_sez = AntennalSEZNCA()

    sensor = arena.step_physics(MotorOut.zeros(), dt_ms=1.0)
    sensor.taste_sweet_bitter_amino = torch.tensor([0.5, 0.0, 0.6])  # Both sugar and yeast present

    # Sugar-deprived but protein-replete fly prefers sugar
    hemo_sugar_hungry = HemolymphState(sugar_level=0.20, amino_acid_level=0.90)
    dig_sugar = al_sez.step_chemosensory(sensor, hemo_sugar_hungry)
    assert dig_sugar.sweet_reward_us > dig_sugar.yeast_protein_drive
    assert dig_sugar.fdg_proboscis_cmd == 1.0

    # Amino-acid (protein) deprived fly activates DA-WED -> switches preference from sugar to yeast/protein!
    hemo_protein_hungry = HemolymphState(sugar_level=0.80, amino_acid_level=0.15)
    dig_protein = al_sez.step_chemosensory(sensor, hemo_protein_hungry)
    assert dig_protein.yeast_protein_drive > dig_protein.sweet_reward_us
    assert dig_protein.fdg_proboscis_cmd == 1.0

    # Visceral crop fullness (> 0.85) shuts off Fdg proboscis extension (BANC crop -> pharynx stop loop!)
    sensor.crop_stretch = 0.92
    dig_full_crop = al_sez.step_chemosensory(sensor, hemo_sugar_hungry)
    assert dig_full_crop.fdg_proboscis_cmd == 0.0


def test_vpm4_autoreceptor_knockdown_escalates_aggression_boxing() -> None:
    """3. Sci Adv 2026 (PMC13475639): VPM4 (OA+Glu) -> MBON-11 (GluCla); knocking down OAa2R/mGluR autoreceptors escalates boxing."""
    mb_wt = MushroomBodyNCA(n_kc=512)
    mb_knockdown = MushroomBodyNCA(n_kc=512)
    mb_knockdown.vpm4_autoreceptor_intact = False

    odor = torch.zeros(50)
    odor[0] = 0.5

    for _ in range(10):
        out_wt = mb_wt.step_mb(odor, vpm4_octopamine_drive=0.85)
        out_kd = mb_knockdown.step_mb(odor, vpm4_octopamine_drive=0.85)

    # Knockdown of presynaptic OAa2R/mGluR autoreceptors causes significantly higher aggressive escalation!
    assert out_kd.aggression_boxing_drive > out_wt.aggression_boxing_drive * 1.45


def test_dfb_sleep_homeostasis_and_creb_ltm_consolidation_from_gamma_to_alpha_beta() -> None:
    """4. Curr Sleep Med Rep 2026: high sleep need triggers dFB Slow-Wave/Active Sleep and CREB transfers gamma1 memory to alpha/beta."""
    arena = FastTensorArena4D()
    ol = OpticLobeNCA()
    cx = CentralComplexNCA()
    mb = MushroomBodyNCA(n_kc=1024)

    cs_plus = torch.zeros(50)
    cs_plus[3] = 0.9
    # Train short-term memory in gamma1 (compartment 0)
    for _ in range(3):
        mb.step_mb(cs_plus, bitter_shock_us=1.0)

    kc_mask = mb.compute_calyx_sparse_kc(cs_plus) > 0
    # Before sleep: gamma1 (0) is depressed (< 0.5), while LTM alpha1 (9) is still naive (~0.98)
    assert float(mb.w_mb[kc_mask, COMP_GAMMA1].mean()) < 0.50
    assert float(mb.w_mb[kc_mask, 9].mean()) > 0.90

    # Sleep deprivation or dCREB2b blocker (creb_intact=False): no transfer to alpha/beta LTM
    mb.consolidate_sleep_creb(creb_intact=False)
    assert float(mb.w_mb[kc_mask, 9].mean()) > 0.90

    # High homeostatic sleep pressure activates dFB (R23E10) -> transitions through Slow-Wave and Active Sleep
    sensor = arena.step_physics(MotorOut.zeros(), dt_ms=4.0)
    vis = ol.step_optic_lobe(sensor, dt_ms=4.0)
    saw_slow_wave = False
    saw_active_rem = False
    for _ in range(45):
        cx_out = cx.step_cx(sensor, vis, sleep_need_alg=0.95, ppl1_wake_da=0.05, dt_ms=10.0)
        if cx_out.sleep_phase == "SLOW_WAVE":
            saw_slow_wave = True
        elif cx_out.sleep_phase == "ACTIVE_REM":
            saw_active_rem = True
            mb.consolidate_sleep_creb(creb_intact=True, transfer_rate=0.60)

    assert saw_slow_wave and saw_active_rem
    # After Active Sleep with intact CREB: LTM compartment alpha1 (9) has consolidated the depressed trace from gamma1!
    assert float(mb.w_mb[kc_mask, 9].mean()) < 0.65
