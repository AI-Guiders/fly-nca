"""Laboratory Session & Protocol Runner — executes published Drosophila experimental protocols on Fly-NCA."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import torch

from fly_nca.app.scenario import SCENARIOS, ScenarioSpec, get_scenario
from fly_nca.body.arena_3d import FastTensorArena3D, OdorPlume3D
from fly_nca.composition import FlyFrameTelemetry, FlyNCAComposition
from fly_nca.core.cell_state import CH_A
from fly_nca.instruments.panel import sample_panel
from fly_nca.nervous.impulse import DebugLevel, ImpulseRecorder, format_impulse_replay
from fly_nca.organs.mushroom_body import COMP_GAMMA1, COMP_GAMMA5


@dataclass
class LabSession:
    """One open Laboratory rig + living FlyNCAComposition + Instrument Panel + Nervous Impulse Recorder."""

    scenario: ScenarioSpec
    fly: FlyNCAComposition
    arena: FastTensorArena3D
    recorder: ImpulseRecorder
    last_telem: FlyFrameTelemetry | None = None
    protocol_measurements: dict[str, Any] = field(default_factory=dict)

    @property
    def meta(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario.id,
            "title": self.scenario.title,
            "laboratory": self.scenario.laboratory.value,
            "paper_citation": self.scenario.paper_citation,
            "genotype": self.scenario.genotype,
            "wetlab_protocol": self.scenario.wetlab_protocol,
            "published_result": self.scenario.published_result,
        }

    def step(
        self,
        frames: int = 1,
        *,
        target_altitude_z: float = 0.0,
        sleep_need_override: float | None = None,
        creb_intact: bool = True,
        visual_darkness: bool = False,
        gtacr1_inhibit_fb_goal: bool = False,
    ) -> FlyFrameTelemetry | None:
        for _ in range(max(0, int(frames))):
            self.last_telem = self.fly.step_cognitive_frame(
                self.arena,
                target_altitude_z=target_altitude_z,
                sleep_need_override=sleep_need_override,
                creb_intact=creb_intact,
                visual_darkness=visual_darkness,
                gtacr1_inhibit_fb_goal=gtacr1_inhibit_fb_goal,
                recorder=self.recorder,
            )
        return self.last_telem

    def panel(self) -> dict[str, Any]:
        return sample_panel(
            self.fly,
            self.arena,
            last_telem=self.last_telem,
            recorder=self.recorder,
        )

    def impulse_replay(self) -> str:
        return format_impulse_replay(self.recorder.last_impulses(limit=200))


def open_lab(
    scenario_id: str,
    *,
    device: str = "cpu",
    store_dir: Path | None = None,
) -> LabSession:
    """Open a configured Laboratory session for a named published scenario."""
    spec = get_scenario(scenario_id)
    fly = FlyNCAComposition(device=device)
    arena = FastTensorArena3D(device=device)
    rec = ImpulseRecorder(level=DebugLevel.FULL, store_dir=store_dir)
    return LabSession(scenario=spec, fly=fly, arena=arena, recorder=rec)


def execute_scenario_protocol(
    scenario_id: str,
    *,
    device: str = "cpu",
    store_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the exact wet-lab experimental protocol for `scenario_id` and return measurements + panel."""
    lab = open_lab(scenario_id, device=device, store_dir=store_dir)
    sid = lab.scenario.id
    fly = lab.fly
    arena = lab.arena

    if sid == "pmc9213513_mb_gamma1_shibire":
        # Sci Rep 2022 (PMC9213513): CS+ (OCT, glom 10) paired with shock vs CS- (MCH, glom 22) unreinforced
        oct_clean = OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=10, intensity=0.95, sigma_m=0.5)
        oct_shock = OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=10, intensity=0.95, sigma_m=0.5, is_bitter_shock=True)
        mch_clean = OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=22, intensity=0.95, sigma_m=0.5)

        arena.plumes = [oct_clean]
        pre_cs_plus = lab.step(1)
        arena.plumes = [mch_clean]
        pre_cs_minus = lab.step(1)
        assert pre_cs_plus is not None and pre_cs_minus is not None

        # Differential conditioning: 4 frames of CS+ + shock
        arena.plumes = [oct_shock]
        for _ in range(4):
            arena.pos_xyz.zero_()
            lab.step(1)

        # Post-conditioning test in T-maze (no shock)
        arena.pos_xyz.zero_()
        arena.plumes = [oct_clean]
        post_cs_plus = lab.step(1)
        arena.pos_xyz.zero_()
        arena.plumes = [mch_clean]
        post_cs_minus = lab.step(1)
        assert post_cs_plus is not None and post_cs_minus is not None

        # Shibire^ts1 blockade at 32 C of MBON-gamma1pedc (0) or MBON-gamma5 (4)
        fly.mushroom_body.set_shibire_blockade([COMP_GAMMA1], blocked=True)
        arena.plumes = [oct_clean]
        shibire_g1 = lab.step(1)
        fly.mushroom_body.set_shibire_blockade([COMP_GAMMA1], blocked=False)

        fly.mushroom_body.set_shibire_blockade([COMP_GAMMA5], blocked=True)
        shibire_g5 = lab.step(1)
        fly.mushroom_body.set_shibire_blockade([COMP_GAMMA5], blocked=False)
        assert shibire_g1 is not None and shibire_g5 is not None

        # Restore intact post-conditioning readout for final panel
        post_cs_plus = lab.step(1)
        assert post_cs_plus is not None

        g1_pre = float(pre_cs_plus.mb.mbon_rates[COMP_GAMMA1].item())
        g1_post = float(post_cs_plus.mb.mbon_rates[COMP_GAMMA1].item())
        g5_pre = float(pre_cs_plus.mb.mbon_rates[COMP_GAMMA5].item())
        g5_post = float(post_cs_plus.mb.mbon_rates[COMP_GAMMA5].item())
        val_cs_plus = float(post_cs_plus.mb.net_learned_valence)
        val_cs_minus = float(post_cs_minus.mb.net_learned_valence)

        lab.protocol_measurements = {
            "g1_pre_cs_plus": round(g1_pre, 4),
            "g1_post_cs_plus": round(g1_post, 4),
            "g1_depression_ratio": round(g1_post / max(g1_pre, 1e-6), 4),
            "g5_pre_cs_plus": round(g5_pre, 4),
            "g5_post_cs_plus": round(g5_post, 4),
            "g5_disinhibition_ratio": round(g5_post / max(g5_pre, 1e-6), 4),
            "net_valence_cs_plus": round(val_cs_plus, 4),
            "net_valence_cs_minus": round(val_cs_minus, 4),
            "shibire_gamma1_valence": round(float(shibire_g1.mb.net_learned_valence), 4),
            "shibire_gamma5_valence": round(float(shibire_g5.mb.net_learned_valence), 4),
            "matches_published_paper": bool(
                g1_post < g1_pre * 0.55
                and g5_post > g5_pre * 1.25
                and val_cs_plus < -0.35
                and abs(val_cs_minus) < 0.08
                and float(shibire_g1.mb.net_learned_valence) == 0.0
                and float(shibire_g5.mb.net_learned_valence) == 0.0
            ),
        }

    elif sid == "pmc6779902_cltm_copper_grid":
        # Nat Commun 2019 (PMC6779902): Copper grid (texture=1.0) + 23 C + light ON conditioning of OCT (glom 12)
        arena.temperature_c = 23.0
        arena.surface_texture_freq = 1.0
        arena.plumes = [
            OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=12, intensity=0.95, sigma_m=0.5, is_bitter_shock=True)
        ]
        lab.step(2)

        # 1) Test in T-maze WITHOUT copper grid (surface_texture_freq = 2.5 -> AMMC mismatch)
        arena.plumes = [OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=12, intensity=0.95, sigma_m=0.5)]
        arena.surface_texture_freq = 2.5
        t_no_grid = lab.step(1)

        # 2) Test WITH copper grid reinstated (1.0), but temperature shifted (23 C -> 32 C)
        arena.surface_texture_freq = 1.0
        arena.temperature_c = 32.0
        t_temp_shift = lab.step(1)

        # 3) Test WITH copper grid (1.0) + 23 C + matching light (Full context reinstatement!)
        arena.temperature_c = 23.0
        t_reinstated = lab.step(1)
        assert t_no_grid is not None and t_temp_shift is not None and t_reinstated is not None

        lab.protocol_measurements = {
            "cltm_gate_reinstated": round(float(t_reinstated.lh_spin.cltm_gate_open), 4),
            "cltm_valence_reinstated": round(float(t_reinstated.lh_spin.cltm_retrieved_valence), 4),
            "cltm_gate_no_copper_grid": round(float(t_no_grid.lh_spin.cltm_gate_open), 4),
            "cltm_valence_no_copper_grid": round(float(t_no_grid.lh_spin.cltm_retrieved_valence), 4),
            "cltm_gate_temp_mismatch_32c": round(float(t_temp_shift.lh_spin.cltm_gate_open), 4),
            "cltm_valence_temp_mismatch_32c": round(float(t_temp_shift.lh_spin.cltm_retrieved_valence), 4),
            "matches_published_paper": bool(
                t_reinstated.lh_spin.cltm_gate_open > 0.75
                and t_reinstated.lh_spin.cltm_retrieved_valence < -0.75
                and t_no_grid.lh_spin.cltm_gate_open == 0.0
                and t_no_grid.lh_spin.cltm_retrieved_valence == 0.0
                and t_temp_shift.lh_spin.cltm_gate_open == 0.0
                and t_temp_shift.lh_spin.cltm_retrieved_valence == 0.0
            ),
        }

    elif sid == "pmc13518903_vt062617_wind_tunnel":
        # Nat Commun 2026 (PMC13518903): 12 cm/s laminar wind + 1% ACV pulses -> odor OFF persistence vs GtACR1
        arena.wind_world_xyz = (0.12, 0.0, 0.0)
        arena.thorax_rpy[2] = -0.5
        fly.central_complex.seed_heading(-0.5)
        arena.plumes = [OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=0, intensity=0.95, sigma_m=0.5)]

        # 12 frames odor ON (ramping across pulses)
        t_on = lab.step(12)
        assert t_on is not None
        bump_odor_on = float(fly.central_complex.state[16:32, 3].max().item())
        mode_odor_on = t_on.cx.nav_mode

        # 20 frames odor OFF (clean compensatory airflow, light OFF -> wild-type persistence)
        arena.plumes.clear()
        t_off_wt = lab.step(20, gtacr1_inhibit_fb_goal=False)
        assert t_off_wt is not None
        bump_odor_off_wt = float(fly.central_complex.state[16:32, 3].max().item())
        mode_odor_off_wt = t_off_wt.cx.nav_mode

        # 470 nm blue light ON: VT062617 > GtACR1 photoinactivation of hDeltaK21
        t_gtacr1 = lab.step(2, gtacr1_inhibit_fb_goal=True)
        assert t_gtacr1 is not None
        bump_gtacr1 = float(fly.central_complex.state[16:32, 3].max().item())

        lab.protocol_measurements = {
            "hdeltak21_bump_odor_on": round(bump_odor_on, 4),
            "nav_mode_odor_on": mode_odor_on,
            "hdeltak21_bump_after_odor_off_wt": round(bump_odor_off_wt, 4),
            "nav_mode_after_odor_off_wt": mode_odor_off_wt,
            "hdeltak21_bump_with_gtacr1_470nm": round(bump_gtacr1, 4),
            "nav_mode_with_gtacr1_470nm": t_gtacr1.cx.nav_mode,
            "matches_published_paper": bool(
                bump_odor_on > 0.25
                and bump_odor_off_wt > bump_odor_on * 0.85
                and mode_odor_off_wt in ("GOAL", "SEARCH")
                and bump_gtacr1 == 0.0
                and t_gtacr1.cx.nav_mode == "BASELINE"
            ),
        }

    elif sid == "pmc7145822_pd2a1_concentration_time_starvation":
        # eLife 2020 (PMC7145822): R37G11-Gal4 PD2a1/b1 in 50x5x1.3 mm linear chambers
        fly.hemolymph.sugar_level = 0.90  # Satiated fly
        val_fed_10e2_80s = fly.lh_spin.evaluate_pd2a1_soft_prior(0.02, 80.0, fly.hemolymph)
        val_fed_10e3_80s = fly.lh_spin.evaluate_pd2a1_soft_prior(0.001, 80.0, fly.hemolymph)
        val_fed_10e2_40s = fly.lh_spin.evaluate_pd2a1_soft_prior(0.02, 40.0, fly.hemolymph)

        fly.hemolymph.sugar_level = 0.10  # 24-h starved fly
        val_starved_10e2_80s = fly.lh_spin.evaluate_pd2a1_soft_prior(0.02, 80.0, fly.hemolymph)
        fly.hemolymph.sugar_level = 0.90
        lab.step(2)

        lab.protocol_measurements = {
            "pd2a1_fed_10e2_80s": round(val_fed_10e2_80s, 4),
            "pd2a1_fed_10e3_80s_low_conc": round(val_fed_10e3_80s, 4),
            "pd2a1_fed_10e2_40s_short_time": round(val_fed_10e2_40s, 4),
            "pd2a1_starved_24h_10e2_80s": round(val_starved_10e2_80s, 4),
            "matches_published_paper": bool(
                val_fed_10e2_80s < -0.20
                and val_fed_10e3_80s == 0.0
                and val_fed_10e2_40s == 0.0
                and val_starved_10e2_80s == 0.0
            ),
        }

    elif sid == "pmc7710706_mdn_lbl40_lul130_epistasis":
        # Nat Commun 2020 (PMC7710706): MDN > CsChrimson backward walking + SS47938 (T3 LBL40) > GtACR2 silencing
        arena.plumes = [
            OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=10, intensity=0.95, sigma_m=0.5, is_bitter_shock=True)
        ]
        t3_peaks_intact: list[float] = []
        for _ in range(10):
            arena.pos_xyz[2] = 0.0
            lab.step(1)
            t3_peaks_intact.append(
                max(
                    float(fly.vnc_autopilot.state[14 + 4 * 4, CH_A].item()),
                    float(fly.vnc_autopilot.state[14 + 5 * 4, CH_A].item()),
                )
            )
        x_backward_intact = float(arena.pos_xyz[0].item())
        t3_lbl40_intact = max(t3_peaks_intact)

        # Apply 470 nm blue light to silence T3 LBL40 (legs 4 and 5 stance cells) via GtACR2
        fly.vnc_autopilot.silence_lbl40_t3()
        t3_peaks_gtacr2: list[float] = []
        t1_peaks_unablated: list[float] = []
        for _ in range(6):
            lab.step(1)
            t3_peaks_gtacr2.append(
                max(
                    float(fly.vnc_autopilot.state[14 + 4 * 4, CH_A].item()),
                    float(fly.vnc_autopilot.state[14 + 5 * 4, CH_A].item()),
                )
            )
            t1_peaks_unablated.append(
                max(
                    float(fly.vnc_autopilot.state[14 + 0 * 4, CH_A].item()),
                    float(fly.vnc_autopilot.state[14 + 1 * 4, CH_A].item()),
                )
            )
        t3_lbl40_gtacr2 = max(t3_peaks_gtacr2)
        t1_lbl40_unablated = max(t1_peaks_unablated)

        lab.protocol_measurements = {
            "x_displacement_mdn_backward_m": round(x_backward_intact, 5),
            "t3_lbl40_stance_intact": round(t3_lbl40_intact, 4),
            "t3_lbl40_stance_gtacr2_silenced": round(t3_lbl40_gtacr2, 4),
            "t1_lbl40_stance_unablated": round(t1_lbl40_unablated, 4),
            "matches_published_paper": bool(
                x_backward_intact < 0.0
                and t3_lbl40_intact > 0.20
                and t3_lbl40_gtacr2 == 0.0
                and t1_lbl40_unablated > 0.15
            ),
        }

    elif sid == "eneuro2025_gf_frazzled_shakb":
        # eNeuro 2025 (DOI 10.1523/ENEURO.0202-25.2025): wt (9.04% shak-B) vs fra^3/fra^4 LOF (5.31%) vs FraICD rescue
        arena.looming_rate = 0.92
        telem_wt = lab.step(1)
        assert telem_wt is not None
        ttmn_v_wt = float(fly.vnc_autopilot.state[39, 0].item())
        jump_wt = float(telem_wt.motor.jump_ttmn_burst)

        fly_lof = FlyNCAComposition(frazzled_state="lof", device=fly.device)
        arena_lof = FastTensorArena3D(device=fly.device)
        arena_lof.looming_rate = 0.92
        telem_lof = fly_lof.step_cognitive_frame(arena_lof)
        ttmn_v_lof = float(fly_lof.vnc_autopilot.state[39, 0].item())
        jump_lof = float(telem_lof.motor.jump_ttmn_burst)

        fly_icd = FlyNCAComposition(frazzled_state="icd_rescue", device=fly.device)
        arena_icd = FastTensorArena3D(device=fly.device)
        arena_icd.looming_rate = 0.92
        telem_icd = fly_icd.step_cognitive_frame(arena_icd)
        ttmn_v_icd = float(fly_icd.vnc_autopilot.state[39, 0].item())
        jump_icd = float(telem_icd.motor.jump_ttmn_burst)

        lab.protocol_measurements = {
            "wt_shakb_pct": 9.04,
            "lof_shakb_pct": 5.31,
            "ttmn_v_wt": round(ttmn_v_wt, 4),
            "ttmn_v_fra_lof": round(ttmn_v_lof, 4),
            "ttmn_v_fra_icd_rescue": round(ttmn_v_icd, 4),
            "jump_burst_wt": round(jump_wt, 4),
            "jump_burst_fra_lof": round(jump_lof, 4),
            "jump_burst_fra_icd_rescue": round(jump_icd, 4),
            "matches_published_paper": bool(
                jump_wt > 0.50
                and ttmn_v_lof < ttmn_v_wt * 0.55
                and jump_lof == 0.0
                and ttmn_v_icd >= ttmn_v_wt
                and jump_icd >= jump_wt
            ),
        }

    elif sid == "pmc13475639_vpm4_autoreceptor_aggression":
        # Sci Adv 2026 (PMC13475639): VPM4 (MB113C-split-Gal4) wt vs OAa2R/mGluR autoreceptor knockdown
        odor = torch.zeros(50, device=fly.device)
        odor[0] = 0.5
        for _ in range(10):
            out_wt = fly.mushroom_body.step_mb(odor, vpm4_octopamine_drive=0.85)
        boxing_wt = float(out_wt.aggression_boxing_drive)

        fly.mushroom_body.vpm4_autoreceptor_intact = False
        fly.mushroom_body.vpm4_auto_r = 0.0
        for _ in range(10):
            out_kd = fly.mushroom_body.step_mb(odor, vpm4_octopamine_drive=0.85)
        boxing_kd = float(out_kd.aggression_boxing_drive)
        lab.step(1)

        lab.protocol_measurements = {
            "boxing_drive_vpm4_wt": round(boxing_wt, 4),
            "boxing_drive_vpm4_oaa2r_mglur_kd": round(boxing_kd, 4),
            "escalation_ratio_kd_vs_wt": round(boxing_kd / max(boxing_wt, 1e-6), 4),
            "matches_published_paper": bool(boxing_kd > boxing_wt * 1.45),
        }

    elif sid == "pmc6451361_hunger_dawed_protein_switch":
        # Open Biol 2019 (PMC6451361): Two-choice sucrose vs yeast under sugar starvation, protein starvation, & full crop
        sensor = arena.step_physics(fly.vnc_autopilot.step_vnc(arena.step_physics(torch.zeros((6, 4))), dt_ms=0.0) if False else None, dt_ms=0.0) if False else None  # noqa: E501
        from fly_nca.body.adapter import MotorOut

        sensor_patch = arena.step_physics(MotorOut.zeros(device=fly.device), dt_ms=0.0)
        sensor_patch.taste_sweet_bitter_amino = torch.tensor([0.55, 0.0, 0.60], device=fly.device)

        fly.hemolymph.sugar_level = 0.15
        fly.hemolymph.amino_acid_level = 0.90
        dig_carb = fly.antennal_sez.step_chemosensory(sensor_patch, fly.hemolymph)

        fly.hemolymph.sugar_level = 0.85
        fly.hemolymph.amino_acid_level = 0.10
        dig_prot = fly.antennal_sez.step_chemosensory(sensor_patch, fly.hemolymph)

        sensor_patch.crop_stretch = 0.92
        dig_full_crop = fly.antennal_sez.step_chemosensory(sensor_patch, fly.hemolymph)
        lab.step(1)

        lab.protocol_measurements = {
            "carb_starved_sweet_us": round(float(dig_carb.sweet_reward_us), 4),
            "carb_starved_yeast_drive": round(float(dig_carb.yeast_protein_drive), 4),
            "carb_starved_fdg_per": round(float(dig_carb.fdg_proboscis_cmd), 4),
            "protein_starved_sweet_us": round(float(dig_prot.sweet_reward_us), 4),
            "protein_starved_yeast_drive": round(float(dig_prot.yeast_protein_drive), 4),
            "protein_starved_fdg_per": round(float(dig_prot.fdg_proboscis_cmd), 4),
            "full_crop_0p92_fdg_per": round(float(dig_full_crop.fdg_proboscis_cmd), 4),
            "matches_published_paper": bool(
                dig_carb.sweet_reward_us > dig_carb.yeast_protein_drive
                and dig_carb.fdg_proboscis_cmd == 1.0
                and dig_prot.yeast_protein_drive > dig_prot.sweet_reward_us
                and dig_prot.fdg_proboscis_cmd == 1.0
                and dig_full_crop.fdg_proboscis_cmd == 0.0
            ),
        }

    elif sid == "pmc13031250_dfb_sleep_creb_consolidation":
        # Curr Sleep Med Rep 2026 (PMC13031250) & eLife 2021 (PMC9477501): dFB sleep states & CREB consolidation
        arena.plumes = [
            OdorPlume3D(pos_xyz=(0.0, 0.0, 0.0), glomerulus_id=3, intensity=0.95, sigma_m=0.5, is_bitter_shock=True)
        ]
        for _ in range(4):
            arena.pos_xyz.zero_()
            lab.step(1)

        pn_cs = fly.antennal_sez.pn_state.clone()
        kc_mask = fly.mushroom_body.compute_calyx_sparse_kc(pn_cs) > 0
        g1_before_sleep = float(fly.mushroom_body.w_mb[kc_mask, COMP_GAMMA1].mean().item())
        alpha_before_sleep = float(fly.mushroom_body.w_mb[kc_mask, 9].mean().item())

        # Test dCREB2b repressor blocker first (no consolidation)
        fly.mushroom_body.consolidate_sleep_creb(creb_intact=False)
        alpha_with_dcreb2b_blocker = float(fly.mushroom_body.w_mb[kc_mask, 9].mean().item())

        # Now run 32 frames under high astrocyte sleep pressure (0.95) with intact CREB
        arena.plumes.clear()
        phases_seen: set[str] = set()
        for _ in range(32):
            t_sleep = lab.step(1, sleep_need_override=0.95, creb_intact=True)
            if t_sleep is not None:
                phases_seen.add(t_sleep.cx.sleep_phase)

        alpha_after_active_sleep = float(fly.mushroom_body.w_mb[kc_mask, 9].mean().item())

        lab.protocol_measurements = {
            "gamma1_stm_weight_pre_sleep": round(g1_before_sleep, 4),
            "alpha1_ltm_weight_pre_sleep": round(alpha_before_sleep, 4),
            "alpha1_ltm_weight_dcreb2b_blocker": round(alpha_with_dcreb2b_blocker, 4),
            "alpha1_ltm_weight_post_active_sleep": round(alpha_after_active_sleep, 4),
            "sleep_phases_observed": sorted(phases_seen),
            "matches_published_paper": bool(
                "SLOW_WAVE" in phases_seen
                and "ACTIVE_REM" in phases_seen
                and alpha_with_dcreb2b_blocker == alpha_before_sleep
                and alpha_after_active_sleep < alpha_before_sleep - 0.05
            ),
        }

    return {
        "scenario": lab.meta,
        "protocol_measurements": lab.protocol_measurements,
        "panel": lab.panel(),
    }


__all__ = ["LabSession", "SCENARIOS", "execute_scenario_protocol", "open_lab"]
