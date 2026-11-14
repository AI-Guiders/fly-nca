"""Laboratory Rig Presets — physical wet-lab experimental apparatuses from published Drosophila papers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LaboratoryPreset(str, Enum):
    """Canonical wet-lab experimental apparatuses reproduced in FastTensorArena4D."""

    T_MAZE_PAVLOVIAN_TUBE = "t_maze_pavlovian_tube"
    COPPER_GRID_CONTEXT_T_MAZE = "copper_grid_context_t_maze"
    LAMINAR_WIND_TUNNEL_BALL = "laminar_wind_tunnel_ball"
    LINEAR_20_CHAMBER_OLFACTOMETER = "linear_20_chamber_olfactometer"
    TETHERED_BALL_200FPS_DEEPLABCUT = "tethered_ball_200fps_deeplabcut"
    GIANT_FIBER_ELECTROPHYS_RIG = "giant_fiber_electrophys_rig"
    SLIDING_DIVIDER_FIGHT_ARENA = "sliding_divider_fight_arena"
    NUTRITIONAL_CAFE_CHAMBER = "nutritional_cafe_chamber"
    SLEEP_ARC_2P_RIG = "sleep_arc_2p_rig"


@dataclass(frozen=True, slots=True)
class LaboratoryRigSpec:
    """Self-contained description of the physical wet-lab apparatus and its sensor/effector mapping."""

    preset: LaboratoryPreset
    title: str
    paper_source: str
    physical_dimensions_and_conditions: str
    arena_mapping: str


LABORATORY_RIGS: dict[LaboratoryPreset, LaboratoryRigSpec] = {
    LaboratoryPreset.T_MAZE_PAVLOVIAN_TUBE: LaboratoryRigSpec(
        preset=LaboratoryPreset.T_MAZE_PAVLOVIAN_TUBE,
        title="Pavlovian Olfactory Conditioning Tube & Two-Arm T-Maze + 2P homer-GCaMP Imaging Rig",
        paper_source=(
            "Hancock et al. / Fiala & Nawrot (2022) Sci Rep 12:10456, DOI 10.1038/s41598-022-14413-5, PMC9213513 "
            "('Visualization of learning-induced synaptic plasticity in output neurons of the Drosophila mushroom body γ-lobe')"
        ),
        physical_dimensions_and_conditions=(
            "Differential aversive conditioning pairing CS+ odor (3-octanol OCT or 4-methylcyclohexanol MCH) with "
            "electric shock (12 pulses of 60–90 V for 60 s) vs unpaired CS- odor; postsynaptic Ca2+ imaged via "
            "MBON-specific homer-GCaMP in γ1–γ5 compartments; behavioral choice tested in a two-arm T-maze under "
            "permissive (23 °C) vs restrictive (32 °C) UAS-Shibire^ts1 blockade of MBON-γ1pedc or MBON-γ5."
        ),
        arena_mapping=(
            "OdorPlume3D for CS+ (glomerulus 10 = OCT) and CS- (glomerulus 22 = MCH); aversive US delivered via "
            "is_bitter_shock=True; UAS-Shibire^ts1 synaptic blockade mapped via MushroomBodyNCA.set_shibire_blockade."
        ),
    ),
    LaboratoryPreset.COPPER_GRID_CONTEXT_T_MAZE: LaboratoryRigSpec(
        preset=LaboratoryPreset.COPPER_GRID_CONTEXT_T_MAZE,
        title="Context-Reinstatement T-Maze with Copper Grid, Monochromatic Light, and Temperature Control",
        paper_source=(
            "Zhao et al. / Zhong (2019) Nat Commun 10:4550, DOI 10.1038/s41467-019-12436-7, PMC6779902 "
            "('Long-term memory is formed immediately without the need for protein synthesis-dependent consolidation in Drosophila')"
        ),
        physical_dimensions_and_conditions=(
            "23 °C, 60% RH room; training tube lined with copper grid: 90 s air -> 60 s CS+ (OCT 15 µl/10 ml or "
            "MCH 10 µl/10 ml) paired with 12× 60 V shocks -> 45 s air -> 60 s CS- -> 45 s air. "
            "Modified T-maze testing arms reinstate the tactile copper grid + training temperature (23 °C vs 32 °C) + "
            "ambient light color."
        ),
        arena_mapping=(
            "FastTensorArena4D with surface_texture_freq=1.0 (copper grid present) vs 2.5 (grid absent/changed), "
            "temperature_c=23.0 vs 32.0, ommatidia_light ON vs OFF, and LHSpinGateNCA 4-branch GluCla conjunction gate."
        ),
    ),
    LaboratoryPreset.LAMINAR_WIND_TUNNEL_BALL: LaboratoryRigSpec(
        preset=LaboratoryPreset.LAMINAR_WIND_TUNNEL_BALL,
        title="Closed-Loop Air-Supported Ball with Rotary Wind Union & 12 cm/s Laminar Wind Tunnel",
        paper_source=(
            "run-time 2P & wind-tunnel setup from Nagel Lab (2026) Nat Commun, DOI 10.1038/s41467-026-75945-2, PMC13518903 "
            "('Neural dynamics for working memory and evidence integration during olfactory navigation in Drosophila')"
        ),
        physical_dimensions_and_conditions=(
            "1) Head-fixed fly on air-supported ball with motorized 3-channel rotary union (0.2–0.6 L/min wind, "
            "0.3 L/min 0.5% apple cider vinegar ACV via 3.5 cm Teflon tube 5 mm ID at 2 cm from antennae) + 2P imaging of VT062617 (hΔK21). "
            "2) Freely-walking shallow acrylic wind tunnel with 12 cm/s laminar airflow, 1 Hz (500 ms) pulses of 1% ACV, "
            "50 Hz Basler IR tracking (850 nm), and 470 nm LED panel (61 µW/mm²) for VT062617 > UAS-GtACR1 silencing."
        ),
        arena_mapping=(
            "FastTensorArena4D wind_world_xyz=(0.12, 0.0, 0.0) m/s (12 cm/s laminar wind), ACV OdorPlume3D pulses "
            "(glomerulus 0), CentralComplexNCA cells 16..31 (VT062617 / hΔK21 with tau_persist=7500 ms), and gtacr1_inhibit_fb_goal flag."
        ),
    ),
    LaboratoryPreset.LINEAR_20_CHAMBER_OLFACTOMETER: LaboratoryRigSpec(
        preset=LaboratoryPreset.LINEAR_20_CHAMBER_OLFACTOMETER,
        title="20-Chamber Polycarbonate Linear Olfactometer (50×5×1.3 mm) with Dual Converging Flows",
        paper_source=(
            "ito / Parnas Lab (2020) eLife 9:e55589 / Sci Rep, PMC7145822 "
            "('Starvation, concentration, and exposure time modulate PD2a1/b1 lateral horn output neuron contribution to odor avoidance')"
        ),
        physical_dimensions_and_conditions=(
            "Single flies in 50 × 5 × 1.3 mm transparent polycarbonate chambers backlit at 940 nm inside a temperature-controlled "
            "incubator; 0.3 L/min odor + 2.7 L/min carrier air split across 20 chambers (0.15 L/min per half-chamber converging at center); "
            "odors tested at 10^-2 (0.01–0.02) vs 10^-3 (0.001) dilution across 40 s, 80 s, and 120 s in fed vs 24-h water-starved flies."
        ),
        arena_mapping=(
            "LHSpinGateNCA.evaluate_pd2a1_soft_prior(aversive_odor_conc, exposure_time_s, hemolymph) with "
            "HemolymphState(sugar_level=0.90 fed vs 0.10 24h-starved)."
        ),
    ),
    LaboratoryPreset.TETHERED_BALL_200FPS_DEEPLABCUT: LaboratoryRigSpec(
        preset=LaboratoryPreset.TETHERED_BALL_200FPS_DEEPLABCUT,
        title="6-mm Polypropylene Air-Supported Ball with 200-fps DeepLabCut 24-DOF Joint Tracking & Dual Optogenetics",
        paper_source=(
            "Feng, Wallace, Dickson et al. (2020) Nat Commun 11:6101, DOI 10.1038/s41467-020-19936-x, PMC7710706 "
            "('Distributed control of motor circuits for backward walking in Drosophila')"
        ),
        physical_dimensions_and_conditions=(
            "Tethered (intact or decapitated) fly on 6-mm Spherotech polypropylene ball; 200-fps video tracking of 6 legs × 4 joints "
            "(body-coxa, coxa-trochanter, femur-tibia, tibia-tarsus); 660 nm red light activates MDN > CsChrimson (25 s trial), "
            "while 470 nm blue light (10 s pulse at t=10 s) silences T3 LBL40 (SS47938, stance tibia flexion) or LUL130 (SS50974, swing leg lift) via GtACR2."
        ),
        arena_mapping=(
            "VNCAutopilotNCA 6-leg loops (cells 14..37: LBL40 stance, LUL130 swing, Coxa_Fwd, Coxa_Bwd) driven by "
            "DescendingJoystickDN(backward_walk_mdn=1.0) with selective GtACR2 silencing via VNCAutopilotNCA.ablation_mask."
        ),
    ),
    LaboratoryPreset.GIANT_FIBER_ELECTROPHYS_RIG: LaboratoryRigSpec(
        preset=LaboratoryPreset.GIANT_FIBER_ELECTROPHYS_RIG,
        title="Giant Fiber (GF -> TTMn) Intracellular Electrophysiology & IMARIS 3D shaking-B Gap Junction Rig",
        paper_source=(
            "Kennedy, Broadie, Godenschwege et al. (2025) eNeuro 12, DOI 10.1523/ENEURO.0202-25.2025 "
            "('Frazzled/DCC Regulates Gap Junction Formation at a Drosophila Giant Synapse')"
        ),
        physical_dimensions_and_conditions=(
            "Recording TTM jump muscle / TTMn motor neuron latency and response frequency in T2 upon Giant Fiber stimulation "
            "across genotypes: control wt (9.04% shaking-B terminal volume, 1.14 ms latency, 79.6% response), "
            "fra^3/fra^4 LOF (5.31% shaking-B volume, 1.23–1.33 ms latency, 51.5–52.7% response), and "
            "R91H05-Gal4 > UAS-FraICD rescue (0.97 ms latency, 98.4% response)."
        ),
        arena_mapping=(
            "VNCAutopilotNCA Port 4 electrical synapse (cell 38 GF -> cell 39 TTMn) parameterized by "
            "frazzled_state in {'wt', 'lof', 'icd_rescue'} using exact eNeuro 2025 shaking-B volume fractions (9.04% vs 5.31%)."
        ),
    ),
    LaboratoryPreset.SLIDING_DIVIDER_FIGHT_ARENA: LaboratoryRigSpec(
        preset=LaboratoryPreset.SLIDING_DIVIDER_FIGHT_ARENA,
        title="22×16 mm Sliding-Divider Male Aggression Chamber with Central Yeast/Sucrose Cup & BORIS Scoring",
        paper_source=(
            "Trannoy & Certel Labs (2026) Sci Adv 12:eaec4153, DOI 10.1126/sciadv.aec4153, PMC13475639 "
            "('An octopamine-glutamate cotransmitting neuron gates aggressive escalation through layered inhibition in Drosophila')"
        ),
        physical_dimensions_and_conditions=(
            "22-mm diameter × 16-mm height sliding chamber at ZT0–ZT3 (25 °C or 29 °C for dTrpA1) with central 13 × 6 mm food cup "
            "(sucrose + rehydrated yeast); 10 min acclimation followed by divider removal and 35 min Basler Pylon recording; "
            "comparing MB113C-split-Gal4 (VPM4) control vs VPM4 > OAα2R-RNAi and VPM4 > mGluR-RNAi (~85% fights escalate to >=2 boxing bouts)."
        ),
        arena_mapping=(
            "MushroomBodyNCA VPM4 cotransmission (OA + Glu -> MBON-11 GluCla in γ1) with presynaptic autoreceptor brake "
            "vpm4_autoreceptor_intact=True (wt) vs False (OAα2R / mGluR knockdown)."
        ),
    ),
    LaboratoryPreset.NUTRITIONAL_CAFE_CHAMBER: LaboratoryRigSpec(
        preset=LaboratoryPreset.NUTRITIONAL_CAFE_CHAMBER,
        title="Two-Choice Nutritional Geometry Chamber (Sucrose Gr5a vs Yeast Ir76b & Bitter Gr66a)",
        paper_source=(
            "Lin, Senapati & Tsao (2019) Open Biol 9:180259, DOI 10.1098/rsob.180259, PMC6451361 "
            "('Neural basis of hunger-driven behaviour in Drosophila')"
        ),
        physical_dimensions_and_conditions=(
            "Adult flies tested under carbohydrate deprivation (low DILPs, high AKH/NPF/sNPF -> TH-VUM DopEcR sensitizes Gr5a sugar "
            "and sNPF/GABA inhibits Gr66a bitter, unlocking Fdg PER and MP1 food-seeking) vs amino-acid / glutamine deprivation "
            "(sensitizes Ir76b yeast GRNs and activates DA-WED -> FB-LAL DopR2 protein seeking while inhibiting PLP DopR1 sugar intake)."
        ),
        arena_mapping=(
            "AntennalSEZNCA + HemolymphState(sugar_level, amino_acid_level) + FastTensorArena4D crop_stretch visceral stop loop."
        ),
    ),
    LaboratoryPreset.SLEEP_ARC_2P_RIG: LaboratoryRigSpec(
        preset=LaboratoryPreset.SLEEP_ARC_2P_RIG,
        title="Sleep Homeostasis & LFP/2P Recording Rig (dFB R23E10 / ER5 Slow-Wave & Active Sleep Consolidation)",
        paper_source=(
            "Curr Sleep Med Rep (2026) DOI 10.1007/s40675-025-00352-w, PMC13031250 "
            "('Current Evidence for Sleep States in Drosophila') & Hulse et al. (2021) eLife 10:e66039, PMC9477501"
        ),
        physical_dimensions_and_conditions=(
            "Multi-stage sleep recording: astrocyte (ALG) sleep pressure activates dFB tangential neurons (R23E10: FB6A/C/E/G/I/Z, FB7A/K) "
            "against wake-promoting PPL1 DANs; transitions through Slow-Wave Sleep (<4 Hz in R5/EB) and Active Sleep (1–5 min wake-like REM bouts "
            "with elevated synaptic/axonal guidance gene expression and CREB-dependent transfer of MB γ memory to α/β LTM)."
        ),
        arena_mapping=(
            "GlialSubstrateNCA (alg_sleep_need) + CentralComplexNCA cells 64..65 (dFB R23E10 & ER5) + "
            "MushroomBodyNCA.consolidate_sleep_creb(creb_intact=True vs UAS-dCREB2b blocker)."
        ),
    ),
}
