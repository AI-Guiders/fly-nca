"""Named Wet-Lab Experimental Scenarios — SSOT mapping Original Paper -> Exact Setup -> Published Result."""

from __future__ import annotations

from dataclasses import dataclass

from fly_nca.app.laboratory import LABORATORY_RIGS, LaboratoryPreset, LaboratoryRigSpec


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Self-contained specification of a wet-lab experiment from a published Drosophila paper."""

    id: str
    title: str
    laboratory: LaboratoryPreset
    paper_citation: str
    genotype: str
    wetlab_protocol: str
    published_result: str
    default_frames: int = 16
    verify_suites: tuple[str, ...] = ("gate", "wetlab_experiments", "instruments")

    @property
    def rig(self) -> LaboratoryRigSpec:
        return LABORATORY_RIGS[self.laboratory]

    @property
    def laboratory_label(self) -> str:
        return self.laboratory.value


SCENARIOS: dict[str, ScenarioSpec] = {
    "pmc9213513_mb_gamma1_shibire": ScenarioSpec(
        id="pmc9213513_mb_gamma1_shibire",
        title="MBON-γ1pedc Associative LTD, γ1-|γ5 Feedforward Balance, & Shibire^ts1 Blockade",
        laboratory=LaboratoryPreset.T_MAZE_PAVLOVIAN_TUBE,
        paper_citation=(
            "Sci Rep 2022, DOI 10.1038/s41598-022-14413-5, PMC9213513 (Figs. 2, 4, 5 — homer-GCaMP & Shibire^ts1 T-maze)"
        ),
        genotype="MBON-γ1pedc > homer-GCaMP / UAS-Shibire^ts1 & MBON-γ5/β'2 > UAS-Shibire^ts1",
        wetlab_protocol=(
            "Pair CS+ odor (OCT, glom 10) with electric shock (12× 60–90V) while CS- (MCH, glom 22) is unreinforced. "
            "Measure postsynaptic homer-GCaMP activity in MBON-γ1pedc (compartment 0) and downstream MBON-γ5 (compartment 4), "
            "then test CS+ vs CS- discrimination at 32 °C under Shibire^ts1 blockade of γ1 or γ5."
        ),
        published_result=(
            "1) Learning-induced synaptic depression (LTD) is specific to MBON-γ1pedc for CS+ (not CS-), which disinhibits "
            "avoidance MBON-γ5 via feedforward inhibition (γ1 -| γ5). 2) Blocking EITHER MBON-γ1pedc OR MBON-γ5 with "
            "Shibire^ts1 collapses CS+/CS- discrimination to chance (net learned valence = 0.0)."
        ),
        default_frames=12,
        verify_suites=("gate", "mb_lh_plasticity", "wetlab_experiments", "instruments"),
    ),
    "pmc6779902_cltm_copper_grid": ScenarioSpec(
        id="pmc6779902_cltm_copper_grid",
        title="Context-Dependent LTM (cLTM): Copper Grid + Temperature + Light Multimodal Gate",
        laboratory=LaboratoryPreset.COPPER_GRID_CONTEXT_T_MAZE,
        paper_citation=(
            "Nat Commun 2019, DOI 10.1038/s41467-019-12436-7, PMC6779902 (Figs. 1a–c, 2a–d, 3–5)"
        ),
        genotype="w1118 / TH-Gal4 > UAS-Shibire^ts1 / dDA1^dumb2 / OK107-Gal4 > UAS-Shibire^ts1",
        wetlab_protocol=(
            "Single-trial conditioning in a tube with copper grid (60V shock + OCT 15 µl/10 ml at 23 °C, 60% RH). "
            "Test retrieval in T-maze with copper grid reinstated + matching light + 23 °C vs mismatching any single "
            "modality (removing copper grid texture, shifting temperature 23 °C -> 32 °C, or turning off light), and "
            "under TH-Gal4/dDA1 dopamine blockade."
        ),
        published_result=(
            "1) Full cLTM retrieval (PI > 0.75) when copper grid + temperature + visual light all match encoding context. "
            "2) Mismatching ANY single contextual modality (grid absent, temperature shift, or light change) abolishes "
            "cLTM retrieval (PI = 0). 3) Blockade of TH-Gal4 / dDA1 during encoding prevents cLTM formation."
        ),
        default_frames=10,
        verify_suites=("gate", "mb_lh_plasticity", "wetlab_experiments", "instruments"),
    ),
    "pmc13518903_vt062617_wind_tunnel": ScenarioSpec(
        id="pmc13518903_vt062617_wind_tunnel",
        title="VT062617 (hΔK21) Persistent Goal Bump (5–10 s after Odor OFF) & GtACR1 Optogenetic Silencing",
        laboratory=LaboratoryPreset.LAMINAR_WIND_TUNNEL_BALL,
        paper_citation=(
            "Nat Commun 2026, DOI 10.1038/s41467-026-75945-2, PMC13518903 (Figs. 1, 3, 5, 6 — Nagel Lab)"
        ),
        genotype="VT062617-Gal4 (hΔK21 FB local) > UAS-GCaMP / UAS-GtACR1 (470 nm 61 µW/mm²)",
        wetlab_protocol=(
            "Deliver pulses of 0.5–1% apple cider vinegar (ACV) in 12 cm/s laminar airflow (1 Hz, 500 ms pulses; "
            "saturates around 4 pulses), then switch odor valve OFF while maintaining clean compensatory airflow. "
            "Compare wild-type (light OFF) vs VT062617 > GtACR1 inhibition (470 nm blue light ON)."
        ),
        published_result=(
            "1) Odor pulses build a localized allocentric goal bump in VT062617 (hΔK21) that persists for 5–10 s "
            "after odor offset, maintaining GOAL/SEARCH upwind navigation. 2) GtACR1 photoinactivation (470 nm) "
            "collapses the persistent FB goal bump and abolishes post-odor upwind persistence."
        ),
        default_frames=36,
        verify_suites=("gate", "cx_navigation", "wetlab_experiments", "instruments"),
    ),
    "pmc7145822_pd2a1_concentration_time_starvation": ScenarioSpec(
        id="pmc7145822_pd2a1_concentration_time_starvation",
        title="LH PD2a1/b1 Naive Avoidance Soft Prior: 10^-2 vs 10^-3 Dilution, 40s vs 80s Time, & 24h Starvation",
        laboratory=LaboratoryPreset.LINEAR_20_CHAMBER_OLFACTOMETER,
        paper_citation=(
            "eLife 2020 / Sci Rep, PMC7145822 (Figs. 3, 4, 5, 6 — R37G11-Gal4 > UAS-Shibire^ts1 in 50×5×1.3 mm chambers)"
        ),
        genotype="R37G11-Gal4 (6–7 cholinergic PD2a1/b1 LHONs) > UAS-Shibire^ts1",
        wetlab_protocol=(
            "Place single flies in 50×5×1.3 mm polycarbonate linear chambers with 0.15 L/min converging odor streams. "
            "Measure PD2a1/b1 contribution to odor avoidance across: (A) satiated flies at 10^-2 dilution for 80–120 s; "
            "(B) low concentration 10^-3 dilution; (C) short exposure 40 s; (D) 24-h starved flies at 10^-2 dilution."
        ),
        published_result=(
            "PD2a1/b1 drives significant odor avoidance ONLY in satiated flies at high concentration (10^-2) after "
            ">=80 s exposure; its behavioral avoidance contribution vanishes (0.0) at 10^-3 dilution, at <=40 s exposure, "
            "and in 24-h starved flies."
        ),
        default_frames=8,
        verify_suites=("gate", "mb_lh_plasticity", "wetlab_experiments", "instruments"),
    ),
    "pmc7710706_mdn_lbl40_lul130_epistasis": ScenarioSpec(
        id="pmc7710706_mdn_lbl40_lul130_epistasis",
        title="MDN Backward Walking & VNC Epistasis: LBL40 (T3 Stance Flexion) and LUL130 (Swing Leg Lift)",
        laboratory=LaboratoryPreset.TETHERED_BALL_200FPS_DEEPLABCUT,
        paper_citation=(
            "Nat Commun 2020, DOI 10.1038/s41467-020-19936-x, PMC7710706 (Figs. 1, 5, 6, 7 — 200-fps DeepLabCut)"
        ),
        genotype="MDN > CsChrimson (660 nm) + SS47938 (T3 LBL40) or SS50974 (LUL130) > GtACR2 (470 nm)",
        wetlab_protocol=(
            "Stimulate MDN with 660 nm red light on a 6-mm air-supported polypropylene ball to induce backward walking "
            "(25 s trial), then apply a 10 s pulse of 470 nm blue light to silence T3 LBL40 (14B lineage) via GtACR2, "
            "and test autonomous decapitated VNC bristle grooming reflex."
        ),
        published_result=(
            "1) MDN activation switches all 6 legs into backward coxa protraction via local VNC half-centers (LBL40 stance "
            "<-> LUL130 swing). 2) Silencing T3 LBL40 (SS47938 > GtACR2) specifically abolishes hindleg tibia flexion "
            "during backward walking. 3) Decapitated VNC executes autonomous bristle grooming, which is subsumed by walking."
        ),
        default_frames=24,
        verify_suites=("gate", "vnc_autopilots", "wetlab_experiments", "instruments"),
    ),
    "eneuro2025_gf_frazzled_shakb": ScenarioSpec(
        id="eneuro2025_gf_frazzled_shakb",
        title="Giant Fiber (GF -> TTMn) Escape Synapse: wt (9.04% shak-B), frazzled LOF (5.31%), & FraICD Rescue",
        laboratory=LaboratoryPreset.GIANT_FIBER_ELECTROPHYS_RIG,
        paper_citation=(
            "eNeuro 2025, DOI 10.1523/ENEURO.0202-25.2025 (Figs. 1, 3, 4, 5 & Table 1 — anti-shaking-B IMARIS & TTM latency)"
        ),
        genotype="Control sibling wt vs fra^3/fra^4 LOF vs w; fra^3/fra^4; R91H05-Gal4/UAS-FraICD",
        wetlab_protocol=(
            "Stimulate Giant Fiber escape pathway (via LC4 looming > 0.743 or direct GF spike) and measure TTMn "
            "depolarization and jump burst across control wt (9.04% shaking-B terminal volume), frazzled LOF (5.31% "
            "shaking-B volume), and UAS-FraICD rescue (P3 transcriptional activation domain intact)."
        ),
        published_result=(
            "1) Control wt (9.04% shak-B) triggers rapid 1-frame electrical GF->TTMn jump (1.14 ms wet-lab latency, "
            "Port4_ShakB_Electrical). 2) frazzled LOF (5.31% shak-B) suffers >45% drop in electrical TTMn depolarization "
            "and fails fast 1-ms jump threshold. 3) UAS-FraICD rescues electrical coupling above wt (0.97 ms wet-lab latency)."
        ),
        default_frames=6,
        verify_suites=("gate", "vnc_autopilots", "wetlab_experiments", "instruments"),
    ),
    "pmc13475639_vpm4_autoreceptor_aggression": ScenarioSpec(
        id="pmc13475639_vpm4_autoreceptor_aggression",
        title="VPM4 (OA+Glu) -> MBON-11 (GluClα) Layered Inhibition & OAα2R/mGluR Autoreceptor Knockdown Boxing",
        laboratory=LaboratoryPreset.SLIDING_DIVIDER_FIGHT_ARENA,
        paper_citation=(
            "Sci Adv 2026, DOI 10.1126/sciadv.aec4153, PMC13475639 (Figs. 1, 2, 3, 5, 6, 7 — Trannoy & Certel Labs)"
        ),
        genotype="MB113C-split-Gal4 (VPM4) control vs VPM4 > OAα2R-RNAi (OAα2R^KD) & VPM4 > mGluR-RNAi (mGluR^KD)",
        wetlab_protocol=(
            "Place male pairs in 22×16 mm sliding-divider arena with central 13×6 mm sucrose/yeast food cup at 25–29 °C. "
            "Activate VPM4 (OA+Glu cotransmission onto MBON-11 γ1pedc GluClα receptors) in wild-type males with intact "
            "presynaptic OAα2R/mGluR autoreceptors vs males with OAα2R/mGluR autoreceptor knockdown."
        ),
        published_result=(
            "Presynaptic OAα2R and mGluR autoreceptors act as a negative-feedback brake on VPM4 release; knocking them "
            "down (OAα2R^KD or mGluR^KD) disinhibits VPM4 -> MBON-11 signaling and escalates high-intensity boxing (>1.45x wt, ~85% fights)."
        ),
        default_frames=12,
        verify_suites=("gate", "mb_lh_plasticity", "wetlab_experiments", "instruments"),
    ),
    "pmc6451361_hunger_dawed_protein_switch": ScenarioSpec(
        id="pmc6451361_hunger_dawed_protein_switch",
        title="Peripheral Hunger Modulation (Gr5a/Gr66a/Ir76b), Central DA-WED Protein Switch, & Fdg/Crop Loop",
        laboratory=LaboratoryPreset.NUTRITIONAL_CAFE_CHAMBER,
        paper_citation=(
            "Open Biol 2019, DOI 10.1098/rsob.180259, PMC6451361 (§4.1–4.4 Figs. 1–3) & BANC v888 visceral crop loop"
        ),
        genotype="Wild-type adult Drosophila under Carbohydrate Deprivation vs Glutamine/Amino-Acid Deprivation vs Full Crop",
        wetlab_protocol=(
            "Present a two-choice food patch containing both sucrose (Gr5a) and yeast amino acids (Ir76b) across three "
            "internal states: (1) sugar-starved / protein-replete (sugar=0.15, amino=0.90); (2) sugar-replete / "
            "protein-starved (sugar=0.85, amino=0.10); (3) distended crop after feeding (crop_stretch=0.92)."
        ),
        published_result=(
            "1) Sugar starvation sensitizes Gr5a (via NPF/TH-VUM/DopEcR), dampens Gr66a (via 77.1% axo-axonic sNPF/GABA gate), "
            "and drives sucrose feeding via Fdg (PER=1.0). 2) Amino-acid starvation activates DA-WED (FB-LAL DopR2 > PLP DopR1), "
            "switching preference from sucrose to yeast. 3) Crop distension (>0.853) shuts off Fdg proboscis extension (PER=0.0)."
        ),
        default_frames=6,
        verify_suites=("gate", "wetlab_experiments", "instruments"),
    ),
    "pmc13031250_dfb_sleep_creb_consolidation": ScenarioSpec(
        id="pmc13031250_dfb_sleep_creb_consolidation",
        title="dFB (R23E10) Homeostatic Sleep States (Slow-Wave & Active Sleep) & CREB γ->α/β LTM Consolidation",
        laboratory=LaboratoryPreset.SLEEP_ARC_2P_RIG,
        paper_citation=(
            "Curr Sleep Med Rep 2026, DOI 10.1007/s40675-025-00352-w, PMC13031250 & eLife 2021 PMC9477501"
        ),
        genotype="R23E10-Gal4 (dFB) + wild-type CREB vs pan-neuronal UAS-dCREB2b repressor",
        wetlab_protocol=(
            "Condition short-term aversive memory in MB γ1 (compartment 0), then elevate astrocyte (ALG) homeostatic "
            "sleep need (0.95) to activate dFB (R23E10) neurons through Slow-Wave and Active Sleep (REM-like) cycles. "
            "Compare consolidated α/β LTM weights (compartments 9..14) in intact CREB vs UAS-dCREB2b blocker flies."
        ),
        published_result=(
            "1) High sleep need activates dFB R23E10, gating motor output (freeze_dn=1.0) and cycling through SLOW_WAVE "
            "and ACTIVE_REM sleep. 2) During ACTIVE_REM, intact CREB consolidates depressed γ1 traces into α/β LTM "
            "compartments, whereas UAS-dCREB2b blocks LTM transfer."
        ),
        default_frames=36,
        verify_suites=("gate", "wetlab_experiments", "instruments"),
    ),
}


def get_scenario(scenario_id: str) -> ScenarioSpec:
    if scenario_id not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario_id!r}. Known: {sorted(SCENARIOS)}")
    return SCENARIOS[scenario_id]
