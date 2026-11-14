"""Laboratory Instrument Catalog — SSOT of Cellular, Glial, Nervous-Impulse, and 3D Body observables.

Mirrors the `mt-lattice-fluid/mt_ca/instruments/catalog.py` architecture so every instrument has:
- a stable `InstrumentId` for JSON / CLI / verification suites,
- a layer classification (`C_cell`, `G_glia`, `N_impulse`, `B_body`),
- a direct reference to the published paper (`PMC` / `DOI` / `BANC v888` table) where the quantity is measured,
- natural and SI physical units.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

InstrumentLayer = Literal["C_cell", "G_glia", "N_impulse", "B_body"]


class InstrumentId(str, Enum):
    """Stable instrument identifiers for laboratory panels, JSON output, and verify suites."""

    # --- C_cell: Cellular & Synaptic Layer ---
    KC_SPARSITY = "kc_sparsity"
    MBON_G1_RATE = "mbon_g1_rate"
    MBON_G5_RATE = "mbon_g5_rate"
    MB_NET_VALENCE = "mb_net_valence"
    MB_ALPHA_LTM_W = "mb_alpha_ltm_w"
    VPM4_BOXING_DRIVE = "vpm4_boxing_drive"
    CLTM_GATE_OPEN = "cltm_gate_open"
    CLTM_VALENCE = "cltm_valence"
    PD2A1_PRIOR = "pd2a1_prior"
    EB_BUMP_PEAK = "eb_bump_peak"
    EB_HEADING_RAD = "eb_heading_rad"
    FB_GOAL_PERSIST = "fb_goal_persist"
    CX_NAV_MODE = "cx_nav_mode"
    DFB_SLEEP_PHASE = "dfb_sleep_phase"
    DN_DNP09_FWD = "dn_dnp09_fwd"
    DN_MDN_BACK = "dn_mdn_back"
    DN_DNA02_STEER = "dn_dna02_steer"
    DN_DNG02_FLIGHT = "dn_dng02_flight"
    GF_TTMN_V = "gf_ttmn_v"

    # --- G_glia: Glial & Hemolymph Milieu Layer ---
    HEMO_SUGAR = "hemo_sugar"
    HEMO_AMINO = "hemo_amino"
    ALG_SLEEP_NEED = "alg_sleep_need"

    # --- N_impulse: Nervous Impulse Wave Layer ---
    IMPULSE_OUTCOME = "impulse_outcome"
    IMPULSE_EVENT_COUNT = "impulse_event_count"
    PORT4_SHAKB_ACTIVE = "port4_shakb_active"

    # --- B_body: 3D Kinematic & Behavioral Effector Layer ---
    POS_X_M = "pos_x_m"
    POS_Y_M = "pos_y_m"
    POS_Z_M = "pos_z_m"
    WING_STROKE_AMP = "wing_stroke_amp"
    TARSAL_LOAD_MEAN = "tarsal_load_mean"
    TRIPOD_COHERENCE = "tripod_coherence"
    SWEET_US = "sweet_us"
    BITTER_US = "bitter_us"
    YEAST_DAWED_DRIVE = "yeast_dawed_drive"
    PROBOSCIS_PER = "proboscis_per"
    JUMP_TTMN_BURST = "jump_ttmn_burst"


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    """Metadata specification of a single laboratory instrument."""

    id: InstrumentId
    title: str
    paper_ref: str
    layer: InstrumentLayer
    unit: str
    note: str


REGISTRY: tuple[InstrumentSpec, ...] = (
    InstrumentSpec(
        InstrumentId.KC_SPARSITY,
        "Kenyon cell active fraction",
        "BANC v888 kenyon_cell ie_ratio=0.046 · Learn Mem 2024 PMC11199948 (APL GABA feedback)",
        "C_cell",
        "fraction [0..1]",
        "~4.6% sparse pattern separation in Mushroom Body Calyx",
    ),
    InstrumentSpec(
        InstrumentId.MBON_G1_RATE,
        "MBON-γ1pedc>α/β firing rate",
        "Sci Rep 2022 PMC9213513 Fig. 2 (homer-GCaMP postsynaptic Ca2+ in γ1)",
        "C_cell",
        "normalized rate [0..1]",
        "Single site of associative aversive STM LTD; feeds forward inhibition onto γ5",
    ),
    InstrumentSpec(
        InstrumentId.MBON_G5_RATE,
        "MBON-γ5 avoidance firing rate",
        "Sci Rep 2022 PMC9213513 Fig. 4 (γ1 -| γ5 feedforward disinhibition)",
        "C_cell",
        "normalized rate [0..1]",
        "Disinhibited when MBON-γ1pedc is depressed by PPL1-γ1pedc LTD",
    ),
    InstrumentSpec(
        InstrumentId.MB_NET_VALENCE,
        "Net learned MBON valence",
        "Sci Rep 2022 PMC9213513 · Front Syst Neurosci 2026 PMC13190573",
        "C_cell",
        "valence [-1..+1]",
        "0.000 in naive fly; negative after aversive conditioning, positive after appetitive",
    ),
    InstrumentSpec(
        InstrumentId.MB_ALPHA_LTM_W,
        "Consolidated α/β LTM synaptic weight",
        "Nat Commun 2019 PMC6779902 · Curr Sleep Med Rep 2026 PMC13031250",
        "C_cell",
        "synaptic weight [0..1]",
        "Mean KC->MBON weight in consolidated α/β compartments (9..14)",
    ),
    InstrumentSpec(
        InstrumentId.VPM4_BOXING_DRIVE,
        "VPM4 -> MBON-11 aggressive boxing escalation",
        "Sci Adv 2026 PMC13475639 Fig. 1–7 (VPM4 OA+Glu -> MBON-11 GluCla & OAa2R/mGluR brake)",
        "C_cell",
        "drive [0..1]",
        "Escalates when VPM4 is active and presynaptic OAa2R/mGluR autoreceptors are knocked down",
    ),
    InstrumentSpec(
        InstrumentId.CLTM_GATE_OPEN,
        "LH-SPIN 4-branch GluClα context gate",
        "Nat Commun 2019 PMC6779902 Fig. 1c, 3–5 · Front Syst Neurosci 2026 PMC13190573",
        "C_cell",
        "gate [0..1]",
        "Requires all 4 context modalities (odor + visual + temperature + AMMC grid texture) to match",
    ),
    InstrumentSpec(
        InstrumentId.CLTM_VALENCE,
        "Retrieved context-dependent LTM (cLTM) valence",
        "Nat Commun 2019 PMC6779902 Fig. 1–2 (protein-synthesis & MB independent, dDA1 dependent)",
        "C_cell",
        "valence [-1..+1]",
        "Drops to exact 0.0 when any single context modality mismatches training",
    ),
    InstrumentSpec(
        InstrumentId.PD2A1_PRIOR,
        "LH PD2a1/b1 naive avoidance soft prior",
        "eLife 2020 PMC7145822 Fig. 3–6 (R37G11-Gal4 concentration, 80s time, & starvation gating)",
        "C_cell",
        "valence [-0.66..0]",
        "Active at 10^-2 & >=80s in fed flies; 0.0 at 10^-3, <=40s, or in starved flies",
    ),
    InstrumentSpec(
        InstrumentId.EB_BUMP_PEAK,
        "EB EPG 16-wedge compass bump peak",
        "eLife 2021 PMC9477501 Fig. 16 (EPG <-> PEN ring attractor)",
        "C_cell",
        "rate [0..1]",
        "Maintains single localized bump in light and total darkness",
    ),
    InstrumentSpec(
        InstrumentId.EB_HEADING_RAD,
        "Decoded EPG head direction phase",
        "eLife 2021 PMC9477501 · eLife 2021 PMC8741217",
        "C_cell",
        "rad [-π..+π]",
        "Circular mean of the 16 EPG compass wedges",
    ),
    InstrumentSpec(
        InstrumentId.FB_GOAL_PERSIST,
        "FB VT062617 (hΔK21) persistent goal bump",
        "Nat Commun 2026 PMC13518903 Fig. 1, 3, 5 (ramping & 5–10s persistence after odor OFF)",
        "C_cell",
        "current [0..1]",
        "Persists with tau=7500 ms after plume loss; abolished by GtACR1",
    ),
    InstrumentSpec(
        InstrumentId.CX_NAV_MODE,
        "Central Complex 3-state navigation controller",
        "Nat Commun 2026 PMC13518903 Fig. 6 (BASELINE / GOAL / SEARCH state transitions)",
        "C_cell",
        "state enum",
        "Emerges from FB hΔK21 goal bump amplitude",
    ),
    InstrumentSpec(
        InstrumentId.DFB_SLEEP_PHASE,
        "dFB (R23E10) & ER5 sleep state",
        "Curr Sleep Med Rep 2026 PMC13031250 · eLife 2021 PMC9477501",
        "C_cell",
        "state enum",
        "WAKE, SLOW_WAVE (<4 Hz R5), or ACTIVE_REM (synaptic CREB consolidation)",
    ),
    InstrumentSpec(
        InstrumentId.DN_DNP09_FWD,
        "DNp09 forward walking descending neuron (cell 66)",
        "BANC v888 PMC13518251 Fig. 3 · Nat Commun 2026 PMC13518903",
        "C_cell",
        "rate [0..1]",
        "Excited by FB goal on ground; shunted by tarsal unloading (flight) or aversion",
    ),
    InstrumentSpec(
        InstrumentId.DN_MDN_BACK,
        "MDN Moonwalker backward walking neuron (cell 67)",
        "Nat Commun 2020 PMC7710706 Fig. 1–7 (MDN -> LBL40/LUL130)",
        "C_cell",
        "rate [0..1]",
        "Recruits T3 LBL40 tibia flexion and T1–T3 LUL130 leg lifting",
    ),
    InstrumentSpec(
        InstrumentId.DN_DNA02_STEER,
        "DNa02 L-R descending steering differential (cells 68..69)",
        "eLife 2021 PMC9477501 · eLife 2021 PMC8741217 (PFL3 ±45° PB shift -> DNa02)",
        "C_cell",
        "diff [-1..+1]",
        "Pure topological PFL3_Left - PFL3_Right readout without atan2",
    ),
    InstrumentSpec(
        InstrumentId.DN_DNG02_FLIGHT,
        "DNg02 flight power descending neuron (cell 70)",
        "BANC v888 PMC13518251 · campaniform tarsal unloading reflex",
        "C_cell",
        "rate [0..1]",
        "Gated open by loss of tarsal ground load or climb demand",
    ),
    InstrumentSpec(
        InstrumentId.GF_TTMN_V,
        "TTMn jump motor neuron membrane potential (cell 39)",
        "eNeuro 2025 DOI 10.1523/ENEURO.0202-25.2025 Fig. 4–5 (GF->TTMn shak-B GJ)",
        "C_cell",
        "potential [-1..+1]",
        "Reflects fast electrical gap junction current from Giant Fiber (9.04% wt vs 5.31% fra LOF)",
    ),
    InstrumentSpec(
        InstrumentId.HEMO_SUGAR,
        "Hemolymph trehalose/glucose satiety level",
        "Open Biol 2019 PMC6451361 Fig. 1 (DILPs / AKH / NPF via SPG BBB)",
        "G_glia",
        "level [0..1]",
        "Low sugar elevates NPF & AKH across SPG blood-brain barrier",
    ),
    InstrumentSpec(
        InstrumentId.HEMO_AMINO,
        "Hemolymph amino-acid / glutamine satiety level",
        "Open Biol 2019 PMC6451361 §4.4 Fig. 2b (DA-WED protein hunger sensor)",
        "G_glia",
        "level [0..1]",
        "Low amino acids activate DA-WED plastic switch toward yeast",
    ),
    InstrumentSpec(
        InstrumentId.ALG_SLEEP_NEED,
        "Astrocyte-like glia (ALG) homeostatic sleep pressure",
        "Curr Sleep Med Rep 2026 PMC13031250 · Glia 2025",
        "G_glia",
        "pressure [0..1]",
        "Drives dFB R23E10 depolarization against PPL1 wake dopamine",
    ),
    InstrumentSpec(
        InstrumentId.IMPULSE_OUTCOME,
        "Cognitive frame nervous impulse outcome",
        "CASA-ADR-0080 · BANC v888 6-organ topological wave",
        "N_impulse",
        "enum",
        "High-level behavioral verdict of the 4.038 ms cognitive frame",
    ),
    InstrumentSpec(
        InstrumentId.IMPULSE_EVENT_COUNT,
        "Synaptic decision events in frame",
        "CASA-ADR-0080 (activate/suppress/score/threshold/select/route)",
        "N_impulse",
        "count",
        "Number of recorded synaptic gate events across OL->AL->MB->LH->CX->VNC",
    ),
    InstrumentSpec(
        InstrumentId.PORT4_SHAKB_ACTIVE,
        "Port 4 (shak-B electrical gap junction) escape routing flag",
        "eNeuro 2025 DOI 10.1523/ENEURO.0202-25.2025",
        "N_impulse",
        "bool",
        "True when GF->TTMn electrical escape route fires during the frame",
    ),
    InstrumentSpec(
        InstrumentId.POS_X_M,
        "Arena X coordinate",
        "FastTensorArena3D SI kinematics",
        "B_body",
        "m",
        "World-frame X position in meters",
    ),
    InstrumentSpec(
        InstrumentId.POS_Y_M,
        "Arena Y coordinate",
        "FastTensorArena3D SI kinematics",
        "B_body",
        "m",
        "World-frame Y position in meters",
    ),
    InstrumentSpec(
        InstrumentId.POS_Z_M,
        "Arena Z altitude",
        "FastTensorArena3D SI kinematics",
        "B_body",
        "m",
        "World-frame Z altitude above ground in meters",
    ),
    InstrumentSpec(
        InstrumentId.WING_STROKE_AMP,
        "200-Hz mean wing stroke amplitude",
        "BANC v888 wing_motor_neuron (64 MNs) · VNC flight ESP",
        "B_body",
        "amplitude [0..1]",
        "Mean Left/Right wing drive from VNC cells 6..7",
    ),
    InstrumentSpec(
        InstrumentId.TARSAL_LOAD_MEAN,
        "Campaniform sensillum mean leg ground load",
        "BANC v888 campaniform_sensillum_neuron (576 cells)",
        "B_body",
        "load [0..1]",
        ">0 when tarsi bear weight on substrate; 0.0 when airborne",
    ),
    InstrumentSpec(
        InstrumentId.TRIPOD_COHERENCE,
        "6-leg Tripod A vs Tripod B anti-phase coherence",
        "Nat Commun 2020 PMC7710706 (LBL40 stance <-> LUL130 swing)",
        "B_body",
        "coherence [0..1]",
        "Separation between Tripod A (L1,R2,L3) and Tripod B (R1,L2,R3)",
    ),
    InstrumentSpec(
        InstrumentId.SWEET_US,
        "Hunger-modulated Gr5a sweet taste signal",
        "Open Biol 2019 PMC6451361 §4.1 Fig. 2a (Gr5a + DopEcR/NPF)",
        "B_body",
        "signal [0..1]",
        "Sensitized by starvation; suppressed during protein hunger via DA-WED",
    ),
    InstrumentSpec(
        InstrumentId.BITTER_US,
        "Hunger-modulated Gr66a bitter taste signal",
        "Open Biol 2019 PMC6451361 §4.2 Fig. 2a (Gr66a + sNPF/GABA/OA-VL)",
        "B_body",
        "signal [0..1]",
        "Dampened by 77.1% axo-axonic presynaptic gating under starvation",
    ),
    InstrumentSpec(
        InstrumentId.YEAST_DAWED_DRIVE,
        "DA-WED + Ir76b yeast/protein seeking drive",
        "Open Biol 2019 PMC6451361 §4.4 Fig. 2b (DA-WED FB-LAL DopR2 vs PLP DopR1)",
        "B_body",
        "drive [0..1]",
        "Outcompetes sugar drive when hemolymph amino acids are depleted",
    ),
    InstrumentSpec(
        InstrumentId.PROBOSCIS_PER,
        "Fdg-driven Proboscis Extension Reflex (PER)",
        "Open Biol 2019 PMC6451361 §4.3 Fig. 2a (Fdg command neuron) · BANC crop stop loop",
        "B_body",
        "{0, 1}",
        "1.0 when hungry and tasting food; 0.0 when satiated or crop stretch > 0.853",
    ),
    InstrumentSpec(
        InstrumentId.JUMP_TTMN_BURST,
        "TTMn tergotrochanteral escape jump burst",
        "eNeuro 2025 DOI 10.1523/ENEURO.0202-25.2025",
        "B_body",
        "burst [0..1]",
        "Triggers immediate vertical takeoff into 3D flight when > 0.5",
    ),
)
