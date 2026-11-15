"""Nervous impulse channels, biological ports, and behavioral outcomes for Fly-NCA (inspired by CASA-ADR-0080)."""

from __future__ import annotations

from enum import StrEnum


class ImpulseChannel(StrEnum):
    """Substrate-wide nervous and developmental channels across embryogenesis, morphogenesis, and 4.038 ms frames."""

    EMBRYOGENESIS_SYNCYTIUM = "embryogenesis_syncytium"
    GENE_ONTOGENY_AUTOMATON = "gene_ontogeny_automaton"
    MORPHOGENESIS_TWO_WAVE = "morphogenesis_two_wave"
    PAN_CONNECTOME_WAVE = "pan_connectome_wave"
    COGNITIVE_FRAME = "cognitive_frame"
    ESCAPE_REFLEX = "escape_reflex"
    SLEEP_CONSOLIDATION = "sleep_consolidation"


class BiologicalPort(StrEnum):
    """Universal biological synaptic, glial, and developmental ports of a Fly-NCA cell."""

    PORT1_ACH = "Port1_ACh_Excitatory"
    PORT2_GLUCLA_GABA = "Port2_GluCla_GABA_Shunt"
    PORT3_AXO_AXONIC = "Port3_AxoAxonic_25p8pct"
    PORT4_SHAKB_GAP = "Port4_ShakB_Electrical"
    PORT5_GLIA = "Port5_Glia_Metabolic"
    # Developmental & Genomic NCA Ports (L0-L1)
    PORT_MATERNAL_MRNA = "Port0_Maternal_Microtubule_mRNA"
    PORT_TRANSLATIONAL_3UTR = "Port0_3UTR_Translational_Repression"
    PORT_PIONEER_TF = "Port0_Zelda_Opa_Pioneer_Enhancer"
    PORT_SXL_TRA_SPLICING = "Port1_Sxl_Tra_Dsx_Fru_Splicing"
    PORT_CAM_DIP_DPR = "Port1_CAM_DIP_Dpr_Adhesion"


class ImpulseOutcome(StrEnum):
    """Committed developmental or efferent behavioral outcome at the end of an impulse wave."""

    # Developmental outcomes (L0-L1)
    EMBRYO_WT_8_SEGMENTS = "embryo_wt_8_segments"
    MUTANT_BICAUDAL_BCD = "mutant_bicaudal_bcd"
    MUTANT_ABDOMENLESS_NOS = "mutant_abdomenless_nos"
    MUTANT_NO_DOUBLING_OPA = "mutant_no_doubling_opa"
    ADULT_ECLOSION_READY = "adult_eclosion_ready"
    LARVA_HATCHED_READY = "larva_hatched_ready"
    # Behavioral outcomes (L2-L4)
    FORWARD_RUN = "forward_run"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    BACKWARD_MDN = "backward_mdn"
    ESCAPE_JUMP = "escape_jump"
    FEED_PER = "feed_per"
    ACTIVE_SLEEP_CREB = "active_sleep_creb"
    QUIET_SLEEP = "quiet_sleep"
    FREEZE = "freeze"
    HOVER_STABILIZE = "hover_stabilize"
    MALE_COURTSHIP_SONG = "male_courtship_song"
    FEMALE_FRUM_COURTSHIP_SONG = "female_frum_courtship_song"
    FEMALE_SONG_RECEPTIVITY = "female_song_receptivity"
    FEMALE_OVIPOSITION = "female_oviposition"
    MALE_BOXING_ESCALATION = "male_boxing_escalation"
