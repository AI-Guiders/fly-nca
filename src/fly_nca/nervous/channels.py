"""Nervous impulse channels, biological ports, and behavioral outcomes for Fly-NCA (inspired by CASA-ADR-0080)."""

from __future__ import annotations

from enum import StrEnum


class ImpulseChannel(StrEnum):
    """Substrate-wide nervous channels across a 4.038 ms cognitive frame."""

    COGNITIVE_FRAME = "cognitive_frame"
    ESCAPE_REFLEX = "escape_reflex"
    SLEEP_CONSOLIDATION = "sleep_consolidation"


class BiologicalPort(StrEnum):
    """The 5 universal biological synaptic/glial ports of a Fly-NCA cell."""

    PORT1_ACH = "Port1_ACh_Excitatory"
    PORT2_GLUCLA_GABA = "Port2_GluCla_GABA_Shunt"
    PORT3_AXO_AXONIC = "Port3_AxoAxonic_25p8pct"
    PORT4_SHAKB_GAP = "Port4_ShakB_Electrical"
    PORT5_GLIA = "Port5_Glia_Metabolic"


class ImpulseOutcome(StrEnum):
    """Committed efferent behavioral outcome at the end of a nervous impulse wave."""

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
