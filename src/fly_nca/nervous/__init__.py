"""Nervous Impulse Trace for Fly-NCA (parity with CASA-ADR-0080)."""

from fly_nca.nervous.channels import BiologicalPort, ImpulseChannel, ImpulseOutcome
from fly_nca.nervous.impulse import (
    DebugLevel,
    ImpulseChannelWriter,
    ImpulseRecord,
    ImpulseRecorder,
    SynapticEvent,
    activate_impulse_trace,
    format_impulse_replay,
    impulse_recorder,
    read_impulse_trace,
    resolve_debug_level,
)

__all__ = [
    "BiologicalPort",
    "DebugLevel",
    "ImpulseChannel",
    "ImpulseChannelWriter",
    "ImpulseOutcome",
    "ImpulseRecord",
    "ImpulseRecorder",
    "SynapticEvent",
    "activate_impulse_trace",
    "format_impulse_replay",
    "impulse_recorder",
    "read_impulse_trace",
    "resolve_debug_level",
]
