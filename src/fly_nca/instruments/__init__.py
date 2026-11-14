"""Laboratory instruments and multi-layer telemetry panel for Fly-NCA."""

from fly_nca.instruments.catalog import InstrumentId, InstrumentLayer, InstrumentSpec, REGISTRY
from fly_nca.instruments.panel import PANEL_SCHEMA, sample_panel

__all__ = [
    "InstrumentId",
    "InstrumentLayer",
    "InstrumentSpec",
    "PANEL_SCHEMA",
    "REGISTRY",
    "sample_panel",
]
