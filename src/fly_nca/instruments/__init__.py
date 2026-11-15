"""Laboratory instruments and multi-layer telemetry panel for Fly-NCA."""

from fly_nca.instruments.catalog import InstrumentId, InstrumentLayer, InstrumentSpec, REGISTRY
from fly_nca.instruments.gene_scanner import (
    CANONICAL_MARKER_DOMAINS,
    MarkerDomainSpec,
    TranscriptomeAtlasScanner,
    TranscriptomeScanReport,
)
from fly_nca.instruments.panel import PANEL_SCHEMA, sample_panel

__all__ = [
    "CANONICAL_MARKER_DOMAINS",
    "InstrumentId",
    "InstrumentLayer",
    "InstrumentSpec",
    "MarkerDomainSpec",
    "PANEL_SCHEMA",
    "REGISTRY",
    "TranscriptomeAtlasScanner",
    "TranscriptomeScanReport",
    "sample_panel",
]
