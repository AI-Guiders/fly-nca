"""Fly-NCA Configurable Laboratory Bench (`laboratory`, `scenario`, `lab`, `cli`)."""

from __future__ import annotations

from typing import Any

from fly_nca.app.laboratory import LABORATORY_RIGS, LaboratoryPreset, LaboratoryRigSpec
from fly_nca.app.scenario import SCENARIOS, ScenarioSpec, get_scenario

__all__ = [
    "LABORATORY_RIGS",
    "LabSession",
    "LaboratoryPreset",
    "LaboratoryRigSpec",
    "SCENARIOS",
    "ScenarioSpec",
    "execute_scenario_protocol",
    "get_scenario",
    "open_lab",
]


def __getattr__(name: str) -> Any:
    if name in ("LabSession", "execute_scenario_protocol", "open_lab"):
        from fly_nca.app import lab

        return getattr(lab, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
