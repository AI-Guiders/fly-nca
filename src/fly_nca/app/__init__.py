"""Fly-NCA Configurable Laboratory Bench (`laboratory`, `scenario`, `lab`, `cli`)."""

from fly_nca.app.lab import LabSession, execute_scenario_protocol, open_lab
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
