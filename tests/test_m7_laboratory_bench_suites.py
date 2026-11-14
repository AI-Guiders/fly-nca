"""Milestone M7 Verification: Configurable Laboratory Bench, 36-Channel Instrument Panel, & 9 Wet-Lab Protocols."""

from __future__ import annotations

import pyarrow.feather  # noqa: F401 - import before torch on Windows

from fly_nca.app import LABORATORY_RIGS, SCENARIOS, execute_scenario_protocol, open_lab
from fly_nca.app.cli import main as app_cli_main
from fly_nca.instruments import InstrumentId, REGISTRY
from verify_checks import SHIP_SUITE_IDS, run_suites
from verify_principles import main as verify_main


def test_laboratory_rigs_and_scenarios_are_self_contained() -> None:
    """Every scenario and rig must carry full paper citation, genotype, wet-lab protocol, and published result."""
    assert len(LABORATORY_RIGS) == 9
    assert len(SCENARIOS) == 9
    for sid, spec in SCENARIOS.items():
        assert spec.id == sid
        assert len(spec.paper_citation) > 20
        assert len(spec.genotype) > 5
        assert len(spec.wetlab_protocol) > 40
        assert len(spec.published_result) > 40
        assert spec.rig.preset == spec.laboratory


def test_instrument_panel_samples_all_36_registered_channels() -> None:
    """LabSession.panel() samples all 36 registered instruments across C_cell, G_glia, N_impulse, and B_body."""
    assert len(REGISTRY) == len(InstrumentId) == 36
    lab = open_lab("pmc13518903_vt062617_wind_tunnel")
    lab.step(2)
    panel = lab.panel()
    for spec in REGISTRY:
        assert spec.id.value in panel


def test_all_9_published_wetlab_protocols_match_empirical_results() -> None:
    """Run all ship verification suites (gate + 9 wet-lab protocols + instruments) and verify 100% PASS."""
    rows = run_suites(SHIP_SUITE_IDS, device="cpu")
    assert len(rows) == 12
    failed = [r for r in rows if not r["ok"]]
    assert failed == []


def test_cli_entrypoints_list_run_panel_and_verify(capsys) -> None:
    """Verify `python -m fly_nca.app list|run|panel` and `verify_principles.py --list` CLI commands."""
    assert app_cli_main(["list"]) == 0
    assert app_cli_main(["panel", "pmc9213513_mb_gamma1_shibire", "--steps", "2", "--json"]) == 0
    assert verify_main(["--list"]) == 0
    res = execute_scenario_protocol("eneuro2025_gf_frazzled_shakb")
    assert res["protocol_measurements"]["matches_published_paper"] is True
    _ = capsys.readouterr()
