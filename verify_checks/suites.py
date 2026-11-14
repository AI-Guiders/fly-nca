"""Verification suites for Fly-NCA — SSOT grouping mirroring `mt-lattice-fluid/verify_checks/suites.py`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

from fly_nca.app.lab import execute_scenario_protocol, open_lab
from fly_nca.app.scenario import SCENARIOS, get_scenario
from fly_nca.core.cell_state import TAU_FAST_MS, TAU_FRAME_MS
from fly_nca.genome.banc_biophysics import BANC_EMPIRICAL_BIOPHYSICS
from fly_nca.genome.morphogenesis import develop_fly_from_genome
from fly_nca.instruments.catalog import InstrumentId, REGISTRY

REPO_ROOT = Path(__file__).resolve().parents[1]


def check_zero_pinv_and_atan2_purity(device: str = "cpu") -> dict[str, Any]:
    """Verify Model Purity Contract: no torch.linalg.pinv in morphogenesis and no atan2 in CX steering."""
    _ = device
    morph_src = (REPO_ROOT / "src" / "fly_nca" / "genome" / "morphogenesis.py").read_text(encoding="utf-8")
    cx_src = (REPO_ROOT / "src" / "fly_nca" / "organs" / "central_complex.py").read_text(encoding="utf-8")
    has_pinv = "linalg.pinv" in morph_src
    # Only decode_circular_bump telemetry helper may use atan2; step_cx and copy_and_shift must not
    step_cx_block = cx_src.split("def step_cx", maxsplit=1)[1]
    has_atan2_in_step = "math.atan2(" in step_cx_block or "torch.atan2(" in step_cx_block
    ok = (not has_pinv) and (not has_atan2_in_step)
    return {
        "id": "P1_zero_pinv_and_atan2",
        "ok": ok,
        "has_pinv_in_morphogenesis": has_pinv,
        "has_atan2_in_step_cx": has_atan2_in_step,
    }


def check_banc_biophysics_and_time_quanta(device: str = "cpu") -> dict[str, Any]:
    """Verify 21 empirical BANC v888 cell classes and mt-lattice-fluid MODEL §4.8 time quanta."""
    f = develop_fly_from_genome(scale=0.05, device=device)
    tau_dan = BANC_EMPIRICAL_BIOPHYSICS["mushroom_body_dopaminergic_neuron"].tau_v_ms
    tau_cx_out = BANC_EMPIRICAL_BIOPHYSICS["central_complex_output_neuron"].tau_v_ms
    ok = (
        len(BANC_EMPIRICAL_BIOPHYSICS) >= 21
        and abs(TAU_FAST_MS - 1.711) < 1e-3
        and abs(TAU_FRAME_MS - 4.038) < 1e-3
        and abs(tau_dan - TAU_FRAME_MS) / TAU_FRAME_MS < 0.02
        and abs(tau_cx_out - TAU_FRAME_MS) / TAU_FRAME_MS < 0.03
        and abs(f.tract_weight("AL", "MB") - 0.3189) < 1e-3
    )
    return {
        "id": "P2_banc_21_classes_and_quanta",
        "ok": ok,
        "n_banc_classes": len(BANC_EMPIRICAL_BIOPHYSICS),
        "tau_fast_ms": TAU_FAST_MS,
        "tau_frame_ms": TAU_FRAME_MS,
        "tau_dan_ms": round(tau_dan, 3),
        "tau_cx_out_ms": round(tau_cx_out, 3),
    }


def _run_wetlab_scenario_check(scenario_id: str, device: str) -> dict[str, Any]:
    res = execute_scenario_protocol(scenario_id, device=device)
    meas = res["protocol_measurements"]
    ok = bool(meas.get("matches_published_paper", False))
    out: dict[str, Any] = {
        "id": scenario_id,
        "ok": ok,
        "paper": res["scenario"]["paper_citation"].split("(")[0].strip(),
    }
    for k, v in meas.items():
        if k != "matches_published_paper":
            out[k] = v
    return out


def check_pmc9213513_mb_gamma1_shibire(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("pmc9213513_mb_gamma1_shibire", device)


def check_pmc6779902_cltm_copper_grid(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("pmc6779902_cltm_copper_grid", device)


def check_pmc13518903_vt062617_wind_tunnel(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("pmc13518903_vt062617_wind_tunnel", device)


def check_pmc7145822_pd2a1_concentration_time_starvation(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("pmc7145822_pd2a1_concentration_time_starvation", device)


def check_pmc7710706_mdn_lbl40_lul130_epistasis(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("pmc7710706_mdn_lbl40_lul130_epistasis", device)


def check_eneuro2025_gf_frazzled_shakb(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("eneuro2025_gf_frazzled_shakb", device)


def check_pmc13475639_vpm4_autoreceptor_aggression(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("pmc13475639_vpm4_autoreceptor_aggression", device)


def check_pmc6451361_hunger_dawed_protein_switch(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("pmc6451361_hunger_dawed_protein_switch", device)


def check_pmc13031250_dfb_sleep_creb_consolidation(device: str = "cpu") -> dict[str, Any]:
    return _run_wetlab_scenario_check("pmc13031250_dfb_sleep_creb_consolidation", device)


def check_instrument_panel_completeness(device: str = "cpu") -> dict[str, Any]:
    """Verify that all 34 registered instruments in REGISTRY are sampled by LabSession.panel()."""
    lab = open_lab("pmc9213513_mb_gamma1_shibire", device=device)
    lab.step(2)
    panel = lab.panel()
    expected_ids = [spec.id.value for spec in REGISTRY]
    missing = [iid for iid in expected_ids if iid not in panel]
    ok = len(missing) == 0 and len(REGISTRY) == len(InstrumentId)
    return {
        "id": "I1_instrument_panel_34_channels",
        "ok": ok,
        "registered_instruments": len(REGISTRY),
        "missing_in_panel": missing,
    }


@dataclass(frozen=True, slots=True)
class VerifySuiteSpec:
    """Named bundle of verification checks — mirrors ScenarioSpec for the laboratory bench."""

    id: str
    description: str
    check_fns: tuple[Callable[..., dict[str, Any]], ...]
    ship: bool = True


VERIFY_SUITES: dict[str, VerifySuiteSpec] = {
    "gate": VerifySuiteSpec(
        id="gate",
        description="Model Purity (zero pinv, zero atan2) + 21 BANC v888 classes + mt-lattice-fluid time quanta.",
        check_fns=(check_zero_pinv_and_atan2_purity, check_banc_biophysics_and_time_quanta),
        ship=True,
    ),
    "mb_lh_plasticity": VerifySuiteSpec(
        id="mb_lh_plasticity",
        description="MB & LH-SPIN wet-lab protocols: PMC9213513 (γ1/γ5), PMC6779902 (cLTM), PMC7145822 (PD2a1), PMC13475639 (VPM4).",
        check_fns=(
            check_pmc9213513_mb_gamma1_shibire,
            check_pmc6779902_cltm_copper_grid,
            check_pmc7145822_pd2a1_concentration_time_starvation,
            check_pmc13475639_vpm4_autoreceptor_aggression,
        ),
        ship=True,
    ),
    "cx_navigation": VerifySuiteSpec(
        id="cx_navigation",
        description="CX navigation & working memory: PMC13518903 (VT062617 hΔK21 5–10s persistence & GtACR1).",
        check_fns=(check_pmc13518903_vt062617_wind_tunnel,),
        ship=True,
    ),
    "vnc_autopilots": VerifySuiteSpec(
        id="vnc_autopilots",
        description="VNC motor epistasis & Giant Fiber GJ: PMC7710706 (MDN->LBL40/LUL130) & eNeuro 2025 (Frazzled shak-B).",
        check_fns=(
            check_pmc7710706_mdn_lbl40_lul130_epistasis,
            check_eneuro2025_gf_frazzled_shakb,
        ),
        ship=True,
    ),
    "wetlab_experiments": VerifySuiteSpec(
        id="wetlab_experiments",
        description="All 9 published Drosophila wet-lab protocols (Original Paper -> Setup -> Published Result).",
        check_fns=(
            check_pmc9213513_mb_gamma1_shibire,
            check_pmc6779902_cltm_copper_grid,
            check_pmc13518903_vt062617_wind_tunnel,
            check_pmc7145822_pd2a1_concentration_time_starvation,
            check_pmc7710706_mdn_lbl40_lul130_epistasis,
            check_eneuro2025_gf_frazzled_shakb,
            check_pmc13475639_vpm4_autoreceptor_aggression,
            check_pmc6451361_hunger_dawed_protein_switch,
            check_pmc13031250_dfb_sleep_creb_consolidation,
        ),
        ship=True,
    ),
    "instruments": VerifySuiteSpec(
        id="instruments",
        description="34-channel Laboratory Instrument Catalog & sample_panel() across C_cell, G_glia, N_impulse, B_body.",
        check_fns=(check_instrument_panel_completeness,),
        ship=True,
    ),
}

SUITE_ORDER: tuple[str, ...] = (
    "gate",
    "mb_lh_plasticity",
    "cx_navigation",
    "vnc_autopilots",
    "wetlab_experiments",
    "instruments",
)

SHIP_SUITE_IDS: tuple[str, ...] = ("gate", "wetlab_experiments", "instruments")


def run_suites(suite_ids: Sequence[str], *, device: str = "cpu") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_checks: set[str] = set()
    for sid in suite_ids:
        spec = VERIFY_SUITES[sid]
        for fn in spec.check_fns:
            if fn.__name__ in seen_checks and len(suite_ids) > 1:
                continue
            seen_checks.add(fn.__name__)
            sig = inspect.signature(fn)
            row = fn(device=device) if "device" in sig.parameters else fn()
            row["suite"] = sid
            rows.append(row)
    return rows


def verify_suite_ids_for_scenarios(scenario_ids: Sequence[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    for sc_id in scenario_ids:
        sc = get_scenario(sc_id)
        for s in sc.verify_suites:
            if s not in ordered:
                ordered.append(s)
    return tuple(ordered)


__all__ = [
    "SCENARIOS",
    "SHIP_SUITE_IDS",
    "SUITE_ORDER",
    "VERIFY_SUITES",
    "VerifySuiteSpec",
    "run_suites",
    "verify_suite_ids_for_scenarios",
]
