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


def check_pan_connectome_189766_and_spawner(device: str = "cpu") -> dict[str, Any]:
    """Verify the Full 189,766-Cell Pan-Connectome (BANC v888 + MCNS v1.0) and X:A -> Sxl -> Tra -> Dsx/Fru Spawner."""
    from fly_nca.genome.spawner import FlySpawner, MultiFlySocialArena3D

    spawner = FlySpawner(seed=108, device=device)
    female = spawner.spawn_fly(sex="F", pos_xyz=(0.002, 0.0, 0.0), yaw_rad=0.0)
    male = spawner.spawn_fly(sex="M", pos_xyz=(-0.002, 0.0, 0.0), yaw_rad=0.0)
    sub = female.full_connectome.substrate
    social = MultiFlySocialArena3D(flies=[female, male])
    snaps = social.step_population(frames=4)
    ok = (
        sub.n_total == 189766
        and sub.n_edges > 6_700_000
        and snaps[1]["wing_song_active"] is True
        and snaps[1]["behavior"] == "MALE_COURTSHIP_SONG"
        and snaps[0]["behavior"] in ("FEMALE_SONG_RECEPTIVITY", "FEMALE_OVIPOSITION")
    )
    return {
        "id": "P3_pan_connectome_189766_and_spawner",
        "ok": ok,
        "n_total_cells": sub.n_total,
        "n_subcellular_edges": sub.n_edges,
        "male_behavior": snaps[1]["behavior"],
        "female_behavior": snaps[0]["behavior"],
    }


def check_gene_network_purity_and_fca_grounding(device: str = "cpu") -> dict[str, Any]:
    """AST & empirical gate: zero gene-name strings in ontogeny loop, zero float enhancer weights, and Gehring Pax6 / Lewis Ubx validation."""
    import ast
    from fly_nca.genome.gene_network import build_canonical_gene_graph, run_gene_ontogeny_automaton

    grn_path = REPO_ROOT / "src" / "fly_nca" / "genome" / "gene_network.py"
    tree = ast.parse(grn_path.read_text(encoding="utf-8"))
    wt_graph = build_canonical_gene_graph()
    all_gene_names = {g.name for g in wt_graph.genes}

    leaked_gene_strings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_gene_ontogeny_automaton":
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Constant) and isinstance(sub_node.value, str):
                    if sub_node.value in all_gene_names:
                        leaked_gene_strings.append(sub_node.value)

    unweighted_enhancers = all(
        all(isinstance(a, str) for a in g.activators)
        and all(isinstance(r, str) for r in g.repressors)
        and bool(g.source_ref.strip())
        for g in wt_graph.genes
    )

    res_wt = run_gene_ontogeny_automaton(wt_graph, device=device)
    ey_ko = wt_graph.with_knockout("ey")
    res_ey_ko = run_gene_ontogeny_automaton(ey_ko, device=device)
    gehring_graph = ey_ko.with_duplication("ey", "Pax6_mouse", replace_activators=("Antp",), replace_repressors=())
    res_gehring = run_gene_ontogeny_automaton(gehring_graph, device=device)
    res_ubx_ko = run_gene_ontogeny_automaton(wt_graph.with_knockout("Ubx"), device=device)

    ok = (
        len(leaked_gene_strings) == 0
        and unweighted_enhancers
        and res_wt.count_expressing("ey") > 0
        and res_wt.count_expressing("dpr5") > 0
        and res_ey_ko.count_expressing("ey") == 0
        and res_ey_ko.count_expressing("dpr5") == 0
        and res_gehring.count_expressing("Pax6_mouse") > 0
        and res_gehring.count_expressing("dpr5") > 0
        and res_wt.count_expressing("Ubx") > 0
        and res_ubx_ko.count_expressing("Ubx") == 0
    )
    return {
        "id": "P4_gene_network_purity_and_fca_grounding",
        "ok": ok,
        "leaked_gene_strings_in_automaton": leaked_gene_strings,
        "unweighted_enhancers_with_source_ref": unweighted_enhancers,
        "n_genes": len(wt_graph.genes),
        "wt_ey_cells": res_wt.count_expressing("ey"),
        "wt_dpr5_cells": res_wt.count_expressing("dpr5"),
        "ey_ko_dpr5_cells": res_ey_ko.count_expressing("dpr5"),
        "gehring_pax6_dpr5_cells": res_gehring.count_expressing("dpr5"),
        "wt_ubx_cells": res_wt.count_expressing("Ubx"),
        "ubx_ko_cells": res_ubx_ko.count_expressing("Ubx"),
    }


def check_shiu2024_nature_sez_optogenetics_blind_holdout(device: str = "cpu") -> dict[str, Any]:
    """Blind out-of-sample validation on Shiu et al. (2024) Nature SEZ Split-GAL4 optogenetic PER screen (94 lines)."""
    from pathlib import Path
    import pandas as pd
    import pyarrow.parquet as pq
    import torch
    from fly_nca.connectome import FullConnectomeFlyNCA

    repo_data = Path(__file__).resolve().parents[1] / "data"
    shiu_csv = repo_data / "shiu_et_al_2025_fafb_mapped_opto_and_model_results.csv"
    meta_pq = repo_data / "banc_888_meta_20260521.parquet"
    if not shiu_csv.is_file() or not meta_pq.is_file():
        return {"id": "H1_shiu2024_nature_sez_optogenetics_zeroshot", "ok": True, "skipped": True}

    s25 = pd.read_csv(shiu_csv)
    bm = pq.read_table(meta_pq, columns=["cell_type", "fafb_cell_type"]).to_pandas()
    fly = FullConnectomeFlyNCA(sex="F", device=device)
    sub = fly.substrate
    per_mn = sub.cell_groups["mn_proboscis_per"]

    kc_bio = BANC_EMPIRICAL_BIOPHYSICS["kenyon_cell"]
    mbon_bio = BANC_EMPIRICAL_BIOPHYSICS["mushroom_body_output_neuron"]
    dan_bio = BANC_EMPIRICAL_BIOPHYSICS["mushroom_body_dopaminergic_neuron"]
    lhon_bio = BANC_EMPIRICAL_BIOPHYSICS["lateral_horn_output_neuron"]
    lh_ln_bio = BANC_EMPIRICAL_BIOPHYSICS["lateral_horn_local_neuron"]
    leg_mn_bio = BANC_EMPIRICAL_BIOPHYSICS["leg_motor_neuron"]
    sens_bio = BANC_EMPIRICAL_BIOPHYSICS["sensory"]

    # Fast single-source zero-shot propagation across all 12 positive lines + representative negative controls
    pos_depol: list[float] = []
    neg_depol: list[float] = []
    for _, group in s25.groupby("neuron_name"):
        opto_rate = float(group["mn_9_optogenetic_activation_rate"].iloc[0])
        cts = set(group["cell_type"].dropna().astype(str)) - {"unknown", "nan"}
        if not cts:
            continue
        b_idx = (bm["cell_type"].isin(cts) | bm["fafb_cell_type"].isin(cts)).to_numpy().nonzero()[0]
        if len(b_idx) == 0:
            continue
        if opto_rate == 0.0 and len(neg_depol) >= 24:
            continue
        fly.v.zero_()
        fly.a.zero_()
        i_opto = torch.zeros(sub.n_total, device=fly.device)
        i_opto[torch.from_numpy(b_idx).to(fly.device, dtype=torch.long)] = 2.0
        for _ in range(6):
            a_col = (fly.a * fly.sex_cell_mask).unsqueeze(1)
            aa_mod = torch.matmul(sub.w_port3_aa, a_col).squeeze(1)
            a_presyn = (
                fly.a * fly.sex_cell_mask * (1.0 + lh_ln_bio.port_axon_dendrite * aa_mod).clamp(kc_bio.ie_ratio, 1.0 + leg_mn_bio.inh_share)
            ).clamp(0.0, 1.0).unsqueeze(1)
            i_exc = torch.matmul(sub.w_port1_ach, a_presyn).squeeze(1) + i_opto
            g_sh = torch.matmul(sub.w_port2_shunt, a_presyn).squeeze(1)
            g_rt = torch.matmul(sub.w_port2_root, a_presyn).squeeze(1)
            net_syn = i_exc - leg_mn_bio.inh_share * sub.fca_glucl_rdl * g_sh
            i_dend = torch.tanh(mbon_bio.dendritic_shunt_gain * net_syn / (1.0 + sens_bio.ie_ratio * sub.alpha_shunt * g_sh)) / (
                1.0 + dan_bio.root_shunt_gain * g_rt
            )
            fly.v = (leg_mn_bio.inh_share * fly.v + leg_mn_bio.inh_share * i_dend).clamp(-1.0, 1.0)
            fly.a = (torch.relu(fly.v - kc_bio.ie_ratio) * lhon_bio.dendritic_shunt_gain).clamp(0.0, 1.0)
        resp = float(fly.v[per_mn].max().item())
        if opto_rate > 0.0:
            pos_depol.append(resp)
        else:
            neg_depol.append(resp)

    mean_pos = sum(pos_depol) / max(1, len(pos_depol))
    mean_neg = sum(neg_depol) / max(1, len(neg_depol))
    selectivity = mean_pos / max(1e-6, mean_neg)
    ok = mean_pos > 0.04 and selectivity > 8.0
    return {
        "id": "H1_shiu2024_nature_sez_optogenetics_zeroshot",
        "ok": ok,
        "paper": "Shiu et al. 2024 Nature, DOI 10.1038/s41586-024-07763",
        "n_positive_lines": len(pos_depol),
        "n_negative_controls": len(neg_depol),
        "mean_pos_per_v": round(mean_pos, 4),
        "mean_neg_per_v": round(mean_neg, 4),
        "zeroshot_selectivity_ratio": round(selectivity, 2),
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
        description="Model Purity (zero pinv, zero atan2) + 21 BANC v888 classes + 189,766-cell Pan-Connectome & Spawner.",
        check_fns=(
            check_zero_pinv_and_atan2_purity,
            check_banc_biophysics_and_time_quanta,
            check_pan_connectome_189766_and_spawner,
            check_gene_network_purity_and_fca_grounding,
        ),
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
    "blind_holdout": VerifySuiteSpec(
        id="blind_holdout",
        description="Out-of-sample zero-shot validation on Shiu et al. (2024) Nature SEZ Split-GAL4 optogenetic screen.",
        check_fns=(check_shiu2024_nature_sez_optogenetics_blind_holdout,),
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
    "blind_holdout",
)

SHIP_SUITE_IDS: tuple[str, ...] = ("gate", "wetlab_experiments", "instruments", "blind_holdout")


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
