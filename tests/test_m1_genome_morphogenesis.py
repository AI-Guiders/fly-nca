"""Milestone M1 Verification: 5-Port NCA Core, Glia, FlyGenome, and Two-Wave Morphogenesis."""

from __future__ import annotations

from pathlib import Path
import pyarrow.feather  # noqa: F401 - import before torch on Windows
import pytest
import torch

from fly_nca.core import (
    CH_A,
    CH_ATP,
    CH_CA,
    CH_PERSIST,
    CH_SLEEP,
    CH_V,
    MOD_AKH,
    MOD_DA_NEG,
    MOD_DA_POS,
    MOD_DILP,
    MOD_NPF,
    TAU_FAST_MS,
    TAU_FRAME_MS,
    CellPhysiologyParams,
    GlialSubstrateNCA,
    HemolymphState,
    UniversalFlyCellUpdater,
    init_cell_state,
    port1_2_dendritic_integration,
    port3_axo_axonic_presynaptic_gate,
    port4_electrical_gap_junction,
)
from fly_nca.genome import (
    SEGMENTS_8,
    FlyGenome,
    assemble_cam_synapses,
    develop_fly_from_genome,
    verify_against_banc_meta,
)
from fly_nca.genome.banc_biophysics import (
    BANC_EMPIRICAL_BIOPHYSICS,
    BANC_ORGAN_INPUT_MATRIX,
)

BANC_META_PATH = Path(r"D:\Experiments\casa-data\connectomes\banc-v888\banc_888_meta.feather")


def test_physical_time_quanta_constants() -> None:
    """Verify Drosophila frame quanta from mt-lattice-fluid MODEL §4.8 and BANC morphometric tau_V bridge."""
    assert abs(TAU_FAST_MS - 1.711) < 1e-3
    assert abs(TAU_FRAME_MS - 4.038) < 1e-3

    # Parameter-free cable-length scaling from sensory (241.53 um -> 1.711 ms) to MB DAN (565.86 um) & CX output (584.59 um)
    # hits TAU_FRAME_MS (4.038 ms) within 2%!
    tau_dan = BANC_EMPIRICAL_BIOPHYSICS["mushroom_body_dopaminergic_neuron"].tau_v_ms
    tau_cx_out = BANC_EMPIRICAL_BIOPHYSICS["central_complex_output_neuron"].tau_v_ms
    assert abs(tau_dan - TAU_FRAME_MS) / TAU_FRAME_MS < 0.02
    assert abs(tau_cx_out - TAU_FRAME_MS) / TAU_FRAME_MS < 0.03


def test_port1_2_glucla_shunt_and_tanh_conjunction_gate() -> None:
    """Verify dendritic GluCla shunting, tanh saturation, and strict AND-gate (PI=0 on missing modality)."""
    n_cells = 2
    params = CellPhysiologyParams.default(n_cells)
    # Cell 0: pure conjunction AND-gate (mu_conj = 1.0, e.g. cLTM LHON / SPIN SMP353)
    # Cell 1: linear dendritic summer (mu_conj = 0.0)
    params.mu_conj[0] = 1.0
    params.mu_conj[1] = 0.0

    # Case A: 4 modalities (odor, light, temp, mech/pressure), all 4 present
    exc_all = torch.tensor([[1.5, 1.5, 1.5, 1.5], [1.5, 1.5, 1.5, 1.5]])
    shunt_zero = torch.zeros_like(exc_all)
    out_all = port1_2_dendritic_integration(exc_all, shunt_zero, params)

    # tanh saturation keeps 4-modality product near 1.0 without explosion
    assert 0.85 < float(out_all[0]) <= 1.0

    # Case B: 1 modality missing (e.g., temperature mismatch -> branch 2 is 0.0)
    exc_missing_one = torch.tensor([[1.5, 1.5, 0.0, 1.5], [1.5, 1.5, 0.0, 1.5]])
    out_missing = port1_2_dendritic_integration(exc_missing_one, shunt_zero, params)

    # Conjunction cell MUST output exact 0.0 (Nat Commun 2019: any context mismatch -> PI=0!)
    assert float(out_missing[0]) == 0.0
    assert float(out_missing[1]) > 4.0

    # Case C: Root aversive shunt (aversion > attraction) suppresses attractive output
    root_aversive = torch.tensor([5.0, 5.0])
    out_shunted = port1_2_dendritic_integration(exc_all, shunt_zero, params, root_shunt=root_aversive)
    assert float(out_shunted[0]) < float(out_all[0]) * 0.10


def test_port3_axo_axonic_compartment_independence_and_autoreceptor() -> None:
    """Verify Port 3 (a-a, 25.8% of synapses): 1 neuron emits distinct outputs across 15 compartments."""
    soma_a = torch.tensor([[0.8]])        # 1 Kenyon cell, soma active
    auto_r = torch.tensor([[0.0]])        # No autoreceptor fatigue yet
    kappa = torch.tensor([[0.5]])

    # 15 compartments along the KC axon: compartment 0 (gamma1) has strong a-a inhibition, compartment 1 has facilitation
    aa_exc = torch.zeros((1, 15))
    aa_inh = torch.zeros((1, 15))
    aa_inh[0, 0] = 9.0
    aa_exc[0, 1] = 0.5

    out_comp = port3_axo_axonic_presynaptic_gate(soma_a, auto_r, kappa, aa_exc=aa_exc, aa_inh=aa_inh)
    assert out_comp.shape == (1, 15)
    assert float(out_comp[0, 0]) < 0.10   # Shunted in compartment gamma1
    assert float(out_comp[0, 1]) > 0.90   # Facilitated in compartment gamma2
    assert abs(float(out_comp[0, 2]) - 0.80) < 1e-5  # Unchanged in compartment gamma3

    # Autoreceptor brake (OAa2R / mGluR knockdown vs intact, Sci Adv 2026)
    auto_high = torch.tensor([[1.0]])
    out_braked = port3_axo_axonic_presynaptic_gate(soma_a, auto_high, torch.tensor([[2.0]]))
    assert float(out_braked.item()) < 0.30


def test_port4_gap_junction_and_glia_eg_capsule_isolation() -> None:
    """Verify Port 4 fast electrical transfer and Port 5 EG glial compartment non-leakage."""
    v = torch.tensor([0.9, 0.0])  # Cell 0 = GF, Cell 1 = TTMn jump motor neuron
    src = torch.tensor([0])
    dst = torch.tensor([1])
    g_gap = torch.tensor([0.8])
    i_gap = port4_electrical_gap_junction(v, src, dst, g_gap, rectifying=False)
    assert float(i_gap[1]) > 0.70
    assert abs(float(i_gap[0] + i_gap[1])) < 1e-6  # Conservation of symmetric electrical current

    # Glia EG capsule isolation across 15 MB compartments
    glia = GlialSubstrateNCA(n_compartments=15)
    glia.inject_compartment_modulator(comp_idx=0, mod_channel=MOD_DA_NEG, amount=0.9)
    assert float(glia.compartment_mods[0, MOD_DA_NEG]) == pytest.approx(0.9)
    assert float(glia.compartment_mods[1:, MOD_DA_NEG].sum()) == 0.0  # Zero leak to compartments 1..14!

    # SPG BBB translates low hemolymph sugar into high NPF & AKH hunger peptides
    hungry_hemo = HemolymphState(sugar_level=0.15)
    cell_sleep = torch.full((10,), 0.4)
    cell_comp = torch.zeros(10, dtype=torch.long)
    glia.step_glia(hungry_hemo, cell_sleep, cell_comp, is_sleeping=False, dt_ms=10.0)
    assert float(glia.compartment_mods[0, MOD_NPF]) == pytest.approx(0.85)
    assert float(glia.compartment_mods[0, MOD_AKH]) > 0.50
    assert float(glia.compartment_mods[0, MOD_DILP]) == pytest.approx(0.15)


def test_two_wave_morphogenesis_and_in_silico_genetics() -> None:
    """Verify Wave 1 (Larva ~3k) -> Wave 2 (Adult), Bates NT law, Broad shift in 14A, and fru/dsx switch."""
    genome = FlyGenome()
    segs = genome.run_french_flag_segmentation_nca()
    assert int(segs.min()) == 0 and int(segs.max()) == len(SEGMENTS_8) - 1

    # 1. Wave 1 Larval Core (~3,000-3,600 cells) vs Wave 2 Adult expansion
    larva = develop_fly_from_genome(genome, stage="larva", sex="F", scale=1.0)
    adult_f = develop_fly_from_genome(genome, stage="adult", sex="F", scale=0.1)
    assert 2500 <= larva.n_cells <= 4000
    assert larva.n_wave2_secondary == 0
    assert adult_f.n_wave2_secondary > adult_f.n_wave1_primary * 2

    # 2. Neurotransmitter fractions match BANC v888 hierarchy (ACh dominant ~50%, Glu & GABA inhibitory)
    nts = adult_f.nt_fractions()
    assert 0.45 <= nts["acetylcholine"] <= 0.62
    assert 0.15 <= nts["glutamate"] <= 0.30
    assert 0.12 <= nts["gaba"] <= 0.25

    # 3. Nature 2026 Ectopic Broad (br) shift in hemilineage 14A:
    # Exact same neuron count in 14A, but shifted axonal/dendritic CAM barcodes!
    adult_br = develop_fly_from_genome(genome, stage="adult", sex="F", scale=0.1, ectopic_broad_14a=True)
    mask_14a = adult_f.lineage_id == 30
    assert int(mask_14a.sum()) == int((adult_br.lineage_id == 30).sum())  # Cell count unchanged!
    cam_diff_14a = (adult_f.surface.k_axon[mask_14a] - adult_br.surface.k_axon[mask_14a]).abs().max()
    cam_diff_other = (adult_f.surface.k_axon[~mask_14a] - adult_br.surface.k_axon[~mask_14a]).abs().max()
    assert float(cam_diff_14a) > 0.20
    assert float(cam_diff_other) == 0.0

    # 4. Sex switch (fru/dsx): ~98% isomorphic scaffold untouched, dimorphic aSP/P1 dendrite inverted!
    adult_m = develop_fly_from_genome(genome, stage="adult", sex="M", scale=0.1)
    iso_mask = ~adult_f.sex_dimorphic
    dim_mask = adult_f.sex_dimorphic
    assert float((adult_f.surface.q_dend[iso_mask] - adult_m.surface.q_dend[iso_mask]).abs().max()) == 0.0
    cos_dim = torch.nn.functional.cosine_similarity(
        adult_f.surface.q_dend[dim_mask], adult_m.surface.q_dend[dim_mask], dim=-1
    )
    assert float(cos_dim.mean()) < 0.20  # Dimorphic dendritic locks rotated/inverted!

    # 5. eNeuro 2025 Frazzled LOF vs ICD rescue on gap junctions
    adult_fra_lof = develop_fly_from_genome(genome, stage="adult", sex="F", scale=0.1, frazzled_state="lof")
    adult_fra_icd = develop_fly_from_genome(genome, stage="adult", sex="F", scale=0.1, frazzled_state="icd_rescue")
    ratio_lof = float(adult_fra_lof.surface.z_gap.norm() / adult_f.surface.z_gap.norm())
    assert abs(ratio_lof - (5.31 / 9.04)) < 1e-5
    # 6. CAM Lock-and-Key inter-organ tract matrix reproduces empirical BANC v888 9x9 organ input fractions
    target_mat = torch.tensor(BANC_ORGAN_INPUT_MATRIX, dtype=torch.float32)
    target_norm = target_mat / target_mat.sum(dim=0, keepdim=True)
    assert float((adult_f.organ_tract_matrix - target_norm).abs().max().item()) < 1e-5
    assert abs(adult_f.tract_weight("OL", "OL") - 0.9648) < 1e-3
    assert abs(adult_f.tract_weight("AL", "MB") - 0.3189) < 1e-3
    assert abs(adult_f.tract_weight("LH_SPIN", "CX") - 0.1192) < 1e-3
    assert abs(adult_f.tract_weight("VNC", "VNC") - 0.8036) < 1e-3
    assert abs(adult_f.tract_weight("AN_DN", "VNC") - 0.1848) < 1e-3


@pytest.mark.skipif(not BANC_META_PATH.is_file(), reason="BANC v888 feather not present on disk")
def test_banc_v888_empirical_ground_truth() -> None:
    """Verify against real local BANC v888 connectome metadata (188,508 cells, 259 hemilineages)."""
    stats = verify_against_banc_meta(BANC_META_PATH)
    assert stats["n_total"] == 188508
    assert stats["n_hemilineages"] == 259
    assert stats["n_cell_types"] == 11504
    assert 0.50 <= stats["ach_fraction"] <= 0.60
    assert stats["secondary_hemilineage_nt_purity"] > 0.80
