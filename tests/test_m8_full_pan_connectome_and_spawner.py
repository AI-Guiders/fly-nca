"""Milestone M8 Verification: Full 189,766-Cell Pan-Connectome (BANC v888 + MCNS v1.0) & Genetic Fly Spawner."""

from __future__ import annotations

import pytest
import pyarrow.feather  # noqa: F401 - import before torch on Windows

from fly_nca.connectome import PAN_CONNECTOME_PATH, PanConnectomeSubstrate
from fly_nca.genome import FlySpawner, MultiFlySocialArena4D


@pytest.mark.skipif(not PAN_CONNECTOME_PATH.is_file(), reason="Compiled 189,766-cell pan-connectome not on disk")
def test_full_189766_pan_connectome_structure_and_5_ports() -> None:
    """1. Verify that PanConnectomeSubstrate loads all 188,508 BANC + 1,258 MCNS male-specific = 189,766 cells & 6.75M edges."""
    sub = PanConnectomeSubstrate.get()
    assert sub.n_banc == 188508
    assert sub.n_male_specific == 1258
    assert sub.n_total == 189766
    assert sub.n_edges > 6_700_000

    # Verify all 5 subcellular port sparse CSR matrices have shape (189766, 189766)
    for mat in (sub.w_port1_ach, sub.w_port2_shunt, sub.w_port2_root, sub.w_port3_aa, sub.w_port5_mod):
        assert mat.shape == (189766, 189766)

    # Verify exact sensory, central circuit, dimorphic, and effector cell index groups
    assert sub.cell_groups["circuit_kc"].numel() == 4552
    assert sub.cell_groups["circuit_lhon"].numel() == 1080
    assert sub.cell_groups["sens_orn_all"].numel() == 3007
    assert sub.cell_groups["dn_gf_dnp01"].numel() == 2
    assert sub.cell_groups["dn_mdn"].numel() == 4
    assert sub.cell_groups["LBL40_t3"].numel() == 2
    assert sub.cell_groups["dimorphic_male_pip10"].numel() == 6
    assert sub.cell_groups["dimorphic_male_p1"].numel() > 100


@pytest.mark.skipif(not PAN_CONNECTOME_PATH.is_file(), reason="Compiled 189,766-cell pan-connectome not on disk")
def test_fly_spawner_valid_genomes_and_male_female_social_interactions() -> None:
    """2. Spawn XX female and XY male flies with valid genomes and verify courtship song, receptivity, and male-male boxing."""
    spawner = FlySpawner(seed=108)

    # A. Male + Female pair near food cup -> Male courts & sings (LC10a -> P1 -> pIP10 -> TN1), Female listens (JO -> pC1)
    female = spawner.spawn_fly(sex="F", pos_xyz=(0.002, 0.0, 0.0), yaw_rad=0.0)
    male = spawner.spawn_fly(sex="M", pos_xyz=(-0.002, 0.0, 0.0), yaw_rad=0.0)
    assert female.genome.is_valid() and male.genome.is_valid()

    # Check chromosomal sex masks on the 189,766-cell pan-connectome (8 male-specific in BANC + 1,258 in MCNS = 1,266)
    sub = female.full_connectome.substrate
    assert int((female.full_connectome.sex_cell_mask[sub.sex_code == 3] == 0).sum().item()) == 1266
    assert int((male.full_connectome.sex_cell_mask[sub.sex_code == 2] == 0).sum().item()) == 946

    pair_arena = MultiFlySocialArena4D(flies=[female, male])
    snaps_pair = pair_arena.step_population(frames=4)
    snap_f, snap_m = snaps_pair[0], snaps_pair[1]

    # Male activates P1 & pIP10 and sings unilateral courtship song toward the female
    assert snap_m["sex"] == "M"
    assert snap_m["wing_song_active"] is True
    assert snap_m["behavior"] == "MALE_COURTSHIP_SONG"
    assert snap_m["male_pip10_rate"] > 0.15

    # Female hears the male's song + cVA and activates pC1 receptivity (or oviposition on the yeast cup)
    assert snap_f["sex"] == "F"
    assert snap_f["behavior"] in ("FEMALE_SONG_RECEPTIVITY", "FEMALE_OVIPOSITION")
    assert snap_f["female_pc1_rate"] > 0.15

    # B. Two Males (X:A=0.5 -> Sxl OFF -> Tra OFF -> DsxM + FruM+) on food cup -> rival cVA -> MALE_BOXING_ESCALATION
    male_1 = spawner.spawn_fly(sex="M", pos_xyz=(0.001, 0.0, 0.0), yaw_rad=0.0)
    male_2 = spawner.spawn_fly(sex="M", pos_xyz=(-0.001, 0.0, 0.0), yaw_rad=3.14)
    fight_arena = MultiFlySocialArena4D(flies=[male_1, male_2])
    snaps_fight = fight_arena.step_population(frames=4)

    assert snaps_fight[0]["wing_song_active"] is False
    assert snaps_fight[0]["behavior"] == "MALE_BOXING_ESCALATION"
    assert snaps_fight[0]["boxing_intensity"] > 0.25
    assert snaps_fight[1]["behavior"] == "MALE_BOXING_ESCALATION"

    # C. Demir & Dickson (2005) / PMC10622783 Experiment:
    # Transgenic female (X:A=1.0 -> Sxl ON -> Tra ON -> DsxF) overexpressing FruM (`fru_m_override=True`):
    # Because Dsx = DsxF, male-specific P1 neurons are ABSENT (`male_p1_rate == 0.0`),
    # yet FruM+ masculinizes VNC song circuits (`pIP10 -> TN1`) so the P1-less female courts and sings!
    wt_target_female = spawner.spawn_fly(sex="F", pos_xyz=(0.002, 0.0, 0.0), yaw_rad=0.0)
    frum_female = spawner.spawn_fly(sex="F", pos_xyz=(-0.002, 0.0, 0.0), yaw_rad=0.0, fru_m_override=True)
    assert frum_female.genome.x_to_autosome_ratio == 1.0
    assert frum_female.genome.dsx_isoform == "DsxF"
    assert frum_female.genome.fru_m_active is True

    frum_arena = MultiFlySocialArena4D(flies=[wt_target_female, frum_female])
    snaps_frum = frum_arena.step_population(frames=4)
    snap_frum_f = snaps_frum[1]
    assert snap_frum_f["male_p1_rate"] == 0.0  # P1 neurons absent under DsxF!
    assert snap_frum_f["wing_song_active"] is True
    assert snap_frum_f["behavior"] == "FEMALE_FRUM_COURTSHIP_SONG"


def test_syncytial_blastoderm_embryogenesis_and_nobel1995_mutants() -> None:
    """3. Verify Syncytial Blastoderm -> Translational Repression -> Gap -> Zelda/Opa -> Hox NCA (zero linspace x!)."""
    from fly_nca.genome import FlyGenome

    genome = FlyGenome()

    # A. Wild-type embryo: self-organizes all 8 Hox neuromeres (0..7) purely from pole boundary diffusion
    wt = genome.run_syncytial_embryogenesis_nca(n_positions=64)
    assert sorted(wt.segment_ids.unique().tolist()) == [0, 1, 2, 3, 4, 5, 6, 7]
    # Translational repression: Caudal is repressed at anterior pole (0) by Bicoid; Hunchback is repressed at posterior (-1) by Nanos
    assert float(wt.caudal_protein[0]) < float(wt.caudal_protein[-1]) * 0.35
    assert float(wt.hunchback_protein[-1]) < float(wt.hunchback_protein[0]) * 0.25
    # Posterior pole cells (primordial germ cells) bud off at posterior tip
    assert float(wt.pole_cells_mask[-1]) == 1.0 and float(wt.pole_cells_mask[0]) == 0.0

    # B. Maternal bicoid^- mutant (Nobel 1995): loses anterior Head & Thorax segments (0, 1, 2) -> bicaudal abdomen!
    bcd_mut = genome.run_syncytial_embryogenesis_nca(n_positions=64, bcd_mutant=True)
    bcd_segs = set(bcd_mut.segment_ids.unique().tolist())
    assert 0 not in bcd_segs and 1 not in bcd_segs and 2 not in bcd_segs

    # C. Maternal nanos^- mutant (Nobel 1995): unrepressed posterior Hunchback wipes out abdominal knirps domain!
    nos_mut = genome.run_syncytial_embryogenesis_nca(n_positions=64, nos_mutant=True)
    assert float(nos_mut.knirps_protein.max().item()) < 0.15

    # D. Zygotic odd-paired (opa^-) pioneer mutant (PLoS Genetics 2021 PMC8516289): loses 7->14 stripe frequency doubling
    opa_mut = genome.run_syncytial_embryogenesis_nca(n_positions=64, opa_mutant=True)
    assert float((opa_mut.opa_14_stripes - opa_mut.zelda_7_stripes).abs().max().item()) == 0.0
