"""Milestone M8 Verification: Full 189,766-Cell Pan-Connectome (BANC v888 + MCNS v1.0) & Genetic Fly Spawner."""

from __future__ import annotations

from pathlib import Path
import pytest
import pyarrow.feather  # noqa: F401 - import before torch on Windows

from fly_nca.connectome import PAN_CONNECTOME_PATH, PanConnectomeSubstrate
from fly_nca.genome import FlySpawner, MultiFlySocialArena3D


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
    """2. Spawn X:A=1.0 (Sxl->Tra->DsxF/Fru-) female and X:A=0.5 (DsxM/FruM+) male flies and verify social behavior."""
    spawner = FlySpawner(seed=108)

    # A. Male + Female pair near food cup -> Male courts & sings (LC10a -> P1 -> pIP10 -> TN1), Female listens (JO -> pC1)
    female = spawner.spawn_fly(sex="F", pos_xyz=(0.002, 0.0, 0.0), yaw_rad=0.0)
    male = spawner.spawn_fly(sex="M", pos_xyz=(-0.002, 0.0, 0.0), yaw_rad=0.0)
    assert female.genome.is_valid() and male.genome.is_valid()

    # Check chromosomal sex masks on the 189,766-cell pan-connectome (8 male-specific in BANC + 1,258 in MCNS = 1,266)
    sub = female.full_connectome.substrate
    assert int((female.full_connectome.sex_cell_mask[sub.sex_code == 3] == 0).sum().item()) == 1266
    assert int((male.full_connectome.sex_cell_mask[sub.sex_code == 2] == 0).sum().item()) == 946

    pair_arena = MultiFlySocialArena3D(flies=[female, male])
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

    # B. Two Males (X:A=0.5 -> Sxl OFF -> Tra OFF -> DsxM + FruM+) on food cup:
    # No artificial stereotypes: on yeast-rich substrate, males may exhibit both VPM4 boxing escalation
    # AND male-male courtship song (Grosjean et al. 2011 Nature / Gill 1963 fruitless chains) purely from connectome dynamics!
    male_1 = spawner.spawn_fly(sex="M", pos_xyz=(0.001, 0.0, 0.0), yaw_rad=0.0)
    male_2 = spawner.spawn_fly(sex="M", pos_xyz=(-0.001, 0.0, 0.0), yaw_rad=3.14)
    fight_arena = MultiFlySocialArena3D(flies=[male_1, male_2])
    snaps_fight = fight_arena.step_population(frames=4)

    assert snaps_fight[0]["boxing_intensity"] > 0.25
    assert snaps_fight[0]["behavior"] in ("MALE_BOXING_ESCALATION", "MALE_COURTSHIP_SONG")
    assert snaps_fight[1]["behavior"] in ("MALE_BOXING_ESCALATION", "MALE_COURTSHIP_SONG")

    # C. Demir & Dickson (2005) / PMC10622783 Experiment:
    # Transgenic female (X:A=1.0 -> Sxl ON -> Tra ON -> DsxF) overexpressing FruM (`fru_m_override=True`):
    # Because Dsx = DsxF, male-specific P1 neurons are ABSENT (`male_p1_rate == 0.0`),
    # yet FruM+ masculinizes VNC song circuits (`pIP10 -> TN1`) so the P1-less female courts and sings!
    wt_target_female = spawner.spawn_fly(sex="F", pos_xyz=(0.002, 0.0, 0.0), yaw_rad=0.0)
    frum_female = spawner.spawn_fly(sex="F", pos_xyz=(-0.002, 0.0, 0.0), yaw_rad=0.0, fru_m_override=True)
    assert frum_female.genome.x_to_autosome_ratio == 1.0
    assert frum_female.genome.dsx_isoform == "DsxF"
    assert frum_female.genome.fru_m_active is True

    frum_arena = MultiFlySocialArena3D(flies=[wt_target_female, frum_female])
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

    # E. Verify ImpulseRecorder traces Embryogenesis (L0), Two-Wave Morphogenesis (L1), and 189,766-cell Pan-Connectome Wave!
    from fly_nca.genome import develop_fly_from_genome
    from fly_nca.nervous import DebugLevel, ImpulseRecorder, format_impulse_replay

    rec = ImpulseRecorder(level=DebugLevel.FULL)
    _ = genome.run_syncytial_embryogenesis_nca(n_positions=64, bcd_mutant=True, recorder=rec)
    _ = develop_fly_from_genome(genome, stage="adult", sex="M", scale=0.05, recorder=rec)
    replay = format_impulse_replay(rec.last_impulses())
    assert "outcome=MUTANT_BICAUDAL_BCD" in replay
    assert "oogenesis.maternal_mrna_poles" in replay
    assert "syncytium.3utr_translational_repression" in replay
    assert "cellularization.zelda_to_opa_frequency_doubling" in replay
    assert "sex_cascade.x_to_a_sxl_tra_dsx_fru" in replay
    assert "metamorphosis.20e_ecdysone_wave2_dati" in replay
    assert "synaptogenesis.cam_dip_dpr_9x9_tracts" in replay


@pytest.mark.skipif(not PAN_CONNECTOME_PATH.is_file(), reason="Compiled 189,766-cell pan-connectome not on disk")
def test_multimodal_courtship_threshold_motor_decode_and_plasticity() -> None:
    """4. Verify multimodal courtship gate (odor vs LC10+ppk23 vs cVA brake), 1,026-cell MotorOut decode, and plasticity."""
    from fly_nca.body import FastTensorArena3D, MotorOut
    from fly_nca.connectome import FullConnectomeFlyNCA

    fly_odor = FullConnectomeFlyNCA(sex="M")
    sub = fly_odor.substrate
    assert sub.soma_xyz_um.shape == (189766, 3)
    assert sub.mirror_partner_idx.shape == (189766,)
    assert int((sub.soma_xyz_um.norm(dim=1) > 0).sum().item()) > 140_000

    arena = FastTensorArena3D(device=fly_odor.device)
    s_dist = arena.step_physics(MotorOut.zeros(device=fly_odor.device), dt_ms=0.0)
    s_dist.small_target_salience = 0.0
    for _ in range(3):
        out_odor = fly_odor.step_full_connectome(
            s_dist, cva_pheromone_conc=0.0, mate_odor_conc=0.85, conspecific_is_female=True
        )
    # Odor alone at a distance stirs P1 slightly but does NOT cross multimodal courtship song threshold!
    assert out_odor.male_p1_courtship_rate < 0.02
    assert out_odor.male_pip10_song_rate < 0.05

    # Multimodal contact (VA1v + LC10a visual + 7,11-HD foreleg ppk23 F-cells -> vAB3/PPN1 -> P1 -> pIP10)
    fly_multi = FullConnectomeFlyNCA(sex="M")
    s_close = arena.step_physics(MotorOut.zeros(device=fly_multi.device), dt_ms=0.0)
    s_close.small_target_salience = 0.85
    for _ in range(3):
        out_multi = fly_multi.step_full_connectome(
            s_close, cva_pheromone_conc=0.0, mate_odor_conc=0.85, conspecific_is_female=True
        )
    motor_multi = fly_multi.decode_motor_effectors()
    assert out_multi.male_p1_courtship_rate > 0.10
    assert out_multi.male_pip10_song_rate > 0.15
    assert float(motor_multi.wing_song_mode[0].item()) > 0.04
    assert motor_multi.leg_joint_drive.shape == (6, 4)

    # Rival male high cVA -> mAL / AN05B023 GABAergic shunt suppresses P1 & pIP10 below song threshold
    fly_rival = FullConnectomeFlyNCA(sex="M")
    for _ in range(3):
        out_rival = fly_rival.step_full_connectome(
            s_close, cva_pheromone_conc=0.85, mate_odor_conc=0.0, conspecific_is_female=False
        )
    assert out_rival.male_p1_courtship_rate < out_multi.male_p1_courtship_rate

    # Verify live sleep consolidation & glial ATP recovery on the 189,766-cell connectome
    sleep_res = fly_multi.step_sleep_consolidation(dcreb2b_blocked=False)
    assert sleep_res["glial_atp_pool"] > 0.90
    assert sleep_res["sleep_debt"] < 0.05


@pytest.mark.skipif(not PAN_CONNECTOME_PATH.is_file(), reason="Compiled 189,766-cell pan-connectome not on disk")
def test_multigenerational_colony_ontogeny_egg_larva_pupa_adult_and_3d_viewer(tmp_path: Path) -> None:
    """5. Verify complete colony ontogeny: Mating -> Oviposition (F1 Egg) -> Hatching (Wave 1 Larva) -> 20E Pupa -> Adult Eclosion."""
    from fly_nca.app.viewer3d import export_colony_3d_viewer_html

    out_html = tmp_path / "colony_3d_viewer.html"
    res = export_colony_3d_viewer_html(output_path=out_html, n_founders=2, frames=18, seed=108)
    assert out_html.is_file()
    assert res["n_final_colony"] >= 3
    event_types = [ev["event"] for ev in res["events"]]
    assert "OVIPOSITION_EGG_LAID" in event_types
    assert "HATCH_EGG_TO_LARVA" in event_types
    assert "PUPARIATION_20E_PULSE" in event_types
    assert "ADULT_ECLOSION_189766_CELLS" in event_types


def test_gene_regulatory_network_automaton_mutations_and_phylogeny() -> None:
    """6. Verify that the pure GeneGraph automaton (zero gene strings in step, unweighted FCA/SPR enhancers) grows distinct phenotypes."""
    from fly_nca.genome.gene_network import build_canonical_gene_graph, run_gene_ontogeny_automaton
    from fly_nca.instruments import TranscriptomeAtlasScanner
    from verify_checks.suites import check_gene_network_purity_and_fca_grounding

    # 1. Verify AST & unweighted enhancer purity gate
    purity = check_gene_network_purity_and_fca_grounding()
    assert purity["ok"] is True
    assert purity["leaked_gene_strings_in_automaton"] == []

    wt_graph = build_canonical_gene_graph()
    ecr_minus_graph = wt_graph.with_knockout("EcR")
    ey_minus_graph = wt_graph.with_knockout("ey")
    # Gehring 1995 (Science): ectopic expression of mouse Pax6 homolog in Antp+ thoracic primordium
    gehring_pax6_graph = ey_minus_graph.with_duplication(
        "ey",
        "Pax6_mouse",
        replace_activators=("Antp",),
        replace_repressors=(),
        feed_into_targets=(("dpr5", "activate"), ("dpr12", "activate"), ("DIP-gamma", "activate")),
    )
    ubx_minus_graph = wt_graph.with_knockout("Ubx")
    notch2nl_graph = wt_graph.with_duplication(
        "CycB",
        "NOTCH2NL",
        replace_activators=("oc",),
        feed_into_targets=(("pros", "repress"), ("Imp", "activate")),
    )

    res_wt = run_gene_ontogeny_automaton(wt_graph)
    res_ecr = run_gene_ontogeny_automaton(ecr_minus_graph)
    res_ey = run_gene_ontogeny_automaton(ey_minus_graph)
    res_gehring = run_gene_ontogeny_automaton(gehring_pax6_graph)
    res_ubx = run_gene_ontogeny_automaton(ubx_minus_graph)
    res_n2nl = run_gene_ontogeny_automaton(notch2nl_graph)

    # 2. Gehring (1995) eyeless / Pax6 deep homology: ey^- ablates eye cells; mouse Pax6 under Antp induces ectopic eye cascade!
    assert res_wt.count_expressing("ey") > 0
    assert res_ey.count_expressing("ey") == 0
    assert res_gehring.count_expressing("Pax6_mouse") > 0
    assert res_gehring.count_expressing("dpr5") > res_ey.count_expressing("dpr5")

    # 3. Lewis (1978/1995) Ubx^- knockout eliminates Ubx+ cells and expands Antp+ / T2_WingMesothorax domain
    assert res_wt.count_expressing("Ubx") > 0
    assert res_ubx.count_expressing("Ubx") == 0
    assert res_ubx.count_expressing("Antp") > res_wt.count_expressing("Antp")

    # 4. NOTCH2NL duplication under forebrain `oc` (Otx) enhancer expands `oc+` pallial progenitors
    assert res_n2nl.count_expressing("oc") > 5 * res_wt.count_expressing("oc")

    # 5. External observer instrument (TranscriptomeAtlasScanner) classifies phenotypes from raw gene expression
    scanner = TranscriptomeAtlasScanner()
    rep_wt = scanner.scan(res_wt)
    rep_ecr = scanner.scan(res_ecr, wild_type_baseline=rep_wt)
    rep_ubx = scanner.scan(res_ubx, wild_type_baseline=rep_wt)
    rep_n2nl = scanner.scan(res_n2nl, wild_type_baseline=rep_wt)

    assert rep_wt.matched_phenotype == "drosophila.melanogaster.wild_type"
    assert rep_ecr.matched_phenotype == "urbilateria.primary_wave1_worm"
    assert rep_ubx.matched_phenotype == "drosophila.melanogaster.ubx_bithorax_homeotic_4wing"
    assert rep_n2nl.matched_phenotype == "chordata.pallium_notch2nl_expanded"
    assert rep_ubx.domain_counts["T3_HaltereMetathorax"] == 0
    assert rep_ubx.domain_counts["T2_WingMesothorax"] > rep_wt.domain_counts["T2_WingMesothorax"]
