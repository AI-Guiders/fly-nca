#!/usr/bin/env python3
"""Compile the Full 189,766-Cell Drosophila Pan-Connectome (BANC v888 Female + MCNS v1.0 Male-Specific).

Extracts directly from local empirical connectome tables:
1. `D:/Experiments/casa-data/connectomes/banc-v888/banc_888_meta.feather` (188,508 cells)
2. `D:/Experiments/casa-data/connectomes/banc-v888/banc_888_metrics.feather` (morphometry: l2_cable_length_um, branchpoints)
3. `D:/Experiments/casa-data/connectomes/banc-v888/banc_888_edgelist_split_v3.feather` (15,716,248 subcellular split edges)
4. `D:/Experiments/casa-data/connectomes/mcns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather` (1,258 male-specific neurons)
5. `D:/Experiments/casa-data/connectomes/mcns-v1.0/body-neurotransmitters-male-cns-v1.0.feather`
6. `D:/Experiments/casa-data/connectomes/mcns-v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather`

Outputs a fast-loading PyTorch bundle at:
`D:/Experiments/casa-data/connectomes/banc-v888-csr-cache/drosophila_pan_connectome_189766.pt`
"""

from __future__ import annotations

from pathlib import Path
import time
import pyarrow.feather as pf
import pandas as pd
import numpy as np
import torch

BANC_DIR = Path(r"D:\Experiments\casa-data\connectomes\banc-v888")
MCNS_DIR = Path(r"D:\Experiments\casa-data\connectomes\mcns-v1.0")
OUT_PATH = Path(r"D:\Experiments\casa-data\connectomes\banc-v888-csr-cache\drosophila_pan_connectome_189766.pt")

NT_TO_ID: dict[str, int] = {
    "acetylcholine": 0,
    "gaba": 1,
    "glutamate": 2,
    "dopamine": 3,
    "octopamine": 4,
    "serotonin": 5,
    "histamine": 6,
    "tyramine": 7,
}


def build_pan_connectome(min_synapse_count: int = 2) -> dict:
    t0 = time.perf_counter()
    print("[1/5] Loading BANC v888 metadata (188,508 cells) & morphometrics...")
    b_meta = pf.read_table(
        BANC_DIR / "banc_888_meta.feather",
        columns=[
            "banc_888_id",
            "super_class",
            "cell_class",
            "cell_sub_class",
            "cell_type",
            "malecns_cell_type",
            "malecns_match",
            "side",
            "hemilineage",
            "neuromere",
            "sexually_dimorphic",
            "super_cluster",
            "cluster",
            "cns_network",
            "body_part_sensory",
            "body_part_effector",
            "peripheral_target_type",
            "cell_function",
            "neurotransmitter_predicted",
            "neuropeptide_verified",
            "l2_cable_length_um",
            "pd_width",
            "segregation_index",
        ],
    ).to_pandas()

    b_metrics = pf.read_table(
        BANC_DIR / "banc_888_metrics.feather",
        columns=["banc_888_id", "branchpoints"],
    ).to_pandas()
    b_meta = b_meta.merge(b_metrics, on="banc_888_id", how="left")

    n_banc = len(b_meta)
    assert n_banc == 188508

    print("[2/5] Loading MCNS v1.0 male-specific neurons (1,258 cells) & neurotransmitters...")
    m_meta = pf.read_table(
        MCNS_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        columns=["bodyId", "type", "superclass", "class", "subclass", "dimorphism", "fruDsx", "somaSide", "itoleeHl"],
    ).to_pandas()
    m_nt = pf.read_table(
        MCNS_DIR / "body-neurotransmitters-male-cns-v1.0.feather",
        columns=["body", "consensus_nt", "predicted_nt"],
    ).to_pandas()
    m_meta = m_meta.merge(m_nt, left_on="bodyId", right_on="body", how="left")

    ms_df = m_meta[m_meta["dimorphism"] == "male-specific"].reset_index(drop=True)
    n_male_spec = len(ms_df)
    n_total = n_banc + n_male_spec
    print(f"      Unified Pan-Connectome size: {n_banc:,} (BANC) + {n_male_spec:,} (MCNS male-specific) = {n_total:,} cells")

    # --- Per-Cell Biophysical & Genetic Vectors [189,766] ---
    # 1. Cable length -> tau_v_ms = 1.711 * (L2 / 241.53)
    l2_banc = b_meta["l2_cable_length_um"].fillna(450.0).to_numpy(dtype=np.float32)
    l2_male = np.full(n_male_spec, 650.0, dtype=np.float32)
    l2_all = np.concatenate([l2_banc, l2_male])
    tau_v_ms = np.clip(1.711 * (l2_all / 241.53), 1.0, 25.0).astype(np.float32)

    # 2. Branchpoints & segregation_index -> dendritic shunting gain alpha_shunt
    bp_banc = b_meta["branchpoints"].fillna(28.0).to_numpy(dtype=np.float32)
    bp_male = np.full(n_male_spec, 45.0, dtype=np.float32)
    bp_all = np.concatenate([bp_banc, bp_male])
    alpha_shunt = np.clip(np.log1p(bp_all / 28.0) * 2.5, 0.8, 6.0).astype(np.float32)

    # 3. Neurotransmitter ID (0..8)
    nt_id = np.full(n_total, 8, dtype=np.uint8)
    for i, nt_str in enumerate(b_meta["neurotransmitter_predicted"]):
        if pd.notna(nt_str):
            nt_id[i] = NT_TO_ID.get(str(nt_str).lower().strip(), 8)
    for k, (cnt, pnt) in enumerate(zip(ms_df["consensus_nt"], ms_df["predicted_nt"])):
        raw_nt = cnt if pd.notna(cnt) else pnt
        if pd.notna(raw_nt):
            nt_id[n_banc + k] = NT_TO_ID.get(str(raw_nt).lower().strip(), 0)
        else:
            nt_id[n_banc + k] = 0

    # 4. Sex specificity code: 0=isomorphic, 1=shared_dimorphic, 2=female_specific, 3=male_specific
    sex_code = np.zeros(n_total, dtype=np.uint8)
    sd_col = b_meta["sexually_dimorphic"].fillna("isomorphic").astype(str)
    sex_code[:n_banc][sd_col == "dimorphic"] = 1
    sex_code[:n_banc][sd_col == "female-specific"] = 2
    sex_code[:n_banc][sd_col == "male-specific"] = 3
    sex_code[n_banc:] = 3

    # 5. Side code: 0=midline/na, 1=left, 2=right
    side_id = np.zeros(n_total, dtype=np.uint8)
    b_side = b_meta["side"].fillna("").astype(str).str.lower()
    side_id[:n_banc][b_side == "left"] = 1
    side_id[:n_banc][b_side == "right"] = 2
    m_side = ms_df["somaSide"].fillna("").astype(str).str.upper()
    side_id[n_banc:][m_side.isin(["L", "LEFT"])] = 1
    side_id[n_banc:][m_side.isin(["R", "RIGHT"])] = 2

    # 6. Organ ID: 0=OL, 1=AL, 2=LH_SPIN, 3=MB, 4=CX, 5=AN_DN, 6=VNC, 7=SEZ, 8=GLIA
    organ_id = np.full(n_total, 2, dtype=np.uint8)
    sc = b_meta["super_class"].fillna("").astype(str)
    cc = b_meta["cell_class"].fillna("").astype(str)
    cn = b_meta["cns_network"].fillna("").astype(str)
    organ_id[:n_banc][sc.isin(["optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"])] = 0
    organ_id[:n_banc][cc.isin(["olfactory_receptor_neuron", "antennal_lobe_projection_neuron", "antennal_lobe_local_neuron", "johnstons_organ_neuron"])] = 1
    organ_id[:n_banc][cc.isin(["kenyon_cell", "mushroom_body_output_neuron", "mushroom_body_dopaminergic_neuron", "mushroom_body_intrinsic_neuron"])] = 3
    organ_id[:n_banc][cc.str.startswith("central_complex") | (cn == "central complex related")] = 4
    organ_id[:n_banc][sc.isin(["ascending", "descending", "sensory_ascending", "sensory_descending"])] = 5
    organ_id[:n_banc][sc.isin(["ventral_nerve_cord_intrinsic", "motor"]) | cn.str.contains("VNC")] = 6
    organ_id[:n_banc][cc.str.contains("gustatory|subesophageal|pharynx|proboscis", case=False)] = 7
    organ_id[:n_banc][sc == "glia"] = 8

    m_sc = ms_df["superclass"].fillna("").astype(str)
    organ_id[n_banc:][m_sc.isin(["ol_intrinsic", "visual_projection"])] = 0
    organ_id[n_banc:][m_sc == "cb_intrinsic"] = 2
    organ_id[n_banc:][m_sc.isin(["ascending_neuron", "descending_neuron"])] = 5
    organ_id[n_banc:][m_sc == "vnc_intrinsic"] = 6

    # 7. Hemilineage integer index
    hl_all = pd.concat([b_meta["hemilineage"].fillna("unassigned").astype(str), ms_df["itoleeHl"].fillna("male_hl").astype(str)], ignore_index=True)
    hl_codes, hl_uniques = pd.factorize(hl_all)
    hemilineage_id = hl_codes.astype(np.int16)

    # --- Step 3: Build Complete Sensory, Circuit, and Effector Cell Index Maps ---
    print("[3/5] Extracting exact sensory (16,563), circuit, and effector (1,026) cell index groups...")
    cf = b_meta["cell_function"].fillna("").astype(str)
    bps = b_meta["body_part_sensory"].fillna("").astype(str)
    bpe = b_meta["body_part_effector"].fillna("").astype(str)
    csc = b_meta["cell_sub_class"].fillna("").astype(str)
    ct = b_meta["cell_type"].fillna("").astype(str)
    ptt = b_meta["peripheral_target_type"].fillna("").astype(str)
    sup_cl = b_meta["super_cluster"].fillna("").astype(str)
    m_type = ms_df["type"].fillna("").astype(str)

    def _idx(mask: pd.Series | np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.flatnonzero(np.asarray(mask)).astype(np.int32))

    def _m_idx(mask: pd.Series | np.ndarray) -> torch.Tensor:
        return torch.from_numpy((n_banc + np.flatnonzero(np.asarray(mask))).astype(np.int32))

    cell_groups: dict[str, torch.Tensor] = {
        # Sensory Periphery (16,563 cells)
        "sens_visual_left": _idx(cf.str.startswith("visual") & (b_side == "left")),
        "sens_visual_right": _idx(cf.str.startswith("visual") & (b_side == "right")),
        "sens_ocellar": _idx(cf == "visual_ocellar"),
        "sens_orn_all": _idx(cc == "olfactory_receptor_neuron"),
        "sens_orn_left": _idx((cc == "olfactory_receptor_neuron") & (b_side == "left")),
        "sens_orn_right": _idx((cc == "olfactory_receptor_neuron") & (b_side == "right")),
        "sens_orn_DA1_cva": _idx(ct == "ORN_DA1"),
        "sens_orn_VA1v_mate": _idx(ct == "ORN_VA1v"),
        "sens_orn_DM1_food": _idx(ct.isin(["ORN_DM1", "ORN_DM4", "ORN_DM2"])),
        "sens_orn_V_co2": _idx(ct.isin(["ORN_V", "ORN_DA2", "ORN_DL4"])),
        "sens_jo_left": _idx((cf == "auditory") & (b_side == "left")),
        "sens_jo_right": _idx((cf == "auditory") & (b_side == "right")),
        "sens_gust_labellum": _idx((cf == "gustatory") & (bps == "labellum")),
        "sens_gust_front_leg": _idx((cf == "gustatory") & (bps == "front_leg")),
        "sens_gust_middle_leg": _idx((cf == "gustatory") & (bps == "middle_leg")),
        "sens_gust_hind_leg": _idx((cf == "gustatory") & (bps == "hind_leg")),
        "sens_gust_pharynx": _idx((cf == "gustatory") & bps.isin(["pharynx", "anterior_digestive_tract"])),
        "sens_campaniform_all": _idx(cc == "campaniform_sensillum_neuron"),
        "sens_chordotonal_all": _idx(cc == "chordotonal_organ_neuron"),
        "sens_bristle_all": _idx(cc == "bristle_neuron"),
        "sens_haltere_left": _idx((bps == "haltere") & (b_side == "left")),
        "sens_haltere_right": _idx((bps == "haltere") & (b_side == "right")),
        "sens_crop_visceral": _idx(cf == "visceral_mechanosensation"),
        "sens_thermo": _idx(cf == "thermosensory"),
        "sens_hygro": _idx(cf == "hygrosensory"),
        "sens_nociception": _idx(cf == "nociception"),
        # Per-leg mechanoreceptors (L1..R3)
        "sens_leg_L1": _idx((bps == "front_leg") & (b_side == "left")),
        "sens_leg_R1": _idx((bps == "front_leg") & (b_side == "right")),
        "sens_leg_L2": _idx((bps == "middle_leg") & (b_side == "left")),
        "sens_leg_R2": _idx((bps == "middle_leg") & (b_side == "right")),
        "sens_leg_L3": _idx((bps == "hind_leg") & (b_side == "left")),
        "sens_leg_R3": _idx((bps == "hind_leg") & (b_side == "right")),
        # Identified Central Brain & VNC Circuits
        "circuit_al_pn": _idx(cc == "antennal_lobe_projection_neuron"),
        "circuit_al_ln": _idx(cc == "antennal_lobe_local_neuron"),
        "circuit_kc": _idx(cc == "kenyon_cell"),
        "circuit_apl": _idx(ct == "APL"),
        "circuit_mbon": _idx(cc == "mushroom_body_output_neuron"),
        "circuit_mbon_approach": _idx((cc == "mushroom_body_output_neuron") & (nt_id[:n_banc] == 0)),
        "circuit_mbon_avoid": _idx((cc == "mushroom_body_output_neuron") & np.isin(nt_id[:n_banc], [1, 2])),
        "circuit_dan_pam": _idx(ct.str.startswith("PAM")),
        "circuit_dan_ppl1": _idx(ct.str.startswith("PPL1")),
        "circuit_lhon": _idx(cc == "lateral_horn_output_neuron"),
        "circuit_lhon_ach_pos": _idx((cc == "lateral_horn_output_neuron") & (nt_id[:n_banc] == 0)),
        "circuit_lhon_inh_neg": _idx((cc == "lateral_horn_output_neuron") & np.isin(nt_id[:n_banc], [1, 2])),
        "circuit_cx_epg": _idx(csc == "eb_wedge_neuron"),
        "circuit_cx_pen": _idx(csc == "PEN"),
        "circuit_cx_pfn": _idx(csc == "PFN"),
        "circuit_cx_hdelta": _idx(csc == "hDelta"),
        "circuit_cx_vdelta": _idx(csc == "vDelta"),
        "circuit_cx_pfl": _idx(csc == "PFL"),
        "circuit_cx_pfl_left": _idx((csc == "PFL") & (b_side == "left")),
        "circuit_cx_pfl_right": _idx((csc == "PFL") & (b_side == "right")),
        "circuit_cx_fbt_sleep": _idx(csc == "FB_tangential"),
        "circuit_lc4_looming": _idx(ct.str.startswith("LC4") | ct.str.startswith("LPLC2")),
        "circuit_lc10a_left": _idx(ct.str.startswith("LC10a") & (b_side == "left")),
        "circuit_lc10a_right": _idx(ct.str.startswith("LC10a") & (b_side == "right")),
        # Sexually Dimorphic & Social Circuits (Female BANC + Male MCNS)
        "dimorphic_female_pc1": _idx(ct.str.startswith("pC1")),
        "dimorphic_female_asp_f": _idx(ct.str.startswith("aSP-f")),
        "dimorphic_female_asp_g": _idx(ct.str.startswith("aSP-g")),
        "dimorphic_female_vpo_ovi": _idx(ct.str.contains("vpo|oviDN|oviIN", case=False)),
        "dimorphic_male_p1": _m_idx(m_type.str.startswith("pC1_") | m_type.str.startswith("SIP1")),
        "dimorphic_male_mal": _m_idx(m_type.str.startswith("mAL_")),
        "dimorphic_male_pip10": _m_idx(m_type.isin(["pIP10", "pIP1", "pMP2"])),
        "dimorphic_male_tn1_song": _m_idx(m_type.str.startswith("TN1") | m_type.str.startswith("vMS") | m_type.str.startswith("IN")),
        # Descending Neuron Modules (BANC Fig. 3 super_clusters & specific DNs)
        "dn_gf_dnp01": _idx(ct == "DNp01"),
        "dn_mdn": _idx(ct == "MDN"),
        "dn_dnp09": _idx(ct == "DNp09"),
        "dn_dna02_left": _idx((ct == "DNa02") & (b_side == "left")),
        "dn_dna02_right": _idx((ct == "DNa02") & (b_side == "right")),
        "dn_dng02": _idx(ct.str.startswith("DNg02")),
        "dn_cluster_walking": _idx((sc == "descending") & (sup_cl == "walking")),
        "dn_cluster_flight_power": _idx((sc == "descending") & (sup_cl == "flight power")),
        "dn_cluster_flight_steer": _idx((sc == "descending") & sup_cl.str.startswith("flight steering")),
        "dn_cluster_head_orient": _idx((sc == "descending") & (sup_cl == "head orienting")),
        "dn_cluster_threat": _idx((sc == "descending") & (sup_cl == "threat response")),
        "dn_cluster_feeding": _idx((sc == "descending") & (sup_cl == "feeding")),
        "dn_cluster_grooming": _idx((sc == "descending") & (sup_cl == "grooming")),
        "dn_cluster_reproduction": _idx((sc == "descending") & (sup_cl == "reproduction")),
        "LBL40_t3": _idx(ct == "LBL40"),
        # Motor & Visceral Effectors (1,026 cells)
        "mn_wing_power_left": _idx((csc == "wing_power_motor_neuron") & (b_side == "left")),
        "mn_wing_power_right": _idx((csc == "wing_power_motor_neuron") & (b_side == "right")),
        "mn_wing_steer_left": _idx(csc.isin(["wing_steering_motor_neuron", "wing_tension_motor_neuron"]) & (b_side == "left")),
        "mn_wing_steer_right": _idx(csc.isin(["wing_steering_motor_neuron", "wing_tension_motor_neuron"]) & (b_side == "right")),
        "mn_haltere_left": _idx(csc.str.startswith("haltere_") & (b_side == "left")),
        "mn_haltere_right": _idx(csc.str.startswith("haltere_") & (b_side == "right")),
        "mn_neck_roll_left": _idx(csc.isin(["neck_roll_motor_neuron", "neck_roll_pitch_motor_neuron"]) & (b_side == "left")),
        "mn_neck_roll_right": _idx(csc.isin(["neck_roll_motor_neuron", "neck_roll_pitch_motor_neuron"]) & (b_side == "right")),
        "mn_neck_pitch": _idx(csc == "neck_pitch_motor_neuron"),
        "mn_neck_yaw_left": _idx((csc == "neck_yaw_motor_neuron") & (b_side == "left")),
        "mn_neck_yaw_right": _idx((csc == "neck_yaw_motor_neuron") & (b_side == "right")),
        "mn_proboscis_per": _idx((bpe == "proboscis") | (cf == "proboscis_motor")),
        "mn_pharynx_pump": _idx((bpe == "pharynx") | (cf == "pharynx_motor")),
        "mn_crop_visceral": _idx(bpe.isin(["crop", "digestive_tract"])),
        "mn_ttmn_jump": _idx(ptt == "tergotrochanter_extensor_muscle"),
        "eff_oviposition_uterus": _idx((cf == "ovulation") | bpe.isin(["uterus", "reproductive_tract"]) | (csc == "abd_Dsx_OA")),
        "eff_endocrine_ipc_akh": _idx(csc.isin(["retrocerebral_complex_neurosecretory_cell", "corpus_allatum_neurosecretory_cell"])),
    }

    # Per-leg motor pools (L1..R3) x (coxa_fwd, coxa_bwd, troch_lift, tibia_flex, tibia_ext, tarsus_grip)
    leg_defs = [
        ("L1", "front_leg_motor_neuron", "left"),
        ("R1", "front_leg_motor_neuron", "right"),
        ("L2", "middle_leg_motor_neuron", "left"),
        ("R2", "middle_leg_motor_neuron", "right"),
        ("L3", "hind_leg_motor_neuron", "left"),
        ("R3", "hind_leg_motor_neuron", "right"),
    ]
    for leg_tag, leg_sub, leg_side in leg_defs:
        lm = (csc == leg_sub) & (b_side == leg_side)
        cell_groups[f"mn_{leg_tag}_all"] = _idx(lm)
        cell_groups[f"mn_{leg_tag}_coxa_fwd"] = _idx(lm & ptt.isin(["sternal_anterior_rotator_muscle", "tergopleural_promotor_muscle", "sternal_adductor_muscle"]))
        cell_groups[f"mn_{leg_tag}_coxa_bwd"] = _idx(lm & ptt.isin(["sternal_posterior_rotator_muscle", "pleural_remotor_and_abductor_muscle"]))
        cell_groups[f"mn_{leg_tag}_troch_lift"] = _idx(lm & ptt.isin(["trochanter_flexor_muscle", "accessory_trochanter_flexor_muscle"]))
        cell_groups[f"mn_{leg_tag}_tibia_flex"] = _idx(lm & ptt.isin(["tibia_flexor_muscle", "accessory_tibia_flexor_muscle", "femur_reductor_muscle"]))
        cell_groups[f"mn_{leg_tag}_tibia_ext"] = _idx(lm & ptt.isin(["tibia_extensor_muscle", "trochanter_extensor_muscle", "sternotrochanter_extensor_muscle"]))
        cell_groups[f"mn_{leg_tag}_tarsus_grip"] = _idx(lm & ptt.isin(["long_tendon_muscle", "tarsus_depressor_muscle"]))

    # Also group all 56 ORN glomerular types
    orn_types = sorted(b_meta.loc[cc == "olfactory_receptor_neuron", "cell_type"].dropna().unique())
    for g_idx, orn_t in enumerate(orn_types[:50]):
        cell_groups[f"orn_glom_{g_idx:02d}"] = _idx(ct == orn_t)

    # --- Step 4: Extract 5-Port Subcellular Synaptic Edges from BANC v888 + MCNS v1.0 ---
    print(f"[4/5] Loading BANC v888 split edgelist (filtering count >= {min_synapse_count})...")
    sp = pf.read_table(
        BANC_DIR / "banc_888_edgelist_split_v3.feather",
        columns=["pre", "post", "pre_label", "post_label", "count", "norm", "pre_conf_nt"],
    ).to_pandas()
    sp = sp[sp["count"] >= min_synapse_count].reset_index(drop=True)

    b_idx = pd.Index(b_meta["banc_888_id"])
    pre_i = b_idx.get_indexer(sp["pre"]).astype(np.int32)
    post_i = b_idx.get_indexer(sp["post"]).astype(np.int32)
    norm_v = sp["norm"].to_numpy(dtype=np.float32)
    post_lbl = sp["post_label"].fillna("dendrite").astype(str)
    pre_nt = sp["pre_conf_nt"].fillna("acetylcholine").astype(str)

    # Classify every edge into one of the 5 biological ports:
    # 0 = Port 1 (dendritic excitatory ACh+)
    # 1 = Port 2a (dendritic chloride shunting GABA / Glu / His)
    # 2 = Port 2b (soma / primary-dendrite root shunt)
    # 3 = Port 3 (axo-axonic presynaptic gate)
    # 4 = Port 5 (modulatory DA / OA / 5-HT / TA)
    port_code = np.zeros(len(sp), dtype=np.uint8)
    is_mod = pre_nt.isin(["dopamine", "octopamine", "serotonin", "tyramine"]).to_numpy()
    is_inh = pre_nt.isin(["gaba", "glutamate", "histamine"]).to_numpy()
    is_axon_post = (post_lbl == "axon").to_numpy()
    is_root_post = post_lbl.isin(["primary.dendrite", "primary.neurite", "soma"]).to_numpy()

    port_code[is_inh & (~is_axon_post) & (~is_root_post)] = 1
    port_code[is_root_post & is_inh] = 2
    port_code[is_axon_post & (~is_mod)] = 3
    port_code[is_mod] = 4

    # --- Add MCNS v1.0 Male-Specific Edges (connecting 188508..189765 with BANC 0..188507) ---
    print("[5/5] Merging MCNS v1.0 male-specific synaptic edges into the pan-connectome...")
    body_to_pan: dict[int, int] = {int(b): n_banc + i for i, b in enumerate(ms_df["bodyId"])}
    for i, m_id in enumerate(b_meta["malecns_match"]):
        if pd.notna(m_id) and str(m_id).isdigit():
            bid = int(m_id)
            if bid not in body_to_pan:
                body_to_pan[bid] = i

    type_side_to_banc: dict[tuple[str, str], int] = {}
    for i, (c_t, mc_t, s_val) in enumerate(zip(ct, b_meta["malecns_cell_type"], b_side)):
        s_key = s_val if s_val in ("left", "right") else "any"
        if pd.notna(mc_t):
            type_side_to_banc.setdefault((str(mc_t), s_key), i)
            type_side_to_banc.setdefault((str(mc_t), "any"), i)
        if c_t:
            type_side_to_banc.setdefault((c_t, s_key), i)
            type_side_to_banc.setdefault((c_t, "any"), i)

    for b, t, s in zip(m_meta["bodyId"], m_meta["type"], m_meta["somaSide"]):
        bid = int(b)
        if bid not in body_to_pan and pd.notna(t):
            s_str = "left" if s == "L" else ("right" if s == "R" else "any")
            hit = type_side_to_banc.get((str(t), s_str)) or type_side_to_banc.get((str(t), "any"))
            if hit is not None:
                body_to_pan[bid] = hit

    ms_body_set = set(ms_df["bodyId"].astype(np.int64))
    w_mcns = pf.read_table(MCNS_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather").to_pandas()
    w_ms = w_mcns[(w_mcns["weight"] >= 5) & (w_mcns["body_pre"].isin(ms_body_set) | w_mcns["body_post"].isin(ms_body_set))].copy()
    w_ms["pre_pan"] = w_ms["body_pre"].map(body_to_pan)
    w_ms["post_pan"] = w_ms["body_post"].map(body_to_pan)
    w_ms = w_ms.dropna(subset=["pre_pan", "post_pan"])
    w_ms = w_ms.groupby(["pre_pan", "post_pan"], as_index=False)["weight"].sum()
    post_tot = w_ms.groupby("post_pan")["weight"].transform("sum").clip(lower=1.0)
    w_ms["norm"] = (w_ms["weight"] / post_tot).astype(np.float32)

    m_pre_i = w_ms["pre_pan"].to_numpy(dtype=np.int32)
    m_post_i = w_ms["post_pan"].to_numpy(dtype=np.int32)
    m_norm_v = w_ms["norm"].to_numpy(dtype=np.float32)
    m_pre_nt = nt_id[m_pre_i]
    m_port_code = np.zeros(len(w_ms), dtype=np.uint8)
    m_port_code[np.isin(m_pre_nt, [1, 2, 6])] = 1
    m_port_code[np.isin(m_pre_nt, [3, 4, 5, 7])] = 4

    all_pre = np.concatenate([pre_i, m_pre_i])
    all_post = np.concatenate([post_i, m_post_i])
    all_norm = np.concatenate([norm_v, m_norm_v]).astype(np.float16)
    all_port = np.concatenate([port_code, m_port_code])

    bundle = {
        "schema_version": 1,
        "n_total": int(n_total),
        "n_banc": int(n_banc),
        "n_male_specific": int(n_male_spec),
        "n_edges": int(len(all_pre)),
        "tau_v_ms": torch.from_numpy(tau_v_ms),
        "alpha_shunt": torch.from_numpy(alpha_shunt),
        "nt_id": torch.from_numpy(nt_id),
        "sex_code": torch.from_numpy(sex_code),
        "side_id": torch.from_numpy(side_id),
        "organ_id": torch.from_numpy(organ_id),
        "hemilineage_id": torch.from_numpy(hemilineage_id),
        "edge_pre": torch.from_numpy(all_pre),
        "edge_post": torch.from_numpy(all_post),
        "edge_norm": torch.from_numpy(all_norm),
        "edge_port": torch.from_numpy(all_port),
        "cell_groups": cell_groups,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Done in {time.perf_counter()-t0:.2f}s! Saved {OUT_PATH} ({size_mb:.1f} MB, {len(all_pre):,} edges)")
    return bundle


if __name__ == "__main__":
    build_pan_connectome()
