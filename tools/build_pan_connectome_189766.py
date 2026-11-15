#!/usr/bin/env python3
"""Compile the Full 189,766-Cell Drosophila Pan-Connectome v2 (BANC v888 + MaleCNS + Frankenbrain + FCA + Influence).

Integrates all local empirical layers from `D:/Experiments/casa-data/`:
1. `repos/BANC-project/data/meta/banc_888_meta_20260521.parquet` (188,508 cells x 165 cols: known_nt, v3 NT, malecns_nblast, etc.)
2. `banc-v888/banc_888_metrics.feather` (morphometry: l2_cable_length_um, branchpoints, segregation_index)
3. `repos/BANC-project/data/meta/franken_meta_20260522.feather` (163,833 cells x 62 cols: EM soma_dcv_density, cell_dcv_density_um, tracts, serial_motif)
4. `banc-v888-extras/zenodo/drosophila_neuropeptides-v1.1.0.zip` & `drosophila_neurotransmitters-v1.1.0.zip` (ground-truth neuropeptides & NTs)
5. `fca/fca_head_10x_stringent.h5ad`, `fca_body_10x_stringent.h5ad`, `fca_banc_map.csv` (Fly Cell Atlas scRNA-seq ion channels: Ca-alpha1T, GluClalpha, Rdl, Sh/Shal/Shab/Shaw, shakB)
6. `banc-v888/influence/influence_all_to_effector_subclass.parquet` + `banc_888_betweenness_afferent_to_efferent_v3.csv` + `banc_888_cns_network_spectral_clustering_v3.csv`
7. `banc-v888-extras/annotations/v888/somas_v1.parquet` (153,892 3D soma coordinates) & `nblast/banc_malecns_reviewed_matches.csv`
8. `banc-v888/banc_888_edgelist_split_v3.feather` (15,716,248 subcellular split edges)
9. `mcns-v1.0/` + `malecns-v0.9/malecns_09_meta.feather` (+ `malecns_09_split_edgelist.feather` when available)

Outputs `D:/Experiments/casa-data/connectomes/banc-v888-csr-cache/drosophila_pan_connectome_189766.pt` (schema_version=2).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
import zipfile
import h5py
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import pyarrow.parquet as pq
import torch

BANC_DIR = Path(r"D:\Experiments\casa-data\connectomes\banc-v888")
BANC_EXTRAS_DIR = Path(r"D:\Experiments\casa-data\connectomes\banc-v888-extras")
BANC_REPO_META_DIR = Path(r"D:\Experiments\casa-data\connectomes\repos\BANC-project\data\meta")
MCNS_DIR = Path(r"D:\Experiments\casa-data\connectomes\mcns-v1.0")
MALECNS09_DIR = Path(r"D:\Experiments\casa-data\connectomes\malecns-v0.9")
FCA_DIR = Path(r"D:\Experiments\casa-data\fca")
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
    "glycine": 1,  # inhibitory chloride channel
}


def _parse_nt_string(val: object) -> int | None:
    if pd.isna(val):
        return None
    s = str(val).lower().strip()
    if not s or s in ("unclear", "unknown", "none", "nan", "na"):
        return None
    for token in s.replace(";", ",").split(","):
        tok = token.strip()
        if tok in NT_TO_ID:
            return NT_TO_ID[tok]
    return None


def _first_not_none(*vals: int | None) -> int | None:
    for v in vals:
        if v is not None:
            return v
    return None


def _extract_fca_channel_profiles(b_meta: pd.DataFrame, n_total: int) -> dict[str, np.ndarray]:
    """Extract empirical scRNA-seq ion-channel expression from FCA head & body .h5ad files via CSC column slices."""
    ca_alpha1t = np.full(n_total, 0.45, dtype=np.float32)
    glucl_rdl = np.full(n_total, 1.00, dtype=np.float32)
    shaker_k = np.full(n_total, 1.00, dtype=np.float32)
    shakb_gap = np.full(n_total, 0.35, dtype=np.float32)

    h5_head = FCA_DIR / "fca_head_10x_stringent.h5ad"
    if not h5_head.is_file():
        return {
            "fca_ca_alpha1t": ca_alpha1t,
            "fca_glucl_rdl": glucl_rdl,
            "fca_shaker_k": shaker_k,
            "fca_shakb_gap": shakb_gap,
        }

    genes_wanted = ["Ca-alpha1T", "GluClalpha", "Rdl", "Sh", "Shal", "Shab", "Shaw", "shakB"]
    class_profiles: dict[str, dict[str, float]] = {}

    for h5_path in (h5_head, FCA_DIR / "fca_body_10x_stringent.h5ad"):
        if not h5_path.is_file():
            continue
        with h5py.File(h5_path, "r") as f:
            raw_names = [g.decode() if isinstance(g, bytes) else str(g) for g in f["raw.var"]["index"][:]]
            g_map = {name: idx for idx, name in enumerate(raw_names)}
            indptr = f["raw.X"]["indptr"][:]
            ann_codes = f["obs"]["annotation"][:]
            ann_cats = [c.decode() if isinstance(c, bytes) else str(c) for c in f["uns"]["annotation_categories"][:]]
            n_cats = len(ann_cats)
            cat_counts = np.bincount(ann_codes[ann_codes >= 0], minlength=n_cats).clip(min=1)

            gene_vecs: dict[str, np.ndarray] = {}
            for g in genes_wanted:
                gi = g_map.get(g)
                if gi is None:
                    gene_vecs[g] = np.zeros(n_cats, dtype=np.float32)
                    continue
                s, e = int(indptr[gi]), int(indptr[gi + 1])
                rows = f["raw.X"]["indices"][s:e]
                vals = np.log1p(f["raw.X"]["data"][s:e])
                valid = ann_codes[rows] >= 0
                sums = np.bincount(ann_codes[rows[valid]], weights=vals[valid], minlength=n_cats)
                gene_vecs[g] = (sums / cat_counts).astype(np.float32)

            for c_idx, c_name in enumerate(ann_cats):
                key = c_name.lower()
                class_profiles[key] = {
                    "ca": float(gene_vecs["Ca-alpha1T"][c_idx]),
                    "cl": float(gene_vecs["GluClalpha"][c_idx] + gene_vecs["Rdl"][c_idx]),
                    "k": float(
                        0.35
                        * (
                            gene_vecs["Sh"][c_idx]
                            + gene_vecs["Shal"][c_idx]
                            + gene_vecs["Shab"][c_idx]
                            + gene_vecs["Shaw"][c_idx]
                        )
                    ),
                    "gap": float(gene_vecs["shakB"][c_idx]),
                }

    # Map FCA cell classes onto BANC cell_type / cell_class / super_class
    ct_ser = b_meta["cell_type"].fillna("").astype(str)
    cc_ser = b_meta["cell_class"].fillna("").astype(str)
    sc_ser = b_meta["super_class"].fillna("").astype(str)
    n_banc = len(b_meta)

    for fca_name, prof in class_profiles.items():
        # Match exact optic lobe / central / sensory / glial classes from FCA
        token = fca_name.split()[-1] if fca_name else ""
        mask = np.zeros(n_banc, dtype=bool)
        if len(token) >= 2 and token[0].isalpha() and any(ch.isdigit() for ch in token):
            mask = ct_ser.str.lower().str.startswith(token).to_numpy()
        elif "kenyon" in fca_name or "mushroom body" in fca_name:
            mask = (cc_ser == "kenyon_cell").to_numpy()
        elif "projection neuron" in fca_name and "antennal" in fca_name:
            mask = (cc_ser == "antennal_lobe_projection_neuron").to_numpy()
        elif "glial" in fca_name or "astrocyte" in fca_name:
            mask = (sc_ser == "glia").to_numpy()
        elif "motor neuron" in fca_name:
            mask = (sc_ser == "motor").to_numpy()
        elif "photoreceptor" in fca_name:
            mask = ct_ser.str.startswith("R1").to_numpy() | ct_ser.str.startswith("R7").to_numpy() | ct_ser.str.startswith("R8").to_numpy()
        if mask.any():
            ca_alpha1t[:n_banc][mask] = np.clip(prof["ca"], 0.1, 2.5)
            glucl_rdl[:n_banc][mask] = np.clip(0.5 + 0.5 * prof["cl"], 0.5, 2.5)
            shaker_k[:n_banc][mask] = np.clip(0.5 + 0.5 * prof["k"], 0.5, 2.5)
            shakb_gap[:n_banc][mask] = np.clip(prof["gap"], 0.1, 2.0)

    return {
        "fca_ca_alpha1t": ca_alpha1t,
        "fca_glucl_rdl": glucl_rdl,
        "fca_shaker_k": shaker_k,
        "fca_shakb_gap": shakb_gap,
    }


def build_pan_connectome(min_synapse_count: int = 2) -> dict:
    t0 = time.perf_counter()
    print("[1/6] Loading BANC v888 full 165-column master parquet (188,508 cells) & morphometrics...")
    parquet_meta_path = BANC_REPO_META_DIR / "banc_888_meta_20260521.parquet"
    if parquet_meta_path.is_file():
        b_meta = pq.read_table(
            parquet_meta_path,
            columns=[
                "banc_888_id",
                "super_class",
                "cell_class",
                "cell_sub_class",
                "cell_type",
                "fafb_cell_type",
                "manc_cell_type",
                "malecns_cell_type",
                "malecns_match",
                "malecns_png_match",
                "malecns_nblast",
                "side",
                "hemilineage",
                "neuromere",
                "sexually_dimorphic",
                "super_cluster",
                "cluster",
                "cns_network",
                "cns_cluster",
                "body_part_sensory",
                "body_part_effector",
                "peripheral_target_type",
                "cell_function",
                "known_nt",
                "neurotransmitter",
                "neurotransmitter_predicted",
                "neurotransmitter_predicted_v3",
                "neuropeptide_verified",
                "l2_cable_length_um",
                "pd_width",
                "segregation_index",
                "an_type",
                "dn_type",
            ],
        ).to_pandas()
    else:
        b_meta = pf.read_table(BANC_DIR / "banc_888_meta.feather").to_pandas()

    b_metrics = pf.read_table(
        BANC_DIR / "banc_888_metrics.feather",
        columns=["banc_888_id", "branchpoints", "mitochondria", "mitochondria_volume", "axon_length", "dend_length"],
    ).to_pandas()
    b_meta = b_meta.merge(b_metrics, on="banc_888_id", how="left")

    n_banc = len(b_meta)
    assert n_banc == 188508

    print("[2/6] Loading MCNS v1.0 + harmonized malecns_09_meta (1,258 male-specific cells) & Lacin/Bates hemilineages...")
    m_meta = pf.read_table(
        MCNS_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        columns=["bodyId", "type", "superclass", "class", "subclass", "dimorphism", "fruDsx", "somaSide", "itoleeHl"],
    ).to_pandas()
    m_nt = pf.read_table(
        MCNS_DIR / "body-neurotransmitters-male-cns-v1.0.feather",
        columns=["body", "consensus_nt", "predicted_nt"],
    ).to_pandas()
    m_meta = m_meta.merge(m_nt, left_on="bodyId", right_on="body", how="left")

    # Join harmonized malecns_09_meta when available for refined hemilineage & NT calls
    m09_path = MALECNS09_DIR / "malecns_09_meta.feather"
    if m09_path.is_file():
        m09 = pf.read_table(
            m09_path,
            columns=["malecns_09_id", "hemilineage", "neurotransmitter_predicted", "neuromere", "tract"],
        ).to_pandas()
        m09["bodyId"] = m09["malecns_09_id"].astype("int64")
        m_meta = m_meta.merge(
            m09[["bodyId", "hemilineage", "neurotransmitter_predicted", "neuromere", "tract"]],
            on="bodyId",
            how="left",
            suffixes=("", "_09"),
        )
        m_meta["itoleeHl"] = m_meta["hemilineage"].combine_first(m_meta["itoleeHl"])

    ms_df = m_meta[m_meta["dimorphism"] == "male-specific"].reset_index(drop=True)
    n_male_spec = len(ms_df)
    n_total = n_banc + n_male_spec
    print(f"      Unified Pan-Connectome size: {n_banc:,} (BANC) + {n_male_spec:,} (MCNS male-specific) = {n_total:,} cells")

    # --- Per-Cell Biophysical, Transcriptomic & Genetic Vectors [189,766] ---
    # 1. Cable length -> tau_v_ms = 1.711 * (L2 / 241.53)
    l2_banc = b_meta["l2_cable_length_um"].fillna(450.0).to_numpy(dtype=np.float32)
    l2_male = np.full(n_male_spec, 520.0, dtype=np.float32)
    l2_male[ms_df["superclass"].fillna("").isin(["descending_neuron", "ascending_neuron"]).to_numpy()] = 241.53
    l2_all = np.concatenate([l2_banc, l2_male])
    tau_v_ms = np.clip(1.711 * (l2_all / 241.53), 1.0, 25.0).astype(np.float32)

    # 2. Branchpoints & segregation_index -> dendritic shunting gain alpha_shunt
    bp_banc = b_meta["branchpoints"].fillna(28.0).to_numpy(dtype=np.float32)
    bp_male = np.full(n_male_spec, 45.0, dtype=np.float32)
    bp_all = np.concatenate([bp_banc, bp_male])
    alpha_shunt = np.clip(np.log1p(bp_all / 28.0) * 2.5, 0.8, 6.0).astype(np.float32)

    # 3. Hemilineage integer index & Lacin/Bates Hemilineage modal neurotransmitter map
    hl_banc = b_meta["hemilineage"].fillna("unassigned").astype(str)
    hl_male = ms_df["itoleeHl"].fillna("male_hl").astype(str)
    hl_all = pd.concat([hl_banc, hl_male], ignore_index=True)
    hl_codes, _ = pd.factorize(hl_all)
    hemilineage_id = hl_codes.astype(np.int16)

    # Compute empirical Lacin/Bates modal neurotransmitter per hemilineage across BANC + MaleCNS
    hl_nt_counts: dict[str, Counter] = defaultdict(lambda: defaultdict(int))
    for hl_name, nt_s in zip(hl_banc, b_meta["neurotransmitter_predicted"]):
        if hl_name not in ("unassigned", "", "na", "nan"):
            nid = _parse_nt_string(nt_s)
            if nid is not None and nid in (0, 1, 2):
                hl_nt_counts[hl_name][nid] += 1
    for hl_name, nt_s in zip( m_meta["itoleeHl"].fillna(""), m_meta["consensus_nt"]):
        hl_s = str(hl_name)
        if hl_s and hl_s not in ("male_hl", "na", "nan"):
            nid = _parse_nt_string(nt_s)
            if nid is not None and nid in (0, 1, 2):
                hl_nt_counts[hl_s][nid] += 1
    hl_modal_nt: dict[str, int] = {
        hl_k: max(counts.items(), key=lambda kv: kv[1])[0]
        for hl_k, counts in hl_nt_counts.items()
        if counts
    }

    # 4. Neurotransmitter ID (0..8) with priority: known_nt > Franken verified/top_nt > BANC curated > v3 > Lacin/Bates hemilineage
    franken_nt_by_bid: dict[str, int] = {}
    franken_nt_by_ct: dict[str, int] = {}
    franken_path = BANC_REPO_META_DIR / "franken_meta_20260522.feather"
    if franken_path.is_file():
        fr_nt_df = pf.read_table(
            franken_path,
            columns=["banc_id", "cell_type", "neurotransmitter_verified", "top_nt"],
        ).to_pandas()
        for b_id_s, c_t_s, n_ver, t_nt in zip(
            fr_nt_df["banc_id"], fr_nt_df["cell_type"], fr_nt_df["neurotransmitter_verified"], fr_nt_df["top_nt"]
        ):
            nid_f = _first_not_none(_parse_nt_string(n_ver), _parse_nt_string(t_nt))
            if nid_f is not None:
                if pd.notna(b_id_s):
                    franken_nt_by_bid[str(b_id_s)] = nid_f
                if pd.notna(c_t_s) and str(c_t_s) not in franken_nt_by_ct:
                    franken_nt_by_ct[str(c_t_s)] = nid_f

    nt_id = np.full(n_total, 8, dtype=np.uint8)
    cotrans_no_mask = np.zeros(n_total, dtype=np.uint8)
    k_nt = b_meta.get("known_nt", pd.Series([None] * n_banc))
    c_nt = b_meta.get("neurotransmitter", pd.Series([None] * n_banc))
    v3_nt = b_meta.get("neurotransmitter_predicted_v3", pd.Series([None] * n_banc))
    p_nt = b_meta["neurotransmitter_predicted"]
    b_ids_str = b_meta["banc_888_id"].astype(str)
    b_cts_str = b_meta["cell_type"].fillna("").astype(str)

    for i in range(n_banc):
        raw_c = str(c_nt.iat[i]) if pd.notna(c_nt.iat[i]) else ""
        if "nitric_oxide" in raw_c:
            cotrans_no_mask[i] = 1
        b_ct_i = b_cts_str.iat[i]
        nid = _first_not_none(
            _parse_nt_string(k_nt.iat[i]),
            franken_nt_by_bid.get(b_ids_str.iat[i]),
            franken_nt_by_ct.get(b_ct_i) if b_ct_i else None,
            _parse_nt_string(c_nt.iat[i]),
            _parse_nt_string(v3_nt.iat[i]),
            _parse_nt_string(p_nt.iat[i]),
        )
        if nid is None:
            nid = hl_modal_nt.get(str(hl_banc.iat[i]), 8)
        nt_id[i] = nid

    sc = b_meta["super_class"].fillna("").astype(str)
    cc = b_meta["cell_class"].fillna("").astype(str)
    cn = b_meta["cns_network"].fillna("").astype(str)
    cf = b_meta["cell_function"].fillna("").astype(str)
    is_sens_banc = sc.isin(["sensory", "sensory_ascending", "sensory_descending"]).to_numpy()
    is_photo_banc = cf.str.startswith("visual").to_numpy()
    # Peripheral sensory neurons (ORN, GRN, JON, campaniform, chordotonal, bristle) in Drosophila are cholinergic (ACh=0),
    # whereas R1-R6 photoreceptors are histaminergic (His=6).
    nt_id[:n_banc][is_sens_banc & (~is_photo_banc)] = 0

    m_09_nt = ms_df.get("neurotransmitter_predicted", pd.Series([None] * n_male_spec))
    for k in range(n_male_spec):
        nid = _first_not_none(
            _parse_nt_string(m_09_nt.iat[k]),
            _parse_nt_string(ms_df["consensus_nt"].iat[k]),
            _parse_nt_string(ms_df["predicted_nt"].iat[k]),
        )
        if nid is None:
            # Lacin/Bates Hemilineage Law resolves unclear cells (e.g. VLPl4_dorsal AVLP743m -> GABA, CREa1_ventral mAL -> GABA)
            hl_k = str(hl_male.iat[k])
            nid = hl_modal_nt.get(hl_k, 1 if str(ms_df["type"].iat[k]).startswith("mAL") else 0)
        nt_id[n_banc + k] = nid

    # 5. Dense-Core Vesicle (DCV) & Neuropeptide Gain [189,766] from Frankenbrain EM + Zenodo 20818144 ground-truth
    print("[3/6] Extracting EM Dense-Core Vesicle (DCV) density, Zenodo neuropeptides, FCA scRNA-seq channels & Influence...")
    dcv_peptidergic_gain = np.full(n_total, 0.20, dtype=np.float32)
    np_ver = b_meta["neuropeptide_verified"].fillna("").astype(str)
    dcv_peptidergic_gain[:n_banc][np_ver != ""] = 1.25

    franken_path = BANC_REPO_META_DIR / "franken_meta_20260522.feather"
    if franken_path.is_file():
        fr = pf.read_table(
            franken_path,
            columns=["banc_id", "cell_type", "fafb_cell_type", "manc_cell_type", "soma_dcv_density", "cell_dcv_density_um", "neuropeptide_verified"],
        ).to_pandas()
        fr_valid = fr[(fr["cell_dcv_density_um"] > 0) | (fr["soma_dcv_density"] > 0)].copy()
        fr_valid["dcv_score"] = np.clip(
            np.maximum(fr_valid["cell_dcv_density_um"].clip(lower=0.0), fr_valid["soma_dcv_density"].clip(lower=0.0)),
            0.05,
            2.5,
        )
        bid_to_dcv = dict(zip(fr_valid["banc_id"].dropna().astype(str), fr_valid["dcv_score"]))
        ct_to_dcv = fr_valid.groupby("cell_type")["dcv_score"].mean().to_dict()
        for i, (bid, ct_i) in enumerate(zip(b_meta["banc_888_id"].astype(str), b_meta["cell_type"].fillna(""))):
            val = bid_to_dcv.get(bid) or ct_to_dcv.get(ct_i)
            if val is not None:
                dcv_peptidergic_gain[i] = max(dcv_peptidergic_gain[i], float(val))

    np_zip = BANC_EXTRAS_DIR / "zenodo" / "drosophila_neuropeptides-v1.1.0.zip"
    if np_zip.is_file():
        with zipfile.ZipFile(np_zip) as zf:
            np_df = pd.read_csv(zf.open("flyconnectome-drosophila_neuropeptides-63e5ac6/gt_np_data.csv"))
            np_types = set(np_df["cell_type"].dropna().astype(str))
            has_gt_np = b_meta["cell_type"].fillna("").isin(np_types).to_numpy()
            dcv_peptidergic_gain[:n_banc][has_gt_np] = np.maximum(dcv_peptidergic_gain[:n_banc][has_gt_np], 1.50)

    # 6. Fly Cell Atlas scRNA-seq Ion Channel & Receptor Profiles
    fca_profiles = _extract_fca_channel_profiles(b_meta, n_total)
    alpha_shunt = (alpha_shunt * fca_profiles["fca_glucl_rdl"]).astype(np.float32)

    # 7. Causal Effector Influence & Afferent-to-Efferent Betweenness
    effector_influence = np.zeros(n_total, dtype=np.float32)
    b_idx = pd.Index(b_meta["banc_888_id"].astype(str))
    inf_path = BANC_DIR / "influence" / "influence_all_to_effector_subclass.parquet"
    if inf_path.is_file():
        inf_df = pq.read_table(inf_path, columns=["source", "influence"]).to_pandas()
        inf_sum = inf_df.groupby("source")["influence"].sum()
        matched_idx = b_idx.get_indexer(inf_sum.index.astype(str))
        valid_m = matched_idx >= 0
        effector_influence[matched_idx[valid_m]] = np.log1p(inf_sum.to_numpy(dtype=np.float32)[valid_m])

    betweenness_aff_eff = np.zeros(n_total, dtype=np.float32)
    betw_path = BANC_DIR / "banc_888_betweenness_afferent_to_efferent_v3.csv"
    if betw_path.is_file():
        bw_df = pd.read_csv(betw_path)
        if "banc_888_id" in bw_df.columns and "betweenness" in bw_df.columns:
            m_idx = b_idx.get_indexer(bw_df["banc_888_id"].astype(str))
            v_m = m_idx >= 0
            betweenness_aff_eff[m_idx[v_m]] = np.log1p(bw_df["betweenness"].fillna(0.0).to_numpy(dtype=np.float32)[v_m])

    # 8. Sex specificity code: 0=isomorphic, 1=shared_dimorphic, 2=female_specific, 3=male_specific
    sex_code = np.zeros(n_total, dtype=np.uint8)
    sd_col = b_meta["sexually_dimorphic"].fillna("isomorphic").astype(str)
    sex_code[:n_banc][sd_col == "dimorphic"] = 1
    sex_code[:n_banc][sd_col == "female-specific"] = 2
    sex_code[:n_banc][sd_col == "male-specific"] = 3
    sex_code[n_banc:] = 3

    # 9. Side code: 0=midline/na, 1=left, 2=right
    side_id = np.zeros(n_total, dtype=np.uint8)
    b_side = b_meta["side"].fillna("").astype(str).str.lower()
    side_id[:n_banc][b_side == "left"] = 1
    side_id[:n_banc][b_side == "right"] = 2
    m_side = ms_df["somaSide"].fillna("").astype(str).str.upper()
    side_id[n_banc:][m_side.isin(["L", "LEFT"])] = 1
    side_id[n_banc:][m_side.isin(["R", "RIGHT"])] = 2

    # 10. Organ ID: 0=OL, 1=AL, 2=LH_SPIN, 3=MB, 4=CX, 5=AN_DN, 6=VNC, 7=SEZ, 8=GLIA
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

    # --- Step 4: Build Complete Sensory, Circuit, Dimorphic, and Effector Cell Index Maps ---
    print("[4/6] Extracting exact sensory (16,563), circuit, dimorphic (P1/mAL/SIP1/AN09B017), and effector groups...")
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

    is_sens = sc.isin(["sensory", "sensory_ascending", "sensory_descending"])
    mc_clean = b_meta["malecns_cell_type"].fillna("").astype(str).str.removeprefix("auto:")
    is_leg_gust = (cf == "gustatory") | ct.str.startswith("LgLG") | mc_clean.str.startswith("LgLG")
    # Kallman, Kim & Scott (2015) Nat Neurosci / MCNS v1.0 connectivity:
    # - LgLG5,6,7,8 -> AN09B017 (vAB3, ACh -> P1) and LgLG1a,2 -> AN05B102 (PPN1, ACh -> P1) are female 7,11-HD F-cells
    # - LgLG3,4 -> AN05B023 / AN13B002 (GABA -| P1) are male 7-tricosene/cVA M-cells
    f_711hd_mask = ct.isin(["LgLG5", "LgLG6", "LgLG7", "LgLG8", "LgLG1a", "LgLG2"]) | mc_clean.isin(
        ["LgLG5", "LgLG6", "LgLG7", "LgLG8", "LgLG1a", "LgLG2"]
    )
    m_cva_mask = (ct.isin(["LgLG3", "LgLG4"]) | mc_clean.isin(["LgLG3", "LgLG4"])) & (~f_711hd_mask)

    cell_groups: dict[str, torch.Tensor] = {
        # Sensory Periphery (strictly afferent sensory / sensory_ascending / sensory_descending cells)
        "sens_visual_left": _idx(is_sens & cf.str.startswith("visual") & (b_side == "left")),
        "sens_visual_right": _idx(is_sens & cf.str.startswith("visual") & (b_side == "right")),
        "sens_ocellar": _idx(cf == "visual_ocellar"),
        "sens_orn_all": _idx(cc == "olfactory_receptor_neuron"),
        "sens_orn_left": _idx((cc == "olfactory_receptor_neuron") & (b_side == "left")),
        "sens_orn_right": _idx((cc == "olfactory_receptor_neuron") & (b_side == "right")),
        "sens_orn_DA1_cva": _idx(ct == "ORN_DA1"),
        "sens_orn_VA1v_mate": _idx(ct == "ORN_VA1v"),
        "sens_orn_DM1_food": _idx(ct.isin(["ORN_DM1", "ORN_DM4", "ORN_DM2"])),
        "sens_orn_V_co2": _idx(ct.isin(["ORN_V", "ORN_DA2", "ORN_DL4"])),
        "sens_jo_left": _idx(is_sens & (cf == "auditory") & (b_side == "left")),
        "sens_jo_right": _idx(is_sens & (cf == "auditory") & (b_side == "right")),
        "sens_gust_labellum": _idx(is_sens & (cf == "gustatory") & (bps == "labellum")),
        "sens_gust_front_leg": _idx(is_sens & is_leg_gust & (bps == "front_leg")),
        "sens_gust_front_leg_female_711hd": _idx(is_sens & is_leg_gust & (bps == "front_leg") & f_711hd_mask),
        "sens_gust_front_leg_male_cva": _idx(is_sens & is_leg_gust & (bps == "front_leg") & m_cva_mask),
        "sens_gust_middle_leg": _idx(is_sens & (cf == "gustatory") & (bps == "middle_leg")),
        "sens_gust_hind_leg": _idx(is_sens & (cf == "gustatory") & (bps == "hind_leg")),
        "sens_gust_pharynx": _idx(is_sens & (cf == "gustatory") & bps.isin(["pharynx", "anterior_digestive_tract"])),
        "sens_campaniform_all": _idx(is_sens & (cc == "campaniform_sensillum_neuron")),
        "sens_chordotonal_all": _idx(is_sens & (cc == "chordotonal_organ_neuron")),
        "sens_bristle_all": _idx(is_sens & (cc == "bristle_neuron")),
        "sens_haltere_left": _idx(is_sens & (bps == "haltere") & (b_side == "left")),
        "sens_haltere_right": _idx(is_sens & (bps == "haltere") & (b_side == "right")),
        "sens_crop_visceral": _idx(is_sens & (csc == "anterior_digestive_tract_multidendritic_neuron")),
        "sens_reproductive_tract": _idx(is_sens & (cc == "reproductive_tract_sensory_neuron")),
        "sens_thermo": _idx(is_sens & (cf == "thermosensory")),
        "sens_hygro": _idx(is_sens & (cf == "hygrosensory")),
        "sens_nociception": _idx(is_sens & (cf == "nociception")),
        # Per-leg mechanoreceptors (L1..R3, strictly excluding all LgLG/gustatory chemoreceptors)
        "sens_leg_L1": _idx(is_sens & (bps == "front_leg") & (~is_leg_gust) & (b_side == "left")),
        "sens_leg_R1": _idx(is_sens & (bps == "front_leg") & (~is_leg_gust) & (b_side == "right")),
        "sens_leg_L2": _idx(is_sens & (bps == "middle_leg") & (~is_leg_gust) & (b_side == "left")),
        "sens_leg_R2": _idx(is_sens & (bps == "middle_leg") & (~is_leg_gust) & (b_side == "right")),
        "sens_leg_L3": _idx(is_sens & (bps == "hind_leg") & (~is_leg_gust) & (b_side == "left")),
        "sens_leg_R3": _idx(is_sens & (bps == "hind_leg") & (~is_leg_gust) & (b_side == "right")),
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
        # Note: Pure P1 is the 126 ACh+ pC1_1a..19 cluster; SIP100m..146m (186 cells) are GluCla- local inhibitory interneurons!
        "dimorphic_female_pc1": _idx(ct.str.startswith("pC1")),
        "dimorphic_female_asp_f": _idx(ct.str.startswith("aSP-f")),
        "dimorphic_female_asp_g": _idx(ct.str.startswith("aSP-g")),
        "dimorphic_female_vpo_ovi": _idx(ct.str.contains("vpo|oviDN|oviIN", case=False)),
        "dimorphic_male_p1": _m_idx(m_type.str.startswith("pC1_")),
        "dimorphic_male_sip1_inh": _m_idx(m_type.str.startswith("SIP1")),
        "dimorphic_male_mal": _m_idx(m_type.str.startswith("mAL_")),
        "dimorphic_male_an09b017": _m_idx(m_type.str.startswith("AN09B017")),
        "dimorphic_male_pip10": _m_idx(m_type.isin(["pIP10", "pIP1", "pMP2"])),
        "dimorphic_male_tn1_song": _m_idx(m_type.str.startswith("TN1") | m_type.str.startswith("vMS") | m_type.str.startswith("IN")),
        "dimorphic_male_fru_only": _m_idx(~m_type.str.startswith("pC1_")),
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
        "eff_oviposition_uterus": _idx((sc == "motor") & (bpe == "uterus")),
        "eff_endocrine_ipc_akh": _idx(csc.isin(["retrocerebral_complex_neurosecretory_cell", "corpus_allatum_neurosecretory_cell"])),
        "wave1_primary": _idx(hl_banc.isin(["putative_primary", "primary"])),
    }

    # Per-leg motor pools (L1..R3)
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

    orn_types = sorted(b_meta.loc[cc == "olfactory_receptor_neuron", "cell_type"].dropna().unique())
    for g_idx, orn_t in enumerate(orn_types[:50]):
        cell_groups[f"orn_glom_{g_idx:02d}"] = _idx(ct == orn_t)

    # --- Step 5: Extract 5-Port Subcellular Synaptic Edges from BANC v888 + MaleCNS ---
    print(f"[5/6] Loading BANC v888 split edgelist (filtering count >= {min_synapse_count})...")
    sp = pf.read_table(
        BANC_DIR / "banc_888_edgelist_split_v3.feather",
        columns=["pre", "post", "pre_label", "post_label", "count", "norm", "pre_conf_nt"],
    ).to_pandas()
    sp = sp[sp["count"] >= min_synapse_count].reset_index(drop=True)

    b_id_idx = pd.Index(b_meta["banc_888_id"])
    pre_i = b_id_idx.get_indexer(sp["pre"]).astype(np.int32)
    post_i = b_id_idx.get_indexer(sp["post"]).astype(np.int32)
    norm_v = sp["norm"].to_numpy(dtype=np.float32)
    post_lbl = sp["post_label"].fillna("dendrite").astype(str)
    b_pre_nt = nt_id[pre_i]

    # Classify every edge into one of the 5 biological ports using curated/verified nt_id[pre_i]:
    # 0 = Port 1 (dendritic excitatory ACh+)
    # 1 = Port 2a (dendritic chloride shunting GABA / Glu / His)
    # 2 = Port 2b (soma / primary-dendrite root shunt)
    # 3 = Port 3 (axo-axonic presynaptic gate: +ACh facilitation, -GABA/Glu presynaptic inhibition)
    # 4 = Port 5 (modulatory DA / OA / 5-HT / TA)
    port_code = np.zeros(len(sp), dtype=np.uint8)
    is_mod = np.isin(b_pre_nt, [3, 4, 5, 7])
    is_inh = np.isin(b_pre_nt, [1, 2, 6])
    is_axon_post = (post_lbl == "axon").to_numpy()
    is_root_post = post_lbl.isin(["primary.dendrite", "primary.neurite", "soma"]).to_numpy()

    port_code[is_inh & (~is_axon_post) & (~is_root_post)] = 1
    port_code[is_root_post & is_inh] = 2
    port_code[is_axon_post & (~is_mod)] = 3
    port_code[is_mod] = 4
    # Signed Port 3: GABA/Glu/His axo-axonic synapses perform presynaptic inhibition (-norm), ACh performs facilitation (+norm)
    norm_v[is_axon_post & (~is_mod) & is_inh] *= -1.0

    # --- Step 6: Cross-Atlas NBLAST + Round-Robin Matching & MaleCNS 5-Port Split Synaptic Edges ---
    print("[6/6] Merging MaleCNS 5-port compartment-split edges with 150k NBLAST + reviewed matches...")
    body_to_pan: dict[int, int] = {int(b): n_banc + i for i, b in enumerate(ms_df["bodyId"])}

    # 1. Reviewed matches from banc_malecns_reviewed_matches.csv
    rev_path = BANC_EXTRAS_DIR / "nblast" / "banc_malecns_reviewed_matches.csv"
    if rev_path.is_file():
        rev = pd.read_csv(rev_path)
        rev_valid = rev[rev["valid"].astype(str).str.lower().isin(["t", "true", "1"])]
        root_idx = b_id_idx.get_indexer(rev_valid["pt_root_id"].astype(str))
        for r_i, q_id in zip(root_idx, rev_valid["query_id"]):
            if r_i >= 0 and pd.notna(q_id) and str(q_id).isdigit():
                bid = int(q_id)
                if bid not in body_to_pan:
                    body_to_pan[bid] = int(r_i)

    # 2. Direct malecns_match & malecns_png_match in BANC v888 master table
    for col_name in ("malecns_match", "malecns_png_match"):
        if col_name in b_meta.columns:
            for i, m_id in enumerate(b_meta[col_name]):
                if pd.notna(m_id) and str(m_id).isdigit():
                    bid = int(m_id)
                    if bid not in body_to_pan:
                        body_to_pan[bid] = i

    # 3. Round-robin per (cell_type, side) — stripping "auto:" prefix (preserves exact cell-type & side identity!)
    type_side_to_banc_list: dict[tuple[str, str], list[int]] = defaultdict(list)
    mc_col = b_meta["malecns_cell_type"].fillna("").astype(str).str.removeprefix("auto:")
    for i, (c_t, mc_t, s_val) in enumerate(zip(ct, mc_col, b_side)):
        s_key = s_val if s_val in ("left", "right") else "any"
        if mc_t and mc_t not in ("nan", "None"):
            type_side_to_banc_list[(mc_t, s_key)].append(i)
            type_side_to_banc_list[(mc_t, "any")].append(i)
        if c_t and c_t != mc_t:
            type_side_to_banc_list[(c_t, s_key)].append(i)
            type_side_to_banc_list[(c_t, "any")].append(i)

    rr_counters: dict[tuple[str, str], int] = defaultdict(int)
    for b, t, s in zip(m_meta["bodyId"], m_meta["type"], m_meta["somaSide"]):
        bid = int(b)
        if bid not in body_to_pan and pd.notna(t):
            t_clean = str(t).removeprefix("auto:")
            s_str = "left" if s == "L" else ("right" if s == "R" else "any")
            key = (t_clean, s_str)
            candidates = type_side_to_banc_list.get(key)
            if not candidates:
                key = (t_clean, "any")
                candidates = type_side_to_banc_list.get(key)
            if candidates:
                idx_pos = rr_counters[key] % len(candidates)
                rr_counters[key] += 1
                body_to_pan[bid] = candidates[idx_pos]

    # 4. NBLAST cross-atlas similarity fallback for untyped neurons (excluding score>=0.99 small-fragment artifacts)
    nbl_path = BANC_EXTRAS_DIR / "nblast" / "banc_malecns_v0.9_nblast.feather"
    if nbl_path.is_file():
        import pyarrow.compute as pc

        nbl_tbl = pf.read_table(nbl_path, columns=["root_888", "match_id", "score"])
        nbl_df = nbl_tbl.filter(
            pc.and_(pc.greater_equal(nbl_tbl["score"], 0.45), pc.less(nbl_tbl["score"], 0.98))
        ).to_pandas()
        nbl_best = nbl_df.sort_values("score", ascending=False).drop_duplicates("match_id")
        r888_idx = b_id_idx.get_indexer(nbl_best["root_888"].astype(str))
        for r_i, m_id in zip(r888_idx, nbl_best["match_id"]):
            if r_i >= 0 and pd.notna(m_id) and str(m_id).isdigit():
                bid = int(m_id)
                if bid not in body_to_pan:
                    body_to_pan[bid] = int(r_i)

    # 5. Extract compartment-split MaleCNS edges from malecns_09_split_edgelist.feather (or fallback to flat weights)
    msplit_path = MALECNS09_DIR / "malecns_09_split_edgelist.feather"
    if msplit_path.is_file():
        import pyarrow as pa
        import pyarrow.compute as pc

        dim_str_ids = m_meta[m_meta["dimorphism"].notna()]["bodyId"].astype(str).tolist()
        ms_id_arr = pa.array(dim_str_ids)
        msplit_tbl = pf.read_table(msplit_path, columns=["pre", "post", "post_label", "count"])
        mask = pc.and_(
            pc.greater_equal(msplit_tbl["count"], 5),
            pc.or_(pc.is_in(msplit_tbl["pre"], value_set=ms_id_arr), pc.is_in(msplit_tbl["post"], value_set=ms_id_arr)),
        )
        w_ms = msplit_tbl.filter(mask).to_pandas()
        w_ms["body_pre"] = w_ms["pre"].astype(np.int64)
        w_ms["body_post"] = w_ms["post"].astype(np.int64)
        w_ms["pre_pan"] = w_ms["body_pre"].map(body_to_pan)
        w_ms["post_pan"] = w_ms["body_post"].map(body_to_pan)
        w_ms = w_ms.dropna(subset=["pre_pan", "post_pan"])
        w_ms = w_ms[w_ms["pre_pan"] != w_ms["post_pan"]]
        w_ms = w_ms.groupby(["pre_pan", "post_pan", "post_label"], as_index=False)["count"].sum()
        w_ms.rename(columns={"count": "weight"}, inplace=True)
    else:
        ms_body_set = set(ms_df["bodyId"].astype(np.int64))
        w_mcns = pf.read_table(MCNS_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather").to_pandas()
        w_ms = w_mcns[(w_mcns["weight"] >= 5) & (w_mcns["body_pre"].isin(ms_body_set) | w_mcns["body_post"].isin(ms_body_set))].copy()
        w_ms["pre_pan"] = w_ms["body_pre"].map(body_to_pan)
        w_ms["post_pan"] = w_ms["body_post"].map(body_to_pan)
        w_ms = w_ms.dropna(subset=["pre_pan", "post_pan"])
        w_ms = w_ms[w_ms["pre_pan"] != w_ms["post_pan"]]
        w_ms["post_label"] = "dendrite"
        w_ms = w_ms.groupby(["pre_pan", "post_pan", "post_label"], as_index=False)["weight"].sum()

    m_pre_i = w_ms["pre_pan"].to_numpy(dtype=np.int32)
    m_post_i = w_ms["post_pan"].to_numpy(dtype=np.int32)
    m_post_lbl = w_ms["post_label"].fillna("dendrite").astype(str)
    m_pre_nt = nt_id[m_pre_i]
    m_is_mod = np.isin(m_pre_nt, [3, 4, 5, 7])
    m_is_inh = np.isin(m_pre_nt, [1, 2, 6])
    m_is_axon_post = (m_post_lbl == "axon").to_numpy()
    p1_pan_set = cell_groups["dimorphic_male_p1"].numpy()
    mal_lh4_set = np.concatenate(
        [
            cell_groups["dimorphic_male_mal"].numpy(),
            (n_banc + np.flatnonzero(m_type.isin(["LH004m", "LH007m"]).to_numpy())).astype(np.int32),
        ]
    )
    m_is_p1_root_shunt = np.isin(m_post_i, p1_pan_set) & np.isin(m_pre_i, mal_lh4_set) & (~m_is_axon_post)
    m_is_root_post = (
        m_post_lbl.isin(["primary_dendrite", "primary_neurite", "primary.dendrite", "primary.neurite", "soma"]).to_numpy()
        | m_is_p1_root_shunt
    )

    m_port_code = np.zeros(len(w_ms), dtype=np.uint8)
    m_port_code[m_is_inh & (~m_is_axon_post) & (~m_is_root_post)] = 1
    m_port_code[m_is_root_post & m_is_inh] = 2
    m_port_code[m_is_axon_post & (~m_is_mod)] = 3
    m_port_code[m_is_mod] = 4

    post_tot = w_ms.groupby("post_pan")["weight"].transform("sum").clip(lower=10.0)
    m_norm_v = np.clip(w_ms["weight"].to_numpy(dtype=np.float32) / post_tot.to_numpy(dtype=np.float32), 0.0005, 0.50)
    m_norm_v[m_is_axon_post & (~m_is_mod) & m_is_inh] *= -1.0

    # Mitochondrial ATP capacity [189,766] from BANC v888 metrics
    mito_cnt = b_meta.get("mitochondria", pd.Series(np.zeros(n_banc))).fillna(0.0).to_numpy(dtype=np.float32)
    mito_atp_capacity = np.ones(n_total, dtype=np.float32)
    mito_atp_capacity[:n_banc] = np.clip(0.6 + 0.25 * np.log1p(mito_cnt), 0.6, 2.2)

    # 3D Soma Coordinates [189,766, 3] in micrometers from v888/somas_v1.parquet
    soma_xyz_um = np.zeros((n_total, 3), dtype=np.float32)
    somas_path = BANC_EXTRAS_DIR / "annotations" / "v888" / "somas_v1.parquet"
    if somas_path.is_file():
        somas_df = pq.read_table(somas_path, columns=["pt_root_id", "pt_position"]).to_pandas()
        s_idx = b_id_idx.get_indexer(somas_df["pt_root_id"].astype(str))
        valid_s = s_idx >= 0
        pos_vox = np.vstack(somas_df["pt_position"].to_numpy())[valid_s].astype(np.float32)
        soma_xyz_um[s_idx[valid_s]] = pos_vox * np.array([0.004, 0.004, 0.045], dtype=np.float32)

    # Bilateral Left<->Right Mirror Homolog Index [189,766] from banc_mirror_nblast.feather
    mirror_partner_idx = np.arange(n_total, dtype=np.int32)
    mir_path = BANC_EXTRAS_DIR / "nblast" / "banc_mirror_nblast.feather"
    if mir_path.is_file():
        import pyarrow.compute as pc

        mir_tbl = pf.read_table(mir_path, columns=["root_888", "match_root_888", "score"])
        mir_df = mir_tbl.filter(
            pc.and_(pc.greater_equal(mir_tbl["score"], 0.45), pc.less(mir_tbl["score"], 0.99))
        ).to_pandas()
        mir_best = mir_df.sort_values("score", ascending=False).drop_duplicates("root_888")
        q_idx = b_id_idx.get_indexer(mir_best["root_888"].astype(str))
        m_idx_b = b_id_idx.get_indexer(mir_best["match_root_888"].astype(str))
        valid_mir = (q_idx >= 0) & (m_idx_b >= 0)
        mirror_partner_idx[q_idx[valid_mir]] = m_idx_b[valid_mir].astype(np.int32)

    all_pre = np.concatenate([pre_i, m_pre_i])
    all_post = np.concatenate([post_i, m_post_i])
    all_norm = np.concatenate([norm_v, m_norm_v]).astype(np.float16)
    all_port = np.concatenate([port_code, m_port_code])

    bundle = {
        "n_total": int(n_total),
        "n_banc": int(n_banc),
        "n_male_specific": int(n_male_spec),
        "n_edges": int(len(all_pre)),
        "tau_v_ms": torch.from_numpy(tau_v_ms),
        "alpha_shunt": torch.from_numpy(alpha_shunt),
        "dcv_peptidergic_gain": torch.from_numpy(dcv_peptidergic_gain),
        "mito_atp_capacity": torch.from_numpy(mito_atp_capacity),
        "soma_xyz_um": torch.from_numpy(soma_xyz_um),
        "mirror_partner_idx": torch.from_numpy(mirror_partner_idx),
        "fca_ca_alpha1t": torch.from_numpy(fca_profiles["fca_ca_alpha1t"]),
        "fca_glucl_rdl": torch.from_numpy(fca_profiles["fca_glucl_rdl"]),
        "fca_shaker_k": torch.from_numpy(fca_profiles["fca_shaker_k"]),
        "fca_shakb_gap": torch.from_numpy(fca_profiles["fca_shakb_gap"]),
        "effector_influence": torch.from_numpy(effector_influence),
        "betweenness_aff_eff": torch.from_numpy(betweenness_aff_eff),
        "cotrans_no_mask": torch.from_numpy(cotrans_no_mask),
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
