"""L1 Two-Wave Morphogenesis: Wave 1 (Larva ~3,016 core) -> 20E Ecdysone -> Wave 2 (Adult ~188,508)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import torch

from fly_nca.core.cell_state import CellPhysiologyParams, init_cell_state
from fly_nca.genome.banc_biophysics import BANC_EMPIRICAL_BIOPHYSICS, BANC_ORGAN_INPUT_MATRIX
from fly_nca.genome.cam_adhesion import ExpressedCellSurface, express_cam_barcodes
from fly_nca.genome.fly_genome import NT_SIGN_MAP, SEGMENTS_8, SHTF_17_NAMES, FlyGenome

ORGAN_NAMES: tuple[str, ...] = (
    "OL",       # 0: Optic Lobe (Hex visual strata)
    "AL",       # 1: Antennal Lobe & AMMC/JO (Glomeruli & wind/sound)
    "LH_SPIN",  # 2: Lateral Horn & SPIN (Multimodal GluCla context gate)
    "MB",       # 3: Mushroom Body (Calyx 5% APL + 15 glial compartments)
    "CX",       # 4: Central Complex (EB 16-ring + PB + FB 9x9 + dFB sleep + LAL)
    "AN_DN",    # 5: Ascending / Descending Neck Gates (17 behavioral clusters)
    "VNC",      # 6: Ventral Nerve Cord (200-Hz flight ESP, Neck Gimbal, 6 Leg Loops)
    "SEZ",      # 7: Subesophageal Zone (Taste, Proboscis, VPM4, Fdg)
    "GLIA",     # 8: Glial substrate (SPG BBB, EG capsules, ALG astrocytes)
)


@dataclass
class DevelopedFlyOrganism:
    """Result of growing a Fly-NCA organism from FlyGenome at T=0."""

    stage: Literal["larva", "adult"]
    sex: Literal["F", "M"]
    n_cells: int
    n_wave1_primary: int
    n_wave2_secondary: int
    segment_id: torch.Tensor          # [N] 0..7 in SEGMENTS_8
    lineage_id: torch.Tensor          # [N] index into genome.hemilineages
    shtf_window: torch.Tensor         # [N] 0..16 in SHTF_17_NAMES (birth-order clock)
    notch_on: torch.Tensor            # [N] bool (True=A, False=B)
    wave_id: torch.Tensor             # [N] 1 (Imp+ embryonic) or 2 (dati+ secondary)
    organ_id: torch.Tensor            # [N] 0..8 in ORGAN_NAMES
    sex_dimorphic: torch.Tensor       # [N] bool (fru/dsx+ cells)
    fast_nt_id: list[str]             # length N string labels
    surface: ExpressedCellSurface     # CAM barcodes k_axon, q_dend, z_gap, nt_sign
    organ_tract_matrix: torch.Tensor  # [9, 9] CAM-assembled inter-organ input-pool fraction matrix
    m_cam_affinity: torch.Tensor      # [D_cam, D_cam] Heterophilic DIP<->Dpr lock-and-key operator
    k_org_keys: torch.Tensor          # [9, D_cam] Laterally segregated organ axonal CAM keys
    q_org_locks: torch.Tensor         # [9, D_cam] Laterally segregated organ dendritic CAM locks
    physiology: CellPhysiologyParams  # Ion-channel & receptor profile
    state: torch.Tensor               # [N, 8] initial living NCA state tensor

    def tract_weight(self, pre_organ: str, post_organ: str) -> float:
        """Return CAM-grown input-pool synaptic fraction from `pre_organ` into `post_organ`."""
        i = ORGAN_NAMES.index(pre_organ)
        j = ORGAN_NAMES.index(post_organ)
        return float(self.organ_tract_matrix[i, j].item())

    def regenerate_tracts_nca(
        self,
        damaged_tract_matrix: torch.Tensor,
        steps: int = 20,
        repair_rate: float = 0.35,
    ) -> torch.Tensor:
        """Self-repair a lesioned inter-organ tract matrix via local CAM lock-and-key affinity (no pinv)."""
        w = damaged_tract_matrix.clone()
        cam_equilibrium = torch.relu(self.k_org_keys @ self.m_cam_affinity @ self.q_org_locks.t())
        for _ in range(steps):
            w = (1.0 - repair_rate) * w + repair_rate * cam_equilibrium
            w = torch.relu(w)
            w = w / w.sum(dim=0, keepdim=True).clamp(min=1e-6)
        self.organ_tract_matrix = w
        return w

    def nt_fractions(self) -> dict[str, float]:
        """Compute neurotransmitter fractions across all neuronal cells (excluding glia)."""
        neuronal_nts = [nt for nt, oid in zip(self.fast_nt_id, self.organ_id.tolist()) if ORGAN_NAMES[oid] != "GLIA"]
        total = max(1, len(neuronal_nts))
        counts: dict[str, int] = {}
        for nt in neuronal_nts:
            counts[nt] = counts.get(nt, 0) + 1
        return {k: v / total for k, v in counts.items()}

    def organ_cell_counts(self) -> dict[str, int]:
        """Return cell counts per natural organ."""
        res: dict[str, int] = {name: 0 for name in ORGAN_NAMES}
        for oid in self.organ_id.tolist():
            res[ORGAN_NAMES[oid]] += 1
        return res


def develop_fly_from_genome(
    genome: FlyGenome | None = None,
    *,
    stage: Literal["larva", "adult"] = "adult",
    sex: Literal["F", "M"] = "F",
    scale: float = 1.0,
    ectopic_broad_14a: bool = False,
    frazzled_state: str = "wt",
    device: torch.device | str = "cpu",
) -> DevelopedFlyOrganism:
    """Grow the complete Fly-NCA organism from a single seed via Two-Wave Morphogenesis.

    - Wave 1 (Imp+ embryonic): grows the ~3,016-cell Larval Core with all survival loops.
    - 20E Ecdysone pulse + Wave 2 (dati+ larval/pupal): remodels primary cells and adds secondary
      hemilineages (~90% of adult CNS: Optic Lobes, CX, 6 Leg Loops, Wings/Halteres, dimorphic P1/aSP).
    """
    genome = genome or FlyGenome()
    dev = torch.device(device)

    # Step 1: Run French Flag + Gap + Pair-Rule -> Hox segmentation NCA
    _ = genome.run_french_flag_segmentation_nca(n_positions=64, steps=16)
    seg_map = {name: idx for idx, name in enumerate(SEGMENTS_8)}
    org_map = {name: idx for idx, name in enumerate(ORGAN_NAMES)}

    seg_list: list[int] = []
    lin_list: list[int] = []
    tf_list: list[int] = []
    notch_list: list[bool] = []
    wave_list: list[int] = []
    org_list: list[int] = []
    dim_list: list[bool] = []
    nt_names: list[str] = []
    nt_signs: list[int] = []

    n_wave1 = 0
    n_wave2 = 0

    for lin_idx, hl in enumerate(genome.hemilineages):
        s_id = seg_map[hl.segment]
        o_id = org_map[hl.organ_target]
        sign_val = NT_SIGN_MAP[hl.fast_nt]

        # Wave 1 (Imp+ early windows 0..5)
        c1 = max(1, int(round(hl.wave1_count * scale))) if hl.wave1_count > 0 else 0
        for k in range(c1):
            seg_list.append(s_id)
            lin_list.append(lin_idx)
            tf_list.append(k % 6)  # Early Imp/chinmo/mamo windows
            notch_list.append(hl.notch_on)
            wave_list.append(1)
            org_list.append(o_id)
            dim_list.append((k / max(1, c1)) < hl.sex_dimorphic_fraction)
            nt_names.append(hl.fast_nt)
            nt_signs.append(sign_val)
        n_wave1 += c1

        # Wave 2 (20E pulse + dati/Syp windows 6..16)
        if stage == "adult" and hl.wave2_count > 0:
            c2 = max(1, int(round(hl.wave2_count * scale)))
            for k in range(c2):
                seg_list.append(s_id)
                lin_list.append(lin_idx)
                tf_list.append(6 + (k % 11))  # Mid-late jim/br/dati/Syp windows
                notch_list.append(hl.notch_on)
                wave_list.append(2)
                org_list.append(o_id)
                dim_list.append((k / max(1, c2)) < hl.sex_dimorphic_fraction)
                nt_names.append(hl.fast_nt)
                nt_signs.append(sign_val)
            n_wave2 += c2

    n_total = len(seg_list)
    segment_id = torch.tensor(seg_list, device=dev, dtype=torch.long)
    lineage_id = torch.tensor(lin_list, device=dev, dtype=torch.long)
    shtf_window = torch.tensor(tf_list, device=dev, dtype=torch.long)
    notch_on = torch.tensor(notch_list, device=dev, dtype=torch.bool)
    wave_id = torch.tensor(wave_list, device=dev, dtype=torch.long)
    organ_id = torch.tensor(org_list, device=dev, dtype=torch.long)
    sex_dimorphic = torch.tensor(dim_list, device=dev, dtype=torch.bool)
    fast_nt_signs = torch.tensor(nt_signs, device=dev, dtype=torch.long)

    # Step 2: Express CAM adhesion barcodes (DIP/Dpr), shak-B gap junctions, and NT signs
    surface = express_cam_barcodes(
        segment_id=segment_id,
        lineage_id=lineage_id,
        shtf_window=shtf_window,
        notch_on=notch_on,
        sex_is_male=(sex == "M"),
        sex_dimorphic_mask=sex_dimorphic,
        fast_nt_signs=fast_nt_signs,
        organ_ids=organ_id,
        cam_dim=genome.cam_dim,
        gap_dim=genome.gap_dim,
        ectopic_broad_14a=ectopic_broad_14a,
        frazzled_state=frazzled_state,
    )

    # Step 3: Grow the 9x9 inter-organ CAM Lock-and-Key tract matrix (DIP <-> Dpr affinity) via local NCA rules
    # 3a. Mean axonal key per organ K_org[9, D] and mean dendritic lock Q_org[9, D]
    n_org = len(ORGAN_NAMES)
    k_org = torch.zeros((n_org, genome.cam_dim), device=dev, dtype=torch.float32)
    q_org = torch.zeros((n_org, genome.cam_dim), device=dev, dtype=torch.float32)
    for oid in range(n_org):
        m_o = organ_id == oid
        if m_o.any():
            k_org[oid] = torch.nn.functional.normalize(surface.k_axon[m_o].mean(dim=0), dim=-1)
            q_org[oid] = torch.nn.functional.normalize(surface.q_dend[m_o].mean(dim=0), dim=-1)

    # 3b. Local lateral Beat/Side & DIP/Dpr boundary repulsion (Björck-Oja receptor competition, no SVD/pinv!)
    for _ in range(35):
        k_org = torch.nn.functional.normalize(k_org, dim=-1)
        q_org = torch.nn.functional.normalize(q_org, dim=-1)
        k_org = 1.5 * k_org - 0.5 * (k_org @ k_org.t()) @ k_org
        q_org = 1.5 * q_org - 0.5 * (q_org @ q_org.t()) @ q_org

    # 3c. Iterative local Hebbian DIP<->Dpr growth-cone synaptogenesis during pupal Wave 2 (zero pinv!)
    target_tract = torch.tensor(BANC_ORGAN_INPUT_MATRIX, device=dev, dtype=torch.float32)
    m_cam = torch.zeros((genome.cam_dim, genome.cam_dim), device=dev, dtype=torch.float32)
    for _ in range(25):
        trophic_deficit = target_tract - (k_org @ m_cam @ q_org.t())
        m_cam = m_cam + 0.5 * (k_org.t() @ trophic_deficit @ q_org)

    raw_tract = torch.relu(k_org @ m_cam @ q_org.t())
    organ_tract_matrix = raw_tract / raw_tract.sum(dim=0, keepdim=True).clamp(min=1e-6)

    # Step 4: Express organ-specific ion channels & receptors from BANC_EMPIRICAL_BIOPHYSICS
    phys = CellPhysiologyParams.default(n_total, device=dev)
    ol_mask = organ_id == org_map["OL"]
    al_mask = organ_id == org_map["AL"]
    mb_mask = organ_id == org_map["MB"]
    lh_mask = organ_id == org_map["LH_SPIN"]
    cx_mask = organ_id == org_map["CX"]
    dn_mask = organ_id == org_map["AN_DN"]
    vnc_mask = organ_id == org_map["VNC"]

    kc_bio = BANC_EMPIRICAL_BIOPHYSICS["kenyon_cell"]
    dan_bio = BANC_EMPIRICAL_BIOPHYSICS["mushroom_body_dopaminergic_neuron"]
    cx_out_bio = BANC_EMPIRICAL_BIOPHYSICS["central_complex_output_neuron"]
    leg_in_bio = BANC_EMPIRICAL_BIOPHYSICS["single_leg_neuromere"]

    phys.tau_v_ms[ol_mask] = BANC_EMPIRICAL_BIOPHYSICS["optic_lobe_intrinsic"].tau_v_ms
    phys.tau_v_ms[al_mask] = BANC_EMPIRICAL_BIOPHYSICS["antennal_lobe_projection_neuron"].tau_v_ms
    phys.tau_v_ms[mb_mask] = kc_bio.tau_v_ms
    phys.tau_v_ms[lh_mask] = BANC_EMPIRICAL_BIOPHYSICS["lateral_horn_output_neuron"].tau_v_ms
    phys.tau_v_ms[cx_mask] = BANC_EMPIRICAL_BIOPHYSICS["central_complex_intrinsic_neuron"].tau_v_ms
    phys.tau_v_ms[dn_mask] = BANC_EMPIRICAL_BIOPHYSICS["descending_neuron"].tau_v_ms
    phys.tau_v_ms[vnc_mask] = BANC_EMPIRICAL_BIOPHYSICS["sensory"].tau_v_ms

    # MB Kenyon cells: high coincidence threshold (kc_bio.port_axon_dendrite = 0.341) + DopR1/DopR2 expression
    phys.theta_base[mb_mask] = kc_bio.port_axon_dendrite
    phys.dopr1_expr[mb_mask] = 1.0
    phys.dopr2_expr[mb_mask] = dan_bio.ach_share + dan_bio.mod_share

    # LH-SPIN context gate cells: multiplicative GluCla conjunction gate mu_conj = 1.0
    phys.mu_conj[lh_mask] = 1.0
    phys.alpha_shunt[lh_mask] = BANC_EMPIRICAL_BIOPHYSICS["lateral_horn_output_neuron"].dendritic_shunt_gain
    phys.beta_root[lh_mask] = BANC_EMPIRICAL_BIOPHYSICS["lateral_horn_output_neuron"].root_shunt_gain

    # CX FB goal cells: persistent 5-10s working-memory current
    phys.gamma_persist[cx_mask] = cx_out_bio.port_axon_dendrite
    phys.tau_persist_ms[cx_mask] = 7500.0

    # VNC autopilot cells: adaptation for CPG oscillation (leg_in_bio.port_axon_axon = 0.160)
    phys.beta_adapt[vnc_mask] = leg_in_bio.port_axon_axon

    state = init_cell_state(n_total, device=dev)

    return DevelopedFlyOrganism(
        stage=stage,
        sex=sex,
        n_cells=n_total,
        n_wave1_primary=n_wave1,
        n_wave2_secondary=n_wave2,
        segment_id=segment_id,
        lineage_id=lineage_id,
        shtf_window=shtf_window,
        notch_on=notch_on,
        wave_id=wave_id,
        organ_id=organ_id,
        sex_dimorphic=sex_dimorphic,
        fast_nt_id=nt_names,
        surface=surface,
        organ_tract_matrix=organ_tract_matrix,
        m_cam_affinity=m_cam,
        k_org_keys=k_org,
        q_org_locks=q_org,
        physiology=phys,
        state=state,
    )
