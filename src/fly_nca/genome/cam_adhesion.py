"""Cell Adhesion Molecule (CAM: DIP/Dpr, Beat/Side) & Gap Junction (Frazzled/shak-B) Self-Assembly."""

from __future__ import annotations

from dataclasses import dataclass
import torch

from fly_nca.genome.fly_genome import NT_SIGN_MAP, SHTF_17_NAMES


@dataclass(frozen=True, slots=True)
class ExpressedCellSurface:
    """Surface adhesion keys, locks, gap-junction innexins, and NT signs expressed by cells [N]."""

    k_axon: torch.Tensor       # [N, D_cam] Presynaptic Dpr/Side keys
    q_dend: torch.Tensor       # [N, D_cam] Postsynaptic DIP/Beat locks
    z_gap: torch.Tensor        # [N, D_gap] shak-B / Innexin7 gap-junction channels (Frazzled ICD regulated)
    nt_sign: torch.Tensor      # [N] Fast neurotransmitter sign in {+1, -1} (Bates 2026 law)
    organ_id: torch.Tensor     # [N] Integer organ/compartment ID


def express_cam_barcodes(
    segment_id: torch.Tensor,
    lineage_id: torch.Tensor,
    shtf_window: torch.Tensor,
    notch_on: torch.Tensor,
    sex_is_male: bool,
    sex_dimorphic_mask: torch.Tensor,
    fast_nt_signs: torch.Tensor,
    organ_ids: torch.Tensor,
    cam_dim: int = 16,
    gap_dim: int = 8,
    ectopic_broad_14a: bool = False,
    frazzled_state: str = "wt",
) -> ExpressedCellSurface:
    """Deterministically compute surface CAM barcodes (k_axon, q_dend) and shak-B gap profile from genetic address.

    Implements:
    1. Birth-order 17-shTF + Lineage + Hox + Notch -> orthogonal DIP/Dpr barcodes.
    2. Nature 2026 ectopic Broad (br, index 7) shift in hemilineage 14A: cell count is unchanged,
       but temporal TF window shifts to Broad (7), rotating k_axon / q_dend arbors!
    3. Sex determination (sxl -> tra -> dsx/fru): isomorphic (~98%) cells have identical barcodes
       in F and M; dimorphic fru/dsx+ cells (e.g., cVA pathway Or67d->DA1->aSP-f/g) flip their
       dendritic lock q_dend between female attraction and male repulsion!
    4. eNeuro 2025 Frazzled/DCC regulation of shak-B gap junctions via P3 transcriptional ICD domain:
       wt = 1.0 (9.04% terminal fill), frazzled_lof = 0.587 (5.31% fill), frazzled_icd_rescue = 1.05 (0.97 ms).
    """
    device = segment_id.device
    n_cells = segment_id.shape[0]

    eff_shtf = shtf_window.clone()
    if ectopic_broad_14a:
        # Lineage 14A_16B_Leg_Glu has index 30 in canonical catalog; shift all its windows to Broad (7)
        br_idx = SHTF_17_NAMES.index("br")
        mask_14a = lineage_id == 30
        eff_shtf[mask_14a] = br_idx

    # Construct deterministic trigonometric basis from (segment, lineage, eff_shtf, notch)
    dims = torch.arange(cam_dim, device=device, dtype=torch.float32).unsqueeze(0)  # [1, D]
    lin_f = lineage_id.float().unsqueeze(1)
    seg_f = segment_id.float().unsqueeze(1)
    tf_f = eff_shtf.float().unsqueeze(1)
    notch_f = notch_on.float().unsqueeze(1) * 2.0 - 1.0

    phase_k = 0.37 * lin_f * (dims + 1.0) + 0.71 * tf_f + 0.29 * seg_f + 0.53 * notch_f
    phase_q = 0.41 * lin_f * (dims + 1.0) - 0.67 * tf_f + 0.31 * seg_f - 0.47 * notch_f

    k_axon = torch.cos(phase_k)
    q_dend = torch.sin(phase_q)

    # Sex switch (fru/dsx) on sexually dimorphic neurons (~2% of CNS, e.g., aSP-f vs aSP-g & P1)
    if sex_is_male:
        dim_idx = sex_dimorphic_mask.bool()
        if dim_idx.any():
            # FruM masculinizes dendrites (rotates q_dend and k_axon in dimorphic slots)
            q_dend[dim_idx] = -torch.roll(q_dend[dim_idx], shifts=1, dims=-1)
            k_axon[dim_idx] = torch.roll(k_axon[dim_idx], shifts=1, dims=-1)

    # Normalize CAM vectors
    k_axon = torch.nn.functional.normalize(k_axon, dim=-1)
    q_dend = torch.nn.functional.normalize(q_dend, dim=-1)

    # Gap junction (shak-B / Innexin7) expression regulated by Frazzled ICD (eNeuro 2025)
    fra_scale = 1.0
    if frazzled_state == "lof":
        fra_scale = 5.31 / 9.04  # Exact eNeuro 2025 terminal GJ volume drop (5.31% vs 9.04% wt)
    elif frazzled_state == "icd_rescue":
        fra_scale = 1.08         # Full P3 transcriptional domain rescue (0.97 ms latency)

    g_dims = torch.arange(gap_dim, device=device, dtype=torch.float32).unsqueeze(0)
    z_gap = fra_scale * torch.relu(torch.cos(0.51 * lin_f + 0.33 * g_dims))

    return ExpressedCellSurface(
        k_axon=k_axon,
        q_dend=q_dend,
        z_gap=z_gap,
        nt_sign=fast_nt_signs.to(torch.float32),
        organ_id=organ_ids,
    )


def assemble_cam_synapses(
    surface: ExpressedCellSurface,
    src_candidates: torch.Tensor,
    dst_candidates: torch.Tensor,
    affinity_matrix: torch.Tensor | None = None,
    theta_syn: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Assemble chemical synapse weights W_{i->j}^{(0)} via DIP <-> Dpr lock-and-key complementarity."""
    k_pre = surface.k_axon[src_candidates]
    q_post = surface.q_dend[dst_candidates]
    if affinity_matrix is not None:
        k_pre = k_pre @ affinity_matrix
    affinity = (k_pre * q_post).sum(dim=-1)
    keep = affinity > theta_syn
    pre_idx = src_candidates[keep]
    post_idx = dst_candidates[keep]
    raw_w = torch.sigmoid(4.0 * (affinity[keep] - theta_syn))
    signed_w = surface.nt_sign[pre_idx] * raw_w
    return pre_idx, post_idx, signed_w
