"""Five Biological Perception Ports for Universal Fly NCA Cells (Section 1.2 & Section 3 of Architecture)."""

from __future__ import annotations

import torch

from fly_nca.core.cell_state import CH_A, CH_AUTO, CH_V, CellPhysiologyParams


def port1_2_dendritic_integration(
    exc_branches: torch.Tensor,
    shunt_branches: torch.Tensor,
    params: CellPhysiologyParams,
    sub_branches: torch.Tensor | None = None,
    root_shunt: torch.Tensor | None = None,
    branch_gain_gamma: float = 2.5,
) -> torch.Tensor:
    """Ports 1 & 2: Dendritic branch excitation (ACh+) with GluCla/GABA shunting and tanh conjunction gate.

    Args:
        exc_branches: [N, B] excitatory branch inputs I_{exc, b}(i, t) >= 0.
        shunt_branches: [N, B] chloride shunting conductances g_{shunt, b}(i, t) >= 0 (GluCla / GABA Rdl).
        params: CellPhysiologyParams with alpha_shunt [N], beta_root [N], mu_conj [N].
        sub_branches: Optional [N, B] subtractive GABA currents I_{sub, b}(i, t) >= 0.
        root_shunt: Optional [N] soma-root aversive shunting conductance g_{root}(i, t) >= 0 (aversion > attraction).
        branch_gain_gamma: Slope factor gamma_b inside tanh(gamma_b * I_{branch, b}) for saturation at 1.0.

    Returns:
        i_dend: [N] integrated dendritic current entering the soma.
    """
    net_exc = exc_branches if sub_branches is None else torch.relu(exc_branches - sub_branches)
    denom_branch = 1.0 + params.alpha_shunt.unsqueeze(-1) * torch.relu(shunt_branches)
    i_branch = net_exc / denom_branch  # [N, B] >= 0

    # Linear branch sum vs multiplicative tanh conjunction gate (cLTM / SPIN SMP353)
    sum_part = i_branch.sum(dim=-1)
    sat_branches = torch.tanh(branch_gain_gamma * i_branch)
    prod_part = torch.prod(sat_branches, dim=-1)

    mu = params.mu_conj.clamp(0.0, 1.0)
    combined = (1.0 - mu) * sum_part + mu * prod_part

    if root_shunt is not None:
        denom_root = 1.0 + params.beta_root * torch.relu(root_shunt)
        combined = combined / denom_root

    return combined


def port3_axo_axonic_presynaptic_gate(
    soma_activity: torch.Tensor,
    autoreceptor_r: torch.Tensor,
    kappa_auto: torch.Tensor,
    aa_exc: torch.Tensor | None = None,
    aa_inh: torch.Tensor | None = None,
) -> torch.Tensor:
    """Port 3: Axo-axonic (a-a, 25.8% of synapses!) presynaptic compartment gate + autoreceptor brake.

    Allows a single neuron i (e.g. Kenyon cell or haltere sensory neuron) to emit different
    effective outputs Out_{i, c}(t) across different compartments c without altering its soma state.

    Args:
        soma_activity: [N] or [N, 1] soma firing output a_i(t).
        autoreceptor_r: [N] or [N, 1] presynaptic autoreceptor state r_i(t) (OAa2R / mGluR).
        kappa_auto: [N] or [N, 1] autoreceptor suppression strength.
        aa_exc: Optional [N] or [N, C] axo-axonic presynaptic facilitation in compartment c.
        aa_inh: Optional [N] or [N, C] axo-axonic presynaptic shunting inhibition in compartment c.

    Returns:
        out_ic: Effective presynaptic terminal output in [0, 1].
    """
    facil = 1.0 if aa_exc is None else (1.0 + torch.relu(aa_exc)).clamp(max=2.0)
    shunt = 1.0 if aa_inh is None else (1.0 + torch.relu(aa_inh))
    auto_brake = 1.0 + torch.relu(kappa_auto) * torch.relu(autoreceptor_r)
    return (soma_activity * facil / (shunt * auto_brake)).clamp(0.0, 1.0)


def port4_electrical_gap_junction(
    v_membrane: torch.Tensor,
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
    g_gap: torch.Tensor,
    rectifying: bool = True,
) -> torch.Tensor:
    """Port 4: Electrical synapse (shak-B / Innexin7, regulated by Frazzled ICD).

    In Drosophila sensory->motor circuits (GF -> TTMn and Haltere -> b1 Wing MN),
    heterotypic shak-B(N+16)/shak-B(L) gap junctions are rectifying diodes (current flows
    from presynaptic sensor/interneuron to postsynaptic motor cell when V_src > V_dst,
    preventing motor drive from back-flooding sensory receptors!).
    """
    i_gap = torch.zeros_like(v_membrane)
    if src_idx.numel() == 0:
        return i_gap
    dv = v_membrane[src_idx] - v_membrane[dst_idx]
    if rectifying:
        flux = g_gap * torch.relu(dv)
        i_gap.index_add_(0, dst_idx, flux)
    else:
        flux = g_gap * dv
        i_gap.index_add_(0, dst_idx, flux)
        i_gap.index_add_(0, src_idx, -flux)
    return i_gap


def sparse_synaptic_gather(
    presyn_output: torch.Tensor,
    pre_idx: torch.Tensor,
    post_idx: torch.Tensor,
    weights: torch.Tensor,
    n_cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather signed chemical synaptic currents into Port 1 (Exc +) and Port 2 (Shunt/Inh -)."""
    exc = torch.zeros(n_cells, device=presyn_output.device, dtype=presyn_output.dtype)
    inh = torch.zeros(n_cells, device=presyn_output.device, dtype=presyn_output.dtype)
    if pre_idx.numel() == 0:
        return exc, inh
    msg = presyn_output[pre_idx] * weights
    pos_mask = weights > 0
    if pos_mask.any():
        exc.index_add_(0, post_idx[pos_mask], msg[pos_mask])
    neg_mask = weights < 0
    if neg_mask.any():
        inh.index_add_(0, post_idx[neg_mask], (-msg[neg_mask]))
    return exc, inh
