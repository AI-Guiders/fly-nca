"""Pure Gene-Regulatory Network (GRN) Cellular Automaton: Genes & Enhancers -> Cells & Synapses.

Zero `organ_id`, zero `segment_id`, zero species constants, and zero hardcoded cell counts:
1. A genome (`GeneGraph`) is purely a graph of `Gene` loci, where each gene has:
   - `activators`: genes that bind its cis-regulatory enhancer to activate transcription (+),
   - `repressors`: genes that bind its enhancer to repress transcription (-),
   - `role`: molecular class of the gene product (`MORPHOGEN`, `TF`, `SIGNAL`, `CELL_CYCLE`, `STOP_CYCLE`, `CAM_KEY`, `CAM_LOCK`, `NT`, `CHANNEL`).
2. Starting from an initial syncytium with maternal anterior (`bcd`) and posterior (`nos`) RNA poles,
   `run_gene_ontogeny_automaton(gene_graph)` steps the gene network `E(t) -> E(t+1)` and physically divides
   any cell where `CELL_CYCLE > STOP_CYCLE` until all cells autonomously activate `STOP_CYCLE` and exit mitosis.
3. Differentiated cells know nothing about human labels like "Mushroom Body" or "Segment T2": each cell only has
   its own gene expression vector `E[c, :]`, and synapses `W[i, j]` self-assemble directly between cells whose
   expressed `CAM_KEY` and `CAM_LOCK` proteins bind each other, signed by each presynaptic cell's expressed `NT` enzyme.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
import torch

GeneRole = Literal[
    "MORPHOGEN",
    "TF",
    "SIGNAL",
    "CELL_CYCLE",
    "STOP_CYCLE",
    "CAM_KEY",
    "CAM_LOCK",
    "NT",
    "CHANNEL",
]


@dataclass(frozen=True, slots=True)
class Gene:
    """Single genetic locus defined strictly by its cis-regulatory enhancer and molecular role."""

    name: str
    role: GeneRole
    activators: tuple[tuple[str, float], ...] = ()
    repressors: tuple[tuple[str, float], ...] = ()
    cam_partners: tuple[tuple[str, float], ...] = ()  # For CAM_KEY genes: which CAM_LOCK proteins they physically bind
    nt_valence: float = 0.0                           # For NT genes: +1.0 (excitatory ACh) or -1.0 (inhibitory GABA/Glu)
    basal: float = 0.0
    decay: float = 0.12
    functional: bool = True


@dataclass(frozen=True, slots=True)
class GeneOntogenyResult:
    """Autonomous outcome of running the GeneGraph cellular automaton from t=0 to cell-cycle exit."""

    n_cells: int
    gene_names: tuple[str, ...]
    expression: torch.Tensor        # [N_cells, G] final gene expression vector of every differentiated cell
    nt_sign: torch.Tensor           # [N_cells] net neurotransmitter sign in [-1, +1] from expressed NT enzymes
    synaptic_weights: torch.Tensor  # [N_cells, N_cells] synaptic matrix assembled purely from CAM_KEY x CAM_LOCK binding
    divisions_per_step: list[int]   # Number of physical cell divisions at each ontogenetic step

    def gene_column(self, gene_name: str) -> torch.Tensor:
        """Return expression vector [N_cells] of a specific gene across all grown cells."""
        idx = self.gene_names.index(gene_name)
        return self.expression[:, idx]

    def count_expressing(self, gene_name: str, threshold: float = 0.5) -> int:
        """Count how many differentiated cells express `gene_name` above `threshold`."""
        return int((self.gene_column(gene_name) > threshold).sum().item())


@dataclass(frozen=True, slots=True)
class GeneGraph:
    """Complete genome represented purely as a set of genes and their enhancer binding sites."""

    genes: tuple[Gene, ...]

    def gene_index(self) -> dict[str, int]:
        return {g.name: i for i, g in enumerate(self.genes)}

    def with_knockout(self, gene_name: str) -> GeneGraph:
        """Return a starting genome where `gene_name` is non-functional (loss-of-function mutation)."""
        return GeneGraph(
            genes=tuple(replace(g, functional=False) if g.name == gene_name else g for g in self.genes)
        )

    def with_duplication(
        self,
        source_gene: str,
        new_gene_name: str,
        *,
        replace_activators: tuple[tuple[str, float], ...] | None = None,
        add_activators: tuple[tuple[str, float], ...] = (),
        add_repressors: tuple[tuple[str, float], ...] = (),
        feed_into_targets: tuple[tuple[str, Literal["activate", "repress"], float], ...] = (),
    ) -> GeneGraph:
        """Return a starting genome with a duplicated gene locus and cis-regulatory enhancer divergence."""
        src = next(g for g in self.genes if g.name == source_gene)
        new_acts = replace_activators if replace_activators is not None else (src.activators + add_activators)
        duplicated = Gene(
            name=new_gene_name,
            role=src.role,
            activators=new_acts,
            repressors=src.repressors + add_repressors,
            cam_partners=src.cam_partners,
            nt_valence=src.nt_valence,
            basal=src.basal,
            decay=src.decay,
            functional=True,
        )
        updated: list[Gene] = []
        target_map = {t_name: (mode, w) for t_name, mode, w in feed_into_targets}
        for g in self.genes:
            if g.name in target_map:
                mode, w = target_map[g.name]
                if mode == "activate":
                    g = replace(g, activators=g.activators + ((new_gene_name, w),))
                else:
                    g = replace(g, repressors=g.repressors + ((new_gene_name, w),))
            updated.append(g)
        updated.append(duplicated)
        return GeneGraph(genes=tuple(updated))

    def compile_matrices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compile enhancer binding sites into [G, G] activation and repression matrices."""
        idx = self.gene_index()
        n_g = len(self.genes)
        w_act = torch.zeros((n_g, n_g), device=device, dtype=torch.float32)
        w_rep = torch.zeros((n_g, n_g), device=device, dtype=torch.float32)
        basal = torch.zeros(n_g, device=device, dtype=torch.float32)
        decay = torch.zeros(n_g, device=device, dtype=torch.float32)
        func_mask = torch.zeros(n_g, device=device, dtype=torch.float32)

        for post_i, g in enumerate(self.genes):
            basal[post_i] = float(g.basal)
            decay[post_i] = float(g.decay)
            func_mask[post_i] = 1.0 if g.functional else 0.0
            for act_name, weight in g.activators:
                if act_name in idx:
                    w_act[idx[act_name], post_i] += float(weight)
            for rep_name, weight in g.repressors:
                if rep_name in idx:
                    w_rep[idx[rep_name], post_i] += float(weight)

        return w_act, w_rep, basal, decay, func_mask


def build_canonical_gene_graph() -> GeneGraph:
    """Construct the canonical bilaterian/dipteran Gene Regulatory Network (GRN)."""
    genes: tuple[Gene, ...] = (
        # --- 1. Maternal & Axial Morphogens (diffuse between neighboring cells) ---
        Gene("bcd", "MORPHOGEN", activators=(("bcd", 0.4),), decay=0.02),
        Gene("nos", "MORPHOGEN", activators=(("nos", 0.4),), decay=0.02),
        Gene("cad", "TF", basal=0.85, repressors=(("bcd", 2.2), ("ey_Six3", 2.0), ("otd_Otx", 2.0), ("Dfd_Scr", 1.6)), decay=0.10),
        # --- 2. Gap Gene Cross-Repression Network ---
        Gene("hb", "TF", activators=(("bcd", 1.8), ("hb", 0.6)), repressors=(("nos", 2.2), ("Kr", 1.2)), decay=0.10),
        Gene("Kr", "TF", activators=(("bcd", 0.9), ("hb", 0.8)), repressors=(("kni", 1.6), ("tll", 1.8)), decay=0.10),
        Gene("kni", "TF", activators=(("cad", 1.6),), repressors=(("hb", 2.0), ("Kr", 1.2), ("tll", 1.6)), decay=0.10),
        Gene("tll", "TF", activators=(("bcd", 1.1), ("nos", 1.4)), repressors=(("Kr", 1.5),), decay=0.10),
        # --- 3. Anterior-Posterior Hox & Head Selector Genes (with Trithorax/Polycomb self-maintenance) ---
        # Antp is repressed by Ubx in posterior thorax: if Ubx is knocked out (Ubx^-), Antp stays ON in posterior thorax!
        Gene("ey_Six3", "TF", activators=(("bcd", 2.0), ("tll", 1.2), ("ey_Six3", 2.2)), repressors=(("cad", 2.2), ("Kr", 1.8), ("Dfd_Scr", 1.5)), decay=0.04),
        Gene("otd_Otx", "TF", activators=(("bcd", 2.0), ("hb", 1.3), ("otd_Otx", 2.3)), repressors=(("Dfd_Scr", 1.5), ("Kr", 2.0), ("cad", 1.8), ("Antp", 2.0)), decay=0.04),
        Gene("Dfd_Scr", "TF", activators=(("bcd", 1.0), ("hb", 1.2), ("Dfd_Scr", 1.75)), repressors=(("otd_Otx", 1.6), ("Antp", 1.6), ("kni", 2.0)), decay=0.04),
        Gene("Antp", "TF", activators=(("hb", 0.6), ("Kr", 1.7), ("Antp", 1.75)), repressors=(("otd_Otx", 2.0), ("Dfd_Scr", 1.5), ("Ubx", 2.4), ("abd_A", 2.4)), decay=0.04),
        Gene("Ubx", "TF", activators=(("Kr", 1.2), ("kni", 1.1), ("Ubx", 1.75)), repressors=(("hb", 1.9), ("abd_A", 2.0), ("Abd_B", 2.2)), decay=0.04),
        Gene("abd_A", "TF", activators=(("kni", 1.8), ("cad", 1.0), ("abd_A", 1.75)), repressors=(("hb", 2.4), ("Kr", 1.6), ("Abd_B", 1.8)), decay=0.04),
        Gene("Abd_B", "TF", activators=(("cad", 1.6), ("tll", 1.5), ("Abd_B", 1.75)), repressors=(("hb", 2.5), ("Kr", 2.2)), decay=0.04),
        # --- 4. Dorso-Ventral Columnar & Glial Selectors ---
        Gene("vnd_Nkx2", "TF", activators=(("vnd_Nkx2", 1.5),), repressors=(("ind_Gsx", 1.4), ("msh_Msx", 1.4)), decay=0.03),
        Gene("ind_Gsx", "TF", activators=(("ind_Gsx", 1.5),), repressors=(("vnd_Nkx2", 1.4), ("msh_Msx", 1.4)), decay=0.03),
        Gene("msh_Msx", "TF", activators=(("msh_Msx", 1.5),), repressors=(("vnd_Nkx2", 1.4), ("ind_Gsx", 1.4)), decay=0.03),
        Gene("gcm_Glia", "TF", activators=(("Notch", 0.9), ("Syp", 1.1), ("gcm_Glia", 1.2)), repressors=(("Imp", 1.8), ("Antp", 0.9)), decay=0.05),
        # --- 5. Neuroblast Temporal Cascade (Imp -> chinmo -> EcR_20E -> br -> dati -> Syp) & Cell Cycle ---
        Gene("Notch", "SIGNAL", activators=(("Cyclin", 0.9),), repressors=(("pros_stop", 1.4),), decay=0.15),
        Gene("Imp", "TF", activators=(("Imp", 0.82), ("otd_Otx", 0.22), ("ey_Six3", 0.28)), repressors=(("Syp", 1.45), ("pros_stop", 1.8)), decay=0.14),
        Gene("chinmo", "TF", activators=(("Imp", 1.3),), repressors=(("br", 1.2), ("Syp", 1.5)), decay=0.15),
        Gene("EcR_20E", "SIGNAL", activators=(("chinmo", 0.95), ("Kr", 0.35), ("otd_Otx", 0.45), ("ey_Six3", 0.55), ("Antp", 0.45)), repressors=(("Imp", 1.1),), decay=0.10),
        Gene("br", "TF", activators=(("EcR_20E", 1.5), ("chinmo", 0.6)), repressors=(("Imp", 1.2), ("Syp", 1.1)), decay=0.12),
        Gene("dati", "TF", activators=(("EcR_20E", 1.4), ("br", 1.1)), repressors=(("Imp", 1.5),), decay=0.10),
        Gene("Syp", "TF", activators=(("chinmo", 0.75), ("br", 1.15), ("dati", 1.25), ("Syp", 0.85)), repressors=(("Imp", 1.35),), decay=0.06),
        Gene("Cyclin", "CELL_CYCLE", activators=(("Imp", 1.5), ("EcR_20E", 1.45), ("otd_Otx", 0.25), ("ey_Six3", 0.35)), repressors=(("pros_stop", 2.4),), decay=0.12),
        Gene("pros_stop", "STOP_CYCLE", activators=(("Syp", 1.75),), repressors=(("Imp", 1.8), ("EcR_20E", 1.25)), decay=0.05),
        # --- 6. Fast Neurotransmitter Synthesis Enzymes (Lacin/Bates Notch Law) ---
        Gene("ChaT_ACh", "NT", activators=(("Notch", 1.6), ("pros_stop", 0.8)), repressors=(("Gad1_GABA", 1.5),), nt_valence=+1.0, decay=0.05),
        Gene("Gad1_GABA", "NT", activators=(("pros_stop", 1.1), ("ind_Gsx", 0.8)), repressors=(("Notch", 1.6),), nt_valence=-1.0, decay=0.05),
        Gene("VGlut_Glu", "NT", activators=(("pros_stop", 1.1), ("msh_Msx", 0.9), ("Antp", 0.5)), repressors=(("Notch", 1.5),), nt_valence=-1.0, decay=0.05),
        Gene("Hdc_His", "NT", activators=(("ey_Six3", 1.6), ("pros_stop", 0.9)), repressors=(("Notch", 1.4), ("otd_Otx", 1.2)), nt_valence=-1.0, decay=0.05),
        # --- 7. Cell Adhesion Molecule (CAM) Surface Proteins (physical ligand-receptor binding pairs) ---
        Gene("Dpr1", "CAM_KEY", activators=(("ey_Six3", 1.8),), repressors=(("Antp", 1.5),), cam_partners=(("DIP_a", 1.0), ("DIP_g", 0.5)), decay=0.05),
        Gene("Dpr2", "CAM_KEY", activators=(("otd_Otx", 1.4), ("Imp", 0.7)), repressors=(("ey_Six3", 1.2),), cam_partners=(("DIP_b", 1.0), ("DIP_a", 0.7)), decay=0.05),
        Gene("Dpr3", "CAM_KEY", activators=(("otd_Otx", 1.6), ("dati", 1.1)), repressors=(("Antp", 1.8),), cam_partners=(("DIP_a", 1.0), ("DIP_g", 0.8)), decay=0.05),
        Gene("Dpr4", "CAM_KEY", activators=(("otd_Otx", 1.5), ("br", 1.2)), repressors=(("Dfd_Scr", 1.4),), cam_partners=(("DIP_d", 1.0),), decay=0.05),
        Gene("Dpr5", "CAM_KEY", activators=(("Dfd_Scr", 1.8),), repressors=(("Antp", 1.2),), cam_partners=(("DIP_z", 1.0), ("DIP_d", 0.6)), decay=0.05),
        Gene("Dpr6", "CAM_KEY", activators=(("Antp", 1.4), ("hb", 0.8)), repressors=(("Ubx", 1.8),), cam_partners=(("DIP_e", 1.0),), decay=0.05),
        Gene("Dpr7", "CAM_KEY", activators=(("Antp", 1.7),), repressors=(("hb", 1.1), ("Ubx", 2.0)), cam_partners=(("DIP_e", 1.0),), decay=0.05),
        Gene("Dpr8", "CAM_KEY", activators=(("Ubx", 1.8),), repressors=(("abd_A", 1.6),), cam_partners=(("DIP_e", 1.0),), decay=0.05),
        Gene("Dpr9", "CAM_KEY", activators=(("abd_A", 1.5), ("Abd_B", 1.5)), repressors=(("Antp", 1.8),), cam_partners=(("DIP_e", 1.0),), decay=0.05),
        Gene("DIP_a", "CAM_LOCK", activators=(("otd_Otx", 1.5), ("chinmo", 0.9)), repressors=(("Antp", 1.5),), decay=0.05),
        Gene("DIP_b", "CAM_LOCK", activators=(("otd_Otx", 1.6), ("Notch", 0.9)), repressors=(("Ubx", 1.8),), decay=0.05),
        Gene("DIP_g", "CAM_LOCK", activators=(("otd_Otx", 1.4), ("dati", 1.1)), repressors=(("abd_A", 1.8),), decay=0.05),
        Gene("DIP_d", "CAM_LOCK", activators=(("Dfd_Scr", 1.1), ("otd_Otx", 1.1), ("br", 1.0)), repressors=(("ey_Six3", 1.4),), decay=0.05),
        Gene("DIP_e", "CAM_LOCK", activators=(("Antp", 1.5), ("Ubx", 1.4), ("abd_A", 1.2)), repressors=(("otd_Otx", 1.8),), decay=0.05),
        Gene("DIP_z", "CAM_LOCK", activators=(("Dfd_Scr", 1.7),), repressors=(("otd_Otx", 1.2),), decay=0.05),
        # --- 8. Ion Channel Genes ---
        Gene("Para_NaV", "CHANNEL", activators=(("pros_stop", 1.4),), decay=0.05),
        Gene("Shaker_KV", "CHANNEL", activators=(("pros_stop", 1.2), ("Antp", 0.6), ("Ubx", 0.6)), decay=0.05),
        Gene("Ca_alpha1T", "CHANNEL", activators=(("pros_stop", 1.1), ("msh_Msx", 0.7)), decay=0.05),
        Gene("GluCla_Rdl", "CHANNEL", activators=(("pros_stop", 1.3), ("otd_Otx", 0.5)), decay=0.05),
    )
    return GeneGraph(genes=genes)


def run_gene_ontogeny_automaton(
    gene_graph: GeneGraph,
    *,
    n_founder_nuclei: int = 48,
    max_ontogeny_steps: int = 18,
    device: torch.device | str = "cpu",
) -> GeneOntogenyResult:
    """Run the autonomous Gene-Regulatory Network + Cell-Division Automaton from t=0 until `STOP_CYCLE` halts mitosis."""
    dev = torch.device(device)
    idx = gene_graph.gene_index()
    n_g = len(gene_graph.genes)
    w_act, w_rep, basal, decay, func_mask = gene_graph.compile_matrices(dev)

    morph_indices = [i for i, g in enumerate(gene_graph.genes) if g.role == "MORPHOGEN"]
    cyc_indices = [i for i, g in enumerate(gene_graph.genes) if g.role == "CELL_CYCLE"]
    stop_indices = [i for i, g in enumerate(gene_graph.genes) if g.role == "STOP_CYCLE"]

    # 1. Syncytial blastoderm initialization: maternal `bcd` at anterior pole, `nos` at posterior pole
    expr = torch.zeros((n_founder_nuclei, n_g), device=dev, dtype=torch.float32)
    if "bcd" in idx:
        expr[:4, idx["bcd"]] = 1.0 * func_mask[idx["bcd"]]
    if "nos" in idx:
        expr[-4:, idx["nos"]] = 1.0 * func_mask[idx["nos"]]

    # Local 1D syncytial morphogen diffusion + enhancer transcription before neuroblast delamination
    for _ in range(80):
        if "bcd" in idx:
            expr[:2, idx["bcd"]] = 1.0 * func_mask[idx["bcd"]]
        if "nos" in idx:
            expr[-2:, idx["nos"]] = 1.0 * func_mask[idx["nos"]]
        for m_i in morph_indices:
            col = expr[:, m_i]
            col_l = torch.cat([col[:1], col[:-1]])
            col_r = torch.cat([col[1:], col[-1:]])
            expr[:, m_i] = (0.50 * col + 0.25 * (col_l + col_r)) * func_mask[m_i]
        net = basal.unsqueeze(0) + expr @ w_act - expr @ w_rep
        act = torch.sigmoid(3.2 * (net - 0.35)) * func_mask.unsqueeze(0)
        for g_i, g in enumerate(gene_graph.genes):
            if g.role not in ("MORPHOGEN", "CELL_CYCLE", "STOP_CYCLE"):
                expr[:, g_i] = (1.0 - decay[g_i]) * expr[:, g_i] + decay[g_i] * act[:, g_i]

    # Lateral D-V symmetry breaking across founder neuroblasts and start of cell-cycle gene expression
    ap_idx = torch.arange(n_founder_nuclei, device=dev)
    if "vnd_Nkx2" in idx:
        expr[ap_idx % 3 == 0, idx["vnd_Nkx2"]] = 0.90 * func_mask[idx["vnd_Nkx2"]]
    if "ind_Gsx" in idx:
        expr[ap_idx % 3 == 1, idx["ind_Gsx"]] = 0.90 * func_mask[idx["ind_Gsx"]]
    if "msh_Msx" in idx:
        expr[ap_idx % 3 == 2, idx["msh_Msx"]] = 0.90 * func_mask[idx["msh_Msx"]]
    if "Imp" in idx:
        expr[:, idx["Imp"]] = 0.95 * func_mask[idx["Imp"]]
    for c_i in cyc_indices:
        expr[:, c_i] = 0.90 * func_mask[c_i]
    for s_i in stop_indices:
        expr[:, s_i] = 0.0

    # 2. Autonomous Gene-Driven Cell Division Loop:
    # Any cell where CELL_CYCLE > STOP_CYCLE divides into two post-mitotic daughters (Notch+ and Notch-)
    # while self-renewing until STOP_CYCLE genes turn ON. If extra repressors strongly suppress STOP_CYCLE
    # in a progenitor, the progenitor itself duplicates (transit-amplifying bRG / INP division)!
    neurons_expr_list: list[torch.Tensor] = []
    divisions_log: list[int] = []

    for step_t in range(max_ontogeny_steps):
        net = basal.unsqueeze(0) + expr @ w_act - expr @ w_rep
        act = torch.sigmoid(3.2 * (net - 0.35)) * func_mask.unsqueeze(0)
        expr = ((1.0 - decay.unsqueeze(0)) * expr + decay.unsqueeze(0) * act) * func_mask.unsqueeze(0)

        cyc_val = expr[:, cyc_indices].max(dim=1).values if cyc_indices else torch.zeros(expr.shape[0], device=dev)
        stop_val = expr[:, stop_indices].max(dim=1).values if stop_indices else torch.ones(expr.shape[0], device=dev)
        dividing_mask = (cyc_val > 0.48) & (stop_val < 0.52)
        n_div = int(dividing_mask.sum().item())
        divisions_log.append(n_div)
        if n_div == 0:
            break

        div_expr = expr[dividing_mask]
        daughter_a = div_expr.clone()
        daughter_b = div_expr.clone()
        if "Notch" in idx:
            daughter_a[:, idx["Notch"]] = 0.95 * func_mask[idx["Notch"]]
            daughter_b[:, idx["Notch"]] = 0.05
        for s_i in stop_indices:
            daughter_a[:, s_i] = 0.95 * func_mask[s_i]
            daughter_b[:, s_i] = 0.95 * func_mask[s_i]

        neurons_expr_list.extend([daughter_a, daughter_b])

        # Mitotic dilution of early Imp RNA and accumulation of late Syp RNA per division
        if "Imp" in idx:
            expr[dividing_mask, idx["Imp"]] *= 0.76
        if "Syp" in idx and "Imp" in idx:
            expr[dividing_mask, idx["Syp"]] = (
                expr[dividing_mask, idx["Syp"]] + 0.14 * (1.0 - expr[dividing_mask, idx["Imp"]])
            ).clamp(0.0, 1.0) * func_mask[idx["Syp"]]

        # If STOP_CYCLE enhancer receives strong net repression (< -0.65), the progenitor itself duplicates!
        if stop_indices and step_t < 6:
            stop_net = net[:, stop_indices[0]]
            inp_expand_mask = dividing_mask & (stop_net < -0.65)
            if inp_expand_mask.any():
                expr = torch.cat([expr, expr[inp_expand_mask].clone()], dim=0)

    expr = torch.cat(neurons_expr_list, dim=0) if neurons_expr_list else expr

    # Terminal post-mitotic gene expression pass (express CAM_KEY, CAM_LOCK, NT, and CHANNEL genes)
    net_final = basal.unsqueeze(0) + expr @ w_act - expr @ w_rep
    act_final = torch.sigmoid(3.2 * (net_final - 0.35)) * func_mask.unsqueeze(0)
    expr = ((0.5 * expr + 0.5 * act_final) * func_mask.unsqueeze(0)).clamp(0.0, 1.0)

    # 3. Direct Synaptic Self-Assembly from Expressed CAM_KEY and CAM_LOCK Proteins (zero organ_id!)
    cam_key_indices = [i for i, g in enumerate(gene_graph.genes) if g.role == "CAM_KEY"]
    cam_lock_indices = [i for i, g in enumerate(gene_graph.genes) if g.role == "CAM_LOCK"]
    lock_pos = {gene_graph.genes[i].name: k for k, i in enumerate(cam_lock_indices)}

    m_bind = torch.zeros((len(cam_key_indices), len(cam_lock_indices)), device=dev, dtype=torch.float32)
    for k_pos, g_i in enumerate(cam_key_indices):
        for partner_name, aff_w in gene_graph.genes[g_i].cam_partners:
            if partner_name in lock_pos:
                m_bind[k_pos, lock_pos[partner_name]] = float(aff_w)

    k_mat = expr[:, cam_key_indices]   # [N, K]
    q_mat = expr[:, cam_lock_indices]  # [N, L]

    # Neurotransmitter sign per cell directly from expressed NT synthesis enzymes
    nt_indices = [i for i, g in enumerate(gene_graph.genes) if g.role == "NT"]
    nt_valences = torch.tensor([gene_graph.genes[i].nt_valence for i in nt_indices], device=dev, dtype=torch.float32)
    nt_sign = torch.tanh((expr[:, nt_indices] * nt_valences.unsqueeze(0)).sum(dim=1) * 3.0)

    raw_syn = torch.relu((k_mat @ m_bind) @ q_mat.t() - 0.25)
    syn_weights = raw_syn * nt_sign.unsqueeze(1)

    return GeneOntogenyResult(
        n_cells=expr.shape[0],
        gene_names=tuple(g.name for g in gene_graph.genes),
        expression=expr,
        nt_sign=nt_sign,
        synaptic_weights=syn_weights,
        divisions_per_step=divisions_log,
    )
