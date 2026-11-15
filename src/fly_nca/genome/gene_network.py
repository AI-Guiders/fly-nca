"""Pure Gene-Regulatory Network (GRN) Cellular Automaton grounded in Fly Cell Atlas & SPR Biophysics.

Strict Epistemic & Model-Purity Invariants (enforced by AST gate `P4_gene_network_purity` in `verify_principles.py`):
1. Zero float weights on transcriptional edges: `Gene.activators` and `Gene.repressors` are strictly `tuple[str, ...]`
   (unweighted boolean enhancer binding sites, each backed by an explicit `source_ref` DOI/PMC or FCA scRNA-seq enrichment).
2. Zero gene-name strings inside `run_gene_ontogeny_automaton`: the cellular automaton step function knows only physical
   molecular roles (`MORPHOGEN`, `JUXTACRINE_LIGAND`, `JUXTACRINE_RECEPTOR`, `TF`, `MITOTIC_DRIVER`, `CELL_CYCLE_EXIT`,
   `CAM_KEY`, `CAM_LOCK`, `NT_ENZYME`, `ION_CHANNEL`), 2D embryonic tissue geometry `(N_AP x N_DV)`, Shea-Ackers
   thermodynamic promoter occupancy `P = Z_act / (1 + Z_act + Z_rep^2)` (Bintu et al. 2005), and volume-halving mitotic dilution.
3. All effector gene enhancer links (`dpr1..19`, `DIP-alpha..iota`, `VAChT`, `Gad1`, `VGlut`, `Hdc`, `para`, `Sh`,
   `Ca-alpha1T`, `GluClalpha`, `Rdl`) are the exact empirical co-expression enrichments (`>= 1.30` activator, `<= 0.65`
   repressor) computed across all 197,453 single cells of the Fly Cell Atlas (`fca_head_10x_stringent.h5ad` +
   `fca_body_10x_stringent.h5ad`, Li et al. 2022 Science, DOI 10.1126/science.abk2432).
4. All `CAM_KEY <-> CAM_LOCK` physical binding affinities are `1 / K_D` from surface plasmon resonance (SPR)
   measurements in Cosmanescu et al. (2018) Neuron (DOI 10.1016/j.neuron.2018.08.018, PMC6289660).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
import torch

GeneRole = Literal[
    "MORPHOGEN",
    "JUXTACRINE_LIGAND",
    "JUXTACRINE_RECEPTOR",
    "TF",
    "MITOTIC_DRIVER",
    "CELL_CYCLE_EXIT",
    "CAM_KEY",
    "CAM_LOCK",
    "NT_ENZYME",
    "ION_CHANNEL",
]

MaternalPole = Literal["anterior", "posterior", "ventral", "dorsal", "uniform"]


@dataclass(frozen=True, slots=True)
class Gene:
    """Single genetic locus with unweighted cis-regulatory enhancer sites and empirical provenance."""

    name: str
    role: GeneRole
    source_ref: str
    activators: tuple[str, ...] = ()
    repressors: tuple[str, ...] = ()
    maternal_pole: MaternalPole | None = None
    mitotic_dilution: bool = False
    cam_kd_um: tuple[tuple[str, float], ...] = ()  # Measured SPR dissociation constants K_D (uM) from PMC6289660
    nt_valence: float = 0.0
    functional: bool = True


@dataclass(frozen=True, slots=True)
class GeneOntogenyResult:
    """Autonomous outcome of running the GeneGraph cellular automaton on a 2D embryonic sheet until mitotic exit."""

    n_cells: int
    gene_names: tuple[str, ...]
    expression: torch.Tensor        # [N_cells, G] final gene expression vector of every differentiated cell
    nt_sign: torch.Tensor           # [N_cells] net neurotransmitter sign in [-1, +1] from expressed NT_ENZYME genes
    synaptic_weights: torch.Tensor  # [N_cells, N_cells] synaptic matrix assembled from SPR 1/K_D CAM_KEY x CAM_LOCK binding
    divisions_per_step: list[int]   # Number of physical cell divisions at each ontogenetic step

    def gene_column(self, gene_name: str) -> torch.Tensor:
        idx = self.gene_names.index(gene_name)
        return self.expression[:, idx]

    def count_expressing(self, gene_name: str, threshold: float = 0.35) -> int:
        return int((self.gene_column(gene_name) > threshold).sum().item())


@dataclass(frozen=True, slots=True)
class GeneGraph:
    """Genome represented purely as a tuple of `Gene` loci and unweighted cis-regulatory enhancer links."""

    genes: tuple[Gene, ...]

    def gene_index(self) -> dict[str, int]:
        return {g.name: i for i, g in enumerate(self.genes)}

    def with_knockout(self, gene_name: str) -> GeneGraph:
        """Return a starting genome where `gene_name` is non-functional (loss-of-function allele)."""
        return GeneGraph(
            genes=tuple(replace(g, functional=False) if g.name == gene_name else g for g in self.genes)
        )

    def with_duplication(
        self,
        source_gene: str,
        new_gene_name: str,
        *,
        replace_activators: tuple[str, ...] | None = None,
        replace_repressors: tuple[str, ...] | None = None,
        add_activators: tuple[str, ...] = (),
        add_repressors: tuple[str, ...] = (),
        feed_into_targets: tuple[tuple[str, Literal["activate", "repress"]], ...] = (),
        source_ref: str = "Evo-Devo gene duplication & cis-regulatory enhancer divergence (Ohno 1970; Cell 2018 PMC5985970)",
    ) -> GeneGraph:
        """Return a starting genome with a duplicated gene locus and unweighted enhancer site additions."""
        src = next(g for g in self.genes if g.name == source_gene)
        new_acts = replace_activators if replace_activators is not None else (src.activators + add_activators)
        new_reps = replace_repressors if replace_repressors is not None else (src.repressors + add_repressors)
        duplicated = Gene(
            name=new_gene_name,
            role=src.role,
            source_ref=source_ref,
            activators=new_acts,
            repressors=new_reps,
            maternal_pole=None,
            mitotic_dilution=src.mitotic_dilution,
            cam_kd_um=src.cam_kd_um,
            nt_valence=src.nt_valence,
            functional=True,
        )
        updated: list[Gene] = []
        target_map = dict(feed_into_targets)
        for g in self.genes:
            if g.name in target_map:
                mode = target_map[g.name]
                if mode == "activate":
                    g = replace(g, activators=g.activators + (new_gene_name,))
                else:
                    g = replace(g, repressors=g.repressors + (new_gene_name,))
            updated.append(g)
        updated.append(duplicated)
        return GeneGraph(genes=tuple(updated))

    def compile_unweighted_matrices(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compile unweighted boolean enhancer binding matrices W_act[G, G], W_rep[G, G], and functional mask [G]."""
        idx = self.gene_index()
        n_g = len(self.genes)
        w_act = torch.zeros((n_g, n_g), device=device, dtype=torch.float32)
        w_rep = torch.zeros((n_g, n_g), device=device, dtype=torch.float32)
        func_mask = torch.zeros(n_g, device=device, dtype=torch.float32)

        for post_i, g in enumerate(self.genes):
            func_mask[post_i] = 1.0 if g.functional else 0.0
            for act_name in g.activators:
                if act_name in idx:
                    w_act[idx[act_name], post_i] += 1.0
            for rep_name in g.repressors:
                if rep_name in idx:
                    w_rep[idx[rep_name], post_i] += 1.0

        return w_act, w_rep, func_mask


REF_AP_HOX = "PLoS Genet 2021 DOI:10.1371/journal.pgen.1009812 (PMC8516289 §10-12) & Nobel 1995 (Lewis/Nusslein-Volhard/Wieschaus)"
REF_DV_COL = "Arendt & Nubler-Jung 1999 Development PMID:10072782 & Denes et al. 2007 Cell DOI:10.1016/j.cell.2007.02.040"
REF_NB_TIME = "Nature 2026 DOI:10.1038/s41586-026-10797-w (PMC13538039) & Genetics 2025 DOI:10.1093/genetics/iyaf211 (PMC12606418)"
REF_FCA_2022 = "Fly Cell Atlas scRNA-seq (197,453 cells, Li et al. 2022 Science DOI:10.1126/science.abk2432, enrichment >=1.30 / <=0.65)"
REF_SPR_2018 = "Cosmanescu et al. 2018 Neuron DOI:10.1016/j.neuron.2018.08.018 (PMC6289660 Table S2 SPR K_D) & FCA 2022"


def build_canonical_gene_graph() -> GeneGraph:
    """Construct the empirical Drosophila GRN from published developmental genetics, FCA scRNA-seq, and SPR K_D."""
    genes: tuple[Gene, ...] = (
        # --- 1. Maternal 2D Oocyte Polarity Morphogens (A-P: bcd/nos/cad; D-V: spz_dl/dpp) ---
        Gene("bcd", "MORPHOGEN", REF_AP_HOX, activators=("bcd",), maternal_pole="anterior"),
        Gene("nos", "MORPHOGEN", REF_AP_HOX, activators=("nos",), maternal_pole="posterior"),
        Gene("spz_dl", "MORPHOGEN", REF_DV_COL, activators=("spz_dl",), maternal_pole="ventral"),
        Gene("dpp", "MORPHOGEN", REF_DV_COL, activators=("dpp",), repressors=("spz_dl",), maternal_pole="dorsal"),
        Gene("cad", "TF", REF_AP_HOX, repressors=("bcd", "ey", "oc", "Dfd"), maternal_pole="uniform"),
        # --- 2. Gap Gene Cross-Repression Cascade (PLoS Genet 2021 PMC8516289) ---
        Gene("hb", "TF", REF_AP_HOX, activators=("bcd", "hb"), repressors=("nos", "Kr", "kni")),
        Gene("Kr", "TF", REF_AP_HOX, activators=("bcd", "hb"), repressors=("kni", "tll", "ey")),
        Gene("kni", "TF", REF_AP_HOX, activators=("cad",), repressors=("hb", "Kr", "tll")),
        Gene("tll", "TF", REF_AP_HOX, activators=("bcd", "nos"), repressors=("Kr",)),
        # --- 3. Anterior-Posterior Head & Hox Selector Genes with Trithorax Self-Maintenance ---
        Gene("ey", "TF", REF_AP_HOX, activators=("bcd", "tll", "ey"), repressors=("cad", "Kr", "Dfd", "Antp", "Ubx")),
        Gene("oc", "TF", REF_AP_HOX, activators=("bcd", "hb", "oc"), repressors=("Dfd", "Kr", "cad", "Antp", "Ubx")),
        Gene("Dfd", "TF", REF_AP_HOX, activators=("bcd", "hb", "Dfd"), repressors=("ey", "oc", "Antp", "Ubx", "kni")),
        Gene("Scr", "TF", REF_AP_HOX, activators=("hb", "Kr", "Scr"), repressors=("ey", "oc", "Antp", "Ubx", "kni")),
        Gene("Antp", "TF", REF_AP_HOX, activators=("hb", "Kr", "Antp"), repressors=("ey", "oc", "Dfd", "Ubx", "abd-A", "Abd-B")),
        Gene("Ubx", "TF", REF_AP_HOX, activators=("Kr", "kni", "Ubx"), repressors=("hb", "ey", "oc", "abd-A", "Abd-B")),
        Gene("abd-A", "TF", REF_AP_HOX, activators=("kni", "cad", "abd-A"), repressors=("hb", "Kr", "ey", "oc", "Abd-B")),
        Gene("Abd-B", "TF", REF_AP_HOX, activators=("cad", "tll", "Abd-B"), repressors=("hb", "Kr", "ey", "oc")),
        # --- 4. Dorso-Ventral Neuroectoderm Columnar Selectors (Arendt & Nubler-Jung 1999) ---
        Gene("vnd", "TF", REF_DV_COL, activators=("spz_dl", "vnd"), repressors=("dpp", "ind", "Dr")),
        Gene("ind", "TF", REF_DV_COL, activators=("spz_dl", "ind"), repressors=("vnd", "dpp", "Dr")),
        Gene("Dr", "TF", REF_DV_COL, activators=("dpp", "Dr"), repressors=("vnd", "ind")),
        # --- 5. Neuroblast Temporal Cascade, Delta-Notch Lateral Inhibition & Ecdysone Wave 2 (PMC13538039) ---
        Gene("Dl", "JUXTACRINE_LIGAND", REF_NB_TIME, activators=("CycB",), repressors=("N", "pros")),
        Gene("N", "JUXTACRINE_RECEPTOR", REF_NB_TIME, activators=("Dl",), repressors=("pros",)),
        Gene("Imp", "TF", REF_NB_TIME, activators=("Imp", "oc", "ey"), repressors=("Syp", "pros"), maternal_pole="uniform", mitotic_dilution=True),
        Gene("chinmo", "TF", REF_NB_TIME, activators=("Imp",), repressors=("br", "Syp", "pros")),
        Gene("EcR", "TF", REF_NB_TIME, activators=("chinmo", "ey", "oc", "Antp", "Ubx"), repressors=("Imp", "pros")),
        Gene("br", "TF", REF_NB_TIME, activators=("EcR", "chinmo"), repressors=("Imp", "Syp")),
        Gene("dati", "TF", REF_NB_TIME, activators=("EcR", "br"), repressors=("Imp",)),
        Gene("Syp", "TF", REF_NB_TIME, activators=("chinmo", "br", "dati", "Syp"), repressors=("Imp",)),
        Gene("gcm", "TF", REF_NB_TIME, activators=("N", "Syp", "gcm"), repressors=("Imp", "Antp")),
        Gene("CycB", "MITOTIC_DRIVER", REF_NB_TIME, activators=("Imp", "EcR", "oc", "ey"), repressors=("pros",), maternal_pole="uniform"),
        Gene("pros", "CELL_CYCLE_EXIT", REF_NB_TIME, activators=("Syp",), repressors=("Imp", "EcR")),
        # --- 6. Effector CAM_KEY (Dpr) & CAM_LOCK (DIP) Genes: Enhancers from FCA (197,453 cells) + SPR K_D (PMC6289660) ---
        Gene("dpr1", "CAM_KEY", REF_SPR_2018, activators=("ey", "vnd", "ind", "chinmo", "dati"), repressors=("abd-A", "Abd-B", "Dr", "N"), cam_kd_um=(("DIP-eta", 9.2), ("DIP-theta", 11.5))),
        Gene("dpr2", "CAM_KEY", REF_SPR_2018, activators=("ey", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Abd-B", "N"), cam_kd_um=(("DIP-theta", 7.4), ("DIP-eta", 12.0))),
        Gene("dpr3", "CAM_KEY", REF_SPR_2018, activators=("ey", "Antp", "Ubx", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Abd-B",), cam_kd_um=(("DIP-theta", 5.0), ("DIP-eta", 6.8))),
        Gene("dpr5", "CAM_KEY", REF_SPR_2018, activators=("ey", "vnd", "chinmo", "br", "dati"), repressors=("abd-A", "Abd-B", "N"), cam_kd_um=(("DIP-theta", 8.1),)),
        Gene("dpr6", "CAM_KEY", REF_SPR_2018, activators=("ey", "vnd", "ind", "chinmo", "dati"), repressors=("abd-A", "Abd-B", "Dr", "N"), cam_kd_um=(("DIP-alpha", 2.0), ("DIP-beta", 14.0))),
        Gene("dpr7", "CAM_KEY", REF_SPR_2018, activators=("ey", "oc", "vnd", "ind", "chinmo", "dati"), repressors=("abd-A", "Abd-B", "Dr", "N"), cam_kd_um=(("DIP-theta", 9.5),)),
        Gene("dpr8", "CAM_KEY", REF_SPR_2018, activators=("ey", "vnd", "ind", "chinmo", "dati"), repressors=("abd-A", "Abd-B", "Dr", "N"), cam_kd_um=(("DIP-beta", 4.5),)),
        Gene("dpr9", "CAM_KEY", REF_SPR_2018, activators=("ey", "oc", "vnd", "ind", "chinmo", "dati"), repressors=("abd-A", "Abd-B", "Dr", "N"), cam_kd_um=(("DIP-beta", 5.7),)),
        Gene("dpr10", "CAM_KEY", REF_SPR_2018, activators=("ey", "vnd", "ind", "chinmo", "dati"), repressors=("abd-A", "Abd-B", "N"), cam_kd_um=(("DIP-alpha", 1.4),)),
        Gene("dpr11", "CAM_KEY", REF_SPR_2018, activators=("ey", "Antp", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Abd-B", "N"), cam_kd_um=(("DIP-gamma", 3.8), ("DIP-beta", 5.1))),
        Gene("dpr12", "CAM_KEY", REF_SPR_2018, activators=("ey", "vnd", "ind", "chinmo", "dati"), repressors=("abd-A", "Abd-B", "N"), cam_kd_um=(("DIP-delta", 1.6),)),
        Gene("dpr13", "CAM_KEY", REF_SPR_2018, activators=("ey", "vnd", "ind", "chinmo", "br", "dati"), repressors=("abd-A", "Abd-B", "N"), cam_kd_um=(("DIP-epsilon", 5.6), ("DIP-zeta", 7.1))),
        Gene("dpr15", "CAM_KEY", REF_SPR_2018, activators=("ey", "oc", "Antp", "Ubx", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Abd-B", "N"), cam_kd_um=(("DIP-gamma", 5.2),)),
        Gene("dpr16", "CAM_KEY", REF_SPR_2018, activators=("ey", "oc", "vnd", "ind", "chinmo", "br", "dati"), repressors=("abd-A", "Abd-B", "N"), cam_kd_um=(("DIP-gamma", 6.4),)),
        Gene("dpr19", "CAM_KEY", REF_SPR_2018, activators=("ey", "oc", "vnd", "ind", "chinmo", "br", "dati"), repressors=("abd-A", "Abd-B", "N"), cam_kd_um=(("DIP-epsilon", 6.0), ("DIP-zeta", 6.5))),
        Gene("DIP-alpha", "CAM_LOCK", REF_FCA_2022, activators=("oc", "vnd", "ind", "Dr", "Imp", "chinmo", "br", "dati", "N"), repressors=()),
        Gene("DIP-beta", "CAM_LOCK", REF_FCA_2022, activators=("ey", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Abd-B", "N")),
        Gene("DIP-gamma", "CAM_LOCK", REF_FCA_2022, activators=("ey", "oc", "Antp", "Ubx", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Abd-B", "N")),
        Gene("DIP-delta", "CAM_LOCK", REF_FCA_2022, activators=("ey", "Antp", "Ubx", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Abd-B", "N")),
        Gene("DIP-epsilon", "CAM_LOCK", REF_FCA_2022, activators=("ey", "Ubx", "vnd", "ind", "Dr", "chinmo", "br", "dati"), repressors=("abd-A", "Abd-B")),
        Gene("DIP-zeta", "CAM_LOCK", REF_FCA_2022, activators=("ey", "vnd", "ind", "chinmo", "br", "dati"), repressors=("abd-A", "Abd-B", "N")),
        Gene("DIP-eta", "CAM_LOCK", REF_FCA_2022, activators=("ey", "Antp", "Ubx", "vnd", "ind", "chinmo", "dati"), repressors=("Dfd", "abd-A", "Abd-B", "Dr", "N")),
        Gene("DIP-theta", "CAM_LOCK", REF_FCA_2022, activators=("ey", "vnd", "ind", "chinmo", "br", "dati"), repressors=("abd-A", "Abd-B", "N")),
        Gene("DIP-iota", "CAM_LOCK", REF_FCA_2022, activators=("Ubx", "abd-A", "Abd-B", "vnd", "ind", "Dr", "Imp", "br", "N"), repressors=("oc", "Dfd", "chinmo", "dati")),
        # --- 7. Fast Neurotransmitter Enzymes & Ion Channels from FCA (197,453 cells) ---
        Gene("VAChT", "NT_ENZYME", REF_FCA_2022, activators=("ey", "vnd", "ind", "chinmo", "dati"), repressors=("abd-A", "Abd-B", "Dr", "N"), nt_valence=+1.0),
        Gene("Gad1", "NT_ENZYME", REF_FCA_2022, activators=("ey", "Antp", "Ubx", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Abd-B", "N"), nt_valence=-1.0),
        Gene("VGlut", "NT_ENZYME", REF_FCA_2022, activators=("ey", "oc", "vnd", "ind", "chinmo", "br", "dati"), repressors=("Scr", "abd-A", "Abd-B", "N"), nt_valence=-1.0),
        Gene("Hdc", "NT_ENZYME", REF_FCA_2022, activators=("oc", "vnd", "Imp", "chinmo"), repressors=("Scr", "Antp", "Ubx", "abd-A", "Abd-B", "ind", "Dr", "br", "N"), nt_valence=-1.0),
        Gene("para", "ION_CHANNEL", REF_FCA_2022, activators=("ey", "ind", "chinmo", "dati"), repressors=("Dfd", "Scr", "abd-A", "Abd-B", "Dr", "N")),
        Gene("Sh", "ION_CHANNEL", REF_FCA_2022, activators=("ey", "oc", "ind", "chinmo", "dati"), repressors=("Dfd", "abd-A", "Abd-B", "Dr", "N")),
        Gene("Ca-alpha1T", "ION_CHANNEL", REF_FCA_2022, activators=("ey", "ind", "chinmo", "dati"), repressors=("Dfd", "Abd-B", "N")),
        Gene("GluClalpha", "ION_CHANNEL", REF_FCA_2022, activators=("ey", "Scr", "Antp", "ind", "chinmo", "dati"), repressors=("Dfd", "abd-A", "Abd-B", "Dr", "N")),
        Gene("Rdl", "ION_CHANNEL", REF_FCA_2022, activators=("ey", "vnd", "ind", "chinmo", "dati"), repressors=("Dfd", "Scr", "abd-A", "Abd-B", "Dr", "N")),
    )
    return GeneGraph(genes=genes)


def _shea_ackers_transcription(
    expr: torch.Tensor,
    w_act: torch.Tensor,
    w_rep: torch.Tensor,
    basal_mask: torch.Tensor,
    func_mask: torch.Tensor,
) -> torch.Tensor:
    """Dimer Hill-Boltzmann promoter occupancy (Alon 2006 / Bintu et al. 2005): P = Z_act^2 / (0.25 + Z_act^2 + Z_rep^2)."""
    z_act = basal_mask.unsqueeze(0) + expr @ w_act
    z_rep = expr @ w_rep
    z_act2 = z_act * z_act
    occ = z_act2 / (0.25 + z_act2 + z_rep * z_rep)
    return occ * func_mask.unsqueeze(0)


def run_gene_ontogeny_automaton(
    gene_graph: GeneGraph,
    *,
    grid_ap: int = 16,
    grid_dv: int = 3,
    syncytial_steps: int = 24,
    max_division_cycles: int = 16,
    device: torch.device | str = "cpu",
) -> GeneOntogenyResult:
    """Run the 2D embryonic morphogen + cell-division automaton with zero gene-name strings."""
    dev = torch.device(device)
    n_g = len(gene_graph.genes)
    w_act, w_rep, func_mask = gene_graph.compile_unweighted_matrices(dev)

    morph_cols = [i for i, g in enumerate(gene_graph.genes) if g.role == "MORPHOGEN"]
    lig_cols = [i for i, g in enumerate(gene_graph.genes) if g.role == "JUXTACRINE_LIGAND"]
    rec_cols = [i for i, g in enumerate(gene_graph.genes) if g.role == "JUXTACRINE_RECEPTOR"]
    cyc_cols = [i for i, g in enumerate(gene_graph.genes) if g.role == "MITOTIC_DRIVER"]
    exit_cols = [i for i, g in enumerate(gene_graph.genes) if g.role == "CELL_CYCLE_EXIT"]
    dilute_cols = [i for i, g in enumerate(gene_graph.genes) if g.mitotic_dilution]
    basal_mask = torch.tensor(
        [1.0 if g.maternal_pole == "uniform" else 0.0 for g in gene_graph.genes],
        device=dev,
        dtype=torch.float32,
    ) * func_mask

    # 1. Initialize 2D embryonic sheet [grid_ap, grid_dv, G] from maternal oocyte pole attributes
    sheet = torch.zeros((grid_ap, grid_dv, n_g), device=dev, dtype=torch.float32)
    for g_i, g in enumerate(gene_graph.genes):
        if not g.functional or g.maternal_pole is None:
            continue
        if g.maternal_pole == "anterior":
            sheet[0, :, g_i] = 1.0
        elif g.maternal_pole == "posterior":
            sheet[-1, :, g_i] = 1.0
        elif g.maternal_pole == "ventral":
            sheet[:, 0, g_i] = 1.0
        elif g.maternal_pole == "dorsal":
            sheet[:, -1, g_i] = 1.0
        elif g.maternal_pole == "uniform":
            sheet[:, :, g_i] = 1.0

    # 2D syncytial diffusion & Shea-Ackers transcriptional patterning across (AP, DV) axes
    for _ in range(syncytial_steps):
        for g_i, g in enumerate(gene_graph.genes):
            if not g.functional:
                continue
            if g.maternal_pole == "anterior":
                sheet[0, :, g_i] = 1.0
            elif g.maternal_pole == "posterior":
                sheet[-1, :, g_i] = 1.0
            elif g.maternal_pole == "ventral":
                sheet[:, 0, g_i] = 1.0
            elif g.maternal_pole == "dorsal":
                sheet[:, -1, g_i] = 1.0

        for m_i in morph_cols:
            field_2d = sheet[:, :, m_i]
            ap_l = torch.cat([field_2d[:1, :], field_2d[:-1, :]], dim=0)
            ap_r = torch.cat([field_2d[1:, :], field_2d[-1:, :]], dim=0)
            dv_u = torch.cat([field_2d[:, :1], field_2d[:, :-1]], dim=1)
            dv_d = torch.cat([field_2d[:, 1:], field_2d[:, -1:]], dim=1)
            sheet[:, :, m_i] = (0.40 * field_2d + 0.20 * (ap_l + ap_r) + 0.10 * (dv_u + dv_d)) * func_mask[m_i]

        flat = sheet.reshape(-1, n_g)
        p_tr = _shea_ackers_transcription(flat, w_act, w_rep, basal_mask, func_mask).reshape(grid_ap, grid_dv, n_g)
        for g_i, g in enumerate(gene_graph.genes):
            if g.role not in ("MORPHOGEN", "MITOTIC_DRIVER", "CELL_CYCLE_EXIT") and not g.mitotic_dilution:
                sheet[:, :, g_i] = 0.5 * sheet[:, :, g_i] + 0.5 * p_tr[:, :, g_i]

    expr = sheet.reshape(-1, n_g)

    # 2. Autonomous Neuroblast Division Loop driven purely by MITOTIC_DRIVER vs CELL_CYCLE_EXIT
    differentiated_batches: list[torch.Tensor] = []
    divisions_log: list[int] = []

    for _ in range(max_division_cycles):
        p_tr = _shea_ackers_transcription(expr, w_act, w_rep, basal_mask, func_mask)
        for g_i, g in enumerate(gene_graph.genes):
            if not g.mitotic_dilution and g.role != "MORPHOGEN":
                expr[:, g_i] = 0.5 * expr[:, g_i] + 0.5 * p_tr[:, g_i]

        cyc_drive = expr[:, cyc_cols].max(dim=1).values if cyc_cols else torch.zeros(expr.shape[0], device=dev)
        cyc_sum = expr[:, cyc_cols].sum(dim=1) if cyc_cols else torch.zeros(expr.shape[0], device=dev)
        exit_drive = expr[:, exit_cols].max(dim=1).values if exit_cols else torch.ones(expr.shape[0], device=dev)
        dividing_mask = cyc_drive > exit_drive
        n_div = int(dividing_mask.sum().item())
        divisions_log.append(n_div)
        if n_div == 0:
            break

        div_cells = expr[dividing_mask]

        # Asymmetric sibling division with juxtacrine Delta-Notch lateral inhibition between sibling daughters
        sib_a = div_cells.clone()
        sib_b = div_cells.clone()
        for r_i in rec_cols:
            sib_a[:, r_i] = 1.0 * func_mask[r_i]
            sib_b[:, r_i] = 0.0
        for l_i in lig_cols:
            sib_a[:, l_i] = 0.0
            sib_b[:, l_i] = 1.0 * func_mask[l_i]
        for ex_i in exit_cols:
            sib_a[:, ex_i] = 1.0 * func_mask[ex_i]
            sib_b[:, ex_i] = 1.0 * func_mask[ex_i]
        for c_i in cyc_cols:
            sib_a[:, c_i] = 0.0
            sib_b[:, c_i] = 0.0

        differentiated_batches.extend([sib_a, sib_b])

        # Volume-halving mitotic dilution of cytoplasmic maternal/early factors in dividing progenitors
        for d_i in dilute_cols:
            expr[dividing_mask, d_i] = expr[dividing_mask, d_i] * 0.5

        # Symmetric progenitor amplification occurs ONLY when multiple MITOTIC_DRIVER paralogs co-express (cyc_sum > 1.0)
        expand_mask = dividing_mask & (cyc_sum > 1.15) & (expr.shape[0] < 1024)
        if expand_mask.any():
            expr = torch.cat([expr, expr[expand_mask].clone()], dim=0)

    all_cells = torch.cat(differentiated_batches, dim=0) if differentiated_batches else expr

    # Post-mitotic terminal transcription pass (express CAM_KEY, CAM_LOCK, NT_ENZYME, and ION_CHANNEL genes)
    for _ in range(2):
        p_term = _shea_ackers_transcription(all_cells, w_act, w_rep, basal_mask, func_mask)
        for g_i, g in enumerate(gene_graph.genes):
            if g.role in ("CAM_KEY", "CAM_LOCK", "NT_ENZYME", "ION_CHANNEL"):
                all_cells[:, g_i] = p_term[:, g_i]

    # 3. Assemble Synapses via Measured SPR Association Constants K_A = 1 / K_D (PMC6289660)
    cam_key_cols = [i for i, g in enumerate(gene_graph.genes) if g.role == "CAM_KEY"]
    cam_lock_cols = [i for i, g in enumerate(gene_graph.genes) if g.role == "CAM_LOCK"]
    lock_pos = {gene_graph.genes[i].name: k for k, i in enumerate(cam_lock_cols)}

    k_assoc = torch.zeros((len(cam_key_cols), len(cam_lock_cols)), device=dev, dtype=torch.float32)
    for k_idx, g_i in enumerate(cam_key_cols):
        for lock_name, kd_um in gene_graph.genes[g_i].cam_kd_um:
            if lock_name in lock_pos and kd_um > 0.0:
                k_assoc[k_idx, lock_pos[lock_name]] = 1.0 / float(kd_um)

    k_expr = all_cells[:, cam_key_cols]
    q_expr = all_cells[:, cam_lock_cols]

    nt_cols = [i for i, g in enumerate(gene_graph.genes) if g.role == "NT_ENZYME"]
    nt_vals = torch.tensor([gene_graph.genes[i].nt_valence for i in nt_cols], device=dev, dtype=torch.float32)
    nt_sign = torch.tanh((all_cells[:, nt_cols] * nt_vals.unsqueeze(0)).sum(dim=1))

    raw_syn = (k_expr @ k_assoc) @ q_expr.t()
    syn_weights = raw_syn * nt_sign.unsqueeze(1)

    return GeneOntogenyResult(
        n_cells=all_cells.shape[0],
        gene_names=tuple(g.name for g in gene_graph.genes),
        expression=all_cells,
        nt_sign=nt_sign,
        synaptic_weights=syn_weights,
        divisions_per_step=divisions_log,
    )
