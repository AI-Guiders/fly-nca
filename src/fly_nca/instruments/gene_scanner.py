"""External Single-Cell Transcriptome & Connectome Atlas Scanner (Observer Instrument).

Lives strictly in `fly_nca/instruments/` OUTSIDE the genomic cellular automaton (`fly_nca/genome/gene_network.py`):
- The genomic automaton (`run_gene_ontogeny_automaton`) knows nothing about `organ_id`, `segment_id`, or species names;
  it only steps gene expression `E[N, G]`, divides cells until `STOP_CYCLE`, and assembles synapses `W[N, N]` via `CAM_KEY x CAM_LOCK`.
- `TranscriptomeAtlasScanner` is the laboratory scRNA-seq / EM-atlas observer instrument that takes a grown
  `GeneOntogenyResult` (`expression[N, G]` and `synaptic_weights[N, N]`), applies marker-gene signatures to
  classify cell populations into anatomical domains (`OL`, `AL`, `MB`, `CX`, `SEZ`, `T1`, `T2_Wing`, `T3_Haltere`, `Abdomen`),
  computes the inter-domain connectome projection matrix, and matches the grown organism against species/mutant reference atlases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import torch

from fly_nca.genome.gene_network import GeneOntogenyResult


@dataclass(frozen=True, slots=True)
class MarkerDomainSpec:
    """Single-cell RNA-seq marker-gene signature for an anatomical domain or cell population."""

    domain_id: str
    positive_markers: tuple[str, ...]
    negative_markers: tuple[str, ...] = ()
    threshold: float = 0.45


@dataclass(frozen=True, slots=True)
class SpeciesReferenceProfile:
    """Reference expectation for matching a grown transcriptome against a known species or mutant phenotype."""

    profile_id: str
    description: str
    expected_rules: tuple[tuple[str, str, float], ...]  # (domain_or_gene, op: '==' | '>' | '<' | 'ratio>', value)


@dataclass(frozen=True, slots=True)
class TranscriptomeScanReport:
    """Observer report produced by scanning a grown `GeneOntogenyResult` against marker signatures."""

    n_cells: int
    n_synapses: int
    excitatory_fraction: float
    inhibitory_fraction: float
    domain_counts: dict[str, int]
    domain_fractions: dict[str, float]
    gene_expression_counts: dict[str, int]
    inter_domain_synapses: dict[str, dict[str, int]]
    matched_phenotype: str
    detected_shifts: list[str]


CANONICAL_MARKER_DOMAINS: tuple[MarkerDomainSpec, ...] = (
    MarkerDomainSpec("OL_OpticLobe", positive_markers=("ey_Six3", "Dpr1"), negative_markers=("Antp", "Ubx")),
    MarkerDomainSpec("MB_Pallium", positive_markers=("otd_Otx", "Dpr3", "ChaT_ACh"), negative_markers=("Antp", "Ubx")),
    MarkerDomainSpec("AL_LH_Olfactory", positive_markers=("otd_Otx", "Dpr2"), negative_markers=("ey_Six3", "Antp")),
    MarkerDomainSpec("CX_BasalGanglia", positive_markers=("otd_Otx", "Dpr4"), negative_markers=("Dfd_Scr", "Antp")),
    MarkerDomainSpec("SEZ_Gnathal", positive_markers=("Dfd_Scr", "Dpr5"), negative_markers=("otd_Otx", "Antp")),
    MarkerDomainSpec("T1_Prothorax", positive_markers=("Antp", "Dpr6"), negative_markers=("Ubx",)),
    MarkerDomainSpec("T2_WingMesothorax", positive_markers=("Antp", "Dpr7"), negative_markers=("Ubx",)),
    MarkerDomainSpec("T3_HaltereMetathorax", positive_markers=("Ubx", "Dpr8"), negative_markers=("abd_A",)),
    MarkerDomainSpec("Abdomen_Visceral", positive_markers=("Dpr9",), negative_markers=("otd_Otx", "Antp")),
)


class TranscriptomeAtlasScanner:
    """External laboratory scanner that maps raw gene expression `E[N, G]` to anatomical domains and species phenotypes."""

    def __init__(self, domains: tuple[MarkerDomainSpec, ...] = CANONICAL_MARKER_DOMAINS) -> None:
        self.domains = domains

    def _cell_mask_for_domain(self, ontogeny: GeneOntogenyResult, spec: MarkerDomainSpec) -> torch.Tensor:
        mask = torch.ones(ontogeny.n_cells, device=ontogeny.expression.device, dtype=torch.bool)
        for pos_g in spec.positive_markers:
            if pos_g in ontogeny.gene_names:
                mask = mask & (ontogeny.gene_column(pos_g) > spec.threshold)
            else:
                return torch.zeros_like(mask)
        for neg_g in spec.negative_markers:
            if neg_g in ontogeny.gene_names:
                mask = mask & (ontogeny.gene_column(neg_g) <= spec.threshold)
        return mask

    def scan(
        self,
        ontogeny: GeneOntogenyResult,
        *,
        wild_type_baseline: TranscriptomeScanReport | None = None,
    ) -> TranscriptomeScanReport:
        """Scan a grown `GeneOntogenyResult` and classify cells, neurotransmitter balance, and phenotype."""
        n_total = max(1, ontogeny.n_cells)
        n_syn = int((ontogeny.synaptic_weights.abs() > 0.05).sum().item())
        exc_frac = float((ontogeny.nt_sign > 0.05).float().mean().item())
        inh_frac = float((ontogeny.nt_sign < -0.05).float().mean().item())

        domain_masks: dict[str, torch.Tensor] = {}
        domain_counts: dict[str, int] = {}
        domain_fractions: dict[str, float] = {}
        for d_spec in self.domains:
            m = self._cell_mask_for_domain(ontogeny, d_spec)
            cnt = int(m.sum().item())
            domain_masks[d_spec.domain_id] = m
            domain_counts[d_spec.domain_id] = cnt
            domain_fractions[d_spec.domain_id] = round(cnt / n_total, 4)

        gene_counts = {g_name: ontogeny.count_expressing(g_name) for g_name in ontogeny.gene_names}

        # Measure inter-domain synaptic connectivity directly from the CAM-assembled synaptic_weights[N, N]
        syn_active = (ontogeny.synaptic_weights.abs() > 0.05)
        inter_domain: dict[str, dict[str, int]] = {}
        for src_d, src_m in domain_masks.items():
            if not src_m.any():
                continue
            row_dict: dict[str, int] = {}
            sub_rows = syn_active[src_m]
            for dst_d, dst_m in domain_masks.items():
                if not dst_m.any():
                    continue
                c_edges = int(sub_rows[:, dst_m].sum().item())
                if c_edges > 0:
                    row_dict[dst_d] = c_edges
            if row_dict:
                inter_domain[src_d] = row_dict

        # Phenotype & evolutionary shift classification by comparing marker domains
        shifts: list[str] = []
        n_ubx = gene_counts.get("Ubx", 0)
        n_antp = gene_counts.get("Antp", 0)
        n_otx = gene_counts.get("otd_Otx", 0)
        n_ecr = gene_counts.get("EcR_20E", 0)
        has_notch2nl = gene_counts.get("NOTCH2NL", 0) > 0

        if n_ubx == 0 and n_antp > 400:
            phenotype = "drosophila.melanogaster.ubx_bithorax_homeotic_4wing"
            shifts.append("Homeotic transformation T3 -> T2: Ubx+/Dpr8+ haltere domain absent (0 cells); Antp+/Dpr7+ wing domain duplicated.")
        elif has_notch2nl or n_otx > 2000:
            phenotype = "chordata.pallium_notch2nl_expanded"
            shifts.append(f"Forebrain heterochronic expansion: otd_Otx+ / Dpr3+ pallial domain expanded to {n_otx} cells via delayed STOP_CYCLE.")
        elif n_ecr == 0 and ontogeny.n_cells < 1200:
            phenotype = "urbilateria.primary_wave1_worm"
            shifts.append("Primary-wave-only ontogeny: EcR_20E metamorphosis absent; cells exited mitosis after primary embryonic wave.")
        else:
            phenotype = "drosophila.melanogaster.wild_type"

        if wild_type_baseline is not None:
            for d_id, cnt in domain_counts.items():
                wt_cnt = wild_type_baseline.domain_counts.get(d_id, 0)
                if wt_cnt > 0 and cnt == 0:
                    shifts.append(f"Domain loss vs WT: {d_id} dropped from {wt_cnt} -> 0 cells.")
                elif wt_cnt == 0 and cnt > 0:
                    shifts.append(f"De novo domain appearance vs WT: {d_id} gained {cnt} cells.")
                elif wt_cnt > 0 and cnt >= 2 * wt_cnt:
                    shifts.append(f"Domain expansion vs WT: {d_id} expanded {cnt / wt_cnt:.2f}x ({wt_cnt} -> {cnt} cells).")

        return TranscriptomeScanReport(
            n_cells=ontogeny.n_cells,
            n_synapses=n_syn,
            excitatory_fraction=round(exc_frac, 4),
            inhibitory_fraction=round(inh_frac, 4),
            domain_counts=domain_counts,
            domain_fractions=domain_fractions,
            gene_expression_counts=gene_counts,
            inter_domain_synapses=inter_domain,
            matched_phenotype=phenotype,
            detected_shifts=shifts,
        )
