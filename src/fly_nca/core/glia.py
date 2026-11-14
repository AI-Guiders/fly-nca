"""Port 5 & Glial Substrate: SPG (BBB), EG (Compartment Capsules), and ALG (Astrocyte Clearance & Sleep)."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

# Volume modulator indices inside each EG-insulated compartment [N_comp, 7]
MOD_DA_POS: int = 0   # PAM Dopamine (reward / appetitive teacher)
MOD_DA_NEG: int = 1   # PPL1 Dopamine (punishment / aversive teacher & wake drive)
MOD_OA: int = 2       # Octopamine (sweet taste / arousal / aggression VPM4)
MOD_5HT: int = 3      # Serotonin (feeding / MP1 modulation)
MOD_NPF: int = 4      # Neuropeptide F / sNPF (hunger gate — suppresses naive aversion)
MOD_AKH: int = 5      # Adipokinetic hormone (glucagon-like starvation / hyperactivity)
MOD_DILP: int = 6     # Drosophila Insulin-Like Peptides (satiety signal from IPCs)

NUM_MOD_CHANNELS: int = 7


@dataclass
class HemolymphState:
    """Internal bodily milieu filtered into the CNS through Subperineurial Glia (SPG / BBB)."""

    sugar_level: float = 0.80      # Hemolymph trehalose/glucose in [0, 1] (drops during locomotion)
    amino_acid_level: float = 0.80 # Protein/yeast reserve in [0, 1] (drives DA-WED when low)
    osmolarity: float = 0.20       # Thirst signal in [0, 1]
    ecdysone_20e: float = 0.0      # 20-hydroxyecdysone hormone pulse (triggers Wave 2 metamorphosis)


class GlialSubstrateNCA(nn.Module):
    """Three glial classes (12,750 glia in BANC v888: SPG, EG, ALG) governing Port 5."""

    def __init__(self, n_compartments: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.n_compartments = int(n_compartments)
        # EG-insulated compartment modulator concentrations m_c(t) [N_comp, 7]
        self.register_buffer(
            "compartment_mods",
            torch.zeros((n_compartments, NUM_MOD_CHANNELS), device=device, dtype=dtype),
        )
        # ALG astrocyte-accumulated global/regional sleep pressure in [0, 1]
        self.register_buffer(
            "alg_sleep_need",
            torch.zeros((n_compartments,), device=device, dtype=dtype),
        )

    def inject_compartment_modulator(self, comp_idx: int | torch.Tensor, mod_channel: int, amount: float | torch.Tensor) -> None:
        """Deposit DCV/synaptic modulator strictly inside EG-bounded compartment `comp_idx` (zero cross-compartment leak!)."""
        self.compartment_mods[comp_idx, mod_channel] = (
            self.compartment_mods[comp_idx, mod_channel] + amount
        ).clamp(0.0, 1.0)

    def step_glia(
        self,
        hemolymph: HemolymphState,
        cell_sleep_pressure: torch.Tensor,
        cell_compartment_id: torch.Tensor,
        is_sleeping: bool = False,
        dt_ms: float = 5.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update SPG (BBB peptide transport), EG (compartment isolation), and ALG (transmitter clearance & sleep)."""
        dt_s = float(dt_ms) * 1e-3
        # 1. ALG astrocyte clearance of extracellular DA/OA/5HT inside each EG capsule
        clearance_decay = max(0.0, 1.0 - 2.5 * dt_s)
        self.compartment_mods[:, :4] *= clearance_decay

        # 2. SPG blood-brain barrier translates hemolymph sugar into systemic NPF/AKH (hunger) vs DILP (satiety)
        hunger = max(0.0, 1.0 - hemolymph.sugar_level)
        satiety = hemolymph.sugar_level
        self.compartment_mods[:, MOD_NPF] = hunger
        self.compartment_mods[:, MOD_AKH] = max(0.0, (0.45 - hemolymph.sugar_level) * 2.0)
        self.compartment_mods[:, MOD_DILP] = satiety

        # 3. ALG integrates local neuronal synaptic fatigue into compartment sleep need
        if cell_sleep_pressure.numel() > 0:
            comp_ids = cell_compartment_id.clamp(0, self.n_compartments - 1)
            counts = torch.bincount(comp_ids, minlength=self.n_compartments).clamp(min=1).to(self.alg_sleep_need.dtype)
            sums = torch.zeros_like(self.alg_sleep_need).index_add_(0, comp_ids, cell_sleep_pressure)
            mean_fatigue = sums / counts
            if is_sleeping:
                self.alg_sleep_need = (self.alg_sleep_need - 0.08 * dt_s).clamp(0.0, 1.0)
            else:
                self.alg_sleep_need = (0.98 * self.alg_sleep_need + 0.02 * mean_fatigue).clamp(0.0, 1.0)

        # Return per-cell ATP supply and sleep clearance rates
        n_cells = cell_sleep_pressure.shape[0]
        device = cell_sleep_pressure.device
        dtype = cell_sleep_pressure.dtype
        atp_rate = (0.0020 if is_sleeping else 0.0005 * (0.3 + 0.7 * satiety)) * float(dt_ms)
        clear_rate = (0.0025 * float(dt_ms)) if is_sleeping else 0.0
        return (
            torch.full((n_cells,), atp_rate, device=device, dtype=dtype),
            torch.full((n_cells,), clear_rate, device=device, dtype=dtype),
        )
