"""L3 AL-NCA & SEZ-NCA: 50 Antennal Glomeruli, Peripheral Hunger Modulation, DA-WED Protein Switch, and Fdg."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from fly_nca.body.adapter import SensorIn
from fly_nca.core.glia import HemolymphState
from fly_nca.genome.banc_biophysics import (
    BANC_EMPIRICAL_BIOPHYSICS,
    BANC_SEZ_DN_EFFERENCE_FRACTION,
    BANC_SPIN_INHIBITORY_FRACTION,
)


@dataclass(frozen=True, slots=True)
class ChemosensoryDigest:
    """Normalized 50-glomerulus olfactory code + hunger-modulated gustatory & DA-WED state."""

    pn_glomeruli_50: torch.Tensor   # [50] Divisively normalized Antennal Lobe PN output
    stereo_odor_diff: float         # Left - Right antennal odor gradient
    sweet_reward_us: float          # Modulated Gr5a sweet signal (-> PAM DANs & Fdg)
    bitter_punish_us: float         # Modulated Gr66a bitter/nociceptive signal (-> PPL1 DANs)
    yeast_protein_drive: float      # DA-WED + Ir76b protein seeking drive (Open Biol 2019)
    fdg_proboscis_cmd: float        # SEZ Fdg feeding command (gated by hunger & crop stretch!)


class AntennalSEZNCA(nn.Module):
    """50-Glomerulus Antennal Lobe NCA (divisive normalization via LN pool) + SEZ Gustatory/Feeding NCA."""

    def __init__(self, n_glomeruli: int = 50, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.n_glomeruli = int(n_glomeruli)
        self.device = torch.device(device)
        self.dtype = dtype
        self.register_buffer("pn_state", torch.zeros(self.n_glomeruli, device=self.device, dtype=dtype))

    def step_chemosensory(
        self,
        sensor: SensorIn,
        hemolymph: HemolymphState,
        sigma_ln: float | None = None,
    ) -> ChemosensoryDigest:
        """Compute divisive normalization across 50 AL glomeruli and peripheral hunger gating in SEZ."""
        al_pn_bio = BANC_EMPIRICAL_BIOPHYSICS["antennal_lobe_projection_neuron"]
        al_ln_bio = BANC_EMPIRICAL_BIOPHYSICS["antennal_lobe_local_neuron"]
        gust_bio = BANC_EMPIRICAL_BIOPHYSICS["taste_bristle_gustatory_neuron"]
        if sigma_ln is None:
            # Measured inhibitory LN synapse share on AL projection neurons in BANC v888 (0.221)
            sigma_ln = al_pn_bio.inh_share
        odor_lr = sensor.odor_glomeruli_lr.to(self.device, dtype=self.dtype)
        mean_orn = 0.5 * (odor_lr[0] + odor_lr[1]).clamp(min=0.0)
        stereo_diff = float((odor_lr[0].sum() - odor_lr[1].sum()).item())

        # Divisive normalization via global GABAergic Local Interneuron (LN) web in AL:
        # Exponent = 1.0 + al_pn_bio.port_axon_dendrite (1.490 from BANC v888!)
        exp_al = 1.0 + al_pn_bio.port_axon_dendrite
        num = torch.pow(mean_orn + 1e-6, exp_al)
        den = (sigma_ln ** exp_al) + torch.pow(mean_orn.mean() + 1e-6, exp_al)
        pn_norm = (num / den).clamp(0.0, 1.0)
        self.pn_state = pn_norm

        # Peripheral & Central Hunger Modulation (Open Biol 2019, PMC6451361):
        # 77.1% axo-axonic synapses on taste_bristle_gustatory_neuron terminals (gust_bio.port_axon_axon = 0.771)!
        hunger = max(0.0, 1.0 - hemolymph.sugar_level)
        protein_hunger = max(0.0, 1.0 - hemolymph.amino_acid_level)

        raw_sweet = float(sensor.taste_sweet_bitter_amino[0].item())
        raw_bitter = float(sensor.taste_sweet_bitter_amino[1].item())
        raw_amino = float(sensor.taste_sweet_bitter_amino[2].item())

        sweet_mod = min(
            1.0,
            raw_sweet
            * (gust_bio.ach_share + (1.0 + gust_bio.ach_share) * hunger)
            * (1.0 - al_pn_bio.port_axon_dendrite * protein_hunger),
        )
        bitter_mod = max(0.0, raw_bitter * (1.0 - gust_bio.port_axon_axon * hunger))

        # DA-WED plastic switch: amino-acid deprivation boosts yeast/protein channel (FB-LAL DopR2)
        # while suppressing sugar preference (PLP DopR1)!
        yeast_drive = (
            min(1.0, (raw_amino + gust_bio.ach_share) * (gust_bio.ie_ratio * 0.75 * protein_hunger))
            if protein_hunger > gust_bio.ach_share
            else (raw_amino * al_pn_bio.inh_share)
        )

        # SEZ Fdg feeding command neuron: triggered by sweet/yeast only when hungry AND crop is not full
        # (BANC local visceral loop: crop stretch > al_ln_bio.ach_share = 0.853 inhibits pharyngeal/proboscis MNs!)
        crop_inhibition = 1.0 if sensor.crop_stretch > al_ln_bio.ach_share else 0.0
        fdg_cmd = 0.0
        if crop_inhibition < 0.5 and (hunger > al_ln_bio.port_dendrite_dendrite or protein_hunger > al_ln_bio.port_dendrite_dendrite):
            if sweet_mod > al_pn_bio.inh_share or (protein_hunger > gust_bio.ach_share and raw_amino > al_pn_bio.inh_share):
                fdg_cmd = 1.0

        return ChemosensoryDigest(
            pn_glomeruli_50=pn_norm,
            stereo_odor_diff=stereo_diff,
            sweet_reward_us=sweet_mod,
            bitter_punish_us=bitter_mod,
            yeast_protein_drive=yeast_drive,
            fdg_proboscis_cmd=fdg_cmd,
        )
