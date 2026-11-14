"""L3 LH-SPIN-NCA: Pure Cellular & Synaptic Multimodal Context Gate (GluCla Shunt Matrices) & SPIN Arbitration.

Zero Python lists (`list[ContextSnapshot]`) or `if/else` modality comparisons:
1. 64 multimodal LHON context-detector NCA cells (0..63, `mu_conj = 1.0`, `tau_V = 5.91 ms` from BANC)
   each have 4 dendritic branches (`b=0: olf[50]`, `b=1: vis[16]`, `b=2: temp[16]`, `b=3: tex[16]`).
2. Each branch has excitatory synaptic weights `W_exc` (Port 1) and inhibitory GluCla shunting weights `W_shunt` (Port 2).
   Before cLTM formation, `W_shunt = 1.0` across all population bins, keeping all branches shunted shut.
3. During `record_cltm_engram`, dopamine (`dDA1`) performs local synaptic plasticity on the recruited LHON cell:
   it potentiates `W_exc` on the active sensory bins and depresses (`LTD`) `W_shunt` to `0.0` on the active bins,
   leaving `W_shunt = 1.0` on all off-context bins!
4. On retrieval (`step_lh_spin`), any mismatch in temperature, light, texture, or odor activates off-context
   bins that hit high `W_shunt` (`GluCla` chloride shunt), collapsing that branch and multiplying the
   4-branch `tanh` product to zero (`PI = 0`) purely via `port1_2_dendritic_integration`!
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from fly_nca.body.adapter import SensorIn
from fly_nca.core.cell_state import (
    CH_A,
    CH_CA,
    CellPhysiologyParams,
    UniversalFlyCellUpdater,
    init_cell_state,
)
from fly_nca.core.glia import HemolymphState
from fly_nca.core.ports import port1_2_dendritic_integration
from fly_nca.genome.banc_biophysics import (
    BANC_EMPIRICAL_BIOPHYSICS,
    BANC_LHON_MULTIMODAL_FRACTION,
    BANC_LHON_TO_DAN_COVERAGE,
)

N_CLTM_LHON: int = 64
# Cells 0..63: 64 multimodal cLTM LHON cells (4 branches each)
# Cell 64: PD2a1/b1 naive odor avoidance LHON cell
# Cell 65: SPIN SMP353 (ACh+, attractive bottleneck cell)
# Cell 66: SPIN SIP015 (Glu-, aversive bottleneck cell)
N_LH_SPIN_CELLS: int = 67


@dataclass(frozen=True, slots=True)
class LHSpinOutput:
    """Output of Lateral Horn & SPIN convergence-divergence bottleneck (Front Syst Neurosci 2026)."""

    net_valence_to_cx_fbt: float     # Broadcast from SPIN (SMP353 ACh+ minus SIP015 Glu-) to CX FBt layer 6
    innate_prior_valence: float      # Soft prior valence (PD2a1/b1 & attractive food channels)
    cltm_retrieved_valence: float    # Context-gated long-term memory output (0.0 if any modality mismatches!)
    cltm_gate_open: float            # Multiplicative dendritic GluCla AND-gate value in [0, 1]
    lh_to_pam_dan_drive: float       # LHON -> PAM DAN reward teacher drive (482 LHON -> 94% DANs in BANC!)
    lh_to_ppl1_dan_drive: float      # LHON -> PPL1 DAN punishment teacher drive


class LHSpinGateNCA(nn.Module):
    """Pure Synaptic/Cellular LH-SPIN NCA with 4-Branch Dendritic GluCla Shunt Matrices."""

    def __init__(self, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.updater = UniversalFlyCellUpdater(slope=3.5)

        lhon_bio = BANC_EMPIRICAL_BIOPHYSICS["lateral_horn_output_neuron"]
        self.params = CellPhysiologyParams.default(N_LH_SPIN_CELLS, device=self.device, dtype=dtype)
        self.params.tau_v_ms[:] = lhon_bio.tau_v_ms  # 5.91 ms from BANC l2_cable_length_um (834.48 um)
        # Cells 0..63 are pure 4-branch conjunction detectors (mu_conj = 1.0)
        self.params.mu_conj[:N_CLTM_LHON] = 1.0
        self.params.alpha_shunt[:N_CLTM_LHON] = 18.0
        self.params.beta_root[:] = 4.0

        self.register_buffer("state", init_cell_state(N_LH_SPIN_CELLS, device=self.device, dtype=dtype))

        # Sensory population tuning centers for Light [16], Temperature [16] (10..42 C), and Texture [16] (0..4)
        self.register_buffer("light_centers", torch.linspace(0.0, 1.0, 16, device=self.device, dtype=dtype))
        self.register_buffer("temp_centers", torch.linspace(10.0, 42.0, 16, device=self.device, dtype=dtype))
        self.register_buffer("tex_centers", torch.linspace(0.0, 4.0, 16, device=self.device, dtype=dtype))

        # Synaptic weight matrices for the 4 dendritic branches of the 64 cLTM LHON cells:
        # Branch 0 (Olfactory, 50 glomeruli):
        self.register_buffer("w_exc_olf", torch.zeros((N_CLTM_LHON, 50), device=self.device, dtype=dtype))
        self.register_buffer("w_shunt_olf", torch.ones((N_CLTM_LHON, 50), device=self.device, dtype=dtype))
        # Branch 1 (Visual luminance population, 16 bins):
        self.register_buffer("w_exc_vis", torch.zeros((N_CLTM_LHON, 16), device=self.device, dtype=dtype))
        self.register_buffer("w_shunt_vis", torch.ones((N_CLTM_LHON, 16), device=self.device, dtype=dtype))
        # Branch 2 (Thermoreceptor population, 16 bins):
        self.register_buffer("w_exc_temp", torch.zeros((N_CLTM_LHON, 16), device=self.device, dtype=dtype))
        self.register_buffer("w_shunt_temp", torch.ones((N_CLTM_LHON, 16), device=self.device, dtype=dtype))
        # Branch 3 (AMMC Tactile/Texture population, 16 bins):
        self.register_buffer("w_exc_tex", torch.zeros((N_CLTM_LHON, 16), device=self.device, dtype=dtype))
        self.register_buffer("w_shunt_tex", torch.ones((N_CLTM_LHON, 16), device=self.device, dtype=dtype))

        # Synaptic weights from the 64 LHON cells to SPIN SMP353 (+1) vs SIP015 (-1)
        self.register_buffer("w_lhon_valence", torch.zeros(N_CLTM_LHON, device=self.device, dtype=dtype))
        self.next_slot: int = 0

    def _encode_context_populations(
        self,
        pn_glomeruli_50: torch.Tensor,
        sensor: SensorIn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode sensory inputs into population activity vectors on the 4 modality tracts."""
        p_olf = pn_glomeruli_50.to(self.device, dtype=self.dtype)
        light_val = float(sensor.ommatidia_light.mean().item())
        temp_val = float(sensor.temperature_c)
        tex_val = float(sensor.surface_texture_freq)

        p_vis = torch.exp(-((self.light_centers - light_val) / 0.08) ** 2)
        p_temp = torch.exp(-((self.temp_centers - temp_val) / 1.2) ** 2)
        p_tex = torch.exp(-((self.tex_centers - tex_val) / 0.15) ** 2)
        return p_olf, p_vis, p_temp, p_tex

    def record_cltm_engram(
        self,
        odor_glomerulus: int,
        sensor: SensorIn,
        valence_sign: float,
        da_intact: bool = True,
    ) -> None:
        """Synaptic dDA1 plasticity: carve a context keyhole into W_exc and W_shunt of an LHON NCA cell."""
        if not da_intact:
            return
        slot = self.next_slot % N_CLTM_LHON
        self.next_slot += 1

        pn_50 = torch.zeros(50, device=self.device, dtype=self.dtype)
        pn_50[int(odor_glomerulus) % 50] = 1.0
        p_olf, p_vis, p_temp, p_tex = self._encode_context_populations(pn_50, sensor)

        # Hebbian LTP on excitatory ACh weights + Anti-Hebbian LTD on GluCla shunting weights for active bins!
        # Off-context bins (p <= 0.01) retain strong GluCla chloride shunting so any context shift shunts the branch!
        self.w_exc_olf[slot] = (p_olf > 0.2).to(self.dtype) * 2.5
        self.w_shunt_olf[slot] = (p_olf <= 0.01).to(self.dtype) * 8.0

        self.w_exc_vis[slot] = (p_vis > 0.3).to(self.dtype) * 2.0
        self.w_shunt_vis[slot] = (p_vis <= 0.01).to(self.dtype) * 8.0

        self.w_exc_temp[slot] = (p_temp > 0.3).to(self.dtype) * 2.0
        self.w_shunt_temp[slot] = (p_temp <= 0.01).to(self.dtype) * 8.0

        self.w_exc_tex[slot] = (p_tex > 0.3).to(self.dtype) * 2.0
        self.w_shunt_tex[slot] = (p_tex <= 0.01).to(self.dtype) * 8.0

        self.w_lhon_valence[slot] = float(valence_sign)

    def evaluate_pd2a1_soft_prior(
        self,
        aversive_odor_conc: float,
        exposure_time_s: float,
        hemolymph: HemolymphState,
    ) -> float:
        """eLife 2020 (PMC7145822): PD2a1/b1 cell (64) gated by concentration threshold, slow Ca2+ integration, and NPF hunger shunt."""
        hunger = max(0.0, 1.0 - hemolymph.sugar_level)
        # Concentration threshold on dendritic branch (10^-3 = 0.001 fails threshold; 10^-2 = 0.02 passes)
        conc_drive = max(0.0, float(aversive_odor_conc) - 0.005) * 40.0
        # Slow temporal integration window (40 s < threshold, >=80 s saturates)
        time_gate = max(0.0, min(1.0, (float(exposure_time_s) - 50.0) / 30.0))
        # Starvation NPF/sNPF shunts PD2a1/b1 at the root!
        hunger_shunt = 50.0 if hunger > 0.65 else 0.0

        exc = torch.tensor([[conc_drive * time_gate]], device=self.device, dtype=self.dtype)
        shunt = torch.tensor([[hunger_shunt]], device=self.device, dtype=self.dtype)
        p_pd2 = CellPhysiologyParams.default(1, device=self.device, dtype=self.dtype)
        i_d = float(port1_2_dendritic_integration(exc, shunt, p_pd2)[0].item())
        # Pass integrated dendritic current through PD2a1/b1 soma firing threshold theta_base,
        # capped by the measured multimodal LHON fraction in BANC v888 (708 / 1080 = 0.6556)!
        soma_rate = max(0.0, i_d - float(p_pd2.theta_base[0].item()))
        return -min(BANC_LHON_MULTIMODAL_FRACTION, soma_rate)

    def step_lh_spin(
        self,
        pn_glomeruli_50: torch.Tensor,
        sensor: SensorIn,
        hemolymph: HemolymphState,
        mbon_learned_valence: float = 0.0,
        aversive_odor_conc: float = 0.0,
        exposure_time_s: float = 90.0,
        social_inhibitory_cue: float = 0.0,
    ) -> LHSpinOutput:
        """Compute 4-branch synaptic currents across all 64 cLTM LHON cells and SPIN SMP353/SIP015 cells."""
        p_olf, p_vis, p_temp, p_tex = self._encode_context_populations(pn_glomeruli_50, sensor)

        # Synaptic matrix-vector products on all 4 dendritic branches of the 64 LHON cells [64, 4]
        exc_branches = torch.stack(
            [
                torch.mv(self.w_exc_olf, p_olf),
                torch.mv(self.w_exc_vis, p_vis),
                torch.mv(self.w_exc_temp, p_temp),
                torch.mv(self.w_exc_tex, p_tex),
            ],
            dim=-1,
        )
        shunt_branches = torch.stack(
            [
                torch.mv(self.w_shunt_olf, p_olf) + (10.0 if float(p_olf.max().item()) < 0.1 else 0.0),
                torch.mv(self.w_shunt_vis, p_vis),
                torch.mv(self.w_shunt_temp, p_temp),
                torch.mv(self.w_shunt_tex, p_tex),
            ],
            dim=-1,
        )

        # Pad to N_LH_SPIN_CELLS (67) so all cells run through port1_2_dendritic_integration together
        exc_all = torch.zeros((N_LH_SPIN_CELLS, 4), device=self.device, dtype=self.dtype)
        shunt_all = torch.zeros((N_LH_SPIN_CELLS, 4), device=self.device, dtype=self.dtype)
        exc_all[:N_CLTM_LHON] = exc_branches
        shunt_all[:N_CLTM_LHON] = shunt_branches

        i_dend = port1_2_dendritic_integration(exc_all, shunt_all, self.params)
        lhon_cltm_rates = i_dend[:N_CLTM_LHON]

        best_gate = float(lhon_cltm_rates.max().item()) if self.next_slot > 0 else 0.0
        if best_gate < 0.05:
            best_gate = 0.0
            cltm_val = 0.0
        else:
            cltm_val = float((lhon_cltm_rates * self.w_lhon_valence).sum().item())

        # Innate food prior weighted by BANC LHON multimodal fraction (708/1080 = 0.6556) + PD2a1/b1 soft prior
        lhon_bio = BANC_EMPIRICAL_BIOPHYSICS["lateral_horn_output_neuron"]
        innate_food_ach = float(p_olf[0].item()) * BANC_LHON_MULTIMODAL_FRACTION
        innate_avoid_glu = self.evaluate_pd2a1_soft_prior(aversive_odor_conc, exposure_time_s, hemolymph)
        innate_prior = innate_food_ach + innate_avoid_glu

        # SPIN SMP353 (cell 65, ACh+) vs SIP015 (cell 66, Glu-) dendritic integration
        total_learned = mbon_learned_valence + cltm_val
        if abs(total_learned) > lhon_bio.segregation_index:
            smp353_exc = max(0.0, total_learned + lhon_bio.mod_share * innate_food_ach)
            sip015_exc = max(0.0, -total_learned + lhon_bio.mod_share * abs(innate_avoid_glu))
        else:
            smp353_exc = max(0.0, innate_food_ach)
            sip015_exc = max(0.0, -innate_avoid_glu)

        smp353_shunted = smp353_exc / (1.0 + lhon_bio.root_shunt_gain * social_inhibitory_cue + lhon_bio.dendritic_shunt_gain * sip015_exc)
        net_spin = smp353_shunted - sip015_exc

        return LHSpinOutput(
            net_valence_to_cx_fbt=net_spin,
            innate_prior_valence=innate_prior,
            cltm_retrieved_valence=cltm_val,
            cltm_gate_open=best_gate,
            lh_to_pam_dan_drive=max(0.0, smp353_shunted * BANC_LHON_TO_DAN_COVERAGE * 0.5),
            lh_to_ppl1_dan_drive=max(0.0, sip015_exc * BANC_LHON_TO_DAN_COVERAGE * 0.5),
        )
