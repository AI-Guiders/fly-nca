"""L3 MB-NCA: 4,552 KC x 15 Glial Compartments, 5% APL Sparse Hash, LTD, DN->DAN Efference, VPM4, & Sleep CREB."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from fly_nca.core.ports import port3_axo_axonic_presynaptic_gate
from fly_nca.genome.banc_biophysics import (
    BANC_EMPIRICAL_BIOPHYSICS,
    BANC_VNC_DN_EFFERENCE_FRACTION,
)

# 15 Compartments across 3 Lobes:
# 0..4:   gamma1..gamma5 (Fast STM, gamma1 = key aversive trace inhibiting gamma5!)
# 5..8:   alpha'1..beta'2 (Intermediate MTM)
# 9..14:  alpha1..beta3 (Consolidated Long-Term Memory LTM, CREB-dependent during Active Sleep)
N_MB_COMPARTMENTS: int = 15
COMP_GAMMA1: int = 0
COMP_GAMMA5: int = 4

# Compartment valence polarity of readout MBONs:
# Even gamma/alpha'/alpha compartments (including gamma1!) have Approach (+) MBONs
# whose depression (LTD) by aversive PPL1 DANs shifts net balance toward Avoidance (-)!
# Odd compartments (including gamma5!) have Avoid (-) MBONs whose depression by reward PAM DANs
# shifts net balance toward Approach (+)!


@dataclass(frozen=True, slots=True)
class MBStepOutput:
    """Output of Mushroom Body NCA on each cognitive frame."""

    kc_sparse_activity: torch.Tensor   # [N_KC] 5% sparse Kenyon cell pattern
    kc_active_fraction: float          # Verified ~0.05 sparsity
    mbon_rates: torch.Tensor           # [15] Readout rates of all 15 compartments
    net_learned_valence: float         # Approach - Avoidance balance (-> SPIN & 140 AN/DN)
    dan_prediction_error: torch.Tensor # [15] Active teacher signals (with DN efference copy!)
    aggression_boxing_drive: float     # VPM4 (OA+Glu) -| MBON-11 (GABA/GluCla) layered disinhibition


class MushroomBodyNCA(nn.Module):
    """Mushroom Body NCA: Calyx (5% APL shunt) + Subcellular [N_KC x 15] Compartment Matrix."""

    def __init__(
        self,
        n_kc: int = 4552,
        n_glomeruli: int = 50,
        claws_per_kc: int = 6,
        target_sparsity: float | None = None,
        seed: int = 42,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_kc = int(n_kc)
        self.n_glomeruli = int(n_glomeruli)
        # Default sparsity = measured I/E synapse ratio on Kenyon cells in BANC v888 (0.046 = 4.6%!)
        kc_bio = BANC_EMPIRICAL_BIOPHYSICS["kenyon_cell"]
        mbon_bio = BANC_EMPIRICAL_BIOPHYSICS["mushroom_body_output_neuron"]
        dan_bio = BANC_EMPIRICAL_BIOPHYSICS["mushroom_body_dopaminergic_neuron"]
        self.target_sparsity = float(target_sparsity) if target_sparsity is not None else kc_bio.ie_ratio
        self.device = torch.device(device)
        self.dtype = dtype

        # 1. Calyx random claw connectivity matrix W_claw [N_KC, 50] (~6 claws per KC with graded bouton weights)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        claw_mat = torch.zeros((self.n_kc, self.n_glomeruli), dtype=dtype)
        for k in range(self.n_kc):
            idx = torch.randperm(self.n_glomeruli, generator=gen)[:claws_per_kc]
            weights = 0.5 + 0.5 * torch.rand(claws_per_kc, generator=gen, dtype=dtype)
            claw_mat[k, idx] = weights / weights.sum()
        self.register_buffer("w_claw", claw_mat.to(self.device))

        # 2. Subcellular compartment synaptic weight matrix W_mb [N_KC, 15] initialized to 1.0
        self.register_buffer(
            "w_mb",
            torch.ones((self.n_kc, N_MB_COMPARTMENTS), device=self.device, dtype=dtype),
        )

        # Compartment learning rates eta_c anchored to measured DAN/MBON modulatory synapse shares in BANC:
        # gamma (0..4): dan_bio.ach_share (0.434) & dan_bio.mod_share (0.358)
        # alpha'/beta' (5..8): mbon_bio.mod_share * 0.5 (0.1185)
        # alpha/beta (9..14): kc_bio.mod_share (0.028)
        eta_gamma = dan_bio.ach_share
        eta_mtm = mbon_bio.mod_share * 0.5
        eta_ltm = kc_bio.mod_share
        eta = torch.tensor(
            [eta_gamma] * 5 + [eta_mtm] * 4 + [eta_ltm] * 6,
            device=self.device,
            dtype=dtype,
        )
        self.register_buffer("eta_learn", eta)

        # Sign of MBON behavioral valence per compartment (+1 = Approach MBON, -1 = Avoid MBON)
        # gamma1 (0) is Approach MBON-g1pedc>a/b (depressed by aversive PPL1-g1pedc!);
        # gamma5 (4) is Avoid MBON-g5 (inhibited feedforward by MBON-g1pedc, depressed by appetitive PAM!)
        mbon_sign = torch.tensor(
            [+1.0, -1.0, +1.0, -1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0, +1.0, -1.0],
            device=self.device,
            dtype=dtype,
        )
        self.register_buffer("mbon_valence_sign", mbon_sign)

        # Compartment ablation / shibire blockade mask [15] (Sci Rep 2022 test!)
        self.register_buffer("mbon_active_mask", torch.ones(N_MB_COMPARTMENTS, device=self.device, dtype=dtype))

        # VPM4 (OA + Glu cotransmission) presynaptic autoreceptor state (Sci Adv 2026 aggression circuit)
        self.vpm4_autoreceptor_intact: bool = True
        self.vpm4_auto_r: float = 0.0

    def set_shibire_blockade(self, comp_indices: list[int], blocked: bool = True) -> None:
        """Reversible shibire^ts1 synaptic blockade of specific MBON compartments (Sci Rep 2022)."""
        for c in comp_indices:
            self.mbon_active_mask[c] = 0.0 if blocked else 1.0

    def compute_calyx_sparse_kc(self, pn_glomeruli_50: torch.Tensor) -> torch.Tensor:
        """Expand 50 AL glomeruli -> 4,552 KCs with APL GABAergic feedback shunting to exact ~5% sparsity."""
        raw_drive = torch.mv(self.w_claw, pn_glomeruli_50.to(self.device, dtype=self.dtype))
        if float(raw_drive.max().item()) <= 1e-6:
            return torch.zeros(self.n_kc, device=self.device, dtype=self.dtype)

        # APL feedback inhibition + high KC coincidence threshold selects exact top ~5% KCs
        k_keep = max(1, int(round(self.n_kc * self.target_sparsity)))
        top_vals, top_idx = torch.topk(raw_drive, k=k_keep)
        kc_act = torch.zeros_like(raw_drive)
        if float(top_vals[0].item()) > 1e-6:
            kc_act[top_idx] = top_vals / float(top_vals[0].item())
        return kc_act

    def step_mb(
        self,
        pn_glomeruli_50: torch.Tensor,
        sweet_reward_us: float = 0.0,
        bitter_shock_us: float = 0.0,
        lh_pam_drive: float = 0.0,
        lh_ppl1_drive: float = 0.0,
        dn_efference_copy: torch.Tensor | None = None,
        vpm4_octopamine_drive: float = 0.0,
        hunger_mp1_gate: float = 0.5,
    ) -> MBStepOutput:
        """Run 1 cognitive step of MB-NCA: 5% KC sparse code, Port 3 a-a compartment readout, LTD, and VPM4."""
        kc_act = self.compute_calyx_sparse_kc(pn_glomeruli_50)
        active_frac = float((kc_act > 0.0).float().mean().item())

        # Port 3 axo-axonic presynaptic compartment gate on KC axons [N_KC, 15]
        # Hunger (NPF inhibiting MP1 DA) gates open the consolidated food-seeking compartments (alpha/beta 9..14)
        aa_exc = torch.zeros((self.n_kc, N_MB_COMPARTMENTS), device=self.device, dtype=self.dtype)
        aa_inh = torch.zeros((self.n_kc, N_MB_COMPARTMENTS), device=self.device, dtype=self.dtype)
        if hunger_mp1_gate < 0.2:
            aa_inh[:, 9:15] = 1.5  # Satiety (MP1 active) dampens LTM food-seeking readout

        kc_comp_out = port3_axo_axonic_presynaptic_gate(
            kc_act.unsqueeze(-1),
            torch.zeros((self.n_kc, 1), device=self.device, dtype=self.dtype),
            torch.zeros((self.n_kc, 1), device=self.device, dtype=self.dtype),
            aa_exc=aa_exc,
            aa_inh=aa_inh,
        )  # [N_KC, 15]

        # Raw compartment readout sum_k W_mb[k, c] * Out_{k, c}^{KC} normalized by active KC count
        norm_div = max(1.0, float((kc_act > 0).sum().item()))
        naive_mbon = kc_comp_out.sum(dim=0) / norm_div
        raw_mbon = (self.w_mb * kc_comp_out).sum(dim=0) / norm_div
        raw_mbon = raw_mbon * self.mbon_active_mask

        # Feedforward lateral inhibition MBON-gamma1pedc (0) -| MBON-gamma5 (4) (Sci Rep 2022, Springer/Nawrot!):
        # Before aversive learning, MBON-g1pedc is high and suppresses MBON-g5 (avoidance).
        # After aversive LTD depresses KC -> MBON-g1pedc, MBON-g1pedc drops, disinhibiting MBON-g5!
        mbon_rates = raw_mbon.clone()
        g1_rate = float(mbon_rates[COMP_GAMMA1].item())
        mbon_rates[COMP_GAMMA5] = mbon_rates[COMP_GAMMA5] / (1.0 + 2.5 * g1_rate)

        # Net learned valence: deviation from naive (W_mb == 1.0) baseline balance (exact 0.0 before any DAN LTD learning!)
        # Note: if either gamma1 (0) or gamma5 (4) is blocked by shibire, the g1<->g5 opponent balance collapses!
        if float(self.mbon_active_mask[COMP_GAMMA1].item()) == 0.0 or float(self.mbon_active_mask[COMP_GAMMA5].item()) == 0.0:
            net_valence = 0.0
        else:
            r0_g1 = float(naive_mbon[COMP_GAMMA1].item())
            r0_g5 = float(naive_mbon[COMP_GAMMA5].item()) / (1.0 + 2.5 * r0_g1)
            baseline_offset = float((naive_mbon * self.mbon_valence_sign).sum().item() + float(naive_mbon[COMP_GAMMA5].item()) - r0_g5)
            net_valence = float((mbon_rates * self.mbon_valence_sign).sum().item() - baseline_offset)

        # Direct Efference Copy from Descending Neurons (37% VNC DNs -> DANs, Science 2023 PMC7614541!)
        eff_bonus = 0.0
        if dn_efference_copy is not None and dn_efference_copy.numel() > 0:
            eff_bonus = BANC_VNC_DN_EFFERENCE_FRACTION * float(dn_efference_copy[0].item())

        # Teacher DAN activity with prediction-error feedback (-MBON_c):
        # - Aversive PPL1 DANs innervate Approach compartments (even indices, especially gamma1 = 0!)
        # - Appetitive PAM DANs innervate Avoid compartments (odd indices & gamma5 = 4!)
        dan_signals = torch.zeros(N_MB_COMPARTMENTS, device=self.device, dtype=self.dtype)
        punish_total = max(0.0, bitter_shock_us + lh_ppl1_drive)
        reward_total = max(0.0, sweet_reward_us + lh_pam_drive + eff_bonus * (1.0 if sweet_reward_us > 0 else 0.0))

        for c in range(N_MB_COMPARTMENTS):
            if self.mbon_valence_sign[c] > 0:
                # Approach MBON compartment: taught (depressed via LTD!) by aversive PPL1 DAN
                if punish_total > 0.05:
                    # Sci Rep 2022: associative aversive plasticity is localized to gamma1 (c=0) in STM!
                    scale_c = 1.0 if (c == COMP_GAMMA1 or c >= 5) else 0.0
                    dan_signals[c] = max(0.0, scale_c * punish_total - 0.2 * float(mbon_rates[c].item()))
            else:
                # Avoid MBON compartment: taught (depressed via LTD!) by appetitive PAM DAN
                if reward_total > 0.05:
                    dan_signals[c] = max(0.0, reward_total - 0.2 * float(mbon_rates[c].item()))

        # Synaptic Plasticity Rule (Anti-Hebbian LTD on active KC x DAN compartments):
        if float(dan_signals.max().item()) > 0.0:
            ltd_delta = (
                self.eta_learn.unsqueeze(0)
                * kc_act.unsqueeze(-1)
                * dan_signals.unsqueeze(0)
                * self.w_mb
            )
            self.w_mb = (self.w_mb - ltd_delta).clamp(0.05, 1.0)

        # VPM4 (OA + Glu cotransmission) -> GluCla inhibition of MBON-11 (gamma1 GABA) -> Aggression escalation
        # Presynaptic autoreceptors OAa2R / mGluR brake VPM4 release; knockdown causes boxing escalation (Sci Adv 2026)!
        if vpm4_octopamine_drive > 0.0:
            self.vpm4_auto_r = min(1.0, 0.7 * self.vpm4_auto_r + 0.6 * vpm4_octopamine_drive)
        else:
            self.vpm4_auto_r *= 0.85

        auto_gain = 2.5 if self.vpm4_autoreceptor_intact else 0.0
        eff_vpm4 = vpm4_octopamine_drive / (1.0 + auto_gain * self.vpm4_auto_r)
        # VPM4 GluCla shunts MBON-11 (which normally holds tonic GABA Rdl inhibition on aggression)
        mbon11_tonic_gaba = 1.0 / (1.0 + 3.0 * eff_vpm4)
        aggression_boxing = max(0.0, 1.0 - mbon11_tonic_gaba)

        return MBStepOutput(
            kc_sparse_activity=kc_act,
            kc_active_fraction=active_frac,
            mbon_rates=mbon_rates,
            net_learned_valence=net_valence,
            dan_prediction_error=dan_signals,
            aggression_boxing_drive=aggression_boxing,
        )

    def consolidate_sleep_creb(self, creb_intact: bool = True, transfer_rate: float = 0.50) -> None:
        """Active Sleep (dFB REM-like phase): transfer memory traces from fast gamma (0..4) to LTM alpha/beta (9..14)."""
        if not creb_intact:
            return
        # Map gamma1 (0, approach) -> alpha1/alpha2/alpha3 (9, 11, 13)
        # and gamma5 (4, avoid) -> beta1/beta2/beta3 (10, 12, 14)
        for dst_c in (9, 11, 13):
            delta = transfer_rate * (self.w_mb[:, COMP_GAMMA1] - self.w_mb[:, dst_c])
            self.w_mb[:, dst_c] = (self.w_mb[:, dst_c] + delta).clamp(0.05, 1.0)
        for dst_c in (10, 12, 14):
            delta = transfer_rate * (self.w_mb[:, COMP_GAMMA5] - self.w_mb[:, dst_c])
            self.w_mb[:, dst_c] = (self.w_mb[:, dst_c] + delta).clamp(0.05, 1.0)
