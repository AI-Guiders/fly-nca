"""Core 8-channel Universal Fly NCA Cell State and physiological parameters (Section 1.1 & 1.3)."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

# Channel indices in state tensor s[N, 8]
CH_V: int = 0        # Membrane potential V_i(t) in [-1, +1]
CH_A: int = 1        # Soma firing output a_i(t) in [0, 1]
CH_CA: int = 2       # Intracellular calcium Ca_i(t) in [0, 1] (coincidence integrator)
CH_PERSIST: int = 3  # Persistent working-memory current p_i(t) in [0, 1] (5-10s in CX FB hDeltaK21)
CH_ADAPT: int = 4    # Spike-frequency adaptation / CPG phase A_i(t) in [0, 1] (VNC / dFB)
CH_AUTO: int = 5     # Presynaptic autoreceptor brake r_i(t) in [0, 1] (OAa2R / mGluR)
CH_ATP: int = 6      # Cellular metabolic reserve E_i(t) in [0, 1]
CH_SLEEP: int = 7    # Local sleep pressure / synaptic fatigue S_i(t) in [0, 1]

NUM_STATE_CHANNELS: int = 8

# Fundamental physical frame quanta for 1 mg Drosophila from mt-lattice-fluid MODEL §4.8:
TAU_FAST_MS: float = 1.711   # Tissue sphere R=0.27 mm -> VNC autopilot / haltere / GF escape quantum (1 tau_fast)
TAU_FRAME_MS: float = 4.038  # Body sphere R=0.63 mm -> 200-248 Hz wingbeat & OL/CB cognitive frame (1 tau_frame)
TAU_FRAME_TO_FAST_RATIO: float = TAU_FRAME_MS / TAU_FAST_MS  # 2.360023... fast quanta per wingbeat frame


@dataclass(frozen=True, slots=True)
class CellPhysiologyParams:
    """Genetically expressed ion-channel and receptor profile per cell [N]."""

    tau_v_ms: torch.Tensor        # [N] Membrane time constant (ms), default ~5.0 ms
    theta_base: torch.Tensor      # [N] Baseline firing threshold
    alpha_shunt: torch.Tensor     # [N] Dendritic branch GluCla / GABA shunt gain
    beta_root: torch.Tensor       # [N] Soma root aversive shunt gain (aversion > attraction)
    mu_conj: torch.Tensor         # [N] Conjunction gate factor in [0, 1] (1.0 = AND-gate via tanh product)
    tau_ca_ms: torch.Tensor       # [N] Calcium decay time constant (ms), default ~100.0 ms
    tau_persist_ms: torch.Tensor  # [N] Persistent current decay (5000..10000 ms in FB hDeltaK21)
    gamma_persist: torch.Tensor   # [N] Persistent bistable self-excitation gain
    tau_adapt_ms: torch.Tensor    # [N] Adaptation decay time constant (ms)
    beta_adapt: torch.Tensor      # [N] Adaptation increment per unit activity
    lambda_adapt: torch.Tensor    # [N] Adaptation hyperpolarizing current weight
    tau_auto_ms: torch.Tensor     # [N] Presynaptic autoreceptor decay (ms), ~200 ms
    beta_auto: torch.Tensor       # [N] Autoreceptor accumulation rate
    kappa_auto: torch.Tensor      # [N] Presynaptic autoreceptor suppression strength (OAa2R / mGluR)
    dopr1_expr: torch.Tensor      # [N] DopR1 (cAMP+ / LTD plasticity) receptor expression
    dopr2_expr: torch.Tensor      # [N] DopR2 receptor expression
    octr_expr: torch.Tensor       # [N] Octopamine receptor (OAMB / OctbR) expression

    @staticmethod
    def default(n_cells: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> CellPhysiologyParams:
        z = lambda val: torch.full((n_cells,), float(val), device=device, dtype=dtype)
        return CellPhysiologyParams(
            tau_v_ms=z(5.0),
            theta_base=z(0.15),
            alpha_shunt=z(2.0),
            beta_root=z(3.0),
            mu_conj=z(0.0),
            tau_ca_ms=z(80.0),
            tau_persist_ms=z(6000.0),
            gamma_persist=z(0.0),
            tau_adapt_ms=z(120.0),
            beta_adapt=z(0.05),
            lambda_adapt=z(0.25),
            tau_auto_ms=z(200.0),
            beta_auto=z(0.10),
            kappa_auto=z(0.50),
            dopr1_expr=z(0.0),
            dopr2_expr=z(0.0),
            octr_expr=z(0.0),
        )


def init_cell_state(n_cells: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Initialize [N, 8] state tensor with resting membrane potential and full ATP."""
    s = torch.zeros((n_cells, NUM_STATE_CHANNELS), device=device, dtype=dtype)
    s[:, CH_ATP] = 1.0
    return s


class UniversalFlyCellUpdater(nn.Module):
    """Universal 8-channel NCA step function F_Genome (Section 1.3)."""

    def __init__(self, slope: float = 4.0) -> None:
        super().__init__()
        self.slope = float(slope)

    def forward(
        self,
        state: torch.Tensor,
        params: CellPhysiologyParams,
        i_dend: torch.Tensor,
        i_gap: torch.Tensor,
        mod_threshold_shift: torch.Tensor | None = None,
        i_persist_reset: torch.Tensor | None = None,
        glial_atp_supply: torch.Tensor | None = None,
        glial_sleep_clearance: torch.Tensor | None = None,
        dt_ms: float = TAU_FAST_MS,
    ) -> torch.Tensor:
        """Advance cell state s[N, 8] by dt_ms (defaults to 1 natural fast quantum TAU_FAST_MS = 1.711 ms)."""
        v = state[:, CH_V]
        a = state[:, CH_A]
        ca = state[:, CH_CA]
        p = state[:, CH_PERSIST]
        adapt = state[:, CH_ADAPT]
        r_auto = state[:, CH_AUTO]
        atp = state[:, CH_ATP]
        sleep_p = state[:, CH_SLEEP]

        dt = float(dt_ms)
        alpha_v = (dt / params.tau_v_ms.clamp(min=dt)).clamp(0.0, 1.0)
        alpha_ca = (dt / params.tau_ca_ms.clamp(min=dt)).clamp(0.0, 1.0)
        alpha_p = (dt / params.tau_persist_ms.clamp(min=dt)).clamp(0.0, 1.0)
        alpha_ad = (dt / params.tau_adapt_ms.clamp(min=dt)).clamp(0.0, 1.0)
        alpha_au = (dt / params.tau_auto_ms.clamp(min=dt)).clamp(0.0, 1.0)

        # 0. Membrane potential V_i(t + dt)
        drive = i_dend + i_gap + params.gamma_persist * p - params.lambda_adapt * adapt
        v_next = ((1.0 - alpha_v) * v + alpha_v * drive).clamp(-1.0, 1.0)

        # 1. Firing output a_i(t + dt) gated by modulated threshold and metabolic ATP availability
        theta_eff = params.theta_base
        if mod_threshold_shift is not None:
            theta_eff = theta_eff + mod_threshold_shift
        atp_gate = (atp / 0.15).clamp(0.0, 1.0)
        a_next = (torch.relu(v_next - theta_eff) * self.slope).clamp(0.0, 1.0) * atp_gate

        # 2. Intracellular Ca2+ coincidence integrator
        ca_next = ((1.0 - alpha_ca) * ca + alpha_ca * a_next).clamp(0.0, 1.0)

        # 3. Persistent working-memory current p_i(t + dt) (active when gamma_persist > 0, e.g. CX FB hDeltaK21)
        reset_term = i_persist_reset if i_persist_reset is not None else torch.zeros_like(p)
        p_pump = (params.gamma_persist > 0).to(p.dtype) * (0.25 * ca_next * torch.relu(i_dend) + 0.04 * (p > 0.25).to(p.dtype))
        p_next = ((1.0 - alpha_p) * p + p_pump - reset_term).clamp(0.0, 1.0)

        # 4. Spike-frequency adaptation / pacemaker phase A_i(t + dt)
        adapt_next = ((1.0 - alpha_ad) * adapt + params.beta_adapt * (dt / 5.0) * a_next).clamp(0.0, 1.0)

        # 5. Presynaptic autoreceptor brake r_i(t + dt) (OAa2R / mGluR)
        r_next = ((1.0 - alpha_au) * r_auto + params.beta_auto * (dt / 5.0) * a_next).clamp(0.0, 1.0)

        # 6. Metabolic ATP reserve E_i(t + dt)
        supply = glial_atp_supply if glial_atp_supply is not None else torch.full_like(atp, 0.0005 * dt)
        atp_next = (atp - 0.0012 * dt * (a_next * a_next) + supply).clamp(0.0, 1.0)

        # 7. Local sleep pressure / synaptic fatigue S_i(t + dt)
        clear = glial_sleep_clearance if glial_sleep_clearance is not None else torch.zeros_like(sleep_p)
        sleep_next = (sleep_p + 0.0004 * dt * a_next - clear).clamp(0.0, 1.0)

        return torch.stack([v_next, a_next, ca_next, p_next, adapt_next, r_next, atp_next, sleep_next], dim=-1)
