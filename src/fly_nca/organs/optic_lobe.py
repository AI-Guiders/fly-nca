"""L3 OL-NCA: 892 Hexagonal Retinotopic Columns, T4/T5 Motion Correlators, MeTu/AOTU, and VPN Glomeruli."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn as nn

from fly_nca.body.adapter import SensorIn
from fly_nca.core.cell_state import TAU_FRAME_MS
from fly_nca.genome.banc_biophysics import BANC_EMPIRICAL_BIOPHYSICS


@dataclass(frozen=True, slots=True)
class VisualProjectionDigest:
    """Output of Optic Lobe (7,316 VPNs + AOTU->BU anterior pathway) sent to Central Brain."""

    bu_compass_ring_16: torch.Tensor   # [16] Topographic MeTu-l/c -> AOTU -> TuBu -> Bulb BU -> EB compass input
    bui_contralateral_inh: float       # Non-topographic MeTu-m -> BUi contralateral suppression (iScience 2020)
    lc4_lplc2_looming: float           # Approaching threat / collision detector -> GF & DN
    lc11_small_target: float           # Small moving object salience
    lc11_target_azimuth_rad: float     # Bearing to small target
    lpt_optic_flow_yaw_fwd: tuple[float, float]  # Global optic flow (yaw rotation, forward translation)


class OpticLobeNCA(nn.Module):
    """892-column hexagonal retinotopic NCA (97.7% intrinsic, 0% direct motor output, Nature 2025)."""

    def __init__(self, n_columns: int = 892, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.n_columns = int(n_columns)
        self.device = torch.device(device)
        self.dtype = dtype
        self.retinal_gain: float = 1.0

        # State per hex column: [L1_ON, L2_OFF, Delayed_ON, Delayed_OFF, T4_front2back, T4_back2front]
        self.register_buffer("col_state", torch.zeros((self.n_columns, 6), device=self.device, dtype=dtype))
        self.register_buffer("prev_lum", torch.zeros((self.n_columns,), device=self.device, dtype=dtype))

    def step_optic_lobe(self, sensor: SensorIn, dt_ms: float = TAU_FRAME_MS) -> VisualProjectionDigest:
        """Run local hexagonal column filtering (Lamina -> Medulla -> Lobula Plate T4/T5 -> AOTU & VPN)."""
        rg = float(self.retinal_gain)
        lum = sensor.ommatidia_light.to(self.device, dtype=self.dtype) * rg
        # Hexagonal lateral neighborhood average via circular shifts
        lat_mean = (
            torch.roll(lum, 1)
            + torch.roll(lum, -1)
            + torch.roll(lum, 6)
            + torch.roll(lum, -6)
            + torch.roll(lum, 12)
            + torch.roll(lum, -12)
        ) / 6.0

        ol_bio = BANC_EMPIRICAL_BIOPHYSICS["optic_lobe_intrinsic"]
        tm_bio = BANC_EMPIRICAL_BIOPHYSICS["transmedullary"]
        lc_bio = BANC_EMPIRICAL_BIOPHYSICS["lobula_columnar"]

        contrast = lum - lat_mean
        dlum = lum - self.prev_lum
        self.prev_lum = lum.clone()

        l1_on = torch.relu(contrast + ol_bio.port_axon_dendrite * dlum)
        l2_off = torch.relu(-contrast - ol_bio.port_axon_dendrite * dlum)

        # Medulla Mi1/Tm1 delayed channels (low-pass filtered over 3x lobula_columnar tau_V = 14.05 ms)
        tau_delay_ms = 3.0 * lc_bio.tau_v_ms
        alpha_delay = min(1.0, float(dt_ms) / tau_delay_ms)
        del_on = (1.0 - alpha_delay) * self.col_state[:, 2] + alpha_delay * self.col_state[:, 0]
        del_off = (1.0 - alpha_delay) * self.col_state[:, 3] + alpha_delay * self.col_state[:, 1]

        # Lobula Plate T4/T5 Reichardt-Barlow motion correlators with GluCla shunting in null direction
        null_shunt_gain = tm_bio.ie_ratio + lc_bio.dendritic_shunt_gain
        t4_f2b = (l1_on * torch.roll(del_on, 1)) / (1.0 + null_shunt_gain * torch.roll(del_on, -1))
        t4_b2f = (l1_on * torch.roll(del_on, -1)) / (1.0 + null_shunt_gain * torch.roll(del_on, 1))

        self.col_state[:, 0] = l1_on
        self.col_state[:, 1] = l2_off
        self.col_state[:, 2] = del_on
        self.col_state[:, 3] = del_off
        self.col_state[:, 4] = t4_f2b
        self.col_state[:, 5] = t4_b2f

        # 1. Anterior Visual Pathway (iScience 2020):
        # Topographic MeTu-l/c -> AOTU SU -> TuBu -> Bulb -> 16 EB wedges
        wedge_angles = torch.linspace(-math.pi, math.pi, 17, device=self.device, dtype=self.dtype)[:-1]
        bu_ring = torch.relu(torch.cos(wedge_angles - sensor.sun_azimuth_rad)) * rg
        # Non-topographic MeTu-m integrates across medulla/lobula -> BUi contralateral suppression (ol_bio.inh_share = 0.380)
        bui_inh = float(lum.mean().item()) * ol_bio.inh_share

        # 2. Lobular VPN Optic Glomeruli (LC4/LPLC2 looming, LC11 small target, LPT optic flow)
        yaw_flow = float((t4_f2b - t4_b2f).mean().item())
        fwd_flow = float((t4_f2b + t4_b2f).mean().item())

        return VisualProjectionDigest(
            bu_compass_ring_16=bu_ring,
            bui_contralateral_inh=bui_inh,
            lc4_lplc2_looming=float(sensor.looming_rate) * rg,
            lc11_small_target=float(sensor.small_target_salience) * rg,
            lc11_target_azimuth_rad=float(sensor.small_target_azimuth_rad),
            lpt_optic_flow_yaw_fwd=(yaw_flow, fwd_flow),
        )
