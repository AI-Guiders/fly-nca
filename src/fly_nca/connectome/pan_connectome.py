"""Full 189,766-Cell Drosophila Pan-Connectome NCA Engine (BANC v888 Female + MCNS v1.0 Male-Specific).

Connects all 188,508 cells of BANC v888 + 1,258 male-specific cells of MCNS v1.0 (189,766 total cells)
and 6,749,514 subcellular synaptic edges across the 5 biological ports:
- Port 1 (`W_port1_ach_dend`): Excitatory dendritic ACh+ synapses
- Port 2a (`W_port2_shunt_dend`): Inhibitory chloride-shunting dendritic GABA/Glu/His synapses
- Port 2b (`W_port2_root_shunt`): Soma / primary-dendrite root shunting synapses
- Port 3 (`W_port3_axo_axonic`): Presynaptic axo-axonic gating synapses (9.15M synapses in BANC)
- Port 5 (`W_port5_modulators`): Neuromodulatory DA / OA / 5-HT / TA synapses

Includes direct sensory periphery injection across all 16,563 sensory neurons (`body_part_sensory`,
`cell_function`, `ORN` glomeruli, `JO` auditory/wind, `Gr5a/Gr66a/Ir76b`, `campaniform`, `chordotonal`,
`bristle`, `haltere`) and direct motor/visceral readout across all 1,026 effector neurons
(`front/middle/hind leg` muscle groups, `wing_power/steering`, `haltere`, `neck`, `proboscis/pharynx`,
`TTMn`, `oviposition/uterus`, and `MCNS` male `P1 -> pIP10 -> TN1` courtship song circuits).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import warnings
import torch
import torch.nn as nn

from fly_nca.body.adapter import MotorOut, SensorIn

PAN_CONNECTOME_PATH = Path(
    r"D:\Experiments\casa-data\connectomes\banc-v888-csr-cache\drosophila_pan_connectome_189766.pt"
)


@dataclass(frozen=True, slots=True)
class FullConnectomeReadout:
    """Per-frame readout of the 189,766-cell pan-connectome wave and dimorphic social circuits."""

    n_active_cells: int
    mean_firing_rate: float
    organ_mean_rates: dict[str, float]
    dn_cluster_rates: dict[str, float]
    male_p1_courtship_rate: float
    male_pip10_song_rate: float
    male_tn1_wing_song_rate: float
    female_pc1_receptivity_rate: float
    female_ovi_oviposition_rate: float
    gf_dnp01_rate: float
    mdn_backward_rate: float
    dnp09_forward_rate: float
    dng02_flight_rate: float
    lbl40_t3_rate: float


def default_device() -> torch.device:
    """Return CUDA GPU device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PanConnectomeSubstrate:
    """Shared read-only structural CSR graph of all 189,766 cells and 6,749,514 subcellular edges."""

    _CACHE: dict[str, "PanConnectomeSubstrate"] = {}

    def __init__(self, path: Path = PAN_CONNECTOME_PATH, device: torch.device | str | None = None) -> None:
        self.device = torch.device(device) if device is not None else default_device()
        if not path.is_file():
            from tools.build_pan_connectome_189766 import build_pan_connectome

            build_pan_connectome()

        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.n_total: int = int(raw["n_total"])
        self.n_banc: int = int(raw["n_banc"])
        self.n_male_specific: int = int(raw["n_male_specific"])
        self.n_edges: int = int(raw["n_edges"])

        self.tau_v_ms: torch.Tensor = raw["tau_v_ms"].to(self.device, dtype=torch.float32)
        self.alpha_shunt: torch.Tensor = raw["alpha_shunt"].to(self.device, dtype=torch.float32)
        n = self.n_total
        self.dcv_peptidergic_gain: torch.Tensor = raw.get(
            "dcv_peptidergic_gain", torch.full((n,), 0.25)
        ).to(self.device, dtype=torch.float32)
        self.mito_atp_capacity: torch.Tensor = raw.get(
            "mito_atp_capacity", torch.ones(n)
        ).to(self.device, dtype=torch.float32)
        self.fca_ca_alpha1t: torch.Tensor = raw.get(
            "fca_ca_alpha1t", torch.full((n,), 0.45)
        ).to(self.device, dtype=torch.float32)
        self.fca_glucl_rdl: torch.Tensor = raw.get(
            "fca_glucl_rdl", torch.ones(n)
        ).to(self.device, dtype=torch.float32)
        self.fca_shaker_k: torch.Tensor = raw.get(
            "fca_shaker_k", torch.ones(n)
        ).to(self.device, dtype=torch.float32)
        self.fca_shakb_gap: torch.Tensor = raw.get(
            "fca_shakb_gap", torch.full((n,), 0.35)
        ).to(self.device, dtype=torch.float32)
        self.effector_influence: torch.Tensor = raw.get(
            "effector_influence", torch.zeros(n)
        ).to(self.device, dtype=torch.float32)
        self.betweenness_aff_eff: torch.Tensor = raw.get(
            "betweenness_aff_eff", torch.zeros(n)
        ).to(self.device, dtype=torch.float32)
        self.soma_xyz_um: torch.Tensor = raw.get(
            "soma_xyz_um", torch.zeros((n, 3))
        ).to(self.device, dtype=torch.float32)
        self.mirror_partner_idx: torch.Tensor = raw.get(
            "mirror_partner_idx", torch.arange(n, dtype=torch.int32)
        ).to(self.device, dtype=torch.long)
        self.cotrans_no_mask: torch.Tensor = raw.get(
            "cotrans_no_mask", torch.zeros(n, dtype=torch.uint8)
        ).to(self.device)
        self.nt_id: torch.Tensor = raw["nt_id"].to(self.device)
        self.sex_code: torch.Tensor = raw["sex_code"].to(self.device)
        self.side_id: torch.Tensor = raw["side_id"].to(self.device)
        self.organ_id: torch.Tensor = raw["organ_id"].to(self.device)
        self.hemilineage_id: torch.Tensor = raw["hemilineage_id"].to(self.device)

        self.cell_groups: dict[str, torch.Tensor] = {
            k: v.to(self.device, dtype=torch.long) for k, v in raw["cell_groups"].items()
        }

        # Precompute vectorized ORN glomerular and 6-leg index tensors (zero Python loops / zero .item() syncs per frame!)
        orn_cells_list: list[torch.Tensor] = []
        orn_glom_list: list[torch.Tensor] = []
        for g_idx in range(50):
            g_c = self.cell_groups.get(f"orn_glom_{g_idx:02d}")
            if g_c is not None and g_c.numel() > 0:
                orn_cells_list.append(g_c)
                orn_glom_list.append(torch.full_like(g_c, g_idx))
        self.orn_cell_idx = torch.cat(orn_cells_list) if orn_cells_list else torch.empty(0, dtype=torch.long, device=self.device)
        self.orn_glom_id = torch.cat(orn_glom_list) if orn_glom_list else torch.empty(0, dtype=torch.long, device=self.device)

        leg_cells_list: list[torch.Tensor] = []
        leg_id_list: list[torch.Tensor] = []
        for leg_i, l_tag in enumerate(("L1", "R1", "L2", "R2", "L3", "R3")):
            l_c = self.cell_groups.get(f"sens_leg_{l_tag}")
            if l_c is not None and l_c.numel() > 0:
                leg_cells_list.append(l_c)
                leg_id_list.append(torch.full_like(l_c, leg_i))
        self.leg_cell_idx = torch.cat(leg_cells_list) if leg_cells_list else torch.empty(0, dtype=torch.long, device=self.device)
        self.leg_id = torch.cat(leg_id_list) if leg_id_list else torch.empty(0, dtype=torch.long, device=self.device)

        pre = raw["edge_pre"].to(torch.long)
        post = raw["edge_post"].to(torch.long)
        norm = raw["edge_norm"].to(torch.float32)
        port = raw["edge_port"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.w_port1_ach = self._build_csr(pre, post, norm, port == 0)
            self.w_port2_shunt = self._build_csr(pre, post, norm, port == 1)
            self.w_port2_root = self._build_csr(pre, post, norm, port == 2)
            self.w_port3_aa = self._build_csr(pre, post, norm, port == 3)
            self.w_port5_mod = self._build_csr(pre, post, norm, port == 4)

    def _build_csr(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        norm: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        idx = torch.stack([post[mask], pre[mask]], dim=0).to(self.device)
        val = norm[mask].to(self.device, dtype=torch.float32)
        coo = torch.sparse_coo_tensor(idx, val, (self.n_total, self.n_total), device=self.device).coalesce()
        return coo.to_sparse_csr()

    @classmethod
    def get(cls, device: torch.device | str | None = None) -> "PanConnectomeSubstrate":
        resolved = torch.device(device) if device is not None else default_device()
        key = str(resolved)
        if key not in cls._CACHE:
            cls._CACHE[key] = cls(device=resolved)
        return cls._CACHE[key]


class FullConnectomeFlyNCA(nn.Module):
    """Complete 189,766-Cell Drosophila NCA Organism running on the empirical 5-port pan-connectome.

    Sex determination follows the Drosophila cell-autonomous X:A -> Sxl -> Tra -> (Dsx, Fru) splicing cascade
    (PMC10622783, Front Mol Neurosci 2023; Kimura et al. 2008; Demir & Dickson 2005; Kohl et al. 2013):
    1) `dsx_isoform` ('DsxF' vs 'DsxM') controls sex-specific cell survival (apoptosis vs survival of P1 vs oviDN/vpoEN).
    2) `fru_m_active` (bool) controls dendritic arborization & synaptic valence (e.g., DA1 -> aSP-f vs aSP-g cVA switch,
       and VNC TN1/vPR6 courtship song generation even in P1-less DsxF + FruM+ females!).
    """

    def __init__(
        self,
        sex: Literal["F", "M"] = "F",
        *,
        dsx_isoform: Literal["DsxF", "DsxM"] | None = None,
        fru_m_active: bool | None = None,
        foraging_Write_gain: float = 1.0,
        dopr1_learning_scale: float = 1.0,
        oaa2r_brake_scale: float = 1.0,
        shakb_gap_scale: float = 1.0,
        snpfr_hunger_scale: float = 1.0,
        courtship_vigor_scale: float = 1.0,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device) if device is not None else default_device()
        self.sex: Literal["F", "M"] = sex
        self.dsx_isoform: Literal["DsxF", "DsxM"] = dsx_isoform or ("DsxF" if sex == "F" else "DsxM")
        self.fru_m_active: bool = fru_m_active if fru_m_active is not None else (sex == "M")
        self.substrate = PanConnectomeSubstrate.get(self.device)
        n = self.substrate.n_total
        cg = self.substrate.cell_groups

        self.foraging_gain = float(foraging_Write_gain)
        self.dopr1_learning_scale = float(dopr1_learning_scale)
        self.oaa2r_brake_scale = float(oaa2r_brake_scale)
        self.shakb_gap_scale = float(shakb_gap_scale)
        self.snpfr_hunger_scale = float(snpfr_hunger_scale)
        self.courtship_vigor_scale = float(courtship_vigor_scale)

        # Cell-autonomous Dsx / Fru survival & expression mask (X:A -> Sxl -> Tra -> DsxF/DsxM + FruM):
        # - DsxF triggers apoptosis of male-specific P1 cluster (PMC10622783: FruM females lack P1 because P1 requires DsxM!)
        # - DsxM triggers apoptosis of female-specific oviDN/vpoEN/uterus neurons
        # - FruM switches dendritic arborization between aSP-f (FruM+) and aSP-g (Fru-) (Kohl et al. 2013 Cell)
        #   and enables male VNC song circuits (TN1/vPR6) and pIP10 descending song drive
        sex_mask = torch.ones(n, device=self.device, dtype=torch.float32)
        if self.dsx_isoform == "DsxF":
            sex_mask[self.substrate.sex_code == 3] = 0.0  # Male-specific Dsx+ P1 dies under DsxF
            if self.fru_m_active:
                # Demir & Dickson (2005) / Kimura et al. (2008) / PMC10622783:
                # In DsxF + FruM+ females, Dsx-dependent P1 (100% dsx+) undergoes apoptosis,
                # whereas the 749 FruM-only (fru_high/fru_low, dsx-) circuits and VNC song circuits survive!
                if "dimorphic_male_fru_only" in cg:
                    sex_mask[cg["dimorphic_male_fru_only"]] = 1.0
                sex_mask[cg["dimorphic_male_tn1_song"]] = 1.0
                sex_mask[cg["dimorphic_male_pip10"]] = 1.0
        else:
            sex_mask[self.substrate.sex_code == 2] = 0.0  # Female-specific oviDN/vpoEN/uterus absent under DsxM
            sex_mask[cg["eff_oviposition_uterus"]] = 0.0
            sex_mask[cg["dimorphic_female_vpo_ovi"]] = 0.0

        # Kohl et al. (2013) Cell / PMC10622783 dendritic switch on cVA pathway (Or67d -> DA1 -> aSP-f vs aSP-g):
        if self.fru_m_active:
            sex_mask[cg["dimorphic_female_asp_g"]] = 0.0  # FruM+ disconnects female-typical aSP-g dendrites
        else:
            sex_mask[cg["dimorphic_female_asp_f"]] = 0.0  # Fru- disconnects male-typical aSP-f dendrites
            sex_mask[cg["dimorphic_male_tn1_song"]] = 0.0
            sex_mask[cg["dimorphic_male_pip10"]] = 0.0
        self.register_buffer("sex_cell_mask", sex_mask)

        # Living per-cell state vectors across all 189,766 cells:
        # v: membrane potential [-1..+1], a: soma firing rate [0..1], adapt: spike adaptation [0..1]
        # synaptic_plasticity_gain: live 3-factor KC->MBON / circuit synaptic efficacy [0.2..2.0]
        self.register_buffer("v", torch.zeros(n, device=self.device, dtype=torch.float32))
        self.register_buffer("a", torch.zeros(n, device=self.device, dtype=torch.float32))
        self.register_buffer("adapt", torch.zeros(n, device=self.device, dtype=torch.float32))
        self.register_buffer("synaptic_plasticity_gain", torch.ones(n, device=self.device, dtype=torch.float32))
        self.glial_atp_pool: float = 1.0
        self.sleep_debt: float = 0.0
        self.step_counter: int = 0

    def group_mean_rate(self, group_name: str) -> float:
        idx = self.substrate.cell_groups.get(group_name)
        if idx is None or idx.numel() == 0:
            return 0.0
        return float(self.a[idx].mean().item())

    def inject_sensory_periphery(
        self,
        sensor: SensorIn,
        *,
        cva_pheromone_conc: float = 0.0,
        mate_odor_conc: float = 0.0,
        conspecific_is_female: bool = True,
        substrate_yeast_rich: bool = False,
        hunger_level: float = 0.3,
    ) -> torch.Tensor:
        """Map SensorIn (16,563 BANC sensory channels) directly onto the 189,766-cell sensory periphery."""
        sub = self.substrate
        n = sub.n_total
        cg = sub.cell_groups
        i_sens = torch.zeros(n, device=self.device, dtype=torch.float32)

        # 1. Visual periphery (strictly R1-R8 retinal & ocellar photoreceptors)
        # Lamina monopolar cells (L1/L2 + Dm9/Lawf surround) high-pass filter steady illumination into spatial contrast
        vis_contrast = float(sensor.ommatidia_light.std().item()) if sensor.ommatidia_light.numel() > 1 else 0.0
        vis_drive = 0.04 + 0.65 * vis_contrast + 0.45 * float(sensor.looming_rate)
        i_sens[cg["sens_visual_left"]] = vis_drive
        i_sens[cg["sens_visual_right"]] = vis_drive
        i_sens[cg["sens_ocellar"]] = abs(float(sensor.head_horizon_rpy[0].item())) + abs(float(sensor.head_horizon_rpy[1].item()))

        # Looming shadow on retina excites peripheral lobula LC4/LPLC2 visual feature detectors
        if sensor.looming_rate > 0.70:
            i_sens[cg["circuit_lc4_looming"]] = float(sensor.looming_rate) * 1.5

        # Small moving target (conspecific mate/rival) excites LC10a visual tracking neurons
        vis_sal = float(sensor.small_target_salience)
        if vis_sal > 0.05:
            sal = vis_sal * self.courtship_vigor_scale
            az = float(sensor.small_target_azimuth_rad)
            if az >= 0.0:
                i_sens[cg["circuit_lc10a_left"]] = sal * 1.3
                i_sens[cg["circuit_lc10a_right"]] = sal * 0.4
            else:
                i_sens[cg["circuit_lc10a_right"]] = sal * 1.3
                i_sens[cg["circuit_lc10a_left"]] = sal * 0.4

        # 2. Vectorized Olfactory periphery (3,007 ORNs across 50 glomerular classes — zero Python loops!)
        odor_mean = (0.5 * (sensor.odor_glomeruli_lr[0] + sensor.odor_glomeruli_lr[1])).to(self.device)
        hunger_boost = 1.0 + 0.4 * self.snpfr_hunger_scale * hunger_level
        if sub.orn_cell_idx.numel() > 0:
            i_sens[sub.orn_cell_idx] = (odor_mean[sub.orn_glom_id] * hunger_boost).clamp(0.0, 1.0)

        # Sex pheromone cVA (Or67d -> ORN_DA1, 224 sensory cells) & mate odor (ORN_VA1v, 171 sensory cells)
        if cva_pheromone_conc > 0.0:
            i_sens[cg["sens_orn_DA1_cva"]] = float(cva_pheromone_conc) * self.courtship_vigor_scale
        if mate_odor_conc > 0.0:
            i_sens[cg["sens_orn_VA1v_mate"]] = float(mate_odor_conc) * self.courtship_vigor_scale

        # 3. Johnston's Organ (1,187 sensory cells: wind vector + courtship song vibration)
        wind_mag = float(sensor.wind_vector_ego.norm().item())
        song_vib = float(sensor.sound_vibration)
        jo_val = min(1.0, wind_mag + song_vib * 1.2 * self.courtship_vigor_scale)
        i_sens[cg["sens_jo_left"]] = jo_val
        i_sens[cg["sens_jo_right"]] = jo_val

        # 4. Gustatory & Tarsal Pheromone periphery (1,585 sensory cells across labellum, legs, pharynx)
        # Thistle et al. (2012) Cell / Kallman, Kim & Scott (2015) Nat Neurosci:
        # - Female 7,11-HD contact (requires close proximity `vis_sal > 0.20` and female conspecific) excites F-cells (LgLG5..8,1a,2 -> vAB3/PPN1 -> P1)
        # - Male 7-tricosene/cVA contact excites M-cells (LgLG3,4 -> AN05B023/AN13B002 GABA -| P1)
        t_vec = sensor.taste_sweet_bitter_amino
        sweet = float(t_vec[0].item())
        bitter = float(t_vec[1].item())
        amino = float(t_vec[2].item()) + (0.6 if substrate_yeast_rich else 0.0)
        i_sens[cg["sens_gust_labellum"]] = min(1.0, sweet * hunger_boost + amino)

        f_cells = cg.get("sens_gust_front_leg_female_711hd", cg["sens_gust_front_leg"])
        m_cells = cg.get("sens_gust_front_leg_male_cva")
        female_711hd_contact = (
            float(mate_odor_conc) * self.courtship_vigor_scale
            if (conspecific_is_female and vis_sal > 0.20)
            else 0.0
        )
        male_7t_contact = (
            float(cva_pheromone_conc) * self.courtship_vigor_scale
            if (not conspecific_is_female or cva_pheromone_conc > 0.45)
            else 0.0
        )
        i_sens[f_cells] = min(1.0, sweet + amino + female_711hd_contact)
        if m_cells is not None:
            i_sens[m_cells] = min(1.0, bitter + male_7t_contact)

        i_sens[cg["sens_gust_hind_leg"]] = min(1.0, amino)
        i_sens[cg["sens_gust_pharynx"]] = min(1.0, sweet + amino)
        i_sens[cg["sens_nociception"]] = bitter

        # 5. Vectorized Legs & Haltere Proprioception / Mechanoreception (zero Python loops!)
        if sub.leg_cell_idx.numel() > 0:
            leg_drive = (
                0.35 * sensor.leg_ground_load.to(self.device) + 0.80 * sensor.bristle_touch.to(self.device)
            ).clamp(0.0, 1.0)
            i_sens[sub.leg_cell_idx] = leg_drive[sub.leg_id]

        halt_mag = float(sensor.haltere_coriolis_rpy.abs().mean().item())
        i_sens[cg["sens_haltere_left"]] = min(1.0, 0.25 + halt_mag)
        i_sens[cg["sens_haltere_right"]] = min(1.0, 0.25 + halt_mag)
        i_sens[cg["sens_crop_visceral"]] = float(sensor.crop_stretch)

        return i_sens * self.sex_cell_mask

    def step_full_connectome(
        self,
        sensor: SensorIn,
        *,
        cva_pheromone_conc: float = 0.0,
        mate_odor_conc: float = 0.0,
        conspecific_is_female: bool = True,
        substrate_yeast_rich: bool = False,
        hunger_level: float = 0.3,
        n_synaptic_hops: int = 6,
        dt_ms: float = 4.038,
        recorder: object | None = None,
    ) -> FullConnectomeReadout:
        """Advance all 189,766 cells by `n_synaptic_hops` through the 5-port sparse CSR connectome matrices.

        Zero central-neuron shortcuts: sensory periphery inputs (`i_sens`) enter ONLY at the 16,563 sensory
        neurons (`sens_*` + peripheral optic glomeruli `LC4/LC10a`) and propagate across the 6,853,423
        empirical synaptic edges (`AL_PN -> KC/MBON/LHON -> CX/P1/pC1 -> DNs -> Motor Neurons`).
        """
        sub = self.substrate
        cg = sub.cell_groups
        i_sens = self.inject_sensory_periphery(
            sensor,
            cva_pheromone_conc=cva_pheromone_conc,
            mate_odor_conc=mate_odor_conc,
            conspecific_is_female=conspecific_is_female,
            substrate_yeast_rich=substrate_yeast_rich,
            hunger_level=hunger_level,
        )

        hop_dt = float(dt_ms) / max(1, int(n_synaptic_hops))
        alpha_v = (hop_dt / sub.tau_v_ms).clamp(0.18, 0.85)
        theta = 0.05

        for _ in range(max(1, int(n_synaptic_hops))):
            # Port 3: Signed axo-axonic presynaptic modulation across all 189,766 axons (+ACh facilitation, -GABA/Glu presynaptic inhibition)
            a_col = (self.a * self.sex_cell_mask).unsqueeze(1)
            aa_mod = torch.matmul(sub.w_port3_aa, a_col).squeeze(1)
            a_presyn = (self.a * self.sex_cell_mask * (1.0 + 0.45 * aa_mod).clamp(0.05, 1.50)).clamp(0.0, 1.0).unsqueeze(1)
            a_mod_presyn = (a_presyn.squeeze(1) * sub.dcv_peptidergic_gain).unsqueeze(1)

            # Port 1 (ACh+ dendritic excitation scaled by live synaptic_plasticity_gain), Port 2 (Cl- shunt), Port 5 (DCV peptidergic)
            i_exc = torch.matmul(sub.w_port1_ach, a_presyn).squeeze(1) * self.synaptic_plasticity_gain + i_sens
            g_shunt = torch.matmul(sub.w_port2_shunt, a_presyn).squeeze(1)
            g_root = torch.matmul(sub.w_port2_root, a_presyn).squeeze(1)
            i_mod = torch.matmul(sub.w_port5_mod, a_mod_presyn).squeeze(1)

            # 5-Port dendritic integration with hyperpolarizing chloride reversal (-0.50*g_shunt) & divisive conductance shunt
            net_syn = i_exc - 0.50 * sub.fca_glucl_rdl * g_shunt
            i_dend = torch.tanh(2.6 * net_syn / (1.0 + 0.30 * sub.alpha_shunt * g_shunt)) / (1.0 + 2.8 * g_root)

            # Post-inhibitory Ca-alpha1T rebound + Shaker K+ adaptation + Glial ATP metabolic envelope
            rebound = 0.08 * sub.fca_ca_alpha1t * torch.relu(-self.v)
            atp_env = (0.5 + 0.5 * self.glial_atp_pool) * sub.mito_atp_capacity.clamp(0.7, 1.4)
            drive = (i_dend + 0.25 * i_mod + rebound - 0.15 * sub.fca_shaker_k * self.adapt) * atp_env
            self.v = ((1.0 - alpha_v) * self.v + alpha_v * drive).clamp(-1.0, 1.0) * self.sex_cell_mask

            # Soma firing rate a_i(t + dt) and spike-frequency adaptation
            self.a = (torch.relu(self.v - theta) * 2.5).clamp(0.0, 1.0) * self.sex_cell_mask
            self.adapt = (0.92 * self.adapt + 0.08 * self.a).clamp(0.0, 1.0)

        self.step_counter += 1

        # Live 3-Factor KC -> MBON Dopaminergic Synaptic Plasticity & Glial Metabolic Homeostasis (Port 5)
        kc_rate = self.group_mean_rate("circuit_kc")
        pam_rate = self.group_mean_rate("circuit_dan_pam")
        ppl1_rate = self.group_mean_rate("circuit_dan_ppl1")
        lr = 0.12 * self.dopr1_learning_scale * self.foraging_gain
        if kc_rate > 0.02 and (pam_rate > 0.05 or ppl1_rate > 0.05):
            # PAM reward depresses avoidance MBONs -> approach; PPL1 punishment depresses approach MBONs -> avoidance
            mbon_av = cg["circuit_mbon_avoid"]
            mbon_ap = cg["circuit_mbon_approach"]
            self.synaptic_plasticity_gain[mbon_av] = (
                self.synaptic_plasticity_gain[mbon_av] * (1.0 - lr * kc_rate * pam_rate)
            ).clamp(0.25, 1.80)
            self.synaptic_plasticity_gain[mbon_ap] = (
                self.synaptic_plasticity_gain[mbon_ap] * (1.0 - lr * kc_rate * ppl1_rate)
            ).clamp(0.25, 1.80)

        mean_act = float(self.a.mean().item())
        self.glial_atp_pool = max(0.20, min(1.0, self.glial_atp_pool - 0.008 * mean_act + 0.002))
        self.sleep_debt = max(0.0, min(1.0, self.sleep_debt + 0.006 * mean_act))
        # Activate 3,121 Glial cells (organ_id == 8) proportionally to metabolic lactate demand & sleep debt
        self.a[sub.organ_id == 8] = min(1.0, 0.25 + 0.75 * (1.0 - self.glial_atp_pool) + 0.50 * self.sleep_debt)

        # Summarize organ & circuit activity
        organ_names = ("OL", "AL", "LH_SPIN", "MB", "CX", "AN_DN", "VNC", "SEZ", "GLIA")
        organ_rates = {
            name: round(float(self.a[sub.organ_id == idx].mean().item()), 4)
            for idx, name in enumerate(organ_names)
        }
        dn_clusters = {
            "walking": round(self.group_mean_rate("dn_cluster_walking"), 4),
            "flight_power": round(self.group_mean_rate("dn_cluster_flight_power"), 4),
            "flight_steer": round(self.group_mean_rate("dn_cluster_flight_steer"), 4),
            "head_orient": round(self.group_mean_rate("dn_cluster_head_orient"), 4),
            "threat": round(self.group_mean_rate("dn_cluster_threat"), 4),
            "feeding": round(self.group_mean_rate("dn_cluster_feeding"), 4),
            "grooming": round(self.group_mean_rate("dn_cluster_grooming"), 4),
            "reproduction": round(self.group_mean_rate("dn_cluster_reproduction"), 4),
        }
        _ = cg
        fem_rec = max(
            self.group_mean_rate("dimorphic_female_pc1"),
            self.group_mean_rate("dimorphic_female_vpo_ovi"),
            self.group_mean_rate("dimorphic_female_asp_g"),
        )
        pip_idx = cg["dimorphic_male_pip10"]
        pip10_rate = float(self.a[pip_idx].topk(min(2, pip_idx.numel())).values.mean().item()) if pip_idx.numel() > 0 else 0.0
        readout = FullConnectomeReadout(
            n_active_cells=int((self.a > 0.01).sum().item()),
            mean_firing_rate=round(float(self.a.mean().item()), 5),
            organ_mean_rates=organ_rates,
            dn_cluster_rates=dn_clusters,
            male_p1_courtship_rate=round(self.group_mean_rate("dimorphic_male_p1"), 4),
            male_pip10_song_rate=round(pip10_rate, 4),
            male_tn1_wing_song_rate=round(self.group_mean_rate("dimorphic_male_tn1_song"), 4),
            female_pc1_receptivity_rate=round(fem_rec, 4),
            female_ovi_oviposition_rate=round(self.group_mean_rate("eff_oviposition_uterus"), 4),
            gf_dnp01_rate=round(self.group_mean_rate("dn_gf_dnp01"), 4),
            mdn_backward_rate=round(self.group_mean_rate("dn_mdn"), 4),
            dnp09_forward_rate=round(self.group_mean_rate("dn_dnp09"), 4),
            dng02_flight_rate=round(self.group_mean_rate("dn_dng02"), 4),
            lbl40_t3_rate=round(self.group_mean_rate("LBL40_t3"), 4),
        )

        from fly_nca.nervous import BiologicalPort, ImpulseChannel, ImpulseOutcome, ImpulseRecorder, impulse_recorder

        rec = recorder if isinstance(recorder, ImpulseRecorder) else impulse_recorder()
        with rec.channel(ImpulseChannel.PAN_CONNECTOME_WAVE) as imp:
            n_sens_active = int((i_sens > 0.01).sum().item())
            n_silenced = int((self.sex_cell_mask == 0).sum().item())
            imp.activate(
                "pan_connectome.sensory_periphery_16563",
                organ="SENSORY_PERIPHERY",
                port=BiologicalPort.PORT1_ACH,
                inputs={
                    "active_sensory_cells": n_sens_active,
                    "cva_pheromone": round(cva_pheromone_conc, 4),
                    "mate_odor": round(mate_odor_conc, 4),
                },
            )
            imp.suppress(
                "pan_connectome.dsx_fru_sex_dimorphism_mask",
                organ="GENOME_DIMORPHISM",
                port=BiologicalPort.PORT_SXL_TRA_SPLICING,
                reason=f"Cell-autonomous {self.dsx_isoform} + FruM={self.fru_m_active} masked {n_silenced} opposite-sex cells",
                rejected_above_floor=n_silenced,
                inputs={"dsx": self.dsx_isoform, "fru_m": self.fru_m_active},
            )
            imp.score(
                "pan_connectome.6hop_wave_189766_cells",
                organ="PAN_CONNECTOME",
                port=BiologicalPort.PORT3_AXO_AXONIC,
                potentials=[
                    {"id": "AL", "score": organ_rates["AL"]},
                    {"id": "MB", "score": organ_rates["MB"]},
                    {"id": "LH_SPIN", "score": organ_rates["LH_SPIN"]},
                    {"id": "CX", "score": organ_rates["CX"]},
                    {"id": "VNC", "score": organ_rates["VNC"]},
                ],
                inputs={"n_active_cells": readout.n_active_cells, "n_hops": n_synaptic_hops},
            )
            imp.route(
                "pan_connectome.dimorphic_and_dn_effectors",
                organ="AN_DN_VNC",
                port=BiologicalPort.PORT1_ACH,
                selected=[
                    k
                    for k, v in (
                        ("male_P1_courtship", readout.male_p1_courtship_rate),
                        ("male_pIP10_song", readout.male_pip10_song_rate),
                        ("female_pC1_receptivity", readout.female_pc1_receptivity_rate),
                        ("female_ovi_uterus", readout.female_ovi_oviposition_rate),
                        ("DN_walking", dn_clusters["walking"]),
                    )
                    if v > 0.10
                ],
                potentials=[
                    {"id": "male_P1", "score": readout.male_p1_courtship_rate},
                    {"id": "male_pIP10", "score": readout.male_pip10_song_rate},
                    {"id": "female_pC1", "score": readout.female_pc1_receptivity_rate},
                    {"id": "female_ovi", "score": readout.female_ovi_oviposition_rate},
                ],
            )
            if readout.male_pip10_song_rate > 0.15 and self.dsx_isoform == "DsxM":
                out_c = ImpulseOutcome.MALE_COURTSHIP_SONG
            elif readout.male_pip10_song_rate > 0.15 and self.dsx_isoform == "DsxF":
                out_c = ImpulseOutcome.FEMALE_FRUM_COURTSHIP_SONG
            elif readout.female_ovi_oviposition_rate > 0.15:
                out_c = ImpulseOutcome.FEMALE_OVIPOSITION
            elif readout.female_pc1_receptivity_rate > 0.15:
                out_c = ImpulseOutcome.FEMALE_SONG_RECEPTIVITY
            else:
                out_c = ImpulseOutcome.FORWARD_RUN
            imp.finish(outcome=out_c)

        return readout

    def decode_motor_effectors(self) -> MotorOut:
        """Decode all 1,026 BANC motor & visceral effector neurons directly into a 4D `MotorOut` command."""
        leg_drive = torch.zeros((6, 4), device=self.device, dtype=torch.float32)
        tarsus_grip = torch.ones((6,), device=self.device, dtype=torch.float32)

        walk_dn = self.group_mean_rate("dn_cluster_walking") + 0.5 * self.group_mean_rate("dn_dnp09")
        mdn_bwd = self.group_mean_rate("dn_mdn")
        gait_dir = -1.0 if mdn_bwd > walk_dn + 0.05 else 1.0

        # Tripod gait phase across L1(0), R1(1), L2(2), R2(3), L3(4), R3(5) modulated by real motor pool rates
        import math

        phase_base = float(self.step_counter) * 0.85 * gait_dir
        tripod_offsets = (0.0, math.pi, math.pi, 0.0, 0.0, math.pi)
        for leg_i, l_tag in enumerate(("L1", "R1", "L2", "R2", "L3", "R3")):
            c_fwd = self.group_mean_rate(f"mn_{l_tag}_coxa_fwd")
            c_bwd = self.group_mean_rate(f"mn_{l_tag}_coxa_bwd")
            t_lift = self.group_mean_rate(f"mn_{l_tag}_troch_lift")
            tb_flex = self.group_mean_rate(f"mn_{l_tag}_tibia_flex")
            tb_ext = self.group_mean_rate(f"mn_{l_tag}_tibia_ext")
            t_grip = self.group_mean_rate(f"mn_{l_tag}_tarsus_grip")

            phi = phase_base + tripod_offsets[leg_i]
            swing = max(0.0, math.sin(phi)) * min(1.0, walk_dn + mdn_bwd + c_fwd + c_bwd)
            stance = max(0.0, -math.sin(phi))

            leg_drive[leg_i, 0] = max(-1.0, min(1.0, gait_dir * (0.65 * math.cos(phi) + 1.5 * (c_fwd - c_bwd))))
            leg_drive[leg_i, 1] = max(-1.0, min(1.0, 0.8 * swing + 1.4 * t_lift))
            leg_drive[leg_i, 2] = max(-1.0, min(1.0, 1.6 * (tb_ext - tb_flex) + 0.5 * stance * gait_dir))
            leg_drive[leg_i, 3] = max(-1.0, min(1.0, 0.4 + 1.2 * t_grip))
            tarsus_grip[leg_i] = max(0.0, min(1.0, 1.0 - 0.85 * swing + 0.4 * t_grip))

        w_l = self.group_mean_rate("mn_wing_power_left")
        w_r = self.group_mean_rate("mn_wing_power_right")
        s_l = self.group_mean_rate("mn_wing_steer_left")
        s_r = self.group_mean_rate("mn_wing_steer_right")
        wing_drive = torch.tensor([min(1.0, w_l * 2.0), min(1.0, w_r * 2.0)], device=self.device, dtype=torch.float32)
        wing_pitch = torch.tensor(
            [max(-1.0, min(1.0, s_l - s_r)), max(-1.0, min(1.0, s_r - s_l))],
            device=self.device,
            dtype=torch.float32,
        )

        pip_idx = self.substrate.cell_groups["dimorphic_male_pip10"]
        pip10_r = float(self.a[pip_idx].topk(min(2, pip_idx.numel())).values.mean().item()) if pip_idx.numel() > 0 else 0.0
        tn1_r = self.group_mean_rate("dimorphic_male_tn1_song")
        song_amp = tn1_r if (self.fru_m_active and pip10_r > 0.15) else 0.0
        wing_song = torch.tensor([song_amp, 0.0], device=self.device, dtype=torch.float32)

        h_l = self.group_mean_rate("mn_haltere_left")
        h_r = self.group_mean_rate("mn_haltere_right")
        haltere_drive = torch.tensor(
            [max(0.2, min(1.0, h_l * 2.0)), max(0.2, min(1.0, h_r * 2.0))],
            device=self.device,
            dtype=torch.float32,
        )

        n_roll = self.group_mean_rate("mn_neck_roll_right") - self.group_mean_rate("mn_neck_roll_left")
        n_pitch = self.group_mean_rate("mn_neck_pitch")
        n_yaw = self.group_mean_rate("mn_neck_yaw_right") - self.group_mean_rate("mn_neck_yaw_left")
        neck_rpy = torch.tensor([n_roll, n_pitch, n_yaw], device=self.device, dtype=torch.float32).clamp(-1.0, 1.0)

        per_val = min(1.0, 2.2 * (self.group_mean_rate("mn_proboscis_per") + 0.5 * self.group_mean_rate("mn_pharynx_pump")))
        jump_val = min(1.0, 2.5 * max(self.group_mean_rate("mn_ttmn_jump"), self.group_mean_rate("dn_gf_dnp01")))
        ovi_curl = min(1.0, 2.0 * self.group_mean_rate("eff_oviposition_uterus"))

        return MotorOut(
            leg_joint_drive=leg_drive,
            tarsus_adhesion=tarsus_grip,
            wing_drive_lr=wing_drive,
            wing_pitch_lr=wing_pitch,
            wing_song_mode=wing_song,
            haltere_drive=haltere_drive,
            neck_torque_rpy=neck_rpy,
            proboscis_per=round(per_val, 4),
            jump_ttmn_burst=round(jump_val, 4),
            abdomen_bend_pitch_yaw=(round(-ovi_curl, 4), 0.0),
        )

    def step_sleep_consolidation(self, *, dcreb2b_blocked: bool = False) -> dict[str, float]:
        """Run active/quiet sleep homeostasis across the 3,121 Glial cells & consolidate MBON plasticity."""
        cg = self.substrate.cell_groups
        fbt_sleep = self.group_mean_rate("circuit_cx_fbt_sleep")
        self.glial_atp_pool = min(1.0, self.glial_atp_pool + 0.25)
        self.sleep_debt = max(0.0, self.sleep_debt - 0.30)
        if not dcreb2b_blocked:
            # Active sleep consolidates short-term MBON depression into stable long-term memory trace
            mbon_all = cg["circuit_mbon"]
            delta = self.synaptic_plasticity_gain[mbon_all] - 1.0
            self.synaptic_plasticity_gain[mbon_all] = (1.0 + 1.15 * delta).clamp(0.20, 1.80)
        return {
            "glial_atp_pool": round(self.glial_atp_pool, 4),
            "sleep_debt": round(self.sleep_debt, 4),
            "fbt_sleep_rate": round(fbt_sleep, 4),
            "mbon_mean_gain": round(float(self.synaptic_plasticity_gain[cg["circuit_mbon"]].mean().item()), 4),
        }
