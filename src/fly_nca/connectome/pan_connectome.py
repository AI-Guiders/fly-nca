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
        self.nt_id: torch.Tensor = raw["nt_id"].to(self.device)
        self.sex_code: torch.Tensor = raw["sex_code"].to(self.device)
        self.side_id: torch.Tensor = raw["side_id"].to(self.device)
        self.organ_id: torch.Tensor = raw["organ_id"].to(self.device)
        self.hemilineage_id: torch.Tensor = raw["hemilineage_id"].to(self.device)

        self.cell_groups: dict[str, torch.Tensor] = {
            k: v.to(self.device, dtype=torch.long) for k, v in raw["cell_groups"].items()
        }

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
            sex_mask[self.substrate.sex_code == 3] = 0.0  # Male-specific P1 dies under DsxF
            if self.fru_m_active:
                # Demir & Dickson (2005) / Clyne & Miesenböck (2008) (PMC10622783):
                # In DsxF + FruM+ females, P1 is absent, but FruM+ VNC song circuits (TN1/pIP10) are active!
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
        self.register_buffer("v", torch.zeros(n, device=self.device, dtype=torch.float32))
        self.register_buffer("a", torch.zeros(n, device=self.device, dtype=torch.float32))
        self.register_buffer("adapt", torch.zeros(n, device=self.device, dtype=torch.float32))

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
        n = self.substrate.n_total
        cg = self.substrate.cell_groups
        i_sens = torch.zeros(n, device=self.device, dtype=torch.float32)

        # 1. Visual periphery (retina + lamina + ocelli: 7,686 cells)
        vis_mean = float(sensor.ommatidia_light.mean().item())
        i_sens[cg["sens_visual_left"]] = 0.35 * vis_mean + 0.45 * float(sensor.looming_rate)
        i_sens[cg["sens_visual_right"]] = 0.35 * vis_mean + 0.45 * float(sensor.looming_rate)
        i_sens[cg["sens_ocellar"]] = abs(float(sensor.head_horizon_rpy[0].item())) + abs(float(sensor.head_horizon_rpy[1].item()))

        # Looming shadow on retina excites peripheral lobula LC4/LPLC2 visual feature detectors
        if sensor.looming_rate > 0.70:
            i_sens[cg["circuit_lc4_looming"]] = float(sensor.looming_rate) * 1.5

        # Small moving target (conspecific mate/rival) excites LC10a visual tracking neurons
        if sensor.small_target_salience > 0.05:
            sal = float(sensor.small_target_salience) * self.courtship_vigor_scale
            az = float(sensor.small_target_azimuth_rad)
            if az >= 0.0:
                i_sens[cg["circuit_lc10a_left"]] = sal * 1.3
                i_sens[cg["circuit_lc10a_right"]] = sal * 0.4
            else:
                i_sens[cg["circuit_lc10a_right"]] = sal * 1.3
                i_sens[cg["circuit_lc10a_left"]] = sal * 0.4

        # 2. Olfactory periphery (3,007 ORNs across 50 glomerular classes)
        odor_mean = 0.5 * (sensor.odor_glomeruli_lr[0] + sensor.odor_glomeruli_lr[1])
        hunger_boost = 1.0 + 0.4 * self.snpfr_hunger_scale * hunger_level
        for g_idx in range(50):
            val = float(odor_mean[g_idx].item())
            if val > 1e-4:
                g_cells = cg.get(f"orn_glom_{g_idx:02d}")
                if g_cells is not None and g_cells.numel() > 0:
                    i_sens[g_cells] = min(1.0, val * hunger_boost)

        # Sex pheromone cVA (Or67d -> ORN_DA1, 224 sensory cells) & mate odor (ORN_VA1v, 171 sensory cells)
        if cva_pheromone_conc > 0.0:
            i_sens[cg["sens_orn_DA1_cva"]] = float(cva_pheromone_conc) * self.courtship_vigor_scale
        if mate_odor_conc > 0.0:
            i_sens[cg["sens_orn_VA1v_mate"]] = float(mate_odor_conc) * self.courtship_vigor_scale

        # 3. Johnston's Organ (1,187 sensory cells: wind vector + courtship song vibration)
        wind_mag = float(sensor.wind_vector_ego.norm().item())
        song_vib = float(sensor.sound_vibration)
        i_sens[cg["sens_jo_left"]] = min(1.0, wind_mag + song_vib * 1.2 * self.courtship_vigor_scale)
        i_sens[cg["sens_jo_right"]] = min(1.0, wind_mag + song_vib * 1.2 * self.courtship_vigor_scale)

        # 4. Gustatory periphery (1,585 sensory cells across labellum, legs, pharynx)
        sweet = float(sensor.taste_sweet_bitter_amino[0].item())
        bitter = float(sensor.taste_sweet_bitter_amino[1].item())
        amino = float(sensor.taste_sweet_bitter_amino[2].item()) + (0.6 if substrate_yeast_rich else 0.0)
        i_sens[cg["sens_gust_labellum"]] = min(1.0, sweet * hunger_boost + amino)
        i_sens[cg["sens_gust_front_leg"]] = min(1.0, sweet + bitter + amino)
        i_sens[cg["sens_gust_hind_leg"]] = min(1.0, amino)
        i_sens[cg["sens_gust_pharynx"]] = min(1.0, sweet + amino)
        i_sens[cg["sens_nociception"]] = bitter

        # 5. Legs & Haltere Proprioception / Mechanoreception (9,135 sensory cells)
        leg_tags = ("L1", "R1", "L2", "R2", "L3", "R3")
        for leg_i, l_tag in enumerate(leg_tags):
            load_i = float(sensor.leg_ground_load[leg_i].item())
            touch_i = float(sensor.bristle_touch[leg_i].item())
            i_sens[cg[f"sens_leg_{l_tag}"]] = min(1.0, 0.6 * load_i + 0.8 * touch_i)

        halt_mag = float(sensor.haltere_coriolis_rpy.abs().mean().item())
        i_sens[cg["sens_haltere_left"]] = min(1.0, 0.3 + halt_mag)
        i_sens[cg["sens_haltere_right"]] = min(1.0, 0.3 + halt_mag)
        i_sens[cg["sens_crop_visceral"]] = float(sensor.crop_stretch)

        _ = conspecific_is_female
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
    ) -> FullConnectomeReadout:
        """Advance all 189,766 cells by `n_synaptic_hops` through the 5-port sparse CSR connectome matrices.

        Zero central-neuron shortcuts: sensory periphery inputs (`i_sens`) enter ONLY at the 16,563 sensory
        neurons (`sens_*` + peripheral optic glomeruli `LC4/LC10a`) and propagate across the 6,749,514
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
            # Port 3: Axo-axonic presynaptic modulation across all 189,766 axons
            a_col = (self.a * self.sex_cell_mask).unsqueeze(1)
            aa_mod = torch.matmul(sub.w_port3_aa, a_col).squeeze(1)
            a_presyn = (self.a * self.sex_cell_mask * (1.0 + 0.35 * aa_mod)).clamp(0.0, 1.0).unsqueeze(1)

            # Port 1 (ACh+ dendritic excitation) & Port 2 (GABA/Glu/His dendritic & root chloride shunting)
            i_exc = torch.matmul(sub.w_port1_ach, a_presyn).squeeze(1) + i_sens
            g_shunt = torch.matmul(sub.w_port2_shunt, a_presyn).squeeze(1)
            g_root = torch.matmul(sub.w_port2_root, a_presyn).squeeze(1)
            i_mod = torch.matmul(sub.w_port5_mod, a_presyn).squeeze(1)

            # 5-Port dendritic integration across all 189,766 cells
            i_dend = torch.tanh(2.6 * i_exc / (1.0 + 0.25 * sub.alpha_shunt * g_shunt)) / (1.0 + 2.0 * g_root)

            # Membrane potential Leaky Integrator with individual empirical cable-length tau_V[189,766]
            drive = i_dend + 0.25 * i_mod - 0.15 * self.adapt
            self.v = ((1.0 - alpha_v) * self.v + alpha_v * drive).clamp(-1.0, 1.0) * self.sex_cell_mask

            # Soma firing rate a_i(t + dt) and spike-frequency adaptation
            self.a = (torch.relu(self.v - theta) * 2.5).clamp(0.0, 1.0) * self.sex_cell_mask
            self.adapt = (0.92 * self.adapt + 0.08 * self.a).clamp(0.0, 1.0)

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
        return FullConnectomeReadout(
            n_active_cells=int((self.a > 0.01).sum().item()),
            mean_firing_rate=round(float(self.a.mean().item()), 5),
            organ_mean_rates=organ_rates,
            dn_cluster_rates=dn_clusters,
            male_p1_courtship_rate=round(self.group_mean_rate("dimorphic_male_p1"), 4),
            male_pip10_song_rate=round(self.group_mean_rate("dimorphic_male_pip10"), 4),
            male_tn1_wing_song_rate=round(self.group_mean_rate("dimorphic_male_tn1_song"), 4),
            female_pc1_receptivity_rate=round(fem_rec, 4),
            female_ovi_oviposition_rate=round(self.group_mean_rate("eff_oviposition_uterus"), 4),
            gf_dnp01_rate=round(self.group_mean_rate("dn_gf_dnp01"), 4),
            mdn_backward_rate=round(self.group_mean_rate("dn_mdn"), 4),
            dnp09_forward_rate=round(self.group_mean_rate("dn_dnp09"), 4),
            dng02_flight_rate=round(self.group_mean_rate("dn_dng02"), 4),
            lbl40_t3_rate=round(self.group_mean_rate("LBL40_t3"), 4),
        )
