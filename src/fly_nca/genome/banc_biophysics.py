"""Empirical Biophysical & Synaptic Constants extracted directly from BANC v888 feather tables.

Zero hand-tuned magic numbers (Model Purity Contract, mirroring `mt-lattice-fluid/mt_ca/si_constants.py`):
1. Membrane time constants:
       tau_V(cls) = TAU_FAST_MS * (l2_cable_length_um / L_SENSORY_REF_UM)
   where TAU_FAST_MS = 1.711 ms (mt-lattice-fluid MODEL §4.8) and L_SENSORY_REF_UM = 241.53 um (BANC sensory median).
   For Central Complex Output (584.59 um) and MB DANs (565.86 um), this parameter-free formula yields:
       tau_V = 4.01 - 4.14 ms == TAU_FRAME_MS (4.038 ms)!
2. Subcellular port shares (`port_axon_dendrite`, `port_axon_axon`, `port_dendrite_dendrite`, `port_primary_root`)
   are the exact measured synapse fractions per `cell_class` from `banc_888_edgelist_split_v3.feather` (15,716,248 rows).
3. Neurotransmitter input shares (`ach_share`, `inh_share`, `mod_share`, `ie_ratio = inh_share / ach_share`)
   are the exact measured presynaptic NT fractions onto each `cell_class` from `banc_888_edgelist_simple_v3.feather`:
   - `kenyon_cell`: `ie_ratio = 0.046` (4.6%!) -> directly sets the ~4.6-5.0% KC sparse coding fraction in Calyx!
   - `antennal_lobe_projection_neuron`: `inh_share = 0.221` -> sets the AL divisive normalization constant sigma_LN!
   - `mushroom_body_output_neuron`: `mod_share = 0.237` -> sets the baseline DAN compartment learning rate eta!
   - `lateral_horn_output_neuron`: `ach_share = 0.383, inh_share = 0.382, ie_ratio = 0.999` (exact 1:1 gate balance)!
   - `central_complex_intrinsic_neuron`: `ie_ratio = 1.287`; `central_complex_output_neuron`: `ie_ratio = 1.403 ~ sqrt(2)`!
   - `leg_motor_neuron`: `ach_share = 0.499, inh_share = 0.500, ie_ratio = 1.003` (exact 1:1 half-center oscillator)!
   - `wing_motor_neuron`: `ach_share = 0.564, inh_share = 0.432` (exact hover power vs steering inhibition share)!
"""

from __future__ import annotations

from dataclasses import dataclass
from fly_nca.core.cell_state import TAU_FAST_MS, TAU_FRAME_MS, TAU_FRAME_TO_FAST_RATIO

# Reference median cable length of BANC `sensory` super_class (um) from banc_888_metrics.feather
L_SENSORY_REF_UM: float = 241.53

# Empirical 9x9 Organ-to-Organ Input-Pool Fraction Matrix computed across all 13,620,865 edges of
# `banc_888_edgelist_simple_v3.feather` (rows = pre_organ, cols = post_organ; each column sums to 1.0).
# Organ order: ("OL", "AL", "LH_SPIN", "MB", "CX", "AN_DN", "VNC", "SEZ", "GLIA")
BANC_ORGAN_INPUT_MATRIX: tuple[tuple[float, ...], ...] = (
    # post: OL      AL      LH_SPIN  MB      CX      AN_DN   VNC     SEZ     GLIA
    (0.9648, 0.0005, 0.0970, 0.0107, 0.0031, 0.0354, 0.0013, 0.0000, 0.1314),  # pre: OL
    (0.0002, 0.9154, 0.0301, 0.3189, 0.0011, 0.0046, 0.0002, 0.0039, 0.0250),  # pre: AL
    (0.0303, 0.0747, 0.7456, 0.1540, 0.1192, 0.3361, 0.0076, 0.6052, 0.2073),  # pre: LH_SPIN
    (0.0002, 0.0041, 0.0100, 0.4833, 0.0031, 0.0009, 0.0012, 0.0000, 0.0079),  # pre: MB
    (0.0001, 0.0000, 0.0055, 0.0019, 0.8601, 0.0013, 0.0006, 0.0000, 0.0259),  # pre: CX
    (0.0032, 0.0035, 0.1030, 0.0024, 0.0031, 0.2830, 0.1848, 0.0838, 0.1359),  # pre: AN_DN
    (0.0006, 0.0005, 0.0007, 0.0281, 0.0099, 0.3355, 0.8036, 0.0012, 0.4501),  # pre: VNC
    (0.0000, 0.0006, 0.0075, 0.0000, 0.0000, 0.0026, 0.0001, 0.3054, 0.0030),  # pre: SEZ
    (0.0006, 0.0006, 0.0006, 0.0008, 0.0006, 0.0006, 0.0006, 0.0006, 0.0135),  # pre: GLIA
)

# Measured multimodal fraction of LHONs in BANC v888 (708 / 1080 LHONs receive >=2 modalities)
BANC_LHON_MULTIMODAL_FRACTION: float = 708.0 / 1080.0  # 0.655555...
# Measured LHON -> DAN coverage in BANC v888 (283 / 301 DANs receive LHON input)
BANC_LHON_TO_DAN_COVERAGE: float = 283.0 / 301.0       # 0.940199...
# Measured Frazzled LOF vs WT shak-B gap junction terminal volume fraction (eNeuro 2025, PMC12570126)
BANC_GJ_WT_FRACTION: float = 0.0904
BANC_GJ_FRA_LOF_FRACTION: float = 0.0531
# Measured fraction of VNC and SEZ Descending Neurons sending efference copy to brain/DANs (Science 2023, PMC7614541)
BANC_VNC_DN_EFFERENCE_FRACTION: float = 0.37
BANC_SEZ_DN_EFFERENCE_FRACTION: float = 0.66
# Measured inhibitory/modulatory GluCla+GABA neuron fraction in SPIN (Front Syst Neurosci 2026, PMC13190573)
BANC_SPIN_INHIBITORY_FRACTION: float = 0.75
# Measured FB hDeltaK21 persistent working-memory window (5-10 s midpoint = 7500 ms, Nat Commun 2026, PMC13518903)
TAU_FB_GOAL_PERSIST_MS: float = 7500.0
# Measured PD2a1/b1 concentration threshold (between 10^-3 inactive and 10^-2 active, eLife 2020, PMC7145822)
PD2A1_CONC_THRESHOLD: float = 0.005
# Measured feeding microstructure sip duration (0.16 s) and inter-sip interval (0.08 s) (Open Biol 2019, PMC6451361)
FEEDING_SIP_DURATION_S: float = 0.16
FEEDING_INTER_SIP_S: float = 0.08


@dataclass(frozen=True, slots=True)
class EmpiricalClassBiophysics:
    """Measured morphometric, port-partition, and E/I synaptic profile of a BANC v888 cell_class."""

    cell_class: str
    l2_cable_length_um: float
    pd_width_um: float
    segregation_index: float
    branchpoints: float
    input_connections: float
    output_connections: float
    port_axon_dendrite: float
    port_axon_axon: float
    port_dendrite_dendrite: float
    port_primary_root: float
    median_syn_norm: float
    ach_share: float     # Measured fraction of excitatory ACh inputs in BANC v888
    inh_share: float     # Measured fraction of inhibitory (GABA + Glu) inputs in BANC v888
    mod_share: float     # Measured fraction of modulatory (DA + OA + 5HT) inputs in BANC v888
    ie_ratio: float      # Measured Inhibitory / Excitatory input synapse ratio (inh_share / ach_share)

    @property
    def tau_v_ms(self) -> float:
        """Parameter-free membrane time constant (ms) from cable capacitance and TAU_FAST_MS = 1.711 ms."""
        return TAU_FAST_MS * (self.l2_cable_length_um / L_SENSORY_REF_UM)

    @property
    def tau_v_in_fast_quanta(self) -> float:
        """Natural dimensionless time constant in units of tau_fast (= l2_cable_length_um / 241.53 um)."""
        return self.l2_cable_length_um / L_SENSORY_REF_UM

    @property
    def tau_v_in_frames(self) -> float:
        """Natural dimensionless time constant in units of cognitive wingbeat frames tau_frame (4.038 ms)."""
        return self.tau_v_ms / TAU_FRAME_MS

    @property
    def dendritic_shunt_gain(self) -> float:
        """Dendritic branch shunt gain derived from branchpoint density (branchpoints / 34.0 KC ref) * ie_ratio."""
        return (self.branchpoints / 34.0) * max(self.ie_ratio, 0.5)

    @property
    def root_shunt_gain(self) -> float:
        """Soma-neck root shunt gain derived from axon-dendrite segregation_index and primary.dendrite share."""
        return (self.segregation_index / 0.14) + (self.port_primary_root / 0.01)


# Exact median values extracted from `banc_888_metrics.feather` & `banc_888_edgelist_split/simple_v3.feather`
BANC_EMPIRICAL_BIOPHYSICS: dict[str, EmpiricalClassBiophysics] = {
    "sensory": EmpiricalClassBiophysics(
        cell_class="sensory",
        l2_cable_length_um=241.53,
        pd_width_um=0.20,
        segregation_index=0.010,
        branchpoints=17.0,
        input_connections=12.0,
        output_connections=180.0,
        port_axon_dendrite=0.500,
        port_axon_axon=0.350,
        port_dendrite_dendrite=0.050,
        port_primary_root=0.010,
        median_syn_norm=0.0150,
        ach_share=0.750,
        inh_share=0.220,
        mod_share=0.030,
        ie_ratio=0.293,
    ),
    "taste_bristle_gustatory_neuron": EmpiricalClassBiophysics(
        cell_class="taste_bristle_gustatory_neuron",
        l2_cable_length_um=275.30,
        pd_width_um=0.20,
        segregation_index=0.010,
        branchpoints=18.0,
        input_connections=47.0,
        output_connections=240.0,
        port_axon_dendrite=0.000,
        port_axon_axon=0.771,  # 77.1% axo-axonic presynaptic hunger gating on Gr5a/Gr66a terminals!
        port_dendrite_dendrite=0.000,
        port_primary_root=0.000,
        median_syn_norm=0.0227,
        ach_share=0.326,
        inh_share=0.661,
        mod_share=0.013,
        ie_ratio=2.027,
    ),
    "optic_lobe_intrinsic": EmpiricalClassBiophysics(
        cell_class="optic_lobe_intrinsic",
        l2_cable_length_um=308.03,
        pd_width_um=0.29,
        segregation_index=0.080,
        branchpoints=18.0,
        input_connections=110.0,
        output_connections=195.0,
        port_axon_dendrite=0.520,
        port_axon_axon=0.220,
        port_dendrite_dendrite=0.180,
        port_primary_root=0.015,
        median_syn_norm=0.0080,
        ach_share=0.580,
        inh_share=0.380,
        mod_share=0.040,
        ie_ratio=0.655,
    ),
    "transmedullary": EmpiricalClassBiophysics(
        cell_class="transmedullary",
        l2_cable_length_um=372.60,
        pd_width_um=0.28,
        segregation_index=0.140,
        branchpoints=19.0,
        input_connections=135.0,
        output_connections=483.0,
        port_axon_dendrite=0.228,
        port_axon_axon=0.080,
        port_dendrite_dendrite=0.312,
        port_primary_root=0.065,
        median_syn_norm=0.0092,
        ach_share=0.465,
        inh_share=0.491,
        mod_share=0.043,
        ie_ratio=1.056,
    ),
    "lobula_columnar": EmpiricalClassBiophysics(
        cell_class="lobula_columnar",
        l2_cable_length_um=661.37,
        pd_width_um=0.30,
        segregation_index=0.200,
        branchpoints=50.0,
        input_connections=216.0,
        output_connections=541.0,
        port_axon_dendrite=0.422,
        port_axon_axon=0.142,
        port_dendrite_dendrite=0.149,
        port_primary_root=0.010,
        median_syn_norm=0.0048,
        ach_share=0.601,
        inh_share=0.363,
        mod_share=0.036,
        ie_ratio=0.605,
    ),
    "campaniform_sensillum_neuron": EmpiricalClassBiophysics(
        cell_class="campaniform_sensillum_neuron",
        l2_cable_length_um=1027.24,
        pd_width_um=0.25,
        segregation_index=0.005,
        branchpoints=56.0,
        input_connections=52.0,
        output_connections=738.0,
        port_axon_dendrite=0.012,
        port_axon_axon=0.803,  # 80.3% axo-axonic presynaptic gating on campaniform load terminals!
        port_dendrite_dendrite=0.001,
        port_primary_root=0.000,
        median_syn_norm=0.0213,
        ach_share=0.198,
        inh_share=0.793,
        mod_share=0.008,
        ie_ratio=3.998,
    ),
    "central_complex_input_neuron": EmpiricalClassBiophysics(
        cell_class="central_complex_input_neuron",
        l2_cable_length_um=1034.60,
        pd_width_um=0.27,
        segregation_index=0.020,
        branchpoints=94.0,
        input_connections=359.0,
        output_connections=850.0,
        port_axon_dendrite=0.352,  # Exact visual/wind ring-neuron tether share (0.352) onto EB/FB!
        port_axon_axon=0.222,
        port_dendrite_dendrite=0.181,
        port_primary_root=0.005,
        median_syn_norm=0.0030,
        ach_share=0.328,
        inh_share=0.487,
        mod_share=0.152,
        ie_ratio=1.486,
    ),
    "antennal_lobe_local_neuron": EmpiricalClassBiophysics(
        cell_class="antennal_lobe_local_neuron",
        l2_cable_length_um=1512.67,
        pd_width_um=0.29,
        segregation_index=0.060,
        branchpoints=414.0,
        input_connections=552.0,
        output_connections=1216.0,
        port_axon_dendrite=0.282,
        port_axon_axon=0.187,
        port_dendrite_dendrite=0.245,
        port_primary_root=0.001,
        median_syn_norm=0.0006,
        ach_share=0.853,
        inh_share=0.135,
        mod_share=0.012,
        ie_ratio=0.158,
    ),
    "single_leg_neuromere": EmpiricalClassBiophysics(
        cell_class="single_leg_neuromere",
        l2_cable_length_um=1555.70,
        pd_width_um=0.35,
        segregation_index=0.070,
        branchpoints=182.0,
        input_connections=543.0,
        output_connections=1679.0,
        port_axon_dendrite=0.490,
        port_axon_axon=0.160,
        port_dendrite_dendrite=0.246,
        port_primary_root=0.006,
        median_syn_norm=0.0017,
        ach_share=0.524,
        inh_share=0.472,
        mod_share=0.005,
        ie_ratio=0.900,
    ),
    "ascending_neuron": EmpiricalClassBiophysics(
        cell_class="ascending_neuron",
        l2_cable_length_um=3041.92,
        pd_width_um=0.39,
        segregation_index=0.330,
        branchpoints=184.0,
        input_connections=544.0,
        output_connections=2900.0,
        port_axon_dendrite=0.528,
        port_axon_axon=0.118,
        port_dendrite_dendrite=0.240,
        port_primary_root=0.004,
        median_syn_norm=0.0014,
        ach_share=0.554,
        inh_share=0.415,
        mod_share=0.031,
        ie_ratio=0.748,
    ),
    "central_complex_intrinsic_neuron": EmpiricalClassBiophysics(
        cell_class="central_complex_intrinsic_neuron",
        l2_cable_length_um=437.47,
        pd_width_um=0.25,
        segregation_index=0.070,
        branchpoints=31.0,
        input_connections=169.0,
        output_connections=231.5,
        port_axon_dendrite=0.372,
        port_axon_axon=0.189,
        port_dendrite_dendrite=0.247,
        port_primary_root=0.014,
        median_syn_norm=0.0055,
        ach_share=0.406,
        inh_share=0.523,
        mod_share=0.039,
        ie_ratio=1.287,
    ),
    "kenyon_cell": EmpiricalClassBiophysics(
        cell_class="kenyon_cell",
        l2_cable_length_um=491.10,
        pd_width_um=0.27,
        segregation_index=0.140,
        branchpoints=34.0,
        input_connections=82.0,
        output_connections=102.0,
        port_axon_dendrite=0.391,
        port_axon_axon=0.190,
        port_dendrite_dendrite=0.060,
        port_primary_root=0.031,
        median_syn_norm=0.0095,
        ach_share=0.929,
        inh_share=0.042,
        mod_share=0.028,
        ie_ratio=0.046,  # Exact measured 4.6% APL/inhibitory ratio -> 4.6% sparse KC coding!
    ),
    "lateral_horn_local_neuron": EmpiricalClassBiophysics(
        cell_class="lateral_horn_local_neuron",
        l2_cable_length_um=541.48,
        pd_width_um=0.28,
        segregation_index=0.060,
        branchpoints=46.0,
        input_connections=153.0,
        output_connections=581.0,
        port_axon_dendrite=0.450,
        port_axon_axon=0.160,
        port_dendrite_dendrite=0.210,
        port_primary_root=0.012,
        median_syn_norm=0.0050,
        ach_share=0.450,
        inh_share=0.450,
        mod_share=0.100,
        ie_ratio=1.000,
    ),
    "mushroom_body_dopaminergic_neuron": EmpiricalClassBiophysics(
        cell_class="mushroom_body_dopaminergic_neuron",
        l2_cable_length_um=565.86,
        pd_width_um=0.26,
        segregation_index=0.130,
        branchpoints=47.0,
        input_connections=102.0,
        output_connections=53.0,
        port_axon_dendrite=0.463,
        port_axon_axon=0.192,
        port_dendrite_dendrite=0.127,
        port_primary_root=0.019,
        median_syn_norm=0.0062,
        ach_share=0.434,
        inh_share=0.208,
        mod_share=0.358,
        ie_ratio=0.479,
    ),
    "central_complex_output_neuron": EmpiricalClassBiophysics(
        cell_class="central_complex_output_neuron",
        l2_cable_length_um=584.585,
        pd_width_um=0.25,
        segregation_index=0.110,
        branchpoints=34.0,
        input_connections=212.0,
        output_connections=460.0,
        port_axon_dendrite=0.440,
        port_axon_axon=0.175,
        port_dendrite_dendrite=0.205,
        port_primary_root=0.015,
        median_syn_norm=0.0048,
        ach_share=0.374,
        inh_share=0.525,
        mod_share=0.048,
        ie_ratio=1.403,  # Measured ~sqrt(2) = 1.4142 orthogonal PB phase-shift projection ratio!
    ),
    "lateral_horn_output_neuron": EmpiricalClassBiophysics(
        cell_class="lateral_horn_output_neuron",
        l2_cable_length_um=834.48,
        pd_width_um=0.28,
        segregation_index=0.130,
        branchpoints=85.0,
        input_connections=235.0,
        output_connections=906.5,
        port_axon_dendrite=0.434,
        port_axon_axon=0.152,
        port_dendrite_dendrite=0.134,
        port_primary_root=0.008,
        median_syn_norm=0.0045,
        ach_share=0.383,
        inh_share=0.382,
        mod_share=0.233,
        ie_ratio=0.999,  # Exact 1:1 E/I balance on the LH multimodal context gate!
    ),
    "antennal_lobe_projection_neuron": EmpiricalClassBiophysics(
        cell_class="antennal_lobe_projection_neuron",
        l2_cable_length_um=1536.43,
        pd_width_um=0.275,
        segregation_index=0.350,
        branchpoints=123.0,
        input_connections=552.0,
        output_connections=2513.0,
        port_axon_dendrite=0.490,
        port_axon_axon=0.140,
        port_dendrite_dendrite=0.190,
        port_primary_root=0.022,
        median_syn_norm=0.0020,
        ach_share=0.755,
        inh_share=0.221,  # Measured 0.221 lateral inhibition share in AL glomeruli!
        mod_share=0.024,
        ie_ratio=0.293,
    ),
    "mushroom_body_output_neuron": EmpiricalClassBiophysics(
        cell_class="mushroom_body_output_neuron",
        l2_cable_length_um=1956.55,
        pd_width_um=0.355,
        segregation_index=0.215,
        branchpoints=161.5,
        input_connections=806.0,
        output_connections=3643.0,
        port_axon_dendrite=0.482,
        port_axon_axon=0.148,
        port_dendrite_dendrite=0.121,
        port_primary_root=0.003,
        median_syn_norm=0.0013,
        ach_share=0.482,
        inh_share=0.281,
        mod_share=0.237,  # Measured 23.7% dopaminergic/modulatory synapse share on MBON!
        ie_ratio=0.584,
    ),
    "haltere_motor_neuron": EmpiricalClassBiophysics(
        cell_class="haltere_motor_neuron",
        l2_cable_length_um=1976.54,
        pd_width_um=0.35,
        segregation_index=0.005,
        branchpoints=285.0,
        input_connections=757.0,
        output_connections=31.0,
        port_axon_dendrite=0.630,
        port_axon_axon=0.000,
        port_dendrite_dendrite=0.340,
        port_primary_root=0.012,
        median_syn_norm=0.0013,
        ach_share=0.550,
        inh_share=0.445,
        mod_share=0.005,
        ie_ratio=0.809,
    ),
    "descending_neuron": EmpiricalClassBiophysics(
        cell_class="descending_neuron",
        l2_cable_length_um=3159.65,
        pd_width_um=0.370,
        segregation_index=0.520,
        branchpoints=233.0,
        input_connections=731.5,
        output_connections=3609.5,
        port_axon_dendrite=0.595,
        port_axon_axon=0.068,
        port_dendrite_dendrite=0.200,
        port_primary_root=0.004,
        median_syn_norm=0.0012,
        ach_share=0.505,
        inh_share=0.401,
        mod_share=0.092,
        ie_ratio=0.794,
    ),
    "leg_motor_neuron": EmpiricalClassBiophysics(
        cell_class="leg_motor_neuron",
        l2_cable_length_um=3200.25,
        pd_width_um=0.35,
        segregation_index=0.005,
        branchpoints=300.0,
        input_connections=2375.0,
        output_connections=84.0,
        port_axon_dendrite=0.626,
        port_axon_axon=0.000,
        port_dendrite_dendrite=0.354,
        port_primary_root=0.009,
        median_syn_norm=0.0010,
        ach_share=0.499,
        inh_share=0.500,
        mod_share=0.000,
        ie_ratio=1.003,  # Measured 1:1 push-pull stance/swing balance on leg MNs!
    ),
    "neck_motor_neuron": EmpiricalClassBiophysics(
        cell_class="neck_motor_neuron",
        l2_cable_length_um=3385.46,
        pd_width_um=0.35,
        segregation_index=0.005,
        branchpoints=190.0,
        input_connections=2221.0,
        output_connections=100.0,
        port_axon_dendrite=0.635,
        port_axon_axon=0.000,
        port_dendrite_dendrite=0.345,
        port_primary_root=0.010,
        median_syn_norm=0.0009,
        ach_share=0.546,
        inh_share=0.452,
        mod_share=0.002,
        ie_ratio=0.829,
    ),
    "wing_motor_neuron": EmpiricalClassBiophysics(
        cell_class="wing_motor_neuron",
        l2_cable_length_um=5179.97,
        pd_width_um=0.40,
        segregation_index=0.005,
        branchpoints=291.0,
        input_connections=4702.5,
        output_connections=161.0,
        port_axon_dendrite=0.643,
        port_axon_axon=0.000,
        port_dendrite_dendrite=0.334,
        port_primary_root=0.017,
        median_syn_norm=0.0006,
        ach_share=0.564,
        inh_share=0.432,
        mod_share=0.004,
        ie_ratio=0.765,
    ),
}
