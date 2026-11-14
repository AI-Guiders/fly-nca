"""L0 FlyGenome: French Flag -> Hox cascade, 17-shTF temporal clock, Notch A/B, and Sex (fru/dsx)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

try:
    import pyarrow.feather as _feather  # Import before torch on Windows to avoid arrow/c10 DLL clash
except ImportError:
    _feather = None

import torch

# Canonical 17 shared temporal transcription factors (17 shTF birth-order code, Nature 2026)
SHTF_17_NAMES: tuple[str, ...] = (
    "Imp",      # 0: Wave 1 embryonic / early secondary marker
    "chinmo",   # 1: Early temporal identity
    "lin-28",   # 2: Early RNA-binding factor
    "mamo",     # 3: Early-mid transition selector
    "Hr39",     # 4: Nuclear receptor temporal step
    "ab",       # 5: Abrupt mid-window factor
    "jim",      # 6: Mid-temporal zinc-finger factor
    "br",       # 7: Broad (terminal selector in VNC/brain; ectopic shift changes 14A arbor identity!)
    "Eip93F",   # 8: Ecdysone-induced mid-late factor
    "dati",     # 9: Late secondary wave marker
    "Syp",      # 10: Syncrip late RNA-binding factor
    "ey",       # 11: Eyeless late temporal TF (MB/CX)
    "svp",      # 12: Seven-up switching factor
    "hb",       # 13: Hunchback early neuroblast window
    "Kr",       # 14: Kruppel window
    "pdm",      # 15: Pdm1/2 window
    "cas",      # 16: Castor late window
)

# Canonical 8 Anterior-Posterior body/CNS neuromere segments after French Flag + Hox patterning
SEGMENTS_8: tuple[str, ...] = (
    "OL",       # 0: Optic Lobe (visual hex strata, 892 columns)
    "CB_ant",   # 1: Anterior Central Brain (AL, MB, LH-SPIN, AOTU)
    "CB_post",  # 2: Posterior Central Brain (CX: PB/EB/FB/NO/LAL, AMMC/WED, AN/DN gates)
    "SEZ",      # 3: Subesophageal Zone (Dfd/Scr Hox: taste, proboscis, Fdg, VPM4)
    "T1",       # 4: Prothoracic neuromere (Antp Hox: front legs L1/R1, neck 49 MN)
    "T2",       # 5: Mesothoracic neuromere (Antp/Ubx Hox: middle legs L2/R2, wings 64 MN, GF->TTMn)
    "T3",       # 6: Metathoracic neuromere (Ubx Hox: hind legs L3/R3 with dominant LBL40, halteres 27 MN)
    "Abdomen",  # 7: Abdominal neuromeres A1-A8 (Abd-A/Abd-B Hox: abdomen 168 MN, visceral)
)

# Fast neurotransmitter sign law (Bates et al., Nature 2026):
# +1 = Acetylcholine (ACh, excitatory), -1 = GABA or Glutamate (inhibitory in CNS via Rdl / GluCla)
NT_SIGN_MAP: dict[str, int] = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "dopamine": +1,
    "octopamine": +1,
    "serotonin": +1,
    "histamine": -1,
    "tyramine": -1,
}


@dataclass(frozen=True, slots=True)
class HemilineageSpec:
    """Specification of a neuroblast hemilineage (Lineage x Notch ON/OFF)."""

    name: str
    segment: str
    notch_on: bool
    fast_nt: str                   # 'acetylcholine' | 'gaba' | 'glutamate' | 'histamine'
    cotransmitter: str | None      # Optional 'dopamine' | 'octopamine' | 'serotonin' | 'neuropeptide'
    organ_target: str              # 'OL' | 'AL' | 'LH_SPIN' | 'MB' | 'CX' | 'AN_DN' | 'VNC' | 'SEZ' | 'GLIA'
    wave1_count: int               # Primary embryonic neurons (Larva core ~3,016 total)
    wave2_count: int               # Secondary post-embryonic neurons (Adult expansion)
    sex_dimorphic_fraction: float = 0.0


def build_canonical_hemilineage_catalog() -> list[HemilineageSpec]:
    """Canonical hemilineage catalog grounded in BANC v888 (reproducing the exact organ & NT proportions)."""
    return [
        # --- Mushroom Body & Olfactory / Valence (CB_ant) ---
        HemilineageSpec("MBp", "CB_ant", True, "acetylcholine", None, "MB", 220, 4332, 0.0),
        HemilineageSpec("MB_APL", "CB_ant", False, "gaba", "octopamine", "MB", 2, 0, 0.0),
        HemilineageSpec("CREa1_PAM", "CB_ant", True, "acetylcholine", "dopamine", "MB", 30, 160, 0.02),
        HemilineageSpec("PPL1_DAN", "CB_ant", False, "glutamate", "dopamine", "MB", 24, 126, 0.02),
        HemilineageSpec("MBON_ACh", "CB_ant", True, "acetylcholine", None, "MB", 20, 32, 0.0),
        HemilineageSpec("MBON_GABA_Glu", "CB_ant", False, "gaba", "glutamate", "MB", 18, 27, 0.0),
        HemilineageSpec("AL_adPN", "CB_ant", True, "acetylcholine", None, "AL", 100, 350, 0.03),
        HemilineageSpec("AL_vPN_LN", "CB_ant", False, "gaba", None, "AL", 60, 240, 0.0),
        HemilineageSpec("LHON_ACh", "CB_ant", True, "acetylcholine", None, "LH_SPIN", 80, 520, 0.08),
        HemilineageSpec("LHON_Glu_GABA", "CB_ant", False, "glutamate", None, "LH_SPIN", 70, 410, 0.08),
        HemilineageSpec("SPIN_GluCla", "CB_ant", False, "glutamate", None, "LH_SPIN", 120, 2000, 0.05),
        HemilineageSpec("SPIN_SMP353", "CB_ant", True, "acetylcholine", None, "LH_SPIN", 40, 684, 0.05),
        HemilineageSpec("P1_aSP_courtship", "CB_ant", True, "acetylcholine", None, "LH_SPIN", 0, 180, 1.0),
        # --- Central Complex & Posterior Brain (CB_post) ---
        HemilineageSpec("DM1_CX_EPG_PEN", "CB_post", True, "acetylcholine", None, "CX", 60, 640, 0.0),
        HemilineageSpec("DM2_CX_Delta7_Ring", "CB_post", False, "glutamate", "gaba", "CX", 40, 520, 0.0),
        HemilineageSpec("DM3_CX_FB_vDelta_hDelta", "CB_post", True, "acetylcholine", "neuropeptide", "CX", 80, 1150, 0.02),
        HemilineageSpec("DM4_CX_dFB_Sleep_PFL", "CB_post", False, "gaba", "acetylcholine", "CX", 40, 420, 0.02),
        HemilineageSpec("WED_AMMC_JO", "CB_post", True, "acetylcholine", None, "AL", 150, 1029, 0.04),
        HemilineageSpec("DN_Command_ACh", "CB_post", True, "acetylcholine", None, "AN_DN", 180, 730, 0.06),
        HemilineageSpec("DN_Inhib_Glu_GABA", "CB_post", False, "glutamate", "gaba", "AN_DN", 80, 326, 0.04),
        # --- SEZ (Taste, Feeding, VPM4 OA+Glu, Fdg) ---
        HemilineageSpec("SEZ_Gr_ACh", "SEZ", True, "acetylcholine", None, "SEZ", 220, 980, 0.02),
        HemilineageSpec("SEZ_VPM4_OAN", "SEZ", True, "glutamate", "octopamine", "SEZ", 24, 96, 0.05),
        HemilineageSpec("SEZ_IPC_NPF_Pept", "SEZ", False, "gaba", "serotonin", "SEZ", 40, 180, 0.0),
        # --- Optic Lobe (OL: Hex Columns & VPNs) ---
        HemilineageSpec("OL_Photoreceptor_His", "OL", False, "histamine", None, "OL", 80, 1720, 0.0),
        HemilineageSpec("OL_Columnar_ACh", "OL", True, "acetylcholine", None, "OL", 200, 38500, 0.0),
        HemilineageSpec("OL_Columnar_Glu", "OL", False, "glutamate", None, "OL", 120, 14800, 0.0),
        HemilineageSpec("OL_Inhib_GABA", "OL", False, "gaba", None, "OL", 100, 11800, 0.0),
        HemilineageSpec("VPNd1_VPNd2", "OL", True, "acetylcholine", None, "OL", 110, 7206, 0.02),
        # --- VNC Serial Thorax & Abdomen (T1, T2, T3, Abdomen) ---
        HemilineageSpec("03A_20A_Leg_Exc", "T1", True, "acetylcholine", None, "VNC", 140, 2200, 0.02),
        HemilineageSpec("06A_09A_12B_19A_Inh", "T1", False, "gaba", None, "VNC", 120, 1800, 0.02),
        HemilineageSpec("14A_16B_Leg_Glu", "T2", False, "glutamate", None, "VNC", 110, 1650, 0.04),
        HemilineageSpec("07B_Flight_Leg_ACh", "T2", True, "acetylcholine", "octopamine", "VNC", 130, 1950, 0.08),
        HemilineageSpec("14B_LBL40_Stance", "T3", True, "acetylcholine", None, "VNC", 90, 1400, 0.02),
        HemilineageSpec("13B_LUL130_Swing", "T3", False, "gaba", None, "VNC", 90, 1350, 0.02),
        HemilineageSpec("AN19B_Ascending", "T2", True, "acetylcholine", None, "AN_DN", 160, 1689, 0.03),
        HemilineageSpec("VNC_Motor_805", "T2", True, "glutamate", "acetylcholine", "VNC", 180, 625, 0.05),
        HemilineageSpec("Abd_Visceral_Motor", "Abdomen", True, "glutamate", "serotonin", "VNC", 100, 950, 0.25),
        # --- Glia (SPG, EG, ALG) ---
        HemilineageSpec("Glia_NPG_EG_ALG", "CB_ant", False, "gaba", None, "GLIA", 208, 12542, 0.0),
    ]


@dataclass(frozen=True, slots=True)
class SyncytialEmbryoReadout:
    """Readout of Syncytial Blastoderm -> Cellularization -> Gap/Pair-Rule/Hox Embryogenesis NCA."""

    segment_ids: torch.Tensor        # [N] Hox neuromere identity in 0..7 (SEGMENTS_8)
    bicoid_protein: torch.Tensor     # [N] Anterior-to-posterior Bcd gradient (diffused from pole 0)
    nanos_protein: torch.Tensor      # [N] Posterior-to-anterior Nos gradient (diffused from pole N-1)
    caudal_protein: torch.Tensor     # [N] Cad gradient (translationally repressed by Bcd in anterior)
    hunchback_protein: torch.Tensor  # [N] Master Hb gradient (translationally repressed by Nos in posterior)
    kruppel_protein: torch.Tensor    # [N] Central Kr gap domain
    knirps_protein: torch.Tensor     # [N] Posterior-abdominal Kni gap domain
    zelda_7_stripes: torch.Tensor    # [N] Early Zelda-activated 7-stripe pair-rule (even-skipped) pattern
    opa_14_stripes: torch.Tensor     # [N] Late Opa-activated frequency-doubled 14-stripe (engrailed) pattern
    pole_cells_mask: torch.Tensor    # [N] Posterior primordial germ cells (budded off first before cellularization)


@dataclass
class FlyGenome:
    """NCA-Genome (L0): syncytial embryogenesis, French Flag enhancers, and CAM expression matrix."""

    species_id: str = "drosophila.melanogaster.banc-v888"
    cam_dim: int = 16
    gap_dim: int = 8
    hemilineages: list[HemilineageSpec] = field(default_factory=build_canonical_hemilineage_catalog)
    # Ectopic transcription factor overrides for in-silico genetics (e.g. Broad shift in 14A, Nature 2026)
    ectopic_tf_overrides: dict[str, str] = field(default_factory=dict)

    def run_syncytial_embryogenesis_nca(
        self,
        n_positions: int = 64,
        syncytial_diffusion_steps: int = 320,
        gap_steps: int = 24,
        *,
        bcd_mutant: bool = False,
        nos_mutant: bool = False,
        opa_mutant: bool = False,
    ) -> SyncytialEmbryoReadout:
        """Pure Local Syncytial Blastoderm -> Cellularization -> Gap/Pair-Rule/Hox NCA (Zero global coordinate x!).

        Implements the exact biological sequence from Nobel 1995 (Nüsslein-Volhard & Wieschaus) and
        PLoS Genetics 2021 (PMC8516289):
        1. Oogenesis: nurse cells anchor `bicoid` mRNA at the anterior pole (index 0) via Staufen after
           microtubule polarity reversal, `nanos` mRNA at the posterior pole (index N-1, where pole cells
           bud off first), and uniform maternal `hunchback` and `caudal` mRNAs across the syncytium.
        2. Syncytial Blastoderm (~5,000 nuclei in shared cytoplasm before cellularization): Bicoid and Nanos
           proteins diffuse purely between adjacent nuclei (`roll(±1)`) with decay — NO global `linspace(0,1)`!
        3. Translational Repression: Bicoid binds the 3'-UTR of `caudal` mRNA to block anterior Caudal translation;
           Nanos binds `hunchback` mRNA to block posterior Hunchback translation.
        4. Early Network (`Zelda` pioneer factor): Hunchback master gradient + mutual cross-repression locks in
           Gap domains (`hb`, `Kr`, `kni`, `tll`) and 7 stripe-specific `even-skipped` pair-rule stripes.
        5. Cellularization & Late Network (`Odd-paired / Opa` pioneer factor): plasma membranes close around
           nuclei, `Opa` rewires pair-rule enhancers for frequency doubling (7 -> 14 `engrailed` parasegments),
           and local `(Bcd, Cad, Hb, Kr, Kni, Tll)` concentrations lock in the 8 Hox neuromeres (0..7).
        """
        bcd = torch.zeros(n_positions, dtype=torch.float32)
        nos = torch.zeros(n_positions, dtype=torch.float32)
        hb_mrna = torch.full((n_positions,), 0.5, dtype=torch.float32)
        cad_mrna = torch.full((n_positions,), 0.5, dtype=torch.float32)
        pole_cells = torch.zeros(n_positions, dtype=torch.float32)
        pole_cells[-2:] = 1.0  # Posterior pole cells (primordial germ cells) segregate first!

        # 1. Pure local 1D neighbor diffusion in the syncytial cytoplasm (no global coordinate x!)
        decay = 0.00045
        for _ in range(max(syncytial_diffusion_steps, 800)):
            if not bcd_mutant:
                bcd[0] = 1.0
            if not nos_mutant:
                nos[-1] = 1.0
            bcd_l = torch.cat([bcd[:1], bcd[:-1]])
            bcd_r = torch.cat([bcd[1:], bcd[-1:]])
            nos_l = torch.cat([nos[:1], nos[:-1]])
            nos_r = torch.cat([nos[1:], nos[-1:]])
            bcd = (1.0 - decay) * (0.50 * bcd + 0.25 * (bcd_l + bcd_r))
            nos = (1.0 - decay) * (0.50 * nos + 0.25 * (nos_l + nos_r))
        if not bcd_mutant:
            bcd[0] = 1.0
        if not nos_mutant:
            nos[-1] = 1.0

        # 2. Translational repression (Bcd -| cad mRNA; Nos -| hb mRNA)
        cad = (cad_mrna * (2.0 / (1.0 + 4.5 * bcd))).clamp(0.0, 1.0)
        hb_mat = (hb_mrna * (2.0 / (1.0 + 6.5 * nos))).clamp(0.0, 1.0)

        # 3. Gap gene cross-repression network (hb, Kr, kni, tll)
        hb = hb_mat.clone()
        kr = torch.zeros(n_positions, dtype=torch.float32)
        kni = torch.zeros(n_positions, dtype=torch.float32)
        tll = torch.zeros(n_positions, dtype=torch.float32)
        tll[-4:] = 1.0
        if not bcd_mutant:
            tll[:2] = 0.7

        for _ in range(gap_steps):
            # Zygotic Hb activated by Bcd + self-cooperativity, repressed by Kr & Nos
            hb_drive = torch.sigmoid(4.5 * (0.85 * bcd + 0.45 * hb - 0.35 - 0.50 * kr - 0.85 * nos))
            if nos_mutant and bcd_mutant:
                hb_drive = torch.zeros_like(hb)
            elif nos_mutant:
                # In nanos^- mutant, unrepressed maternal hb_mat remains high in the posterior!
                hb_drive = torch.maximum(hb_drive, hb_mat * 0.85)
            # Kr bell-curve activation at intermediate Hb (French Flag middle threshold), repressed by Kni & Tll
            kr_drive = torch.relu(1.0 - 2.2 * (hb - 0.48).abs()) * torch.sigmoid(5.0 * (0.55 - kni - tll))
            # Kni activated by Caudal in low-Hb region, strongly repressed by Hb, Kr, and Tll
            kni_drive = torch.sigmoid(7.0 * (cad - 1.25 * hb - 0.75 * kr - 1.2 * tll - 0.12))
            hb = (0.70 * hb + 0.30 * hb_drive).clamp(0.0, 1.0)
            kr = (0.70 * kr + 0.30 * kr_drive).clamp(0.0, 1.0)
            kni = (0.70 * kni + 0.30 * kni_drive).clamp(0.0, 1.0)

        # 4. Early Zelda 7-stripe pair-rule enhancers & late Opa 14-stripe frequency doubling (PMC8516289 §12)
        local_morphogen_phase = (bcd / (bcd + cad + 1e-4)).clamp(0.0, 1.0)
        zelda_7 = (torch.cos(2.0 * 3.14159265 * 7.0 * local_morphogen_phase) > 0.15).to(torch.float32)
        opa_14 = (
            zelda_7
            if opa_mutant
            else (torch.cos(2.0 * 3.14159265 * 14.0 * local_morphogen_phase) > 0.15).to(torch.float32)
        )

        # 5. Hox neuromere assignment (0..7 in SEGMENTS_8) purely from local morphogen & Gap concentrations!
        seg_ids = torch.zeros(n_positions, dtype=torch.long)
        for i in range(n_positions):
            b = float(bcd[i].item())
            c = float(cad[i].item())
            h = float(hb[i].item())
            k = float(kr[i].item())
            kn = float(kni[i].item())
            tl = float(tll[i].item())
            if b > 0.78 and h > 0.80:
                seg_ids[i] = 0  # Brain_PCD (otd/btd/labial)
            elif b > 0.62 and h > 0.72:
                seg_ids[i] = 1  # SEZ_Gnathal (Dfd/Scr)
            elif b > 0.48 and h > 0.60:
                seg_ids[i] = 2  # T1_Prothorax (Antp + high Hb)
            elif k >= 0.35 and b >= 0.34:
                seg_ids[i] = 3  # T2_Mesothorax (Antp + anterior Kr where Hb represses Ubx)
            elif k >= 0.20 and b >= 0.22:
                seg_ids[i] = 4  # T3_Metathorax (Ubx + posterior Kr)
            elif kn >= 0.35 and c < 0.78:
                seg_ids[i] = 5  # A1_A4_Abdomen anterior (abd-A + anterior Kni)
            elif kn >= 0.25 and tl < 0.50:
                seg_ids[i] = 6  # A1_A4_Abdomen posterior (abd-A + posterior Kni)
            else:
                seg_ids[i] = 7  # A5_A8_Abdomen (Abd-B + Tll)

        return SyncytialEmbryoReadout(
            segment_ids=seg_ids,
            bicoid_protein=bcd,
            nanos_protein=nos,
            caudal_protein=cad,
            hunchback_protein=hb,
            kruppel_protein=kr,
            knirps_protein=kni,
            zelda_7_stripes=zelda_7,
            opa_14_stripes=opa_14,
            pole_cells_mask=pole_cells,
        )

    def run_french_flag_segmentation_nca(self, n_positions: int = 64, steps: int = 16) -> torch.Tensor:
        """Run syncytial blastoderm -> Gap/Pair-rule/Hox NCA and return the 8 Hox neuromere IDs [n_positions]."""
        readout = self.run_syncytial_embryogenesis_nca(n_positions=n_positions, gap_steps=steps)
        return readout.segment_ids


def verify_against_banc_meta(meta_feather_path: str | Path) -> dict[str, float | int]:
    """Read real BANC v888 metadata feather to extract empirical ground-truth targets for M1 tests."""
    import pyarrow.feather as feather

    df = feather.read_feather(
        str(meta_feather_path),
        columns=["hemilineage", "cell_type", "neurotransmitter_predicted"],
    )
    n_total = len(df)
    n_hemilineages = int(df["hemilineage"].nunique())
    n_cell_types = int(df["cell_type"].nunique())

    nt_series = df["neurotransmitter_predicted"].dropna()
    nt_counts = nt_series.value_counts(normalize=True)

    hl_df = df.dropna(subset=["hemilineage", "neurotransmitter_predicted"])
    hl_df_sec = hl_df[hl_df["hemilineage"] != "putative_primary"]
    purity = float(
        hl_df_sec.groupby("hemilineage")["neurotransmitter_predicted"]
        .agg(lambda s: s.value_counts().iloc[0] / len(s))
        .mean()
    )

    return {
        "n_total": n_total,
        "n_hemilineages": n_hemilineages,
        "n_cell_types": n_cell_types,
        "ach_fraction": float(nt_counts.get("acetylcholine", 0.0)),
        "glu_fraction": float(nt_counts.get("glutamate", 0.0)),
        "gaba_fraction": float(nt_counts.get("gaba", 0.0)),
        "da_fraction": float(nt_counts.get("dopamine", 0.0)),
        "secondary_hemilineage_nt_purity": purity,
    }
