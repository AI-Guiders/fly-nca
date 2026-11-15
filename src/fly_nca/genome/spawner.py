"""Genetic Fly Spawner & Multi-Fly 3D Social Arena (X:A -> Sxl -> Tra -> Dsx/Fru on the 189,766-Cell Pan-Connectome).

In *Drosophila melanogaster*, sex is determined cell-autonomously NOT by a Y chromosome (unlike mammals),
but by the X-chromosome dose / X-to-autosome (`X:A`) ratio activating the `Sxl -> tra -> dsx / fru`
alternative splicing cascade (Bridges 1921; Cline & Meyer; Front Mol Neurosci 2023 PMC10622783):
- `X:A = 1.0` (2X:2A, including XXY) -> `Sxl ON` -> `Tra ON` -> `DsxF` + `Fru-` (Female);
- `X:A = 0.5` (1X:2A, including X0)  -> `Sxl OFF` -> `Tra OFF` -> `DsxM` + `FruM+` (Male);
- `Dsx` (`DsxF` vs `DsxM`) controls sex-specific cell survival (`P1` survives only under `DsxM`; `oviDN/vpoEN` under `DsxF`);
- `FruM` controls dendritic arborization (`DA1 -> aSP-f` vs `aSP-g`, Kohl et al. 2013) and VNC song circuits (`TN1/pIP10`),
  so `DsxF + FruM+` females (Demir & Dickson 2005) lack `P1` neurons yet still court and sing!
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import TYPE_CHECKING, Any, Literal
import torch

from fly_nca.body.adapter import MotorOut, SensorIn
from fly_nca.body.arena_3d import FastTensorArena3D, OdorPlume3D
from fly_nca.connectome.pan_connectome import FullConnectomeFlyNCA, FullConnectomeReadout
from fly_nca.genome.fly_genome import FlyGenome

if TYPE_CHECKING:
    from fly_nca.composition import FlyFrameTelemetry, FlyNCAComposition

SocialBehaviorLabel = Literal[
    "EGG_SYNCYTIUM_BLASTODERM",
    "LARVA_WAVE1_CRAWLING",
    "PUPA_20E_METAMORPHOSIS",
    "FORAGING_ROVER",
    "FORAGING_SITTER",
    "FEEDING_PER",
    "GROOMING_FRONT_LEGS",
    "MALE_COURTSHIP_SONG",
    "FEMALE_FRUM_COURTSHIP_SONG",
    "MALE_BOXING_ESCALATION",
    "FEMALE_SONG_RECEPTIVITY",
    "FEMALE_OVIPOSITION",
    "ESCAPE_JUMP_GF",
    "SLEEP",
]

LifeStage = Literal["egg", "larva", "pupa", "adult"]


@dataclass(frozen=True, slots=True)
class IndividualFlyGenome:
    """Valid individual Drosophila genome with X:A -> Sxl -> Tra -> (DsxF/DsxM, Fru-/FruM) cascade & allelic variation."""

    fly_id: str
    sex: Literal["F", "M"]
    x_to_autosome_ratio: float          # 1.0 (2X:2A female) or 0.5 (1X:2A male, e.g. X0 or XY)
    sxl_active: bool                    # Sex-lethal splicing switch (ON when X:A >= 1.0)
    tra_active: bool                    # transformer splicing switch (ON when Sxl is ON)
    dsx_isoform: Literal["DsxF", "DsxM"]  # doublesex isoform (DsxF under Tra ON, DsxM under Tra OFF)
    fru_m_active: bool                  # fruitless male isoform FruM (ON when Tra OFF, or in fru^M allele)
    foraging_allele: Literal["rover", "sitter"]
    foraging_locomotion_gain: float
    dopr1_learning_scale: float
    oaa2r_brake_scale: float
    shakb_gap_scale: float
    snpfr_hunger_scale: float
    courtship_vigor_scale: float
    initial_sugar_level: float
    initial_amino_level: float
    generation: int = 0
    maternal_id: str | None = None
    paternal_id: str | None = None

    def is_valid(self) -> bool:
        """Verify that X:A -> Sxl -> Tra -> Dsx/Fru cascade and allelic parameters are physiologically valid."""
        expected_sxl = self.x_to_autosome_ratio >= 0.95
        expected_dsx = "DsxF" if self.tra_active else "DsxM"
        return (
            self.sex in ("F", "M")
            and self.x_to_autosome_ratio in (0.5, 1.0)
            and self.sxl_active == expected_sxl
            and self.tra_active == self.sxl_active
            and self.dsx_isoform == expected_dsx
            and self.foraging_allele in ("rover", "sitter")
            and 0.65 <= self.foraging_locomotion_gain <= 1.40
            and 0.65 <= self.dopr1_learning_scale <= 1.40
            and 0.50 <= self.oaa2r_brake_scale <= 1.50
            and 0.70 <= self.shakb_gap_scale <= 1.30
            and 0.65 <= self.snpfr_hunger_scale <= 1.40
            and 0.65 <= self.courtship_vigor_scale <= 1.40
            and 0.05 <= self.initial_sugar_level <= 1.0
            and 0.05 <= self.initial_amino_level <= 1.0
        )


@dataclass
class SpawnedFly:
    """A living individual fly across its ontogenetic lifecycle (egg -> larva -> pupa -> 189,766-cell adult)."""

    genome: IndividualFlyGenome
    full_connectome: FullConnectomeFlyNCA
    composition: FlyNCAComposition
    arena: FastTensorArena3D
    life_stage: LifeStage = "adult"
    stage_progress: float = 1.0           # [0..1] progress computed directly from the living genomic/morphogenetic NCA!
    larval_mass_mg: float = 1.20          # Critical weight ~1.05 mg triggers neuroendocrine 20E pupariation pulse!
    pupal_hapf: float = 48.0              # Morphogenetic Wave 2 + CAM tract assembly progress (6 -> 48 h.a.p.f.)
    ecdysone_20e_titer: float = 0.0       # Steroid hormone 20-hydroxyecdysone titer [0..1]
    embryo_bcd: torch.Tensor | None = None
    embryo_nos: torch.Tensor | None = None
    embryo_hb: torch.Tensor | None = None
    embryo_kr: torch.Tensor | None = None
    embryo_kni: torch.Tensor | None = None
    embryo_steps: int = 0
    pupa_tract_w: torch.Tensor | None = None
    is_mated: bool = False
    mate_genome: IndividualFlyGenome | None = None
    eggs_laid: int = 0
    last_connectome_readout: FullConnectomeReadout | None = None
    last_telemetry: FlyFrameTelemetry | None = None
    current_behavior: SocialBehaviorLabel = "FORAGING_ROVER"
    wing_song_active: bool = False
    boxing_intensity: float = 0.0
    oviposition_active: bool = False
    receptivity_score: float = 0.0

    @property
    def pos_xyz(self) -> tuple[float, float, float]:
        return (
            float(self.arena.pos_xyz[0].item()),
            float(self.arena.pos_xyz[1].item()),
            float(self.arena.pos_xyz[2].item()),
        )


class FlySpawner:
    """Spawner that generates valid individual male & female Drosophila genomes and instantiates 189,766-cell flies."""

    def __init__(self, seed: int = 42, device: torch.device | str | None = None) -> None:
        self.rng = random.Random(seed)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.spawn_counter: int = 0

    def sample_valid_genome(
        self,
        sex: Literal["F", "M"] | None = None,
        foraging_allele: Literal["rover", "sitter"] | None = None,
        *,
        fru_m_override: bool | None = None,
    ) -> IndividualFlyGenome:
        """Sample a valid individual genome with X:A -> Sxl -> Tra -> Dsx/Fru cascade and natural polygenic variation."""
        self.spawn_counter += 1
        chosen_sex: Literal["F", "M"] = sex if sex in ("F", "M") else ("F" if self.rng.random() < 0.5 else "M")
        # Drosophila sex determination is driven by X-chromosome dose / X:A ratio (1.0 -> Sxl ON, 0.5 -> Sxl OFF)
        x_to_a = 1.0 if chosen_sex == "F" else 0.5
        sxl_on = x_to_a >= 0.95
        tra_on = sxl_on
        dsx_iso: Literal["DsxF", "DsxM"] = "DsxF" if tra_on else "DsxM"
        fru_m_on = fru_m_override if fru_m_override is not None else (not tra_on)

        chosen_for: Literal["rover", "sitter"] = (
            foraging_allele
            if foraging_allele in ("rover", "sitter")
            else ("rover" if self.rng.random() < 0.55 else "sitter")
        )
        loco_gain = (
            self.rng.uniform(1.08, 1.24) if chosen_for == "rover" else self.rng.uniform(0.78, 0.94)
        )
        genome = IndividualFlyGenome(
            fly_id=f"fly_{self.spawn_counter:03d}_{chosen_sex}_{chosen_for}",
            sex=chosen_sex,
            x_to_autosome_ratio=x_to_a,
            sxl_active=sxl_on,
            tra_active=tra_on,
            dsx_isoform=dsx_iso,
            fru_m_active=fru_m_on,
            foraging_allele=chosen_for,
            foraging_locomotion_gain=round(loco_gain, 4),
            dopr1_learning_scale=round(self.rng.uniform(0.88, 1.18), 4),
            oaa2r_brake_scale=round(self.rng.uniform(0.72, 1.28), 4),
            shakb_gap_scale=round(self.rng.uniform(0.92, 1.10), 4),
            snpfr_hunger_scale=round(self.rng.uniform(0.88, 1.18), 4),
            courtship_vigor_scale=round(self.rng.uniform(0.88, 1.22), 4),
            initial_sugar_level=round(self.rng.uniform(0.30, 0.75), 4),
            initial_amino_level=round(self.rng.uniform(0.30, 0.75), 4),
        )
        assert genome.is_valid()
        return genome

    def recombine_genomes(
        self,
        maternal: IndividualFlyGenome,
        paternal: IndividualFlyGenome,
        *,
        sex: Literal["F", "M"] | None = None,
    ) -> IndividualFlyGenome:
        """Mendelian + polygenic recombination of maternal and paternal Drosophila genomes (F1..Fn generation)."""
        self.spawn_counter += 1
        chosen_sex: Literal["F", "M"] = sex if sex in ("F", "M") else ("F" if self.rng.random() < 0.5 else "M")
        x_to_a = 1.0 if chosen_sex == "F" else 0.5
        sxl_on = x_to_a >= 0.95
        tra_on = sxl_on
        dsx_iso: Literal["DsxF", "DsxM"] = "DsxF" if tra_on else "DsxM"
        fru_m_on = not tra_on

        # Mendelian inheritance at foraging locus (oskar/foraging: rover vs sitter)
        chosen_for: Literal["rover", "sitter"] = (
            maternal.foraging_allele if self.rng.random() < 0.5 else paternal.foraging_allele
        )
        gen_num = max(maternal.generation, paternal.generation) + 1

        def _blend(a: float, b: float, lo: float, hi: float) -> float:
            w = self.rng.uniform(0.25, 0.75)
            mut = self.rng.gauss(0.0, 0.025)
            return round(max(lo, min(hi, w * a + (1.0 - w) * b + mut)), 4)

        loco_base = _blend(maternal.foraging_locomotion_gain, paternal.foraging_locomotion_gain, 0.70, 1.35)
        if chosen_for == "rover":
            loco_base = max(1.05, loco_base)
        else:
            loco_base = min(0.96, loco_base)

        genome = IndividualFlyGenome(
            fly_id=f"fly_{self.spawn_counter:03d}_G{gen_num}_{chosen_sex}_{chosen_for}",
            sex=chosen_sex,
            x_to_autosome_ratio=x_to_a,
            sxl_active=sxl_on,
            tra_active=tra_on,
            dsx_isoform=dsx_iso,
            fru_m_active=fru_m_on,
            foraging_allele=chosen_for,
            foraging_locomotion_gain=round(loco_base, 4),
            dopr1_learning_scale=_blend(maternal.dopr1_learning_scale, paternal.dopr1_learning_scale, 0.70, 1.35),
            oaa2r_brake_scale=_blend(maternal.oaa2r_brake_scale, paternal.oaa2r_brake_scale, 0.55, 1.45),
            shakb_gap_scale=_blend(maternal.shakb_gap_scale, paternal.shakb_gap_scale, 0.75, 1.25),
            snpfr_hunger_scale=_blend(maternal.snpfr_hunger_scale, paternal.snpfr_hunger_scale, 0.70, 1.35),
            courtship_vigor_scale=_blend(maternal.courtship_vigor_scale, paternal.courtship_vigor_scale, 0.70, 1.35),
            initial_sugar_level=0.65,
            initial_amino_level=0.65,
            generation=gen_num,
            maternal_id=maternal.fly_id,
            paternal_id=paternal.fly_id,
        )
        assert genome.is_valid()
        return genome

    def spawn_fly(
        self,
        sex: Literal["F", "M"] | None = None,
        foraging_allele: Literal["rover", "sitter"] | None = None,
        pos_xyz: tuple[float, float, float] | None = None,
        yaw_rad: float | None = None,
        *,
        fru_m_override: bool | None = None,
        life_stage: LifeStage = "adult",
        genome_override: IndividualFlyGenome | None = None,
    ) -> SpawnedFly:
        """Spawn one complete living fly (at stage 'egg', 'larva', 'pupa', or 'adult') from a valid genome."""
        from fly_nca.composition import FlyNCAComposition

        ind_genome = genome_override or self.sample_valid_genome(
            sex=sex,
            foraging_allele=foraging_allele,
            fru_m_override=fru_m_override,
        )
        full_conn = FullConnectomeFlyNCA(
            sex=ind_genome.sex,
            dsx_isoform=ind_genome.dsx_isoform,
            fru_m_active=ind_genome.fru_m_active,
            foraging_Write_gain=ind_genome.foraging_locomotion_gain,
            dopr1_learning_scale=ind_genome.dopr1_learning_scale,
            oaa2r_brake_scale=ind_genome.oaa2r_brake_scale,
            shakb_gap_scale=ind_genome.shakb_gap_scale,
            snpfr_hunger_scale=ind_genome.snpfr_hunger_scale,
            courtship_vigor_scale=ind_genome.courtship_vigor_scale,
            device=self.device,
        )
        comp = FlyNCAComposition(
            genome=FlyGenome(),
            stage="larva" if life_stage in ("egg", "larva") else "adult",
            sex=ind_genome.sex,
            device=self.device,
        )
        comp.hemolymph.sugar_level = ind_genome.initial_sugar_level
        comp.hemolymph.amino_acid_level = ind_genome.initial_amino_level
        comp.mushroom_body.eta_learn = comp.mushroom_body.eta_learn * ind_genome.dopr1_learning_scale

        arena = FastTensorArena3D(device=self.device)
        if pos_xyz is not None:
            arena.pos_xyz = torch.tensor(pos_xyz, device=self.device, dtype=torch.float32)
        else:
            arena.pos_xyz = torch.tensor(
                [self.rng.uniform(-0.008, 0.008), self.rng.uniform(-0.008, 0.008), 0.0],
                device=self.device,
                dtype=torch.float32,
            )
        init_yaw = yaw_rad if yaw_rad is not None else self.rng.uniform(-math.pi, math.pi)
        arena.thorax_rpy[2] = float(init_yaw)
        comp.central_complex.seed_heading(float(init_yaw))

        if life_stage == "egg":
            beh: SocialBehaviorLabel = "EGG_SYNCYTIUM_BLASTODERM"
            prog, mass, hapf = 0.0, 0.15, 0.0
        elif life_stage == "larva":
            beh = "LARVA_WAVE1_CRAWLING"
            prog, mass, hapf = 0.0, 0.45, 0.0
        elif life_stage == "pupa":
            beh = "PUPA_20E_METAMORPHOSIS"
            prog, mass, hapf = 0.0, 1.20, 6.0
        else:
            beh = "FORAGING_ROVER" if ind_genome.foraging_allele == "rover" else "FORAGING_SITTER"
            prog, mass, hapf = 1.0, 1.20, 48.0

        full_conn.set_ontogeny_stage(life_stage, prog)

        return SpawnedFly(
            genome=ind_genome,
            full_connectome=full_conn,
            composition=comp,
            arena=arena,
            life_stage=life_stage,
            stage_progress=prog,
            larval_mass_mg=mass,
            pupal_hapf=hapf,
            current_behavior=beh,
        )

    def spawn_population(
        self,
        n_flies: int = 6,
        *,
        sex_sequence: list[Literal["F", "M"]] | None = None,
    ) -> list[SpawnedFly]:
        """Spawn a population of N flies (mix of X:A=1.0 DsxF/Fru- females and X:A=0.5 DsxM/FruM+ males)."""
        population: list[SpawnedFly] = []
        for i in range(n_flies):
            s: Literal["F", "M"] | None = (
                sex_sequence[i] if (sex_sequence and i < len(sex_sequence)) else ("F" if i % 2 == 0 else "M")
            )
            angle = 2.0 * math.pi * i / max(1, n_flies)
            radius = 0.004  # 4 mm social circle around central food cup
            px = radius * math.cos(angle)
            py = radius * math.sin(angle)
            population.append(self.spawn_fly(sex=s, pos_xyz=(px, py, 0.0), yaw_rad=angle + math.pi))
        return population


@dataclass
class MultiFlySocialArena3D:
    """Shared 3D arena hosting a multi-generational Drosophila colony (egg -> larva -> pupa -> adult).

    External Environment Contract:
    The arena ONLY advances time (`dt = 4.038 ms`) and supplies physical environmental fields
    (`temperature_c`, `wind_world_xyz`, `surface_texture_freq`, `food_yeast_radius_m`, odor plumes,
    acoustic vibrations, and 3D physical contact). Every behavior, courtship song, receptivity,
    and uterus motor contraction (`eff_oviposition_uterus`) is computed purely inside each fly's
    autonomous 189,766-cell connectome!
    """

    flies: list[SpawnedFly]
    shared_plumes: list[OdorPlume3D] = field(default_factory=list)
    wind_world_xyz: tuple[float, float, float] = (0.12, 0.0, 0.0)
    ambient_temperature_c: float = 25.0
    agar_texture_freq: float = 8.0
    food_yeast_radius_m: float = 0.008
    time_ms: float = 0.0
    spawner: FlySpawner | None = None
    max_colony_size: int = 24
    colony_events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.shared_plumes:
            # Central sucrose + yeast food cup at (0, 0, 0) as in Sci Adv 2026 & Open Biol 2019
            self.shared_plumes.append(
                OdorPlume3D(
                    pos_xyz=(0.0, 0.0, 0.0),
                    glomerulus_id=0,
                    intensity=0.95,
                    sigma_m=0.025,
                    is_sweet_food=True,
                    is_yeast_protein=True,
                )
            )
        if self.spawner is None and self.flies:
            self.spawner = FlySpawner(seed=108, device=self.flies[0].arena.pos_xyz.device)

    def step_population(self, frames: int = 1, *, enable_ontogeny_and_reproduction: bool = False) -> list[dict[str, Any]]:
        """Advance all organisms by `frames` cognitive frames (4.038 ms each) with social & ontogenetic coupling."""
        from fly_nca.genome.morphogenesis import develop_fly_from_genome

        snapshots: list[dict[str, Any]] = []
        for _ in range(max(1, int(frames))):
            self.time_ms += 4.038
            snapshots = []
            new_offspring: list[SpawnedFly] = []

            for i, f_i in enumerate(self.flies):
                f_i.arena.time_ms = self.time_ms
                f_i.arena.plumes = list(self.shared_plumes)
                f_i.arena.wind_world_xyz = self.wind_world_xyz
                f_i.arena.ambient_temperature_c = self.ambient_temperature_c
                f_i.arena.surface_texture_freq = self.agar_texture_freq
                xi, yi, zi = f_i.pos_xyz
                dist_to_cup = math.hypot(xi, yi)
                on_yeast_cup = dist_to_cup < self.food_yeast_radius_m

                # --- Ontogenetic Stages driven purely by the Genomic & Morphogenetic Cellular Automaton ---
                if f_i.life_stage == "egg":
                    # Step the Syncytial Blastoderm -> Gap/Pair-Rule/Hox Genomic Automaton directly inside the egg!
                    n_pos = 64
                    if f_i.embryo_bcd is None or f_i.embryo_nos is None or f_i.embryo_hb is None or f_i.embryo_kr is None or f_i.embryo_kni is None:
                        f_i.embryo_bcd = torch.zeros(n_pos, dtype=torch.float32)
                        f_i.embryo_nos = torch.zeros(n_pos, dtype=torch.float32)
                        f_i.embryo_hb = torch.full((n_pos,), 0.5, dtype=torch.float32)
                        f_i.embryo_kr = torch.zeros(n_pos, dtype=torch.float32)
                        f_i.embryo_kni = torch.zeros(n_pos, dtype=torch.float32)
                        f_i.embryo_steps = 0

                    bcd = f_i.embryo_bcd
                    nos = f_i.embryo_nos
                    hb = f_i.embryo_hb
                    kr = f_i.embryo_kr
                    kni = f_i.embryo_kni
                    decay = 0.00045
                    # Local syncytial neighbor diffusion steps (`roll(±1)`) per tick
                    for _ in range(400):
                        bcd[0] = 1.0
                        nos[-1] = 1.0
                        bcd_l = torch.cat([bcd[:1], bcd[:-1]])
                        bcd_r = torch.cat([bcd[1:], bcd[-1:]])
                        nos_l = torch.cat([nos[:1], nos[:-1]])
                        nos_r = torch.cat([nos[1:], nos[-1:]])
                        bcd = (1.0 - decay) * (0.50 * bcd + 0.25 * (bcd_l + bcd_r))
                        nos = (1.0 - decay) * (0.50 * nos + 0.25 * (nos_l + nos_r))
                    bcd[0] = 1.0
                    nos[-1] = 1.0
                    cad = (0.5 * (2.0 / (1.0 + 4.5 * bcd))).clamp(0.0, 1.0)
                    if f_i.embryo_steps == 0:
                        hb = (0.5 * (2.0 / (1.0 + 6.5 * nos))).clamp(0.0, 1.0)
                    tll = torch.zeros(n_pos, dtype=torch.float32)
                    tll[-4:] = 1.0
                    tll[:2] = 0.7
                    for _ in range(15):
                        hb_drive = torch.sigmoid(4.5 * (0.85 * bcd + 0.45 * hb - 0.35 - 0.50 * kr - 0.85 * nos))
                        kr_drive = torch.relu(1.0 - 2.2 * (hb - 0.48).abs()) * torch.sigmoid(5.0 * (0.55 - kni - tll))
                        kni_drive = torch.sigmoid(7.0 * (cad - 1.25 * hb - 0.75 * kr - 1.2 * tll - 0.12))
                        hb = (0.70 * hb + 0.30 * hb_drive).clamp(0.0, 1.0)
                        kr = (0.70 * kr + 0.30 * kr_drive).clamp(0.0, 1.0)
                        kni = (0.70 * kni + 0.30 * kni_drive).clamp(0.0, 1.0)

                    f_i.embryo_bcd, f_i.embryo_nos, f_i.embryo_hb, f_i.embryo_kr, f_i.embryo_kni = bcd, nos, hb, kr, kni
                    f_i.embryo_steps += 1

                    # Evaluate Hox neuromere commitment (0..7) directly from the genomic automaton state!
                    committed_segs: set[int] = set()
                    for idx_n in range(n_pos):
                        b_v, c_v, h_v, k_v, kn_v, tl_v = (
                            float(bcd[idx_n]),
                            float(cad[idx_n]),
                            float(hb[idx_n]),
                            float(kr[idx_n]),
                            float(kni[idx_n]),
                            float(tll[idx_n]),
                        )
                        if b_v > 0.78 and h_v > 0.80:
                            committed_segs.add(0)
                        elif b_v > 0.62 and h_v > 0.72:
                            committed_segs.add(1)
                        elif b_v > 0.48 and h_v > 0.60:
                            committed_segs.add(2)
                        elif k_v >= 0.35 and b_v >= 0.34:
                            committed_segs.add(3)
                        elif k_v >= 0.20 and b_v >= 0.22:
                            committed_segs.add(4)
                        elif kn_v >= 0.35 and c_v < 0.78:
                            committed_segs.add(5)
                        elif kn_v >= 0.25 and tl_v < 0.50:
                            committed_segs.add(6)
                        else:
                            committed_segs.add(7)

                    # Egg hatches when all 8 Hox neuromeres (0..7) have self-organized and Wave 1 neurogenesis completes!
                    f_i.stage_progress = min(1.0, (len(committed_segs) / 8.0) * min(1.0, f_i.embryo_steps / 2.0))
                    f_i.full_connectome.set_ontogeny_stage("egg", f_i.stage_progress)
                    f_i.current_behavior = "EGG_SYNCYTIUM_BLASTODERM"
                    n_egg_cells = int(f_i.full_connectome.sex_cell_mask.sum().item())
                    if len(committed_segs) == 8 and f_i.embryo_steps >= 2:
                        _ = f_i.composition.genome.run_syncytial_embryogenesis_nca(n_positions=64)
                        f_i.life_stage = "larva"
                        f_i.stage_progress = 0.0
                        f_i.larval_mass_mg = 0.35
                        f_i.full_connectome.set_ontogeny_stage("larva", 0.0)
                        f_i.current_behavior = "LARVA_WAVE1_CRAWLING"
                        self.colony_events.append(
                            {"time_ms": round(self.time_ms, 2), "event": "HATCH_EGG_TO_LARVA", "fly_id": f_i.genome.fly_id}
                        )
                    snapshots.append(self._build_snapshot(f_i, n_active=n_egg_cells))
                    continue

                if f_i.life_stage == "larva":
                    # Crawling Wave 1 larva running purely on its 3,961 empirical BANC primary cells (`wave1_primary`):
                    # Zero scripted `dx, dy = -xi, -yi` attraction! Movement & pharyngeal ingestion come strictly from the Wave 1 connectome!
                    f_i.full_connectome.set_ontogeny_stage("larva", f_i.stage_progress)
                    sensor_l = f_i.arena.step_physics(MotorOut.zeros(device=f_i.arena.pos_xyz.device), dt_ms=0.0)
                    conn_l = f_i.full_connectome.step_full_connectome(
                        sensor_l,
                        substrate_yeast_rich=on_yeast_cup,
                        hunger_level=0.8,
                        n_synaptic_hops=4,
                    )
                    larva_motor = f_i.full_connectome.decode_motor_effectors()
                    larva_motor.leg_joint_drive *= 0.15  # Larval peristaltic crawling speed (~1.5 mm/s)
                    larva_motor.wing_drive_lr.zero_()
                    larva_motor.jump_ttmn_burst = 0.0
                    _ = f_i.arena.step_physics(larva_motor, dt_ms=4.038)
                    # Metabolic assimilation driven by the larva's own pharyngeal/proboscis motor neurons & IPC neurosecretory cells!
                    ipc_rate = f_i.full_connectome.group_mean_rate("eff_endocrine_ipc_akh")
                    feed_gain = (0.30 if dist_to_cup < 0.010 else 0.15) * max(0.25, float(larva_motor.proboscis_per) + ipc_rate + 0.5)
                    f_i.larval_mass_mg = round(f_i.larval_mass_mg + feed_gain, 4)
                    f_i.stage_progress = min(1.0, f_i.larval_mass_mg / 1.05)
                    f_i.current_behavior = "LARVA_WAVE1_CRAWLING"
                    if f_i.larval_mass_mg >= 1.05:
                        # Critical weight triggers prothoracic gland 20-hydroxyecdysone (20E) steroid pulse -> pupariation!
                        f_i.ecdysone_20e_titer = 1.0
                        f_i.life_stage = "pupa"
                        f_i.pupal_hapf = 6.0
                        f_i.stage_progress = 0.125
                        f_i.full_connectome.set_ontogeny_stage("pupa", f_i.stage_progress)
                        f_i.current_behavior = "PUPA_20E_METAMORPHOSIS"
                        self.colony_events.append(
                            {
                                "time_ms": round(self.time_ms, 2),
                                "event": "PUPARIATION_20E_PULSE",
                                "fly_id": f_i.genome.fly_id,
                                "larval_mass_mg": f_i.larval_mass_mg,
                            }
                        )
                    snapshots.append(self._build_snapshot(f_i, n_active=conn_l.n_active_cells, conn_out=conn_l))
                    continue

                if f_i.life_stage == "pupa":
                    # Meng et al. (2026) Nature pupal stages: Wave 1 pruning + Wave 2 (dati+) secondary neurogenesis
                    # & local CAM (DIP/Dpr + Beat/Side) lock-and-key tract assembly (`K_axon @ M_cam @ Q_dend^T`)!
                    dev = f_i.composition.developed
                    if f_i.pupa_tract_w is None:
                        f_i.pupa_tract_w = dev.organ_tract_matrix.clone() * 0.25
                    cam_eq = torch.relu(dev.k_org_keys @ dev.m_cam_affinity @ dev.q_org_locks.t())
                    cam_eq = cam_eq / cam_eq.sum(dim=0, keepdim=True).clamp(min=1e-6)
                    for _ in range(8):
                        f_i.pupa_tract_w = 0.65 * f_i.pupa_tract_w + 0.35 * cam_eq
                        f_i.pupa_tract_w = f_i.pupa_tract_w / f_i.pupa_tract_w.sum(dim=0, keepdim=True).clamp(min=1e-6)
                    tract_err = float((f_i.pupa_tract_w - cam_eq).abs().max().item())
                    f_i.pupal_hapf = min(48.0, f_i.pupal_hapf + 21.0 * (1.0 - min(0.8, tract_err)))
                    f_i.stage_progress = min(1.0, f_i.pupal_hapf / 48.0)
                    f_i.ecdysone_20e_titer = round(max(0.1, 1.0 - 0.5 * f_i.stage_progress), 3)
                    f_i.full_connectome.set_ontogeny_stage("pupa", f_i.stage_progress)
                    f_i.current_behavior = "PUPA_20E_METAMORPHOSIS"
                    n_pupa_cells = int(f_i.full_connectome.sex_cell_mask.sum().item())
                    if f_i.pupal_hapf >= 48.0 and tract_err < 0.05:
                        f_i.composition.developed = develop_fly_from_genome(
                            f_i.composition.genome,
                            stage="adult",
                            sex=f_i.genome.sex,
                            scale=0.02,
                            device=f_i.arena.pos_xyz.device,
                        )
                        f_i.life_stage = "adult"
                        f_i.stage_progress = 1.0
                        f_i.ecdysone_20e_titer = 0.0
                        f_i.full_connectome.set_ontogeny_stage("adult", 1.0)
                        f_i.current_behavior = (
                            "FORAGING_ROVER" if f_i.genome.foraging_allele == "rover" else "FORAGING_SITTER"
                        )
                        self.colony_events.append(
                            {
                                "time_ms": round(self.time_ms, 2),
                                "event": "ADULT_ECLOSION_189766_CELLS",
                                "fly_id": f_i.genome.fly_id,
                                "sex": f_i.genome.sex,
                                "generation": f_i.genome.generation,
                            }
                        )
                    snapshots.append(self._build_snapshot(f_i, n_active=n_pupa_cells))
                    continue

                # --- Adult Imago (189,766 cells, 6,853,423 5-port synaptic edges) ---
                nearest_dist = 999.0
                nearest_j: SpawnedFly | None = None
                for j, f_j in enumerate(self.flies):
                    if i == j or f_j.life_stage != "adult":
                        continue
                    xj, yj, zj = f_j.pos_xyz
                    d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_j = f_j

                cva_conc = 0.0
                mate_conc = 0.0
                neighbor_is_female = True
                song_heard = 0.0

                if nearest_j is not None and nearest_dist < 0.030:
                    f_i.arena.small_target_pos_xyz = nearest_j.pos_xyz
                    prox = math.exp(-(nearest_dist**2) / (2.0 * 0.010**2))
                    neighbor_is_female = nearest_j.genome.sex == "F"
                    if nearest_j.genome.sex == "M":
                        cva_conc = 0.85 * prox  # Male cuticular pheromone 11-cis-vaccenyl acetate (cVA)
                        if nearest_j.wing_song_active:
                            song_heard = 0.90 * prox  # 160 Hz sine / 35 ms IPI pulse courtship song
                    else:
                        mate_conc = 0.85 * prox  # Female cuticular hydrocarbon pheromone (7,11-HD -> VA1v & ppk23 F-cells)
                        cva_conc = 0.0  # Virgin females lack male anti-aphrodisiac cVA
                else:
                    f_i.arena.small_target_pos_xyz = None

                # Step 6 NCA organs (hemolymph & telemetry without overwriting the 189,766-cell connectome's body adapter)
                telem = f_i.composition.step_cognitive_frame(f_i.arena, step_body=False)
                f_i.last_telemetry = telem

                # Construct full SensorIn with conspecific acoustic song vibration & tactile contact for the 189,766-cell connectome
                sensor = f_i.arena.step_physics(MotorOut.zeros(device=f_i.arena.pos_xyz.device), dt_ms=0.0)
                sensor.sound_vibration = song_heard
                if nearest_j is not None and nearest_dist < 0.030:
                    sensor.small_target_salience = max(float(sensor.small_target_salience), 0.75)
                if nearest_j is not None and nearest_dist < 0.0038:
                    # Physical cuticle-to-cuticle contact (< 3.8 mm) excites front/middle leg tactile bristles!
                    touch_mag = min(1.0, (0.0038 - nearest_dist) / 0.0018)
                    dx_j = nearest_j.pos_xyz[0] - xi
                    dy_j = nearest_j.pos_xyz[1] - yi
                    yaw_i = float(f_i.arena.thorax_rpy[2].item())
                    rel_lat = -math.sin(yaw_i) * dx_j + math.cos(yaw_i) * dy_j
                    if rel_lat >= 0.0:
                        sensor.bristle_touch[0] = max(float(sensor.bristle_touch[0]), touch_mag)
                        sensor.bristle_touch[2] = max(float(sensor.bristle_touch[2]), 0.7 * touch_mag)
                    else:
                        sensor.bristle_touch[1] = max(float(sensor.bristle_touch[1]), touch_mag)
                        sensor.bristle_touch[3] = max(float(sensor.bristle_touch[3]), 0.7 * touch_mag)

                hunger = max(0.0, 1.0 - f_i.composition.hemolymph.sugar_level)
                # Physical sex-peptide molecule on uterus SPSN receptors is present ONLY if physical copulation occurred!
                sex_peptide_conc = 0.85 if f_i.is_mated else 0.0

                conn_out = f_i.full_connectome.step_full_connectome(
                    sensor,
                    cva_pheromone_conc=cva_conc,
                    mate_odor_conc=mate_conc,
                    conspecific_is_female=neighbor_is_female,
                    substrate_yeast_rich=on_yeast_cup,
                    hunger_level=hunger,
                    sex_peptide_uterus=sex_peptide_conc,
                )
                f_i.last_connectome_readout = conn_out
                pan_motor = f_i.full_connectome.decode_motor_effectors()
                _ = f_i.arena.step_physics(pan_motor, dt_ms=4.038)

                # Read out motor & circuit states directly from the fly's own 189,766-cell connectome!
                f_i.wing_song_active = float(pan_motor.wing_song_mode[0].item()) > 0.02
                f_i.oviposition_active = conn_out.female_ovi_oviposition_rate > 0.12
                f_i.receptivity_score = round(conn_out.female_pc1_receptivity_rate, 4)
                f_i.boxing_intensity = 0.0

                if f_i.genome.fru_m_active and nearest_j is not None and not neighbor_is_female and nearest_dist < 0.015:
                    vpm4_drive = (0.75 + 0.25 * cva_conc) / max(0.4, f_i.genome.oaa2r_brake_scale)
                    mb_agg = f_i.composition.mushroom_body.step_mb(
                        telem.chem.pn_glomeruli_50,
                        vpm4_octopamine_drive=min(1.0, vpm4_drive),
                    )
                    f_i.boxing_intensity = round(float(mb_agg.aggression_boxing_drive), 4)
                    f_i.current_behavior = (
                        "MALE_BOXING_ESCALATION" if not f_i.wing_song_active else "MALE_COURTSHIP_SONG"
                    )
                if f_i.wing_song_active:
                    f_i.current_behavior = (
                        "MALE_COURTSHIP_SONG"
                        if f_i.genome.dsx_isoform == "DsxM"
                        else "FEMALE_FRUM_COURTSHIP_SONG"
                    )
                    telem.motor.wing_song_mode[0] = float(conn_out.male_tn1_wing_song_rate)
                elif f_i.boxing_intensity > 0.0:
                    f_i.current_behavior = "MALE_BOXING_ESCALATION"
                elif f_i.oviposition_active:
                    f_i.current_behavior = "FEMALE_OVIPOSITION"
                elif conn_out.female_pc1_receptivity_rate > 0.15:
                    f_i.current_behavior = "FEMALE_SONG_RECEPTIVITY"
                elif conn_out.dn_cluster_rates.get("grooming", 0.0) > 0.06:
                    f_i.current_behavior = "GROOMING_FRONT_LEGS"
                elif pan_motor.proboscis_per > 0.25 or telem.motor.proboscis_per > 0.5:
                    f_i.current_behavior = "FEEDING_PER"
                else:
                    f_i.current_behavior = (
                        "FORAGING_ROVER" if f_i.genome.foraging_allele == "rover" else "FORAGING_SITTER"
                    )

                # Physical transfer of seminal sex-peptide occurs when a courting male and receptive female are in physical contact
                if (
                    not f_i.genome.fru_m_active
                    and not f_i.is_mated
                    and conn_out.female_pc1_receptivity_rate > 0.15
                    and nearest_j is not None
                    and nearest_j.genome.sex == "M"
                    and (nearest_j.wing_song_active or (nearest_j.last_connectome_readout is not None and nearest_j.last_connectome_readout.male_pip10_song_rate > 0.15))
                    and nearest_dist < 0.0095
                ):
                    f_i.is_mated = True
                    f_i.mate_genome = nearest_j.genome
                    self.colony_events.append(
                        {
                            "time_ms": round(self.time_ms, 2),
                            "event": "COPULATION_SEX_PEPTIDE_TRANSFERRED",
                            "female_id": f_i.genome.fly_id,
                            "male_id": nearest_j.genome.fly_id,
                            "pc1_receptivity": f_i.receptivity_score,
                        }
                    )

                # Physical egg extrusion occurs purely when the fly's own 6 uterus motor neurons (CMU / EFFabg95f) contract!
                if (
                    enable_ontogeny_and_reproduction
                    and f_i.oviposition_active
                    and f_i.mate_genome is not None
                    and f_i.eggs_laid == 0
                    and (len(self.flies) + len(new_offspring) < self.max_colony_size)
                    and self.spawner is not None
                ):
                    pat_genome = f_i.mate_genome
                    child_genome = self.spawner.recombine_genomes(f_i.genome, pat_genome)
                    egg_pos = (
                        round(xi + self.spawner.rng.uniform(-0.0015, 0.0015), 5),
                        round(yi + self.spawner.rng.uniform(-0.0015, 0.0015), 5),
                        0.0,
                    )
                    egg_fly = self.spawner.spawn_fly(
                        pos_xyz=egg_pos,
                        life_stage="egg",
                        genome_override=child_genome,
                    )
                    new_offspring.append(egg_fly)
                    f_i.eggs_laid += 1
                    self.colony_events.append(
                        {
                            "time_ms": round(self.time_ms, 2),
                            "event": "OVIPOSITION_EGG_LAID",
                            "mother_id": f_i.genome.fly_id,
                            "father_id": pat_genome.fly_id,
                            "egg_id": child_genome.fly_id,
                            "sex": child_genome.sex,
                            "generation": child_genome.generation,
                        }
                    )

                snapshots.append(self._build_snapshot(f_i, n_active=conn_out.n_active_cells, conn_out=conn_out))

            if new_offspring:
                self.flies.extend(new_offspring)

        return snapshots

    @staticmethod
    def _build_snapshot(
        f_i: SpawnedFly,
        *,
        n_active: int,
        conn_out: FullConnectomeReadout | None = None,
    ) -> dict[str, Any]:
        return {
            "fly_id": f_i.genome.fly_id,
            "generation": f_i.genome.generation,
            "life_stage": f_i.life_stage,
            "stage_progress": round(f_i.stage_progress, 3),
            "larval_mass_mg": round(f_i.larval_mass_mg, 3),
            "pupal_hapf": round(f_i.pupal_hapf, 1),
            "sex": f_i.genome.sex,
            "x_to_a": f_i.genome.x_to_autosome_ratio,
            "sxl_tra": f_i.genome.tra_active,
            "dsx": f_i.genome.dsx_isoform,
            "fru_m": f_i.genome.fru_m_active,
            "foraging_allele": f_i.genome.foraging_allele,
            "pos_xyz": [round(c, 5) for c in f_i.pos_xyz],
            "yaw_rad": round(float(f_i.arena.thorax_rpy[2].item()), 4),
            "leg_angles": [round(float(x), 3) for x in f_i.arena.leg_angles[:, 0].tolist()],
            "organ_rates": conn_out.organ_mean_rates if conn_out is not None else {},
            "behavior": f_i.current_behavior,
            "n_active_cells_189k": n_active,
            "wing_song_active": f_i.wing_song_active,
            "boxing_intensity": f_i.boxing_intensity,
            "receptivity_score": f_i.receptivity_score,
            "oviposition_active": f_i.oviposition_active,
            "male_p1_rate": conn_out.male_p1_courtship_rate if conn_out is not None else 0.0,
            "male_pip10_rate": conn_out.male_pip10_song_rate if conn_out is not None else 0.0,
            "female_pc1_rate": conn_out.female_pc1_receptivity_rate if conn_out is not None else 0.0,
            "female_ovi_rate": conn_out.female_ovi_oviposition_rate if conn_out is not None else 0.0,
        }
