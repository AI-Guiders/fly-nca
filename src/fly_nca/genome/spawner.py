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
    stage_progress: float = 1.0           # [0..1] progress through current developmental stage
    larval_mass_mg: float = 1.20          # Critical weight ~1.0 mg triggers 20E ecdysone pupariation pulse!
    pupal_hapf: float = 48.0              # Hours after puparium formation (0 -> 6 -> 24 -> 36 -> 48 h.a.p.f.)
    ecdysone_20e_titer: float = 0.0       # Steroid hormone 20-hydroxyecdysone titer [0..1]
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
    """Shared 3D arena hosting a multi-generational Drosophila colony (egg -> larva -> pupa -> adult)."""

    flies: list[SpawnedFly]
    shared_plumes: list[OdorPlume3D] = field(default_factory=list)
    wind_world_xyz: tuple[float, float, float] = (0.12, 0.0, 0.0)
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
                f_i.arena.plumes = list(self.shared_plumes)
                f_i.arena.wind_world_xyz = self.wind_world_xyz
                xi, yi, zi = f_i.pos_xyz
                dist_to_cup = math.hypot(xi, yi)
                on_yeast_cup = dist_to_cup < 0.008

                # --- Ontogenetic Stages L0-L1: EGG -> LARVA (Wave 1) -> PUPA (20E Wave 2) -> ADULT ---
                if f_i.life_stage == "egg":
                    f_i.stage_progress = min(1.0, f_i.stage_progress + 0.35)
                    f_i.current_behavior = "EGG_SYNCYTIUM_BLASTODERM"
                    if f_i.stage_progress >= 1.0:
                        _ = f_i.composition.genome.run_syncytial_embryogenesis_nca(n_positions=64)
                        f_i.life_stage = "larva"
                        f_i.stage_progress = 0.0
                        f_i.larval_mass_mg = 0.35
                        f_i.current_behavior = "LARVA_WAVE1_CRAWLING"
                        self.colony_events.append(
                            {"time_ms": round(self.time_ms, 2), "event": "HATCH_EGG_TO_LARVA", "fly_id": f_i.genome.fly_id}
                        )
                    snapshots.append(self._build_snapshot(f_i, n_active=64))
                    continue

                if f_i.life_stage == "larva":
                    # Crawling Wave 1 larva (~3,016 primary cells, zero neck bottleneck): chemotaxis toward yeast cup
                    dx, dy = -xi, -yi
                    d_norm = math.hypot(dx, dy) + 1e-6
                    step_m = 0.0004 * f_i.genome.foraging_locomotion_gain
                    if d_norm > 0.002:
                        f_i.arena.pos_xyz[0] += step_m * (dx / d_norm)
                        f_i.arena.pos_xyz[1] += step_m * (dy / d_norm)
                    feed_gain = 0.30 if dist_to_cup < 0.010 else 0.15
                    f_i.larval_mass_mg = round(f_i.larval_mass_mg + feed_gain, 4)
                    f_i.stage_progress = min(1.0, f_i.larval_mass_mg / 1.05)
                    f_i.current_behavior = "LARVA_WAVE1_CRAWLING"
                    if f_i.larval_mass_mg >= 1.05:
                        # Critical weight triggers prothoracic gland 20-hydroxyecdysone (20E) steroid pulse -> pupariation!
                        f_i.ecdysone_20e_titer = 1.0
                        f_i.life_stage = "pupa"
                        f_i.pupal_hapf = 6.0
                        f_i.stage_progress = 0.125
                        f_i.current_behavior = "PUPA_20E_METAMORPHOSIS"
                        self.colony_events.append(
                            {
                                "time_ms": round(self.time_ms, 2),
                                "event": "PUPARIATION_20E_PULSE",
                                "fly_id": f_i.genome.fly_id,
                                "larval_mass_mg": f_i.larval_mass_mg,
                            }
                        )
                    snapshots.append(self._build_snapshot(f_i, n_active=f_i.composition.developed.n_wave1_primary))
                    continue

                if f_i.life_stage == "pupa":
                    # Meng et al. (2026) Nature pupal stages: 6 -> 24 -> 36 -> 48 h.a.p.f.
                    # Wave 1 pruning + Wave 2 (dati+) secondary neurogenesis & CAM (DIP/Dpr + Beat/Side) tract wiring
                    f_i.pupal_hapf = min(48.0, f_i.pupal_hapf + 14.0)
                    f_i.stage_progress = min(1.0, f_i.pupal_hapf / 48.0)
                    f_i.ecdysone_20e_titer = round(max(0.1, 1.0 - 0.5 * f_i.stage_progress), 3)
                    f_i.current_behavior = "PUPA_20E_METAMORPHOSIS"
                    if f_i.pupal_hapf >= 48.0:
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
                    snapshots.append(self._build_snapshot(f_i, n_active=f_i.composition.developed.n_cells))
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

                # Step 3D body + 6 NCA organs
                telem = f_i.composition.step_cognitive_frame(f_i.arena)
                f_i.last_telemetry = telem

                # Construct full SensorIn with conspecific acoustic song vibration for the 189,766-cell connectome
                sensor = f_i.arena.step_physics(MotorOut.zeros(device=f_i.arena.pos_xyz.device), dt_ms=0.0)
                sensor.sound_vibration = song_heard
                if nearest_j is not None and nearest_dist < 0.030:
                    sensor.small_target_salience = max(float(sensor.small_target_salience), 0.75)

                hunger = max(0.0, 1.0 - f_i.composition.hemolymph.sugar_level)

                conn_out = f_i.full_connectome.step_full_connectome(
                    sensor,
                    cva_pheromone_conc=cva_conc,
                    mate_odor_conc=mate_conc,
                    conspecific_is_female=neighbor_is_female,
                    substrate_yeast_rich=on_yeast_cup,
                    hunger_level=hunger,
                )
                f_i.last_connectome_readout = conn_out
                pan_motor = f_i.full_connectome.decode_motor_effectors()
                _ = f_i.arena.step_physics(pan_motor, dt_ms=0.5)

                # Coupling between 189,766-cell dimorphic circuits and behavioral state
                f_i.wing_song_active = False
                f_i.boxing_intensity = 0.0
                f_i.oviposition_active = False
                f_i.receptivity_score = 0.0

                if f_i.genome.fru_m_active:
                    if conn_out.male_pip10_song_rate > 0.15 and neighbor_is_female:
                        f_i.wing_song_active = True
                        f_i.current_behavior = (
                            "MALE_COURTSHIP_SONG"
                            if f_i.genome.dsx_isoform == "DsxM"
                            else "FEMALE_FRUM_COURTSHIP_SONG"
                        )
                        telem.motor.wing_song_mode[0] = float(conn_out.male_tn1_wing_song_rate)
                    elif nearest_j is not None and not neighbor_is_female and nearest_dist < 0.015:
                        # Male-male encounter on food patch -> VPM4 OA+Glu boxing escalation
                        vpm4_drive = (0.75 + 0.25 * cva_conc) / max(0.4, f_i.genome.oaa2r_brake_scale)
                        mb_agg = f_i.composition.mushroom_body.step_mb(
                            telem.chem.pn_glomeruli_50,
                            vpm4_octopamine_drive=min(1.0, vpm4_drive),
                        )
                        f_i.boxing_intensity = round(float(mb_agg.aggression_boxing_drive), 4)
                        f_i.current_behavior = "MALE_BOXING_ESCALATION"
                    elif telem.motor.proboscis_per > 0.5:
                        f_i.current_behavior = "FEEDING_PER"
                    else:
                        f_i.current_behavior = (
                            "FORAGING_ROVER" if f_i.genome.foraging_allele == "rover" else "FORAGING_SITTER"
                        )
                else:
                    # Wild-type Female (X:A=1.0 -> Sxl ON -> Tra ON -> DsxF, Fru-)
                    if conn_out.female_pc1_receptivity_rate > 0.15 and song_heard > 0.1:
                        f_i.receptivity_score = round(conn_out.female_pc1_receptivity_rate, 4)
                        f_i.current_behavior = "FEMALE_SONG_RECEPTIVITY"
                        if nearest_j is not None and nearest_j.genome.sex == "M":
                            f_i.is_mated = True
                            f_i.mate_genome = nearest_j.genome
                    elif conn_out.female_ovi_oviposition_rate > 0.15 and on_yeast_cup:
                        f_i.oviposition_active = True
                        f_i.current_behavior = "FEMALE_OVIPOSITION"
                        if nearest_j is not None and nearest_j.genome.sex == "M":
                            f_i.is_mated = True
                            f_i.mate_genome = nearest_j.genome
                    elif telem.motor.proboscis_per > 0.5:
                        f_i.current_behavior = "FEEDING_PER"
                    else:
                        f_i.current_behavior = (
                            "FORAGING_ROVER" if f_i.genome.foraging_allele == "rover" else "FORAGING_SITTER"
                        )

                    # When a mated/courted female activates oviposition (or receptivity + oviDN on yeast cup), lay fertilized F1 egg!
                    if (
                        enable_ontogeny_and_reproduction
                        and f_i.eggs_laid == 0
                        and on_yeast_cup
                        and conn_out.female_ovi_oviposition_rate > 0.12
                        and (f_i.is_mated or (nearest_j is not None and nearest_j.genome.sex == "M"))
                        and (len(self.flies) + len(new_offspring) < self.max_colony_size)
                        and self.spawner is not None
                    ):
                        pat_genome = f_i.mate_genome or (
                            nearest_j.genome if nearest_j is not None else f_i.genome
                        )
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
