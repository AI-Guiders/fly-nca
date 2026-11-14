"""Genetic Fly Spawner & Multi-Fly 4D Social Arena (`XX` Females + `XY` Males on the 189,766-Cell Pan-Connectome).

Generates valid individual *Drosophila melanogaster* genomes with natural polygenic variation
(`for^R` rover vs `for^s` sitter foraging polymorphism, `DopR1` learning sensitivity, `OAα2R/mGluR`
presynaptic aggression brake, `shak-B` electrical coupling, `sNPFR` hunger gain, and `FruM / DsxF`
courtship/receptivity vigor), instantiates each fly with the full 189,766-cell pan-connectome
(`BANC v888` + `MCNS v1.0`), and simulates multi-fly social & foraging interactions in 4D spacetime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import TYPE_CHECKING, Any, Literal
import torch

from fly_nca.body.adapter import MotorOut, SensorIn
from fly_nca.body.arena_4d import FastTensorArena4D, OdorPlume3D
from fly_nca.connectome.pan_connectome import FullConnectomeFlyNCA, FullConnectomeReadout
from fly_nca.genome.fly_genome import FlyGenome

if TYPE_CHECKING:
    from fly_nca.composition import FlyFrameTelemetry, FlyNCAComposition

SocialBehaviorLabel = Literal[
    "FORAGING_ROVER",
    "FORAGING_SITTER",
    "FEEDING_PER",
    "MALE_COURTSHIP_SONG",
    "MALE_BOXING_ESCALATION",
    "FEMALE_SONG_RECEPTIVITY",
    "FEMALE_OVIPOSITION",
    "ESCAPE_JUMP_GF",
    "SLEEP",
]


@dataclass(frozen=True, slots=True)
class IndividualFlyGenome:
    """Valid individual Drosophila genome with chromosomal sex (XX/XY) and natural allelic variation."""

    fly_id: str
    sex: Literal["F", "M"]
    foraging_allele: Literal["rover", "sitter"]
    foraging_locomotion_gain: float
    dopr1_learning_scale: float
    oaa2r_brake_scale: float
    shakb_gap_scale: float
    snpfr_hunger_scale: float
    courtship_vigor_scale: float
    initial_sugar_level: float
    initial_amino_level: float

    def is_valid(self) -> bool:
        """Verify that all allelic parameters lie within viable physiological bounds."""
        return (
            self.sex in ("F", "M")
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
    """A living individual fly combining its IndividualFlyGenome, 189,766-cell Pan-Connectome, and 4D Body."""

    genome: IndividualFlyGenome
    full_connectome: FullConnectomeFlyNCA
    composition: FlyNCAComposition
    arena: FastTensorArena4D
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

    def __init__(self, seed: int = 42, device: torch.device | str = "cpu") -> None:
        self.rng = random.Random(seed)
        self.device = torch.device(device)
        self.spawn_counter: int = 0

    def sample_valid_genome(
        self,
        sex: Literal["F", "M"] | None = None,
        foraging_allele: Literal["rover", "sitter"] | None = None,
    ) -> IndividualFlyGenome:
        """Sample a valid individual genome with natural polygenic variation."""
        self.spawn_counter += 1
        chosen_sex: Literal["F", "M"] = sex if sex in ("F", "M") else ("F" if self.rng.random() < 0.5 else "M")
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

    def spawn_fly(
        self,
        sex: Literal["F", "M"] | None = None,
        foraging_allele: Literal["rover", "sitter"] | None = None,
        pos_xyz: tuple[float, float, float] | None = None,
        yaw_rad: float | None = None,
    ) -> SpawnedFly:
        """Spawn one complete 189,766-cell living fly from a valid genome."""
        from fly_nca.composition import FlyNCAComposition

        ind_genome = self.sample_valid_genome(sex=sex, foraging_allele=foraging_allele)
        full_conn = FullConnectomeFlyNCA(
            sex=ind_genome.sex,
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
            sex=ind_genome.sex,
            device=self.device,
        )
        comp.hemolymph.sugar_level = ind_genome.initial_sugar_level
        comp.hemolymph.amino_acid_level = ind_genome.initial_amino_level
        comp.mushroom_body.eta_learn = comp.mushroom_body.eta_learn * ind_genome.dopr1_learning_scale

        arena = FastTensorArena4D(device=self.device)
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

        return SpawnedFly(
            genome=ind_genome,
            full_connectome=full_conn,
            composition=comp,
            arena=arena,
            current_behavior="FORAGING_ROVER" if ind_genome.foraging_allele == "rover" else "FORAGING_SITTER",
        )

    def spawn_population(
        self,
        n_flies: int = 6,
        *,
        sex_sequence: list[Literal["F", "M"]] | None = None,
    ) -> list[SpawnedFly]:
        """Spawn a population of N flies (mix of XX females and XY males with distinct valid genomes)."""
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
class MultiFlySocialArena4D:
    """Shared 4D spacetime arena hosting multiple spawned male & female 189,766-cell flies."""

    flies: list[SpawnedFly]
    shared_plumes: list[OdorPlume3D] = field(default_factory=list)
    wind_world_xyz: tuple[float, float, float] = (0.12, 0.0, 0.0)
    time_ms: float = 0.0

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

    def step_population(self, frames: int = 1) -> list[dict[str, Any]]:
        """Advance all spawned flies by `frames` cognitive frames (4.038 ms each) with social & sensory coupling."""
        snapshots: list[dict[str, Any]] = []
        for _ in range(max(1, int(frames))):
            self.time_ms += 4.038
            snapshots = []
            n = len(self.flies)

            for i, f_i in enumerate(self.flies):
                f_i.arena.plumes = list(self.shared_plumes)
                f_i.arena.wind_world_xyz = self.wind_world_xyz

                # Find closest conspecific neighbor j != i in 3D space
                nearest_dist = 999.0
                nearest_j: SpawnedFly | None = None
                xi, yi, zi = f_i.pos_xyz
                for j, f_j in enumerate(self.flies):
                    if i == j:
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
                        mate_conc = 0.85 * prox  # Female cuticular hydrocarbon pheromone (7,11-HD -> VA1v)
                        cva_conc = 0.25 * prox
                else:
                    f_i.arena.small_target_pos_xyz = None

                # Step 4D body + 6 NCA organs
                telem = f_i.composition.step_cognitive_frame(f_i.arena)
                f_i.last_telemetry = telem

                # Construct full SensorIn with conspecific acoustic song vibration for the 189,766-cell connectome
                sensor = f_i.arena.step_physics(MotorOut.zeros(device=f_i.arena.pos_xyz.device), dt_ms=0.0)
                sensor.sound_vibration = song_heard
                if nearest_j is not None and nearest_dist < 0.030:
                    sensor.small_target_salience = max(float(sensor.small_target_salience), 0.75)

                hunger = max(0.0, 1.0 - f_i.composition.hemolymph.sugar_level)
                dist_to_cup = math.hypot(xi, yi)
                on_yeast_cup = dist_to_cup < 0.008

                conn_out = f_i.full_connectome.step_full_connectome(
                    sensor,
                    cva_pheromone_conc=cva_conc,
                    mate_odor_conc=mate_conc,
                    conspecific_is_female=neighbor_is_female,
                    substrate_yeast_rich=on_yeast_cup,
                    hunger_level=hunger,
                )
                f_i.last_connectome_readout = conn_out

                # Coupling between 189,766-cell dimorphic circuits and behavioral state
                f_i.wing_song_active = False
                f_i.boxing_intensity = 0.0
                f_i.oviposition_active = False
                f_i.receptivity_score = 0.0

                if f_i.genome.sex == "M":
                    if conn_out.male_pip10_song_rate > 0.15 and neighbor_is_female:
                        f_i.wing_song_active = True
                        f_i.current_behavior = "MALE_COURTSHIP_SONG"
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
                    # Female (XX)
                    if conn_out.female_pc1_receptivity_rate > 0.15 and song_heard > 0.1:
                        f_i.receptivity_score = round(conn_out.female_pc1_receptivity_rate, 4)
                        f_i.current_behavior = "FEMALE_SONG_RECEPTIVITY"
                    elif conn_out.female_ovi_oviposition_rate > 0.15 and on_yeast_cup:
                        f_i.oviposition_active = True
                        f_i.current_behavior = "FEMALE_OVIPOSITION"
                    elif telem.motor.proboscis_per > 0.5:
                        f_i.current_behavior = "FEEDING_PER"
                    else:
                        f_i.current_behavior = (
                            "FORAGING_ROVER" if f_i.genome.foraging_allele == "rover" else "FORAGING_SITTER"
                        )

                snapshots.append(
                    {
                        "fly_id": f_i.genome.fly_id,
                        "sex": f_i.genome.sex,
                        "foraging_allele": f_i.genome.foraging_allele,
                        "pos_xyz": [round(c, 5) for c in f_i.pos_xyz],
                        "behavior": f_i.current_behavior,
                        "n_active_cells_189k": conn_out.n_active_cells,
                        "wing_song_active": f_i.wing_song_active,
                        "boxing_intensity": f_i.boxing_intensity,
                        "receptivity_score": f_i.receptivity_score,
                        "oviposition_active": f_i.oviposition_active,
                        "male_p1_rate": conn_out.male_p1_courtship_rate,
                        "male_pip10_rate": conn_out.male_pip10_song_rate,
                        "female_pc1_rate": conn_out.female_pc1_receptivity_rate,
                        "female_ovi_rate": conn_out.female_ovi_oviposition_rate,
                    }
                )
        return snapshots
