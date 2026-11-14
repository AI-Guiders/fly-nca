"""Full Closed-Loop Composition of NCA-Fly: Genome -> 6 Topological NCA Organs + Glia + 4D Body Arena."""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from fly_nca.body import FastTensorArena4D, MotorOut, SensorIn
from fly_nca.core import GlialSubstrateNCA, HemolymphState
from fly_nca.genome import FlyGenome, develop_fly_from_genome
from fly_nca.organs import (
    AntennalSEZNCA,
    CentralComplexNCA,
    ChemosensoryDigest,
    CXNavigationReadout,
    DescendingJoystickDN,
    LHSpinGateNCA,
    LHSpinOutput,
    MBStepOutput,
    MushroomBodyNCA,
    OpticLobeNCA,
    VisualProjectionDigest,
    VNCAutopilotNCA,
)


@dataclass(frozen=True, slots=True)
class FlyFrameTelemetry:
    """Complete telemetry of 1 cognitive frame (~4 ms = 4 fast 1-ms VNC autopilot ticks)."""

    pos_xyz: tuple[float, float, float]
    time_ms: float
    vis: VisualProjectionDigest
    chem: ChemosensoryDigest
    mb: MBStepOutput
    lh_spin: LHSpinOutput
    cx: CXNavigationReadout
    motor: MotorOut


class FlyNCAComposition(nn.Module):
    """Unified compositional NCA-Fly running at dual physical time quanta (1 ms VNC / 4 ms OL+CB)."""

    def __init__(
        self,
        genome: FlyGenome | None = None,
        sex: str = "F",
        frazzled_state: str = "wt",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.genome = genome or FlyGenome()
        self.developed = develop_fly_from_genome(
            self.genome,
            stage="adult",
            sex="M" if sex == "M" else "F",
            scale=0.05,
            frazzled_state=frazzled_state,
            device=self.device,
        )

        # Instantiate the 6 natural topological NCA organs + Glia + Hemolymph
        self.glia = GlialSubstrateNCA(n_compartments=15, device=self.device, dtype=dtype)
        self.hemolymph = HemolymphState()
        self.optic_lobe = OpticLobeNCA(device=self.device, dtype=dtype)
        self.antennal_sez = AntennalSEZNCA(device=self.device, dtype=dtype)
        self.lh_spin = LHSpinGateNCA(device=self.device, dtype=dtype)
        self.mushroom_body = MushroomBodyNCA(device=self.device, dtype=dtype)
        self.central_complex = CentralComplexNCA(device=self.device, dtype=dtype)
        self.vnc_autopilot = VNCAutopilotNCA(frazzled_state=frazzled_state, device=self.device, dtype=dtype)

        self.last_an_efference = torch.zeros(5, device=self.device, dtype=dtype)

    def step_cognitive_frame(
        self,
        arena: FastTensorArena4D,
        target_altitude_z: float = 0.0,
        sleep_need_override: float | None = None,
        creb_intact: bool = True,
        visual_darkness: bool = False,
        gtacr1_inhibit_fb_goal: bool = False,
    ) -> FlyFrameTelemetry:
        """Execute 1 cognitive frame (4.0 ms = 1 wingbeat) composed of 1 CB/OL update + 4 x 1.0 ms VNC ticks."""
        sensor: SensorIn = arena.step_physics(MotorOut.zeros(device=self.device, dtype=self.dtype), dt_ms=0.0)

        # 1. Autonomous Optic Lobe Radar (OL-NCA, 892 hex columns -> AOTU/BU + VPN)
        vis = self.optic_lobe.step_optic_lobe(sensor, dt_ms=4.0)

        # 2. Antennal Lobe (50 glomeruli divisive normalization) & SEZ Gustatory/Feeding NCA
        chem = self.antennal_sez.step_chemosensory(sensor, self.hemolymph)

        # 3. Mushroom Body (Calyx APL sparse KC -> 15 compartments + DN->DAN efference copy weighted by CAM tract)
        hunger = max(0.0, 1.0 - self.hemolymph.sugar_level)
        w_al_to_mb = self.developed.tract_weight("AL", "MB") / 0.3189
        w_vnc_to_mb = self.developed.tract_weight("VNC", "MB") / 0.0281
        mb_read = self.mushroom_body.step_mb(
            pn_glomeruli_50=chem.pn_glomeruli_50 * w_al_to_mb,
            sweet_reward_us=chem.sweet_reward_us,
            bitter_shock_us=chem.bitter_punish_us,
            dn_efference_copy=self.last_an_efference * w_vnc_to_mb,
            hunger_mp1_gate=hunger,
        )

        # 4. Lateral Horn & SPIN Multimodal Context Gate (GluCla AND-gate + soft priors)
        lh_out = self.lh_spin.step_lh_spin(
            pn_glomeruli_50=chem.pn_glomeruli_50,
            sensor=sensor,
            hemolymph=self.hemolymph,
            mbon_learned_valence=mb_read.net_learned_valence,
        )

        # 5. Central Complex Navigator & dFB Sleep Switch (weighted by CAM-assembled LH_SPIN -> CX tract!)
        w_lh_to_cx = self.developed.tract_weight("LH_SPIN", "CX") / 0.1192
        sleep_need = (
            float(sleep_need_override)
            if sleep_need_override is not None
            else float(self.glia.alg_sleep_need.mean().item())
        )
        cx_out = self.central_complex.step_cx(
            sensor=sensor,
            vis=vis,
            valence_gate_signal=lh_out.net_valence_to_cx_fbt * w_lh_to_cx,
            target_altitude_z=target_altitude_z,
            current_z=float(arena.pos_xyz[2].item()),
            sleep_need_alg=sleep_need,
            visual_darkness=visual_darkness,
            gtacr1_inhibit_fb_goal=gtacr1_inhibit_fb_goal,
            dt_ms=4.0,
        )

        # If in Active Sleep (REM-like phase of dFB), consolidate MB gamma traces into alpha/beta via CREB!
        if cx_out.sleep_phase == "ACTIVE_REM":
            self.mushroom_body.consolidate_sleep_creb(creb_intact=creb_intact, transfer_rate=0.35)

        # Combine CX steering with SEZ feeding and LC4 looming -> GF escape on the descending neck bundle
        dn_cmd = DescendingJoystickDN(
            forward_walk_dnp09=cx_out.dn_command.forward_walk_dnp09,
            backward_walk_mdn=cx_out.dn_command.backward_walk_mdn,
            turn_left_dna02=cx_out.dn_command.turn_left_dna02,
            turn_right_dna02=cx_out.dn_command.turn_right_dna02,
            flight_power_dng02=cx_out.dn_command.flight_power_dng02,
            flight_lift_dn=cx_out.dn_command.flight_lift_dn,
            gf_looming_spike=(1.2 if vis.lc4_lplc2_looming > 0.75 else 0.0),
            freeze_dn=cx_out.dn_command.freeze_dn,
            proboscis_fdg=chem.fdg_proboscis_cmd,
        )

        # 6. Run 4 fast ticks (4 x 1.0 ms = 4.0 ms) of VNC 200-Hz Autopilot & 4D Arena physics
        motor = MotorOut.zeros(device=self.device, dtype=self.dtype)
        for _ in range(4):
            motor, self.last_an_efference = self.vnc_autopilot.step_autopilot(sensor, dn=dn_cmd, dt_ms=1.0)
            sensor = arena.step_physics(motor, dt_ms=1.0)

        return FlyFrameTelemetry(
            pos_xyz=(
                float(arena.pos_xyz[0].item()),
                float(arena.pos_xyz[1].item()),
                float(arena.pos_xyz[2].item()),
            ),
            time_ms=arena.time_ms,
            vis=vis,
            chem=chem,
            mb=mb_read,
            lh_spin=lh_out,
            cx=cx_out,
            motor=motor,
        )
