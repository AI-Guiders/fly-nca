"""M6 Verification: Nervous Impulse Trace (synaptic wave observability across all 6 NCA organs & 5 ports)."""

from __future__ import annotations

from pathlib import Path

from fly_nca.body import FastTensorArena4D, OdorPlume3D
from fly_nca.composition import FlyNCAComposition
from fly_nca.nervous import (
    DebugLevel,
    ImpulseOutcome,
    ImpulseRecorder,
    format_impulse_replay,
    read_impulse_trace,
)


def test_nervous_impulse_trace_records_full_synaptic_wave_and_jsonl(tmp_path: Path) -> None:
    """Verify that ImpulseRecorder captures activate, suppress, score, threshold, select, and route events."""
    fly = FlyNCAComposition()
    arena = FastTensorArena4D(
        plumes=[OdorPlume3D(pos_xyz=(0.02, 0.0, 0.0), glomerulus_id=0, intensity=1.0, is_sweet_food=True)]
    )
    rec = ImpulseRecorder(level=DebugLevel.FULL, store_dir=tmp_path)

    # Frame 1: Normal navigation toward odor plume
    fly.step_cognitive_frame(arena, recorder=rec)

    # Frame 2: Predator looming spike triggers Giant Fiber shak-B electrical threshold -> ESCAPE_JUMP
    arena.looming_rate = 0.95
    fly.step_cognitive_frame(arena, recorder=rec)

    # Frame 3..34: High glial sleep need triggers dFB SLOW_WAVE -> ACTIVE_REM (at 120 ms) -> MB CREB consolidation
    arena.looming_rate = 0.0
    for _ in range(32):
        fly.step_cognitive_frame(arena, sleep_need_override=0.95, creb_intact=True, recorder=rec)

    records = rec.last_impulses()
    assert len(records) == 34

    # Verify JSONL persistence and filtering
    persisted = read_impulse_trace(tmp_path, limit=40)
    assert len(persisted) == 34

    # Frame 1 touches all 6 organs in topological order: OL -> AL_SEZ -> MB -> LH_SPIN -> CX -> VNC
    assert persisted[0]["organs_touched"] == ["OL", "AL_SEZ", "MB", "LH_SPIN", "CX", "VNC"]
    kinds_0 = [ev["kind"] for ev in persisted[0]["events"]]
    assert "activate" in kinds_0
    assert "suppress" in kinds_0  # APL 95.4% GABA sparse KC inhibition
    assert "score" in kinds_0     # MBON 15-compartment potentials
    assert "select" in kinds_0    # AL glomeruli, LH-SPIN valence, CX 16-ring wedges
    assert "route" in kinds_0     # VNC descending bundle routing

    # Frame 2 must record the LC4 -> GF electrical threshold and ESCAPE_JUMP outcome
    assert persisted[1]["outcome"] == ImpulseOutcome.ESCAPE_JUMP.value
    assert any(
        ev["kind"] == "threshold" and "Giant Fiber" in ev.get("reason", "")
        for ev in persisted[1]["events"]
    )

    # Sleep frames transition from QUIET_SLEEP (SLOW_WAVE) to ACTIVE_SLEEP_CREB (ACTIVE_REM)
    outcomes = [row["outcome"] for row in persisted[2:]]
    assert ImpulseOutcome.QUIET_SLEEP.value in outcomes
    assert ImpulseOutcome.ACTIVE_SLEEP_CREB.value in outcomes
    assert any(
        ev["kind"] == "threshold" and "CREB" in ev.get("reason", "")
        for row in persisted[2:]
        for ev in row["events"]
    )

    # Human-readable replay formatting includes organs, ports, suppressed KCs, and winners
    replay_text = format_impulse_replay(records)
    assert "Port2_GluCla_GABA_Shunt" in replay_text
    assert "Port4_ShakB_Electrical" in replay_text
    assert "ESCAPE_JUMP" in replay_text
    assert "ACTIVE_SLEEP_CREB" in replay_text
