"""CLI for the Fly-NCA Configurable Laboratory Bench (`list`, `run <scenario>`, `panel <scenario>`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from fly_nca.app.lab import execute_scenario_protocol, open_lab
from fly_nca.app.scenario import SCENARIOS
from fly_nca.genome.spawner import FlySpawner, MultiFlySocialArena4D


def _device_default() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_list(_: argparse.Namespace) -> int:
    print(f"{'scenario':46}  {'laboratory':32}  {'paper'}")
    print("-" * 124)
    for sid, spec in SCENARIOS.items():
        print(f"{sid:46}  {spec.laboratory.value:32}  {spec.paper_citation[:44]}")
    print("\nRun protocol:  python -m fly_nca.app.cli run <scenario>")
    print("Sample panel:  python -m fly_nca.app.cli panel <scenario>")
    print("Verify suites: python verify_principles.py [--list | --ship | --scenario <id>]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    payload = execute_scenario_protocol(args.scenario, device=args.device)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json or not args.json_out:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    lab = open_lab(args.scenario, device=args.device)
    lab.step(args.steps)
    payload = {"scenario": lab.meta, "panel": lab.panel()}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json or not args.json_out:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_spawn(args: argparse.Namespace) -> int:
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    spawner = FlySpawner(seed=args.seed, device=args.device)
    flies = spawner.spawn_population(n_flies=args.count)
    social_arena = MultiFlySocialArena4D(flies=flies)
    snapshots = social_arena.step_population(frames=args.steps)
    payload = {
        "n_flies": len(flies),
        "pan_connectome_cells_per_fly": flies[0].full_connectome.substrate.n_total,
        "pan_connectome_edges_per_fly": flies[0].full_connectome.substrate.n_edges,
        "genomes": [
            {
                "fly_id": f.genome.fly_id,
                "sex": f.genome.sex,
                "foraging_allele": f.genome.foraging_allele,
                "locomotion_gain": f.genome.foraging_locomotion_gain,
                "dopr1_learning": f.genome.dopr1_learning_scale,
                "oaa2r_brake": f.genome.oaa2r_brake_scale,
                "courtship_vigor": f.genome.courtship_vigor_scale,
            }
            for f in flies
        ],
        "final_frame": snapshots,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json or not args.json_out:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fly-NCA Configurable Wet-Lab Bench")
    sub = p.add_subparsers(dest="command", required=True)
    ids = list(SCENARIOS.keys())

    sub.add_parser("list", help="List all wet-lab experimental scenarios and apparatuses").set_defaults(func=cmd_list)

    run_p = sub.add_parser("run", help="Run exact wet-lab protocol from the original paper")
    run_p.add_argument("scenario", choices=ids)
    run_p.add_argument("--device", default=_device_default())
    run_p.add_argument("--json", action="store_true")
    run_p.add_argument("--json-out", type=str, default=None)
    run_p.set_defaults(func=cmd_run)

    panel_p = sub.add_parser("panel", help="Sample all 34 laboratory instruments on a scenario")
    panel_p.add_argument("scenario", choices=ids)
    panel_p.add_argument("--steps", type=int, default=4)
    panel_p.add_argument("--device", default=_device_default())
    panel_p.add_argument("--json", action="store_true")
    panel_p.add_argument("--json-out", type=str, default=None)
    panel_p.set_defaults(func=cmd_panel)

    spawn_p = sub.add_parser("spawn", help="Spawn a population of XX female & XY male 189,766-cell flies in 4D arena")
    spawn_p.add_argument("--count", type=int, default=4)
    spawn_p.add_argument("--steps", type=int, default=4)
    spawn_p.add_argument("--seed", type=int, default=42)
    spawn_p.add_argument("--device", default=_device_default())
    spawn_p.add_argument("--json", action="store_true")
    spawn_p.add_argument("--json-out", type=str, default=None)
    spawn_p.set_defaults(func=cmd_spawn)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
