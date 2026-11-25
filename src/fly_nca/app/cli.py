"""CLI for the Fly-NCA Configurable Laboratory Bench (`list`, `run <scenario>`, `panel <scenario>`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from fly_nca.app.scenario import SCENARIOS


def _device_default() -> str:
    return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"


def _require_torch() -> None:
    if torch is None:
        print(
            "Error: PyTorch (`torch`) is not installed in the active Python environment.\n"
            "On a fresh clone, create and activate a virtual environment first:\n"
            "  python3 -m venv .venv\n"
            "  source .venv/bin/activate   # Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1)


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
    _require_torch()
    from fly_nca.app.lab import execute_scenario_protocol

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    payload = execute_scenario_protocol(args.scenario, device=args.device)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.json or not args.json_out:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def cmd_panel(args: argparse.Namespace) -> int:
    _require_torch()
    from fly_nca.app.lab import open_lab

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
    _require_torch()
    from fly_nca.genome.spawner import FlySpawner, MultiFlySocialArena3D

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    spawner = FlySpawner(seed=args.seed, device=args.device)
    flies = spawner.spawn_population(n_flies=args.count)
    social_arena = MultiFlySocialArena3D(flies=flies)
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


def cmd_colony3d(args: argparse.Namespace) -> int:
    from fly_nca.app.viewer3d import export_colony_3d_viewer_html
    from fly_nca.body.mujoco_flygym import serve_live_mujoco_colony

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if getattr(args, "live", False):
        serve_live_mujoco_colony(
            output_html_path=args.out,
            n_founders=args.founders,
            seed=args.seed,
            port=args.port,
            device=args.device,
        )
        return 0
    res = export_colony_3d_viewer_html(
        output_path=args.out,
        n_founders=args.founders,
        frames=args.frames,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
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

    spawn_p = sub.add_parser("spawn", help="Spawn a population of X:A->Sxl->Tra->Dsx/Fru 189,766-cell flies in 3D arena")
    spawn_p.add_argument("--count", type=int, default=4)
    spawn_p.add_argument("--steps", type=int, default=4)
    spawn_p.add_argument("--seed", type=int, default=42)
    spawn_p.add_argument("--device", default=_device_default())
    spawn_p.add_argument("--json", action="store_true")
    spawn_p.add_argument("--json-out", type=str, default=None)
    spawn_p.set_defaults(func=cmd_spawn)

    col_p = sub.add_parser("colony3d", help="Run multi-generational 3D colony (Egg->Larva->Pupa->Adult) & export WebGL 3D viewer")
    col_p.add_argument("--founders", type=int, default=4)
    col_p.add_argument("--frames", type=int, default=16)
    col_p.add_argument("--seed", type=int, default=108)
    col_p.add_argument("--out", type=str, default="colony_3d_viewer.html")
    col_p.add_argument("--live", action="store_true", help="Run unbounded t=0 -> infinity real-time SSE simulation stream")
    col_p.add_argument("--port", type=int, default=8765)
    col_p.add_argument("--device", default=_device_default())
    col_p.set_defaults(func=cmd_colony3d)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
