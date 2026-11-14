"""CLI for the Fly-NCA Configurable Laboratory Bench (`list`, `run <scenario>`, `panel <scenario>`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from fly_nca.app.lab import execute_scenario_protocol, open_lab
from fly_nca.app.scenario import SCENARIOS


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

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
