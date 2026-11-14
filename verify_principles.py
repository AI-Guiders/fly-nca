#!/usr/bin/env python3
"""Verify Fly-NCA principles and published wet-lab experimental benchmarks.

Mirrors `mt-lattice-fluid/verify_principles.py`:
- Check definitions & grouping: `verify_checks/suites.py` (SSOT)
- Laboratory scenarios & apparatuses: `fly_nca/app/scenario.py` & `fly_nca/app/laboratory.py`
- 34-channel Laboratory Instrument Panel: `fly_nca/instruments/catalog.py` & `panel.py`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO = Path(__file__).resolve().parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pyarrow.feather  # noqa: F401 - import before torch on Windows
import torch

from fly_nca.app.scenario import SCENARIOS, get_scenario
from verify_checks.suites import (
    SHIP_SUITE_IDS,
    SUITE_ORDER,
    VERIFY_SUITES,
    run_suites,
    verify_suite_ids_for_scenarios,
)


def cmd_list(_: argparse.Namespace) -> int:
    print(f"{'suite':22}  {'ship':4}  description")
    print("-" * 100)
    for sid in SUITE_ORDER:
        spec = VERIFY_SUITES[sid]
        ship = "yes" if spec.ship else "no"
        print(f"{sid:22}  {ship:4}  {spec.description}")
    print("\nWet-Lab Scenarios (same ids as `python -m fly_nca.app run <id>`):")
    print(f"{'scenario':46}  {'laboratory':32}  verify_suites")
    print("-" * 116)
    for sid, spec in SCENARIOS.items():
        suites = ",".join(spec.verify_suites)
        print(f"{sid:46}  {spec.laboratory.value:32}  {suites}")
    print("\nRun: python verify_principles.py --scenario pmc13518903_vt062617_wind_tunnel")
    print("     python verify_principles.py [--ship | --suite wetlab_experiments ...]")
    return 0


def _print_row(row: dict) -> None:
    status = "PASS" if row["ok"] else "FAIL"
    payload = {k: v for k, v in row.items() if k not in ("id", "ok", "suite")}
    print(f"{row['id']:46}  {status}  {payload}")


def _print_results(device: str, rows: list[dict], *, by_suite: bool) -> None:
    print(f"device={device}")
    print("-" * 100)
    if not by_suite:
        for row in rows:
            _print_row(row)
        return
    current: str | None = None
    for row in rows:
        sid = row.get("suite", "?")
        if sid != current:
            current = sid
            spec = VERIFY_SUITES.get(sid)
            desc = spec.description if spec else ""
            print(f"\n=== {sid} ===  {desc}")
        _print_row(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fly-NCA First-Principles & Wet-Lab Verification")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="List verify suites and wet-lab scenarios.")
    parser.add_argument(
        "--ship",
        action="store_true",
        help=f"Ship profile: {', '.join(SHIP_SUITE_IDS)}.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        dest="suites",
        metavar="ID",
        help="Run one suite; repeat for several. Default: all ship suites.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        metavar="ID",
        help="Run verify profile for a specific wet-lab scenario.",
    )
    args = parser.parse_args(argv)

    if args.list:
        return cmd_list(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if args.scenarios:
        scenario_ids = []
        for raw in args.scenarios:
            for part in raw.split(","):
                part = part.strip()
                if part:
                    get_scenario(part)
                    scenario_ids.append(part)
        suite_ids = list(verify_suite_ids_for_scenarios(scenario_ids))
    elif args.suites:
        suite_ids = []
        for raw in args.suites:
            for part in raw.split(","):
                part = part.strip()
                if part:
                    if part not in VERIFY_SUITES:
                        parser.error(f"Unknown suite {part!r}. Known: {list(SUITE_ORDER)}")
                    suite_ids.append(part)
    else:
        suite_ids = list(SHIP_SUITE_IDS)

    rows = run_suites(suite_ids, device=args.device)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        _print_results(args.device, rows, by_suite=len(suite_ids) > 1)

    failed = [r for r in rows if not r["ok"]]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
