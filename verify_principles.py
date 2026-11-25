#!/usr/bin/env python3
"""Verify Fly-NCA principles and published wet-lab experimental benchmarks.

Mirrors `mt-lattice-fluid/verify_principles.py`:
- Check definitions & grouping: `verify_checks/suites.py` (SSOT)
- Laboratory scenarios & apparatuses: `fly_nca/app/scenario.py` & `fly_nca/app/laboratory.py`
- 34-channel Laboratory Instrument Panel: `fly_nca/instruments/catalog.py` & `panel.py`
"""

import os
import subprocess
import sys

# If invoked via `python` on a system where `python` is Python 2.x, re-exec under `python3`.
if sys.version_info[0] < 3:
    os.execvp("python3", ["python3", os.path.abspath(__file__)] + sys.argv[1:])

import argparse
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parent
_SRC = _REPO / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _venv_python(repo_root):
    if os.name == "nt":
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python3"


def _bootstrap_venv_and_reexec(argv):
    """Automatically create `.venv`, install `requirements.txt`, and re-exec inside `.venv`."""
    if os.environ.get("FLY_NCA_BOOTSTRAPPED") == "1":
        return None
    venv_dir = _REPO / ".venv"
    venv_py = _venv_python(_REPO)
    req_file = _REPO / "requirements.txt"
    if not venv_py.is_file():
        print("[fly-nca] Creating virtual environment in {} ...".format(venv_dir), file=sys.stderr)
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    check_cmd = [str(venv_py), "-c", "import pyarrow.feather, torch, pandas"]
    if subprocess.call(check_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        print("[fly-nca] Installing dependencies from {} into .venv ...".format(req_file.name), file=sys.stderr)
        subprocess.check_call([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(venv_py), "-m", "pip", "install", "-r", str(req_file)])
    env = os.environ.copy()
    env["FLY_NCA_BOOTSTRAPPED"] = "1"
    cmd = [str(venv_py), os.path.abspath(__file__)] + list(argv if argv is not None else sys.argv[1:])
    return subprocess.call(cmd, env=env)


try:
    import pyarrow.feather as _pa_feather  # noqa: F401 - import before torch on Windows when available
except ImportError:
    _pa_feather = None

try:
    import torch
except ImportError:
    torch = None

from fly_nca.app.scenario import SCENARIOS, get_scenario
from verify_checks.suites import (
    SHIP_SUITE_IDS,
    SUITE_ORDER,
    VERIFY_SUITES,
    run_suites,
    verify_suite_ids_for_scenarios,
)


def cmd_list(_):
    print("{:<22}  {:<4}  description".format("suite", "ship"))
    print("-" * 100)
    for sid in SUITE_ORDER:
        spec = VERIFY_SUITES[sid]
        ship = "yes" if spec.ship else "no"
        print("{:<22}  {:<4}  {}".format(sid, ship, spec.description))
    print("\nWet-Lab Scenarios (same ids as `python3 -m fly_nca.app run <id>`):")
    print("{:<46}  {:<32}  verify_suites".format("scenario", "laboratory"))
    print("-" * 116)
    for sid, spec in SCENARIOS.items():
        suites = ",".join(spec.verify_suites)
        print("{:<46}  {:<32}  {}".format(sid, spec.laboratory.value, suites))
    print("\nRun: python3 verify_principles.py --scenario pmc13518903_vt062617_wind_tunnel")
    print("     python3 verify_principles.py [--ship | --suite wetlab_experiments ...]")
    return 0


def _print_row(row):
    status = "PASS" if row["ok"] else "FAIL"
    payload = {k: v for k, v in row.items() if k not in ("id", "ok", "suite")}
    print("{:<46}  {}  {}".format(row["id"], status, payload))


def _print_results(device, rows, by_suite=False):
    print("device={}".format(device))
    print("-" * 100)
    if not by_suite:
        for row in rows:
            _print_row(row)
        return
    current = None
    for row in rows:
        sid = row.get("suite", "?")
        if sid != current:
            current = sid
            spec = VERIFY_SUITES.get(sid)
            desc = spec.description if spec else ""
            print("\n=== {} ===  {}".format(sid, desc))
        _print_row(row)


def main(argv=None):
    has_cuda = bool(torch is not None and torch.cuda.is_available())
    parser = argparse.ArgumentParser(description="Fly-NCA First-Principles & Wet-Lab Verification")
    parser.add_argument("--device", default="cuda" if has_cuda else "cpu")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list", action="store_true", help="List verify suites and wet-lab scenarios.")
    parser.add_argument(
        "--ship",
        action="store_true",
        help="Ship profile: {}.".format(", ".join(SHIP_SUITE_IDS)),
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

    if torch is None or _pa_feather is None:
        rc = _bootstrap_venv_and_reexec(argv)
        if rc is not None:
            return rc
        print("Error: failed to bootstrap .venv with torch and pyarrow.", file=sys.stderr)
        return 1

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
