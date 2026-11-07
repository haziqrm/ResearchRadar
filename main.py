"""Rebuild every output file, in dependency order.

Usage:
    python main.py                 # run every stage that exists yet
    python main.py --only a d      # run only some stages
    python main.py --list          # show stages and whether each exists

Each stage is a module run as `python -m <module>`. A stage whose module
hasn't been written yet is skipped with a note, so this always runs.
Owners: add your runner module at the path below (or change the path here
and tell the team). The app itself is started separately with Streamlit.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys

# key -> (owner, module, what it writes). Order matters: A feeds D's backtest.
STAGES: dict[str, tuple[str, str, str]] = {
    "a": ("A", "src.signal_decay.run_decay", "factors.csv, decay_results.csv"),
    "b": ("B", "src.research_map.run_map", "network_*.csv, area_forecasts.csv"),
    "d": ("D", "src.live.run_live", "crowding.csv, backtest_results.csv, briefing.*"),
}


def module_exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="+", choices=sorted(STAGES), metavar="STAGE",
                        help="stages to run (a, b, d); default all")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    args = parser.parse_args()

    if args.list:
        for key, (owner, module, outputs) in STAGES.items():
            state = "ready" if module_exists(module) else "not written yet"
            print(f"{key}  {owner}  {module:<28} {state:<16} -> {outputs}")
        return 0

    failed = []
    for key in args.only or STAGES:
        owner, module, outputs = STAGES[key]
        if not module_exists(module):
            print(f"[skip] {owner}: {module} not written yet")
            continue
        print(f"[run]  {owner}: python -m {module}  -> {outputs}", flush=True)
        if subprocess.run([sys.executable, "-m", module]).returncode != 0:
            failed.append(owner)

    if failed:
        print(f"[fail] stages failed: {', '.join(failed)}")
        return 1
    print("[done] start the app with: streamlit run src/app/<entry>.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
