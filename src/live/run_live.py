"""Run D's live-layer pipeline in order.

Usage:
    python -m src.live.run_live [--only fetch tag ...] [--refresh]

Stages that aren't written yet are skipped, so this always runs.
"""

from __future__ import annotations

import argparse
import importlib.util

# stage -> module with a main(argv) function. Order is the dependency order.
STAGES: dict[str, str] = {
    "fetch": "src.live.arxiv_feed",
    # Gemini tags arXiv papers into the 8 map areas. Free tier: 20 requests a day, so a
    # run stops at the quota and the next one carries on from the cache. The older
    # src.live.tagging (D's original areas, for the paper -> signal match) is no longer
    # a stage; run it by hand if needed.
    "map_tags": "src.live.map_tags",
    "crowding": "src.live.crowding",       # OpenAlex by year (default); --source arxiv later
    "backtest": "src.live.backtest",
    "briefing": "src.live.briefing",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run D's live-layer pipeline.")
    parser.add_argument("--only", nargs="+", choices=list(STAGES), metavar="STAGE")
    parser.add_argument("--refresh", action="store_true",
                        help="refetch the newest arXiv months")
    args = parser.parse_args(argv)

    for stage in args.only or STAGES:
        module = STAGES[stage]
        if importlib.util.find_spec(module) is None:
            print(f"[skip] {stage}: {module} not written yet")
            continue
        print(f"[run]  {stage}")
        stage_args = ["--refresh"] if stage == "fetch" and args.refresh else []
        importlib.import_module(module).main(stage_args)


if __name__ == "__main__":
    main()
