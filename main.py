"""Main entry point for the Stock Screener Engine.

Commands
--------
    python main.py screen
        Run the full market screening pass (daily + intraday) across the
        configured universe and print ranked long / swing signals.

    python main.py analyze <SYMBOL>
        Run a deep single-stock analysis for the given NSE symbol and print
        a structured investment report covering technicals, fundamentals,
        multi-horizon assessments, risk flags, and NLP event signals.

        Example:
            python main.py analyze RELIANCE
            python main.py analyze TCS

    python main.py invalidation
        Evaluate open broker positions for stop-loss/thesis/time invalidation
        and write date-stamped reports under data/signals.
"""

from __future__ import annotations

import argparse
import json

from stock_screener_engine.app import run_live_invalidation_daily, run_screen, run_single_stock
from stock_screener_engine.config.settings import load_settings


def _json_default(obj: object) -> str:
    """Fallback serialiser for types json.dumps can't handle (e.g. date)."""
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Stock Screener Engine — NSE equity research toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # -- screen -------------------------------------------------------------
    subparsers.add_parser(
        "screen",
        help="Run full market screening (daily + intraday) across the universe.",
    )

    # -- analyze ------------------------------------------------------------
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Deep single-stock analysis. Example: python main.py analyze RELIANCE",
    )
    analyze_parser.add_argument(
        "symbol",
        type=str,
        help="NSE ticker symbol, e.g. RELIANCE, TCS, INFY",
    )

    # -- invalidation -------------------------------------------------------
    subparsers.add_parser(
        "invalidation",
        help="Run daily live invalidation checks on open broker positions.",
    )

    args = parser.parse_args()

    if args.command == "screen":
        result = run_screen()
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "analyze":
        result = run_single_stock(args.symbol.strip().upper())
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "invalidation":
        settings = load_settings()
        result = run_live_invalidation_daily(settings)
        print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

