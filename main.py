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
"""

from __future__ import annotations

import argparse
import json

from stock_screener_engine.app import run_screen, run_single_stock


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

    # ── screen ──────────────────────────────────────────────────────────────
    subparsers.add_parser(
        "screen",
        help="Run full market screening (daily + intraday) across the universe.",
    )

    # ── analyze ─────────────────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Deep single-stock analysis.  Example: python main.py analyze RELIANCE",
    )
    analyze_parser.add_argument(
        "symbol",
        type=str,
        help="NSE ticker symbol, e.g. RELIANCE, TCS, INFY",
    )

    args = parser.parse_args()

    if args.command == "screen":
        result = run_screen()
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "analyze":
        result = run_single_stock(args.symbol.strip().upper())
        print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

