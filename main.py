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
from datetime import date

from stock_screener_engine.app import (
    run_data_foundation,
    run_data_quality,
    run_deepdive_report,
    run_document_ingest,
    run_live_invalidation_daily,
    run_screen,
    run_sector_rankings,
    run_single_stock,
)
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

    scan_parser = subparsers.add_parser(
        "scan",
        help="Run upgraded professional market scan.",
    )
    scan_parser.add_argument("--mode", choices=["daily", "swing", "full"], default="full")

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

    deepdive_parser = subparsers.add_parser(
        "deepdive",
        help="Company deep-dive research report.",
    )
    deepdive_parser.add_argument("symbol", type=str)
    deepdive_parser.add_argument("--document", type=str, default=None)
    deepdive_parser.add_argument("--format", choices=["json", "markdown"], default="json")

    sector_parser = subparsers.add_parser(
        "sector-rankings",
        help="Run sector intelligence rankings.",
    )
    sector_parser.add_argument("--format", choices=["json", "markdown"], default="json")

    sector_report_parser = subparsers.add_parser(
        "sector-report",
        help="Run sector intelligence report for one sector.",
    )
    sector_report_parser.add_argument("--sector", required=True)

    doc_parser = subparsers.add_parser(
        "document-ingest",
        help="Ingest a local PDF/text financial document.",
    )
    doc_parser.add_argument("--symbol", required=True)
    doc_parser.add_argument("--file", required=True)
    doc_parser.add_argument("--document-type", default="unknown")

    foundation_parser = subparsers.add_parser(
        "data-foundation",
        help="Build canonical security/calendar/OHLCV/corporate-action store.",
    )
    foundation_parser.add_argument("--start", required=True)
    foundation_parser.add_argument("--end", required=True)
    foundation_parser.add_argument("--symbols", default="")
    foundation_parser.add_argument("--interval", default="1d")

    quality_parser = subparsers.add_parser(
        "data-quality",
        help="Check canonical data quality and source reconciliation.",
    )
    quality_parser.add_argument("--start", required=True)
    quality_parser.add_argument("--end", required=True)
    quality_parser.add_argument("--symbols", default="")
    quality_parser.add_argument("--interval", default="1d")

    # -- invalidation -------------------------------------------------------
    subparsers.add_parser(
        "invalidation",
        help="Run daily live invalidation checks on open broker positions.",
    )

    args = parser.parse_args()

    if args.command == "screen":
        result = run_screen()
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "scan":
        result = run_screen()
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "analyze":
        result = run_single_stock(args.symbol.strip().upper())
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "deepdive":
        result = run_deepdive_report(
            args.symbol.strip().upper(),
            document_path=args.document,
            output_format=args.format,
        )
        print(result.get("markdown") if args.format == "markdown" else json.dumps(result, indent=2, default=_json_default))

    elif args.command == "sector-rankings":
        result = run_sector_rankings()
        print(result.get("markdown") if args.format == "markdown" else json.dumps(result, indent=2, default=_json_default))

    elif args.command == "sector-report":
        result = run_sector_rankings()
        reports = [
            report
            for report in result["sector_rankings"]
            if str(report.get("sector", "")).lower() == args.sector.lower()
        ]
        print(json.dumps({"sector_report": reports[0] if reports else None}, indent=2, default=_json_default))

    elif args.command == "document-ingest":
        result = run_document_ingest(
            symbol=args.symbol.strip().upper(),
            file_path=args.file,
            document_type=args.document_type,
        )
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "data-foundation":
        result = run_data_foundation(
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            symbols=_parse_symbols(args.symbols),
            interval=args.interval,
        )
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "data-quality":
        result = run_data_quality(
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            symbols=_parse_symbols(args.symbols),
            interval=args.interval,
        )
        print(json.dumps(result, indent=2, default=_json_default))

    elif args.command == "invalidation":
        settings = load_settings()
        result = run_live_invalidation_daily(settings)
        print(json.dumps(result, indent=2, default=_json_default))


def _parse_symbols(text: str) -> list[str] | None:
    values = [part.strip().upper() for part in text.split(",") if part.strip()]
    return values or None


if __name__ == "__main__":
    main()
