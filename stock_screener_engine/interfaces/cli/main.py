"""CLI for the Indian equity intelligence engine."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from stock_screener_engine.app import (
    run_data_foundation,
    run_data_quality,
    run_deepdive_report,
    run_document_ingest,
    run_financials_ingest,
    run_peer_report,
    run_screen,
    run_sector_rankings,
    run_sector_peer_report,
    run_security_master_ingest,
    run_shareholding_ingest,
    run_single_stock,
    run_valuation_ingest,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="stock-engine",
        description="Indian equity research, signal, document, and sector intelligence engine",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    scan = subparsers.add_parser("scan", help="Run stock signal scan")
    scan.add_argument("--mode", choices=["daily", "swing", "full"], default="full")
    scan.add_argument("--format", choices=["json", "table", "markdown"], default="json")
    _add_source_arg(scan)

    analyze = subparsers.add_parser("analyze", help="Run single-stock analysis")
    analyze.add_argument("symbol")
    _add_source_arg(analyze)

    deepdive = subparsers.add_parser("deepdive", help="Run company deep-dive report")
    deepdive.add_argument("symbol")
    deepdive.add_argument("--include-documents", action="store_true")
    deepdive.add_argument("--document", type=str, default=None)
    deepdive.add_argument("--format", choices=["json", "markdown"], default="json")
    _add_source_arg(deepdive)

    sector_report = subparsers.add_parser("sector-report", help="Render one sector report")
    sector_report.add_argument("--sector", required=True)
    sector_report.add_argument("--include-peers", action="store_true")
    sector_report.add_argument("--as-of", default=None, help="Peer-ranking cutoff YYYY-MM-DD")
    sector_report.add_argument("--format", choices=["json", "markdown"], default="json")
    _add_source_arg(sector_report)

    sector_rankings = subparsers.add_parser("sector-rankings", help="Rank all covered sectors")
    sector_rankings.add_argument("--format", choices=["json", "markdown"], default="json")
    _add_source_arg(sector_rankings)

    peer_report = subparsers.add_parser("peer-report", help="Build canonical peer-comparison report")
    peer_report.add_argument("symbol")
    peer_report.add_argument("--as-of", default=None, help="Point-in-time cutoff YYYY-MM-DD")
    peer_report.add_argument("--format", choices=["json", "markdown"], default="json")

    doc = subparsers.add_parser("document-ingest", help="Ingest a local PDF/text financial document")
    doc.add_argument("--symbol", required=True)
    doc.add_argument("--file", required=True)
    doc.add_argument("--company-name", default=None)
    doc.add_argument("--document-type", default="unknown")

    foundation = subparsers.add_parser("data-foundation", help="Build canonical data foundation store")
    foundation.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    foundation.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    foundation.add_argument("--symbols", default="", help="Comma-separated symbol override")
    foundation.add_argument("--interval", default="1d")
    _add_source_arg(foundation)

    quality = subparsers.add_parser("data-quality", help="Check canonical data quality")
    quality.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    quality.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    quality.add_argument("--symbols", default="", help="Comma-separated symbol override")
    quality.add_argument("--interval", default="1d")

    security_master = subparsers.add_parser("security-master-ingest", help="Ingest local CSV security master rows")
    security_master.add_argument("--file", required=True)
    security_master.add_argument("--venue", default=None)

    financials = subparsers.add_parser("financials-ingest", help="Ingest local CSV financial statements")
    financials.add_argument("--symbol", required=True)
    financials.add_argument("--file", required=True)
    financials.add_argument("--as-of", required=True, help="Point-in-time cutoff YYYY-MM-DD")
    financials.add_argument("--venue", default=None)

    valuation = subparsers.add_parser("valuation-ingest", help="Ingest local CSV market-cap/share-count data")
    valuation.add_argument("--symbol", required=True)
    valuation.add_argument("--file", required=True)
    valuation.add_argument("--as-of", required=True, help="Point-in-time cutoff YYYY-MM-DD")
    valuation.add_argument("--venue", default=None)

    shareholding = subparsers.add_parser("shareholding-ingest", help="Ingest local CSV promoter/FII/DII holdings")
    shareholding.add_argument("--symbol", required=True)
    shareholding.add_argument("--file", required=True)
    shareholding.add_argument("--as-of", required=True, help="Point-in-time cutoff YYYY-MM-DD")
    shareholding.add_argument("--venue", default=None)

    explain = subparsers.add_parser("explain", help="Explain one stock's current signal")
    explain.add_argument("symbol")
    _add_source_arg(explain)

    export = subparsers.add_parser("export-report", help="Export a company report to stdout or file")
    export.add_argument("symbol")
    export.add_argument("--format", choices=["json", "markdown"], default="markdown")
    export.add_argument("--output", type=str, default=None)
    _add_source_arg(export)

    args = parser.parse_args(argv)
    config_path = args.config
    _apply_source_override(getattr(args, "source", None))

    if args.command == "scan":
        result = run_screen(config_path=config_path)
        payload = _scan_payload(result, mode=args.mode, fmt=args.format)
        _emit(payload, fmt=args.format)
        return

    if args.command == "analyze":
        _emit(run_single_stock(args.symbol, config_path=config_path))
        return

    if args.command == "deepdive":
        document_path = args.document if args.include_documents else None
        result = run_deepdive_report(
            args.symbol,
            config_path=config_path,
            document_path=document_path,
            output_format=args.format,
        )
        _emit(result.get("markdown", result), fmt=args.format)
        return

    if args.command == "sector-rankings":
        result = run_sector_rankings(config_path=config_path)
        _emit(result.get("markdown", result) if args.format == "markdown" else result, fmt=args.format)
        return

    if args.command == "sector-report":
        result = run_sector_rankings(config_path=config_path)
        reports = [
            report
            for report in result["sector_rankings"]
            if str(report.get("sector", "")).lower() == args.sector.lower()
        ]
        peer_payload = None
        if args.include_peers:
            peer_payload = run_sector_peer_report(
                args.sector,
                config_path=config_path,
                as_of=date.fromisoformat(args.as_of) if args.as_of else None,
                output_format=args.format,
            )
        if args.format == "markdown":
            from stock_screener_engine.sector.sector_report import SectorIntelligenceBuilder
            from stock_screener_engine.sector.sector_schemas import SectorIntelligenceReport

            objects = [SectorIntelligenceReport(**report) for report in reports]
            markdown = SectorIntelligenceBuilder().render_markdown(objects, sector=args.sector)
            if isinstance(peer_payload, dict):
                markdown = markdown.rstrip() + "\n\n" + str(peer_payload.get("markdown", "")).strip() + "\n"
            _emit(markdown, fmt="markdown")
        else:
            payload = {"sector_report": reports[0] if reports else None}
            if peer_payload is not None:
                payload["peer_comparison"] = peer_payload
            _emit(payload)
        return

    if args.command == "peer-report":
        result = run_peer_report(
            args.symbol,
            config_path=config_path,
            as_of=date.fromisoformat(args.as_of) if args.as_of else None,
            output_format=args.format,
        )
        _emit(result.get("markdown", result), fmt=args.format)
        return

    if args.command == "document-ingest":
        result = run_document_ingest(
            symbol=args.symbol,
            file_path=args.file,
            config_path=config_path,
            company_name=args.company_name,
            document_type=args.document_type,
        )
        _emit(result)
        return

    if args.command == "data-foundation":
        result = run_data_foundation(
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
        )
        _emit(result)
        return

    if args.command == "data-quality":
        result = run_data_quality(
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
        )
        _emit(result)
        return

    if args.command == "security-master-ingest":
        result = run_security_master_ingest(
            file_path=args.file,
            config_path=config_path,
            venue=args.venue,
        )
        _emit(result)
        return

    if args.command == "financials-ingest":
        result = run_financials_ingest(
            symbol=args.symbol,
            file_path=args.file,
            as_of=date.fromisoformat(args.as_of),
            config_path=config_path,
            venue=args.venue,
        )
        _emit(result)
        return

    if args.command == "valuation-ingest":
        result = run_valuation_ingest(
            symbol=args.symbol,
            file_path=args.file,
            as_of=date.fromisoformat(args.as_of),
            config_path=config_path,
            venue=args.venue,
        )
        _emit(result)
        return

    if args.command == "shareholding-ingest":
        result = run_shareholding_ingest(
            symbol=args.symbol,
            file_path=args.file,
            as_of=date.fromisoformat(args.as_of),
            config_path=config_path,
            venue=args.venue,
        )
        _emit(result)
        return

    if args.command == "explain":
        result = run_deepdive_report(args.symbol, config_path=config_path, output_format="json")
        explanation = {
            "symbol": result["symbol"],
            "final_verdict": result["final_verdict"],
            "top_positive": result["source_analysis"].get("key_drivers", {}).get("top_positive", []),
            "top_negative": result["source_analysis"].get("key_drivers", {}).get("top_negative", []),
            "risk_flags": result["source_analysis"].get("risk_flags", []),
        }
        _emit(explanation)
        return

    if args.command == "export-report":
        result = run_deepdive_report(args.symbol, config_path=config_path, output_format=args.format)
        payload = result.get("markdown", result)
        if args.output:
            Path(args.output).write_text(payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str), encoding="utf-8")
        else:
            _emit(payload, fmt=args.format)


def _scan_payload(result: dict[str, object], mode: str, fmt: str) -> object:
    if fmt == "table":
        reports = result.get("professional_signal_reports", {})
        if not isinstance(reports, dict):
            return []
        rows = reports.get("console_rows", {})
        if not isinstance(rows, dict):
            return []
        if mode == "daily":
            return rows.get("long_term", [])
        if mode == "swing":
            return rows.get("swing", [])
        return {"long_term": rows.get("long_term", []), "swing": rows.get("swing", [])}
    if fmt == "markdown":
        reports = result.get("professional_signal_reports", {})
        if isinstance(reports, dict):
            if mode == "swing":
                return reports.get("markdown_top_swing", "")
            return reports.get("markdown_top_long", "")
    return result


def _emit(payload: object, fmt: str = "json") -> None:
    if fmt == "markdown" and isinstance(payload, str):
        print(payload)
    elif fmt == "table":
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(json.dumps(payload, indent=2, default=str))


def _parse_symbols(text: str) -> list[str] | None:
    values = [part.strip().upper() for part in text.split(",") if part.strip()]
    return values or None


def _add_source_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=["canonical", "nse_http", "nse", "yfinance", "zerodha", "icici", "breeze", "mock"],
        default=None,
        help="Market data source override for this command",
    )


def _apply_source_override(source: str | None) -> None:
    if source:
        os.environ["SSE_MARKET_PROVIDER"] = source
