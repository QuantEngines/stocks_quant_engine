"""CLI for the Indian equity intelligence engine."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

from stock_screener_engine.app import (
    run_backtest_readiness,
    run_broker_health,
    run_conviction_calibration,
    run_data_foundation,
    run_data_entitlements,
    run_data_quality,
    run_data_readiness,
    run_data_source_coverage,
    run_data_source_priority,
    run_deepdive_report,
    run_document_ingest,
    run_engine_backtest,
    run_exchange_delivery_ingest,
    run_exchange_foundation_status,
    run_factor_ingest,
    run_factor_qa,
    run_factor_template,
    run_finedge_factor_export,
    run_finedge_inspect,
    run_finedge_onboarding_plan,
    run_finedge_probe,
    run_fmp_probe,
    run_financials_ingest,
    run_indianapi_probe,
    run_missing_data_list,
    run_peer_report,
    run_screen,
    run_sector_rankings,
    run_sector_peer_report,
    run_security_master_ingest,
    run_shareholding_ingest,
    run_single_stock,
    run_forward_return_labels,
    run_market_refresh,
    run_technical_backtest,
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
    scan.add_argument("--symbols", default="", help="Comma-separated symbol override")
    scan.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    scan.add_argument("--readiness-check", choices=["off", "warn", "enforce"], default="warn")
    scan.add_argument("--readiness-start", default=None, help="Readiness start date YYYY-MM-DD")
    scan.add_argument("--readiness-end", default=None, help="Readiness/as-of date YYYY-MM-DD")
    scan.add_argument("--readiness-lookback-years", type=int, default=5)
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

    sector_rankings_cmd = subparsers.add_parser("sector-rankings", help="Rank all covered sectors")
    sector_rankings_cmd.add_argument("--format", choices=["json", "markdown"], default="json")
    _add_source_arg(sector_rankings_cmd)

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
    foundation.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    foundation.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    foundation.add_argument("--lookback-years", type=int, default=None, help="Compute start date from end date")
    foundation.add_argument("--symbols", default="", help="Comma-separated symbol override")
    foundation.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    foundation.add_argument("--interval", default="1d")
    _add_source_arg(foundation)

    quality = subparsers.add_parser("data-quality", help="Check canonical data quality")
    quality.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    quality.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    quality.add_argument("--lookback-years", type=int, default=None, help="Compute start date from end date")
    quality.add_argument("--symbols", default="", help="Comma-separated symbol override")
    quality.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    quality.add_argument("--interval", default="1d")

    source_coverage = subparsers.add_parser(
        "data-source-coverage",
        help="Aggregate canonical, broker, and vendor-trial data-source coverage",
    )
    source_coverage.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    source_coverage.add_argument("--end", default=None, help="End/as-of date YYYY-MM-DD; defaults to today")
    source_coverage.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    source_coverage.add_argument("--symbols", default="", help="Comma-separated symbol override")
    source_coverage.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    source_coverage.add_argument("--venue", default=None)
    source_coverage.add_argument("--interval", default="1d")
    source_coverage.add_argument("--format", choices=["json", "table", "markdown"], default="markdown")

    source_priority = subparsers.add_parser(
        "data-source-priority",
        help="Show canonical source priority by data domain",
    )
    source_priority.add_argument("--format", choices=["json", "table", "markdown"], default="markdown")

    missing_data = subparsers.add_parser(
        "missing-data-list",
        help="Publish the refined missing-data list after reusable sibling-engine coverage",
    )
    missing_data.add_argument("--quant-root", default=None, help="Parent Quant Engines folder; defaults to current repo parent")
    missing_data.add_argument("--no-cross-engine", action="store_true", help="Do not inspect sibling engines for reusable data")
    missing_data.add_argument("--format", choices=["json", "table", "markdown"], default="markdown")

    data_readiness = subparsers.add_parser(
        "data-readiness",
        help="Evaluate hard data-coverage gates for scans, research, and backtests",
    )
    data_readiness.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    data_readiness.add_argument("--end", default=None, help="End/as-of date YYYY-MM-DD; defaults to today")
    data_readiness.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    data_readiness.add_argument("--mode", default="long-term-scan", help="swing-scan, long-term-scan, deep-research, or backtest")
    data_readiness.add_argument("--symbols", default="", help="Comma-separated symbol override")
    data_readiness.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    data_readiness.add_argument("--venue", default=None)
    data_readiness.add_argument("--interval", default="1d")
    data_readiness.add_argument("--format", choices=["json", "table", "markdown"], default="markdown")

    exchange_status = subparsers.add_parser(
        "exchange-foundation-status",
        help="Report NSE/BSE exchange-foundation coverage and remaining blockers",
    )
    exchange_status.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    exchange_status.add_argument("--end", default=None, help="End/as-of date YYYY-MM-DD; defaults to today")
    exchange_status.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    exchange_status.add_argument("--symbols", default="", help="Comma-separated symbol override")
    exchange_status.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    exchange_status.add_argument("--venue", default=None)
    exchange_status.add_argument("--interval", default="1d")
    exchange_status.add_argument("--format", choices=["json", "table", "markdown"], default="markdown")

    exchange_delivery = subparsers.add_parser(
        "exchange-delivery-ingest",
        help="Ingest an official NSE/BSE delivery-turnover CSV into canonical storage",
    )
    exchange_delivery.add_argument("--file", required=True, help="External delivery/turnover CSV path")
    exchange_delivery.add_argument("--trade-date", default=None, help="Default trade date YYYY-MM-DD if file lacks a date column")
    exchange_delivery.add_argument("--venue", default=None)
    exchange_delivery.add_argument("--source-id", default="")
    exchange_delivery.add_argument("--format", choices=["json", "table"], default="json")

    data_entitlements = subparsers.add_parser(
        "data-entitlements",
        help="Report configured data-source plans, endpoints, symbol entitlements, and licensing notes",
    )
    data_entitlements.add_argument("--symbols", default="", help="Comma-separated symbol override")
    data_entitlements.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    data_entitlements.add_argument("--format", choices=["json", "table", "markdown"], default="markdown")

    refresh = subparsers.add_parser("refresh-market", help="Refresh canonical market data with retries and quality gates")
    refresh.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    refresh.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    refresh.add_argument("--lookback-days", type=int, default=10, help="Compute start date from end date")
    refresh.add_argument("--symbols", default="", help="Comma-separated symbol override")
    refresh.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    refresh.add_argument("--interval", default="1d")
    refresh.add_argument("--batch-size", type=int, default=25)
    refresh.add_argument("--retries", type=int, default=2)
    refresh.add_argument("--retry-delay-seconds", type=float, default=2.0)
    refresh.add_argument("--run-scan", action="store_true")
    refresh.add_argument("--scan-mode", choices=["daily", "swing", "full"], default="swing")
    _add_source_arg(refresh)

    broker_health = subparsers.add_parser("broker-health", help="Compare Zerodha and ICICI Breeze market-data health")
    broker_health.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    broker_health.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    broker_health.add_argument("--lookback-days", type=int, default=10, help="Compute start date from end date")
    broker_health.add_argument("--symbols", default="", help="Comma-separated symbol override")
    broker_health.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    broker_health.add_argument("--sources", default="zerodha,icici_breeze", help="Comma-separated broker sources")
    broker_health.add_argument("--interval", default="1d")
    broker_health.add_argument("--sample-size", type=int, default=None)
    broker_health.add_argument("--price-tolerance-pct", type=float, default=1.0)
    broker_health.add_argument("--retries", type=int, default=2)
    broker_health.add_argument("--retry-delay-seconds", type=float, default=1.0)
    broker_health.add_argument("--primary-source", default="zerodha")
    broker_health.add_argument("--lagged-sources", default="icici_breeze", help="Comma-separated sources allowed one-session EOD lag")
    broker_health.add_argument("--format", choices=["json", "table"], default="json")

    indianapi_probe = subparsers.add_parser(
        "indianapi-probe",
        help="Probe IndianAPI coverage for fundamentals, shareholding, analyst, and history endpoints",
    )
    indianapi_probe.add_argument("--symbols", required=True, help="Comma-separated symbols")
    indianapi_probe.add_argument(
        "--check",
        default="stock,financials,shareholding,analyst,forecasts,history",
        help=(
            "Comma-separated checks: search,stock,financials,shareholding,analyst,"
            "forecasts,history,corporate_actions,announcements,news,trending,"
            "nse_most_active,bse_most_active,price_shockers,week_52,ipo,all"
        ),
    )
    indianapi_probe.add_argument("--stock-base-url", default=None)
    indianapi_probe.add_argument("--analyst-base-url", default=None)
    indianapi_probe.add_argument("--api-key-env", default="SSE_INDIANAPI_API_KEY")
    indianapi_probe.add_argument("--timeout-seconds", type=int, default=20)
    indianapi_probe.add_argument("--retries", type=int, default=1)
    indianapi_probe.add_argument("--retry-delay-seconds", type=float, default=0.5)
    indianapi_probe.add_argument("--format", choices=["json", "table"], default="json")

    fmp_probe = subparsers.add_parser(
        "fmp-probe",
        help="Probe Financial Modeling Prep coverage for Indian equity fundamentals and prices",
    )
    fmp_probe.add_argument("--symbols", required=True, help="Comma-separated symbols")
    fmp_probe.add_argument(
        "--check",
        default="smoke",
        help=(
            "Comma-separated checks: search,profile,quote,income_statement,balance_sheet,"
            "cash_flow,ratios,key_metrics,enterprise_values,market_cap,shares_float,"
            "price_history,analyst_estimates,ratings,grades,transcripts,statements,financials,smoke,all"
        ),
    )
    fmp_probe.add_argument("--base-url", default=None)
    fmp_probe.add_argument("--api-key-env", default="SSE_FMP_API_KEY")
    fmp_probe.add_argument("--timeout-seconds", type=int, default=5)
    fmp_probe.add_argument("--retries", type=int, default=0)
    fmp_probe.add_argument("--retry-delay-seconds", type=float, default=0.5)
    fmp_probe.add_argument("--period", choices=["annual", "quarter"], default="annual")
    fmp_probe.add_argument("--limit", type=int, default=5)
    fmp_probe.add_argument("--price-start", default=None, help="Optional price-history start date YYYY-MM-DD")
    fmp_probe.add_argument("--price-end", default=None, help="Optional price-history end date YYYY-MM-DD")
    fmp_probe.add_argument("--exact-symbols", action="store_true", help="Probe only the submitted symbols without .NS/.BO fallbacks")
    fmp_probe.add_argument("--format", choices=["json", "table"], default="json")

    finedge_probe = subparsers.add_parser(
        "finedge-probe",
        help="Probe FinEdge coverage for Indian fundamentals, ownership, prices, and events",
    )
    finedge_probe.add_argument("--symbols", default="", help="Comma-separated symbols")
    finedge_probe.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    finedge_probe.add_argument(
        "--check",
        default="smoke",
        help=(
            "Comma-separated checks: stock_symbols,company_profile,financials,segment_revenue,"
            "notes,ratios,financial_metrics,basic_financials,quote,daily_quotes,"
            "daily_price_ratios,annual_price_ratios,shareholding_pattern,"
            "shareholding_summary,ownership_current,ownership_history,beneficial_owners,"
            "declarations,corporate_actions,dividends,announcements,credit_ratings,"
            "investor_presentations,investor_call_transcripts,results_calendar,ipo_calendar,"
            "index_master,index_market_history,index_valuation_history,health,smoke,"
            "fundamentals,ownership,events,prices,all"
        ),
    )
    finedge_probe.add_argument("--base-url", default=None)
    finedge_probe.add_argument("--api-key-env", default="SSE_FINEDGE_API_KEY")
    finedge_probe.add_argument("--timeout-seconds", type=int, default=8)
    finedge_probe.add_argument("--retries", type=int, default=0)
    finedge_probe.add_argument("--retry-delay-seconds", type=float, default=0.5)
    finedge_probe.add_argument("--statement-type", default="s", help="FinEdge statement_type, e.g. s or c")
    finedge_probe.add_argument("--statement-code", default="pl", help="FinEdge statement_code, e.g. pl/bs/cf")
    finedge_probe.add_argument("--period", default="annual")
    finedge_probe.add_argument("--ratio-type", default="pr")
    finedge_probe.add_argument("--metrics-ratio-type", default="gr")
    finedge_probe.add_argument("--shareholding-period", default="quarterly")
    finedge_probe.add_argument("--from-date", default=None)
    finedge_probe.add_argument("--to-date", default=None)
    finedge_probe.add_argument("--index-symbol", default="NIFTY 50")
    finedge_probe.add_argument("--format", choices=["json", "table"], default="json")

    finedge_inspect = subparsers.add_parser(
        "finedge-inspect",
        help="Inspect sanitized FinEdge response schemas for field mapping",
    )
    finedge_inspect.add_argument("--symbols", default="", help="Comma-separated symbols")
    finedge_inspect.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    finedge_inspect.add_argument(
        "--check",
        default="fundamentals",
        help=(
            "Comma-separated checks or aliases supported by finedge-probe, e.g. "
            "fundamentals,ownership,prices,events,financials,ratios"
        ),
    )
    finedge_inspect.add_argument("--base-url", default=None)
    finedge_inspect.add_argument("--api-key-env", default="SSE_FINEDGE_API_KEY")
    finedge_inspect.add_argument("--timeout-seconds", type=int, default=8)
    finedge_inspect.add_argument("--retries", type=int, default=0)
    finedge_inspect.add_argument("--retry-delay-seconds", type=float, default=0.5)
    finedge_inspect.add_argument("--statement-type", default="s", help="FinEdge statement_type, e.g. s or c")
    finedge_inspect.add_argument("--statement-code", default="pl", help="FinEdge statement_code, e.g. pl/bs/cf")
    finedge_inspect.add_argument("--period", default="annual")
    finedge_inspect.add_argument("--ratio-type", default="pr")
    finedge_inspect.add_argument("--metrics-ratio-type", default="gr")
    finedge_inspect.add_argument("--shareholding-period", default="quarterly")
    finedge_inspect.add_argument("--from-date", default=None)
    finedge_inspect.add_argument("--to-date", default=None)
    finedge_inspect.add_argument("--index-symbol", default="NIFTY 50")
    finedge_inspect.add_argument("--max-depth", type=int, default=4)
    finedge_inspect.add_argument("--max-fields", type=int, default=80)
    finedge_inspect.add_argument("--max-list-items", type=int, default=25)
    finedge_inspect.add_argument("--format", choices=["json", "table"], default="json")

    finedge_factor_export = subparsers.add_parser(
        "finedge-factor-export",
        help="Export FinEdge financials and ownership into reviewable factor CSVs",
    )
    finedge_factor_export.add_argument("--symbols", default="", help="Comma-separated symbols")
    finedge_factor_export.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    finedge_factor_export.add_argument("--output-root", required=True, help="External/ignored output folder for factor CSVs")
    finedge_factor_export.add_argument("--as-of", required=True, help="Point-in-time cutoff date YYYY-MM-DD")
    finedge_factor_export.add_argument("--sections", default="financials,valuations,shareholding", help="financials,valuations,shareholding,banking,all")
    finedge_factor_export.add_argument("--base-url", default=None)
    finedge_factor_export.add_argument("--api-key-env", default="SSE_FINEDGE_API_KEY")
    finedge_factor_export.add_argument("--timeout-seconds", type=int, default=8)
    finedge_factor_export.add_argument("--retries", type=int, default=0)
    finedge_factor_export.add_argument("--retry-delay-seconds", type=float, default=0.5)
    finedge_factor_export.add_argument("--venue", default=None)
    finedge_factor_export.add_argument("--statement-type", default="s", help="FinEdge statement_type, e.g. s or c")
    finedge_factor_export.add_argument("--period", default="annual")
    finedge_factor_export.add_argument("--shareholding-period", default="quarterly")
    finedge_factor_export.add_argument("--format", choices=["json", "table"], default="json")

    finedge_onboarding = subparsers.add_parser(
        "finedge-onboarding-plan",
        help="Create a local paid-FinEdge onboarding checklist and post-subscription command sequence",
    )
    finedge_onboarding.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    finedge_onboarding.add_argument("--end", default=None, help="End/as-of date YYYY-MM-DD; defaults to today")
    finedge_onboarding.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    finedge_onboarding.add_argument("--symbols", default="", help="Comma-separated symbol override")
    finedge_onboarding.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    finedge_onboarding.add_argument("--venue", default=None)
    finedge_onboarding.add_argument("--interval", default="1d")
    finedge_onboarding.add_argument("--factor-root", default=None, help="External/ignored factor root used in generated commands")
    finedge_onboarding.add_argument("--format", choices=["json", "table", "markdown"], default="markdown")

    readiness = subparsers.add_parser("backtest-readiness", help="Check if canonical data is backtest-ready")
    readiness.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    readiness.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    readiness.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    readiness.add_argument("--symbols", default="", help="Comma-separated symbol override")
    readiness.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    readiness.add_argument("--interval", default="1d")
    readiness.add_argument("--min-history-years", type=float, default=5.0)
    readiness.add_argument("--min-history-rows", type=int, default=None)
    readiness.add_argument("--horizons", default="5,20,60", help="Comma-separated forward-return horizons in bars")
    readiness.add_argument("--require-fundamentals", action="store_true")

    labels = subparsers.add_parser("backtest-labels", help="Generate canonical forward-return labels")
    labels.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    labels.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    labels.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    labels.add_argument("--symbols", default="", help="Comma-separated symbol override")
    labels.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    labels.add_argument("--universe-policy", choices=["current", "eligible_history"], default="current")
    labels.add_argument("--min-history-rows", type=int, default=1000)
    labels.add_argument("--horizons", default="5,20,60", help="Comma-separated forward-return horizons in bars")
    labels.add_argument("--interval", default="1d")

    technical_backtest = subparsers.add_parser("technical-backtest", help="Evaluate first-pass technical ranking")
    technical_backtest.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    technical_backtest.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    technical_backtest.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    technical_backtest.add_argument("--symbols", default="", help="Comma-separated symbol override")
    technical_backtest.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    technical_backtest.add_argument("--universe-policy", choices=["current", "eligible_history"], default="eligible_history")
    technical_backtest.add_argument("--min-history-rows", type=int, default=1000)
    technical_backtest.add_argument("--min-lookback", type=int, default=220)
    technical_backtest.add_argument("--horizons", default="5,20,60", help="Comma-separated forward-return horizons in bars")
    technical_backtest.add_argument("--interval", default="1d")
    technical_backtest.add_argument("--round-trip-cost-bps", type=float, default=None)
    technical_backtest.add_argument("--slippage-bps", type=float, default=5.0)

    engine_backtest = subparsers.add_parser("engine-backtest", help="Evaluate engine scores historically")
    engine_backtest.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    engine_backtest.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    engine_backtest.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    engine_backtest.add_argument("--symbols", default="", help="Comma-separated symbol override")
    engine_backtest.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    engine_backtest.add_argument("--universe-policy", choices=["current", "eligible_history"], default="eligible_history")
    engine_backtest.add_argument("--score-type", choices=["swing", "long_term", "conviction"], default="swing")
    engine_backtest.add_argument("--min-history-rows", type=int, default=1000)
    engine_backtest.add_argument("--min-lookback", type=int, default=220)
    engine_backtest.add_argument("--horizons", default="5,20,60", help="Comma-separated forward-return horizons in bars")
    engine_backtest.add_argument("--interval", default="1d")
    engine_backtest.add_argument("--round-trip-cost-bps", type=float, default=None)
    engine_backtest.add_argument("--slippage-bps", type=float, default=5.0)

    conviction_calibrate = subparsers.add_parser(
        "conviction-calibrate",
        help="Run engine backtest and persist latest conviction evidence artifact",
    )
    conviction_calibrate.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    conviction_calibrate.add_argument("--end", default=None, help="End date YYYY-MM-DD; defaults to today")
    conviction_calibrate.add_argument("--lookback-years", type=int, default=5, help="Compute start date from end date")
    conviction_calibrate.add_argument("--symbols", default="", help="Comma-separated symbol override")
    conviction_calibrate.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    conviction_calibrate.add_argument("--universe-policy", choices=["current", "eligible_history"], default="eligible_history")
    conviction_calibrate.add_argument("--score-type", choices=["swing", "long_term", "conviction"], default="conviction")
    conviction_calibrate.add_argument("--min-history-rows", type=int, default=1000)
    conviction_calibrate.add_argument("--min-lookback", type=int, default=220)
    conviction_calibrate.add_argument("--horizons", default="5,20,60", help="Comma-separated forward-return horizons in bars")
    conviction_calibrate.add_argument("--interval", default="1d")
    conviction_calibrate.add_argument("--round-trip-cost-bps", type=float, default=None)
    conviction_calibrate.add_argument("--slippage-bps", type=float, default=5.0)
    conviction_calibrate.add_argument("--output-path", default=None, help="Optional explicit JSON artifact path")

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

    factor_template = subparsers.add_parser("factor-template", help="Create external PIT factor CSV templates")
    factor_template.add_argument("--output-root", required=True)
    factor_template.add_argument("--as-of", required=True, help="Template valuation as-of date YYYY-MM-DD")
    factor_template.add_argument("--symbols", default="", help="Comma-separated symbol override")
    factor_template.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    factor_template.add_argument("--overwrite", action="store_true")

    factor_ingest = subparsers.add_parser("factor-ingest", help="Bulk ingest external PIT factor CSV files")
    factor_ingest.add_argument("--root", required=True)
    factor_ingest.add_argument("--as-of", required=True, help="Point-in-time cutoff YYYY-MM-DD")
    factor_ingest.add_argument("--symbols", default="", help="Comma-separated symbol override")
    factor_ingest.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    factor_ingest.add_argument("--venue", default=None)
    factor_ingest.add_argument("--min-coverage", type=float, default=1.0)
    factor_ingest.add_argument("--sections", default="financials,valuations,shareholding", help="financials,valuations,shareholding,banking,all")

    factor_qa = subparsers.add_parser("factor-qa", help="QA canonical factor coverage, latest values, and mapping warnings")
    factor_qa.add_argument("--as-of", default=None, help="Point-in-time cutoff YYYY-MM-DD; defaults to today")
    factor_qa.add_argument("--symbols", default="", help="Comma-separated symbol override")
    factor_qa.add_argument("--universe-file", default=None, help="External CSV/plain-text universe file")
    factor_qa.add_argument("--venue", default=None)
    factor_qa.add_argument("--statement-type", default=None, help="Optional canonical statement_type filter")
    factor_qa.add_argument("--format", choices=["json", "table", "markdown"], default="table")

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
        result = run_screen(
            config_path=config_path,
            scan_mode=args.mode,
            symbols=_parse_symbols(args.symbols),
            universe_file=args.universe_file,
            readiness_check=args.readiness_check,
            readiness_as_of=date.fromisoformat(args.readiness_end) if args.readiness_end else None,
            readiness_start=date.fromisoformat(args.readiness_start) if args.readiness_start else None,
            readiness_lookback_years=args.readiness_lookback_years,
        )
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
        rankings_payload = result.get("sector_rankings", [])
        sector_rankings = rankings_payload if isinstance(rankings_payload, list) else []
        reports = [
            report
            for report in sector_rankings
            if isinstance(report, Mapping)
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
        start, end = _resolve_date_range(args, parser)
        result = run_data_foundation(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
        )
        _emit(result)
        return

    if args.command == "data-quality":
        start, end = _resolve_date_range(args, parser)
        result = run_data_quality(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
        )
        _emit(result)
        return

    if args.command == "data-source-coverage":
        start, end = _resolve_date_range(args, parser)
        result = run_data_source_coverage(
            as_of=end,
            start=start,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            venue=args.venue,
        )
        _emit(_data_source_coverage_payload(result, args.format), fmt=args.format)
        return

    if args.command == "data-source-priority":
        result = run_data_source_priority(config_path=config_path)
        _emit(_data_source_priority_payload(result, args.format), fmt=args.format)
        return

    if args.command == "missing-data-list":
        result = run_missing_data_list(
            config_path=config_path,
            quant_root=args.quant_root,
            include_cross_engine=not args.no_cross_engine,
        )
        _emit(_missing_data_payload(result, args.format), fmt=args.format)
        return

    if args.command == "data-readiness":
        start, end = _resolve_date_range(args, parser)
        result = run_data_readiness(
            as_of=end,
            start=start,
            mode=args.mode,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            venue=args.venue,
        )
        _emit(_data_readiness_payload(result, args.format), fmt=args.format)
        return

    if args.command == "exchange-foundation-status":
        start, end = _resolve_date_range(args, parser)
        result = run_exchange_foundation_status(
            as_of=end,
            start=start,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            venue=args.venue,
        )
        _emit(_exchange_foundation_payload(result, args.format), fmt=args.format)
        return

    if args.command == "exchange-delivery-ingest":
        result = run_exchange_delivery_ingest(
            file_path=args.file,
            trade_date=date.fromisoformat(args.trade_date) if args.trade_date else None,
            config_path=config_path,
            venue=args.venue,
            source_id=args.source_id,
        )
        _emit(_exchange_delivery_payload(result, args.format), fmt=args.format)
        return

    if args.command == "data-entitlements":
        result = run_data_entitlements(
            symbols=_parse_symbols(args.symbols),
            universe_file=args.universe_file,
            config_path=config_path,
        )
        _emit(_data_entitlements_payload(result, args.format), fmt=args.format)
        return

    if args.command == "refresh-market":
        start, end = _resolve_refresh_date_range(args)
        result = run_market_refresh(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            batch_size=args.batch_size,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            run_scan=args.run_scan,
            scan_mode=args.scan_mode,
        )
        _emit(result)
        return

    if args.command == "broker-health":
        start, end = _resolve_refresh_date_range(args)
        result = run_broker_health(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            universe_file=args.universe_file,
            config_path=config_path,
            sources=_parse_symbols(args.sources),
            interval=args.interval,
            sample_size=args.sample_size,
            price_tolerance_pct=args.price_tolerance_pct,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            primary_source=args.primary_source,
            lagged_sources=_parse_symbols(args.lagged_sources),
        )
        _emit(_broker_health_payload(result, args.format), fmt=args.format)
        return

    if args.command == "indianapi-probe":
        result = run_indianapi_probe(
            symbols=_parse_symbols(args.symbols) or [],
            checks=_parse_csv(args.check),
            config_path=config_path,
            stock_base_url=args.stock_base_url,
            analyst_base_url=args.analyst_base_url,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        _emit(_indianapi_probe_payload(result, args.format), fmt=args.format)
        return

    if args.command == "fmp-probe":
        result = run_fmp_probe(
            symbols=_parse_symbols(args.symbols) or [],
            checks=_parse_csv(args.check),
            config_path=config_path,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            period=args.period,
            limit=args.limit,
            price_start=date.fromisoformat(args.price_start) if args.price_start else None,
            price_end=date.fromisoformat(args.price_end) if args.price_end else None,
            exact_symbols=args.exact_symbols,
        )
        _emit(_fmp_probe_payload(result, args.format), fmt=args.format)
        return

    if args.command == "finedge-probe":
        result = run_finedge_probe(
            symbols=_parse_symbols(args.symbols) or [],
            checks=_parse_csv(args.check),
            config_path=config_path,
            universe_file=args.universe_file,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            statement_type=args.statement_type,
            statement_code=args.statement_code,
            period=args.period,
            ratio_type=args.ratio_type,
            metrics_ratio_type=args.metrics_ratio_type,
            shareholding_period=args.shareholding_period,
            from_date=args.from_date,
            to_date=args.to_date,
            index_symbol=args.index_symbol,
        )
        _emit(_finedge_probe_payload(result, args.format), fmt=args.format)
        return

    if args.command == "finedge-inspect":
        result = run_finedge_inspect(
            symbols=_parse_symbols(args.symbols) or [],
            checks=_parse_csv(args.check),
            config_path=config_path,
            universe_file=args.universe_file,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            statement_type=args.statement_type,
            statement_code=args.statement_code,
            period=args.period,
            ratio_type=args.ratio_type,
            metrics_ratio_type=args.metrics_ratio_type,
            shareholding_period=args.shareholding_period,
            from_date=args.from_date,
            to_date=args.to_date,
            index_symbol=args.index_symbol,
            max_depth=args.max_depth,
            max_fields=args.max_fields,
            max_list_items=args.max_list_items,
        )
        _emit(_finedge_inspect_payload(result, args.format), fmt=args.format)
        return

    if args.command == "finedge-factor-export":
        result = run_finedge_factor_export(
            symbols=_parse_symbols(args.symbols) or [],
            output_root=args.output_root,
            as_of=date.fromisoformat(args.as_of),
            config_path=config_path,
            universe_file=args.universe_file,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay_seconds,
            venue=args.venue,
            statement_type=args.statement_type,
            period=args.period,
            shareholding_period=args.shareholding_period,
            sections=_parse_csv(args.sections),
        )
        _emit(_finedge_factor_export_payload(result, args.format), fmt=args.format)
        return

    if args.command == "finedge-onboarding-plan":
        start, end = _resolve_date_range(args, parser)
        result = run_finedge_onboarding_plan(
            as_of=end,
            start=start,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            venue=args.venue,
            factor_root=args.factor_root,
        )
        _emit(_finedge_onboarding_payload(result, args.format), fmt=args.format)
        return

    if args.command == "backtest-readiness":
        start, end = _resolve_date_range(args, parser)
        result = run_backtest_readiness(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            min_history_years=args.min_history_years,
            min_history_rows=args.min_history_rows,
            horizons=_parse_int_csv(args.horizons),
            require_fundamentals=args.require_fundamentals,
        )
        _emit(result)
        return

    if args.command == "backtest-labels":
        start, end = _resolve_date_range(args, parser)
        result = run_forward_return_labels(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            universe_policy=args.universe_policy,
            min_history_rows=args.min_history_rows,
            horizons=_parse_int_csv(args.horizons),
        )
        _emit(result)
        return

    if args.command == "technical-backtest":
        start, end = _resolve_date_range(args, parser)
        result = run_technical_backtest(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            universe_policy=args.universe_policy,
            min_history_rows=args.min_history_rows,
            min_lookback=args.min_lookback,
            horizons=_parse_int_csv(args.horizons),
            round_trip_cost_bps=args.round_trip_cost_bps,
            slippage_bps=args.slippage_bps,
        )
        _emit(result)
        return

    if args.command == "engine-backtest":
        start, end = _resolve_date_range(args, parser)
        result = run_engine_backtest(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            universe_policy=args.universe_policy,
            min_history_rows=args.min_history_rows,
            min_lookback=args.min_lookback,
            horizons=_parse_int_csv(args.horizons),
            score_type=args.score_type,
            round_trip_cost_bps=args.round_trip_cost_bps,
            slippage_bps=args.slippage_bps,
        )
        _emit(result)
        return

    if args.command == "conviction-calibrate":
        start, end = _resolve_date_range(args, parser)
        result = run_conviction_calibration(
            start=start,
            end=end,
            symbols=_parse_symbols(args.symbols),
            config_path=config_path,
            interval=args.interval,
            universe_file=args.universe_file,
            universe_policy=args.universe_policy,
            min_history_rows=args.min_history_rows,
            min_lookback=args.min_lookback,
            horizons=_parse_int_csv(args.horizons),
            score_type=args.score_type,
            round_trip_cost_bps=args.round_trip_cost_bps,
            slippage_bps=args.slippage_bps,
            output_path=args.output_path,
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

    if args.command == "factor-template":
        result = run_factor_template(
            output_root=args.output_root,
            as_of=date.fromisoformat(args.as_of),
            symbols=_parse_symbols(args.symbols),
            universe_file=args.universe_file,
            config_path=config_path,
            overwrite=args.overwrite,
        )
        _emit(result)
        return

    if args.command == "factor-ingest":
        result = run_factor_ingest(
            root=args.root,
            as_of=date.fromisoformat(args.as_of),
            symbols=_parse_symbols(args.symbols),
            universe_file=args.universe_file,
            config_path=config_path,
            venue=args.venue,
            min_coverage=args.min_coverage,
            sections=_parse_csv(args.sections),
        )
        _emit(result)
        return

    if args.command == "factor-qa":
        as_of = date.fromisoformat(args.as_of) if args.as_of else date.today()
        result = run_factor_qa(
            as_of=as_of,
            symbols=_parse_symbols(args.symbols),
            universe_file=args.universe_file,
            config_path=config_path,
            venue=args.venue,
            statement_type=args.statement_type,
        )
        _emit(_factor_qa_payload(result, args.format), fmt=args.format)
        return

    if args.command == "explain":
        result = run_deepdive_report(args.symbol, config_path=config_path, output_format="json")
        source_analysis = _mapping(result.get("source_analysis"))
        key_drivers = _mapping(source_analysis.get("key_drivers"))
        explanation = {
            "symbol": result["symbol"],
            "final_verdict": result["final_verdict"],
            "top_positive": key_drivers.get("top_positive", []),
            "top_negative": key_drivers.get("top_negative", []),
            "risk_flags": source_analysis.get("risk_flags", []),
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
            payload: object = rows.get("long_term", [])
            return _with_scan_readiness_if_needed(result, payload)
        if mode == "swing":
            payload = rows.get("swing", [])
            return _with_scan_readiness_if_needed(result, payload)
        payload = {"long_term": rows.get("long_term", []), "swing": rows.get("swing", [])}
        return _with_scan_readiness_if_needed(result, payload)
    if fmt == "markdown":
        reports = result.get("professional_signal_reports", {})
        if isinstance(reports, dict):
            readiness = _scan_readiness_markdown(result)
            if mode == "swing":
                markdown = str(reports.get("markdown_top_swing", ""))
            else:
                markdown = str(reports.get("markdown_top_long", ""))
            return (readiness + "\n" + markdown).strip() + "\n" if readiness else markdown
    return result


def _with_scan_readiness_if_needed(result: Mapping[str, object], payload: object) -> object:
    readiness = result.get("data_readiness")
    if not isinstance(readiness, Mapping):
        return payload
    if readiness.get("decision") == "pass" and not result.get("scan_blocked"):
        return payload
    rows = readiness.get("console_rows")
    return {
        "scan_blocked": bool(result.get("scan_blocked")),
        "data_readiness": rows if isinstance(rows, list) else [],
        "signals": payload,
    }


def _scan_readiness_markdown(result: Mapping[str, object]) -> str:
    readiness = result.get("data_readiness")
    if not isinstance(readiness, Mapping):
        return ""
    if readiness.get("decision") == "pass" and not result.get("scan_blocked"):
        return ""
    markdown = readiness.get("markdown")
    return str(markdown).strip() if isinstance(markdown, str) else ""


def _broker_health_payload(result: dict[str, object], fmt: str) -> object:
    if fmt != "table":
        return result
    sources = result.get("source_reports", {})
    if not isinstance(sources, Mapping):
        return []
    rows = []
    for source, report in sources.items():
        if not isinstance(report, Mapping):
            continue
        rows.append(
            {
                "source": source,
                "role": report.get("role"),
                "enabled": report.get("enabled"),
                "quote_coverage": report.get("quote_coverage"),
                "historical_coverage": report.get("historical_coverage"),
                "stale_symbols": len(report.get("stale_symbols", [])) if isinstance(report.get("stale_symbols"), list) else 0,
                "lagged_symbols": len(report.get("lagged_symbols", [])) if isinstance(report.get("lagged_symbols"), list) else 0,
                "errors": report.get("source_errors", []),
                "notes": report.get("source_notes", []),
            }
        )
    return rows


def _data_source_coverage_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "markdown":
        markdown = result.get("markdown")
        return markdown if isinstance(markdown, str) else ""
    if fmt == "table":
        rows = result.get("console_rows")
        return rows if isinstance(rows, list) else []
    return result


def _data_source_priority_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "markdown":
        markdown = result.get("markdown")
        return markdown if isinstance(markdown, str) else ""
    if fmt == "table":
        rows = result.get("rows")
        return rows if isinstance(rows, list) else []
    return result


def _missing_data_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "markdown":
        markdown = result.get("markdown")
        return markdown if isinstance(markdown, str) else ""
    if fmt == "table":
        rows = []
        for row in result.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            if row.get("status") == "covered_upstream_not_wired":
                continue
            rows.append(
                {
                    "variable": row.get("name"),
                    "domain": row.get("domain"),
                    "priority": row.get("priority"),
                    "status": row.get("status"),
                    "preferred_sources": row.get("preferred_sources"),
                    "action": row.get("procurement_action"),
                }
            )
        return rows
    return result


def _data_readiness_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "markdown":
        markdown = result.get("markdown")
        return markdown if isinstance(markdown, str) else ""
    if fmt == "table":
        rows = result.get("console_rows")
        return rows if isinstance(rows, list) else []
    return result


def _exchange_foundation_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "markdown":
        markdown = result.get("markdown")
        return markdown if isinstance(markdown, str) else ""
    if fmt == "table":
        rows = result.get("domains")
        return rows if isinstance(rows, list) else []
    return result


def _exchange_delivery_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "table":
        return [
            {
                "venue": result.get("venue"),
                "file": result.get("file"),
                "input_rows": result.get("input_rows"),
                "persisted": result.get("persisted"),
                "symbols": result.get("symbols"),
                "passed": result.get("passed"),
            }
        ]
    return result


def _data_entitlements_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "markdown":
        markdown = result.get("markdown")
        return markdown if isinstance(markdown, str) else ""
    if fmt == "table":
        rows = result.get("sources")
        if not isinstance(rows, list):
            return []
        out = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            out.append(
                {
                    "source": row.get("display_name"),
                    "plan": row.get("plan_name"),
                    "status": row.get("status"),
                    "enabled": row.get("enabled"),
                    "entitled_symbols": row.get("entitled_symbol_count"),
                    "entitlement_coverage": row.get("entitlement_coverage"),
                    "storage_rights": row.get("storage_rights"),
                    "redistribution_rights": row.get("redistribution_rights"),
                }
            )
        return out
    return result


def _finedge_onboarding_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "markdown":
        markdown = result.get("markdown")
        return markdown if isinstance(markdown, str) else ""
    if fmt == "table":
        rows = result.get("target_domains")
        return rows if isinstance(rows, list) else []
    return result


def _indianapi_probe_payload(result: dict[str, object], fmt: str) -> object:
    return _probe_coverage_payload(result, fmt)


def _fmp_probe_payload(result: dict[str, object], fmt: str) -> object:
    return _probe_coverage_payload(result, fmt)


def _finedge_probe_payload(result: dict[str, object], fmt: str) -> object:
    return _probe_coverage_payload(result, fmt)


def _finedge_inspect_payload(result: dict[str, object], fmt: str) -> object:
    if fmt != "table":
        return result
    rows = []
    market_report = result.get("market_report")
    if isinstance(market_report, Mapping):
        rows.extend(_schema_rows_from_checks("market", market_report.get("checks")))
    for symbol_report in result.get("symbol_reports", []):
        if not isinstance(symbol_report, Mapping):
            continue
        scope = str(symbol_report.get("symbol") or "unknown")
        rows.extend(_schema_rows_from_checks(scope, symbol_report.get("checks")))
    return rows or _probe_coverage_payload(result, fmt)


def _finedge_factor_export_payload(result: dict[str, object], fmt: str) -> object:
    if fmt != "table":
        return result
    row_counts = result.get("row_counts") if isinstance(result.get("row_counts"), Mapping) else {}
    files = result.get("files") if isinstance(result.get("files"), Mapping) else {}
    issue_count = len(result.get("issues", [])) if isinstance(result.get("issues"), list) else 0
    rows = []
    for section in ("financials", "valuations", "shareholding", "banking", "ownership_details"):
        file_info = files.get(section) if isinstance(files, Mapping) else None
        rows.append(
            {
                "section": section,
                "rows": row_counts.get(section, 0) if isinstance(row_counts, Mapping) else 0,
                "path": file_info.get("path") if isinstance(file_info, Mapping) else "",
                "passed": result.get("passed"),
                "issues": issue_count,
            }
        )
    return rows


def _factor_qa_payload(result: dict[str, object], fmt: str) -> object:
    if fmt == "table":
        rows = result.get("console_rows")
        return rows if isinstance(rows, list) else []
    if fmt == "markdown":
        markdown = result.get("markdown")
        return markdown if isinstance(markdown, str) else ""
    return result


def _schema_rows_from_checks(scope: str, checks: object) -> list[dict[str, object]]:
    if not isinstance(checks, Mapping):
        return []
    rows = []
    for check, payload in checks.items():
        if not isinstance(payload, Mapping):
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
        primary = summary.get("primary_record_set") if isinstance(summary, Mapping) else None
        if not isinstance(primary, Mapping):
            primary = {}
        rows.append(
            {
                "scope": scope,
                "check": check,
                "ok": payload.get("ok"),
                "root_type": summary.get("root_type") if isinstance(summary, Mapping) else None,
                "record_set_count": summary.get("record_set_count") if isinstance(summary, Mapping) else 0,
                "primary_path": primary.get("path"),
                "primary_rows": primary.get("item_count"),
                "primary_field_count": primary.get("field_count"),
                "primary_fields": list(primary.get("fields", []))[:20] if isinstance(primary.get("fields"), list) else [],
                "date_like_fields": list(primary.get("date_like_fields", []))[:12] if isinstance(primary.get("date_like_fields"), list) else [],
                "numeric_like_fields": list(primary.get("numeric_like_fields", []))[:12] if isinstance(primary.get("numeric_like_fields"), list) else [],
                "error": payload.get("error", ""),
            }
        )
    return rows


def _probe_coverage_payload(result: dict[str, object], fmt: str) -> object:
    if fmt != "table":
        return result
    coverage = result.get("coverage", {})
    if not isinstance(coverage, Mapping):
        return []
    rows = []
    for check, payload in coverage.items():
        if not isinstance(payload, Mapping):
            continue
        rows.append(
            {
                "check": check,
                "ok": payload.get("ok"),
                "total": payload.get("total"),
                "coverage": payload.get("coverage"),
                "sample_resolved_symbols": payload.get("sample_resolved_symbols", []),
                "sample_errors": payload.get("sample_errors", []),
            }
        )
    return rows


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


def _parse_csv(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _parse_int_csv(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    return values or [5, 20, 60]


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _resolve_date_range(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[date, date]:
    end = date.fromisoformat(args.end) if getattr(args, "end", None) else date.today()
    if getattr(args, "start", None):
        return date.fromisoformat(args.start), end
    years = getattr(args, "lookback_years", None)
    if years is None:
        parser.error("--start or --lookback-years is required")
    return _years_before(end, int(years)), end


def _resolve_refresh_date_range(args: argparse.Namespace) -> tuple[date, date]:
    end = date.fromisoformat(args.end) if getattr(args, "end", None) else date.today()
    if getattr(args, "start", None):
        return date.fromisoformat(args.start), end
    return end - timedelta(days=int(getattr(args, "lookback_days", 10))), end


def _years_before(end: date, years: int) -> date:
    try:
        return end.replace(year=end.year - years)
    except ValueError:
        return end.replace(year=end.year - years, day=28)


def _add_source_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=["canonical", "nse_http", "nse", "yfinance", "zerodha", "icici", "breeze", "icici_breeze", "mock"],
        default=None,
        help="Market data source override for this command",
    )


def _apply_source_override(source: str | None) -> None:
    if source:
        os.environ["SSE_MARKET_PROVIDER"] = source
