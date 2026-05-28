"""Aggregate data-source coverage reporting for the canonical research stack."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

from stock_screener_engine.config.settings import DataSourceEntitlementSettings
from stock_screener_engine.data_sources.schemas import SecurityMasterRecord
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.market_data_store import MarketDataStore


class DataSourceCoverageReporter:
    """Summarize coverage, source roles, and remaining data gaps.

    The reporter intentionally reads only canonical tables and existing quality
    reports. It does not call vendor APIs, so it is safe to run frequently.
    """

    def __init__(
        self,
        *,
        store: MarketDataStore,
        file_store: LocalFileStorage,
        venue: str = "NSE",
        entitlements: Sequence[DataSourceEntitlementSettings] | None = None,
    ) -> None:
        self.store = store
        self.file_store = file_store
        self.venue = venue.strip().upper() or "NSE"
        self.entitlements = list(entitlements or [])

    def build(
        self,
        *,
        symbols: Sequence[str],
        as_of: date,
        start: date,
        interval: str = "1d",
    ) -> dict[str, object]:
        normalized_symbols = _normalize_symbols(symbols)
        metadata_records = self.store.get_security_master(normalized_symbols)
        metadata = {record.symbol: record for record in metadata_records}
        artifacts = self._load_latest_artifacts()

        domain_rows = self._domain_rows(
            symbols=normalized_symbols,
            metadata=metadata,
            as_of=as_of,
            start=start,
            interval=interval,
        )
        source_rows = self._source_rows(domain_rows=domain_rows, artifacts=artifacts)
        _apply_entitlements(
            domain_rows=domain_rows,
            source_rows=source_rows,
            entitlements=self.entitlements,
            symbols=normalized_symbols,
        )
        gross = _gross_coverage(domain_rows)
        entitlement_report = build_data_entitlement_report(self.entitlements, symbols=normalized_symbols)
        report: dict[str, object] = {
            "pipeline": "data_source_coverage",
            "run_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "venue": self.venue,
            "as_of": as_of.isoformat(),
            "start": start.isoformat(),
            "interval": interval,
            "universe": {
                "symbols_requested": len(normalized_symbols),
                "symbols": normalized_symbols,
                "financial_services_symbols": _financial_symbols(normalized_symbols, metadata),
            },
            "gross_coverage": gross,
            "domains": domain_rows,
            "sources": source_rows,
            "entitlements": entitlement_report["sources"],
            "entitlement_summary": entitlement_report["summary"],
            "needed_next": _needed_next(domain_rows=domain_rows, source_rows=source_rows),
            "artifact_evidence": _artifact_evidence(artifacts),
        }
        report["console_rows"] = _console_rows(report)
        report["markdown"] = render_data_source_coverage_markdown(report)
        quality_dir = self.file_store.root / "quality"
        json_path = quality_dir / "data_source_coverage_report.json"
        markdown_path = quality_dir / "data_source_coverage_report.md"
        report["artifacts"] = {"json": str(json_path), "markdown": str(markdown_path)}
        json_path = self.file_store.save_json(report, filename="data_source_coverage_report.json", subdir="quality")
        markdown_path = self.file_store.save_text(
            str(report["markdown"]),
            filename="data_source_coverage_report.md",
            subdir="quality",
        )
        return report

    def _domain_rows(
        self,
        *,
        symbols: Sequence[str],
        metadata: Mapping[str, SecurityMasterRecord],
        as_of: date,
        start: date,
        interval: str,
    ) -> list[dict[str, object]]:
        security = _metadata_coverage(symbols, metadata)
        ohlcv = self.store.coverage_summary(symbols, start=start, end=as_of, interval=interval)
        financials = self.store.financial_statement_coverage(symbols, as_of=as_of, venue=self.venue)
        valuations = self.store.equity_valuation_coverage(symbols, as_of=as_of, venue=self.venue)
        shareholding = self.store.shareholding_coverage(symbols, as_of=as_of, venue=self.venue)
        banking_all = self.store.banking_factor_coverage(symbols, as_of=as_of, venue=self.venue)
        financial_services = _financial_symbols(symbols, metadata)
        banking_applicable = self.store.banking_factor_coverage(financial_services, as_of=as_of, venue=self.venue)
        corporate_actions = self._corporate_action_coverage(symbols)
        delivery = self.store.delivery_turnover_coverage(symbols, as_of=as_of, venue=self.venue)

        rows = [
            _domain_row(
                domain="security_master",
                label="Security master metadata",
                primary_source_id="nse_bse",
                source="NSE/BSE lists + manual/vendor enrichment",
                coverage=security["coverage"],
                covered=security["symbols_with_metadata"],
                total=security["symbols_requested"],
                status=_status(security["coverage"], strong=0.95, usable=0.80),
                latest="",
                covered_symbols=security["symbols_with_metadata_list"],
                applicable_symbols=symbols,
                gaps=security["missing_symbols"],
                notes="Sector and company identity are required before sector reports, peer work, and source confidence.",
            ),
            _domain_row(
                domain="daily_ohlcv",
                label="Daily OHLCV history",
                primary_source_id="zerodha",
                source="Canonical DB from yfinance bootstrap, Zerodha refresh, and exchange/broker adapters",
                coverage=ohlcv["coverage"],
                covered=ohlcv["symbols_with_bars"],
                total=ohlcv["symbols_requested"],
                status=_status(ohlcv["coverage"], strong=0.95, usable=0.80),
                latest=_latest_from_rows(ohlcv.get("rows_by_symbol"), "max_ts"),
                covered_symbols=sorted((ohlcv.get("rows_by_symbol") or {}).keys()) if isinstance(ohlcv.get("rows_by_symbol"), Mapping) else [],
                applicable_symbols=symbols,
                gaps=ohlcv["missing_symbols"],
                notes="Core technical, label, and backtest foundation. Needs corporate-action and source reconciliation depth for production.",
            ),
            _domain_row(
                domain="corporate_actions",
                label="Corporate actions",
                primary_source_id="nse_bse",
                source="NSE/BSE or vendor corporate-action feed",
                coverage=corporate_actions["coverage"],
                covered=corporate_actions["symbols_with_actions"],
                total=corporate_actions["symbols_requested"],
                status="partial" if corporate_actions["symbols_with_actions"] else "gap",
                latest=corporate_actions["latest_action_date"],
                covered_symbols=corporate_actions["symbols_with_actions_list"],
                applicable_symbols=symbols,
                gaps=corporate_actions["symbols_without_actions"],
                notes="Event-style coverage can be naturally sparse, but long-horizon research still needs a reliable adjustment feed.",
            ),
            _domain_row(
                domain="delivery_turnover",
                label="Delivery and turnover",
                primary_source_id="nse_bse",
                source="NSE/BSE delivery-turnover files",
                coverage=delivery["coverage"],
                covered=delivery["symbols_with_delivery"],
                total=delivery["symbols_requested"],
                status=_status(delivery["coverage"], strong=0.95, usable=0.70),
                latest=_latest_from_values(delivery.get("latest_delivery_by_symbol")),
                covered_symbols=sorted((delivery.get("latest_delivery_by_symbol") or {}).keys()) if isinstance(delivery.get("latest_delivery_by_symbol"), Mapping) else [],
                applicable_symbols=symbols,
                gaps=delivery["missing_symbols"],
                notes="High-value Indian-market participation feature for swing signals; now supported through canonical delivery_turnover ingestion.",
            ),
            _domain_row(
                domain="financials",
                label="Point-in-time financial statements",
                primary_source_id="finedge",
                source="FinEdge Basic sample now; FinEdge paid/vendor/filings needed for broad coverage",
                coverage=financials["coverage"],
                covered=financials["symbols_with_statements"],
                total=financials["symbols_requested"],
                status=_status(financials["coverage"], strong=0.90, usable=0.70),
                latest=_latest_from_values(financials.get("latest_period_by_symbol")),
                covered_symbols=sorted((financials.get("latest_period_by_symbol") or {}).keys()) if isinstance(financials.get("latest_period_by_symbol"), Mapping) else [],
                applicable_symbols=symbols,
                gaps=financials["missing_symbols"],
                notes="Basic FinEdge proves mapper path for ITC, RELIANCE, and HDFCBANK; broad universe needs paid/sample-expanded access.",
            ),
            _domain_row(
                domain="valuations",
                label="Market cap and valuation facts",
                primary_source_id="finedge",
                source="FinEdge quote/valuation mapper plus internal derived ratios",
                coverage=valuations["coverage"],
                covered=valuations["symbols_with_valuations"],
                total=valuations["symbols_requested"],
                status=_status(valuations["coverage"], strong=0.90, usable=0.70),
                latest=_latest_from_values(valuations.get("latest_valuation_by_symbol")),
                covered_symbols=sorted((valuations.get("latest_valuation_by_symbol") or {}).keys()) if isinstance(valuations.get("latest_valuation_by_symbol"), Mapping) else [],
                applicable_symbols=symbols,
                gaps=valuations["missing_symbols"],
                notes="Needed for PE/PB/history, market-cap buckets, and valuation risk. Broad historical market-cap history is still open.",
            ),
            _domain_row(
                domain="shareholding",
                label="Shareholding and ownership",
                primary_source_id="finedge",
                source="FinEdge Basic sample now; NSE/BSE/vendor ownership history needed for broad coverage",
                coverage=shareholding["coverage"],
                covered=shareholding["symbols_with_shareholding"],
                total=shareholding["symbols_requested"],
                status=_status(shareholding["coverage"], strong=0.90, usable=0.70),
                latest=_latest_from_values(shareholding.get("latest_period_by_symbol")),
                covered_symbols=sorted((shareholding.get("latest_period_by_symbol") or {}).keys()) if isinstance(shareholding.get("latest_period_by_symbol"), Mapping) else [],
                applicable_symbols=symbols,
                gaps=shareholding["missing_symbols"],
                notes="Required for promoter/FII/DII, pledge, governance, and ownership-trend conviction.",
            ),
            _domain_row(
                domain="banking_factors",
                label="Bank/NBFC-specific factors",
                primary_source_id="finedge",
                source="FinEdge bank mapper plus future bank/financial vendor fields",
                coverage=banking_applicable["coverage"],
                covered=banking_applicable["symbols_with_banking_factors"],
                total=banking_applicable["symbols_requested"],
                status=_status(banking_applicable["coverage"], strong=0.90, usable=0.70)
                if financial_services
                else "not_applicable",
                latest=_latest_from_values(banking_applicable.get("latest_period_by_symbol")),
                covered_symbols=sorted((banking_applicable.get("latest_period_by_symbol") or {}).keys()) if isinstance(banking_applicable.get("latest_period_by_symbol"), Mapping) else [],
                applicable_symbols=financial_services,
                gaps=banking_applicable["missing_symbols"],
                notes=(
                    "Coverage is measured only on Financial Services symbols. "
                    f"All-symbol banking table coverage is {banking_all['coverage']}."
                ),
            ),
            _domain_row(
                domain="events_documents",
                label="Events, filings, and documents",
                primary_source_id="finedge",
                source="NSE/BSE filings, FinEdge announcements/ratings/presentations/transcripts, document pipeline",
                coverage=0.0,
                covered=0,
                total=len(symbols),
                status="gap",
                latest="",
                covered_symbols=[],
                applicable_symbols=symbols,
                gaps=list(symbols),
                notes="Scaffolds exist, but robust canonical event/document coverage is not yet populated.",
            ),
            _domain_row(
                domain="historical_universe",
                label="Historical constituents and delistings",
                primary_source_id="nse_bse",
                source="NSE/BSE or institutional vendor",
                coverage=0.0,
                covered=0,
                total=len(symbols),
                status="gap",
                latest="",
                covered_symbols=[],
                applicable_symbols=symbols,
                gaps=list(symbols),
                notes="Required to remove survivorship bias before claiming world-class backtest evidence.",
            ),
        ]
        return rows

    def _corporate_action_coverage(self, symbols: Sequence[str]) -> dict[str, object]:
        latest_by_symbol: dict[str, str] = {}
        for symbol in symbols:
            actions = self.store.get_corporate_actions(symbol=symbol, venue=self.venue)
            if actions:
                latest_by_symbol[symbol] = max(action.ex_date for action in actions)
        missing = [symbol for symbol in symbols if symbol not in latest_by_symbol]
        return {
            "symbols_requested": len(symbols),
            "symbols_with_actions": len(latest_by_symbol),
            "symbols_with_actions_list": sorted(latest_by_symbol),
            "symbols_without_actions": missing,
            "coverage": round(len(latest_by_symbol) / len(symbols), 4) if symbols else 0.0,
            "latest_action_date": _latest_from_values(latest_by_symbol),
            "latest_action_by_symbol": latest_by_symbol,
        }

    def _load_latest_artifacts(self) -> dict[str, dict[str, object]]:
        root = self.file_store.root
        known = {
            "data_quality": root / "quality" / "data_quality_report.json",
            "broker_health": root / "quality" / "broker_health_report.json",
            "factor_ingest": root / "quality" / "factor_bootstrap_ingest_report.json",
            "factor_qa": root / "quality" / "factor_qa_report.json",
            "finedge_probe": root / "quality" / "finedge_probe_report.json",
            "fmp_probe": root / "quality" / "fmp_probe_report.json",
            "indianapi_probe": root / "quality" / "indianapi_probe_report.json",
        }
        artifacts: dict[str, dict[str, object]] = {}
        for name, path in known.items():
            payload = _read_json(path)
            if payload is not None:
                artifacts[name] = {"path": str(path), "payload": payload, "mtime": _mtime(path)}

        latest_finedge_export = _latest_matching(root / "factors", "finedge_factor_export_report.json")
        if latest_finedge_export is not None:
            payload = _read_json(latest_finedge_export)
            if payload is not None:
                artifacts["finedge_factor_export"] = {
                    "path": str(latest_finedge_export),
                    "payload": payload,
                    "mtime": _mtime(latest_finedge_export),
                }
        return artifacts

    def _source_rows(
        self,
        *,
        domain_rows: Sequence[Mapping[str, object]],
        artifacts: Mapping[str, Mapping[str, object]],
    ) -> list[dict[str, object]]:
        domain = {str(row.get("domain")): row for row in domain_rows}
        broker_health = _artifact_payload(artifacts, "broker_health")
        source_reports = broker_health.get("source_reports") if isinstance(broker_health.get("source_reports"), Mapping) else {}
        zerodha = source_reports.get("zerodha") if isinstance(source_reports, Mapping) and isinstance(source_reports.get("zerodha"), Mapping) else {}
        breeze = (
            source_reports.get("icici_breeze")
            if isinstance(source_reports, Mapping) and isinstance(source_reports.get("icici_breeze"), Mapping)
            else {}
        )
        finedge_export = _artifact_payload(artifacts, "finedge_factor_export")
        finedge_coverage = _source_coverage(
            [
                domain.get("financials", {}),
                domain.get("valuations", {}),
                domain.get("shareholding", {}),
            ]
        )
        data_quality = _artifact_payload(artifacts, "data_quality")
        ohlcv = domain.get("daily_ohlcv", {})

        return [
            {
                "source_id": "nse_bse",
                "source": "NSE/BSE",
                "role": "Official exchange/reference source",
                "current_use": "Security master, exchange adapters, filings/corporate-action direction, future delivery/announcements.",
                "gross_coverage": _source_coverage(
                    [
                        domain.get("security_master", {}),
                        domain.get("corporate_actions", {}),
                        domain.get("delivery_turnover", {}),
                    ]
                ),
                "status": "primary_partial",
                "evidence": "Security metadata is present; direct filings can still be throttled/403 and corporate-action coverage is not yet production-grade.",
                "needed": "Robust bhavcopy, delivery, corporate-action, announcement, historical-constituent, and symbol-change ingestion.",
            },
            {
                "source_id": "zerodha",
                "source": "Zerodha Kite Connect",
                "role": "Primary live/broker market-data source",
                "current_use": "Quotes, broker-backed historical refresh, broker health, future paper/live execution after compliance gates.",
                "gross_coverage": _broker_source_coverage(zerodha),
                "status": "primary_proven" if zerodha else "configured_or_missing_report",
                "evidence": _broker_evidence("zerodha", zerodha),
                "needed": "Keep as live/refresh source; maintain index/instrument mapping and rate-limit handling.",
            },
            {
                "source_id": "icici_breeze",
                "source": "ICICI Breeze",
                "role": "Alternate broker/reconciliation source",
                "current_use": "Historical reconciliation, quote fallback where reliable, future broker redundancy.",
                "gross_coverage": _broker_source_coverage(breeze),
                "status": "primary_partial" if breeze else "configured_or_missing_report",
                "evidence": _broker_evidence("icici_breeze", breeze),
                "needed": "Diagnose remaining quote reliability, stock-code mapping, throttling, and lag policy.",
            },
            {
                "source_id": "finedge",
                "source": "FinEdge",
                "role": "Primary candidate for fundamentals, valuations, shareholding, events, and documents",
                "current_use": "Basic-plan factor development and canonical ingest for ITC, RELIANCE, HDFCBANK.",
                "gross_coverage": finedge_coverage,
                "status": "primary_paid_candidate",
                "evidence": _finedge_evidence(finedge_export, domain),
                "needed": "Paid/sample-expanded Nifty 50/Nifty 500 coverage, bank fields, quarterly history, bulk access, and licensing.",
            },
            {
                "source_id": "yfinance",
                "source": "Yahoo Finance / yfinance",
                "role": "Bootstrap/fallback market-data source",
                "current_use": "Initial daily OHLCV bootstrap and fallback market history.",
                "gross_coverage": float(ohlcv.get("coverage", 0.0) or 0.0),
                "status": "development_fallback",
                "evidence": _data_quality_evidence(data_quality),
                "needed": "Do not rely on it alone for institutional data; reconcile against exchange/broker/vendor sources.",
            },
            {
                "source_id": "satellite_candidates",
                "source": "IndianAPI / FMP",
                "role": "Paused satellite/candidate sources",
                "current_use": "Probe infrastructure and vendor discovery evidence only.",
                "gross_coverage": 0.0,
                "status": "paused",
                "evidence": "IndianAPI free key did not cover core statements/history; FMP hit key-level 429 during trials.",
                "needed": "Revisit only after FinEdge and official-source priorities are clearer.",
            },
        ]


def render_data_source_coverage_markdown(report: Mapping[str, object]) -> str:
    gross = report.get("gross_coverage") if isinstance(report.get("gross_coverage"), Mapping) else {}
    entitlement_summary = report.get("entitlement_summary") if isinstance(report.get("entitlement_summary"), Mapping) else {}
    lines = [
        "# Data Source Coverage",
        "",
        f"- Venue: {report.get('venue')}",
        f"- As of: {report.get('as_of')}",
        f"- Lookback start: {report.get('start')}",
        f"- Universe size: {_nested(report, 'universe', 'symbols_requested')}",
        f"- Critical coverage average: {_pct(gross.get('critical_domain_average'))}",
        f"- Factor coverage average: {_pct(gross.get('factor_domain_average'))}",
        f"- Entitlement sources tracked: {entitlement_summary.get('source_count', 0)}",
        "",
        "## Domain Coverage",
        "",
        "| Domain | Source | Covered | Total | Coverage | Entitled | Entitlement | Status | Latest | Main Gap |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in report.get("domains", []):
        if not isinstance(row, Mapping):
            continue
        entitlement = row.get("entitlement") if isinstance(row.get("entitlement"), Mapping) else {}
        lines.append(
            "| {label} | {source} | {covered} | {total} | {coverage} | {entitled} | {entitlement_coverage} | {status} | {latest} | {gap} |".format(
                label=row.get("label", row.get("domain", "")),
                source=row.get("source", ""),
                covered=row.get("covered", 0),
                total=row.get("total", 0),
                coverage=_pct(row.get("coverage")),
                entitled=entitlement.get("entitled_count", ""),
                entitlement_coverage=_pct(entitlement.get("entitlement_coverage")),
                status=row.get("status", ""),
                latest=row.get("latest", ""),
                gap=_short_gap(row.get("gaps")),
            )
        )

    lines.extend(
        [
            "",
            "## Source Roles",
            "",
            "| Source | Plan | Role | Gross Coverage | Entitlement | Status | Needed |",
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )
    for row in report.get("sources", []):
        if not isinstance(row, Mapping):
            continue
        entitlement = row.get("entitlement") if isinstance(row.get("entitlement"), Mapping) else {}
        lines.append(
            "| {source} | {plan} | {role} | {coverage} | {entitlement_coverage} | {status} | {needed} |".format(
                source=row.get("source", ""),
                plan=entitlement.get("plan_name", ""),
                role=row.get("role", ""),
                coverage=_pct(row.get("gross_coverage")),
                entitlement_coverage=_pct(entitlement.get("entitlement_coverage")),
                status=row.get("status", ""),
                needed=row.get("needed", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Entitlement Registry",
            "",
            "| Source | Plan | Status | Enabled | Symbols Entitled | Storage | Redistribution | Next Action |",
            "| --- | --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in report.get("entitlements", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {source} | {plan} | {status} | {enabled} | {count} | {storage} | {redistribution} | {next_action} |".format(
                source=row.get("display_name", row.get("source_id", "")),
                plan=row.get("plan_name", ""),
                status=row.get("status", ""),
                enabled=row.get("enabled", ""),
                count=row.get("entitled_symbol_count", ""),
                storage=row.get("storage_rights", ""),
                redistribution=row.get("redistribution_rights", ""),
                next_action=row.get("next_action", ""),
            )
        )

    lines.extend(["", "## Needed Next", ""])
    for item in report.get("needed_next", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def build_data_entitlement_report(
    entitlements: Sequence[DataSourceEntitlementSettings],
    *,
    symbols: Sequence[str] | None = None,
) -> dict[str, object]:
    normalized_symbols = _normalize_symbols(symbols or [])
    rows = [_entitlement_payload(entitlement, normalized_symbols) for entitlement in entitlements]
    return {
        "pipeline": "data_entitlements",
        "run_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "summary": {
            "source_count": len(rows),
            "enabled_sources": sum(1 for row in rows if row.get("enabled") is True),
            "sources_requiring_license_review": sum(
                1 for row in rows if str(row.get("license_status", "")).lower() not in {"confirmed", "credentials_required"}
            ),
        },
        "sources": rows,
        "markdown": render_data_entitlements_markdown({"sources": rows}),
    }


def render_data_entitlements_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Data Entitlements",
        "",
        "| Source | Plan | Status | Enabled | Domains | Symbols | Storage | Redistribution | Next Action |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in report.get("sources", []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {source} | {plan} | {status} | {enabled} | {domains} | {symbols} | {storage} | {redistribution} | {next_action} |".format(
                source=row.get("display_name", row.get("source_id", "")),
                plan=row.get("plan_name", ""),
                status=row.get("status", ""),
                enabled=row.get("enabled", ""),
                domains=", ".join(row.get("domains", [])) if isinstance(row.get("domains"), list) else "",
                symbols=row.get("entitled_symbol_count", ""),
                storage=row.get("storage_rights", ""),
                redistribution=row.get("redistribution_rights", ""),
                next_action=row.get("next_action", ""),
            )
        )
    return "\n".join(lines) + "\n"


def _apply_entitlements(
    *,
    domain_rows: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    entitlements: Sequence[DataSourceEntitlementSettings],
    symbols: Sequence[str],
) -> None:
    by_source = {entitlement.source_id: entitlement for entitlement in entitlements}
    for row in domain_rows:
        source_id = str(row.get("primary_source_id", ""))
        entitlement = by_source.get(source_id)
        if entitlement is None:
            continue
        universe = row.get("applicable_symbols")
        applicable = _normalize_symbols(universe if isinstance(universe, list) else list(symbols))
        covered = set(_normalize_symbols(row.get("covered_symbols") if isinstance(row.get("covered_symbols"), list) else []))
        entitlement_payload = _domain_entitlement_payload(entitlement, applicable, covered)
        row["entitlement"] = entitlement_payload
        if entitlement_payload.get("coverage_explanation"):
            row["entitlement_explanation"] = entitlement_payload["coverage_explanation"]

    for row in source_rows:
        source_id = str(row.get("source_id", ""))
        entitlement = by_source.get(source_id)
        if entitlement is None:
            continue
        row["entitlement"] = _source_entitlement_payload(entitlement, symbols)


def _domain_entitlement_payload(
    entitlement: DataSourceEntitlementSettings,
    applicable_symbols: Sequence[str],
    covered_symbols: set[str],
) -> dict[str, object]:
    entitled = set(_entitled_symbols(entitlement, applicable_symbols))
    applicable = set(applicable_symbols)
    covered_entitled = sorted(covered_symbols & entitled)
    entitlement_coverage = round(len(entitled) / len(applicable), 4) if applicable else 0.0
    within_entitlement_coverage = round(len(covered_entitled) / len(entitled), 4) if entitled else 0.0
    missing_entitled = sorted(entitled - covered_symbols)
    missing_not_entitled = sorted(applicable - entitled)
    explanation = ""
    if entitled and within_entitlement_coverage >= 0.999 and missing_not_entitled:
        explanation = "Covered symbols match the current source entitlement; remaining gap is mainly plan/access scope."
    elif missing_entitled:
        explanation = "Some symbols are entitled but missing canonical data; investigate mapper, ingestion, or stale data."
    elif not entitled and applicable:
        explanation = "No symbols are currently entitled by this source/plan for the selected universe."
    return {
        "source_id": entitlement.source_id,
        "display_name": entitlement.display_name,
        "plan_name": entitlement.plan_name,
        "status": entitlement.status,
        "enabled": entitlement.enabled,
        "applicable_count": len(applicable),
        "entitled_count": len(entitled),
        "covered_entitled_count": len(covered_entitled),
        "entitlement_coverage": entitlement_coverage,
        "within_entitlement_coverage": within_entitlement_coverage,
        "missing_entitled_symbols": missing_entitled,
        "not_entitled_symbols": missing_not_entitled,
        "known_limits": entitlement.known_limits,
        "coverage_explanation": explanation,
    }


def _source_entitlement_payload(entitlement: DataSourceEntitlementSettings, symbols: Sequence[str]) -> dict[str, object]:
    entitled = _entitled_symbols(entitlement, symbols)
    total = len(_normalize_symbols(symbols))
    return {
        "source_id": entitlement.source_id,
        "display_name": entitlement.display_name,
        "plan_name": entitlement.plan_name,
        "status": entitlement.status,
        "enabled": entitlement.enabled,
        "domains": entitlement.domains,
        "entitled_symbol_count": len(entitled),
        "symbols_requested": total,
        "entitlement_coverage": round(len(entitled) / total, 4) if total else 0.0,
        "storage_rights": entitlement.storage_rights,
        "redistribution_rights": entitlement.redistribution_rights,
        "commercial_use_rights": entitlement.commercial_use_rights,
        "license_status": entitlement.license_status,
        "known_limits": entitlement.known_limits,
        "next_action": entitlement.next_action,
    }


def _entitlement_payload(
    entitlement: DataSourceEntitlementSettings,
    symbols: Sequence[str],
) -> dict[str, object]:
    entitled = _entitled_symbols(entitlement, symbols)
    total = len(_normalize_symbols(symbols))
    return {
        "source_id": entitlement.source_id,
        "display_name": entitlement.display_name,
        "role": entitlement.role,
        "status": entitlement.status,
        "plan_name": entitlement.plan_name,
        "enabled": entitlement.enabled,
        "domains": entitlement.domains,
        "endpoint_groups": entitlement.endpoint_groups,
        "allowed_symbols": entitlement.allowed_symbols,
        "allowed_universes": entitlement.allowed_universes,
        "credential_envs": entitlement.credential_envs,
        "rate_limit_per_minute": entitlement.rate_limit_per_minute,
        "daily_call_limit": entitlement.daily_call_limit,
        "storage_rights": entitlement.storage_rights,
        "redistribution_rights": entitlement.redistribution_rights,
        "commercial_use_rights": entitlement.commercial_use_rights,
        "license_status": entitlement.license_status,
        "known_limits": entitlement.known_limits,
        "next_action": entitlement.next_action,
        "notes": entitlement.notes,
        "symbols_requested": total,
        "entitled_symbol_count": len(entitled),
        "entitlement_coverage": round(len(entitled) / total, 4) if total else 0.0,
        "entitled_symbols": entitled,
    }


def _entitled_symbols(entitlement: DataSourceEntitlementSettings, symbols: Sequence[str]) -> list[str]:
    normalized = _normalize_symbols(symbols)
    if not entitlement.enabled:
        return []
    allowed = [symbol.upper() for symbol in entitlement.allowed_symbols]
    if "*" in allowed or "ALL" in {item.upper() for item in entitlement.allowed_universes}:
        return normalized
    allowed_set = set(allowed)
    return [symbol for symbol in normalized if symbol in allowed_set]


def _domain_row(
    *,
    domain: str,
    label: str,
    primary_source_id: str,
    source: str,
    coverage: object,
    covered: object,
    total: object,
    status: str,
    latest: object,
    covered_symbols: object,
    applicable_symbols: object,
    gaps: object,
    notes: str,
) -> dict[str, object]:
    gap_list = [str(item) for item in gaps] if isinstance(gaps, list) else []
    covered_list = [str(item).upper() for item in covered_symbols] if isinstance(covered_symbols, list) else []
    applicable_list = [str(item).upper() for item in applicable_symbols] if isinstance(applicable_symbols, list) else []
    return {
        "domain": domain,
        "label": label,
        "primary_source_id": primary_source_id,
        "source": source,
        "coverage": float(coverage or 0.0),
        "covered": int(covered or 0),
        "total": int(total or 0),
        "status": status,
        "latest": latest or "",
        "covered_symbols": covered_list,
        "applicable_symbols": applicable_list,
        "missing_count": len(gap_list),
        "gaps": gap_list,
        "notes": notes,
    }


def _metadata_coverage(symbols: Sequence[str], metadata: Mapping[str, SecurityMasterRecord]) -> dict[str, object]:
    available = {
        symbol: metadata[symbol].sector
        for symbol in symbols
        if symbol in metadata and metadata[symbol].sector and metadata[symbol].sector != "Unknown"
    }
    missing = [symbol for symbol in symbols if symbol not in available]
    return {
        "symbols_requested": len(symbols),
        "symbols_with_metadata": len(available),
        "symbols_with_metadata_list": sorted(available),
        "missing_symbols": missing,
        "coverage": round(len(available) / len(symbols), 4) if symbols else 0.0,
        "sector_by_symbol": available,
    }


def _gross_coverage(domain_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_domain = {str(row.get("domain")): float(row.get("coverage", 0.0) or 0.0) for row in domain_rows}
    critical_domains = ["security_master", "daily_ohlcv", "financials", "valuations", "shareholding"]
    factor_domains = ["financials", "valuations", "shareholding"]
    return {
        "critical_domains": critical_domains,
        "factor_domains": factor_domains,
        "critical_domain_average": round(_average(by_domain.get(domain, 0.0) for domain in critical_domains), 4),
        "factor_domain_average": round(_average(by_domain.get(domain, 0.0) for domain in factor_domains), 4),
        "market_data_coverage": by_domain.get("daily_ohlcv", 0.0),
        "metadata_coverage": by_domain.get("security_master", 0.0),
        "financial_statement_coverage": by_domain.get("financials", 0.0),
        "valuation_coverage": by_domain.get("valuations", 0.0),
        "shareholding_coverage": by_domain.get("shareholding", 0.0),
        "banking_applicable_coverage": by_domain.get("banking_factors", 0.0),
    }


def _source_coverage(rows: Sequence[Mapping[str, object]]) -> float:
    values = [float(row.get("coverage", 0.0) or 0.0) for row in rows if row]
    return round(_average(values), 4) if values else 0.0


def _broker_source_coverage(report: Mapping[str, object]) -> float:
    if not report:
        return 0.0
    values = []
    for key in ("quote_coverage", "historical_coverage"):
        value = report.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return round(_average(values), 4) if values else 0.0


def _finedge_export_coverage(report: Mapping[str, object]) -> float:
    if not report:
        return 0.0
    per_symbol = report.get("per_symbol")
    if not isinstance(per_symbol, Mapping):
        return 0.0
    requested = len(per_symbol)
    if not requested:
        return 0.0
    covered = 0
    for payload in per_symbol.values():
        if not isinstance(payload, Mapping):
            continue
        if any(int(payload.get(key, 0) or 0) > 0 for key in ("financial_rows", "valuation_rows", "shareholding_rows", "banking_rows")):
            covered += 1
    return round(covered / requested, 4)


def _needed_next(*, domain_rows: Sequence[Mapping[str, object]], source_rows: Sequence[Mapping[str, object]]) -> list[str]:
    by_domain = {str(row.get("domain")): row for row in domain_rows}
    items: list[str] = []
    if float(by_domain.get("financials", {}).get("coverage", 0.0) or 0.0) < 0.80:
        items.append("Buy or secure sample-expanded FinEdge coverage before broad factor ingestion; Basic is only a three-symbol sandbox.")
    if float(by_domain.get("daily_ohlcv", {}).get("coverage", 0.0) or 0.0) >= 0.95:
        items.append("Keep Zerodha/Breeze/yfinance market data as usable, but add official exchange reconciliation and corporate-action adjustments.")
    if float(by_domain.get("historical_universe", {}).get("coverage", 0.0) or 0.0) < 0.80:
        items.append("Acquire historical constituents, delistings, symbol changes, and corporate actions before trusting long-horizon backtests.")
    if float(by_domain.get("events_documents", {}).get("coverage", 0.0) or 0.0) < 0.50:
        items.append("Populate event/document feeds from NSE/BSE and FinEdge once fundamentals coverage is stable.")
    if any(str(row.get("source")) == "ICICI Breeze" and float(row.get("gross_coverage", 0.0) or 0.0) < 0.95 for row in source_rows):
        items.append("Continue Breeze quote reliability work; treat Breeze as reconciliation/historical fallback until quote coverage is consistently high.")
    items.append("Keep the data entitlement registry current whenever a source plan, symbol scope, endpoint, or licensing term changes.")
    return items


def _console_rows(report: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    for row in report.get("domains", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "kind": "domain",
                "name": row.get("label"),
                "coverage": row.get("coverage"),
                "covered": row.get("covered"),
                "total": row.get("total"),
                "status": row.get("status"),
                "latest": row.get("latest"),
                "missing": row.get("missing_count"),
            }
        )
    for row in report.get("sources", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "kind": "source",
                "name": row.get("source"),
                "coverage": row.get("gross_coverage"),
                "covered": "",
                "total": "",
                "status": row.get("status"),
                "latest": "",
                "missing": "",
            }
        )
    return rows


def _artifact_evidence(artifacts: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        name: {
            "path": payload.get("path"),
            "mtime": payload.get("mtime"),
            "pipeline": _artifact_payload(artifacts, name).get("pipeline"),
            "passed": _artifact_payload(artifacts, name).get("passed"),
        }
        for name, payload in artifacts.items()
    }


def _artifact_payload(artifacts: Mapping[str, Mapping[str, object]], name: str) -> Mapping[str, object]:
    payload = artifacts.get(name, {}).get("payload") if isinstance(artifacts.get(name), Mapping) else {}
    return payload if isinstance(payload, Mapping) else {}


def _finedge_evidence(report: Mapping[str, object], domain: Mapping[str, Mapping[str, object]]) -> str:
    if report:
        coverage = _finedge_export_coverage(report)
        issues = report.get("issues")
        issue_count = len(issues) if isinstance(issues, list) else 0
        return f"Latest FinEdge export coverage {coverage:.2f}; issues {issue_count}; Basic currently proves three-symbol sandbox coverage."
    return (
        "Canonical factor tables show financial coverage "
        f"{float(domain.get('financials', {}).get('coverage', 0.0) or 0.0):.2f}, but no latest FinEdge export report was found."
    )


def _broker_evidence(source: str, report: Mapping[str, object]) -> str:
    if not report:
        return f"No latest broker-health report found for {source}."
    return (
        f"enabled={report.get('enabled')}, quote={report.get('quote_coverage')}, "
        f"historical={report.get('historical_coverage')}, role={report.get('role')}"
    )


def _data_quality_evidence(report: Mapping[str, object]) -> str:
    coverage = report.get("coverage") if isinstance(report.get("coverage"), Mapping) else {}
    if coverage:
        return f"Latest canonical data-quality coverage {coverage.get('coverage')} across {coverage.get('symbols_requested')} symbols."
    flags = report.get("quality_flags") if isinstance(report.get("quality_flags"), Mapping) else {}
    ohlcv = flags.get("ohlcv") if isinstance(flags.get("ohlcv"), Mapping) else {}
    metrics = ohlcv.get("metrics") if isinstance(ohlcv.get("metrics"), Mapping) else {}
    if metrics:
        return f"Latest OHLCV quality metrics: coverage={metrics.get('coverage')}, rows={metrics.get('row_count')}."
    return "No latest data-quality report found."


def _status(coverage: object, *, strong: float, usable: float) -> str:
    value = float(coverage or 0.0)
    if value >= strong:
        return "strong"
    if value >= usable:
        return "usable_partial"
    if value > 0:
        return "partial"
    return "gap"


def _financial_symbols(symbols: Sequence[str], metadata: Mapping[str, SecurityMasterRecord]) -> list[str]:
    return [symbol for symbol in symbols if _is_financial_business(metadata.get(symbol))]


def _is_financial_business(record: SecurityMasterRecord | None) -> bool:
    text = " ".join([record.sector, record.industry]).lower() if record else ""
    return any(token in text for token in ("bank", "financial", "nbfc", "finance", "insurance"))


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for symbol in symbols:
        value = symbol.strip().upper()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


def _latest_from_rows(rows_by_symbol: object, field: str) -> str:
    if not isinstance(rows_by_symbol, Mapping):
        return ""
    values = [
        str(row.get(field))
        for row in rows_by_symbol.values()
        if isinstance(row, Mapping) and row.get(field)
    ]
    return max(values) if values else ""


def _latest_from_values(values: object) -> str:
    if not isinstance(values, Mapping):
        return ""
    dates = [str(value) for value in values.values() if value]
    return max(dates) if dates else ""


def _latest_matching(root: Path, filename: str) -> Path | None:
    if not root.exists():
        return None
    matches = [path for path in root.glob(f"**/{filename}") if path.is_file()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _mtime(path: Path) -> str:
    return datetime.utcfromtimestamp(path.stat().st_mtime).replace(microsecond=0).isoformat() + "Z"


def _average(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _nested(report: Mapping[str, object], key: str, subkey: str) -> object:
    nested = report.get(key)
    return nested.get(subkey) if isinstance(nested, Mapping) else ""


def _pct(value: object) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value) * 100:.1f}%"


def _short_gap(value: object, limit: int = 6) -> str:
    if not isinstance(value, list):
        return ""
    if not value:
        return "None"
    shown = ", ".join(str(item) for item in value[:limit])
    remainder = len(value) - limit
    return f"{shown} +{remainder} more" if remainder > 0 else shown
