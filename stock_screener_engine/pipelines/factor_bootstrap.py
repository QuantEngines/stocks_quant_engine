"""Bulk point-in-time factor template and ingestion workflow."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.data_sources.financials.ingestion import FinancialStatementIngestor
from stock_screener_engine.data_sources.schemas import BankingFactorRecord, EquityValuationRecord, ShareholdingRecord
from stock_screener_engine.monitoring.factor_quality import FactorQualityValidator
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.market_data_store import MarketDataStore


FINANCIAL_COLUMNS = [
    "symbol",
    "period_end",
    "filing_date",
    "statement_type",
    "revenue",
    "ebit",
    "net_income",
    "operating_cash_flow",
    "capex",
    "total_debt",
    "equity",
    "total_assets",
    "current_assets",
    "current_liabilities",
    "interest_expense",
    "source_id",
]

VALUATION_COLUMNS = [
    "symbol",
    "as_of",
    "market_cap",
    "shares_outstanding",
    "free_float_market_cap",
    "enterprise_value",
    "currency",
    "source_id",
]

SHAREHOLDING_COLUMNS = [
    "symbol",
    "period_end",
    "filing_date",
    "promoter_pct",
    "fii_pct",
    "dii_pct",
    "public_pct",
    "source_id",
]

BANKING_COLUMNS = [
    "symbol",
    "period_end",
    "filing_date",
    "net_interest_income",
    "net_interest_margin_pct",
    "advances_growth_pct",
    "deposits_growth_pct",
    "casa_ratio_pct",
    "gnpa_ratio_pct",
    "nnpa_ratio_pct",
    "provision_coverage_ratio_pct",
    "credit_cost_pct",
    "capital_adequacy_ratio_pct",
    "cet1_ratio_pct",
    "cost_to_income_ratio_pct",
    "roa_pct",
    "roe_pct",
    "loan_to_deposit_ratio_pct",
    "source_id",
]


class FactorBootstrapPipeline:
    """Create and ingest external point-in-time factor CSVs."""

    def __init__(
        self,
        settings: AppSettings,
        store: MarketDataStore | None = None,
        file_store: LocalFileStorage | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or MarketDataStore(settings.storage.sqlite_path)
        self.file_store = file_store or LocalFileStorage(settings.storage.root_dir)

    def create_templates(
        self,
        symbols: Sequence[str],
        output_root: str,
        as_of: date,
        overwrite: bool = False,
    ) -> dict[str, object]:
        root = Path(output_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        normalized_symbols = _normalize_symbols(symbols)
        files = {
            "financials": self._write_template(
                path=root / "financials.csv",
                fieldnames=FINANCIAL_COLUMNS,
                rows=[_financial_template_row(symbol) for symbol in normalized_symbols],
                overwrite=overwrite,
            ),
            "valuations": self._write_template(
                path=root / "valuations.csv",
                fieldnames=VALUATION_COLUMNS,
                rows=[_valuation_template_row(symbol, as_of) for symbol in normalized_symbols],
                overwrite=overwrite,
            ),
            "shareholding": self._write_template(
                path=root / "shareholding.csv",
                fieldnames=SHAREHOLDING_COLUMNS,
                rows=[_shareholding_template_row(symbol) for symbol in normalized_symbols],
                overwrite=overwrite,
            ),
            "banking": self._write_template(
                path=root / "banking.csv",
                fieldnames=BANKING_COLUMNS,
                rows=[_banking_template_row(symbol) for symbol in normalized_symbols],
                overwrite=overwrite,
            ),
        }
        return {
            "pipeline": "factor_bootstrap_template",
            "as_of": as_of.isoformat(),
            "output_root": str(root),
            "symbols": len(normalized_symbols),
            "files": files,
        }

    def ingest(
        self,
        symbols: Sequence[str],
        root: str,
        as_of: date,
        venue: str | None = None,
        min_coverage: float = 1.0,
        sections: Sequence[str] | None = None,
    ) -> dict[str, object]:
        factor_root = Path(root).expanduser()
        canonical_venue = (venue or self.settings.runtime_data.canonical_venue).strip().upper()
        normalized_symbols = _normalize_symbols(symbols)
        normalized_sections = _normalize_factor_sections(sections or ["financials", "valuations", "shareholding"])
        requested = set(normalized_symbols)

        financial_report = (
            self._ingest_financials(
                rows=_load_factor_rows(factor_root, "financials", "financials.csv"),
                requested_symbols=requested,
                venue=canonical_venue,
                as_of=as_of,
            )
            if "financials" in normalized_sections
            else _skipped_section_report("financials")
        )
        valuation_report = (
            self._ingest_valuations(
                rows=_load_factor_rows(factor_root, "valuations", "valuations.csv"),
                requested_symbols=requested,
                venue=canonical_venue,
                as_of=as_of,
            )
            if "valuations" in normalized_sections
            else _skipped_section_report("valuations")
        )
        shareholding_report = (
            self._ingest_shareholding(
                rows=_load_factor_rows(factor_root, "shareholding", "shareholding.csv"),
                requested_symbols=requested,
                venue=canonical_venue,
                as_of=as_of,
            )
            if "shareholding" in normalized_sections
            else _skipped_section_report("shareholding")
        )
        banking_report = (
            self._ingest_banking(
                rows=_load_factor_rows(factor_root, "banking", "banking.csv"),
                requested_symbols=requested,
                venue=canonical_venue,
                as_of=as_of,
            )
            if "banking" in normalized_sections
            else _skipped_section_report("banking")
        )

        coverage: dict[str, object] = {}
        if "financials" in normalized_sections:
            coverage["financials"] = self.store.financial_statement_coverage(normalized_symbols, as_of=as_of, venue=canonical_venue)
        if "valuations" in normalized_sections:
            coverage["valuations"] = self.store.equity_valuation_coverage(normalized_symbols, as_of=as_of, venue=canonical_venue)
        if "shareholding" in normalized_sections:
            coverage["shareholding"] = self.store.shareholding_coverage(normalized_symbols, as_of=as_of, venue=canonical_venue)
        if "banking" in normalized_sections:
            coverage["banking"] = self.store.banking_factor_coverage(normalized_symbols, as_of=as_of, venue=canonical_venue)
        coverage_values = [
            _coverage_value(cast(Mapping[str, object], coverage[section]))
            for section in normalized_sections
            if section in coverage
        ]
        section_reports = {
            "financials": financial_report,
            "valuations": valuation_report,
            "shareholding": shareholding_report,
            "banking": banking_report,
        }
        input_coverage_values = [
            float(cast(Any, section_reports[section].get("input_coverage") or 0.0))
            for section in normalized_sections
        ]
        has_input_errors = any(_section_has_error(section_reports[section]) for section in normalized_sections)
        report = {
            "pipeline": "factor_bootstrap_ingest",
            "root": str(factor_root),
            "venue": canonical_venue,
            "as_of": as_of.isoformat(),
            "symbols": len(normalized_symbols),
            "sections": normalized_sections,
            "min_coverage": float(min_coverage),
            "passed": (
                bool(coverage_values)
                and not has_input_errors
                and all(value >= float(min_coverage) for value in coverage_values)
                and all(value >= float(min_coverage) for value in input_coverage_values)
            ),
            "financials": financial_report,
            "valuations": valuation_report,
            "shareholding": shareholding_report,
            "banking": banking_report,
            "coverage": coverage,
        }
        report_path = self.file_store.save_json(
            report,
            filename="factor_bootstrap_ingest_report.json",
            subdir="quality",
        )
        report["report_path"] = str(report_path)
        return report

    def close(self) -> None:
        self.store.close()

    def _write_template(
        self,
        path: Path,
        fieldnames: list[str],
        rows: list[dict[str, object]],
        overwrite: bool,
    ) -> dict[str, object]:
        if path.exists() and not overwrite:
            return {"path": str(path), "status": "exists", "rows": 0}
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return {"path": str(path), "status": "written", "rows": len(rows)}

    def _ingest_financials(
        self,
        rows: list[dict[str, str]],
        requested_symbols: set[str],
        venue: str,
        as_of: date,
    ) -> dict[str, object]:
        grouped, skipped = _group_rows_by_symbol(rows, requested_symbols)
        accepted = []
        rejected_rows = skipped
        quality_issues: list[dict[str, object]] = []
        per_symbol: dict[str, dict[str, object]] = {}
        ingestor = FinancialStatementIngestor()
        validator = FactorQualityValidator()
        for symbol, symbol_rows in sorted(grouped.items()):
            ingested = ingestor.ingest_rows(symbol_rows, venue=venue, symbol=symbol, as_of=as_of)
            quality = validator.validate(ingested.records, as_of=as_of)
            rejected_rows += ingested.rejected_rows
            issues = [asdict(issue) for issue in quality.issues]
            quality_issues.extend(issues)
            persisted_for_symbol = 0
            if quality.passed and ingested.records:
                accepted.extend(ingested.records)
                persisted_for_symbol = len(ingested.records)
            else:
                rejected_rows += len(ingested.records)
            per_symbol[symbol] = {
                "accepted": persisted_for_symbol,
                "rejected_rows": ingested.rejected_rows + (0 if quality.passed else len(ingested.records)),
                "quality_issues": issues,
            }

        persisted = self.store.upsert_financial_statements(accepted) if accepted else 0
        report = {
            "input_rows": len(rows),
            "accepted": len(accepted),
            "rejected_rows": rejected_rows,
            "persisted": persisted,
            "quality_issues": quality_issues,
            "per_symbol": per_symbol,
        }
        _add_input_coverage(
            report,
            section="financials",
            rows=rows,
            requested_symbols=requested_symbols,
            accepted_symbols=[record.symbol for record in accepted],
        )
        return report

    def _ingest_valuations(
        self,
        rows: list[dict[str, str]],
        requested_symbols: set[str],
        venue: str,
        as_of: date,
    ) -> dict[str, object]:
        accepted: list[EquityValuationRecord] = []
        rejected_rows = 0
        issues: list[dict[str, object]] = []
        per_symbol: dict[str, dict[str, object]] = {}
        for idx, row in enumerate(rows, start=2):
            symbol = _row_symbol(row)
            if not _symbol_in_scope(symbol, requested_symbols):
                rejected_rows += 1
                continue
            per_symbol.setdefault(symbol, {"accepted": 0, "rejected_rows": 0, "quality_issues": []})
            try:
                row_as_of = _csv_date(row.get("as_of") or row.get("date"))
                if row_as_of > as_of:
                    rejected_rows += 1
                    per_symbol[symbol]["rejected_rows"] = int(per_symbol[symbol]["rejected_rows"]) + 1
                    continue
                market_cap = _csv_float(row.get("market_cap"))
                if market_cap <= 0:
                    raise ValueError("market_cap must be positive")
                accepted.append(
                    EquityValuationRecord(
                        venue=venue,
                        symbol=symbol,
                        as_of=row_as_of,
                        market_cap=market_cap,
                        shares_outstanding=_csv_float(row.get("shares_outstanding")),
                        free_float_market_cap=_csv_float(row.get("free_float_market_cap")),
                        enterprise_value=_csv_float(row.get("enterprise_value")),
                        currency=str(row.get("currency") or "INR").strip() or "INR",
                        source_id=str(row.get("source_id") or "").strip(),
                    )
                )
                per_symbol[symbol]["accepted"] = int(per_symbol[symbol]["accepted"]) + 1
            except ValueError as exc:
                rejected_rows += 1
                issue = {"row": idx, "symbol": symbol or "", "severity": "error", "message": str(exc)}
                issues.append(issue)
                if symbol:
                    per_symbol.setdefault(symbol, {"accepted": 0, "rejected_rows": 0, "quality_issues": []})
                    per_symbol[symbol]["rejected_rows"] = int(per_symbol[symbol]["rejected_rows"]) + 1
                    cast(list[dict[str, object]], per_symbol[symbol]["quality_issues"]).append(issue)

        persisted = self.store.upsert_equity_valuations(accepted) if accepted else 0
        report = {
            "input_rows": len(rows),
            "accepted": len(accepted),
            "rejected_rows": rejected_rows,
            "persisted": persisted,
            "quality_issues": issues,
            "per_symbol": per_symbol,
        }
        _add_input_coverage(
            report,
            section="valuations",
            rows=rows,
            requested_symbols=requested_symbols,
            accepted_symbols=[record.symbol for record in accepted],
        )
        return report

    def _ingest_shareholding(
        self,
        rows: list[dict[str, str]],
        requested_symbols: set[str],
        venue: str,
        as_of: date,
    ) -> dict[str, object]:
        accepted: list[ShareholdingRecord] = []
        rejected_rows = 0
        issues: list[dict[str, object]] = []
        per_symbol: dict[str, dict[str, object]] = {}
        for idx, row in enumerate(rows, start=2):
            symbol = _row_symbol(row)
            if not _symbol_in_scope(symbol, requested_symbols):
                rejected_rows += 1
                continue
            per_symbol.setdefault(symbol, {"accepted": 0, "rejected_rows": 0, "quality_issues": []})
            try:
                period_end = _csv_date(row.get("period_end"))
                filing_date = _csv_date(row.get("filing_date"))
                if period_end > as_of or filing_date > as_of:
                    rejected_rows += 1
                    per_symbol[symbol]["rejected_rows"] = int(per_symbol[symbol]["rejected_rows"]) + 1
                    continue
                promoter = _csv_float(row.get("promoter_pct"))
                fii = _csv_float(row.get("fii_pct"))
                dii = _csv_float(row.get("dii_pct"))
                public = _csv_float(row.get("public_pct"))
                if public == 0.0:
                    public = max(0.0, 100.0 - promoter - fii - dii)
                values = [promoter, fii, dii, public]
                if any(value < 0.0 or value > 100.0 for value in values):
                    raise ValueError("holding percentages must be between 0 and 100")
                if sum(values) > 101.0:
                    raise ValueError("holding percentages sum above 101")
                accepted.append(
                    ShareholdingRecord(
                        venue=venue,
                        symbol=symbol,
                        period_end=period_end,
                        filing_date=filing_date,
                        promoter_pct=promoter,
                        fii_pct=fii,
                        dii_pct=dii,
                        public_pct=public,
                        source_id=str(row.get("source_id") or "").strip(),
                    )
                )
                per_symbol[symbol]["accepted"] = int(per_symbol[symbol]["accepted"]) + 1
            except ValueError as exc:
                rejected_rows += 1
                issue = {"row": idx, "symbol": symbol or "", "severity": "error", "message": str(exc)}
                issues.append(issue)
                if symbol:
                    per_symbol.setdefault(symbol, {"accepted": 0, "rejected_rows": 0, "quality_issues": []})
                    per_symbol[symbol]["rejected_rows"] = int(per_symbol[symbol]["rejected_rows"]) + 1
                    cast(list[dict[str, object]], per_symbol[symbol]["quality_issues"]).append(issue)

        persisted = self.store.upsert_shareholding(accepted) if accepted else 0
        report = {
            "input_rows": len(rows),
            "accepted": len(accepted),
            "rejected_rows": rejected_rows,
            "persisted": persisted,
            "quality_issues": issues,
            "per_symbol": per_symbol,
        }
        _add_input_coverage(
            report,
            section="shareholding",
            rows=rows,
            requested_symbols=requested_symbols,
            accepted_symbols=[record.symbol for record in accepted],
        )
        return report

    def _ingest_banking(
        self,
        rows: list[dict[str, str]],
        requested_symbols: set[str],
        venue: str,
        as_of: date,
    ) -> dict[str, object]:
        accepted: list[BankingFactorRecord] = []
        rejected_rows = 0
        issues: list[dict[str, object]] = []
        per_symbol: dict[str, dict[str, object]] = {}
        for idx, row in enumerate(rows, start=2):
            symbol = _row_symbol(row)
            if not _symbol_in_scope(symbol, requested_symbols):
                rejected_rows += 1
                continue
            per_symbol.setdefault(symbol, {"accepted": 0, "rejected_rows": 0, "quality_issues": []})
            try:
                period_end = _csv_date(row.get("period_end"))
                filing_date = _csv_date(row.get("filing_date"))
                if period_end > as_of or filing_date > as_of:
                    rejected_rows += 1
                    per_symbol[symbol]["rejected_rows"] = int(per_symbol[symbol]["rejected_rows"]) + 1
                    continue
                record = BankingFactorRecord(
                    venue=venue,
                    symbol=symbol,
                    period_end=period_end,
                    filing_date=filing_date,
                    net_interest_income=_csv_float(row.get("net_interest_income")),
                    net_interest_margin_pct=_csv_float(row.get("net_interest_margin_pct")),
                    advances_growth_pct=_csv_float(row.get("advances_growth_pct")),
                    deposits_growth_pct=_csv_float(row.get("deposits_growth_pct")),
                    casa_ratio_pct=_csv_float(row.get("casa_ratio_pct")),
                    gnpa_ratio_pct=_csv_float(row.get("gnpa_ratio_pct")),
                    nnpa_ratio_pct=_csv_float(row.get("nnpa_ratio_pct")),
                    provision_coverage_ratio_pct=_csv_float(row.get("provision_coverage_ratio_pct")),
                    credit_cost_pct=_csv_float(row.get("credit_cost_pct")),
                    capital_adequacy_ratio_pct=_csv_float(row.get("capital_adequacy_ratio_pct")),
                    cet1_ratio_pct=_csv_float(row.get("cet1_ratio_pct")),
                    cost_to_income_ratio_pct=_csv_float(row.get("cost_to_income_ratio_pct")),
                    roa_pct=_csv_float(row.get("roa_pct")),
                    roe_pct=_csv_float(row.get("roe_pct")),
                    loan_to_deposit_ratio_pct=_csv_float(row.get("loan_to_deposit_ratio_pct")),
                    source_id=str(row.get("source_id") or "").strip(),
                )
                row_issues = _banking_factor_issues(record)
                if any(issue["severity"] == "error" for issue in row_issues):
                    rejected_rows += 1
                    per_symbol[symbol]["rejected_rows"] = int(per_symbol[symbol]["rejected_rows"]) + 1
                    issues.extend([{**issue, "row": idx, "symbol": symbol} for issue in row_issues])
                    cast(list[dict[str, object]], per_symbol[symbol]["quality_issues"]).extend(row_issues)
                    continue
                accepted.append(record)
                per_symbol[symbol]["accepted"] = int(per_symbol[symbol]["accepted"]) + 1
                warning_issues = [{**issue, "row": idx, "symbol": symbol} for issue in row_issues]
                issues.extend(warning_issues)
                cast(list[dict[str, object]], per_symbol[symbol]["quality_issues"]).extend(warning_issues)
            except ValueError as exc:
                rejected_rows += 1
                issue = {"row": idx, "symbol": symbol or "", "severity": "error", "message": str(exc)}
                issues.append(issue)
                if symbol:
                    per_symbol.setdefault(symbol, {"accepted": 0, "rejected_rows": 0, "quality_issues": []})
                    per_symbol[symbol]["rejected_rows"] = int(per_symbol[symbol]["rejected_rows"]) + 1
                    cast(list[dict[str, object]], per_symbol[symbol]["quality_issues"]).append(issue)

        persisted = self.store.upsert_banking_factors(accepted) if accepted else 0
        report = {
            "input_rows": len(rows),
            "accepted": len(accepted),
            "rejected_rows": rejected_rows,
            "persisted": persisted,
            "quality_issues": issues,
            "per_symbol": per_symbol,
        }
        _add_input_coverage(
            report,
            section="banking",
            rows=rows,
            requested_symbols=requested_symbols,
            accepted_symbols=[record.symbol for record in accepted],
        )
        return report


def _load_factor_rows(root: Path, subdir: str, combined_filename: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    combined = root / combined_filename
    if combined.exists():
        rows.extend(_read_csv(combined))

    folder = root / subdir
    if folder.exists():
        for path in sorted(folder.glob("*.csv")):
            for row in _read_csv(path):
                row.setdefault("symbol", path.stem.upper())
                if not str(row.get("symbol") or "").strip():
                    row["symbol"] = path.stem.upper()
                rows.append(row)
    return rows


def _normalize_factor_sections(sections: Sequence[str]) -> list[str]:
    out: list[str] = []
    for section in sections:
        value = section.strip().lower().replace("-", "_")
        if not value:
            continue
        if value in {"all", "both"}:
            values = ["financials", "valuations", "shareholding", "banking"]
        elif value in {"valuation", "valuations"}:
            values = ["valuations"]
        elif value in {"financials", "shareholding", "banking", "banking_factors", "financial_sector_factors"}:
            value = "banking" if value in {"banking_factors", "financial_sector_factors"} else value
            values = [value]
        else:
            raise ValueError(f"unsupported factor section '{section}'")
        for item in values:
            if item not in out:
                out.append(item)
    return out or ["financials", "valuations", "shareholding"]


def _skipped_section_report(section: str) -> dict[str, object]:
    return {
        "input_rows": 0,
        "accepted": 0,
        "rejected_rows": 0,
        "persisted": 0,
        "quality_issues": [],
        "skipped": True,
        "section": section,
    }


def _add_input_coverage(
    report: dict[str, object],
    *,
    section: str,
    rows: list[dict[str, str]],
    requested_symbols: set[str],
    accepted_symbols: Sequence[str],
) -> None:
    scoped_input_symbols = {
        symbol
        for row in rows
        if (symbol := _row_symbol(row)) and _symbol_in_scope(symbol, requested_symbols)
    }
    accepted = {symbol.strip().upper() for symbol in accepted_symbols if symbol.strip()}
    missing = sorted(requested_symbols - accepted)
    denominator = len(requested_symbols)
    report["input_symbols"] = sorted(scoped_input_symbols)
    report["accepted_symbols"] = sorted(accepted)
    report["input_missing_symbols"] = missing
    report["input_coverage"] = round(len(accepted) / denominator, 4) if denominator else 0.0
    issues = cast(list[dict[str, object]], report.setdefault("quality_issues", []))
    if requested_symbols and not rows:
        issues.append(
            {
                "section": section,
                "severity": "error",
                "message": (
                    f"selected factor section '{section}' has no input rows; "
                    f"provide {section}.csv or omit it with --sections"
                ),
            }
        )
    elif requested_symbols and rows and not accepted:
        issues.append(
            {
                "section": section,
                "severity": "error",
                "message": f"selected factor section '{section}' produced no accepted rows",
            }
        )


def _section_has_error(report: Mapping[str, object]) -> bool:
    issues = report.get("quality_issues")
    if not isinstance(issues, list):
        return False
    return any(isinstance(issue, Mapping) and issue.get("severity") == "error" for issue in issues)


def _coverage_value(report: Mapping[str, object]) -> float:
    return float(cast(Any, report.get("coverage") or 0.0))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return [
            {str(key or "").strip(): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(fh)
        ]


def _group_rows_by_symbol(
    rows: list[dict[str, str]],
    requested_symbols: set[str],
) -> tuple[dict[str, list[dict[str, str]]], int]:
    grouped: dict[str, list[dict[str, str]]] = {}
    skipped = 0
    for row in rows:
        symbol = _row_symbol(row)
        if not _symbol_in_scope(symbol, requested_symbols):
            skipped += 1
            continue
        grouped.setdefault(symbol, []).append(row)
    return grouped, skipped


def _row_symbol(row: dict[str, str]) -> str:
    return str(
        row.get("symbol")
        or row.get("Symbol")
        or row.get("SYMBOL")
        or row.get("tradingsymbol")
        or row.get("ticker")
        or ""
    ).strip().upper()


def _symbol_in_scope(symbol: str, requested_symbols: set[str]) -> bool:
    if not symbol:
        return False
    return not requested_symbols or symbol in requested_symbols


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for symbol in symbols:
        normalized = symbol.strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _financial_template_row(symbol: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "period_end": "",
        "filing_date": "",
        "statement_type": "quarterly",
        "revenue": "",
        "ebit": "",
        "net_income": "",
        "operating_cash_flow": "",
        "capex": "",
        "total_debt": "",
        "equity": "",
        "total_assets": "",
        "current_assets": "",
        "current_liabilities": "",
        "interest_expense": "",
        "source_id": "",
    }


def _valuation_template_row(symbol: str, as_of: date) -> dict[str, object]:
    return {
        "symbol": symbol,
        "as_of": as_of.isoformat(),
        "market_cap": "",
        "shares_outstanding": "",
        "free_float_market_cap": "",
        "enterprise_value": "",
        "currency": "INR",
        "source_id": "",
    }


def _shareholding_template_row(symbol: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "period_end": "",
        "filing_date": "",
        "promoter_pct": "",
        "fii_pct": "",
        "dii_pct": "",
        "public_pct": "",
        "source_id": "",
    }


def _banking_template_row(symbol: str) -> dict[str, object]:
    return {
        "symbol": symbol,
        "period_end": "",
        "filing_date": "",
        "net_interest_income": "",
        "net_interest_margin_pct": "",
        "advances_growth_pct": "",
        "deposits_growth_pct": "",
        "casa_ratio_pct": "",
        "gnpa_ratio_pct": "",
        "nnpa_ratio_pct": "",
        "provision_coverage_ratio_pct": "",
        "credit_cost_pct": "",
        "capital_adequacy_ratio_pct": "",
        "cet1_ratio_pct": "",
        "cost_to_income_ratio_pct": "",
        "roa_pct": "",
        "roe_pct": "",
        "loan_to_deposit_ratio_pct": "",
        "source_id": "",
    }


def _banking_factor_issues(record: BankingFactorRecord) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    populated = [
        record.net_interest_income,
        record.net_interest_margin_pct,
        record.advances_growth_pct,
        record.deposits_growth_pct,
        record.casa_ratio_pct,
        record.gnpa_ratio_pct,
        record.nnpa_ratio_pct,
        record.provision_coverage_ratio_pct,
        record.credit_cost_pct,
        record.capital_adequacy_ratio_pct,
        record.cet1_ratio_pct,
        record.cost_to_income_ratio_pct,
        record.roa_pct,
        record.roe_pct,
        record.loan_to_deposit_ratio_pct,
    ]
    if not any(abs(value) > 1e-9 for value in populated):
        issues.append({"severity": "error", "message": "banking factor row has no populated metrics"})
    elif sum(1 for value in populated if abs(value) > 1e-9) / len(populated) < 0.5:
        issues.append({"severity": "warning", "message": "banking factor row has sparse metric coverage; review vendor field mapping"})
    for field in (
        "net_interest_margin_pct",
        "casa_ratio_pct",
        "gnpa_ratio_pct",
        "nnpa_ratio_pct",
        "provision_coverage_ratio_pct",
        "credit_cost_pct",
        "capital_adequacy_ratio_pct",
        "cet1_ratio_pct",
        "cost_to_income_ratio_pct",
        "roa_pct",
        "roe_pct",
        "loan_to_deposit_ratio_pct",
    ):
        value = float(getattr(record, field))
        if value < 0.0 or value > 200.0:
            issues.append({"severity": "error", "message": f"{field} must be between 0 and 200"})
    for field in ("advances_growth_pct", "deposits_growth_pct"):
        value = float(getattr(record, field))
        if value < -100.0 or value > 500.0:
            issues.append({"severity": "error", "message": f"{field} is outside expected growth bounds"})
    if record.nnpa_ratio_pct > record.gnpa_ratio_pct and record.gnpa_ratio_pct > 0:
        issues.append({"severity": "warning", "message": "NNPA ratio is above GNPA ratio; review source mapping"})
    if 0.0 < record.capital_adequacy_ratio_pct < 10.5:
        issues.append({"severity": "warning", "message": "capital adequacy ratio is below 10.5%"})
    if record.gnpa_ratio_pct > 8.0:
        issues.append({"severity": "warning", "message": "GNPA ratio is above 8%"})
    return issues


def _csv_date(value: object) -> date:
    text = str(value or "").strip()
    if not text:
        raise ValueError("date is required")
    return date.fromisoformat(text)


def _csv_float(value: object, default: float = 0.0) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return default
    return float(text)
