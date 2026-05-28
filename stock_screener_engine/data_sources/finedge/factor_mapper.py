"""Map FinEdge payloads into local factor bootstrap CSVs.

The mapper is intentionally conservative: it exports normalized factor rows for
review and downstream ``factor-ingest`` instead of writing raw vendor payloads or
silently mutating canonical tables.
"""

from __future__ import annotations

import csv
import json
from calendar import monthrange
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from stock_screener_engine.data_sources.finedge.client import FinEdgeClient, JSONPayload
from stock_screener_engine.pipelines.factor_bootstrap import BANKING_COLUMNS, FINANCIAL_COLUMNS, SHAREHOLDING_COLUMNS, VALUATION_COLUMNS


FINANCIAL_STATEMENT_CODES = ("pl", "bs", "cf")

_REVENUE_ALIASES = [
    "revenue",
    "revenueFromOperations",
    "revenueFromSaleOfProducts",
    "revenueFromSaleOfProduct",
    "saleOfProducts",
    "sales",
    "netSales",
    "totalRevenue",
    "operatingRevenue",
    "income",
    "totalIncome",
    "interestEarned",
]
_EBIT_ALIASES = [
    "ebit",
    "earningsBeforeInterestAndTax",
    "operatingProfit",
    "profitBeforeTax",
    "profitLossBeforeTax",
    "profitBeforeExceptionalItemsAndTax",
]
_NET_INCOME_ALIASES = [
    "netIncome",
    "netProfit",
    "profitAfterTax",
    "pat",
    "profitLossForPeriod",
    "profitLossForThePeriod",
    "profitForPeriod",
    "profitLoss",
]
_OCF_ALIASES = [
    "operatingCashFlow",
    "cashFlowsFromOperatingActivities",
    "netCashFlowsFromUsedInOperatingActivities",
    "cashFlowsFromUsedInOperatingActivities",
    "netCashFromOperatingActivities",
    "cashFlowFromOperatingActivities",
    "netCashGeneratedFromOperatingActivities",
]
_CAPEX_ALIASES = [
    "capex",
    "capitalExpenditure",
    "purchaseOfFixed&IntangibleAssets",
    "purchaseOfPPEClassifiedAsInvesting",
    "purchaseOfTangibleAssetsClassifiedAsInvesting",
    "purchaseOfIntangibleAssetsClassifiedAsInvesting",
    "purchaseOfPropertyPlantAndEquipment",
    "purchaseOfPropertyPlantAndEquipmentAndIntangibleAssets",
    "paymentsToAcquirePropertyPlantAndEquipment",
    "paymentsToAcquirePropertyPlantAndEquipmentAndIntangibleAssets",
]
_TOTAL_DEBT_ALIASES = ["totalDebt", "borrowings", "totalBorrowings", "debt"]
_DEBT_COMPONENT_ALIASES = [
    "borrowingsCurrent",
    "borrowingsNoncurrent",
    "debtSecurities",
    "currentDebt",
    "nonCurrentDebt",
    "leaseLiabilitiesCurrent",
    "leaseLiabilitiesNoncurrent",
]
_EQUITY_ALIASES = ["equity", "totalEquity", "totalShareholdersEquity", "shareholdersFunds"]
_EQUITY_COMPONENT_ALIASES = ["equityShareCapital", "otherEquity", "reservesAndSurplus", "capital", "reserves"]
_TOTAL_ASSET_ALIASES = ["assets", "totalAssets"]
_CURRENT_ASSET_ALIASES = ["currentAssets"]
_CURRENT_LIABILITY_ALIASES = ["currentLiabilities"]
_INTEREST_ALIASES = ["interestExpense", "interestExpended", "financeCosts", "financeCost", "interestAndFinanceCharges"]
_INTEREST_INCOME_ALIASES = ["interestIncome", "interestEarned", "interestAndDiscountOnAdvancesBills", "interestOnAdvances", "interestOnInvestments"]
_BANK_NII_ALIASES = ["netInterestIncome", "netInterestRevenue", "interestEarnedNet", "interestIncomeNet", "netInterestEarned"]
_BANK_ADVANCES_ALIASES = ["advances", "grossAdvances", "netAdvances", "loans", "netLoans", "loansAndAdvances", "customerAdvances"]
_BANK_DEPOSITS_ALIASES = ["deposits", "totalDeposits", "customerDeposits"]
_BANK_NIM_ALIASES = ["netInterestMargin", "nim", "netInterestMarginPct"]
_BANK_CASA_ALIASES = ["casaRatio", "casa", "casaPct"]
_BANK_GNPA_ALIASES = ["gnpaRatio", "grossNpaRatio", "grossNonPerformingAssetsRatio", "percentageOfGrossNpa"]
_BANK_NNPA_ALIASES = ["nnpaRatio", "netNpaRatio", "netNonPerformingAssetsRatio", "percentageOfNpa"]
_BANK_PCR_ALIASES = ["provisionCoverageRatio", "pcr"]
_BANK_CREDIT_COST_ALIASES = ["creditCost", "creditCostRatio"]
_BANK_CAR_ALIASES = ["capitalAdequacyRatio", "capitalAdequacy", "car"]
_BANK_CET1_ALIASES = ["cet1Ratio", "cet1"]
_BANK_COST_INCOME_ALIASES = ["costToIncomeRatio", "costIncomeRatio"]
_BANK_ROA_ALIASES = ["roa", "returnOnAsset", "returnOnAssets"]
_BANK_ROE_ALIASES = ["roe", "returnOnEquity"]
_BANK_LDR_ALIASES = ["loanToDepositRatio", "creditDepositRatio", "loanDepositRatio", "cdRatio"]


@dataclass
class FinEdgeFactorExport:
    financial_rows: list[dict[str, object]] = field(default_factory=list)
    valuation_rows: list[dict[str, object]] = field(default_factory=list)
    shareholding_rows: list[dict[str, object]] = field(default_factory=list)
    banking_rows: list[dict[str, object]] = field(default_factory=list)
    ownership_detail_rows: list[dict[str, object]] = field(default_factory=list)
    issues: list[dict[str, object]] = field(default_factory=list)
    per_symbol: dict[str, dict[str, object]] = field(default_factory=dict)


class FinEdgeFactorMapper:
    """Export FinEdge financial and ownership data to factor CSV shape."""

    def __init__(
        self,
        client: FinEdgeClient,
        *,
        venue: str = "NSE",
        statement_type: str = "s",
        period: str = "annual",
        shareholding_period: str = "quarterly",
    ) -> None:
        self.client = client
        self.venue = venue.strip().upper() or "NSE"
        self.statement_type = statement_type.strip().lower() or "s"
        self.period = period.strip().lower() or "annual"
        self.shareholding_period = shareholding_period.strip().lower() or "quarterly"

    def export(
        self,
        *,
        symbols: Sequence[str],
        as_of: date,
        output_root: str,
        sections: Sequence[str] = ("financials", "valuations", "shareholding"),
    ) -> dict[str, object]:
        normalized_symbols = _normalize_symbols(symbols)
        normalized_sections = _normalize_sections(sections)
        root = Path(output_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        result = FinEdgeFactorExport()
        for symbol in normalized_symbols:
            result.per_symbol[symbol] = {"financial_rows": 0, "valuation_rows": 0, "shareholding_rows": 0, "banking_rows": 0, "ownership_detail_rows": 0, "issues": []}
            if "financials" in normalized_sections:
                try:
                    rows = self.map_financials(symbol=symbol, as_of=as_of)
                    result.financial_rows.extend(rows)
                    result.per_symbol[symbol]["financial_rows"] = len(rows)
                except Exception as exc:  # noqa: BLE001 - mapper should continue per symbol.
                    _add_issue(result, symbol, "financials", str(exc))
            if "valuations" in normalized_sections:
                try:
                    rows = self.map_valuation(symbol=symbol, as_of=as_of)
                    result.valuation_rows.extend(rows)
                    result.per_symbol[symbol]["valuation_rows"] = len(rows)
                except Exception as exc:  # noqa: BLE001 - mapper should continue per symbol.
                    _add_issue(result, symbol, "valuations", str(exc))
            if "shareholding" in normalized_sections:
                try:
                    mapped = self.map_shareholding(symbol=symbol, as_of=as_of)
                    result.shareholding_rows.extend(mapped["shareholding_rows"])
                    result.ownership_detail_rows.extend(mapped["ownership_detail_rows"])
                    result.per_symbol[symbol]["shareholding_rows"] = len(mapped["shareholding_rows"])
                    result.per_symbol[symbol]["ownership_detail_rows"] = len(mapped["ownership_detail_rows"])
                    for issue in mapped["issues"]:
                        _add_issue(result, symbol, "shareholding", str(issue))
                except Exception as exc:  # noqa: BLE001 - mapper should continue per symbol.
                    _add_issue(result, symbol, "shareholding", str(exc))
            if "banking" in normalized_sections:
                try:
                    rows = self.map_banking_factors(symbol=symbol, as_of=as_of)
                    result.banking_rows.extend(rows)
                    result.per_symbol[symbol]["banking_rows"] = len(rows)
                except Exception as exc:  # noqa: BLE001 - mapper should continue per symbol.
                    _add_issue(result, symbol, "banking", str(exc))

        files: dict[str, dict[str, object]] = {}
        if "financials" in normalized_sections:
            files["financials"] = _write_csv(root / "financials.csv", FINANCIAL_COLUMNS, result.financial_rows)
        if "valuations" in normalized_sections:
            files["valuations"] = _write_csv(root / "valuations.csv", VALUATION_COLUMNS, result.valuation_rows)
        if "shareholding" in normalized_sections:
            files["shareholding"] = _write_csv(root / "shareholding.csv", SHAREHOLDING_COLUMNS, result.shareholding_rows)
            files["ownership_details"] = _write_csv(
                root / "finedge_ownership_details.csv",
                _ownership_detail_columns(result.ownership_detail_rows),
                result.ownership_detail_rows,
            )
        if "banking" in normalized_sections:
            files["banking"] = _write_csv(root / "banking.csv", BANKING_COLUMNS, result.banking_rows)

        report = {
            "pipeline": "finedge_factor_export",
            "run_at": datetime.utcnow().isoformat() + "Z",
            "as_of": as_of.isoformat(),
            "venue": self.venue,
            "source": "finedge",
            "statement_type": self.statement_type,
            "period": self.period,
            "shareholding_period": self.shareholding_period,
            "sections": normalized_sections,
            "symbols_requested": len(normalized_symbols),
            "output_root": str(root),
            "passed": bool(result.financial_rows or result.valuation_rows or result.shareholding_rows or result.banking_rows),
            "row_counts": {
                "financials": len(result.financial_rows),
                "valuations": len(result.valuation_rows),
                "shareholding": len(result.shareholding_rows),
                "banking": len(result.banking_rows),
                "ownership_details": len(result.ownership_detail_rows),
            },
            "files": files,
            "issues": result.issues,
            "per_symbol": result.per_symbol,
            "notes": [
                "No raw FinEdge payloads are persisted by this mapper.",
                "Review exported CSVs before running factor-ingest into canonical storage.",
                "Pledge and detailed ownership fields are exported to finedge_ownership_details.csv until canonical schema is extended.",
            ],
        }
        report_path = root / "finedge_factor_export_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    def map_financials(self, *, symbol: str, as_of: date) -> list[dict[str, object]]:
        payloads = {
            code: self.client.financials(
                symbol,
                statement_type=self.statement_type,
                statement_code=code,
                period=self.period,
            )
            for code in FINANCIAL_STATEMENT_CODES
        }
        statement_rows = {code: _financial_records(payloads[code]) for code in FINANCIAL_STATEMENT_CODES}
        period_keys = _period_keys(statement_rows)
        rows = []
        for key in period_keys:
            pl = statement_rows["pl"].get(key, {})
            bs = statement_rows["bs"].get(key, {})
            cf = statement_rows["cf"].get(key, {})
            period_end = _best_period_end(pl, bs, cf)
            if period_end is None or period_end > as_of:
                continue
            filing_date = _best_filing_date(pl, bs, cf) or period_end
            if filing_date > as_of:
                continue
            row = {
                "symbol": symbol,
                "period_end": period_end.isoformat(),
                "filing_date": filing_date.isoformat(),
                "statement_type": _canonical_statement_type(self.period, self.statement_type),
                "revenue": _first_number(pl, _REVENUE_ALIASES),
                "ebit": _derived_ebit(pl),
                "net_income": _first_number(pl, _NET_INCOME_ALIASES),
                "operating_cash_flow": _first_number(cf, _OCF_ALIASES),
                "capex": abs(_first_number(cf, _CAPEX_ALIASES)),
                "total_debt": _first_number(bs, _TOTAL_DEBT_ALIASES) or _sum_numbers(bs, _DEBT_COMPONENT_ALIASES),
                "equity": _first_number(bs, _EQUITY_ALIASES) or _sum_numbers(bs, _EQUITY_COMPONENT_ALIASES),
                "total_assets": _first_number(bs, _TOTAL_ASSET_ALIASES),
                "current_assets": _first_number(bs, _CURRENT_ASSET_ALIASES),
                "current_liabilities": _first_number(bs, _CURRENT_LIABILITY_ALIASES),
                "interest_expense": abs(_first_number(pl, _INTEREST_ALIASES)),
                "source_id": f"finedge:{symbol}:{self.statement_type}:{self.period}:{period_end.isoformat()}",
            }
            rows.append(row)
        rows.sort(key=lambda row: str(row["period_end"]), reverse=True)
        return rows

    def map_valuation(self, *, symbol: str, as_of: date) -> list[dict[str, object]]:
        payload = self.client.quote([symbol])
        quote = payload.get(symbol) if isinstance(payload, Mapping) else None
        if not isinstance(quote, Mapping):
            return []
        quote_as_of = _quote_as_of(quote) or as_of
        if quote_as_of > as_of:
            return []
        market_cap = _market_cap_rupees(quote)
        if market_cap <= 0:
            return []
        shares = _first_number(quote, ["shares", "sharesOutstanding"])
        return [
            {
                "symbol": symbol,
                "as_of": quote_as_of.isoformat(),
                "market_cap": market_cap,
                "shares_outstanding": shares,
                "free_float_market_cap": 0.0,
                "enterprise_value": 0.0,
                "currency": "INR",
                "source_id": f"finedge:{symbol}:quote:{quote_as_of.isoformat()}",
            }
        ]

    def map_banking_factors(self, *, symbol: str, as_of: date) -> list[dict[str, object]]:
        payloads = {
            code: self.client.financials(
                symbol,
                statement_type=self.statement_type,
                statement_code=code,
                period=self.period,
            )
            for code in ("pl", "bs")
        }
        ratio_payload = self.client.ratios(symbol, statement_type=self.statement_type, ratio_type="pr")
        basic_pl_payload = self.client.basic_financials(
            symbol,
            statement_type=self.statement_type,
            statement_code="pl",
        )
        basic_bs_payload = self.client.basic_financials(
            symbol,
            statement_type=self.statement_type,
            statement_code="bs",
        )
        statement_rows = {
            "pl": _financial_records(payloads["pl"]),
            "bs": _financial_records(payloads["bs"]),
            "ratios": _records_by_key(ratio_payload, "ratios"),
            "basic_pl": _records_by_key(basic_pl_payload, "ratios"),
            "basic_bs": _records_by_key(basic_bs_payload, "ratios"),
        }
        period_keys = _period_keys(statement_rows)
        base_rows: list[
            tuple[
                str,
                date,
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, Any],
                Mapping[str, Any],
            ]
        ] = []
        for key in period_keys:
            pl = statement_rows["pl"].get(key, {})
            bs = statement_rows["bs"].get(key, {})
            ratios = statement_rows["ratios"].get(key, {})
            basic_pl = statement_rows["basic_pl"].get(key, {})
            basic_bs = statement_rows["basic_bs"].get(key, {})
            period_end = _best_period_end(pl, bs, ratios, basic_pl, basic_bs)
            if period_end is None or period_end > as_of:
                continue
            base_rows.append((key, period_end, pl, bs, ratios, basic_pl, basic_bs))

        by_period = {
            period_end: (pl, bs, ratios, basic_pl, basic_bs)
            for _, period_end, pl, bs, ratios, basic_pl, basic_bs in base_rows
        }
        sorted_periods = sorted(by_period)
        rows: list[dict[str, object]] = []
        for period_end in sorted_periods:
            pl, bs, ratios, basic_pl, basic_bs = by_period[period_end]
            filing_date = _best_filing_date(pl, bs, ratios, basic_pl, basic_bs) or period_end
            if filing_date > as_of:
                continue
            previous_period = _previous_period(sorted_periods, period_end)
            previous_rows = by_period.get(previous_period, ({}, {}, {}, {}, {})) if previous_period is not None else ({}, {}, {}, {}, {})
            previous_bs = previous_rows[1]
            previous_basic_bs = previous_rows[4]
            advances = _first_number_from_rows((bs, basic_bs), _BANK_ADVANCES_ALIASES)
            deposits = _first_number(bs, _BANK_DEPOSITS_ALIASES)
            operating_expenses = _first_number(basic_pl, ["operatingExpenses"])
            operating_revenue = _first_number(basic_pl, ["operatingRevenue"])
            row = {
                "symbol": symbol,
                "period_end": period_end.isoformat(),
                "filing_date": filing_date.isoformat(),
                "net_interest_income": _bank_net_interest_income(pl),
                "net_interest_margin_pct": _percent_value(_first_number_from_rows((ratios, pl), _BANK_NIM_ALIASES)),
                "advances_growth_pct": _growth_pct(
                    advances,
                    _first_number_from_rows((previous_bs, previous_basic_bs), _BANK_ADVANCES_ALIASES),
                ),
                "deposits_growth_pct": _growth_pct(deposits, _first_number(previous_bs, _BANK_DEPOSITS_ALIASES)),
                "casa_ratio_pct": _percent_value(_first_number(bs, _BANK_CASA_ALIASES)),
                "gnpa_ratio_pct": _percent_value(_first_number_from_rows((basic_pl, bs), _BANK_GNPA_ALIASES)),
                "nnpa_ratio_pct": _percent_value(_first_number_from_rows((basic_pl, bs), _BANK_NNPA_ALIASES)),
                "provision_coverage_ratio_pct": _percent_value(_first_number_from_rows((basic_pl, bs), _BANK_PCR_ALIASES)),
                "credit_cost_pct": _percent_value(_first_number_from_rows((basic_pl, pl), _BANK_CREDIT_COST_ALIASES)),
                "capital_adequacy_ratio_pct": _percent_value(_first_number_from_rows((basic_pl, basic_bs, bs), _BANK_CAR_ALIASES)),
                "cet1_ratio_pct": _percent_value(_first_number_from_rows((basic_pl, basic_bs, bs), _BANK_CET1_ALIASES)),
                "cost_to_income_ratio_pct": _percent_value(_first_number_from_rows((ratios, basic_pl, pl), _BANK_COST_INCOME_ALIASES) or _safe_ratio_pct(operating_expenses, operating_revenue)),
                "roa_pct": _percent_value(_first_number_from_rows((ratios, basic_pl, pl), _BANK_ROA_ALIASES)),
                "roe_pct": _percent_value(_first_number_from_rows((ratios, basic_pl, pl), _BANK_ROE_ALIASES)),
                "loan_to_deposit_ratio_pct": _percent_value(_first_number_from_rows((basic_bs, bs), _BANK_LDR_ALIASES) or _safe_ratio_pct(advances, deposits)),
                "source_id": f"finedge:{symbol}:banking:{self.statement_type}:{self.period}:{period_end.isoformat()}",
            }
            if any(float(row.get(key) or 0.0) != 0.0 for key in BANKING_COLUMNS if key not in {"symbol", "period_end", "filing_date", "source_id"}):
                rows.append(row)
        rows.sort(key=lambda row: str(row["period_end"]), reverse=True)
        return rows

    def map_shareholding(self, *, symbol: str, as_of: date) -> dict[str, Any]:
        pattern_payload = self.client.shareholding_pattern(symbol, period=self.shareholding_period)
        history_payload = self.client.ownership_history(symbol, period=self.shareholding_period)
        period_label = _latest_pattern_period(pattern_payload)
        period_end = _parse_date_like(period_label) or _shareholding_period_end(pattern_payload, history_payload) or as_of
        filing_date = period_end if period_end <= as_of else as_of
        buckets = _shareholding_buckets_from_pattern(pattern_payload, period_label)
        shareholding_row = {
            "symbol": symbol,
            "period_end": period_end.isoformat(),
            "filing_date": filing_date.isoformat(),
            "promoter_pct": _round_pct(buckets["promoter"]),
            "fii_pct": _round_pct(buckets["fii"]),
            "dii_pct": _round_pct(buckets["dii"]),
            "public_pct": _round_pct(_public_pct(buckets)),
            "source_id": f"finedge:{symbol}:shareholding:{self.shareholding_period}:{period_end.isoformat()}",
        }
        issues: list[str] = []
        if period_end > as_of:
            issues.append(f"shareholding period_end {period_end.isoformat()} is after as_of {as_of.isoformat()}; filing_date clamped")
        total = sum(float(shareholding_row[key]) for key in ("promoter_pct", "fii_pct", "dii_pct", "public_pct"))
        if total > 101.0:
            issues.append(f"shareholding percentages sum to {total:.2f}; review category mapping")
        current_rows = _latest_ownership_history_rows(history_payload)
        detail_rows = [
            _ownership_detail_row(symbol=symbol, period_end=period_end, source="ownership_history", row=row)
            for row in current_rows
        ]
        return {
            "shareholding_rows": [shareholding_row],
            "ownership_detail_rows": detail_rows,
            "issues": issues,
        }


def _financial_records(payload: JSONPayload) -> dict[str, Mapping[str, Any]]:
    return _records_by_key(payload, "financials")


def _records_by_key(payload: JSONPayload, payload_key: str) -> dict[str, Mapping[str, Any]]:
    records = _as_record_list(payload, payload_key)
    out: dict[str, Mapping[str, Any]] = {}
    for idx, row in enumerate(records):
        period_key = _period_key(row) or f"row:{idx}"
        out[period_key] = row
    return out


def _period_keys(statement_rows: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[str]:
    keys = set()
    for rows in statement_rows.values():
        keys.update(rows.keys())
    return sorted(keys, reverse=True)


def _period_key(row: Mapping[str, Any]) -> str:
    period_end = _best_period_end(row)
    if period_end is not None:
        return period_end.isoformat()
    for key in ("year", "period", "quarter"):
        value = row.get(key)
        if value:
            return str(value).strip()
    return ""


def _best_period_end(*rows: Mapping[str, Any]) -> date | None:
    for row in rows:
        for key in ("period_end", "periodEnd", "period", "quarter", "year", "date"):
            parsed = _parse_date_like(row.get(key))
            if parsed is not None:
                return parsed
    return None


def _best_filing_date(*rows: Mapping[str, Any]) -> date | None:
    for row in rows:
        for key in ("filing_date", "filingDate", "result_date", "resultDate", "published_at", "publishedAt"):
            parsed = _parse_date_like(row.get(key))
            if parsed is not None:
                return parsed
    return None


def _quote_as_of(quote: Mapping[str, Any]) -> date | None:
    value = quote.get("tradetime") or quote.get("trade_time") or quote.get("as_of") or quote.get("date")
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return _parse_date_like(value)


def _market_cap_rupees(quote: Mapping[str, Any]) -> float:
    raw_market_cap = _first_number(quote, ["market_cap", "marketCap"])
    if raw_market_cap <= 0:
        return 0.0
    shares = _first_number(quote, ["shares", "sharesOutstanding"])
    price = _first_number(quote, ["current_price", "currentPrice", "price", "lastPrice"])
    implied = shares * price
    if implied > 0:
        crore_scaled = raw_market_cap * 10_000_000.0
        if abs(crore_scaled - implied) / implied <= 0.35:
            return crore_scaled
        if abs(raw_market_cap - implied) / implied <= 0.35:
            return raw_market_cap
    return raw_market_cap * 10_000_000.0 if raw_market_cap < 10_000_000_000.0 else raw_market_cap


def _shareholding_period_end(*payloads: JSONPayload) -> date | None:
    for payload in payloads:
        if not isinstance(payload, Mapping):
            continue
        for key in ("period_end", "periodEnd", "period", "quarter"):
            parsed = _parse_date_like(payload.get(key))
            if parsed is not None:
                return parsed
    return None


def _latest_pattern_period(payload: JSONPayload) -> str:
    if not isinstance(payload, Mapping):
        return ""
    columns = payload.get("columns")
    if not isinstance(columns, list):
        return ""
    candidates = [(parsed, str(value)) for value in columns if (parsed := _parse_date_like(value)) is not None]
    if not candidates:
        return str(columns[-1]).strip() if columns else ""
    return max(candidates, key=lambda item: item[0])[1]


def _latest_ownership_history_rows(payload: JSONPayload) -> list[Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    items = payload.get("ownership_history")
    if not isinstance(items, list):
        return []
    candidates: list[tuple[date, list[Mapping[str, Any]]]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        parsed = _parse_date_like(item.get("header") or item.get("period") or item.get("quarter"))
        data = item.get("data")
        rows = [row for row in data if isinstance(row, Mapping)] if isinstance(data, list) else []
        if parsed is not None and rows:
            candidates.append((parsed, rows))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        data = item.get("data")
        rows = [row for row in data if isinstance(row, Mapping)] if isinstance(data, list) else []
        if rows:
            return rows
    return []


def _as_record_list(payload: JSONPayload, key: str) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    value = payload.get(key)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        return [value]
    return []


def _first_number(row: Mapping[str, Any], aliases: Sequence[str]) -> float:
    normalized = _normalized_row(row)
    for alias in aliases:
        value = normalized.get(_normalize_key(alias))
        parsed = _number(value)
        if parsed is not None:
            return parsed
    return 0.0


def _first_number_from_rows(rows: Sequence[Mapping[str, Any]], aliases: Sequence[str]) -> float:
    for row in rows:
        value = _first_number(row, aliases)
        if value:
            return value
    return 0.0


def _sum_numbers(row: Mapping[str, Any], aliases: Sequence[str]) -> float:
    normalized = _normalized_row(row)
    total = 0.0
    seen = set()
    for alias in aliases:
        key = _normalize_key(alias)
        if key in seen:
            continue
        seen.add(key)
        parsed = _number(normalized.get(key))
        if parsed is not None:
            total += parsed
    return total


def _bank_net_interest_income(row: Mapping[str, Any]) -> float:
    direct = _first_number(row, _BANK_NII_ALIASES)
    if direct:
        return direct
    interest_income = _first_number(row, _INTEREST_INCOME_ALIASES)
    interest_expense = abs(_first_number(row, _INTEREST_ALIASES))
    return interest_income - interest_expense if interest_income else 0.0


def _growth_pct(current: float, previous: float) -> float:
    if abs(previous) < 1e-9:
        return 0.0
    return round(((current - previous) / abs(previous)) * 100.0, 4)


def _safe_ratio_pct(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-9:
        return 0.0
    return round((numerator / denominator) * 100.0, 4)


def _percent_value(value: float) -> float:
    if abs(value) < 1e-9:
        return 0.0
    if -1.0 <= value <= 1.0:
        return round(value * 100.0, 4)
    return value


def _previous_period(periods: Sequence[date], period_end: date) -> date | None:
    try:
        idx = list(periods).index(period_end)
    except ValueError:
        return None
    return periods[idx - 1] if idx > 0 else None


def _derived_ebit(row: Mapping[str, Any]) -> float:
    direct = _first_number(row, _EBIT_ALIASES)
    if direct:
        return direct
    net_income = _first_number(row, _NET_INCOME_ALIASES)
    finance_cost = abs(_first_number(row, _INTEREST_ALIASES))
    tax = _sum_numbers(row, ["currentTax", "deferredTax", "taxExpense", "incomeTaxExpense"])
    return net_income + finance_cost + tax


def _normalized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {_normalize_key(key): value for key, value in row.items()}


def _normalize_key(value: object) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "--", "NA", "N/A", "null", "None"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        parsed = float(text)
    except ValueError:
        return None
    return -parsed if negative else parsed


def _parse_date_like(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    compact = text.replace("-", "").replace("/", "")
    if compact.isdigit() and len(compact) == 8:
        try:
            return date(int(compact[:4]), int(compact[4:6]), int(compact[6:8]))
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    month_date = _parse_month_year(text)
    if month_date is not None:
        return month_date
    year = _extract_year(text)
    if year is not None:
        return date(year, 3, 31)
    return None


def _parse_month_year(text: str) -> date | None:
    months = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }
    lowered = text.lower().replace(",", " ")
    parts = [part.strip() for part in lowered.split() if part.strip()]
    if len(parts) < 2:
        return None
    month = next((months[part[:3]] for part in parts if part[:3] in months), None)
    year = next((int(part) for part in parts if part.isdigit() and len(part) == 4), None)
    if month is None or year is None:
        return None
    return date(year, month, monthrange(year, month)[1])


def _extract_year(text: str) -> int | None:
    digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
    for token in digits:
        if len(token) == 4:
            year = int(token)
            if 1900 <= year <= 2200:
                return year
    return None


def _canonical_statement_type(period: str, statement_type: str) -> str:
    base = "quarterly" if "quarter" in period.lower() else "annual"
    suffix = "consolidated" if statement_type.lower().startswith("c") else "standalone"
    return f"{base}_{suffix}"


def _shareholding_buckets_from_pattern(payload: JSONPayload, period_label: str) -> dict[str, float]:
    buckets = {"promoter": 0.0, "fii": 0.0, "dii": 0.0, "public": 0.0}
    if not isinstance(payload, Mapping):
        return buckets
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return buckets
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        label = " ".join(
            str(row.get(key) or "")
            for key in ("catagory", "category", "name", "group", "sub_group")
        )
        data = row.get("data")
        pct = _number(data.get(period_label)) if isinstance(data, Mapping) and period_label else None
        if pct is None:
            continue
        bucket = _classify_holder(label)
        if bucket:
            buckets[bucket] += pct
    return buckets


def _classify_holder(label: str) -> str:
    text = _normalize_key(label)
    if "promoter" in text:
        return "promoter"
    if "institutionsforeign" in text or any(token in text for token in ("fii", "fpi", "foreignportfolio", "foreigninstitutional", "foreigninvestors")):
        return "fii"
    if "institutionsdomestic" in text or any(token in text for token in ("mutualfund", "insurance", "domesticinstitutional", "dii", "alternateinvestmentfund")):
        return "dii"
    if any(token in text for token in ("noninstitutions", "goverments", "government", "public")) and "nonpublic" not in text:
        return "public"
    return ""


def _public_pct(buckets: Mapping[str, float]) -> float:
    explicit = float(buckets.get("public") or 0.0)
    if explicit > 0:
        return explicit
    return max(0.0, 100.0 - float(buckets.get("promoter", 0.0)) - float(buckets.get("fii", 0.0)) - float(buckets.get("dii", 0.0)))


def _round_pct(value: float) -> float:
    return round(max(0.0, float(value)), 4)


def _ownership_detail_row(symbol: str, period_end: date, source: str, row: Mapping[str, Any]) -> dict[str, object]:
    return {
        "symbol": symbol,
        "period_end": period_end.isoformat(),
        "source": source,
        "shareholder_name": row.get("shareholder_name") or row.get("header") or "",
        "shareholding_pct": _first_number(row, ["shareholdingPct", "shareholdingPctConv"]),
        "pledged_shares": _first_number(row, ["pledgedShares"]),
        "pledged_shares_pct": _first_number(row, ["pledgedSharesPct"]),
        "locked_in_shares": _first_number(row, ["lockedInShares"]),
        "locked_in_shares_pct": _first_number(row, ["lockedInSharesPct"]),
        "total_shares": _first_number(row, ["totalShares"]),
        "total_shareholders": _first_number(row, ["totalShareholders"]),
        "source_id": f"finedge:{symbol}:ownership_detail:{period_end.isoformat()}",
    }


def _ownership_detail_columns(rows: Sequence[Mapping[str, object]]) -> list[str]:
    preferred = [
        "symbol",
        "period_end",
        "source",
        "shareholder_name",
        "shareholding_pct",
        "pledged_shares",
        "pledged_shares_pct",
        "locked_in_shares",
        "locked_in_shares_pct",
        "total_shares",
        "total_shareholders",
        "source_id",
    ]
    extra = sorted({str(key) for row in rows for key in row.keys()} - set(preferred))
    return preferred + extra


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return {"path": str(path), "rows": len(rows)}


def _normalize_symbols(symbols: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for symbol in symbols:
        normalized = symbol.strip().upper()
        if normalized and normalized not in seen:
            seen.add(normalized)
            out.append(normalized)
    return out


def _normalize_sections(sections: Sequence[str]) -> list[str]:
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
            raise ValueError(f"unsupported FinEdge factor section '{section}'")
        for item in values:
            if item not in out:
                out.append(item)
    return out or ["financials", "valuations", "shareholding"]


def _add_issue(result: FinEdgeFactorExport, symbol: str, section: str, message: str) -> None:
    issue = {"symbol": symbol, "section": section, "message": message}
    result.issues.append(issue)
    per_symbol = result.per_symbol.setdefault(symbol, {"financial_rows": 0, "valuation_rows": 0, "shareholding_rows": 0, "banking_rows": 0, "ownership_detail_rows": 0, "issues": []})
    issues = per_symbol.setdefault("issues", [])
    if isinstance(issues, list):
        issues.append(issue)
