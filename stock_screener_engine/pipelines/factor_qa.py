"""Canonical factor QA reporting for financial, valuation, and ownership data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date

from stock_screener_engine.data_sources.schemas import (
    BankingFactorRecord,
    EquityValuationRecord,
    FinancialStatementRecord,
    SecurityMasterRecord,
    ShareholdingRecord,
)
from stock_screener_engine.monitoring.factor_quality import FactorQualityValidator
from stock_screener_engine.storage.market_data_store import MarketDataStore


class CanonicalFactorQAReporter:
    """Builds a reviewable point-in-time QA view over canonical factor tables."""

    def __init__(self, store: MarketDataStore, venue: str = "NSE") -> None:
        self.store = store
        self.venue = venue.strip().upper() or "NSE"
        self.financial_validator = FactorQualityValidator()

    def build(
        self,
        *,
        symbols: Sequence[str],
        as_of: date,
        statement_type: str | None = None,
    ) -> dict[str, object]:
        normalized_symbols = _normalise_symbols(symbols)
        metadata = {record.symbol: record for record in self.store.get_security_master(normalized_symbols)}
        rows = [
            self._symbol_report(
                symbol=symbol,
                as_of=as_of,
                metadata=metadata.get(symbol),
                statement_type=statement_type,
            )
            for symbol in normalized_symbols
        ]
        summary = _summary(rows)
        return {
            "pipeline": "factor_qa",
            "venue": self.venue,
            "as_of": as_of.isoformat(),
            "statement_type": statement_type or "latest",
            "passed": summary["error_count"] == 0,
            "summary": summary,
            "coverage": {
                "financials": self.store.financial_statement_coverage(normalized_symbols, as_of=as_of, venue=self.venue),
                "valuations": self.store.equity_valuation_coverage(normalized_symbols, as_of=as_of, venue=self.venue),
                "shareholding": self.store.shareholding_coverage(normalized_symbols, as_of=as_of, venue=self.venue),
                "banking": self.store.banking_factor_coverage(normalized_symbols, as_of=as_of, venue=self.venue),
            },
            "symbols": rows,
            "console_rows": [_console_row(row) for row in rows],
            "markdown": render_factor_qa_markdown(
                {
                    "venue": self.venue,
                    "as_of": as_of.isoformat(),
                    "passed": summary["error_count"] == 0,
                    "summary": summary,
                    "symbols": rows,
                }
            ),
        }

    def _symbol_report(
        self,
        *,
        symbol: str,
        as_of: date,
        metadata: SecurityMasterRecord | None,
        statement_type: str | None,
    ) -> dict[str, object]:
        financial = self.store.latest_financial_statement_as_of(
            symbol=symbol,
            as_of=as_of,
            venue=self.venue,
            statement_type=statement_type,
        )
        valuation = self.store.latest_equity_valuation_as_of(symbol=symbol, as_of=as_of, venue=self.venue)
        shareholding = self.store.latest_shareholding_as_of(symbol=symbol, as_of=as_of, venue=self.venue)
        banking = self.store.latest_banking_factor_as_of(symbol=symbol, as_of=as_of, venue=self.venue)
        issues = self._quality_issues(
            symbol=symbol,
            as_of=as_of,
            metadata=metadata,
            financial=financial,
            valuation=valuation,
            shareholding=shareholding,
            banking=banking,
        )
        error_count = sum(1 for issue in issues if issue["severity"] == "error")
        warning_count = sum(1 for issue in issues if issue["severity"] == "warning")
        return {
            "symbol": symbol,
            "company_name": metadata.company_name if metadata else "",
            "sector": metadata.sector if metadata else "Unknown",
            "industry": metadata.industry if metadata else "Unknown",
            "status": "error" if error_count else "warning" if warning_count else "ok",
            "quality_score": max(0, 100 - (30 * error_count) - (8 * warning_count)),
            "financials": _financial_payload(financial, as_of),
            "valuation": _valuation_payload(valuation, as_of),
            "shareholding": _shareholding_payload(shareholding, as_of),
            "banking": _banking_payload(banking, as_of),
            "derived_metrics": _derived_metrics(financial, valuation),
            "banking_metrics": _banking_metrics(banking),
            "quality_issues": issues,
        }

    def _quality_issues(
        self,
        *,
        symbol: str,
        as_of: date,
        metadata: SecurityMasterRecord | None,
        financial: FinancialStatementRecord | None,
        valuation: EquityValuationRecord | None,
        shareholding: ShareholdingRecord | None,
        banking: BankingFactorRecord | None,
    ) -> list[dict[str, object]]:
        issues: list[dict[str, object]] = []
        if metadata is None:
            issues.append(_issue(symbol, "warning", "metadata", "security master metadata is missing"))

        if financial is None:
            issues.append(_issue(symbol, "error", "financials", "no eligible financial statement found as of cutoff"))
        else:
            validator_report = self.financial_validator.validate([financial], as_of=as_of)
            for item in validator_report.issues:
                issues.append(_issue(symbol, item.severity, "financials", item.message))
            if financial.period_end < _days_before(as_of, 550):
                issues.append(_issue(symbol, "warning", "financials", "latest financial statement is older than 550 days"))
            if financial.total_assets <= 0:
                issues.append(_issue(symbol, "warning", "financials", "total assets are non-positive"))
            if financial.revenue <= 0 and not _is_financial_business(metadata):
                issues.append(_issue(symbol, "warning", "financials", "revenue is non-positive for a non-financial company"))

        if valuation is None:
            issues.append(_issue(symbol, "error", "valuation", "no eligible valuation record found as of cutoff"))
        else:
            if valuation.as_of > as_of:
                issues.append(_issue(symbol, "error", "valuation", "valuation as_of is after cutoff"))
            if valuation.as_of < _days_before(as_of, 10):
                issues.append(_issue(symbol, "warning", "valuation", "latest valuation record is older than 10 days"))
            if valuation.market_cap <= 0:
                issues.append(_issue(symbol, "error", "valuation", "market cap is non-positive"))
            if valuation.shares_outstanding <= 0:
                issues.append(_issue(symbol, "warning", "valuation", "shares outstanding are missing or non-positive"))

        if shareholding is None:
            issues.append(_issue(symbol, "warning", "shareholding", "no eligible shareholding record found as of cutoff"))
        else:
            if shareholding.period_end > as_of or shareholding.filing_date > as_of:
                issues.append(_issue(symbol, "error", "shareholding", "shareholding record violates point-in-time cutoff"))
            if shareholding.period_end < _days_before(as_of, 220):
                issues.append(_issue(symbol, "warning", "shareholding", "latest shareholding period is older than 220 days"))
            total_holding = shareholding.promoter_pct + shareholding.fii_pct + shareholding.dii_pct + shareholding.public_pct
            if total_holding < 99.0 or total_holding > 101.0:
                issues.append(_issue(symbol, "warning", "shareholding", f"holding buckets sum to {total_holding:.2f}%"))

        metrics = _derived_metrics(financial, valuation)
        pe = metrics.get("pe_ratio")
        pb = metrics.get("pb_ratio")
        if isinstance(pe, float) and pe > 300:
            issues.append(_issue(symbol, "warning", "valuation", "PE ratio is above 300; review units and mapping"))
        if isinstance(pb, float) and pb > 50:
            issues.append(_issue(symbol, "warning", "valuation", "PB ratio is above 50; review units and mapping"))
        if _is_financial_business(metadata):
            if banking is None:
                issues.append(_issue(symbol, "warning", "banking", "bank/NBFC-specific factor row is missing"))
                issues.append(_issue(symbol, "warning", "financials", "generic statement factors are present, but bank/NBFC-specific factors are still needed"))
            else:
                if banking.period_end < _days_before(as_of, 220):
                    issues.append(_issue(symbol, "warning", "banking", "latest banking factor period is older than 220 days"))
                missing_groups = _missing_banking_quality_groups(banking)
                if missing_groups:
                    issues.append(
                        _issue(
                            symbol,
                            "warning",
                            "banking",
                            "banking factor row is present but missing critical groups: " + ", ".join(missing_groups),
                        )
                    )
                if _banking_metric_coverage(banking) < 0.5:
                    issues.append(_issue(symbol, "warning", "banking", "banking factor row has sparse metric coverage; review vendor field mapping"))
                if banking.gnpa_ratio_pct > 8.0:
                    issues.append(_issue(symbol, "warning", "banking", "GNPA ratio is above 8%"))
                if banking.nnpa_ratio_pct > banking.gnpa_ratio_pct and banking.gnpa_ratio_pct > 0:
                    issues.append(_issue(symbol, "warning", "banking", "NNPA ratio is above GNPA ratio; review source mapping"))
                if 0.0 < banking.capital_adequacy_ratio_pct < 10.5:
                    issues.append(_issue(symbol, "warning", "banking", "capital adequacy ratio is below 10.5%"))
        return issues


def render_factor_qa_markdown(report: Mapping[str, object]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), Mapping) else {}
    lines = [
        "# Canonical Factor QA",
        "",
        f"- Venue: {report.get('venue')}",
        f"- As of: {report.get('as_of')}",
        f"- Passed: {report.get('passed')}",
        f"- Symbols: {summary.get('symbol_count', 0)}",
        f"- Errors: {summary.get('error_count', 0)}",
        f"- Warnings: {summary.get('warning_count', 0)}",
        "",
        "| Symbol | Status | Score | Financial Period | Valuation Date | Shareholding Period | Banking Period | PE | PB | Banking Score | Issues |",
        "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report.get("symbols", []):
        if not isinstance(row, Mapping):
            continue
        financials = row.get("financials") if isinstance(row.get("financials"), Mapping) else {}
        valuation = row.get("valuation") if isinstance(row.get("valuation"), Mapping) else {}
        shareholding = row.get("shareholding") if isinstance(row.get("shareholding"), Mapping) else {}
        banking = row.get("banking") if isinstance(row.get("banking"), Mapping) else {}
        metrics = row.get("derived_metrics") if isinstance(row.get("derived_metrics"), Mapping) else {}
        banking_metrics = row.get("banking_metrics") if isinstance(row.get("banking_metrics"), Mapping) else {}
        issues = row.get("quality_issues") if isinstance(row.get("quality_issues"), list) else []
        issue_text = "; ".join(str(issue.get("message", "")) for issue in issues if isinstance(issue, Mapping)) or "None"
        lines.append(
            "| {symbol} | {status} | {score} | {period} | {valuation_date} | {shareholding_period} | {banking_period} | {pe} | {pb} | {banking_score} | {issues} |".format(
                symbol=row.get("symbol"),
                status=row.get("status"),
                score=row.get("quality_score"),
                period=financials.get("period_end", ""),
                valuation_date=valuation.get("as_of", ""),
                shareholding_period=shareholding.get("period_end", ""),
                banking_period=banking.get("period_end", ""),
                pe=_display_number(metrics.get("pe_ratio")),
                pb=_display_number(metrics.get("pb_ratio")),
                banking_score=_display_number(banking_metrics.get("banking_quality_score")),
                issues=issue_text.replace("|", "/"),
            )
        )
    return "\n".join(lines)


def _financial_payload(record: FinancialStatementRecord | None, as_of: date) -> dict[str, object]:
    if record is None:
        return {"available": False}
    payload = asdict(record)
    payload["period_end"] = record.period_end.isoformat()
    payload["filing_date"] = record.filing_date.isoformat()
    payload["age_days"] = (as_of - record.period_end).days
    return {"available": True, **payload}


def _valuation_payload(record: EquityValuationRecord | None, as_of: date) -> dict[str, object]:
    if record is None:
        return {"available": False}
    payload = asdict(record)
    payload["as_of"] = record.as_of.isoformat()
    payload["age_days"] = (as_of - record.as_of).days
    payload["implied_price"] = _round_or_none(_safe_div(record.market_cap, record.shares_outstanding))
    return {"available": True, **payload}


def _shareholding_payload(record: ShareholdingRecord | None, as_of: date) -> dict[str, object]:
    if record is None:
        return {"available": False}
    payload = asdict(record)
    payload["period_end"] = record.period_end.isoformat()
    payload["filing_date"] = record.filing_date.isoformat()
    payload["age_days"] = (as_of - record.period_end).days
    payload["total_pct"] = _round_or_none(record.promoter_pct + record.fii_pct + record.dii_pct + record.public_pct)
    return {"available": True, **payload}


def _banking_payload(record: BankingFactorRecord | None, as_of: date) -> dict[str, object]:
    if record is None:
        return {"available": False}
    payload = asdict(record)
    payload["period_end"] = record.period_end.isoformat()
    payload["filing_date"] = record.filing_date.isoformat()
    payload["age_days"] = (as_of - record.period_end).days
    return {"available": True, **payload}


def _banking_metrics(record: BankingFactorRecord | None) -> dict[str, object]:
    if record is None:
        return {}
    asset_quality = _weighted_score(
        (
            (_inverse_pct_score_optional(record.gnpa_ratio_pct, 0.0, 8.0), 0.45),
            (_inverse_pct_score_optional(record.nnpa_ratio_pct, 0.0, 4.0), 0.35),
            (_pct_score_optional(record.provision_coverage_ratio_pct, 40.0, 80.0), 0.20),
        )
    )
    capital_strength = _weighted_score(
        (
            (_pct_score_optional(record.capital_adequacy_ratio_pct, 10.5, 18.0), 0.65),
            (_pct_score_optional(record.cet1_ratio_pct, 8.0, 15.0), 0.35),
        )
    )
    franchise_strength = _weighted_score(
        (
            (_pct_score_optional(record.casa_ratio_pct, 25.0, 45.0), 0.45),
            (_pct_score_optional(record.deposits_growth_pct, 0.0, 18.0), 0.30),
            (_pct_score_optional(record.advances_growth_pct, 0.0, 20.0), 0.25),
        )
    )
    profitability = _weighted_score(
        (
            (_pct_score_optional(record.roa_pct, 0.5, 2.0), 0.50),
            (_pct_score_optional(record.roe_pct, 8.0, 18.0), 0.35),
            (_pct_score_optional(record.net_interest_margin_pct, 2.0, 5.0), 0.15),
        )
    )
    efficiency = _weighted_score(((_inverse_pct_score_optional(record.cost_to_income_ratio_pct, 35.0, 65.0), 1.0),))
    composite = 0.26 * asset_quality + 0.24 * capital_strength + 0.20 * franchise_strength + 0.20 * profitability + 0.10 * efficiency
    return {
        "banking_quality_score": round(composite * 100.0, 2),
        "asset_quality_score": round(asset_quality * 100.0, 2),
        "capital_strength_score": round(capital_strength * 100.0, 2),
        "franchise_strength_score": round(franchise_strength * 100.0, 2),
        "bank_profitability_score": round(profitability * 100.0, 2),
        "efficiency_score": round(efficiency * 100.0, 2),
        "banking_metric_coverage": round(_banking_metric_coverage(record), 4),
        "populated_banking_metrics": _populated_banking_metric_count(record),
        "expected_banking_metrics": len(_BANKING_METRIC_FIELDS),
    }


def _derived_metrics(
    financial: FinancialStatementRecord | None,
    valuation: EquityValuationRecord | None,
) -> dict[str, object]:
    if financial is None:
        return {}
    market_cap = valuation.market_cap if valuation is not None else 0.0
    return {
        "pe_ratio": _round_or_none(_safe_div(market_cap, financial.net_income)),
        "pb_ratio": _round_or_none(_safe_div(market_cap, financial.equity)),
        "earnings_yield": _round_or_none(_safe_div(financial.net_income, market_cap)),
        "roe": _round_or_none(_safe_div(financial.net_income, financial.equity)),
        "roa": _round_or_none(_safe_div(financial.net_income, financial.total_assets)),
        "ebit_margin": _round_or_none(_safe_div(financial.ebit, financial.revenue)),
        "net_margin": _round_or_none(_safe_div(financial.net_income, financial.revenue)),
        "cfo_to_pat": _round_or_none(_safe_div(financial.operating_cash_flow, financial.net_income)),
        "fcf_margin": _round_or_none(_safe_div(financial.operating_cash_flow - financial.capex, financial.revenue)),
        "debt_to_equity": _round_or_none(_safe_div(financial.total_debt, financial.equity)),
        "current_ratio": _round_or_none(_safe_div(financial.current_assets, financial.current_liabilities)),
        "interest_coverage": _round_or_none(_safe_div(financial.ebit, financial.interest_expense)),
    }


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    error_count = sum(
        1
        for row in rows
        for issue in row.get("quality_issues", [])
        if isinstance(issue, Mapping) and issue.get("severity") == "error"
    )
    warning_count = sum(
        1
        for row in rows
        for issue in row.get("quality_issues", [])
        if isinstance(issue, Mapping) and issue.get("severity") == "warning"
    )
    financials = sum(1 for row in rows if _available(row.get("financials")))
    valuations = sum(1 for row in rows if _available(row.get("valuation")))
    shareholding = sum(1 for row in rows if _available(row.get("shareholding")))
    banking = sum(1 for row in rows if _available(row.get("banking")))
    scores = [float(row.get("quality_score", 0.0)) for row in rows]
    count = len(rows)
    return {
        "symbol_count": count,
        "error_count": error_count,
        "warning_count": warning_count,
        "financials_available": financials,
        "valuations_available": valuations,
        "shareholding_available": shareholding,
        "banking_available": banking,
        "average_quality_score": round(sum(scores) / count, 2) if count else 0.0,
    }


def _console_row(row: Mapping[str, object]) -> dict[str, object]:
    financials = row.get("financials") if isinstance(row.get("financials"), Mapping) else {}
    valuation = row.get("valuation") if isinstance(row.get("valuation"), Mapping) else {}
    shareholding = row.get("shareholding") if isinstance(row.get("shareholding"), Mapping) else {}
    banking = row.get("banking") if isinstance(row.get("banking"), Mapping) else {}
    metrics = row.get("derived_metrics") if isinstance(row.get("derived_metrics"), Mapping) else {}
    banking_metrics = row.get("banking_metrics") if isinstance(row.get("banking_metrics"), Mapping) else {}
    issues = row.get("quality_issues") if isinstance(row.get("quality_issues"), list) else []
    errors = [issue for issue in issues if isinstance(issue, Mapping) and issue.get("severity") == "error"]
    warnings = [issue for issue in issues if isinstance(issue, Mapping) and issue.get("severity") == "warning"]
    return {
        "symbol": row.get("symbol"),
        "sector": row.get("sector"),
        "status": row.get("status"),
        "quality_score": row.get("quality_score"),
        "financial_period": financials.get("period_end", ""),
        "valuation_as_of": valuation.get("as_of", ""),
        "shareholding_period": shareholding.get("period_end", ""),
        "pe": metrics.get("pe_ratio"),
        "pb": metrics.get("pb_ratio"),
        "roe": metrics.get("roe"),
        "debt_to_equity": metrics.get("debt_to_equity"),
        "cfo_to_pat": metrics.get("cfo_to_pat"),
        "promoter_pct": shareholding.get("promoter_pct"),
        "fii_pct": shareholding.get("fii_pct"),
        "dii_pct": shareholding.get("dii_pct"),
        "banking_period": banking.get("period_end", ""),
        "nim_pct": banking.get("net_interest_margin_pct"),
        "gnpa_pct": banking.get("gnpa_ratio_pct"),
        "nnpa_pct": banking.get("nnpa_ratio_pct"),
        "casa_pct": banking.get("casa_ratio_pct"),
        "car_pct": banking.get("capital_adequacy_ratio_pct"),
        "banking_quality_score": banking_metrics.get("banking_quality_score"),
        "banking_metric_coverage": banking_metrics.get("banking_metric_coverage"),
        "errors": len(errors),
        "warnings": len(warnings),
        "top_issue": _first_issue_text(issues),
    }


def _issue(symbol: str, severity: str, section: str, message: str) -> dict[str, object]:
    return {"symbol": symbol, "severity": severity, "section": section, "message": message}


def _first_issue_text(issues: object) -> str:
    if not isinstance(issues, list):
        return ""
    for issue in issues:
        if isinstance(issue, Mapping):
            return str(issue.get("message", ""))
    return ""


def _available(payload: object) -> bool:
    return isinstance(payload, Mapping) and payload.get("available") is True


def _normalise_symbols(symbols: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        normalized = symbol.strip().upper()
        if normalized and normalized not in seen:
            out.append(normalized)
            seen.add(normalized)
    return out


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator <= 0 or abs(denominator) < 1e-9:
        return None
    return numerator / denominator


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def _display_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    return ""


def _pct_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _inverse_pct_score(value: float, low: float, high: float) -> float:
    return 1.0 - _pct_score(value, low, high)


def _pct_score_optional(value: float, low: float, high: float) -> float | None:
    if abs(float(value)) < 1e-9:
        return None
    return _pct_score(value, low, high)


def _inverse_pct_score_optional(value: float, low: float, high: float) -> float | None:
    if abs(float(value)) < 1e-9:
        return None
    return _inverse_pct_score(value, low, high)


def _weighted_score(components: Sequence[tuple[float | None, float]]) -> float:
    configured_weight = sum(weight for _, weight in components)
    if configured_weight <= 0:
        return 0.0
    return sum((score or 0.0) * weight for score, weight in components) / configured_weight


_BANKING_METRIC_FIELDS = (
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
)


def _populated_banking_metric_count(record: BankingFactorRecord) -> int:
    return sum(1 for field in _BANKING_METRIC_FIELDS if abs(float(getattr(record, field))) > 1e-9)


def _banking_metric_coverage(record: BankingFactorRecord) -> float:
    return _populated_banking_metric_count(record) / len(_BANKING_METRIC_FIELDS)


def _missing_banking_quality_groups(record: BankingFactorRecord) -> list[str]:
    missing = []
    if record.net_interest_margin_pct <= 0 and record.roa_pct <= 0 and record.roe_pct <= 0:
        missing.append("profitability")
    if record.gnpa_ratio_pct <= 0 and record.nnpa_ratio_pct <= 0 and record.provision_coverage_ratio_pct <= 0:
        missing.append("asset quality")
    if record.capital_adequacy_ratio_pct <= 0 and record.cet1_ratio_pct <= 0:
        missing.append("capital adequacy")
    if record.casa_ratio_pct <= 0 and record.deposits_growth_pct <= 0:
        missing.append("deposit franchise")
    return missing


def _days_before(as_of: date, days: int) -> date:
    return date.fromordinal(as_of.toordinal() - days)


def _is_financial_business(metadata: SecurityMasterRecord | None) -> bool:
    if metadata is None:
        return False
    text = f"{metadata.sector} {metadata.industry}".lower()
    return any(token in text for token in ("bank", "financial", "finance", "nbfc", "insurance"))
