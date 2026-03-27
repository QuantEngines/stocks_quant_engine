"""Factor-quality validation for ingested financial statements."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from stock_screener_engine.data_sources.schemas import FactorQualityIssue, FinancialStatementRecord


@dataclass(frozen=True)
class FactorQualityReport:
    passed: bool
    issues: list[FactorQualityIssue]


class FactorQualityValidator:
    """Applies accounting sanity checks to statement records."""

    def validate(self, records: list[FinancialStatementRecord], as_of: date) -> FactorQualityReport:
        issues: list[FactorQualityIssue] = []
        for rec in records:
            if rec.period_end > as_of or rec.filing_date > as_of:
                issues.append(
                    FactorQualityIssue(
                        symbol=rec.symbol,
                        as_of=as_of,
                        severity="error",
                        message="point-in-time violation: record dated after as_of",
                    )
                )

            if rec.total_assets < rec.current_assets:
                issues.append(
                    FactorQualityIssue(
                        symbol=rec.symbol,
                        as_of=as_of,
                        severity="warning",
                        message="current_assets exceed total_assets",
                    )
                )

            if rec.equity <= 0 and rec.net_income > 0:
                issues.append(
                    FactorQualityIssue(
                        symbol=rec.symbol,
                        as_of=as_of,
                        severity="warning",
                        message="positive net income with non-positive equity",
                    )
                )

            if rec.revenue < 0:
                issues.append(
                    FactorQualityIssue(
                        symbol=rec.symbol,
                        as_of=as_of,
                        severity="error",
                        message="negative revenue",
                    )
                )

            if rec.interest_expense > 0 and rec.ebit < 0:
                issues.append(
                    FactorQualityIssue(
                        symbol=rec.symbol,
                        as_of=as_of,
                        severity="warning",
                        message="negative EBIT with non-zero interest expense",
                    )
                )

        has_error = any(i.severity == "error" for i in issues)
        return FactorQualityReport(passed=not has_error, issues=issues)
