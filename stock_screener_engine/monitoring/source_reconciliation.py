"""Source reconciliation checks across exchange and broker market feeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence

from stock_screener_engine.data_sources.schemas import OHLCVBar


@dataclass(frozen=True)
class SourceReconciliationIssue:
    symbol: str
    ts: str
    severity: str
    message: str
    venues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SourceReconciliationReport:
    passed: bool
    issues: list[SourceReconciliationIssue]
    metrics: dict[str, float | int]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": dict(self.metrics),
        }


class SourceReconciler:
    def __init__(
        self,
        max_close_divergence_pct: float = 1.0,
        max_volume_divergence_pct: float = 50.0,
    ) -> None:
        self.max_close_divergence_pct = max_close_divergence_pct
        self.max_volume_divergence_pct = max_volume_divergence_pct

    def reconcile(
        self,
        bars: Sequence[OHLCVBar],
        requested_symbols: Sequence[str] | None = None,
    ) -> SourceReconciliationReport:
        issues: list[SourceReconciliationIssue] = []
        groups: dict[tuple[str, str], list[OHLCVBar]] = {}
        for bar in bars:
            groups.setdefault((bar.symbol.upper(), bar.ts), []).append(bar)

        requested = {s.strip().upper() for s in requested_symbols or [] if s.strip()}
        observed = {bar.symbol.upper() for bar in bars}
        for symbol in sorted(requested - observed):
            issues.append(
                SourceReconciliationIssue(
                    symbol=symbol,
                    ts="",
                    severity="error",
                    message="No bars observed from any source",
                )
            )

        multi_source_groups = 0
        for (symbol, ts), rows in sorted(groups.items()):
            venues = sorted({row.venue.upper() for row in rows})
            if len(venues) < 2:
                continue
            multi_source_groups += 1
            close_values = [row.close for row in rows if row.close > 0]
            volume_values = [row.volume for row in rows if row.volume >= 0]
            close_div = _pct_divergence(close_values)
            volume_div = _pct_divergence(volume_values)
            if close_div > self.max_close_divergence_pct:
                issues.append(
                    SourceReconciliationIssue(
                        symbol=symbol,
                        ts=ts,
                        severity="error",
                        message=(
                            f"Close divergence {close_div:.2f}% exceeds "
                            f"{self.max_close_divergence_pct:.2f}%"
                        ),
                        venues=venues,
                    )
                )
            if volume_div > self.max_volume_divergence_pct:
                issues.append(
                    SourceReconciliationIssue(
                        symbol=symbol,
                        ts=ts,
                        severity="warning",
                        message=(
                            f"Volume divergence {volume_div:.2f}% exceeds "
                            f"{self.max_volume_divergence_pct:.2f}%"
                        ),
                        venues=venues,
                    )
                )

        blocking = any(issue.severity == "error" for issue in issues)
        return SourceReconciliationReport(
            passed=not blocking,
            issues=issues,
            metrics={
                "bar_count": len(bars),
                "symbol_count": len(observed),
                "multi_source_groups": multi_source_groups,
                "issue_count": len(issues),
            },
        )


def _pct_divergence(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    lo = min(values)
    hi = max(values)
    if lo <= 0:
        return 0.0 if hi <= 0 else 100.0
    return (hi / lo - 1.0) * 100.0
