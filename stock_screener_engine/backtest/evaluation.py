"""Unified evaluation facade for signal, ranking, sector, and NLP diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from stock_screener_engine.backtest.cross_sectional import CrossSectionalBacktester


@dataclass(frozen=True)
class EvaluationReport:
    scope: str
    metrics: dict[str, float]
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class EvaluationEngine:
    """Small facade over existing backtest metrics with room for expansion."""

    def evaluate_stock_signals(
        self,
        scores: list[float],
        forward_returns: list[float],
        scope: str = "stock_signals",
    ) -> EvaluationReport:
        stats = CrossSectionalBacktester().evaluate_panel(scores=scores, forward_returns=forward_returns)
        diagnostics: list[str] = []
        if stats.ic <= 0:
            diagnostics.append("Rank IC is non-positive; score ordering is not adding cross-sectional alpha.")
        if stats.quantile_spread <= 0:
            diagnostics.append("Top quantile did not outperform bottom quantile.")
        return EvaluationReport(
            scope=scope,
            metrics={
                "hit_rate": stats.hit_rate,
                "avg_return": stats.avg_return,
                "max_drawdown": stats.max_drawdown,
                "quantile_spread": stats.quantile_spread,
                "information_ratio": stats.information_ratio,
                "ic": stats.ic,
                "ic_t_stat": stats.ic_t_stat,
            },
            diagnostics=diagnostics,
        )

    def evaluate_sector_rotation(
        self,
        sector_scores: list[float],
        forward_relative_returns: list[float],
    ) -> EvaluationReport:
        return self.evaluate_stock_signals(
            scores=sector_scores,
            forward_returns=forward_relative_returns,
            scope="sector_rotation",
        )

    def evaluate_document_signal(
        self,
        document_feature_scores: list[float],
        forward_event_returns: list[float],
    ) -> EvaluationReport:
        return self.evaluate_stock_signals(
            scores=document_feature_scores,
            forward_returns=forward_event_returns,
            scope="document_nlp_signals",
        )
