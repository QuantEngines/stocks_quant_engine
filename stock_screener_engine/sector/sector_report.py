"""Sector scoring, ranking, and report generation."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from stock_screener_engine.core.entities import FeatureVector, ScoreCard, SignalResult
from stock_screener_engine.sector.sector_schemas import SectorIntelligenceReport


class SectorIntelligenceBuilder:
    def build_from_engine_output(self, output: Mapping[str, object]) -> list[SectorIntelligenceReport]:
        features = output.get("features")
        scores = output.get("scores")
        long_signals = output.get("long_signals")
        swing_signals = output.get("swing_signals")
        return self.build(
            features=features if isinstance(features, list) else [],
            scores=scores if isinstance(scores, list) else [],
            long_signals=long_signals if isinstance(long_signals, list) else [],
            swing_signals=swing_signals if isinstance(swing_signals, list) else [],
        )

    def build(
        self,
        features: Iterable[FeatureVector],
        scores: Iterable[ScoreCard],
        long_signals: Iterable[SignalResult],
        swing_signals: Iterable[SignalResult],
    ) -> list[SectorIntelligenceReport]:
        sector_by_symbol = _sector_by_symbol(long_signals, swing_signals)
        features_by_sector: dict[str, list[FeatureVector]] = defaultdict(list)
        scores_by_symbol = {score.symbol: score for score in scores}
        for feature in features:
            sector = sector_by_symbol.get(feature.symbol, "Unknown")
            features_by_sector[sector].append(feature)

        long_by_sector = _signals_by_sector(long_signals)
        swing_by_sector = _signals_by_sector(swing_signals)
        reports = [
            self._build_sector(
                sector=sector,
                features=sector_features,
                scores_by_symbol=scores_by_symbol,
                long_signals=long_by_sector.get(sector, []),
                swing_signals=swing_by_sector.get(sector, []),
            )
            for sector, sector_features in features_by_sector.items()
        ]
        return sorted(reports, key=lambda r: r.sector_score, reverse=True)

    def render_markdown(self, reports: list[SectorIntelligenceReport], sector: str | None = None) -> str:
        selected = [r for r in reports if r.sector.lower() == sector.lower()] if sector else reports
        lines: list[str] = ["# Sector Intelligence", ""]
        for report in selected:
            lines.extend(
                [
                    f"## {report.sector}",
                    f"Stance: {report.stance}",
                    f"Sector Score: {report.sector_score}/100",
                    "",
                    "Drivers:",
                    *[f"- {item}" for item in report.thesis],
                    "",
                    "Risks:",
                    *[f"- {item}" for item in report.risks],
                    "",
                    "Best Expressions:",
                    f"- long-term: {', '.join(report.best_long_term_stocks) or 'unavailable'}",
                    f"- swing: {', '.join(report.best_swing_candidates) or 'unavailable'}",
                    f"- avoid/watchlist: {', '.join(report.avoid_watchlist_stocks) or 'unavailable'}",
                    "",
                ]
            )
        return "\n".join(lines).strip() + "\n"

    def _build_sector(
        self,
        sector: str,
        features: list[FeatureVector],
        scores_by_symbol: Mapping[str, ScoreCard],
        long_signals: list[SignalResult],
        swing_signals: list[SignalResult],
    ) -> SectorIntelligenceReport:
        vals = [dict(f.values) for f in features]
        momentum_score = _score100(
            _avg(
                vals,
                [
                    "trend_strength",
                    "momentum_strength",
                    "relative_strength_proxy",
                    "cross_sectional_momentum_rank",
                    "sector_relative_momentum_rank",
                ],
            )
        )
        fundamentals_score = _score100(
            _avg(
                vals,
                [
                    "growth_quality",
                    "profitability_quality",
                    "balance_sheet_health",
                    "cash_flow_quality",
                    "cross_sectional_quality_rank",
                    "sector_relative_quality_rank",
                ],
            )
        )
        valuation_score = _score100(_avg(vals, ["valuation_sanity", "cross_sectional_value_rank", "quality_value_composite"]))
        avg_risk_penalty = _avg_score_penalty([scores_by_symbol[f.symbol] for f in features if f.symbol in scores_by_symbol])
        risk_score = round(max(0.0, 100.0 - avg_risk_penalty * (100.0 / 30.0)), 2)
        event_macro_score = _score100(_avg(vals, ["event_catalyst", "sentiment_score", "sector_momentum", "market_regime_score"]))
        sector_score = round(
            0.30 * momentum_score
            + 0.25 * fundamentals_score
            + 0.18 * valuation_score
            + 0.17 * risk_score
            + 0.10 * event_macro_score,
            2,
        )

        stance = "overweight" if sector_score >= 70.0 else ("underweight" if sector_score < 45.0 else "neutral")
        avoid = _unique_symbols([
            signal.symbol
            for signal in sorted(long_signals + swing_signals, key=lambda s: s.score)
            if signal.category.endswith("reject")
        ])[:5]

        return SectorIntelligenceReport(
            sector=sector,
            sector_score=sector_score,
            momentum_score=momentum_score,
            fundamentals_score=fundamentals_score,
            valuation_score=valuation_score,
            risk_score=risk_score,
            event_macro_score=event_macro_score,
            stance=stance,
            best_long_term_stocks=[
                s.symbol
                for s in sorted(long_signals, key=lambda s: s.score, reverse=True)
                if s.category == "long_term_candidate"
            ][:5],
            best_swing_candidates=[
                s.symbol
                for s in sorted(swing_signals, key=lambda s: s.score, reverse=True)
                if s.category == "swing_candidate"
            ][:5],
            avoid_watchlist_stocks=avoid,
            thesis=_sector_thesis(momentum_score, fundamentals_score, valuation_score, event_macro_score),
            risks=_sector_risks(fundamentals_score, valuation_score, risk_score, event_macro_score),
            monitorables=[
                "Sector relative strength versus Nifty",
                "Breadth: stocks above 50/200 DMA",
                "Earnings breadth and margin trend",
                "Policy, input-cost, and currency sensitivity",
            ],
            coverage={
                "stock_count": len(features),
                "scored_stock_count": sum(1 for f in features if f.symbol in scores_by_symbol),
                "avg_feature_coverage": round(_avg(vals, ["feature_coverage_score"]), 4),
                "avg_research_readiness": round(_avg(vals, ["research_readiness_score"]), 4),
            },
        )


def _sector_by_symbol(*signal_groups: Iterable[SignalResult]) -> dict[str, str]:
    out: dict[str, str] = {}
    for signals in signal_groups:
        for signal in signals:
            if signal.sector:
                out[signal.symbol] = signal.sector
    return out


def _signals_by_sector(signals: Iterable[SignalResult]) -> dict[str, list[SignalResult]]:
    out: dict[str, list[SignalResult]] = defaultdict(list)
    for signal in signals:
        out[signal.sector or "Unknown"].append(signal)
    return out


def _avg(rows: list[dict[str, float]], keys: list[str]) -> float:
    vals: list[float] = []
    for row in rows:
        for key in keys:
            if key in row:
                val = float(row[key])
                if key in {"event_catalyst", "sentiment_score", "sector_momentum", "market_regime_score"}:
                    val = (val + 1.0) / 2.0
                vals.append(max(0.0, min(1.0, val)))
    return sum(vals) / len(vals) if vals else 0.0


def _score100(value: float) -> float:
    return round(max(0.0, min(1.0, value)) * 100.0, 2)


def _avg_score_penalty(scores: list[ScoreCard]) -> float:
    if not scores:
        return 30.0
    return sum(score.risk_penalty for score in scores) / len(scores)


def _sector_thesis(momentum: float, fundamentals: float, valuation: float, event_macro: float) -> list[str]:
    out: list[str] = []
    if momentum >= 65:
        out.append("strong relative/price momentum across covered stocks")
    if fundamentals >= 65:
        out.append("fundamental quality and growth features are improving")
    if valuation >= 65:
        out.append("valuation setup is supportive versus sector/history proxies")
    if event_macro >= 60:
        out.append("event and regime context is constructive")
    if not out:
        out.append("no dominant positive sector edge from current covered universe")
    return out


def _sector_risks(fundamentals: float, valuation: float, risk: float, event_macro: float) -> list[str]:
    out: list[str] = []
    if fundamentals < 20:
        out.append("fundamental coverage is missing or too sparse for high-conviction sector calls")
    if valuation < 45:
        out.append("valuation risk or lack of valuation support")
    if risk < 55:
        out.append("liquidity, volatility, leverage, or governance risk is elevated")
    if event_macro < 45:
        out.append("event, sentiment, or macro/regime backdrop is weak")
    if not out:
        out.append("watch for reversal in breadth, earnings momentum, or policy backdrop")
    return out


def _unique_symbols(symbols: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for symbol in symbols:
        normalized = symbol.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out
