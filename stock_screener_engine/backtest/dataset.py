"""Canonical backtest dataset builders from stored OHLCV bars."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from statistics import mean, pstdev
from typing import Mapping, Sequence

from stock_screener_engine.backtest.calibration import CalibrationDatasetBuilder, ModelCalibrator
from stock_screener_engine.backtest.costs import IndianEquityCostModel
from stock_screener_engine.core.entities import MarketSnapshot
from stock_screener_engine.core.features import FeatureEngine
from stock_screener_engine.core.scoring import LongTermScorer, RiskPenaltyScorer, SwingScorer, build_score_card
from stock_screener_engine.data_sources.schemas import OHLCVBar
from stock_screener_engine.storage.market_data_store import MarketDataStore


@dataclass(frozen=True)
class UniverseSelection:
    policy: str
    selected_symbols: list[str]
    rejected_symbols: dict[str, str]
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ForwardReturnLabel:
    as_of: str
    symbol: str
    horizon: int
    close: float
    forward_close: float
    forward_return: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TechnicalScoreRow:
    as_of: str
    symbol: str
    score: float
    components: Mapping[str, float]

    def to_flat_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of,
            "symbol": self.symbol,
            "score": self.score,
            **{f"component_{key}": value for key, value in self.components.items()},
        }


@dataclass(frozen=True)
class EngineScoreRow:
    as_of: str
    symbol: str
    sector: str
    score_type: str
    score: float
    long_term_score: float
    swing_score: float
    risk_penalty: float
    conviction: float
    components: Mapping[str, float]

    def to_flat_dict(self) -> dict[str, object]:
        return {
            "as_of": self.as_of,
            "symbol": self.symbol,
            "sector": self.sector,
            "score_type": self.score_type,
            "score": self.score,
            "long_term_score": self.long_term_score,
            "swing_score": self.swing_score,
            "risk_penalty": self.risk_penalty,
            "conviction": self.conviction,
            **{f"component_{key}": value for key, value in self.components.items()},
        }


class BacktestUniverseSelector:
    """Select current or history-eligible symbols for point-in-time tests."""

    def __init__(self, store: MarketDataStore, venue: str = "NSE", interval: str = "1d") -> None:
        self.store = store
        self.venue = venue
        self.interval = interval

    def select(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        policy: str = "current",
        min_history_rows: int = 1000,
        horizons: Sequence[int] = (5, 20, 60),
    ) -> UniverseSelection:
        requested = [symbol.strip().upper() for symbol in symbols if symbol.strip()]
        normalized_policy = policy.strip().lower()
        if normalized_policy not in {"current", "eligible_history", "history_eligible"}:
            raise ValueError("universe policy must be 'current' or 'eligible_history'")

        selected: list[str] = []
        rejected: dict[str, str] = {}
        row_counts: dict[str, int] = {}
        max_horizon = max([int(h) for h in horizons], default=0)
        for symbol in requested:
            bars = self.store.get_ohlcv(
                symbol=symbol,
                start=start,
                end=end,
                venue=self.venue,
                interval=self.interval,
                adjusted=True,
            )
            row_count = len(bars)
            row_counts[symbol] = row_count
            if not bars:
                rejected[symbol] = "missing_ohlcv"
                continue
            if normalized_policy in {"eligible_history", "history_eligible"}:
                if row_count < min_history_rows:
                    rejected[symbol] = f"insufficient_history:{row_count}<{min_history_rows}"
                    continue
                if row_count <= max_horizon:
                    rejected[symbol] = f"insufficient_forward_labels:{row_count}<={max_horizon}"
                    continue
            selected.append(symbol)

        return UniverseSelection(
            policy="eligible_history" if normalized_policy == "history_eligible" else normalized_policy,
            selected_symbols=selected,
            rejected_symbols=rejected,
            diagnostics={
                "requested_symbols": len(requested),
                "selected_symbols": len(selected),
                "rejected_symbols": len(rejected),
                "min_history_rows": min_history_rows,
                "horizons": [int(h) for h in horizons],
                "row_counts": row_counts,
            },
        )


class ForwardReturnLabelBuilder:
    """Build point-in-time forward-return labels from close-to-close bars."""

    def __init__(self, store: MarketDataStore, venue: str = "NSE", interval: str = "1d") -> None:
        self.store = store
        self.venue = venue
        self.interval = interval

    def build(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        horizons: Sequence[int],
    ) -> list[ForwardReturnLabel]:
        labels: list[ForwardReturnLabel] = []
        horizon_values = sorted({int(h) for h in horizons if int(h) > 0})
        if not horizon_values:
            return labels

        for symbol in symbols:
            bars = self.store.get_ohlcv(
                symbol=symbol,
                start=start,
                end=end,
                venue=self.venue,
                interval=self.interval,
                adjusted=True,
            )
            labels.extend(_labels_for_symbol(symbol, bars, horizon_values))
        labels.sort(key=lambda row: (row.as_of, row.symbol, row.horizon))
        return labels


class TechnicalRankingDatasetBuilder:
    """Build a simple technical ranking panel for first-pass swing evaluation."""

    def __init__(
        self,
        store: MarketDataStore,
        venue: str = "NSE",
        interval: str = "1d",
        min_lookback: int = 220,
    ) -> None:
        self.store = store
        self.venue = venue
        self.interval = interval
        self.min_lookback = min_lookback

    def build_scores(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        max_horizon: int,
    ) -> list[TechnicalScoreRow]:
        rows: list[TechnicalScoreRow] = []
        for symbol in symbols:
            bars = self.store.get_ohlcv(
                symbol=symbol,
                start=start,
                end=end,
                venue=self.venue,
                interval=self.interval,
                adjusted=True,
            )
            rows.extend(_technical_scores_for_symbol(symbol, bars, self.min_lookback, max_horizon))
        rows.sort(key=lambda row: (row.as_of, row.symbol))
        return rows

    def evaluate(
        self,
        scores: Sequence[TechnicalScoreRow],
        labels: Sequence[ForwardReturnLabel],
        horizons: Sequence[int],
        sector_by_symbol: Mapping[str, str] | None = None,
        cost_model: IndianEquityCostModel | None = None,
    ) -> dict[str, object]:
        score_map = {(row.as_of, row.symbol): row.score for row in scores}
        return evaluate_ranking_panel(
            score_map=score_map,
            labels=labels,
            horizons=horizons,
            sector_by_symbol=sector_by_symbol,
            cost_model=cost_model,
        )


class EngineScoreDatasetBuilder:
    """Build historical score panels using the engine's feature/scoring stack."""

    def __init__(
        self,
        store: MarketDataStore,
        venue: str = "NSE",
        interval: str = "1d",
        min_lookback: int = 220,
        feature_engine: FeatureEngine | None = None,
        long_term_scorer: LongTermScorer | None = None,
        swing_scorer: SwingScorer | None = None,
        risk_scorer: RiskPenaltyScorer | None = None,
    ) -> None:
        self.store = store
        self.venue = venue
        self.interval = interval
        self.min_lookback = min_lookback
        self.feature_engine = feature_engine or FeatureEngine()
        self.long_term_scorer = long_term_scorer or LongTermScorer()
        self.swing_scorer = swing_scorer or SwingScorer()
        self.risk_scorer = risk_scorer or RiskPenaltyScorer()

    def build_scores(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        max_horizon: int,
        score_type: str = "swing",
    ) -> list[EngineScoreRow]:
        normalized_score_type = score_type.strip().lower()
        if normalized_score_type not in {"swing", "long_term", "conviction"}:
            raise ValueError("score_type must be swing, long_term, or conviction")
        sectors = _sector_map(self.store, symbols)
        rows: list[EngineScoreRow] = []
        for symbol in symbols:
            bars = self.store.get_ohlcv(
                symbol=symbol,
                start=start,
                end=end,
                venue=self.venue,
                interval=self.interval,
                adjusted=True,
            )
            rows.extend(self._score_symbol(symbol, bars, sectors.get(symbol, "Unknown"), max_horizon, normalized_score_type))
        rows.sort(key=lambda row: (row.as_of, row.symbol))
        return rows

    def evaluate(
        self,
        scores: Sequence[EngineScoreRow],
        labels: Sequence[ForwardReturnLabel],
        horizons: Sequence[int],
        cost_model: IndianEquityCostModel | None = None,
    ) -> dict[str, object]:
        score_map = {(row.as_of, row.symbol): row.score for row in scores}
        sector_by_symbol = {row.symbol: row.sector for row in scores}
        return evaluate_ranking_panel(
            score_map=score_map,
            labels=labels,
            horizons=horizons,
            sector_by_symbol=sector_by_symbol,
            cost_model=cost_model,
        )

    def _score_symbol(
        self,
        symbol: str,
        bars: list[OHLCVBar],
        sector: str,
        max_horizon: int,
        score_type: str,
    ) -> list[EngineScoreRow]:
        rows: list[EngineScoreRow] = []
        if len(bars) < self.min_lookback + max_horizon + 1:
            return rows
        bar_dicts = [_bar_to_dict(bar) for bar in bars]
        last_score_index = len(bars) - max_horizon - 1
        for idx in range(self.min_lookback - 1, last_score_index + 1):
            bar = bars[idx]
            as_of = date.fromisoformat(bar.ts)
            market = MarketSnapshot(
                symbol=symbol,
                as_of=as_of,
                sector=sector,
                exchange=self.venue,
                close=bar.close,
                open_price=bar.open,
                high=bar.high,
                low=bar.low,
                volume=bar.volume,
                delivery_ratio=0.5,
                avg_volume_20d=_avg_volume(bars, idx, lookback=20),
            )
            fv = self.feature_engine.compute(
                market=market,
                fundamentals=None,
                governance=None,
                historical_bars=bar_dicts[: idx + 1],
                index_bars=None,
                sentiment_score=0.0,
                news_sentiment=0.0,
                event_signal=0.0,
                market_regime_score=0.0,
                sector_momentum=0.0,
                text_feature_values=None,
            )
            score_card = build_score_card(
                fv,
                long_term_scorer=self.long_term_scorer,
                swing_scorer=self.swing_scorer,
                risk_scorer=self.risk_scorer,
            )
            score = {
                "swing": score_card.swing_score,
                "long_term": score_card.long_term_score,
                "conviction": score_card.conviction,
            }[score_type]
            rows.append(
                EngineScoreRow(
                    as_of=bar.ts,
                    symbol=symbol,
                    sector=sector,
                    score_type=score_type,
                    score=score,
                    long_term_score=score_card.long_term_score,
                    swing_score=score_card.swing_score,
                    risk_penalty=score_card.risk_penalty,
                    conviction=score_card.conviction,
                    components=score_card.component_scores,
                )
            )
        return rows


def _labels_for_symbol(
    symbol: str,
    bars: list[OHLCVBar],
    horizons: Sequence[int],
) -> list[ForwardReturnLabel]:
    labels: list[ForwardReturnLabel] = []
    closes = [bar.close for bar in bars]
    for idx, bar in enumerate(bars):
        if bar.close <= 0:
            continue
        for horizon in horizons:
            fwd_idx = idx + int(horizon)
            if fwd_idx >= len(bars):
                continue
            fwd_close = closes[fwd_idx]
            if fwd_close <= 0:
                continue
            labels.append(
                ForwardReturnLabel(
                    as_of=bar.ts,
                    symbol=symbol.strip().upper(),
                    horizon=int(horizon),
                    close=bar.close,
                    forward_close=fwd_close,
                    forward_return=(fwd_close / bar.close) - 1.0,
                )
            )
    return labels


def evaluate_ranking_panel(
    score_map: Mapping[tuple[str, str], float],
    labels: Sequence[ForwardReturnLabel],
    horizons: Sequence[int],
    sector_by_symbol: Mapping[str, str] | None = None,
    cost_model: IndianEquityCostModel | None = None,
) -> dict[str, object]:
    horizon_values = [int(h) for h in horizons]
    gross_returns = {
        (label.as_of, label.symbol, label.horizon): label.forward_return
        for label in labels
        if (label.as_of, label.symbol) in score_map
    }
    model = cost_model or IndianEquityCostModel(explicit_round_trip_bps=0.0, slippage_bps_per_side=0.0)
    net_returns = {key: model.net_return(value) for key, value in gross_returns.items()}

    rows = CalibrationDatasetBuilder().build(
        scores_by_date_symbol=dict(score_map),
        returns_by_date_symbol_horizon=gross_returns,
        horizons=horizon_values,
    )
    gross_report = ModelCalibrator().evaluate(rows, horizons=horizon_values)
    net_rows = CalibrationDatasetBuilder().build(
        scores_by_date_symbol=dict(score_map),
        returns_by_date_symbol_horizon=net_returns,
        horizons=horizon_values,
    )
    net_report = ModelCalibrator().evaluate(net_rows, horizons=horizon_values)
    gross_metrics = _horizon_quantile_metrics(score_map, gross_returns, horizon_values)
    net_metrics = _horizon_quantile_metrics(score_map, net_returns, horizon_values)
    sector_ic = _sector_neutral_ic(score_map, gross_returns, sector_by_symbol or {}, horizon_values)
    sector_ic_net = _sector_neutral_ic(score_map, net_returns, sector_by_symbol or {}, horizon_values)

    return {
        "rows_evaluated": len(rows),
        "horizons": horizon_values,
        "quantile_ic": gross_report.quantile_ic,
        "turnover_top_quantile": gross_report.turnover_top_quantile,
        "decay": gross_report.decay,
        "net_quantile_ic": net_report.quantile_ic,
        "gross_horizon_metrics": gross_metrics,
        "net_horizon_metrics": net_metrics,
        "sector_neutral_ic": sector_ic,
        "sector_neutral_ic_net": sector_ic_net,
        "cost_model": model.to_dict(),
    }


def _technical_scores_for_symbol(
    symbol: str,
    bars: list[OHLCVBar],
    min_lookback: int,
    max_horizon: int,
) -> list[TechnicalScoreRow]:
    rows: list[TechnicalScoreRow] = []
    if len(bars) < min_lookback + max_horizon + 1:
        return rows
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    last_score_index = len(bars) - max_horizon - 1
    for idx in range(min_lookback - 1, last_score_index + 1):
        history_close = closes[: idx + 1]
        history_volume = volumes[: idx + 1]
        components = _technical_score_components(history_close, history_volume)
        score = round(sum(components.values()) / len(components), 6) if components else 0.0
        rows.append(
            TechnicalScoreRow(
                as_of=bars[idx].ts,
                symbol=symbol.strip().upper(),
                score=score,
                components=components,
            )
        )
    return rows


def _horizon_quantile_metrics(
    score_map: Mapping[tuple[str, str], float],
    returns: Mapping[tuple[str, str, int], float],
    horizons: Sequence[int],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    by_date: dict[str, list[tuple[str, float]]] = {}
    for as_of, symbol in score_map:
        by_date.setdefault(as_of, []).append((symbol, score_map[(as_of, symbol)]))
    for horizon in horizons:
        top_returns: list[float] = []
        bottom_returns: list[float] = []
        spreads: list[float] = []
        hit_rates: list[float] = []
        evaluated_dates = 0
        for as_of, date_scores in sorted(by_date.items()):
            panel: list[tuple[str, float, float]] = []
            for symbol, score in date_scores:
                key = (as_of, symbol, int(horizon))
                if key in returns:
                    panel.append((symbol, score, returns[key]))
            if len(panel) < 5:
                continue
            evaluated_dates += 1
            panel.sort(key=lambda row: row[1], reverse=True)
            q = max(1, len(panel) // 5)
            top = [row[2] for row in panel[:q]]
            bottom = [row[2] for row in panel[-q:]]
            top_avg = mean(top)
            bottom_avg = mean(bottom)
            top_returns.append(top_avg)
            bottom_returns.append(bottom_avg)
            spreads.append(top_avg - bottom_avg)
            hit_rates.append(sum(1 for value in top if value > 0) / len(top))
        out[str(horizon)] = {
            "evaluated_dates": float(evaluated_dates),
            "avg_top_quantile_return": round(mean(top_returns), 6) if top_returns else 0.0,
            "avg_bottom_quantile_return": round(mean(bottom_returns), 6) if bottom_returns else 0.0,
            "avg_quantile_spread": round(mean(spreads), 6) if spreads else 0.0,
            "top_quantile_hit_rate": round(mean(hit_rates), 6) if hit_rates else 0.0,
        }
    return out


def _sector_neutral_ic(
    score_map: Mapping[tuple[str, str], float],
    returns: Mapping[tuple[str, str, int], float],
    sector_by_symbol: Mapping[str, str],
    horizons: Sequence[int],
) -> dict[str, float]:
    if not sector_by_symbol:
        return {str(int(horizon)): 0.0 for horizon in horizons}
    by_date: dict[str, list[tuple[str, float]]] = {}
    for as_of, symbol in score_map:
        by_date.setdefault(as_of, []).append((symbol, score_map[(as_of, symbol)]))
    out: dict[str, float] = {}
    for horizon in horizons:
        day_ics: list[float] = []
        for as_of, date_scores in sorted(by_date.items()):
            panel: list[tuple[str, str, float, float]] = []
            for symbol, score in date_scores:
                key = (as_of, symbol, int(horizon))
                if key in returns:
                    panel.append((symbol, sector_by_symbol.get(symbol, "Unknown"), score, returns[key]))
            residual_scores, residual_returns = _sector_residuals(panel)
            if len(residual_scores) >= 5:
                day_ics.append(_spearman(residual_scores, residual_returns))
        out[str(int(horizon))] = sum(day_ics) / len(day_ics) if day_ics else 0.0
    return out


def _sector_residuals(panel: Sequence[tuple[str, str, float, float]]) -> tuple[list[float], list[float]]:
    by_sector: dict[str, list[tuple[float, float]]] = {}
    for _, sector, score, ret in panel:
        by_sector.setdefault(sector, []).append((score, ret))
    residual_scores: list[float] = []
    residual_returns: list[float] = []
    for values in by_sector.values():
        if len(values) < 2:
            continue
        score_avg = mean([score for score, _ in values])
        return_avg = mean([ret for _, ret in values])
        for score, ret in values:
            residual_scores.append(score - score_avg)
            residual_returns.append(ret - return_avg)
    return residual_scores, residual_returns


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _ranks(list(x))
    ry = _ranks(list(y))
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx <= 1e-12 or dy <= 1e-12:
        return 0.0
    return num / (dx * dy)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            out[indexed[k][0]] = rank
        i = j + 1
    return out


def _technical_score_components(close: list[float], volume: list[float]) -> dict[str, float]:
    price = close[-1]
    ma20 = _sma(close, 20)
    ma50 = _sma(close, 50)
    ma200 = _sma(close, 200)
    vol20 = _sma(volume, 20)
    vol60 = _sma(volume, 60)
    return {
        "momentum_20d": _score_return(_momentum(close, 20), scale=0.20),
        "momentum_60d": _score_return(_momentum(close, 60), scale=0.35),
        "price_vs_50dma": _score_return(_ratio_return(price, ma50), scale=0.20),
        "price_vs_200dma": _score_return(_ratio_return(price, ma200), scale=0.35),
        "ma_alignment": _ma_alignment_score(price, ma20, ma50, ma200),
        "volume_participation": _score_return(_ratio_return(vol20, vol60), scale=0.75),
        "volatility_control": 1.0 - _clamp01(_realized_vol(close, 60) / 0.04),
    }


def _sma(values: Sequence[float], period: int) -> float:
    if len(values) < period:
        return 0.0
    return sum(values[-period:]) / period


def _momentum(values: Sequence[float], lookback: int) -> float:
    if len(values) <= lookback:
        return 0.0
    prev = values[-lookback - 1]
    if prev <= 0:
        return 0.0
    return (values[-1] / prev) - 1.0


def _ratio_return(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) - 1.0


def _score_return(value: float, scale: float) -> float:
    if scale <= 0:
        return 0.5
    return _clamp01(0.5 + (value / scale) * 0.5)


def _ma_alignment_score(price: float, ma20: float, ma50: float, ma200: float) -> float:
    checks = [
        price > ma20 > 0,
        ma20 > ma50 > 0,
        ma50 > ma200 > 0,
        price > ma200 > 0,
    ]
    return sum(1 for ok in checks if ok) / len(checks)


def _realized_vol(close: Sequence[float], period: int) -> float:
    if len(close) < period + 1:
        return 0.0
    returns: list[float] = []
    window = close[-period - 1 :]
    for prev, current in zip(window[:-1], window[1:]):
        if prev > 0:
            returns.append((current / prev) - 1.0)
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) if len(set(returns)) > 1 else 0.0


def summarize_forward_labels(labels: Sequence[ForwardReturnLabel]) -> dict[str, object]:
    by_horizon: dict[int, list[float]] = {}
    for label in labels:
        by_horizon.setdefault(label.horizon, []).append(label.forward_return)
    return {
        str(horizon): {
            "count": len(values),
            "avg_forward_return": round(mean(values), 6) if values else 0.0,
            "hit_rate": round(sum(1 for value in values if value > 0) / len(values), 6) if values else 0.0,
        }
        for horizon, values in sorted(by_horizon.items())
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sector_map(store: MarketDataStore, symbols: Sequence[str]) -> dict[str, str]:
    return {record.symbol: record.sector or "Unknown" for record in store.get_security_master(symbols)}


def _bar_to_dict(bar: OHLCVBar) -> dict[str, float | str]:
    return {
        "date": bar.ts,
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "volume": bar.volume,
    }


def _avg_volume(bars: Sequence[OHLCVBar], idx: int, lookback: int = 20) -> float:
    window = bars[max(0, idx - lookback + 1) : idx + 1]
    if not window:
        return 0.0
    return sum(bar.volume for bar in window) / len(window)
