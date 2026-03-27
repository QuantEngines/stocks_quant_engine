"""Feature engine split by category and designed for extension.

Each category method is independently testable and returns a plain ``dict``.
The ``compute()`` method is the full pipeline; ``compute_from_snapshot()``
provides backward-compatible access for demos and unit tests that use the
unified ``StockSnapshot``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from stock_screener_engine.core.entities import (
    FundamentalsSnapshot,
    GovernanceSnapshot,
    MarketSnapshot,
    StockSnapshot,
    FeatureVector,
)
from stock_screener_engine.core.feature_specs import (
    FEAT_BALANCE_SHEET_HEALTH,
    FEAT_CASH_FLOW_QUALITY,
    FEAT_DELIVERY_RATIO,
    FEAT_EVENT_DECAY_WEIGHTED_SCORE,
    FEAT_EVENT_CATALYST,
    FEAT_EVENT_MOMENTUM_SCORE,
    FEAT_EVENT_RISK_SCORE,
    FEAT_EVENT_STRENGTH_SCORE,
    FEAT_EARNINGS_STABILITY,
    FEAT_GOVERNANCE_EVENT,
    FEAT_GOVERNANCE_PROXY,
    FEAT_GROWTH_QUALITY,
    FEAT_LEVERAGE_TREND,
    FEAT_MARKET_REGIME,
    FEAT_MOMENTUM_STRENGTH,
    FEAT_NEWS_SENTIMENT,
    FEAT_OPERATING_MARGIN,
    FEAT_PROFITABILITY,
    FEAT_RELATIVE_STRENGTH,
    FEAT_REVENUE_GROWTH,
    FEAT_ROLLING_PB_ZSCORE,
    FEAT_ROLLING_PE_ZSCORE,
    FEAT_SENTIMENT_MOMENTUM,
    FEAT_SENTIMENT_RECENT,
    FEAT_SENTIMENT_TREND,
    FEAT_SECTOR_PB_ZSCORE,
    FEAT_SECTOR_PE_ZSCORE,
    FEAT_SECTOR_MOMENTUM,
    FEAT_SENTIMENT_SCORE,
    FEAT_TREND_STRENGTH,
    FEAT_VALUATION_SANITY,
    FEAT_GOVERNANCE_FLAG_SCORE,
    FEAT_CATALYST_PRESENCE_FLAG,
    FEAT_UNCERTAINTY_PENALTY,
    FEAT_RECENT_EVENT_COUNT,
    FEAT_MANAGEMENT_TONE_SCORE,
    FEAT_EARNINGS_SENTIMENT_SCORE,
    FEAT_EVENT_CLUSTER_SCORE,
    FEAT_DECAYED_EVENT_SIGNAL,
    FEAT_TRANSCRIPT_QUALITY_SIGNAL,
    FEAT_RECENT_POSITIVE_EVENT_SCORE,
    FEAT_RECENT_NEGATIVE_EVENT_SCORE,
    FEAT_CATALYST_STRENGTH_SCORE,
    FEAT_GOVERNANCE_RISK_SCORE,
    FEAT_HIGH_IMPACT_EVENT_FLAG,
    FEAT_VOLATILITY_REGIME,
    FEAT_VOLUME_CONFIRMATION,
    FEAT_PE_RATIO,
    FEAT_PB_RATIO,
    FEAT_DEBT_TO_EQUITY,
    FEAT_CFO_PAT_RATIO,
    FEAT_BREAKOUT_SCORE,
    FEAT_COMPRESSION_SCORE,
    FEAT_PRICE_ACCELERATION,
    FEAT_ACTIVITY_VS_AVG,
)
from stock_screener_engine.core.technical_indicators import adx, atr, breakout_compression, momentum, rolling_beta


@dataclass
class FeatureEngine:
    """Compute normalised [0, 1] feature values from domain snapshots.

    Feature design principles
    -------------------------
    * All features are normalised to roughly [0, 1] (may exceed for strong
      signals).  Negative values are used for adverse signals.
    * Each category method is a pure function: no IO, no side effects.
    * Adding a new feature category means adding a new ``_xxx_features``
      method and calling it inside ``compute()``.
    """

    include_sentiment: bool = True
    include_event_signals: bool = True
    include_regime_features: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        market: MarketSnapshot,
        fundamentals: FundamentalsSnapshot | None = None,
        governance: GovernanceSnapshot | None = None,
        historical_bars: list[dict] | None = None,
        index_bars: list[dict] | None = None,
        sentiment_score: float = 0.0,
        news_sentiment: float = 0.0,
        event_signal: float = 0.0,
        market_regime_score: float = 0.0,
        sector_momentum: float = 0.0,
        valuation_context: Mapping[str, Mapping[str, float]] | None = None,
        text_feature_values: Mapping[str, float] | None = None,
    ) -> FeatureVector:
        """Compute feature vector from granular snapshot types."""
        values: dict[str, float] = {}

        values.update(
            self._fundamental_features(
                fundamentals,
                valuation_metrics=(valuation_context or {}).get(market.symbol),
            )
        )
        values.update(self._technical_features(market, historical_bars=historical_bars, index_bars=index_bars))
        values.update(self._governance_features(governance))

        if self.include_event_signals:
            has_event = event_signal != 0.0
            gov_score = governance.insider_activity_score if governance else 0.0
            values.update(self._event_features(gov_score, event_signal, has_event))

        if self.include_sentiment:
            values.update(self._sentiment_features(sentiment_score, news_sentiment))

        if self.include_regime_features:
            values.update(self._regime_features(market_regime_score, sector_momentum))

        values.update(self._text_features(text_feature_values))

        return FeatureVector(symbol=market.symbol, as_of=market.as_of, values=values)

    def compute_from_snapshot(
        self,
        snapshot: StockSnapshot,
        historical_bars: list[dict] | None = None,
        index_bars: list[dict] | None = None,
        sentiment_score: float = 0.0,
        event_signal: float = 0.0,
        market_regime_score: float = 0.0,
        valuation_context: Mapping[str, Mapping[str, float]] | None = None,
        text_feature_values: Mapping[str, float] | None = None,
    ) -> FeatureVector:
        """Backward-compatible entry point for unified ``StockSnapshot``."""
        market = MarketSnapshot(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            sector=snapshot.sector,
            close=snapshot.close,
            volume=snapshot.volume,
            delivery_ratio=snapshot.delivery_ratio,
        )
        fundamentals = FundamentalsSnapshot(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            pe_ratio=snapshot.pe_ratio,
            roe=snapshot.roe,
            debt_to_equity=snapshot.debt_to_equity,
            earnings_growth_yoy=snapshot.earnings_growth,
            free_cash_flow_margin=snapshot.free_cash_flow_margin,
        )
        governance = GovernanceSnapshot(
            symbol=snapshot.symbol,
            as_of=snapshot.as_of,
            promoter_holding_change_qoq=snapshot.promoter_holding_change,
            insider_activity_score=snapshot.insider_activity_score,
        )
        return self.compute(
            market=market,
            fundamentals=fundamentals,
            governance=governance,
            historical_bars=historical_bars,
            index_bars=index_bars,
            sentiment_score=sentiment_score,
            event_signal=event_signal,
            market_regime_score=market_regime_score,
            valuation_context=valuation_context,
            text_feature_values=text_feature_values,
        )

    # ------------------------------------------------------------------
    # Feature category methods (pure functions, independently testable)
    # ------------------------------------------------------------------

    def _fundamental_features(
        self,
        f: FundamentalsSnapshot | None,
        valuation_metrics: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        if f is None:
            return {k: 0.0 for k in [
                FEAT_GROWTH_QUALITY, FEAT_PROFITABILITY, FEAT_BALANCE_SHEET_HEALTH,
                FEAT_CASH_FLOW_QUALITY, FEAT_VALUATION_SANITY,
                FEAT_REVENUE_GROWTH, FEAT_OPERATING_MARGIN,
                FEAT_EARNINGS_STABILITY, FEAT_LEVERAGE_TREND,
                FEAT_SECTOR_PE_ZSCORE, FEAT_SECTOR_PB_ZSCORE,
                FEAT_ROLLING_PE_ZSCORE, FEAT_ROLLING_PB_ZSCORE,
                FEAT_PE_RATIO, FEAT_PB_RATIO, FEAT_DEBT_TO_EQUITY, FEAT_CFO_PAT_RATIO,
            ]}

        metrics = valuation_metrics or {}
        pe_sector_z = float(metrics.get("sector_pe_zscore", 0.0))
        pb_sector_z = float(metrics.get("sector_pb_zscore", 0.0))
        pe_rolling_z = float(metrics.get("rolling_pe_zscore", 0.0))
        pb_rolling_z = float(metrics.get("rolling_pb_zscore", 0.0))

        base_valuation = _clamp(max(0.0, 1.0 - (f.pe_ratio / 100.0)))
        valuation_sanity = _clamp(
            0.40 * base_valuation
            + 0.20 * _z_to_inverse_score(pe_sector_z)
            + 0.15 * _z_to_inverse_score(pb_sector_z)
            + 0.15 * _z_to_inverse_score(pe_rolling_z)
            + 0.10 * _z_to_inverse_score(pb_rolling_z)
        )

        growth_spread = abs(f.earnings_growth_yoy - f.revenue_growth_yoy)
        earnings_stability = _clamp(
            0.55 * _clamp(1.0 - (growth_spread / 0.35))
            + 0.45 * _clamp((f.net_profit_margin + 0.05) / 0.35)
        )

        leverage_trend = _clamp(
            0.65 * _clamp(1.0 - min(3.0, max(0.0, f.debt_to_equity)) / 3.0)
            + 0.35 * _clamp(f.interest_coverage / 12.0)
        )

        return {
            FEAT_GROWTH_QUALITY: _clamp(f.earnings_growth_yoy),
            FEAT_PROFITABILITY: _clamp(f.roe),
            FEAT_BALANCE_SHEET_HEALTH: _clamp(max(0.0, 1.0 - f.debt_to_equity)),
            FEAT_CASH_FLOW_QUALITY: _clamp(f.free_cash_flow_margin),
            FEAT_VALUATION_SANITY: valuation_sanity,
            FEAT_REVENUE_GROWTH: _clamp(f.revenue_growth_yoy),
            FEAT_OPERATING_MARGIN: _clamp(f.operating_margin),
            FEAT_EARNINGS_STABILITY: earnings_stability,
            FEAT_LEVERAGE_TREND: leverage_trend,
            FEAT_SECTOR_PE_ZSCORE: pe_sector_z,
            FEAT_SECTOR_PB_ZSCORE: pb_sector_z,
            FEAT_ROLLING_PE_ZSCORE: pe_rolling_z,
            FEAT_ROLLING_PB_ZSCORE: pb_rolling_z,
            # raw ratios for scoring functions that apply own normalisation
            FEAT_PE_RATIO: max(0.0, f.pe_ratio),
            FEAT_PB_RATIO: max(0.0, f.pb_ratio),
            FEAT_DEBT_TO_EQUITY: max(0.0, f.debt_to_equity),
            FEAT_CFO_PAT_RATIO: _cfo_pat_ratio(f),
        }

    def _governance_features(self, g: GovernanceSnapshot | None) -> dict[str, float]:
        if g is None:
            return {FEAT_GOVERNANCE_PROXY: 0.0}
        holding_signal = _clamp((g.promoter_holding_change_qoq + 0.1) / 0.2)
        insider_signal = _clamp((g.insider_activity_score + 1.0) / 2.0)
        audit_penalty = 0.0 if g.audit_opinion == "clean" else (
            0.3 if g.audit_opinion == "qualified" else 1.0
        )
        combined = (holding_signal * 0.5 + insider_signal * 0.5) * (1.0 - audit_penalty)
        return {FEAT_GOVERNANCE_PROXY: _clamp(combined)}

    def _technical_features(
        self,
        m: MarketSnapshot,
        historical_bars: list[dict] | None = None,
        index_bars: list[dict] | None = None,
    ) -> dict[str, float]:
        vol_norm = _clamp(m.volume / 20_000_000.0) if m.volume > 0 else 0.0
        delivery_signal = _clamp(m.delivery_ratio)
        if not historical_bars:
            return {
                FEAT_TREND_STRENGTH: delivery_signal,
                FEAT_MOMENTUM_STRENGTH: delivery_signal,
                FEAT_VOLATILITY_REGIME: 0.5,
                FEAT_VOLUME_CONFIRMATION: vol_norm,
                FEAT_RELATIVE_STRENGTH: delivery_signal,
                FEAT_DELIVERY_RATIO: delivery_signal,
                FEAT_BREAKOUT_SCORE: 0.5,
                FEAT_COMPRESSION_SCORE: 0.5,
                FEAT_PRICE_ACCELERATION: 0.0,
                FEAT_ACTIVITY_VS_AVG: _activity_vs_avg(m, None),
            }

        high = [float(b.get("high", 0.0) or 0.0) for b in historical_bars]
        low = [float(b.get("low", 0.0) or 0.0) for b in historical_bars]
        close = [float(b.get("close", 0.0) or 0.0) for b in historical_bars]
        if len(close) < 5:
            return {
                FEAT_TREND_STRENGTH: delivery_signal,
                FEAT_MOMENTUM_STRENGTH: delivery_signal,
                FEAT_VOLATILITY_REGIME: 0.5,
                FEAT_VOLUME_CONFIRMATION: vol_norm,
                FEAT_RELATIVE_STRENGTH: delivery_signal,
                FEAT_DELIVERY_RATIO: delivery_signal,
                FEAT_BREAKOUT_SCORE: 0.5,
                FEAT_COMPRESSION_SCORE: 0.5,
                FEAT_PRICE_ACCELERATION: 0.0,
                FEAT_ACTIVITY_VS_AVG: _activity_vs_avg(m, None),
            }

        atr_value = atr(high, low, close, period=14)
        adx_value = adx(high, low, close, period=14)
        mom_value = momentum(close, lookback=20)
        compression = breakout_compression(close, lookback=20)

        relative_strength = delivery_signal
        if index_bars:
            index_close = [float(b.get("close", 0.0) or 0.0) for b in index_bars]
            beta = rolling_beta(close, index_close, period=60)
            relative_strength = _clamp(1.0 - min(2.0, abs(beta - 1.0)) / 2.0)

        atr_norm = 0.0 if close[-1] <= 0 else min(2.0, atr_value / close[-1])
        volatility_regime = _clamp(1.0 - min(1.0, atr_norm / 0.08))
        trend_strength = _clamp(adx_value / 50.0)
        momentum_strength = _clamp((mom_value + 0.2) / 0.4)
        breakout_signal = _clamp(compression)

        # breakout proximity: how close current close is to range high
        range_hi = max(close[-20:])
        range_lo = min(close[-20:])
        range_span = range_hi - range_lo
        _bk = _clamp((close[-1] - range_lo) / range_span) if range_span > 0 else 0.5
        # compression: tight ATR relative to range = good setup
        _cp = _clamp(1.0 - min(1.0, atr_norm / 0.06)) if atr_norm > 0 else 0.5
        # price acceleration: 2nd derivative of 10-day return
        _pa = _price_acceleration(close)
        return {
            FEAT_TREND_STRENGTH: _clamp(0.7 * trend_strength + 0.3 * breakout_signal),
            FEAT_MOMENTUM_STRENGTH: momentum_strength,
            FEAT_VOLATILITY_REGIME: volatility_regime,
            FEAT_VOLUME_CONFIRMATION: vol_norm,
            FEAT_RELATIVE_STRENGTH: relative_strength,
            FEAT_DELIVERY_RATIO: delivery_signal,
            FEAT_BREAKOUT_SCORE: _bk,
            FEAT_COMPRESSION_SCORE: _cp,
            FEAT_PRICE_ACCELERATION: _pa,
            FEAT_ACTIVITY_VS_AVG: _activity_vs_avg(m, historical_bars),
        }

    def _event_features(
        self, governance_score: float, event_signal: float, has_event: bool
    ) -> dict[str, float]:
        catalyst = max(-1.0, min(1.0, event_signal))
        gov_event = _clamp(max(0.0, governance_score + (event_signal / 2.0 if has_event else 0.0)))
        return {
            FEAT_EVENT_CATALYST: catalyst,
            FEAT_GOVERNANCE_EVENT: gov_event,
        }

    def _sentiment_features(self, sentiment: float, news_sentiment: float) -> dict[str, float]:
        return {
            FEAT_SENTIMENT_SCORE: max(-1.0, min(1.0, sentiment)),
            FEAT_NEWS_SENTIMENT: max(-1.0, min(1.0, news_sentiment)),
        }

    def _regime_features(self, regime: float, sector_momentum: float) -> dict[str, float]:
        return {
            FEAT_MARKET_REGIME: max(-1.0, min(1.0, regime)),
            FEAT_SECTOR_MOMENTUM: max(-1.0, min(1.0, sector_momentum)),
        }

    def _text_features(self, values: Mapping[str, float] | None) -> dict[str, float]:
        base = {
            FEAT_SENTIMENT_RECENT: 0.0,
            FEAT_SENTIMENT_TREND: 0.0,
            FEAT_SENTIMENT_MOMENTUM: 0.0,
            FEAT_EVENT_STRENGTH_SCORE: 0.0,
            FEAT_EVENT_RISK_SCORE: 0.0,
            FEAT_GOVERNANCE_FLAG_SCORE: 0.0,
            FEAT_CATALYST_PRESENCE_FLAG: 0.0,
            FEAT_EVENT_MOMENTUM_SCORE: 0.0,
            FEAT_EVENT_DECAY_WEIGHTED_SCORE: 0.0,
            FEAT_UNCERTAINTY_PENALTY: 0.0,
            FEAT_RECENT_EVENT_COUNT: 0.0,
            FEAT_HIGH_IMPACT_EVENT_FLAG: 0.0,
            FEAT_MANAGEMENT_TONE_SCORE: 0.0,
            FEAT_EARNINGS_SENTIMENT_SCORE: 0.0,
            FEAT_EVENT_CLUSTER_SCORE: 0.0,
            FEAT_DECAYED_EVENT_SIGNAL: 0.0,
            FEAT_TRANSCRIPT_QUALITY_SIGNAL: 0.0,
            FEAT_RECENT_POSITIVE_EVENT_SCORE: 0.0,
            FEAT_RECENT_NEGATIVE_EVENT_SCORE: 0.0,
            FEAT_CATALYST_STRENGTH_SCORE: 0.0,
            FEAT_GOVERNANCE_RISK_SCORE: 0.0,
        }
        if not values:
            return base
        out = dict(base)
        for key in out:
            raw = float(values.get(key, out[key]))
            if key in {
                FEAT_SENTIMENT_RECENT,
                FEAT_SENTIMENT_TREND,
                FEAT_SENTIMENT_MOMENTUM,
                FEAT_EVENT_DECAY_WEIGHTED_SCORE,
                FEAT_MANAGEMENT_TONE_SCORE,
                FEAT_EARNINGS_SENTIMENT_SCORE,
                FEAT_DECAYED_EVENT_SIGNAL,
            }:
                out[key] = max(-1.0, min(1.0, raw))
            else:
                out[key] = _clamp(raw)
        return out



def _cfo_pat_ratio(f) -> float:
    """Cash-flow-to-earnings quality proxy.

    A value near 1.0 means FCF matches accounting profit (good quality).
    Capped at 3.0 to avoid outlier dominance.
    """
    if f.net_profit_margin <= 1e-6:
        return 1.0
    ratio = f.free_cash_flow_margin / f.net_profit_margin
    return max(0.0, min(3.0, ratio))


def _price_acceleration(close: list[float], short: int = 10, long: int = 20) -> float:
    """Second derivative of price: momentum of momentum.

    Positive = accelerating uptrend; negative = decelerating or reversing.
    Clipped to [-0.20, 0.20] for feature stability.
    """
    if len(close) < long + 1:
        return 0.0
    mom_recent = (close[-1] - close[-short]) / max(1e-8, abs(close[-short]))
    mom_older  = (close[-short] - close[-long]) / max(1e-8, abs(close[-long]))
    return max(-0.20, min(0.20, mom_recent - mom_older))


def _activity_vs_avg(m, bars) -> float:
    """Current session volume relative to 20-day average.

    Returns a ratio clamped to [0.1, 5.0]; 1.0 = exactly average.
    Uses MarketSnapshot.avg_volume_20d when available, else derives from bars.
    """
    from stock_screener_engine.core.entities import MarketSnapshot
    avg = 0.0
    if isinstance(m, MarketSnapshot) and m.avg_volume_20d > 0:
        avg = m.avg_volume_20d
    elif bars and len(bars) >= 10:
        vols = [float(b.get("volume", 0.0) or 0.0) for b in bars[-20:]]
        avg = sum(vols) / len(vols) if vols else 0.0
    if avg > 0 and m.volume > 0:
        return max(0.1, min(5.0, m.volume / avg))
    return 1.0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _z_to_inverse_score(value: float, bound: float = 3.0) -> float:
    bounded = max(-bound, min(bound, float(value)))
    return _clamp((bound - bounded) / (2.0 * bound))
