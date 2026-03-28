"""Research engine orchestration that remains broker-agnostic."""

from __future__ import annotations

import json
import logging
from datetime import datetime, time, timedelta
from pathlib import Path

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.backtest.calibration import (
    CalibrationReport,
    TunedWeightPriors,
    WeightPriorAutoTuner,
)
from stock_screener_engine.core.entities import (
    FeatureVector,
    FundamentalsSnapshot,
    GovernanceSnapshot,
    MarketSnapshot,
    ScoreCard,
    SignalResult,
    StockSnapshot,
)
from stock_screener_engine.core.features import FeatureEngine
from stock_screener_engine.core.ml_ranking import LinearRankModel
from stock_screener_engine.core.regime_detection import RegimeDetector, RegimeThresholds
from stock_screener_engine.core.ranking import rank_by_long_term, rank_by_swing
from stock_screener_engine.core.scoring import (
    LongTermScorer,
    LongTermWeights,
    RegimeSwitchConfig,
    RiskPenaltyScorer,
    RiskPenaltyWeights,
    SwingScorer,
    SwingWeights,
    build_score_card,
)
from stock_screener_engine.core.signals import SignalGenerator
from stock_screener_engine.core.text_feature_integration import TextFeatureIntegration
from stock_screener_engine.core.universe import UniverseSelector
from stock_screener_engine.core.valuation_normalization import RollingSectorValuationNormalizer
from stock_screener_engine.data_sources.base.interfaces import (
    FinancialsProvider,
    MarketDataProvider,
    TextEventProvider,
)
from stock_screener_engine.execution.portfolio_adapter import (
    PortfolioConstructionAdapter,
    PortfolioConstraints,
)
from stock_screener_engine.features.text_features.aggregator import to_feature_map
from stock_screener_engine.monitoring.data_quality import DataQualityChecker
from stock_screener_engine.nlp.event_engine.pipeline import TextIntelligencePipeline

logger = logging.getLogger(__name__)


class ResearchEngine:
    def __init__(
        self,
        settings: AppSettings,
        market_data: MarketDataProvider,
        text_data: TextEventProvider,
        financials: FinancialsProvider | None = None,
        text_pipeline: TextIntelligencePipeline | None = None,
    ) -> None:
        self.settings = settings
        self.market_data = market_data
        self.text_data = text_data
        self.financials = financials  # optional; falls back to StockSnapshot fields if None
        self.text_pipeline = text_pipeline

        self.universe_selector = UniverseSelector(min_liquidity=settings.features.min_liquidity_threshold)
        self.feature_engine = FeatureEngine(
            include_sentiment=settings.features.include_sentiment,
            include_event_signals=settings.features.include_event_signals,
            include_regime_features=settings.features.include_regime_features,
        )

        long_weight_values = {
            k: getattr(settings.scoring.long_term_weights, k)
            for k in settings.scoring.long_term_weights.__dataclass_fields__
        }
        swing_weight_values = {
            k: getattr(settings.scoring.swing_weights, k)
            for k in settings.scoring.swing_weights.__dataclass_fields__
        }

        tuned = self._auto_tuned_weight_priors(long_weight_values, swing_weight_values)
        if tuned is not None:
            long_weight_values = tuned.long_term
            swing_weight_values = tuned.swing

        self.regime_detector = RegimeDetector(
            thresholds=RegimeThresholds(
                bull_threshold=settings.scoring.regime_switching.bull_threshold,
                bear_threshold=settings.scoring.regime_switching.bear_threshold,
            )
        )
        regime_cfg = RegimeSwitchConfig(
            enabled=settings.scoring.regime_switching.enabled,
            bull_threshold=settings.scoring.regime_switching.bull_threshold,
            bear_threshold=settings.scoring.regime_switching.bear_threshold,
        )

        self.long_scorer = LongTermScorer(
            weights=LongTermWeights(**long_weight_values),
            regime_switch=regime_cfg,
            regime_profiles=settings.scoring.long_term_regime_profiles,
        )
        self.swing_scorer = SwingScorer(
            weights=SwingWeights(**swing_weight_values),
            regime_switch=regime_cfg,
            regime_profiles=settings.scoring.swing_regime_profiles,
        )
        self.risk_scorer = RiskPenaltyScorer(
            max_penalty=settings.scoring.max_risk_penalty,
            weights=RiskPenaltyWeights(
                liquidity_risk=settings.scoring.risk_weights.liquidity_risk,
                volatility_risk=settings.scoring.risk_weights.volatility_risk,
                leverage_risk=settings.scoring.risk_weights.leverage_risk,
                earnings_instability_risk=settings.scoring.risk_weights.earnings_instability_risk,
                event_uncertainty_risk=settings.scoring.risk_weights.event_uncertainty_risk,
                governance_risk=settings.scoring.risk_weights.governance_risk,
                text_uncertainty_risk=settings.scoring.risk_weights.text_uncertainty_risk,
            ),
        )
        self.signal_generator = SignalGenerator(
            long_term_min_score=settings.scoring.long_term_min_score,
            swing_min_score=settings.scoring.swing_min_score,
            short_min_score=settings.scoring.short_min_score,
        )
        self.valuation_normalizer = RollingSectorValuationNormalizer()
        self.text_feature_integration = TextFeatureIntegration()
        self.portfolio_adapter = PortfolioConstructionAdapter()
        self.quality = DataQualityChecker()

    def run(self, symbols: list[str] | None = None, regime_score: float | None = None) -> dict[str, list]:
        symbols = symbols or self.market_data.get_universe()
        snapshots = self.market_data.get_snapshots(symbols)

        if regime_score is None:
            regime_snapshot = self._detect_regime_snapshot(snapshots)
            regime_score = float(regime_snapshot["score"])
        else:
            regime_snapshot = {
                "label": self._label_for_regime_score(regime_score),
                "score": float(regime_score),
                "momentum": 0.0,
                "realized_volatility": 0.0,
                "breadth": 0.5,
            }

        snapshot_quality = self.quality.validate_snapshots(snapshots)
        if not snapshot_quality.passed:
            logger.warning("Snapshot quality issues: %s", snapshot_quality.issues)

        selected = self.universe_selector.select(snapshots)
        features, sector_map, text_feature_rows = self._compute_features(selected, regime_score=regime_score)

        feature_quality = self.quality.validate_features(features)
        if not feature_quality.passed:
            logger.warning("Feature quality issues: %s", feature_quality.issues)

        score_cards: list[ScoreCard] = []
        long_signals: list[SignalResult] = []
        swing_signals: list[SignalResult] = []
        short_signals: list[SignalResult] = []

        for fv in features:
            card = build_score_card(fv, self.long_scorer, self.swing_scorer, self.risk_scorer)
            _, _, risk_flags = self.risk_scorer.score(fv)
            sector = sector_map.get(fv.symbol, "")
            score_cards.append(card)
            long_signals.append(
                self.signal_generator.build_long_term_signal(card, risk_flags=risk_flags, sector=sector)
            )
            swing_signals.append(
                self.signal_generator.build_swing_signal(card, risk_flags=risk_flags, sector=sector)
            )
            short_signals.append(
                self.signal_generator.build_short_signal(fv, risk_flags=risk_flags, sector=sector)
            )

        long_sorted  = sorted(long_signals,  key=lambda s: s.score, reverse=True)
        swing_sorted = sorted(swing_signals, key=lambda s: s.score, reverse=True)
        short_sorted = sorted(short_signals, key=lambda s: s.score, reverse=True)

        top_long  = long_sorted[: self.settings.scoring.ranking.top_k_long_term]
        top_swing = swing_sorted[: self.settings.scoring.ranking.top_k_swing]
        top_short = [s for s in short_sorted if s.category == "short_candidate"][
            : self.settings.scoring.ranking.top_k_swing
        ]

        price_by_symbol = {s.symbol: s.close for s in selected}
        volume_by_symbol = {s.symbol: s.volume for s in selected}

        portfolio_cfg = self.settings.scoring.ranking.portfolio
        long_portfolio_positions = []
        long_portfolio_rejected = []
        swing_portfolio_positions = []
        swing_portfolio_rejected = []
        if portfolio_cfg.enabled:
            long_min_notional = (
                portfolio_cfg.long_min_position_notional
                if portfolio_cfg.long_min_position_notional is not None
                else portfolio_cfg.min_position_notional
            )
            swing_min_notional = (
                portfolio_cfg.swing_min_position_notional
                if portfolio_cfg.swing_min_position_notional is not None
                else portfolio_cfg.min_position_notional
            )
            long_sector_targets = (
                portfolio_cfg.long_sector_target_weights
                if portfolio_cfg.long_sector_target_weights is not None
                else portfolio_cfg.sector_target_weights
            )
            swing_sector_targets = (
                portfolio_cfg.swing_sector_target_weights
                if portfolio_cfg.swing_sector_target_weights is not None
                else portfolio_cfg.sector_target_weights
            )

            long_portfolio = self.portfolio_adapter.construct(
                ranked_signals=top_long,
                sector_by_symbol=sector_map,
                price_by_symbol=price_by_symbol,
                volume_by_symbol=volume_by_symbol,
                constraints=PortfolioConstraints(
                    max_positions=portfolio_cfg.max_positions_long,
                    max_sector_positions=portfolio_cfg.max_sector_positions,
                    min_avg_daily_volume=portfolio_cfg.min_avg_daily_volume,
                    max_single_position_weight=portfolio_cfg.max_single_position_weight,
                    capital_base=portfolio_cfg.capital_base,
                    min_position_notional=long_min_notional,
                    sector_target_weights=long_sector_targets,
                    sector_target_tolerance=portfolio_cfg.sector_target_tolerance,
                ),
            )
            swing_portfolio = self.portfolio_adapter.construct(
                ranked_signals=top_swing,
                sector_by_symbol=sector_map,
                price_by_symbol=price_by_symbol,
                volume_by_symbol=volume_by_symbol,
                constraints=PortfolioConstraints(
                    max_positions=portfolio_cfg.max_positions_swing,
                    max_sector_positions=portfolio_cfg.max_sector_positions,
                    min_avg_daily_volume=portfolio_cfg.min_avg_daily_volume,
                    max_single_position_weight=portfolio_cfg.max_single_position_weight,
                    capital_base=portfolio_cfg.capital_base,
                    min_position_notional=swing_min_notional,
                    sector_target_weights=swing_sector_targets,
                    sector_target_tolerance=portfolio_cfg.sector_target_tolerance,
                ),
            )
            long_portfolio_positions = long_portfolio.positions
            long_portfolio_rejected = long_portfolio.rejected
            swing_portfolio_positions = swing_portfolio.positions
            swing_portfolio_rejected = swing_portfolio.rejected

        ml_ranked: list[tuple[str, float]] = []
        ml_long_ranked: list[ScoreCard] = []
        ml_swing_ranked: list[ScoreCard] = []
        model = self._load_ml_rank_model()
        if model is not None:
            ml_ranked = model.rank({fv.symbol: dict(fv.values) for fv in features})
            ml_by_symbol = {symbol: score for symbol, score in ml_ranked}
            ml_long_ranked = sorted(
                score_cards,
                key=lambda c: (ml_by_symbol.get(c.symbol, float("-inf")), c.symbol),
                reverse=True,
            )
            ml_swing_ranked = list(ml_long_ranked)

        return {
            "regime_snapshot": regime_snapshot,
            "features": features,
            "text_features": text_feature_rows,
            "scores": score_cards,
            "long_ranked": rank_by_long_term(score_cards),
            "swing_ranked": rank_by_swing(score_cards),
            "ml_ranked": ml_ranked,
            "ml_long_ranked": ml_long_ranked,
            "ml_swing_ranked": ml_swing_ranked,
            "long_signals": long_sorted,
            "swing_signals": swing_sorted,
            "short_signals": short_sorted,
            "long_signals_top": top_long,
            "swing_signals_top": top_swing,
            "short_signals_top": top_short,
            "long_portfolio_positions": long_portfolio_positions,
            "long_portfolio_rejected": long_portfolio_rejected,
            "swing_portfolio_positions": swing_portfolio_positions,
            "swing_portfolio_rejected": swing_portfolio_rejected,
        }

    def _detect_regime_snapshot(self, snapshots: list[StockSnapshot]) -> dict[str, float | str]:
        if not snapshots:
            return {
                "label": "neutral",
                "score": 0.0,
                "momentum": 0.0,
                "realized_volatility": 0.0,
                "breadth": 0.5,
            }

        end = max(s.as_of for s in snapshots)
        start = end - timedelta(days=140)
        try:
            index_bars = self.market_data.get_historical("^NSEI", interval="1d", start=start, end=end)
            snap = self.regime_detector.detect(index_bars=index_bars)
            return {
                "label": snap.label,
                "score": snap.score,
                "momentum": snap.momentum,
                "realized_volatility": snap.realized_volatility,
                "breadth": snap.breadth,
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("Regime detection fallback to neutral due to: %s", exc)
            return {
                "label": "neutral",
                "score": 0.0,
                "momentum": 0.0,
                "realized_volatility": 0.0,
                "breadth": 0.5,
            }

    def _load_ml_rank_model(self) -> LinearRankModel | None:
        path = Path(self.settings.storage.root_dir) / "calibration" / "ml_rank_model_latest.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            return LinearRankModel.from_payload(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load ML rank model from %s: %s", path, exc)
            return None

    def _label_for_regime_score(self, score: float) -> str:
        if score >= self.settings.scoring.regime_switching.bull_threshold:
            return "bull"
        if score <= self.settings.scoring.regime_switching.bear_threshold:
            return "bear"
        return "neutral"

    def _compute_features(
        self, snapshots: list[StockSnapshot], regime_score: float
    ) -> tuple[list[FeatureVector], dict[str, str], list[dict[str, float | str]]]:
        events = self.text_data.get_recent_events([s.symbol for s in snapshots])
        sector_map: dict[str, str] = {s.symbol: s.sector for s in snapshots}
        fundamental_map: dict[str, FundamentalsSnapshot] = {}
        governance_map: dict[str, GovernanceSnapshot] = {}
        if self.financials is not None:
            symbols = [s.symbol for s in snapshots]
            fundamental_map = self.financials.get_fundamentals(symbols)
            governance_map = self.financials.get_governance(symbols)

        pe_by_symbol: dict[str, float] = {}
        pb_by_symbol: dict[str, float] = {}
        for snap in snapshots:
            fin = fundamental_map.get(snap.symbol)
            pe_by_symbol[snap.symbol] = fin.pe_ratio if fin is not None else snap.pe_ratio
            pb_by_symbol[snap.symbol] = fin.pb_ratio if fin is not None else 0.0

        valuation_context = self.valuation_normalizer.normalize(
            sector_by_symbol=sector_map,
            pe_by_symbol=pe_by_symbol,
            pb_by_symbol=pb_by_symbol,
        )

        text_feature_map: dict[str, dict[str, float]] = {}
        if self.text_pipeline is not None and self.settings.nlp.enabled:
            as_of_dt = datetime.combine(max(s.as_of for s in snapshots), time(15, 30)) if snapshots else datetime.utcnow()
            feature_set = self.text_pipeline.run(
                symbols=[s.symbol for s in snapshots],
                as_of=as_of_dt,
                lookback_days=self.settings.nlp.lookback_days,
            )
            text_feature_map = to_feature_map(feature_set)

        text_feature_rows: list[dict[str, float | str]] = []
        for symbol, fmap in text_feature_map.items():
            text_feature_rows.append({"symbol": symbol, **fmap})

        vectors: list[FeatureVector] = []
        for snap in snapshots:
            sentiment = self.text_data.get_sentiment_score(snap.symbol)
            event_signal = 0.2 if events.get(snap.symbol) else 0.0
            end = snap.as_of
            start = end - timedelta(days=140)
            bars = self.market_data.get_historical(snap.symbol, interval="1d", start=start, end=end)
            index_bars = self.market_data.get_historical("^NSEI", interval="1d", start=start, end=end)

            if self.financials is None:
                vector = self.feature_engine.compute_from_snapshot(
                    snapshot=snap,
                    historical_bars=bars,
                    index_bars=index_bars,
                    sentiment_score=sentiment,
                    event_signal=event_signal,
                    market_regime_score=regime_score,
                    valuation_context=valuation_context,
                    text_feature_values=self.text_feature_integration.merge(snap.symbol, text_feature_map),
                )
            else:
                market = MarketSnapshot(
                    symbol=snap.symbol,
                    as_of=snap.as_of,
                    sector=snap.sector,
                    close=snap.close,
                    volume=snap.volume,
                    delivery_ratio=snap.delivery_ratio,
                )
                vector = self.feature_engine.compute(
                    market=market,
                    fundamentals=fundamental_map.get(snap.symbol),
                    governance=governance_map.get(snap.symbol),
                    historical_bars=bars,
                    index_bars=index_bars,
                    sentiment_score=sentiment,
                    event_signal=event_signal,
                    market_regime_score=regime_score,
                    valuation_context=valuation_context,
                    text_feature_values=self.text_feature_integration.merge(snap.symbol, text_feature_map),
                )
            vectors.append(vector)
        return vectors, sector_map, text_feature_rows

    def _auto_tuned_weight_priors(
        self,
        long_weights: dict[str, float],
        swing_weights: dict[str, float],
    ) -> TunedWeightPriors | None:
        cfg = self.settings.scoring.calibration_auto_tune
        if not cfg.enabled:
            return None

        report_path = Path(cfg.report_path)
        if not report_path.exists():
            logger.warning("Calibration report not found for auto-tune: %s", report_path)
            return None

        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            report_raw = payload.get("report", payload)
            report = CalibrationReport(
                quantile_ic={int(k): float(v) for k, v in report_raw.get("quantile_ic", {}).items()},
                turnover_top_quantile={int(k): float(v) for k, v in report_raw.get("turnover_top_quantile", {}).items()},
                decay={int(k): float(v) for k, v in report_raw.get("decay", {}).items()},
            )
            tuned = WeightPriorAutoTuner().tune(
                report=report,
                long_term_weights=long_weights,
                swing_weights=swing_weights,
                learning_rate=cfg.learning_rate,
            )
            logger.info("Applied calibration auto-tuned priors: %s", tuned.diagnostics)
            return tuned
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Failed to apply calibration auto-tune priors: %s", exc)
            return None
