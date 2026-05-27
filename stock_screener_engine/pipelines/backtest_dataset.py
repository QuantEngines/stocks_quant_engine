"""Build forward-return labels and first-pass technical backtest reports."""

from __future__ import annotations

from datetime import date, datetime
from typing import Sequence
from uuid import uuid4

from stock_screener_engine.backtest.dataset import (
    BACKTEST_DATASET_SCHEMA_VERSION,
    ENGINE_SCORE_FEATURE_VERSION,
    FORWARD_LABEL_FEATURE_VERSION,
    BacktestUniverseSelector,
    EngineScoreDatasetBuilder,
    ForwardReturnLabelBuilder,
    TECHNICAL_SCORE_FEATURE_VERSION,
    TechnicalRankingDatasetBuilder,
    summarize_forward_labels,
)
from stock_screener_engine.backtest.costs import IndianEquityCostModel
from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.core.features import FeatureEngine
from stock_screener_engine.core.scoring import (
    LongTermScorer,
    LongTermWeights,
    RegimeSwitchConfig,
    RiskPenaltyScorer,
    SwingScorer,
    SwingWeights,
)
from stock_screener_engine.core.scoring_risk import RiskPenaltyWeights
from stock_screener_engine.storage.local_files import LocalFileStorage
from stock_screener_engine.storage.market_data_store import MarketDataStore


class BacktestDatasetPipeline:
    """Orchestrate canonical backtest dataset generation from local market data."""

    def __init__(
        self,
        settings: AppSettings,
        store: MarketDataStore | None = None,
        file_store: LocalFileStorage | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or MarketDataStore(settings.storage.sqlite_path)
        self.file_store = file_store or LocalFileStorage(settings.storage.root_dir)

    def build_forward_labels(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        horizons: Sequence[int],
        universe_policy: str = "current",
        min_history_rows: int = 1000,
        interval: str = "1d",
    ) -> dict[str, object]:
        selection = self._select_universe(symbols, start, end, horizons, universe_policy, min_history_rows, interval)
        run_context = self._run_context(
            pipeline="forward_return_labels",
            end=end,
            interval=interval,
        )
        label_lineage = self._lineage_for_rows(
            run_context=run_context,
            feature_version=FORWARD_LABEL_FEATURE_VERSION,
            model_version="",
            quality_status="passed" if selection.selected_symbols else "empty",
        )
        labels = ForwardReturnLabelBuilder(
            store=self.store,
            venue=self.settings.runtime_data.canonical_venue,
            interval=interval,
        ).build(
            symbols=selection.selected_symbols,
            start=start,
            end=end,
            horizons=horizons,
            lineage=label_lineage,
        )
        rows = [label.to_dict() for label in labels]
        labels_path = self.file_store.save_rows_csv(
            rows,
            filename="forward_return_labels.csv",
            subdir="backtest",
        )
        report = {
            "pipeline": "forward_return_labels",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "horizons": [int(h) for h in horizons],
            "universe": selection.to_dict(),
            "label_summary": summarize_forward_labels(labels),
            "labels_persisted": len(labels),
            "labels_path": str(labels_path),
            "lineage": {
                "run": run_context,
                "labels": label_lineage,
            },
        }
        report_path = self.file_store.save_json(report, filename="forward_return_labels_report.json", subdir="backtest")
        report["report_path"] = str(report_path)
        return report

    def evaluate_technical_ranking(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        horizons: Sequence[int],
        universe_policy: str = "eligible_history",
        min_history_rows: int = 1000,
        min_lookback: int = 220,
        interval: str = "1d",
        cost_model: IndianEquityCostModel | None = None,
    ) -> dict[str, object]:
        selection = self._select_universe(symbols, start, end, horizons, universe_policy, min_history_rows, interval)
        run_context = self._run_context(
            pipeline="technical_ranking_backtest",
            end=end,
            interval=interval,
        )
        label_lineage = self._lineage_for_rows(
            run_context=run_context,
            feature_version=FORWARD_LABEL_FEATURE_VERSION,
            model_version="",
            quality_status="passed" if selection.selected_symbols else "empty",
        )
        score_lineage = self._lineage_for_rows(
            run_context=run_context,
            feature_version=TECHNICAL_SCORE_FEATURE_VERSION,
            model_version="technical_ranking.v1",
            quality_status="passed" if selection.selected_symbols else "empty",
        )
        max_horizon = max([int(h) for h in horizons], default=0)
        labels = ForwardReturnLabelBuilder(
            store=self.store,
            venue=self.settings.runtime_data.canonical_venue,
            interval=interval,
        ).build(
            symbols=selection.selected_symbols,
            start=start,
            end=end,
            horizons=horizons,
            lineage=label_lineage,
        )
        builder = TechnicalRankingDatasetBuilder(
            store=self.store,
            venue=self.settings.runtime_data.canonical_venue,
            interval=interval,
            min_lookback=min_lookback,
        )
        scores = builder.build_scores(
            symbols=selection.selected_symbols,
            start=start,
            end=end,
            max_horizon=max_horizon,
            lineage=score_lineage,
        )
        sector_by_symbol = {
            record.symbol: record.sector or "Unknown"
            for record in self.store.get_security_master(selection.selected_symbols)
        }
        evaluation = builder.evaluate(
            scores=scores,
            labels=labels,
            horizons=horizons,
            sector_by_symbol=sector_by_symbol,
            cost_model=cost_model,
        )
        label_rows = [label.to_dict() for label in labels]
        score_rows = [score.to_flat_dict() for score in scores]
        labels_path = self.file_store.save_rows_csv(
            label_rows,
            filename="technical_forward_return_labels.csv",
            subdir="backtest",
        )
        scores_path = self.file_store.save_rows_csv(
            score_rows,
            filename="technical_ranking_scores.csv",
            subdir="backtest",
        )
        report = {
            "pipeline": "technical_ranking_backtest",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "horizons": [int(h) for h in horizons],
            "universe": selection.to_dict(),
            "label_summary": summarize_forward_labels(labels),
            "score_rows": len(scores),
            "label_rows": len(labels),
            "evaluation": evaluation,
            "artifacts": {
                "labels_csv": str(labels_path),
                "scores_csv": str(scores_path),
            },
            "lineage": {
                "run": run_context,
                "labels": label_lineage,
                "scores": score_lineage,
            },
        }
        report_path = self.file_store.save_json(report, filename="technical_ranking_evaluation.json", subdir="backtest")
        report["artifacts"]["report_json"] = str(report_path)
        return report

    def evaluate_engine_scores(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        horizons: Sequence[int],
        universe_policy: str = "eligible_history",
        min_history_rows: int = 1000,
        min_lookback: int = 220,
        interval: str = "1d",
        score_type: str = "swing",
        cost_model: IndianEquityCostModel | None = None,
    ) -> dict[str, object]:
        selection = self._select_universe(symbols, start, end, horizons, universe_policy, min_history_rows, interval)
        run_context = self._run_context(
            pipeline="engine_score_backtest",
            end=end,
            interval=interval,
        )
        label_lineage = self._lineage_for_rows(
            run_context=run_context,
            feature_version=FORWARD_LABEL_FEATURE_VERSION,
            model_version="",
            quality_status="passed" if selection.selected_symbols else "empty",
        )
        score_lineage = self._lineage_for_rows(
            run_context=run_context,
            feature_version=ENGINE_SCORE_FEATURE_VERSION,
            model_version=f"engine_{score_type.strip().lower()}_scoring.v1",
            quality_status="passed" if selection.selected_symbols else "empty",
        )
        max_horizon = max([int(h) for h in horizons], default=0)
        labels = ForwardReturnLabelBuilder(
            store=self.store,
            venue=self.settings.runtime_data.canonical_venue,
            interval=interval,
        ).build(
            symbols=selection.selected_symbols,
            start=start,
            end=end,
            horizons=horizons,
            lineage=label_lineage,
        )
        builder = EngineScoreDatasetBuilder(
            store=self.store,
            venue=self.settings.runtime_data.canonical_venue,
            interval=interval,
            min_lookback=min_lookback,
            feature_engine=FeatureEngine(
                include_sentiment=self.settings.features.include_sentiment,
                include_event_signals=self.settings.features.include_event_signals,
                include_regime_features=self.settings.features.include_regime_features,
            ),
            long_term_scorer=_long_term_scorer(self.settings),
            swing_scorer=_swing_scorer(self.settings),
            risk_scorer=_risk_scorer(self.settings),
        )
        scores = builder.build_scores(
            symbols=selection.selected_symbols,
            start=start,
            end=end,
            max_horizon=max_horizon,
            score_type=score_type,
            lineage=score_lineage,
        )
        evaluation = builder.evaluate(
            scores=scores,
            labels=labels,
            horizons=horizons,
            cost_model=cost_model,
        )
        label_rows = [label.to_dict() for label in labels]
        score_rows = [score.to_flat_dict() for score in scores]
        labels_path = self.file_store.save_rows_csv(
            label_rows,
            filename="engine_forward_return_labels.csv",
            subdir="backtest",
        )
        scores_path = self.file_store.save_rows_csv(
            score_rows,
            filename=f"engine_{score_type}_scores.csv",
            subdir="backtest",
        )
        report = {
            "pipeline": "engine_score_backtest",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "interval": interval,
            "horizons": [int(h) for h in horizons],
            "score_type": score_type,
            "universe": selection.to_dict(),
            "label_summary": summarize_forward_labels(labels),
            "score_rows": len(scores),
            "label_rows": len(labels),
            "evaluation": evaluation,
            "artifacts": {
                "labels_csv": str(labels_path),
                "scores_csv": str(scores_path),
            },
            "lineage": {
                "run": run_context,
                "labels": label_lineage,
                "scores": score_lineage,
            },
        }
        report_path = self.file_store.save_json(report, filename=f"engine_{score_type}_evaluation.json", subdir="backtest")
        report["artifacts"]["report_json"] = str(report_path)
        return report

    def _select_universe(
        self,
        symbols: Sequence[str],
        start: date,
        end: date,
        horizons: Sequence[int],
        universe_policy: str,
        min_history_rows: int,
        interval: str,
    ):
        return BacktestUniverseSelector(
            store=self.store,
            venue=self.settings.runtime_data.canonical_venue,
            interval=interval,
        ).select(
            symbols=symbols,
            start=start,
            end=end,
            policy=universe_policy,
            min_history_rows=min_history_rows,
            horizons=horizons,
        )

    def _run_context(self, pipeline: str, end: date, interval: str) -> dict[str, str]:
        run_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return {
            "run_id": f"{pipeline}:{run_at.replace(':', '').replace('-', '')}:{uuid4().hex[:8]}",
            "pipeline": pipeline,
            "source_id": f"{self.settings.runtime_data.canonical_venue}:ohlcv_bars:{interval}",
            "source_timestamp": end.isoformat(),
            "ingested_at": run_at,
            "schema_version": BACKTEST_DATASET_SCHEMA_VERSION,
        }

    @staticmethod
    def _lineage_for_rows(
        run_context: dict[str, str],
        feature_version: str,
        model_version: str,
        quality_status: str,
    ) -> dict[str, str]:
        return {
            "run_id": run_context["run_id"],
            "source_id": run_context["source_id"],
            "source_timestamp": run_context["source_timestamp"],
            "ingested_at": run_context["ingested_at"],
            "quality_status": quality_status,
            "schema_version": run_context["schema_version"],
            "feature_version": feature_version,
            "model_version": model_version,
        }

    def close(self) -> None:
        self.store.close()


def _long_term_scorer(settings: AppSettings) -> LongTermScorer:
    w = settings.scoring.long_term_weights
    return LongTermScorer(
        weights=LongTermWeights(
            growth_quality=w.growth_quality,
            profitability_quality=w.profitability_quality,
            balance_sheet_health=w.balance_sheet_health,
            cash_flow_quality=w.cash_flow_quality,
            valuation_sanity=w.valuation_sanity,
            governance_proxy=w.governance_proxy,
            event_catalyst=w.event_catalyst,
            regime_tailwind=w.regime_tailwind,
        ),
        regime_switch=_regime_switch(settings),
        regime_profiles=settings.scoring.long_term_regime_profiles,
    )


def _swing_scorer(settings: AppSettings) -> SwingScorer:
    w = settings.scoring.swing_weights
    return SwingScorer(
        weights=SwingWeights(
            trend_strength=w.trend_strength,
            momentum_strength=w.momentum_strength,
            relative_strength_proxy=w.relative_strength_proxy,
            volatility_regime=w.volatility_regime,
            volume_confirmation=w.volume_confirmation,
            event_catalyst=w.event_catalyst,
            sentiment_score=w.sentiment_score,
        ),
        regime_switch=_regime_switch(settings),
        regime_profiles=settings.scoring.swing_regime_profiles,
    )


def _risk_scorer(settings: AppSettings) -> RiskPenaltyScorer:
    w = settings.scoring.risk_weights
    return RiskPenaltyScorer(
        max_penalty=settings.scoring.max_risk_penalty,
        weights=RiskPenaltyWeights(
            liquidity_risk=w.liquidity_risk,
            volatility_risk=w.volatility_risk,
            leverage_risk=w.leverage_risk,
            earnings_instability_risk=w.earnings_instability_risk,
            event_uncertainty_risk=w.event_uncertainty_risk,
            governance_risk=w.governance_risk,
            text_uncertainty_risk=w.text_uncertainty_risk,
        ),
    )


def _regime_switch(settings: AppSettings) -> RegimeSwitchConfig:
    r = settings.scoring.regime_switching
    return RegimeSwitchConfig(
        enabled=r.enabled,
        bull_threshold=r.bull_threshold,
        bear_threshold=r.bear_threshold,
    )
