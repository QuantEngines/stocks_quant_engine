from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from stock_screener_engine.backtest.costs import IndianEquityCostModel
from stock_screener_engine.backtest.dataset import (
    BacktestUniverseSelector,
    EngineScoreDatasetBuilder,
    ForwardReturnLabelBuilder,
)
from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.data_sources.schemas import OHLCVBar
from stock_screener_engine.pipelines.backtest_dataset import BacktestDatasetPipeline
from stock_screener_engine.storage.market_data_store import MarketDataStore


def test_backtest_universe_selector_supports_current_and_history_eligible(tmp_path: Path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    start = date(2026, 1, 1)
    try:
        store.upsert_ohlcv(_bars("AAA", start, 12))
        store.upsert_ohlcv(_bars("BBB", start, 4))

        selector = BacktestUniverseSelector(store=store)
        current = selector.select(["AAA", "BBB"], start=start, end=start + timedelta(days=20), policy="current")
        eligible = selector.select(
            ["AAA", "BBB"],
            start=start,
            end=start + timedelta(days=20),
            policy="eligible_history",
            min_history_rows=10,
            horizons=[1, 5],
        )

        assert current.selected_symbols == ["AAA", "BBB"]
        assert eligible.selected_symbols == ["AAA"]
        assert "BBB" in eligible.rejected_symbols
    finally:
        store.close()


def test_forward_return_label_builder_generates_close_to_close_labels(tmp_path: Path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    start = date(2026, 1, 1)
    try:
        store.upsert_ohlcv(_bars("AAA", start, 5, start_price=100.0, step=10.0))

        labels = ForwardReturnLabelBuilder(store=store).build(
            symbols=["AAA"],
            start=start,
            end=start + timedelta(days=10),
            horizons=[1, 2],
        )

        first = labels[0]
        assert first.as_of == "2026-01-01"
        assert first.horizon == 1
        assert round(first.forward_return, 4) == 0.1
        assert first.schema_version == "backtest_dataset.v1"
        assert first.feature_version == "forward_return_label.v1"
        assert len(labels) == 7
    finally:
        store.close()


def test_backtest_dataset_pipeline_persists_labels_and_technical_evaluation(tmp_path: Path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "market.db")),
    )
    store = MarketDataStore(settings.storage.sqlite_path)
    start = date(2026, 1, 1)
    try:
        for idx, symbol in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
            store.upsert_ohlcv(_bars(symbol, start, 30, start_price=100.0 + idx, step=1.0 + idx * 0.2))
        pipeline = BacktestDatasetPipeline(settings=settings, store=store)

        labels_report = pipeline.build_forward_labels(
            symbols=["AAA", "BBB", "CCC", "DDD", "EEE"],
            start=start,
            end=start + timedelta(days=40),
            horizons=[1, 5],
            min_history_rows=10,
        )
        evaluation = pipeline.evaluate_technical_ranking(
            symbols=["AAA", "BBB", "CCC", "DDD", "EEE"],
            start=start,
            end=start + timedelta(days=40),
            horizons=[1, 5],
            min_history_rows=10,
            min_lookback=5,
        )

        assert labels_report["labels_persisted"] > 0
        assert labels_report["lineage"]["run"]["run_id"]
        assert labels_report["lineage"]["labels"]["feature_version"] == "forward_return_label.v1"
        assert Path(str(labels_report["labels_path"])).exists()
        assert evaluation["score_rows"] > 0
        assert evaluation["evaluation"]["rows_evaluated"] > 0
        assert evaluation["lineage"]["scores"]["model_version"] == "technical_ranking.v1"
        assert Path(str(evaluation["artifacts"]["report_json"])).exists()
        scores_header = Path(str(evaluation["artifacts"]["scores_csv"])).read_text(encoding="utf-8").splitlines()[0]
        assert "run_id" in scores_header
        assert "feature_version" in scores_header
    finally:
        store.close()


def test_engine_score_builder_uses_engine_scoring_stack(tmp_path: Path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    start = date(2026, 1, 1)
    try:
        store.upsert_ohlcv(_bars("AAA", start, 30, start_price=100.0, step=1.0))

        rows = EngineScoreDatasetBuilder(store=store, min_lookback=5).build_scores(
            symbols=["AAA"],
            start=start,
            end=start + timedelta(days=40),
            max_horizon=5,
            score_type="swing",
        )

        assert rows
        assert rows[0].symbol == "AAA"
        assert rows[0].score_type == "swing"
        assert rows[0].swing_score >= 0.0
        assert rows[0].schema_version == "backtest_dataset.v1"
        assert rows[0].feature_version == "engine_feature_stack.v1"
        assert rows[0].to_flat_dict()["model_version"] == "engine_scoring.v1"
        assert "swing_trend_strength" in rows[0].components
    finally:
        store.close()


def test_dataset_builders_accept_explicit_lineage(tmp_path: Path) -> None:
    store = MarketDataStore(str(tmp_path / "market.db"))
    start = date(2026, 1, 1)
    lineage = {
        "run_id": "unit-run",
        "source_id": "NSE:ohlcv_bars:1d",
        "source_timestamp": "2026-01-31",
        "ingested_at": "2026-02-01T00:00:00Z",
        "quality_status": "passed",
        "schema_version": "backtest_dataset.v1",
        "feature_version": "custom_feature.v1",
        "model_version": "custom_model.v1",
    }
    try:
        store.upsert_ohlcv(_bars("AAA", start, 30, start_price=100.0, step=1.0))

        labels = ForwardReturnLabelBuilder(store=store).build(
            symbols=["AAA"],
            start=start,
            end=start + timedelta(days=40),
            horizons=[1],
            lineage=lineage,
        )
        scores = EngineScoreDatasetBuilder(store=store, min_lookback=5).build_scores(
            symbols=["AAA"],
            start=start,
            end=start + timedelta(days=40),
            max_horizon=5,
            score_type="swing",
            lineage=lineage,
        )

        assert labels[0].run_id == "unit-run"
        assert labels[0].source_id == "NSE:ohlcv_bars:1d"
        assert labels[0].model_version == "custom_model.v1"
        assert scores[0].run_id == "unit-run"
        assert scores[0].quality_status == "passed"
        assert scores[0].to_flat_dict()["feature_version"] == "custom_feature.v1"
    finally:
        store.close()


def test_transaction_cost_model_reduces_forward_return() -> None:
    model = IndianEquityCostModel(explicit_round_trip_bps=25.0)

    assert model.round_trip_fraction() == 0.0025
    assert model.net_return(0.05) == 0.0475


def _bars(
    symbol: str,
    start: date,
    count: int,
    start_price: float = 100.0,
    step: float = 1.0,
) -> list[OHLCVBar]:
    rows: list[OHLCVBar] = []
    for i in range(count):
        price = start_price + i * step
        rows.append(
            OHLCVBar(
                venue="NSE",
                symbol=symbol,
                ts=(start + timedelta(days=i)).isoformat(),
                open=price,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000.0 + i,
            )
        )
    return rows
