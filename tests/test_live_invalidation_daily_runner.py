from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

from stock_screener_engine.config.settings import StorageSettings, load_settings
from stock_screener_engine.pipelines.live_invalidation_daily import run_live_invalidation_daily_job


class _DummyBroker:
    def is_enabled(self) -> bool:
        return True

    def get_positions(self) -> list[dict]:
        return [
            {
                "symbol": "AAA",
                "quantity": 10,
                "avg_price": 100.0,
                "ltp": 89.0,
                "entry_date": (date.today() - timedelta(days=2)).isoformat(),
                "thesis_flags": ["earnings_momentum"],
            },
            {
                "symbol": "BBB",
                "quantity": 0,
                "avg_price": 200.0,
                "ltp": 198.0,
            },
        ]

    def get_quote(self, symbols: list[str]) -> dict[str, dict]:
        return {symbol: {"ltp": 90.0} for symbol in symbols}


def test_run_live_invalidation_daily_writes_dated_reports(tmp_path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=StorageSettings(
            root_dir=str(tmp_path),
            sqlite_path=str(tmp_path / "metadata.db"),
        ),
    )

    out = run_live_invalidation_daily_job(settings, adapters={"dummy": _DummyBroker()})
    assert out["brokers_checked"] == ["dummy"]
    assert out["positions_evaluated"] == 1

    date_label = date.today().isoformat()
    json_path = tmp_path / "signals" / f"live_invalidation_{date_label}.json"
    csv_path = tmp_path / "signals" / f"live_invalidation_{date_label}.csv"
    assert json_path.exists()
    assert csv_path.exists()
