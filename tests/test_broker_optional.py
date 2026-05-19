from __future__ import annotations

from stock_screener_engine.config.settings import BrokerIntegrationSettings
from stock_screener_engine.data_sources.market.broker_market_data_provider import BrokerMarketDataProvider
from stock_screener_engine.data_sources.broker.breeze_adapter import BreezeAdapter
from stock_screener_engine.data_sources.broker.zerodha_adapter import ZerodhaAdapter


def test_broker_adapters_disabled_without_credentials() -> None:
    settings = BrokerIntegrationSettings(
        enabled=True,
        api_key_env="MISSING_A",
        api_secret_env="MISSING_B",
        token_env="MISSING_C",
    )

    zerodha = ZerodhaAdapter(settings)
    breeze = BreezeAdapter(settings)

    assert zerodha.is_enabled() is False
    assert breeze.is_enabled() is False


class _EnabledBroker:
    def is_enabled(self) -> bool:
        return True

    def get_instruments(self) -> list[dict]:
        return [{"exchange": "NSE", "tradingsymbol": "RELIANCE"}]

    def get_quote(self, symbols):
        return {symbol: {"ltp": 2500.0, "volume": 1_200_000} for symbol in symbols}

    def get_historical(self, symbol, interval, start, end):
        return [
            {"date": start.isoformat(), "open": 2450.0, "high": 2510.0, "low": 2440.0, "close": 2500.0, "volume": 1_100_000}
        ]

    def place_order(self, order_request):
        return {}

    def get_positions(self):
        return []

    def get_holdings(self):
        return []

    def get_order_history(self, order_id):
        return []


def test_broker_market_data_provider_builds_market_only_snapshot() -> None:
    provider = BrokerMarketDataProvider(_EnabledBroker(), universe=["RELIANCE"], broker_name="dummy")
    snapshots = provider.get_snapshots(["RELIANCE"])

    assert len(snapshots) == 1
    assert snapshots[0].close == 2500.0
    assert snapshots[0].pe_ratio == 0.0
    assert snapshots[0].roe == 0.0
