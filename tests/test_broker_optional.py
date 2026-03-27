from __future__ import annotations

from stock_screener_engine.config.settings import BrokerIntegrationSettings
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
