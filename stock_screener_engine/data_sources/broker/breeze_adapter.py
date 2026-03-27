"""Optional ICICI Breeze adapter scaffold (disabled by default)."""

from __future__ import annotations

from stock_screener_engine.config.settings import BrokerIntegrationSettings
from stock_screener_engine.data_sources.broker._optional_base import OptionalBrokerAdapterBase


class BreezeAdapter(OptionalBrokerAdapterBase):
    def __init__(self, settings: BrokerIntegrationSettings) -> None:
        super().__init__(enabled=settings.enabled, credentials=settings.credentials(), broker_name="breeze")
