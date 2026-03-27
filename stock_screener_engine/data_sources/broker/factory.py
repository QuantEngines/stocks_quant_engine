"""Factory for optional broker adapter creation."""

from __future__ import annotations

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.data_sources.base.interfaces import BrokerAdapter
from stock_screener_engine.data_sources.broker.breeze_adapter import BreezeAdapter
from stock_screener_engine.data_sources.broker.zerodha_adapter import ZerodhaAdapter


def build_broker_adapters(settings: AppSettings) -> dict[str, BrokerAdapter]:
    return {
        "zerodha": ZerodhaAdapter(settings.integrations.zerodha),
        "breeze": BreezeAdapter(settings.integrations.breeze),
    }
