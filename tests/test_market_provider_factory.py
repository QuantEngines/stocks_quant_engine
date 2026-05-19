from __future__ import annotations

from dataclasses import replace

from stock_screener_engine.app import _build_financials_provider, _build_market_provider
from stock_screener_engine.config.settings import (
    BrokerIntegrationSettings,
    IntegrationSettings,
    RuntimeDataSettings,
    load_settings,
)
from stock_screener_engine.data_sources.market.broker_market_data_provider import BrokerMarketDataProvider
from stock_screener_engine.data_sources.financials.sqlite_financials_provider import SQLiteFinancialsProvider
from stock_screener_engine.data_sources.market.http_market_data_provider import NSEHTTPMarketDataProvider
from stock_screener_engine.data_sources.market.sqlite_market_data_provider import SQLiteMarketDataProvider


def test_build_market_provider_uses_nse_http_for_nse_http_config() -> None:
    settings = load_settings()
    settings = replace(
        settings,
        runtime_data=RuntimeDataSettings(
            market_provider="nse_http",
            market_universe=["RELIANCE"],
            news_provider=settings.runtime_data.news_provider,
            filings_provider=settings.runtime_data.filings_provider,
            transcripts_provider=settings.runtime_data.transcripts_provider,
        ),
    )

    provider = _build_market_provider(settings)
    assert isinstance(provider, NSEHTTPMarketDataProvider)


def test_build_market_provider_uses_canonical_sqlite_source(tmp_path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, sqlite_path=str(tmp_path / "market.db")),
        runtime_data=RuntimeDataSettings(
            market_provider="canonical",
            market_universe=["RELIANCE"],
            news_provider=settings.runtime_data.news_provider,
            filings_provider=settings.runtime_data.filings_provider,
            transcripts_provider=settings.runtime_data.transcripts_provider,
        ),
    )

    provider = _build_market_provider(settings)
    try:
        assert isinstance(provider, SQLiteMarketDataProvider)
    finally:
        provider.close()


def test_canonical_market_source_auto_uses_canonical_financials(tmp_path) -> None:
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, sqlite_path=str(tmp_path / "market.db")),
        runtime_data=RuntimeDataSettings(
            market_provider="canonical",
            market_universe=["RELIANCE"],
            news_provider=settings.runtime_data.news_provider,
            filings_provider=settings.runtime_data.filings_provider,
            transcripts_provider=settings.runtime_data.transcripts_provider,
        ),
    )

    provider = _build_financials_provider(settings)
    try:
        assert isinstance(provider, SQLiteFinancialsProvider)
    finally:
        provider.close()


def test_build_market_provider_can_use_zerodha_as_market_source(monkeypatch) -> None:
    monkeypatch.setenv("Z_API_KEY", "key")
    monkeypatch.setenv("Z_API_SECRET", "secret")
    monkeypatch.setenv("Z_TOKEN", "token")
    settings = load_settings()
    zerodha = BrokerIntegrationSettings(
        enabled=True,
        api_key_env="Z_API_KEY",
        api_secret_env="Z_API_SECRET",
        token_env="Z_TOKEN",
    )
    settings = replace(
        settings,
        runtime_data=RuntimeDataSettings(
            market_provider="zerodha",
            market_universe=["RELIANCE"],
            news_provider=settings.runtime_data.news_provider,
            filings_provider=settings.runtime_data.filings_provider,
            transcripts_provider=settings.runtime_data.transcripts_provider,
        ),
        integrations=IntegrationSettings(zerodha=zerodha, breeze=settings.integrations.breeze),
    )

    provider = _build_market_provider(settings)
    assert isinstance(provider, BrokerMarketDataProvider)


def test_build_market_provider_can_use_icici_breeze_as_market_source(monkeypatch) -> None:
    monkeypatch.setenv("B_API_KEY", "key")
    monkeypatch.setenv("B_API_SECRET", "secret")
    monkeypatch.setenv("B_TOKEN", "token")
    settings = load_settings()
    breeze = BrokerIntegrationSettings(
        enabled=True,
        api_key_env="B_API_KEY",
        api_secret_env="B_API_SECRET",
        token_env="B_TOKEN",
    )
    settings = replace(
        settings,
        runtime_data=RuntimeDataSettings(
            market_provider="icici_breeze",
            market_universe=["RELIANCE"],
            news_provider=settings.runtime_data.news_provider,
            filings_provider=settings.runtime_data.filings_provider,
            transcripts_provider=settings.runtime_data.transcripts_provider,
        ),
        integrations=IntegrationSettings(zerodha=settings.integrations.zerodha, breeze=breeze),
    )

    provider = _build_market_provider(settings)
    assert isinstance(provider, BrokerMarketDataProvider)
