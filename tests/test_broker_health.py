from __future__ import annotations

from datetime import date

from stock_screener_engine.app import run_broker_health


class _HealthBroker:
    def __init__(
        self,
        enabled: bool = True,
        quote_price: float = 100.0,
        close_price: float = 100.0,
        latest_date: str = "2026-05-27",
        fail_quote: bool = False,
        fail_history_symbols: set[str] | None = None,
    ) -> None:
        self._enabled = enabled
        self._quote_price = quote_price
        self._close_price = close_price
        self._latest_date = latest_date
        self._fail_quote = fail_quote
        self._fail_history_symbols = fail_history_symbols or set()

    def is_enabled(self) -> bool:
        return self._enabled

    def get_instruments(self) -> list[dict]:
        return []

    def get_quote(self, symbols):
        if self._fail_quote:
            raise RuntimeError("quote failed")
        return {symbol: {"ltp": self._quote_price, "volume": 1000} for symbol in symbols}

    def get_historical(self, symbol, interval, start, end):
        if symbol in self._fail_history_symbols:
            raise RuntimeError("historical failed")
        return [
            {
                "date": self._latest_date,
                "open": self._close_price,
                "high": self._close_price,
                "low": self._close_price,
                "close": self._close_price,
                "volume": 1000,
            }
        ]

    def place_order(self, order_request):
        return {}

    def get_positions(self):
        return []

    def get_holdings(self):
        return []

    def get_order_history(self, order_id):
        return []


def test_broker_health_compares_zerodha_and_icici_breeze(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))
    monkeypatch.setattr(
        "stock_screener_engine.app.build_broker_adapters",
        lambda settings: {
            "zerodha": _HealthBroker(quote_price=100.0, close_price=100.0),
            "breeze": _HealthBroker(quote_price=101.5, close_price=101.5),
        },
    )

    report = run_broker_health(
        start=date(2026, 5, 25),
        end=date(2026, 5, 27),
        symbols=["AAA", "BBB"],
        sources=["zerodha", "icici"],
        price_tolerance_pct=1.0,
    )

    assert report["passed"] is True
    assert report["sources"] == ["zerodha", "icici_breeze"]
    assert report["source_reports"]["zerodha"]["quote_coverage"] == 1.0
    assert report["source_reports"]["icici_breeze"]["historical_coverage"] == 1.0
    assert report["reconciliation"]["price_mismatch_count"] == 2
    assert report["reconciliation"]["preferred_source_counts"]["zerodha"] == 2
    assert (tmp_path / "quality" / "broker_health_report.json").exists()


def test_broker_health_reports_disabled_source_without_crashing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))
    monkeypatch.setattr(
        "stock_screener_engine.app.build_broker_adapters",
        lambda settings: {
            "zerodha": _HealthBroker(enabled=False),
            "breeze": _HealthBroker(quote_price=100.0, close_price=100.0),
        },
    )

    report = run_broker_health(
        start=date(2026, 5, 25),
        end=date(2026, 5, 27),
        symbols=["AAA"],
        sources=["zerodha", "breeze"],
    )

    assert report["passed"] is True
    assert report["source_reports"]["zerodha"]["enabled"] is False
    assert report["symbol_reports"][0]["sources"]["zerodha"]["errors"] == [
        "zerodha disabled or missing credentials"
    ]


def test_broker_health_surfaces_enabled_zero_coverage(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))
    monkeypatch.setattr(
        "stock_screener_engine.app.build_broker_adapters",
        lambda settings: {
            "breeze": _HealthBroker(quote_price=0.0, close_price=0.0),
        },
    )

    report = run_broker_health(
        start=date(2026, 5, 25),
        end=date(2026, 5, 27),
        symbols=["AAA"],
        sources=["icici_breeze"],
        retries=0,
    )

    assert report["source_reports"]["icici_breeze"]["source_errors"] == [
        "1 quote failures",
        "1 historical failures",
    ]
    assert "no usable quote returned" in report["symbol_reports"][0]["sources"]["icici_breeze"]["errors"]
    assert "no usable historical bars returned" in report["symbol_reports"][0]["sources"]["icici_breeze"]["errors"]


def test_broker_health_retries_transient_quote_and_history_failures(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))

    class _FlakyBroker(_HealthBroker):
        def __init__(self) -> None:
            super().__init__(quote_price=100.0, close_price=100.0)
            self.quote_calls = 0
            self.history_calls = 0

        def get_quote(self, symbols):
            self.quote_calls += 1
            if self.quote_calls == 1:
                raise RuntimeError("Too many requests")
            return super().get_quote(symbols)

        def get_historical(self, symbol, interval, start, end):
            self.history_calls += 1
            if self.history_calls == 1:
                raise RuntimeError("Too many requests")
            return super().get_historical(symbol, interval, start, end)

    broker = _FlakyBroker()
    monkeypatch.setattr(
        "stock_screener_engine.app.build_broker_adapters",
        lambda settings: {
            "zerodha": broker,
        },
    )

    report = run_broker_health(
        start=date(2026, 5, 25),
        end=date(2026, 5, 27),
        symbols=["AAA"],
        sources=["zerodha"],
        retries=1,
        retry_delay_seconds=0,
    )

    source = report["source_reports"]["zerodha"]
    view = report["symbol_reports"][0]["sources"]["zerodha"]
    assert source["quote_coverage"] == 1.0
    assert source["historical_coverage"] == 1.0
    assert source["quote_retry_symbols"] == ["AAA"]
    assert source["historical_retry_symbols"] == ["AAA"]
    assert view["quote_attempts"] == 2
    assert view["historical_attempts"] == 2


def test_broker_health_classifies_lagged_reconciliation_source(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))
    monkeypatch.setattr(
        "stock_screener_engine.app.build_broker_adapters",
        lambda settings: {
            "breeze": _HealthBroker(quote_price=100.0, close_price=100.0, latest_date="2026-05-26"),
        },
    )

    report = run_broker_health(
        start=date(2026, 5, 25),
        end=date(2026, 5, 27),
        symbols=["AAA"],
        sources=["icici_breeze"],
        retries=0,
        lagged_sources=["icici_breeze"],
    )

    source = report["source_reports"]["icici_breeze"]
    view = report["symbol_reports"][0]["sources"]["icici_breeze"]
    assert source["role"] == "lagged_reconciliation"
    assert source["stale_symbols"] == []
    assert source["lagged_symbols"] == ["AAA"]
    assert view["lagged"] is True
    assert view["staleness_status"] == "lagged_expected"


def test_broker_health_redacts_credentials_from_errors(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SSE_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("SSE_SQLITE_PATH", str(tmp_path / "market.db"))
    monkeypatch.setenv("SSE_ENABLE_ZERODHA", "true")
    monkeypatch.setenv("SSE_ZERODHA_API_KEY", "public_key")
    monkeypatch.setenv("SSE_ZERODHA_API_SECRET", "secret_value")
    monkeypatch.setenv("SSE_ZERODHA_ACCESS_TOKEN", "token_value")

    class _LeakyBroker(_HealthBroker):
        def get_quote(self, symbols):
            raise RuntimeError("token_value rejected")

    monkeypatch.setattr(
        "stock_screener_engine.app.build_broker_adapters",
        lambda settings: {
            "zerodha": _LeakyBroker(),
            "breeze": _HealthBroker(enabled=False),
        },
    )

    report = run_broker_health(
        start=date(2026, 5, 25),
        end=date(2026, 5, 27),
        symbols=["AAA"],
        sources=["zerodha"],
        retries=0,
    )

    error = report["symbol_reports"][0]["sources"]["zerodha"]["errors"][0]
    assert "token_value" not in error
    assert "[redacted] rejected" == error
