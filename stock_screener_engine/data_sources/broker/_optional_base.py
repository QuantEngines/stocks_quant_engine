"""Common optional-broker behavior with graceful failure semantics."""

from __future__ import annotations

from datetime import date
from typing import Iterable

from stock_screener_engine.data_sources.base.interfaces import BrokerAdapter, OrderRequest


class OptionalBrokerAdapterBase(BrokerAdapter):
    def __init__(self, enabled: bool, credentials: dict[str, str | None], broker_name: str) -> None:
        self._enabled = enabled
        self._credentials = credentials
        self._broker_name = broker_name

    def is_enabled(self) -> bool:
        required = all(self._credentials.values())
        return self._enabled and required

    def _guard(self) -> None:
        if not self.is_enabled():
            raise RuntimeError(
                f"{self._broker_name} adapter disabled or missing credentials. "
                f"Core screener can run without broker APIs."
            )

    def get_instruments(self) -> list[dict]:
        self._guard()
        return []

    def get_quote(self, symbols: Iterable[str]) -> dict[str, dict]:
        self._guard()
        return {symbol: {} for symbol in symbols}

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        self._guard()
        return []

    def place_order(self, order_request: OrderRequest) -> dict:
        self._guard()
        return {"status": "not_implemented", "broker": self._broker_name, "symbol": order_request.symbol}

    def get_positions(self) -> list[dict]:
        self._guard()
        return []

    def get_holdings(self) -> list[dict]:
        self._guard()
        return []

    def get_order_history(self, order_id: str) -> list[dict]:
        self._guard()
        return []
