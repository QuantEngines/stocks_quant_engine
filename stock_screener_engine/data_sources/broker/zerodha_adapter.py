"""Optional Zerodha adapter scaffold (disabled by default)."""

from __future__ import annotations

from datetime import date
from typing import Any, Iterable, cast

from stock_screener_engine.config.settings import BrokerIntegrationSettings
from stock_screener_engine.data_sources.base.interfaces import OrderRequest
from stock_screener_engine.data_sources.broker._optional_base import OptionalBrokerAdapterBase


class ZerodhaAdapter(OptionalBrokerAdapterBase):
    def __init__(self, settings: BrokerIntegrationSettings, client: object | None = None) -> None:
        super().__init__(enabled=settings.enabled, credentials=settings.credentials(), broker_name="zerodha")
        self._client = client
        self._instrument_cache: list[dict] | None = None

    def _kite(self):
        self._guard()
        if self._client is not None:
            return self._client
        try:
            from kiteconnect import KiteConnect  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("Zerodha data source requires the optional 'kiteconnect' package") from exc

        creds = self.credentials
        kite = KiteConnect(api_key=creds["api_key"])
        kite.set_access_token(creds["token"])
        self._client = kite
        return kite

    def get_instruments(self) -> list[dict]:
        if self._instrument_cache is None:
            instruments = self._kite().instruments("NSE")
            self._instrument_cache = list(instruments or [])
        return list(self._instrument_cache)

    def get_quote(self, symbols: Iterable[str]) -> dict[str, dict]:
        symbols = [str(s).strip().upper() for s in symbols if str(s).strip()]
        if not symbols:
            return {}
        keys = [s if ":" in s else f"NSE:{s}" for s in symbols]
        raw = self._kite().quote(keys)
        out: dict[str, dict] = {}
        for symbol, key in zip(symbols, keys):
            payload = raw.get(key, {}) if isinstance(raw, dict) else {}
            out[symbol] = {
                "ltp": _safe_float(payload.get("last_price")),
                "last_price": _safe_float(payload.get("last_price")),
                "volume": _safe_float(payload.get("volume")),
                "ohlc": payload.get("ohlc", {}),
            }
        return out

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        token = self._instrument_token(symbol)
        rows = self._kite().historical_data(
            instrument_token=token,
            from_date=start,
            to_date=end,
            interval=_kite_interval(interval),
        )
        return [_normalize_bar(row) for row in rows or []]

    def place_order(self, order_request: OrderRequest) -> dict:
        kite = self._kite()
        payload = {
            "variety": getattr(kite, "VARIETY_REGULAR", "regular"),
            "exchange": getattr(kite, "EXCHANGE_NSE", "NSE"),
            "tradingsymbol": order_request.symbol,
            "transaction_type": order_request.side,
            "quantity": order_request.quantity,
            "product": getattr(kite, "PRODUCT_CNC", "CNC"),
            "order_type": order_request.order_type,
        }
        if order_request.price is not None:
            payload["price"] = order_request.price
        order_id = kite.place_order(**payload)
        return {"status": "submitted", "order_id": str(order_id), "broker": self.broker_name}

    def get_positions(self) -> list[dict]:
        payload = self._kite().positions()
        if isinstance(payload, dict):
            return list(payload.get("net", []))
        return list(payload or [])

    def get_holdings(self) -> list[dict]:
        return list(self._kite().holdings() or [])

    def get_order_history(self, order_id: str) -> list[dict]:
        return list(self._kite().order_history(order_id) or [])

    def _instrument_token(self, symbol: str) -> int:
        clean = symbol.split(":", maxsplit=1)[-1].strip().upper()
        for row in self.get_instruments():
            if str(row.get("tradingsymbol", "")).strip().upper() == clean:
                token = row.get("instrument_token")
                if token is not None:
                    return int(token)
        raise RuntimeError(f"Zerodha instrument token not found for {symbol}")


def _kite_interval(interval: str) -> str:
    mapping = {
        "1d": "day",
        "day": "day",
        "1m": "minute",
        "3m": "3minute",
        "5m": "5minute",
        "10m": "10minute",
        "15m": "15minute",
        "30m": "30minute",
        "60m": "60minute",
    }
    return mapping.get(interval.lower(), interval)


def _normalize_bar(row: dict) -> dict:
    ts = row.get("date") or row.get("timestamp") or row.get("datetime")
    date_text = ts.isoformat() if ts is not None and hasattr(ts, "isoformat") else str(ts or "")
    return {
        "date": date_text,
        "open": _safe_float(row.get("open")),
        "high": _safe_float(row.get("high")),
        "low": _safe_float(row.get("low")),
        "close": _safe_float(row.get("close")),
        "volume": int(_safe_float(row.get("volume"))),
    }


def _safe_float(value: object) -> float:
    try:
        return float(cast(Any, value)) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
