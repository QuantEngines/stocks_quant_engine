"""Optional ICICI Breeze adapter scaffold (disabled by default)."""

from __future__ import annotations

from datetime import date, datetime, time
import logging
from typing import Any, Iterable, cast

from stock_screener_engine.config.settings import BrokerIntegrationSettings
from stock_screener_engine.data_sources.base.interfaces import OrderRequest
from stock_screener_engine.data_sources.broker._optional_base import OptionalBrokerAdapterBase


logger = logging.getLogger(__name__)


class BreezeAdapter(OptionalBrokerAdapterBase):
    def __init__(self, settings: BrokerIntegrationSettings, client: object | None = None) -> None:
        super().__init__(enabled=settings.enabled, credentials=settings.credentials(), broker_name="breeze")
        self._client = client

    def _breeze(self):
        self._guard()
        if self._client is not None:
            return self._client
        try:
            from breeze_connect import BreezeConnect  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise RuntimeError("ICICI Breeze data source requires the optional 'breeze-connect' package") from exc

        creds = self.credentials
        breeze = BreezeConnect(api_key=creds["api_key"])
        breeze.generate_session(api_secret=creds["api_secret"], session_token=creds["token"])
        self._client = breeze
        return breeze

    def get_instruments(self) -> list[dict]:
        self._guard()
        return []

    def get_quote(self, symbols: Iterable[str]) -> dict[str, dict]:
        client = self._breeze()
        out: dict[str, dict] = {}
        for raw_symbol in symbols:
            symbol = str(raw_symbol).strip().upper()
            if not symbol:
                continue
            try:
                payload = client.get_quotes(stock_code=symbol, exchange_code="NSE")
            except Exception as exc:
                logger.warning("Breeze quote lookup failed for %s; continuing without quote: %s", symbol, exc)
                out[symbol] = {"ltp": 0.0, "last_price": 0.0, "volume": 0.0, "error": str(exc)}
                continue
            row = _first_success_row(payload)
            out[symbol] = {
                "ltp": _safe_float(_pick(row, "ltp", "last_price", "last")),
                "last_price": _safe_float(_pick(row, "ltp", "last_price", "last")),
                "volume": _safe_float(_pick(row, "total_quantity_traded", "volume")),
                "error": _payload_error(payload) if not row else "",
            }
        return out

    def get_historical(self, symbol: str, interval: str, start: date, end: date) -> list[dict]:
        payload = self._breeze().get_historical_data_v2(
            interval=_breeze_interval(interval),
            from_date=_breeze_dt(start, market_open=True),
            to_date=_breeze_dt(end, market_open=False),
            stock_code=symbol.strip().upper(),
            exchange_code="NSE",
            product_type="cash",
        )
        rows = payload.get("Success", []) if isinstance(payload, dict) else []
        return [_normalize_bar(row) for row in rows or []]

    def place_order(self, order_request: OrderRequest) -> dict:
        payload = self._breeze().place_order(
            stock_code=order_request.symbol,
            exchange_code="NSE",
            product="cash",
            action=order_request.side.lower(),
            order_type=order_request.order_type.lower(),
            quantity=str(order_request.quantity),
            price=str(order_request.price or 0),
            validity="day",
        )
        row = _first_success_row(payload)
        order_id = _pick(row, "order_id", "orderid", "OrderId")
        return {"status": "submitted" if order_id else "unknown", "order_id": order_id, "broker": self.broker_name}

    def get_positions(self) -> list[dict]:
        payload = self._breeze().get_portfolio_positions()
        rows = payload.get("Success", []) if isinstance(payload, dict) else []
        return list(rows or [])

    def get_holdings(self) -> list[dict]:
        payload = self._breeze().get_demat_holdings()
        rows = payload.get("Success", []) if isinstance(payload, dict) else []
        return list(rows or [])

    def get_order_history(self, order_id: str) -> list[dict]:
        payload = self._breeze().get_order_detail(exchange_code="NSE", order_id=order_id)
        rows = payload.get("Success", []) if isinstance(payload, dict) else []
        return list(rows or [])


def _breeze_interval(interval: str) -> str:
    mapping = {
        "1d": "1day",
        "day": "1day",
        "1m": "1minute",
        "5m": "5minute",
        "30m": "30minute",
    }
    return mapping.get(interval.lower(), interval)


def _breeze_dt(value: date, market_open: bool) -> str:
    t = time(9, 15) if market_open else time(15, 30)
    return datetime.combine(value, t).isoformat()


def _first_success_row(payload: object) -> dict:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("Success", payload)
    if isinstance(rows, list):
        return rows[0] if rows and isinstance(rows[0], dict) else {}
    return rows if isinstance(rows, dict) else {}


def _payload_error(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    for key in ("Error", "error", "Message", "message", "Status", "status"):
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _pick(row: dict, *keys: str) -> object:
    for key in keys:
        if key in row:
            return row[key]
    return None


def _normalize_bar(row: dict) -> dict:
    return {
        "date": str(_pick(row, "datetime", "date", "time") or ""),
        "open": _safe_float(_pick(row, "open", "open_price")),
        "high": _safe_float(_pick(row, "high", "high_price")),
        "low": _safe_float(_pick(row, "low", "low_price")),
        "close": _safe_float(_pick(row, "close", "close_price")),
        "volume": int(_safe_float(_pick(row, "volume", "total_quantity_traded"))),
    }


def _safe_float(value: object) -> float:
    try:
        return float(cast(Any, value)) if value is not None and str(value).strip() else 0.0
    except (TypeError, ValueError):
        return 0.0
