"""Daily live invalidation job wired to broker open positions."""

from __future__ import annotations

import logging
from datetime import date, datetime

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.data_sources.base.interfaces import BrokerAdapter
from stock_screener_engine.data_sources.broker.factory import build_broker_adapters
from stock_screener_engine.monitoring.live_invalidation import ActiveSignal
from stock_screener_engine.pipelines.live_invalidation import LiveInvalidationPipeline

logger = logging.getLogger(__name__)


def run_live_invalidation_daily_job(
    settings: AppSettings,
    adapters: dict[str, BrokerAdapter] | None = None,
) -> dict[str, object]:
    adapters = adapters or build_broker_adapters(settings)
    invalidation = LiveInvalidationPipeline(settings)

    active_signals: list[ActiveSignal] = []
    latest_price_by_symbol: dict[str, float] = {}
    thesis_flags_by_symbol: dict[str, list[str]] = {}
    brokers_checked: list[str] = []

    for broker_name, adapter in adapters.items():
        if not adapter.is_enabled():
            continue

        brokers_checked.append(broker_name)
        try:
            positions = adapter.get_positions()
        except RuntimeError as exc:
            logger.warning("Skipping broker %s: %s", broker_name, exc)
            continue

        quotes: dict[str, dict] = {}
        symbols_needing_quote: list[str] = []
        for row in positions:
            symbol = extract_symbol(row)
            if symbol and extract_latest_price(row) <= 0.0:
                symbols_needing_quote.append(symbol)
        if symbols_needing_quote:
            try:
                quotes = adapter.get_quote(symbols_needing_quote)
            except RuntimeError as exc:
                logger.warning("Quote fetch failed for %s: %s", broker_name, exc)
                quotes = {}

        for row in positions:
            signal = position_to_active_signal(row)
            if signal is None:
                continue
            active_signals.append(signal)

            price = extract_latest_price(row)
            if price <= 0.0:
                price = extract_quote_price(quotes.get(signal.symbol, {}))
            latest_price_by_symbol[signal.symbol] = price
            thesis_flags_by_symbol[signal.symbol] = extract_thesis_flags(row)

    payload = invalidation.run(
        active_signals=active_signals,
        latest_price_by_symbol=latest_price_by_symbol,
        thesis_flags_by_symbol=thesis_flags_by_symbol,
        as_of=date.today(),
        run_label=date.today().isoformat(),
    )
    return {
        "brokers_checked": brokers_checked,
        "positions_evaluated": len(active_signals),
        "report": payload,
    }


def extract_symbol(position: dict) -> str:
    for key in ("symbol", "tradingsymbol", "trading_symbol", "instrument", "ticker"):
        value = position.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return ""


def position_to_active_signal(position: dict) -> ActiveSignal | None:
    symbol = extract_symbol(position)
    if not symbol:
        return None

    qty = extract_float(position, "quantity", "qty", "net_qty", "net_quantity", default=1.0)
    if abs(qty) < 1e-9:
        return None

    side_text = str(position.get("side") or position.get("direction") or "").strip().lower()
    if "short" in side_text or "sell" in side_text:
        side = "short"
    elif "long" in side_text or "buy" in side_text:
        side = "long"
    else:
        side = "short" if qty < 0 else "long"

    entry_price = extract_float(
        position,
        "entry_price",
        "avg_price",
        "average_price",
        "buy_price",
        "price",
        default=0.0,
    )
    if entry_price <= 0.0:
        return None

    entered_on = extract_date(position.get("entered_on") or position.get("entry_date") or position.get("buy_date"))
    stop_loss_pct = extract_float(position, "stop_loss_pct", "stop_pct", default=0.08)
    max_holding_days = int(extract_float(position, "max_holding_days", "holding_days_limit", default=30.0))

    return ActiveSignal(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        entered_on=entered_on,
        stop_loss_pct=max(0.0, stop_loss_pct),
        max_holding_days=max(1, max_holding_days),
        required_thesis_flags=extract_thesis_flags(position),
    )


def extract_latest_price(position: dict) -> float:
    return extract_float(position, "ltp", "last_price", "mark_price", "close", "price", default=0.0)


def extract_quote_price(quote: dict) -> float:
    return extract_float(quote, "ltp", "last_price", "price", "close", default=0.0)


def extract_thesis_flags(position: dict) -> list[str]:
    raw = position.get("thesis_flags") or position.get("required_thesis_flags") or []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [x.strip() for x in raw.split(",") if x.strip()]
    return []


def extract_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y/%m/%d"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                pass
    return date.today()


def extract_float(position: dict, *keys: str, default: float) -> float:
    for key in keys:
        value = position.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float(default)
