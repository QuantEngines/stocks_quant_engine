"""Execution abstractions separated from research engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


@dataclass(frozen=True)
class ExecutionOrder:
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    broker_order_id: str | None
    message: str
