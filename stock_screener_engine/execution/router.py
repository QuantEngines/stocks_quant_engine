"""Broker router that can be no-op when integration is disabled."""

from __future__ import annotations

import logging

from stock_screener_engine.data_sources.base.interfaces import BrokerAdapter, OrderRequest
from stock_screener_engine.execution.interfaces import ExecutionOrder, ExecutionResult

logger = logging.getLogger(__name__)


class ExecutionRouter:
    def __init__(self, broker: BrokerAdapter | None = None) -> None:
        self.broker = broker

    def submit(self, order: ExecutionOrder) -> ExecutionResult:
        if self.broker is None or not self.broker.is_enabled():
            logger.info("Execution skipped: broker integration disabled")
            return ExecutionResult(status="skipped", broker_order_id=None, message="Broker disabled")

        payload = self.broker.place_order(
            OrderRequest(
                symbol=order.symbol,
                side=order.side.value,
                quantity=order.quantity,
                order_type=order.order_type.value,
                price=order.limit_price,
            )
        )
        return ExecutionResult(
            status=str(payload.get("status", "unknown")),
            broker_order_id=payload.get("order_id"),
            message=str(payload.get("message", "")),
        )
