"""Exchange adapter scaffold for NSE/BSE feeds."""

from __future__ import annotations

from typing import Sequence

from stock_screener_engine.data_sources.base.interfaces import ExchangeAdapter


class NSEExchangeAdapter(ExchangeAdapter):
    """Placeholder exchange adapter that satisfies the ``ExchangeAdapter`` contract.

    TODO:
    - Integrate official exchange bhavcopy/announcement feeds via NSE's
      data API or a compliant third-party data vendor.
    - Normalize corporate actions (dividends, splits, bonuses, buybacks).
    - Normalize exchange notices (SEBI orders, trading halt notices).
    """

    def get_corporate_actions(self, symbols: Sequence[str]) -> dict[str, list[dict]]:
        return {symbol: [] for symbol in symbols}

    def get_exchange_announcements(self, symbols: Sequence[str]) -> dict[str, list[dict]]:
        return {symbol: [] for symbol in symbols}
