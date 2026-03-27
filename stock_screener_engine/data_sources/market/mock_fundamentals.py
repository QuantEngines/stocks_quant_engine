"""Mock FinancialsProvider for local research and demo workflows."""

from __future__ import annotations

from datetime import date
from typing import Sequence

from stock_screener_engine.core.entities import FundamentalsSnapshot, GovernanceSnapshot
from stock_screener_engine.data_sources.base.interfaces import FinancialsProvider


class MockFinancialsProvider(FinancialsProvider):
    """Returns deterministic synthetic fundamentals and governance data.

    All values are derived from the symbol string so the mock is fully
    repeatable without any external network calls.
    """

    def get_fundamentals(self, symbols: Sequence[str]) -> dict[str, FundamentalsSnapshot]:
        today = date.today()
        result: dict[str, FundamentalsSnapshot] = {}
        for idx, symbol in enumerate(symbols):
            factor = ((idx + 1) / (len(symbols) + 2)) if len(symbols) > 0 else 0.5
            result[symbol] = FundamentalsSnapshot(
                symbol=symbol,
                as_of=today,
                pe_ratio=12.0 + idx * 2.5,
                pb_ratio=1.5 + factor * 3.0,
                roe=0.10 + factor * 0.30,
                roa=0.05 + factor * 0.15,
                roce=0.12 + factor * 0.22,
                debt_to_equity=0.10 + (1.0 - factor) * 1.20,
                current_ratio=1.2 + factor * 1.8,
                interest_coverage=3.0 + factor * 12.0,
                earnings_growth_yoy=0.04 + factor * 0.40,
                revenue_growth_yoy=0.06 + factor * 0.30,
                free_cash_flow_margin=0.05 + factor * 0.30,
                operating_margin=0.08 + factor * 0.28,
                net_profit_margin=0.05 + factor * 0.20,
            )
        return result

    def get_governance(self, symbols: Sequence[str]) -> dict[str, GovernanceSnapshot]:
        today = date.today()
        result: dict[str, GovernanceSnapshot] = {}
        for idx, symbol in enumerate(symbols):
            factor = ((idx + 1) / (len(symbols) + 2)) if len(symbols) > 0 else 0.5
            result[symbol] = GovernanceSnapshot(
                symbol=symbol,
                as_of=today,
                promoter_holding_pct=40.0 + factor * 30.0,
                promoter_holding_change_qoq=-0.02 + factor * 0.08,
                institutional_holding_pct=20.0 + factor * 25.0,
                fii_holding_pct=5.0 + factor * 20.0,
                dii_holding_pct=10.0 + factor * 15.0,
                insider_activity_score=0.2 + factor * 0.6,
                audit_opinion="clean",
            )
        return result
