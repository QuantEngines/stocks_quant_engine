"""Core immutable domain entities exchanged between modules.

Design notes
------------
* ``MarketSnapshot``      – daily OHLCV and derived market data.
* ``FundamentalsSnapshot`` – quarterly/annual financial metrics.
* ``GovernanceSnapshot``  – shareholding patterns and insider signals.
* ``StockSnapshot``       – unified convenience type for demos/tests; in production
                            prefer the three granular types above.
* ``FeatureVector``       – named float features keyed by constants in ``feature_specs``.
* ``ScoreCard``           – all three adjusted scores + component breakdown.
* ``SignalExplanation``   – human-readable explanation payload.
* ``SignalResult``        – final signal with category and explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping


# ---------------------------------------------------------------------------
# Granular domain snapshots (preferred for production data flows)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarketSnapshot:
    """Daily OHLCV and derived market data — updates every trading day."""

    symbol: str
    as_of: date
    sector: str
    exchange: str = "NSE"
    close: float = 0.0
    open_price: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: float = 0.0
    delivery_volume: float = 0.0
    delivery_ratio: float = 0.0
    market_cap: float = 0.0
    avg_volume_20d: float = 0.0


@dataclass(frozen=True)
class FundamentalsSnapshot:
    """Quarterly/annual financial metrics — update frequency: quarterly.

    ``as_of`` is the fiscal period-end date, NOT the filing date.
    """

    symbol: str
    as_of: date
    pe_ratio: float = 0.0
    pb_ratio: float = 0.0
    roe: float = 0.0
    roa: float = 0.0
    roce: float = 0.0
    debt_to_equity: float = 0.0
    current_ratio: float = 0.0
    interest_coverage: float = 0.0
    earnings_growth_yoy: float = 0.0
    revenue_growth_yoy: float = 0.0
    free_cash_flow_margin: float = 0.0
    operating_margin: float = 0.0
    net_profit_margin: float = 0.0


@dataclass(frozen=True)
class GovernanceSnapshot:
    """Shareholding patterns and insider activity — update frequency: quarterly.

    ``insider_activity_score``:
        positive  → net insider buying (bullish governance signal)
        negative  → net insider selling (caution)
    ``audit_opinion``: "clean", "qualified", "adverse", or "unknown"
    """

    symbol: str
    as_of: date
    promoter_holding_pct: float = 0.0
    promoter_holding_change_qoq: float = 0.0
    institutional_holding_pct: float = 0.0
    fii_holding_pct: float = 0.0
    dii_holding_pct: float = 0.0
    insider_activity_score: float = 0.0
    audit_opinion: str = "clean"


# ---------------------------------------------------------------------------
# Unified convenience snapshot (demos, tests, simple integrations)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StockSnapshot:
    """Unified convenience snapshot combining market + fundamentals + governance.

    Use the granular types (``MarketSnapshot``, ``FundamentalsSnapshot``,
    ``GovernanceSnapshot``) in production data flows where the adapters serve
    data at the appropriate frequencies.
    """

    symbol: str
    as_of: date
    sector: str
    close: float
    volume: float
    delivery_ratio: float
    pe_ratio: float
    roe: float
    debt_to_equity: float
    earnings_growth: float
    free_cash_flow_margin: float
    promoter_holding_change: float
    insider_activity_score: float


# ---------------------------------------------------------------------------
# Downstream pipeline types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureVector:
    symbol: str
    as_of: date
    values: Mapping[str, float]


@dataclass(frozen=True)
class ScoreCard:
    symbol: str
    as_of: date
    long_term_score: float
    swing_score: float
    risk_penalty: float
    conviction: float
    component_scores: Mapping[str, float]


@dataclass(frozen=True)
class SignalExplanation:
    signal_type: str
    score: float
    top_positive_drivers: list[str] = field(default_factory=list)
    top_negative_drivers: list[str] = field(default_factory=list)
    ranking_reason: str = ""
    rejection_reason: str | None = None
    holding_horizon: str = ""
    risk_flags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    entry_logic: str = "placeholder"
    invalidation_logic: str = "placeholder"


@dataclass(frozen=True)
class SignalResult:
    symbol: str
    category: str
    score: float
    explanation: SignalExplanation
    sector: str = ""
