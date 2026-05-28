"""Bank/NBFC-specific feature helpers.

The core engine treats bank factors as optional. When the financials provider
can supply them, these helpers expose normalized feature values for conviction,
reporting, and later sector-specific scoring. When they are absent, financial
sector names carry an explicit coverage gap instead of silently relying only on
generic industrial ratios.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


BANKING_SECTOR_APPLICABLE = "banking_sector_applicable"
BANKING_FACTOR_AVAILABLE = "banking_factor_available"
BANKING_METRIC_COVERAGE = "banking_metric_coverage"
BANKING_DATA_CONFIDENCE = "banking_data_confidence"
BANKING_FACTOR_GAP = "financial_sector_factor_gap"
BANKING_QUALITY_SCORE = "banking_quality_score"
BANK_ASSET_QUALITY_SCORE = "bank_asset_quality_score"
BANK_CAPITAL_STRENGTH_SCORE = "bank_capital_strength_score"
BANK_FRANCHISE_STRENGTH_SCORE = "bank_franchise_strength_score"
BANK_PROFITABILITY_SCORE = "bank_profitability_score"
BANK_EFFICIENCY_SCORE = "bank_efficiency_score"

BANK_NII = "bank_net_interest_income"
BANK_NIM_PCT = "bank_nim_pct"
BANK_ADVANCES_GROWTH_PCT = "bank_advances_growth_pct"
BANK_DEPOSITS_GROWTH_PCT = "bank_deposits_growth_pct"
BANK_CASA_PCT = "bank_casa_pct"
BANK_GNPA_PCT = "bank_gnpa_pct"
BANK_NNPA_PCT = "bank_nnpa_pct"
BANK_PCR_PCT = "bank_provision_coverage_pct"
BANK_CREDIT_COST_PCT = "bank_credit_cost_pct"
BANK_CAR_PCT = "bank_capital_adequacy_pct"
BANK_CET1_PCT = "bank_cet1_pct"
BANK_COST_INCOME_PCT = "bank_cost_to_income_pct"
BANK_ROA_PCT = "bank_roa_pct"
BANK_ROE_PCT = "bank_roe_pct"
BANK_LDR_PCT = "bank_loan_to_deposit_pct"


_BANKING_METRIC_ATTRS = (
    "net_interest_income",
    "net_interest_margin_pct",
    "advances_growth_pct",
    "deposits_growth_pct",
    "casa_ratio_pct",
    "gnpa_ratio_pct",
    "nnpa_ratio_pct",
    "provision_coverage_ratio_pct",
    "credit_cost_pct",
    "capital_adequacy_ratio_pct",
    "cet1_ratio_pct",
    "cost_to_income_ratio_pct",
    "roa_pct",
    "roe_pct",
    "loan_to_deposit_ratio_pct",
)


def banking_features_for(*, sector: str, industry: str = "", record: object | None = None) -> dict[str, float]:
    """Return normalized bank-factor features for a symbol.

    Scores are 0..1. Raw banking ratios retain percentage units so reports can
    show the same values analysts expect, for example NIM 3.12 rather than
    0.0312.
    """

    applicable = _is_financial_business(sector=sector, industry=industry)
    if not applicable:
        return {BANKING_SECTOR_APPLICABLE: 0.0}

    if record is None:
        return {
            BANKING_SECTOR_APPLICABLE: 1.0,
            BANKING_FACTOR_AVAILABLE: 0.0,
            BANKING_METRIC_COVERAGE: 0.0,
            BANKING_DATA_CONFIDENCE: 0.0,
            BANKING_FACTOR_GAP: 1.0,
            BANKING_QUALITY_SCORE: 0.0,
        }

    coverage = _metric_coverage(record)
    asset_quality = _weighted_score(
        (
            (_inverse_pct_score_optional(_value(record, "gnpa_ratio_pct"), 0.0, 8.0), 0.45),
            (_inverse_pct_score_optional(_value(record, "nnpa_ratio_pct"), 0.0, 4.0), 0.35),
            (_pct_score_optional(_value(record, "provision_coverage_ratio_pct"), 40.0, 80.0), 0.20),
        )
    )
    capital_strength = _weighted_score(
        (
            (_pct_score_optional(_value(record, "capital_adequacy_ratio_pct"), 10.5, 18.0), 0.65),
            (_pct_score_optional(_value(record, "cet1_ratio_pct"), 8.0, 15.0), 0.35),
        )
    )
    franchise = _weighted_score(
        (
            (_pct_score_optional(_value(record, "casa_ratio_pct"), 25.0, 45.0), 0.45),
            (_pct_score_optional(_value(record, "deposits_growth_pct"), 0.0, 18.0), 0.30),
            (_pct_score_optional(_value(record, "advances_growth_pct"), 0.0, 20.0), 0.25),
        )
    )
    profitability = _weighted_score(
        (
            (_pct_score_optional(_value(record, "roa_pct"), 0.5, 2.0), 0.50),
            (_pct_score_optional(_value(record, "roe_pct"), 8.0, 18.0), 0.35),
            (_pct_score_optional(_value(record, "net_interest_margin_pct"), 2.0, 5.0), 0.15),
        )
    )
    efficiency = _weighted_score(
        ((_inverse_pct_score_optional(_value(record, "cost_to_income_ratio_pct"), 35.0, 65.0), 1.0),)
    )
    quality = (
        0.26 * asset_quality
        + 0.24 * capital_strength
        + 0.20 * franchise
        + 0.20 * profitability
        + 0.10 * efficiency
    )

    return {
        BANKING_SECTOR_APPLICABLE: 1.0,
        BANKING_FACTOR_AVAILABLE: 1.0,
        BANKING_METRIC_COVERAGE: round(coverage, 6),
        BANKING_DATA_CONFIDENCE: round(coverage, 6),
        BANKING_FACTOR_GAP: round(max(0.0, 1.0 - coverage), 6),
        BANKING_QUALITY_SCORE: round(quality, 6),
        BANK_ASSET_QUALITY_SCORE: round(asset_quality, 6),
        BANK_CAPITAL_STRENGTH_SCORE: round(capital_strength, 6),
        BANK_FRANCHISE_STRENGTH_SCORE: round(franchise, 6),
        BANK_PROFITABILITY_SCORE: round(profitability, 6),
        BANK_EFFICIENCY_SCORE: round(efficiency, 6),
        BANK_NII: _value(record, "net_interest_income"),
        BANK_NIM_PCT: _value(record, "net_interest_margin_pct"),
        BANK_ADVANCES_GROWTH_PCT: _value(record, "advances_growth_pct"),
        BANK_DEPOSITS_GROWTH_PCT: _value(record, "deposits_growth_pct"),
        BANK_CASA_PCT: _value(record, "casa_ratio_pct"),
        BANK_GNPA_PCT: _value(record, "gnpa_ratio_pct"),
        BANK_NNPA_PCT: _value(record, "nnpa_ratio_pct"),
        BANK_PCR_PCT: _value(record, "provision_coverage_ratio_pct"),
        BANK_CREDIT_COST_PCT: _value(record, "credit_cost_pct"),
        BANK_CAR_PCT: _value(record, "capital_adequacy_ratio_pct"),
        BANK_CET1_PCT: _value(record, "cet1_ratio_pct"),
        BANK_COST_INCOME_PCT: _value(record, "cost_to_income_ratio_pct"),
        BANK_ROA_PCT: _value(record, "roa_pct"),
        BANK_ROE_PCT: _value(record, "roe_pct"),
        BANK_LDR_PCT: _value(record, "loan_to_deposit_ratio_pct"),
    }


def _is_financial_business(*, sector: str, industry: str = "") -> bool:
    text = f"{sector} {industry}".lower()
    return any(token in text for token in ("bank", "financial", "finance", "nbfc", "insurance"))


def _metric_coverage(record: object) -> float:
    populated = sum(1 for attr in _BANKING_METRIC_ATTRS if abs(_value(record, attr)) > 1e-9)
    return populated / len(_BANKING_METRIC_ATTRS)


def _value(record: object, attr: str) -> float:
    try:
        return float(getattr(record, attr, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _pct_score_optional(value: float, low: float, high: float) -> float | None:
    if abs(float(value)) < 1e-9:
        return None
    return _pct_score(value, low, high)


def _inverse_pct_score_optional(value: float, low: float, high: float) -> float | None:
    if abs(float(value)) < 1e-9:
        return None
    return 1.0 - _pct_score(value, low, high)


def _pct_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def _weighted_score(components: Sequence[tuple[float | None, float]]) -> float:
    configured_weight = sum(weight for _, weight in components)
    if configured_weight <= 0:
        return 0.0
    return sum((score or 0.0) * weight for score, weight in components) / configured_weight
