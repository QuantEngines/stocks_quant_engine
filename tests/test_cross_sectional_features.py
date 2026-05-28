from __future__ import annotations

from datetime import date

from stock_screener_engine.core.cross_sectional_features import CrossSectionalFeatureEnricher
from stock_screener_engine.core.entities import FeatureVector


def _fv(symbol: str, *, momentum: float, quality: float, value: float, liquidity: float) -> FeatureVector:
    return FeatureVector(
        symbol=symbol,
        as_of=date(2026, 5, 28),
        values={
            "trend_strength": momentum,
            "momentum_strength": momentum,
            "relative_strength_proxy": momentum,
            "breakout_score": momentum,
            "price_acceleration": (momentum - 0.5) / 5.0,
            "growth_quality": quality,
            "profitability_quality": quality,
            "cash_flow_quality": quality,
            "balance_sheet_health": quality,
            "earnings_stability": quality,
            "governance_proxy": quality,
            "valuation_sanity": value,
            "sector_pe_zscore": 0.0,
            "sector_pb_zscore": 0.0,
            "rolling_pe_zscore": 0.0,
            "pe_ratio": 20.0,
            "pb_ratio": 3.0,
            "cfo_pat_ratio": 1.0,
            "volume_confirmation": liquidity,
            "activity_vs_avg": 1.0 + liquidity,
            "market_regime_score": 0.0,
            "sector_momentum": 0.0,
        },
    )


def test_cross_sectional_enricher_adds_universe_and_sector_ranks() -> None:
    enriched = CrossSectionalFeatureEnricher().enrich(
        [
            _fv("AAA", momentum=0.9, quality=0.8, value=0.7, liquidity=0.9),
            _fv("BBB", momentum=0.3, quality=0.4, value=0.4, liquidity=0.3),
            _fv("CCC", momentum=0.6, quality=0.9, value=0.8, liquidity=0.6),
        ],
        sector_by_symbol={"AAA": "Banks", "BBB": "Banks", "CCC": "IT"},
    )

    by_symbol = {row.symbol: row.values for row in enriched}

    assert by_symbol["AAA"]["cross_sectional_momentum_rank"] == 1.0
    assert by_symbol["BBB"]["cross_sectional_momentum_rank"] == 0.0
    assert by_symbol["AAA"]["sector_relative_momentum_rank"] == 1.0
    assert by_symbol["BBB"]["sector_relative_momentum_rank"] == 0.0
    assert by_symbol["CCC"]["sector_relative_momentum_rank"] == 0.5
    assert by_symbol["CCC"]["cross_sectional_quality_rank"] == 1.0
    assert 0.0 <= by_symbol["AAA"]["research_readiness_score"] <= 1.0


def test_cross_sectional_enricher_marks_sparse_feature_coverage() -> None:
    sparse = FeatureVector(
        symbol="SPARSE",
        as_of=date(2026, 5, 28),
        values={"trend_strength": 0.5, "momentum_strength": 0.5},
    )

    enriched = CrossSectionalFeatureEnricher().enrich([sparse], sector_by_symbol={"SPARSE": "Unknown"})[0]

    assert enriched.values["feature_coverage_score"] < 0.5
    assert enriched.values["research_readiness_score"] < 0.5
