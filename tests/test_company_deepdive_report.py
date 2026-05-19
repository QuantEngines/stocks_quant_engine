from __future__ import annotations

from stock_screener_engine.research.company_deepdive.report import CompanyDeepDiveBuilder


def test_company_deepdive_builder_produces_25_sections() -> None:
    analysis = {
        "symbol": "AAA",
        "company_name": "AAA Ltd",
        "sector": "Capital Goods",
        "as_of": "2026-01-01",
        "scores": {"long_term_score": 70.0, "swing_score": 62.0, "risk_penalty": 8.0},
        "directional": {"bias": "bullish", "interpretation": "Bullish bias."},
        "key_drivers": {"top_positive": ["Strong growth"], "top_negative": ["Valuation risk"]},
        "fundamentals": {"pe_ratio": 25.0, "pb_ratio": 4.0, "roe_pct": 18.0, "debt_to_equity": 0.3},
        "all_features": {"valuation_sanity": 0.6, "cash_flow_quality": 0.7, "balance_sheet_health": 0.8},
        "investment_horizons": {
            "long_term": {"verdict": "buy", "rationale": "Quality is constructive."},
            "swing": {"verdict": "buy", "rationale": "Momentum is constructive."},
        },
        "news": {"headline_count": 1, "lookback_days": 30, "headlines": ["Order win announced"]},
    }
    docs = {
        "quality_score": 0.8,
        "facts": [{"kind": "financial", "label": "revenue", "value": "1000", "confidence": 0.6}],
        "management_commentary": ["Demand outlook remains constructive."],
    }
    peers = {
        "sector": "Capital Goods",
        "peer_count": 3,
        "target": {
            "composite_rank": 1,
            "valuation_rank": 2,
            "quality_rank": 1,
            "growth_rank": 1,
            "risk_rank": 2,
        },
        "peers": [{"symbol": "AAA"}],
        "composite_leaders": ["AAA", "BBB"],
        "warnings": [],
    }

    report = CompanyDeepDiveBuilder().build(analysis, document_insights=docs, peer_insights=peers)
    markdown = CompanyDeepDiveBuilder().render_markdown(report)

    assert len(report.sections) == 25
    assert report.final_verdict["directional_bias"] == "bullish"
    assert report.sections[12].data_points["target_composite_rank"] == 1
    assert "# AAA Ltd (AAA) Deep-Dive" in markdown
