from __future__ import annotations

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.core.engine import ResearchEngine
from stock_screener_engine.data_sources.market.mock_market_data import MockIndianMarketDataProvider
from stock_screener_engine.data_sources.text.mock_text_adapter import MockTextEventProvider
from stock_screener_engine.reporting.signal_report import (
    build_signal_reports,
    render_signal_markdown,
    signal_reports_to_console_rows,
)


def test_professional_signal_report_has_required_sections() -> None:
    settings = load_settings()
    engine = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    )
    output = engine.run(regime_score=0.0)

    reports = build_signal_reports(
        output["features"],
        output["scores"],
        output["long_signals"],
        signal_type="long_term",
        limit=3,
    )

    assert reports
    payload = reports[0].to_dict()
    for key in [
        "identity",
        "summary",
        "technical",
        "fundamentals",
        "banking",
        "valuation",
        "event_nlp",
        "risk",
        "conviction",
        "peer_context",
        "explanation",
    ]:
        assert key in payload
    assert payload["summary"]["rank"] == 1
    assert "missing_data_warnings" in payload["explanation"]


def test_signal_report_markdown_and_console_rows() -> None:
    settings = load_settings()
    output = ResearchEngine(
        settings=settings,
        market_data=MockIndianMarketDataProvider(),
        text_data=MockTextEventProvider(),
    ).run(regime_score=0.0)
    reports = build_signal_reports(
        output["features"],
        output["scores"],
        output["swing_signals"],
        signal_type="swing",
        limit=1,
    )

    markdown = render_signal_markdown(reports[0])
    rows = signal_reports_to_console_rows(reports)

    assert "# " in markdown
    assert "Signal Report" in markdown
    assert rows[0]["symbol"] == reports[0].identity.symbol
