from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from stock_screener_engine.config.settings import load_settings
from stock_screener_engine.pipelines.document_pipeline import DocumentIntelligencePipeline


def test_document_pipeline_extracts_sections_facts_and_commentary(tmp_path: Path) -> None:
    report_file = tmp_path / "annual_report.txt"
    report_file.write_text(
        """
        Business Overview
        The company saw revenue of 12,500 crore and EBITDA 2,100 crore.
        Management Discussion
        Demand outlook remains strong with capex of 500 crore planned.
        Risk Factors
        Commodity price risk and foreign exchange risk remain key monitorables.
        Corporate Governance
        There were no qualified opinion matters, but related party transactions are disclosed.
        """,
        encoding="utf-8",
    )
    settings = load_settings()
    settings = replace(
        settings,
        storage=replace(settings.storage, root_dir=str(tmp_path), sqlite_path=str(tmp_path / "metadata.db")),
    )

    result = DocumentIntelligencePipeline(settings).run(
        symbol="AAA",
        file_path=str(report_file),
        company_name="AAA Ltd",
        document_type="annual_report",
    )
    payload = result.to_dict()

    assert payload["symbol"] == "AAA"
    assert payload["facts"]
    assert payload["management_commentary"]
    assert payload["risk_factors"]
    assert "business_overview" in payload["section_map"]
    assert list((tmp_path / "documents").glob("*.json"))
