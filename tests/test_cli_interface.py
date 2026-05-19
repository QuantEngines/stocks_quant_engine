from __future__ import annotations

from stock_screener_engine.interfaces.cli import main as cli_main


def test_cli_scan_table_uses_professional_rows(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SSE_MARKET_PROVIDER", raising=False)
    monkeypatch.setattr(
        cli_main,
        "run_screen",
        lambda config_path=None: {
            "professional_signal_reports": {
                "console_rows": {
                    "long_term": [{"symbol": "AAA", "score": 80.0}],
                    "swing": [{"symbol": "BBB", "score": 70.0}],
                }
            }
        },
    )

    cli_main.main(["scan", "--mode", "daily", "--format", "table"])
    out = capsys.readouterr().out

    assert "AAA" in out


def test_cli_source_override_sets_market_provider(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SSE_MARKET_PROVIDER", raising=False)

    def fake_run_screen(config_path=None):
        return {
            "source": cli_main.os.environ.get("SSE_MARKET_PROVIDER"),
            "professional_signal_reports": {"console_rows": {"long_term": [], "swing": []}},
        }

    monkeypatch.setattr(cli_main, "run_screen", fake_run_screen)

    cli_main.main(["scan", "--source", "canonical", "--format", "json"])
    out = capsys.readouterr().out

    assert '"source": "canonical"' in out


def test_cli_document_ingest_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_document_ingest",
        lambda **kwargs: {"symbol": kwargs["symbol"], "facts": []},
    )

    cli_main.main(["document-ingest", "--symbol", "AAA", "--file", "report.txt"])
    out = capsys.readouterr().out

    assert '"symbol": "AAA"' in out


def test_cli_sector_rankings_json_does_not_emit_markdown_string(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_sector_rankings",
        lambda config_path=None: {
            "sector_rankings": [{"sector": "IT", "sector_score": 75.0}],
            "markdown": "# Sector Intelligence",
        },
    )

    cli_main.main(["sector-rankings", "--format", "json"])
    out = capsys.readouterr().out

    assert out.lstrip().startswith("{")
    assert '"sector_rankings"' in out
    assert "# Sector Intelligence" in out


def test_cli_peer_report_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_peer_report",
        lambda *args, **kwargs: {
            "symbol": args[0],
            "as_of": kwargs["as_of"].isoformat(),
            "peer_count": 3,
        },
    )

    cli_main.main(["peer-report", "AAA", "--as-of", "2026-05-01"])
    out = capsys.readouterr().out

    assert '"symbol": "AAA"' in out
    assert '"peer_count": 3' in out


def test_cli_sector_report_can_include_peer_payload(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_sector_rankings",
        lambda config_path=None: {
            "sector_rankings": [{"sector": "IT", "sector_score": 75.0}],
            "markdown": "# Sector Intelligence",
        },
    )
    monkeypatch.setattr(
        cli_main,
        "run_sector_peer_report",
        lambda *args, **kwargs: {"sector": args[0], "peer_count": 3},
    )

    cli_main.main(["sector-report", "--sector", "IT", "--include-peers"])
    out = capsys.readouterr().out

    assert '"peer_comparison"' in out
    assert '"peer_count": 3' in out


def test_cli_data_foundation_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_data_foundation",
        lambda **kwargs: {
            "passed": True,
            "start": kwargs["start"].isoformat(),
            "end": kwargs["end"].isoformat(),
            "symbols": kwargs["symbols"],
        },
    )

    cli_main.main([
        "data-foundation",
        "--start",
        "2026-01-01",
        "--end",
        "2026-01-02",
        "--symbols",
        "AAA,BBB",
    ])
    out = capsys.readouterr().out

    assert '"passed": true' in out
    assert '"AAA"' in out


def test_cli_data_quality_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_data_quality",
        lambda **kwargs: {"passed": True, "coverage": {"coverage": 1.0}},
    )

    cli_main.main(["data-quality", "--start", "2026-01-01", "--end", "2026-01-02"])
    out = capsys.readouterr().out

    assert '"coverage": 1.0' in out


def test_cli_security_master_ingest_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_security_master_ingest",
        lambda **kwargs: {
            "source_file": kwargs["file_path"],
            "venue": kwargs["venue"],
            "persisted": 2,
        },
    )

    cli_main.main(["security-master-ingest", "--file", "securities.csv", "--venue", "NSE"])
    out = capsys.readouterr().out

    assert '"source_file": "securities.csv"' in out
    assert '"persisted": 2' in out


def test_cli_financials_ingest_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_financials_ingest",
        lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "as_of": kwargs["as_of"].isoformat(),
            "persisted": 2,
        },
    )

    cli_main.main([
        "financials-ingest",
        "--symbol",
        "AAA",
        "--file",
        "statements.csv",
        "--as-of",
        "2026-05-01",
    ])
    out = capsys.readouterr().out

    assert '"symbol": "AAA"' in out
    assert '"persisted": 2' in out


def test_cli_valuation_ingest_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_valuation_ingest",
        lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "as_of": kwargs["as_of"].isoformat(),
            "persisted": 1,
        },
    )

    cli_main.main([
        "valuation-ingest",
        "--symbol",
        "AAA",
        "--file",
        "valuations.csv",
        "--as-of",
        "2026-05-01",
    ])
    out = capsys.readouterr().out

    assert '"symbol": "AAA"' in out
    assert '"persisted": 1' in out


def test_cli_shareholding_ingest_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_shareholding_ingest",
        lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "as_of": kwargs["as_of"].isoformat(),
            "persisted": 1,
        },
    )

    cli_main.main([
        "shareholding-ingest",
        "--symbol",
        "AAA",
        "--file",
        "shareholding.csv",
        "--as-of",
        "2026-05-01",
    ])
    out = capsys.readouterr().out

    assert '"symbol": "AAA"' in out
    assert '"persisted": 1' in out
