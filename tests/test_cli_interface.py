from __future__ import annotations

from stock_screener_engine.interfaces.cli import main as cli_main


def test_cli_scan_table_uses_professional_rows(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SSE_MARKET_PROVIDER", raising=False)
    monkeypatch.setattr(
        cli_main,
        "run_screen",
        lambda **kwargs: {
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

    def fake_run_screen(**kwargs):
        return {
            "source": cli_main.os.environ.get("SSE_MARKET_PROVIDER"),
            "readiness_check": kwargs["readiness_check"],
            "professional_signal_reports": {"console_rows": {"long_term": [], "swing": []}},
        }

    monkeypatch.setattr(cli_main, "run_screen", fake_run_screen)

    cli_main.main(["scan", "--source", "canonical", "--format", "json"])
    out = capsys.readouterr().out

    assert '"source": "canonical"' in out
    assert '"readiness_check": "warn"' in out


def test_cli_scan_passes_readiness_and_universe_args(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_screen",
        lambda **kwargs: {
            "scan_mode": kwargs["scan_mode"],
            "symbols": kwargs["symbols"],
            "universe_file": kwargs["universe_file"],
            "readiness_check": kwargs["readiness_check"],
            "readiness_lookback_years": kwargs["readiness_lookback_years"],
            "professional_signal_reports": {"console_rows": {"long_term": [], "swing": []}},
        },
    )

    cli_main.main([
        "scan",
        "--symbols",
        "ITC,RELIANCE",
        "--universe-file",
        "/tmp/nifty50.csv",
        "--readiness-check",
        "enforce",
        "--readiness-lookback-years",
        "3",
        "--format",
        "json",
    ])
    out = capsys.readouterr().out

    assert '"symbols": [' in out
    assert '"ITC"' in out
    assert '"/tmp/nifty50.csv"' in out
    assert '"readiness_check": "enforce"' in out
    assert '"readiness_lookback_years": 3' in out


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
            "universe_file": kwargs["universe_file"],
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


def test_cli_data_source_coverage_emits_markdown(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_data_source_coverage",
        lambda **kwargs: {
            "pipeline": "data_source_coverage",
            "as_of": kwargs["as_of"].isoformat(),
            "start": kwargs["start"].isoformat(),
            "console_rows": [{"kind": "domain", "name": "Daily OHLCV"}],
            "markdown": "# Data Source Coverage\n",
        },
    )

    cli_main.main(["data-source-coverage", "--end", "2026-05-28", "--lookback-years", "5", "--format", "markdown"])
    out = capsys.readouterr().out

    assert out.startswith("# Data Source Coverage")


def test_cli_data_readiness_emits_markdown(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_data_readiness",
        lambda **kwargs: {
            "pipeline": "data_readiness",
            "mode": kwargs["mode"],
            "decision": "block",
            "console_rows": [{"domain": "financials", "severity": "block"}],
            "markdown": "# Data Readiness Gate\n",
        },
    )

    cli_main.main([
        "data-readiness",
        "--end",
        "2026-05-28",
        "--mode",
        "long-term-scan",
        "--format",
        "markdown",
    ])
    out = capsys.readouterr().out

    assert out.startswith("# Data Readiness Gate")


def test_cli_exchange_foundation_status_emits_markdown(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_exchange_foundation_status",
        lambda **kwargs: {
            "pipeline": "exchange_foundation_status",
            "as_of": kwargs["as_of"].isoformat(),
            "domains": [{"domain": "delivery_turnover", "coverage": 0.5}],
            "markdown": "# NSE/BSE Exchange Foundation Status\n",
        },
    )

    cli_main.main([
        "exchange-foundation-status",
        "--end",
        "2026-05-28",
        "--universe-file",
        "/tmp/nifty50.csv",
        "--format",
        "markdown",
    ])
    out = capsys.readouterr().out

    assert out.startswith("# NSE/BSE Exchange Foundation Status")


def test_cli_exchange_delivery_ingest_emits_table(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_exchange_delivery_ingest",
        lambda **kwargs: {
            "pipeline": "exchange_delivery_ingest",
            "venue": kwargs["venue"],
            "file": kwargs["file_path"],
            "input_rows": 2,
            "persisted": 2,
            "symbols": 2,
            "passed": True,
        },
    )

    cli_main.main([
        "exchange-delivery-ingest",
        "--file",
        "/tmp/delivery.csv",
        "--venue",
        "NSE",
        "--trade-date",
        "2026-05-28",
        "--format",
        "table",
    ])
    out = capsys.readouterr().out

    assert '"persisted": 2' in out
    assert '"/tmp/delivery.csv"' in out


def test_cli_data_source_priority_emits_table(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_data_source_priority",
        lambda **kwargs: {
            "pipeline": "data_source_priority",
            "rows": [{"domain": "financials", "primary": ["finedge"], "status": "sandbox_ready"}],
            "markdown": "# Data Source Priority Map\n",
        },
    )

    cli_main.main(["data-source-priority", "--format", "table"])
    out = capsys.readouterr().out

    assert '"domain": "financials"' in out
    assert '"status": "sandbox_ready"' in out


def test_cli_missing_data_list_emits_markdown(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_missing_data_list",
        lambda **kwargs: {
            "pipeline": "missing_data_list",
            "rows": [{"name": "filing_date", "status": "still_required"}],
            "markdown": "# Stock Engine Missing Data List\n",
        },
    )

    cli_main.main(["missing-data-list", "--format", "markdown"])
    out = capsys.readouterr().out

    assert out.startswith("# Stock Engine Missing Data List")


def test_cli_data_entitlements_emits_table(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_data_entitlements",
        lambda **kwargs: {
            "pipeline": "data_entitlements",
            "sources": [
                {
                    "display_name": "FinEdge",
                    "plan_name": "Basic Free",
                    "status": "basic_sandbox",
                    "enabled": True,
                    "entitled_symbol_count": 3,
                    "entitlement_coverage": 0.06,
                    "storage_rights": "vendor_terms_required",
                    "redistribution_rights": "not_confirmed",
                }
            ],
            "markdown": "# Data Entitlements\n",
        },
    )

    cli_main.main(["data-entitlements", "--symbols", "ITC,RELIANCE,HDFCBANK", "--format", "table"])
    out = capsys.readouterr().out

    assert "FinEdge" in out
    assert "Basic Free" in out


def test_cli_refresh_market_delegates(monkeypatch, capsys) -> None:
    monkeypatch.delenv("SSE_MARKET_PROVIDER", raising=False)
    monkeypatch.setattr(
        cli_main,
        "run_market_refresh",
        lambda **kwargs: {
            "pipeline": "market_refresh",
            "start": kwargs["start"].isoformat(),
            "end": kwargs["end"].isoformat(),
            "symbols": kwargs["symbols"],
            "batch_size": kwargs["batch_size"],
            "retries": kwargs["retries"],
            "run_scan": kwargs["run_scan"],
            "source": cli_main.os.environ.get("SSE_MARKET_PROVIDER"),
        },
    )

    cli_main.main([
        "refresh-market",
        "--source",
        "zerodha",
        "--start",
        "2026-05-19",
        "--end",
        "2026-05-27",
        "--symbols",
        "AAA,BBB",
        "--batch-size",
        "10",
        "--retries",
        "3",
        "--run-scan",
    ])
    out = capsys.readouterr().out

    assert '"pipeline": "market_refresh"' in out
    assert '"source": "zerodha"' in out
    assert '"batch_size": 10' in out
    assert '"retries": 3' in out
    assert '"run_scan": true' in out


def test_cli_broker_health_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_broker_health",
        lambda **kwargs: {
            "pipeline": "broker_health",
            "start": kwargs["start"].isoformat(),
            "end": kwargs["end"].isoformat(),
            "sources": kwargs["sources"],
            "source_reports": {
                "zerodha": {
                    "enabled": True,
                    "quote_coverage": 1.0,
                    "historical_coverage": 1.0,
                    "stale_symbols": [],
                    "source_errors": [],
                }
            },
        },
    )

    cli_main.main([
        "broker-health",
        "--start",
        "2026-05-19",
        "--end",
        "2026-05-27",
        "--symbols",
        "AAA,BBB",
        "--sources",
        "zerodha,icici_breeze",
        "--format",
        "table",
    ])
    out = capsys.readouterr().out

    assert '"source": "zerodha"' in out
    assert '"quote_coverage": 1.0' in out
    assert '"historical_coverage": 1.0' in out


def test_cli_indianapi_probe_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_indianapi_probe",
        lambda **kwargs: {
            "pipeline": "indianapi_probe",
            "symbols": kwargs["symbols"],
            "checks": kwargs["checks"],
            "stock_base_url": kwargs["stock_base_url"],
            "coverage": {"stock": {"ok": 2, "total": 2, "coverage": 1.0, "sample_errors": []}},
        },
    )

    cli_main.main([
        "indianapi-probe",
        "--symbols",
        "AAA,BBB",
        "--check",
        "stock,financials",
        "--stock-base-url",
        "https://stock.example.test",
        "--format",
        "table",
    ])
    out = capsys.readouterr().out

    assert '"check": "stock"' in out
    assert '"coverage": 1.0' in out


def test_cli_fmp_probe_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_fmp_probe",
        lambda **kwargs: {
            "pipeline": "fmp_probe",
            "symbols": kwargs["symbols"],
            "checks": kwargs["checks"],
            "base_url": kwargs["base_url"],
            "price_start": kwargs["price_start"].isoformat(),
            "price_end": kwargs["price_end"].isoformat(),
            "coverage": {
                "income_statement": {
                    "ok": 2,
                    "total": 2,
                    "coverage": 1.0,
                    "sample_resolved_symbols": ["AAA.NS"],
                    "sample_errors": [],
                }
            },
        },
    )

    cli_main.main([
        "fmp-probe",
        "--symbols",
        "AAA,BBB",
        "--check",
        "income_statement,ratios",
        "--base-url",
        "https://fmp.example.test",
        "--price-start",
        "2026-01-01",
        "--price-end",
        "2026-05-01",
        "--format",
        "table",
    ])
    out = capsys.readouterr().out

    assert '"check": "income_statement"' in out
    assert '"sample_resolved_symbols": [' in out
    assert '"AAA.NS"' in out


def test_cli_finedge_probe_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_finedge_probe",
        lambda **kwargs: {
            "pipeline": "finedge_probe",
            "symbols": kwargs["symbols"],
            "checks": kwargs["checks"],
            "statement_type": kwargs["statement_type"],
            "statement_code": kwargs["statement_code"],
            "coverage": {
                "financials": {
                    "ok": 2,
                    "total": 2,
                    "coverage": 1.0,
                    "sample_errors": [],
                }
            },
        },
    )

    cli_main.main([
        "finedge-probe",
        "--symbols",
        "ITC,RELIANCE",
        "--check",
        "financials,ratios",
        "--statement-type",
        "c",
        "--statement-code",
        "pl",
        "--format",
        "table",
    ])
    out = capsys.readouterr().out

    assert '"check": "financials"' in out
    assert '"coverage": 1.0' in out


def test_cli_finedge_probe_accepts_universe_file(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_finedge_probe",
        lambda **kwargs: {
            "pipeline": "finedge_probe",
            "symbols": kwargs["symbols"],
            "universe_file": kwargs["universe_file"],
            "checks": kwargs["checks"],
        },
    )

    cli_main.main([
        "finedge-probe",
        "--universe-file",
        "/tmp/nifty50.csv",
        "--check",
        "smoke",
    ])
    out = capsys.readouterr().out

    assert '"symbols": []' in out
    assert '"/tmp/nifty50.csv"' in out


def test_cli_finedge_inspect_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_finedge_inspect",
        lambda **kwargs: {
            "pipeline": "finedge_schema_inspection",
            "symbols": kwargs["symbols"],
            "checks": kwargs["checks"],
            "statement_type": kwargs["statement_type"],
            "statement_code": kwargs["statement_code"],
            "symbol_reports": [
                {
                    "symbol": "ITC",
                    "checks": {
                        "financials": {
                            "ok": True,
                            "error": "",
                            "summary": {
                                "root_type": "dict",
                                "record_set_count": 1,
                                "primary_record_set": {
                                    "path": "$.financials",
                                    "item_count": 8,
                                    "field_count": 24,
                                    "fields": ["period", "revenue"],
                                    "date_like_fields": ["period"],
                                    "numeric_like_fields": ["revenue"],
                                },
                            },
                        }
                    },
                }
            ],
            "market_report": {"checks": {}},
        },
    )

    cli_main.main([
        "finedge-inspect",
        "--symbols",
        "ITC",
        "--check",
        "financials",
        "--statement-type",
        "c",
        "--statement-code",
        "pl",
        "--format",
        "table",
    ])
    out = capsys.readouterr().out

    assert '"scope": "ITC"' in out
    assert '"check": "financials"' in out
    assert '"primary_path": "$.financials"' in out
    assert '"revenue"' in out


def test_cli_finedge_inspect_accepts_universe_file(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_finedge_inspect",
        lambda **kwargs: {
            "pipeline": "finedge_schema_inspection",
            "symbols": kwargs["symbols"],
            "universe_file": kwargs["universe_file"],
            "checks": kwargs["checks"],
            "symbol_reports": [],
            "market_report": {"checks": {}},
        },
    )

    cli_main.main([
        "finedge-inspect",
        "--universe-file",
        "/tmp/nifty50.csv",
        "--check",
        "financials",
    ])
    out = capsys.readouterr().out

    assert '"symbols": []' in out
    assert '"/tmp/nifty50.csv"' in out


def test_cli_finedge_factor_export_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_finedge_factor_export",
        lambda **kwargs: {
            "pipeline": "finedge_factor_export",
            "symbols": kwargs["symbols"],
            "as_of": kwargs["as_of"].isoformat(),
            "output_root": kwargs["output_root"],
            "sections": kwargs["sections"],
            "passed": True,
            "row_counts": {"financials": 2, "valuations": 2, "shareholding": 1, "ownership_details": 3},
            "files": {
                "financials": {"path": "factor_root/financials.csv", "rows": 2},
                "valuations": {"path": "factor_root/valuations.csv", "rows": 2},
                "shareholding": {"path": "factor_root/shareholding.csv", "rows": 1},
                "ownership_details": {"path": "factor_root/finedge_ownership_details.csv", "rows": 3},
            },
            "issues": [],
        },
    )

    cli_main.main([
        "finedge-factor-export",
        "--symbols",
        "ITC,RELIANCE",
        "--as-of",
        "2026-05-28",
        "--output-root",
        "factor_root",
        "--sections",
        "financials,shareholding",
        "--format",
        "table",
    ])
    out = capsys.readouterr().out

    assert '"section": "financials"' in out
    assert '"rows": 2' in out
    assert "factor_root/financials.csv" in out


def test_cli_finedge_factor_export_accepts_universe_file(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_finedge_factor_export",
        lambda **kwargs: {
            "pipeline": "finedge_factor_export",
            "symbols": kwargs["symbols"],
            "universe_file": kwargs["universe_file"],
            "as_of": kwargs["as_of"].isoformat(),
            "output_root": kwargs["output_root"],
            "sections": kwargs["sections"],
            "passed": True,
            "row_counts": {"financials": 0, "valuations": 0, "shareholding": 0, "ownership_details": 0},
            "files": {},
            "issues": [],
        },
    )

    cli_main.main([
        "finedge-factor-export",
        "--universe-file",
        "/tmp/nifty50.csv",
        "--as-of",
        "2026-05-28",
        "--output-root",
        "factor_root",
    ])
    out = capsys.readouterr().out

    assert '"symbols": []' in out
    assert '"/tmp/nifty50.csv"' in out


def test_cli_finedge_onboarding_plan_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_finedge_onboarding_plan",
        lambda **kwargs: {
            "pipeline": "finedge_onboarding_plan",
            "as_of": kwargs["as_of"].isoformat(),
            "target_domains": [{"domain": "financials", "coverage": 0.06}],
            "markdown": "# FinEdge Paid Onboarding Plan\n",
        },
    )

    cli_main.main([
        "finedge-onboarding-plan",
        "--end",
        "2026-05-28",
        "--symbols",
        "ITC,RELIANCE,HDFCBANK",
        "--format",
        "markdown",
    ])
    out = capsys.readouterr().out

    assert out.startswith("# FinEdge Paid Onboarding Plan")


def test_cli_backtest_readiness_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_backtest_readiness",
        lambda **kwargs: {
            "passed": True,
            "start": kwargs["start"].isoformat(),
            "end": kwargs["end"].isoformat(),
            "horizons": kwargs["horizons"],
            "min_history_rows": kwargs["min_history_rows"],
        },
    )

    cli_main.main([
        "backtest-readiness",
        "--end",
        "2026-05-18",
        "--lookback-years",
        "5",
        "--symbols",
        "AAA,BBB",
        "--horizons",
        "5,20",
        "--min-history-rows",
        "1000",
    ])
    out = capsys.readouterr().out

    assert '"passed": true' in out
    assert '"horizons": [' in out
    assert '"min_history_rows": 1000' in out


def test_cli_backtest_labels_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_forward_return_labels",
        lambda **kwargs: {
            "pipeline": "forward_return_labels",
            "universe_policy": kwargs["universe_policy"],
            "horizons": kwargs["horizons"],
        },
    )

    cli_main.main([
        "backtest-labels",
        "--end",
        "2026-05-18",
        "--lookback-years",
        "5",
        "--symbols",
        "AAA,BBB",
        "--universe-policy",
        "eligible_history",
        "--horizons",
        "5,20",
    ])
    out = capsys.readouterr().out

    assert '"pipeline": "forward_return_labels"' in out
    assert '"universe_policy": "eligible_history"' in out
    assert '"horizons": [' in out


def test_cli_technical_backtest_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_technical_backtest",
        lambda **kwargs: {
            "pipeline": "technical_ranking_backtest",
            "universe_policy": kwargs["universe_policy"],
            "min_lookback": kwargs["min_lookback"],
        },
    )

    cli_main.main([
        "technical-backtest",
        "--end",
        "2026-05-18",
        "--lookback-years",
        "5",
        "--symbols",
        "AAA,BBB",
        "--min-lookback",
        "120",
    ])
    out = capsys.readouterr().out

    assert '"pipeline": "technical_ranking_backtest"' in out
    assert '"universe_policy": "eligible_history"' in out
    assert '"min_lookback": 120' in out


def test_cli_engine_backtest_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_engine_backtest",
        lambda **kwargs: {
            "pipeline": "engine_score_backtest",
            "score_type": kwargs["score_type"],
            "round_trip_cost_bps": kwargs["round_trip_cost_bps"],
        },
    )

    cli_main.main([
        "engine-backtest",
        "--end",
        "2026-05-18",
        "--lookback-years",
        "5",
        "--symbols",
        "AAA,BBB",
        "--score-type",
        "swing",
        "--round-trip-cost-bps",
        "35",
    ])
    out = capsys.readouterr().out

    assert '"pipeline": "engine_score_backtest"' in out
    assert '"score_type": "swing"' in out
    assert '"round_trip_cost_bps": 35.0' in out


def test_cli_conviction_calibrate_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_conviction_calibration",
        lambda **kwargs: {
            "pipeline": "conviction_calibration",
            "score_type": kwargs["score_type"],
            "horizons": kwargs["horizons"],
            "output_path": kwargs["output_path"],
        },
    )

    cli_main.main([
        "conviction-calibrate",
        "--end",
        "2026-05-18",
        "--lookback-years",
        "5",
        "--symbols",
        "AAA,BBB",
        "--horizons",
        "5,20",
        "--output-path",
        "/tmp/calibration_report_latest.json",
    ])
    out = capsys.readouterr().out

    assert '"pipeline": "conviction_calibration"' in out
    assert '"score_type": "conviction"' in out
    assert '"horizons": [' in out
    assert '"/tmp/calibration_report_latest.json"' in out


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


def test_cli_factor_template_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_factor_template",
        lambda **kwargs: {
            "pipeline": "factor_bootstrap_template",
            "output_root": kwargs["output_root"],
            "symbols": kwargs["symbols"],
            "overwrite": kwargs["overwrite"],
        },
    )

    cli_main.main([
        "factor-template",
        "--output-root",
        "factor_root",
        "--as-of",
        "2026-05-01",
        "--symbols",
        "AAA,BBB",
        "--overwrite",
    ])
    out = capsys.readouterr().out

    assert '"pipeline": "factor_bootstrap_template"' in out
    assert '"AAA"' in out
    assert '"overwrite": true' in out


def test_cli_factor_ingest_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_factor_ingest",
        lambda **kwargs: {
            "pipeline": "factor_bootstrap_ingest",
            "root": kwargs["root"],
            "min_coverage": kwargs["min_coverage"],
        },
    )

    cli_main.main([
        "factor-ingest",
        "--root",
        "factor_root",
        "--as-of",
        "2026-05-01",
        "--symbols",
        "AAA,BBB",
        "--min-coverage",
        "0.8",
    ])
    out = capsys.readouterr().out

    assert '"pipeline": "factor_bootstrap_ingest"' in out
    assert '"min_coverage": 0.8' in out


def test_cli_factor_qa_delegates(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli_main,
        "run_factor_qa",
        lambda **kwargs: {
            "pipeline": "factor_qa",
            "as_of": kwargs["as_of"].isoformat(),
            "console_rows": [{"symbol": "AAA", "status": "ok"}],
            "markdown": "# QA",
        },
    )

    cli_main.main([
        "factor-qa",
        "--as-of",
        "2026-05-01",
        "--symbols",
        "AAA,BBB",
        "--format",
        "table",
    ])
    out = capsys.readouterr().out

    assert '"symbol": "AAA"' in out
    assert '"status": "ok"' in out
