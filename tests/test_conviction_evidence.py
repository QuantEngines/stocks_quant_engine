from __future__ import annotations

import json

import pytest

from stock_screener_engine.core.conviction_evidence import load_backtest_evidence


def test_load_backtest_evidence_reads_engine_evaluation_payload(tmp_path) -> None:
    path = tmp_path / "engine_conviction_evaluation.json"
    path.write_text(
        json.dumps(
            {
                "evaluation": {
                    "quantile_ic": {"5": 0.04, "20": 0.08},
                    "net_quantile_ic": {"5": 0.03, "20": 0.06},
                    "decay": {"5": 0.02, "20": 0.04},
                    "net_horizon_metrics": {
                        "5": {"top_quantile_hit_rate": 0.6, "avg_quantile_spread": 0.01},
                        "20": {"top_quantile_hit_rate": 0.7, "avg_quantile_spread": 0.03},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    evidence, diagnostics = load_backtest_evidence(path)

    assert diagnostics["backtest_evidence_loaded"] is True
    assert evidence["backtest_information_coefficient"] == pytest.approx(0.045)
    assert evidence["backtest_hit_rate"] == pytest.approx(0.65)
    assert evidence["calibration_score"] > 0.5
