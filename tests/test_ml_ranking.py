from __future__ import annotations

from stock_screener_engine.core.ml_ranking import CalibrationSample, LinearMlRanker


def test_linear_ml_ranker_prefers_higher_predicted_alpha() -> None:
    samples = [
        CalibrationSample("A", {"quality": 0.2, "momentum": -0.1}, -0.03),
        CalibrationSample("B", {"quality": 0.4, "momentum": 0.0}, 0.01),
        CalibrationSample("C", {"quality": 0.8, "momentum": 0.5}, 0.06),
        CalibrationSample("D", {"quality": 0.7, "momentum": 0.3}, 0.04),
    ]

    model = LinearMlRanker().fit(samples, feature_names=["quality", "momentum"])

    ranked = model.rank(
        {
            "X": {"quality": 0.35, "momentum": 0.1},
            "Y": {"quality": 0.9, "momentum": 0.6},
            "Z": {"quality": 0.5, "momentum": 0.05},
        }
    )

    assert ranked[0][0] == "Y"
    assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]


def test_linear_ml_ranker_rejects_empty_input() -> None:
    ranker = LinearMlRanker()

    try:
        ranker.fit([], feature_names=["quality"])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "samples" in str(exc)
