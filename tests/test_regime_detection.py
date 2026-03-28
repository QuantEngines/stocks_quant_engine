from __future__ import annotations

from stock_screener_engine.core.regime_detection import RegimeDetector


def _bars_from_closes(closes: list[float]) -> list[dict]:
    return [{"close": c} for c in closes]


def test_regime_detector_identifies_bull_regime() -> None:
    closes = [100 + i * 1.2 for i in range(30)]
    detector = RegimeDetector()

    snap = detector.detect(
        index_bars=_bars_from_closes(closes),
        universe_returns=[0.01, 0.02, 0.015, 0.005, 0.0, 0.01],
    )

    assert snap.label == "bull"
    assert snap.score > 0.0


def test_regime_detector_identifies_bear_regime() -> None:
    closes = [120 - i * 1.8 for i in range(30)]
    detector = RegimeDetector()

    snap = detector.detect(
        index_bars=_bars_from_closes(closes),
        universe_returns=[-0.02, -0.01, -0.03, 0.0, -0.015, -0.01],
    )

    assert snap.label == "bear"
    assert snap.score < 0.0
