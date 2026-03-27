from __future__ import annotations

from stock_screener_engine.core.normalizers import (
    clamp,
    inverse_linear_score,
    linear_score,
    symmetric_score,
    threshold_score,
)


def test_clamp() -> None:
    assert clamp(-1.0) == 0.0
    assert clamp(0.5) == 0.5
    assert clamp(3.0) == 1.0


def test_linear_score() -> None:
    assert linear_score(0.0, 0.0, 10.0) == 0.0
    assert linear_score(10.0, 0.0, 10.0) == 1.0


def test_inverse_linear_score() -> None:
    assert inverse_linear_score(0.0, 0.0, 10.0) == 1.0
    assert inverse_linear_score(10.0, 0.0, 10.0) == 0.0


def test_threshold_score() -> None:
    assert threshold_score(9.0, threshold=10.0, width=2.0) > 0.0
    assert threshold_score(11.0, threshold=10.0, width=2.0) == 1.0


def test_symmetric_score() -> None:
    assert symmetric_score(1.0, center=1.0, tolerance=1.0) == 1.0
    assert symmetric_score(2.0, center=1.0, tolerance=1.0) == 0.0
