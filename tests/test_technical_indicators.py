from __future__ import annotations

from stock_screener_engine.core.technical_indicators import adx, atr, breakout_compression, momentum, rolling_beta


def test_atr_positive_for_moving_series() -> None:
    high = [10, 11, 12, 12.5, 13]
    low = [9, 9.5, 10.5, 11.0, 11.5]
    close = [9.5, 10.5, 11.5, 12.0, 12.7]
    assert atr(high, low, close, period=3) > 0


def test_adx_non_negative() -> None:
    high = [10, 11, 12, 13, 14, 14.5, 15, 16, 16.5, 17, 18, 19, 19.5, 20, 20.5, 21]
    low = [9, 9.8, 10.8, 11.5, 12.0, 12.4, 13.2, 14.2, 14.8, 15.1, 16, 16.7, 17.0, 18, 18.2, 18.8]
    close = [9.7, 10.7, 11.6, 12.5, 13.0, 13.4, 14.0, 15.0, 15.5, 16.2, 17.0, 18.0, 18.4, 19.2, 19.6, 20.3]
    assert adx(high, low, close, period=14) >= 0


def test_rolling_beta_close_to_one_for_scaled_series() -> None:
    idx = [100.0, 102.0, 101.0, 103.0, 104.0, 106.0, 105.0, 107.0, 109.0, 108.0]
    asset = [x * 1.5 for x in idx]
    beta = rolling_beta(asset, idx, period=10)
    assert 0.7 <= beta <= 1.3


def test_breakout_compression_higher_when_range_tight() -> None:
    tight = [100, 100.4, 100.3, 100.5, 100.2, 100.6, 100.4, 100.5, 100.3, 100.4,
             100.5, 100.3, 100.2, 100.4, 100.3, 100.5, 100.4, 100.3, 100.2, 100.4]
    wide = [100.0, 103.0, 97.0, 104.0, 96.0, 105.0, 95.0, 106.0, 94.0, 107.0,
            93.0, 108.0, 92.0, 109.0, 91.0, 110.0, 90.0, 111.0, 89.0, 112.0]
    assert breakout_compression(tight, lookback=20) > breakout_compression(wide, lookback=20)


def test_momentum_positive_for_uptrend() -> None:
    close = [float(100 + i) for i in range(30)]
    assert momentum(close, lookback=20) > 0
