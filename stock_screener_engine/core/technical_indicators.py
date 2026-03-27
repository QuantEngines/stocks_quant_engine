"""Bar-based technical indicator utilities.

No external dependencies required; formulas are implemented in pure Python.
"""

from __future__ import annotations


def atr(high: list[float], low: list[float], close: list[float], period: int = 14) -> float:
    if len(close) < 2:
        return 0.0
    trs: list[float] = []
    for i in range(1, len(close)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        trs.append(max(0.0, tr))
    if not trs:
        return 0.0
    window = trs[-period:] if len(trs) >= period else trs
    return sum(window) / len(window)


def adx(high: list[float], low: list[float], close: list[float], period: int = 14) -> float:
    if len(close) < period + 2:
        return 0.0

    plus_dm: list[float] = []
    minus_dm: list[float] = []
    tr_list: list[float] = []

    for i in range(1, len(close)):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0.0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0.0)
        tr = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        tr_list.append(max(1e-12, tr))

    n = min(period, len(tr_list))
    plus_di = 100.0 * (sum(plus_dm[-n:]) / n) / (sum(tr_list[-n:]) / n)
    minus_di = 100.0 * (sum(minus_dm[-n:]) / n) / (sum(tr_list[-n:]) / n)
    denom = plus_di + minus_di
    if denom <= 1e-12:
        return 0.0
    dx = abs(plus_di - minus_di) / denom * 100.0
    return dx


def rolling_beta(asset_close: list[float], index_close: list[float], period: int = 60) -> float:
    n = min(period, len(asset_close), len(index_close))
    if n < 3:
        return 0.0
    a = _returns(asset_close[-n:])
    b = _returns(index_close[-n:])
    nret = min(len(a), len(b))
    if nret < 3:
        return 0.0
    a = a[-nret:]
    b = b[-nret:]
    mean_a = sum(a) / nret
    mean_b = sum(b) / nret
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b)) / (nret - 1)
    var_b = sum((y - mean_b) ** 2 for y in b) / (nret - 1)
    if var_b <= 1e-12:
        return 0.0
    return cov / var_b


def breakout_compression(close: list[float], lookback: int = 20) -> float:
    if len(close) < lookback:
        return 0.0
    window = close[-lookback:]
    hi = max(window)
    lo = min(window)
    if lo <= 0:
        return 0.0
    compression = (hi - lo) / lo
    if compression <= 1e-12:
        return 1.0
    return 1.0 / (1.0 + compression)


def momentum(close: list[float], lookback: int = 20) -> float:
    if len(close) <= lookback:
        return 0.0
    prev = close[-lookback - 1]
    if prev == 0:
        return 0.0
    return (close[-1] / prev) - 1.0


def _returns(close: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(close)):
        prev = close[i - 1]
        out.append(0.0 if prev == 0 else (close[i] / prev) - 1.0)
    return out
