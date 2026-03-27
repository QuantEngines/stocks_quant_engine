"""Walk-forward evaluation structure scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: str
    train_end: str
    test_start: str
    test_end: str


class WalkForwardPlanner:
    def build_windows(self, dates: list[str], train_size: int, test_size: int) -> list[WalkForwardWindow]:
        windows: list[WalkForwardWindow] = []
        start = 0
        while start + train_size + test_size <= len(dates):
            train = dates[start : start + train_size]
            test = dates[start + train_size : start + train_size + test_size]
            windows.append(
                WalkForwardWindow(
                    train_start=train[0],
                    train_end=train[-1],
                    test_start=test[0],
                    test_end=test[-1],
                )
            )
            start += test_size
        return windows
