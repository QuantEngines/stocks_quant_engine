"""Walk-forward evaluation planner and result container."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WalkForwardWindow:
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str


@dataclass(frozen=True)
class WalkForwardWindowResult:
    """Evaluation metrics for a single walk-forward test window."""

    window: WalkForwardWindow
    ic:             float = 0.0    # Spearman IC over the test window
    ic_t_stat:      float = 0.0    # t-statistic for IC != 0
    hit_rate:       float = 0.0    # fraction of signals with positive return
    avg_return:     float = 0.0    # mean signal return in test window
    n_signals:      int   = 0      # number of signals evaluated


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward evaluation across all windows."""

    windows: list[WalkForwardWindowResult] = field(default_factory=list)

    def mean_ic(self) -> float:
        valid = [w.ic for w in self.windows if w.n_signals > 0]
        return sum(valid) / len(valid) if valid else 0.0

    def ic_stability(self) -> float:
        """Fraction of windows with IC > 0 (higher = more consistent alpha)."""
        valid = [w for w in self.windows if w.n_signals > 0]
        if not valid:
            return 0.0
        return sum(1 for w in valid if w.ic > 0) / len(valid)

    def mean_hit_rate(self) -> float:
        valid = [w.hit_rate for w in self.windows if w.n_signals > 0]
        return sum(valid) / len(valid) if valid else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "n_windows":      float(len(self.windows)),
            "mean_ic":        self.mean_ic(),
            "ic_stability":   self.ic_stability(),
            "mean_hit_rate":  self.mean_hit_rate(),
        }


class WalkForwardPlanner:
    """Generates non-overlapping train/test windows from a sorted date list."""

    def build_windows(
        self,
        dates: list[str],
        train_size: int,
        test_size: int,
    ) -> list[WalkForwardWindow]:
        windows: list[WalkForwardWindow] = []
        start = 0
        while start + train_size + test_size <= len(dates):
            train = dates[start : start + train_size]
            test  = dates[start + train_size : start + train_size + test_size]
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
