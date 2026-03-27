"""Event study analysis for catalyst-based signal evaluation.

Computes:
* avg_pre_event, avg_post_event  — mean cumulative return before / after event
* post_minus_pre                 — simple abnormal return proxy
* win_rate                       — fraction of events where post > pre
* car                            — mean cumulative abnormal return (= post - pre)
* t_stat                         — paired t-test: H0: post_return == pre_return
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EventWindowReturn:
    symbol: str
    event_date: str
    pre_event_return: float
    post_event_return: float


@dataclass(frozen=True)
class EventStudySummary:
    n_events: int
    avg_pre_event: float
    avg_post_event: float
    post_minus_pre: float
    win_rate: float
    car: float              # cumulative abnormal return = avg(post - pre)
    car_t_stat: float       # paired t-stat for CAR != 0
    car_std: float          # cross-event std of abnormal returns


class EventStudy:
    def summarize(self, windows: list[EventWindowReturn]) -> dict[str, float]:
        """Legacy dict interface for backward compatibility."""
        s = self.analyze(windows)
        return {
            "avg_pre_event":  s.avg_pre_event,
            "avg_post_event": s.avg_post_event,
            "post_minus_pre": s.post_minus_pre,
            "win_rate":       s.win_rate,
            "car":            s.car,
            "car_t_stat":     s.car_t_stat,
        }

    def analyze(self, windows: list[EventWindowReturn]) -> EventStudySummary:
        """Return a typed summary with all diagnostics."""
        n = len(windows)
        if n == 0:
            return EventStudySummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        pre  = [w.pre_event_return  for w in windows]
        post = [w.post_event_return for w in windows]
        abns = [p - r for p, r in zip(post, pre)]   # abnormal returns

        avg_pre  = sum(pre)  / n
        avg_post = sum(post) / n
        avg_abn  = sum(abns) / n
        wins     = sum(1 for a in abns if a > 0)

        # std of abnormal returns
        if n > 1:
            var_abn = sum((a - avg_abn) ** 2 for a in abns) / (n - 1)
            std_abn = math.sqrt(var_abn)
        else:
            std_abn = 0.0

        # paired t-stat: mean_abn / (std_abn / sqrt(n))
        se     = std_abn / math.sqrt(n) if n > 1 and std_abn > 1e-12 else 1e-12
        t_stat = avg_abn / se

        return EventStudySummary(
            n_events       = n,
            avg_pre_event  = avg_pre,
            avg_post_event = avg_post,
            post_minus_pre = avg_post - avg_pre,
            win_rate       = wins / n,
            car            = avg_abn,
            car_t_stat     = t_stat,
            car_std        = std_abn,
        )
