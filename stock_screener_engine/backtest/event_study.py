"""Event study scaffold for catalyst-based analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EventWindowReturn:
    symbol: str
    event_date: str
    pre_event_return: float
    post_event_return: float


class EventStudy:
    def summarize(self, windows: list[EventWindowReturn]) -> dict[str, float]:
        if not windows:
            return {"avg_pre_event": 0.0, "avg_post_event": 0.0, "post_minus_pre": 0.0}
        avg_pre = sum(w.pre_event_return for w in windows) / len(windows)
        avg_post = sum(w.post_event_return for w in windows) / len(windows)
        return {
            "avg_pre_event": avg_pre,
            "avg_post_event": avg_post,
            "post_minus_pre": avg_post - avg_pre,
        }
