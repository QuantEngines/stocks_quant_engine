"""NSE/BSE-style trading calendar support with explicit holiday overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterable

from stock_screener_engine.data_sources.schemas import MarketSessionRecord


@dataclass(frozen=True)
class MarketCalendar:
    venue: str = "NSE"
    holidays: frozenset[date] = field(default_factory=frozenset)
    special_trading_days: frozenset[date] = field(default_factory=frozenset)

    def is_trading_day(self, day: date) -> bool:
        if day in self.special_trading_days:
            return True
        if day in self.holidays:
            return False
        return day.weekday() < 5

    def sessions(self, start: date, end: date) -> list[MarketSessionRecord]:
        if end < start:
            raise ValueError("end date must be >= start date")
        out: list[MarketSessionRecord] = []
        day = start
        while day <= end:
            if day in self.special_trading_days:
                session_type = "special"
                reason = "special_trading_day"
            elif day in self.holidays:
                session_type = "holiday"
                reason = "configured_holiday"
            elif day.weekday() >= 5:
                session_type = "weekend"
                reason = "weekend"
            else:
                session_type = "regular"
                reason = ""
            out.append(
                MarketSessionRecord(
                    venue=self.venue,
                    session_date=day,
                    is_trading_day=self.is_trading_day(day),
                    session_type=session_type,
                    reason=reason,
                )
            )
            day += timedelta(days=1)
        return out

    def trading_days(self, start: date, end: date) -> list[date]:
        return [session.session_date for session in self.sessions(start, end) if session.is_trading_day]

    @classmethod
    def from_dates(
        cls,
        venue: str = "NSE",
        holidays: Iterable[date] | None = None,
        special_trading_days: Iterable[date] | None = None,
    ) -> "MarketCalendar":
        return cls(
            venue=venue,
            holidays=frozenset(holidays or []),
            special_trading_days=frozenset(special_trading_days or []),
        )
