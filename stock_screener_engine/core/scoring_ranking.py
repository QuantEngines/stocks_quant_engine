"""Ranking helpers for signal candidates and scorecards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from stock_screener_engine.core.entities import ScoreCard


SortMetric = Literal["long_term_score", "swing_score", "conviction"]


@dataclass(frozen=True)
class RankingConfig:
    min_long_term_score: float = 0.0
    min_swing_score: float = 0.0
    top_k_long_term: int = 50
    top_k_swing: int = 50


def rank_score_cards(cards: Iterable[ScoreCard], metric: SortMetric) -> list[ScoreCard]:
    return sorted(cards, key=lambda c: (getattr(c, metric), c.conviction, c.symbol), reverse=True)


def filter_long_term(cards: Iterable[ScoreCard], cfg: RankingConfig) -> list[ScoreCard]:
    rows = [c for c in cards if c.long_term_score >= cfg.min_long_term_score]
    return rank_score_cards(rows, "long_term_score")[: cfg.top_k_long_term]


def filter_swing(cards: Iterable[ScoreCard], cfg: RankingConfig) -> list[ScoreCard]:
    rows = [c for c in cards if c.swing_score >= cfg.min_swing_score]
    return rank_score_cards(rows, "swing_score")[: cfg.top_k_swing]
