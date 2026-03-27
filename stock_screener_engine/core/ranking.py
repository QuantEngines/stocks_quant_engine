"""Ranking utilities for generated scores."""

from __future__ import annotations

from collections.abc import Iterable

from stock_screener_engine.core.entities import ScoreCard
from stock_screener_engine.core.scoring_ranking import rank_score_cards


def rank_by_long_term(scores: Iterable[ScoreCard]) -> list[ScoreCard]:
    return rank_score_cards(scores, metric="long_term_score")


def rank_by_swing(scores: Iterable[ScoreCard]) -> list[ScoreCard]:
    return rank_score_cards(scores, metric="swing_score")
