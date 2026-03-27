"""Future ML ranker interface placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLRankerConfig:
    model_name: str = "placeholder_xgboost"
    enabled: bool = False


class MLRanker:
    def __init__(self, config: MLRankerConfig | None = None) -> None:
        self.config = config or MLRankerConfig()

    def rank(self, scored_symbols: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return sorted(scored_symbols, key=lambda x: x[1], reverse=True)
