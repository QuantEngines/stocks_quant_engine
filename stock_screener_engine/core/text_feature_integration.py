"""Integration bridge for NLP text features into core feature vectors."""

from __future__ import annotations


class TextFeatureIntegration:
    def merge(self, symbol: str, text_feature_map: dict[str, dict[str, float]] | None) -> dict[str, float]:
        if not text_feature_map:
            return {}
        return dict(text_feature_map.get(symbol, {}))
