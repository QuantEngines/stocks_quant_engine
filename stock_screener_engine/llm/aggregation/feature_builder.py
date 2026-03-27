"""Builds plain numeric feature maps from LLM-aggregated text feature sets."""

from __future__ import annotations

from stock_screener_engine.features.text_features.aggregator import to_feature_map
from stock_screener_engine.nlp.schemas.events import TextFeatureSet


class LLMFeatureBuilder:
    def build(self, feature_set: TextFeatureSet) -> dict[str, dict[str, float]]:
        return to_feature_map(feature_set)
