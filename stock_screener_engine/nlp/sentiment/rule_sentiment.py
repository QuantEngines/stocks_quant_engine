"""Domain-aware rule sentiment for earnings, governance, outlook, and macro dimensions."""

from __future__ import annotations

from stock_screener_engine.nlp.schemas.events import NormalizedDocument, SentimentSignal, SentimentType

_SENTIMENT_RULES: dict[SentimentType, tuple[tuple[str, ...], tuple[str, ...]]] = {
    SentimentType.EARNINGS: (("beat", "margin expansion", "strong earnings", "growth"), ("miss", "margin pressure", "weak earnings")),
    SentimentType.GOVERNANCE: (("promoter buying", "clean audit", "resolved"), ("resigned", "audit issue", "litigation", "penalty")),
    SentimentType.OUTLOOK: (("guidance raised", "optimistic", "acceleration"), ("guidance cut", "cautious", "slowdown")),
    SentimentType.MACRO: (("tailwind", "policy support", "demand rebound"), ("headwind", "inflation", "rate hike")),
}


class RuleSentimentEngine:
    def analyze(self, doc: NormalizedDocument) -> list[SentimentSignal]:
        text = f"{doc.title} {doc.body_text}".lower()
        out: list[SentimentSignal] = []
        for sentiment_type, (positives, negatives) in _SENTIMENT_RULES.items():
            pos = sum(p in text for p in positives)
            neg = sum(n in text for n in negatives)
            total = pos + neg
            polarity = 0.0 if total == 0 else (pos - neg) / max(1, total)
            intensity = min(1.0, 0.2 + 0.2 * total)
            confidence = min(1.0, 0.3 + 0.25 * total)
            out.append(
                SentimentSignal(
                    polarity_score=max(-1.0, min(1.0, polarity)),
                    confidence=confidence,
                    sentiment_type=sentiment_type,
                    intensity=intensity,
                )
            )
        return out
