"""Rule-based extraction of management commentary, risks, and governance cues."""

from __future__ import annotations

import re


_COMMENTARY_KEYWORDS = [
    "outlook",
    "demand",
    "margin",
    "order book",
    "capex",
    "capacity",
    "pricing",
    "cost pressure",
    "rural",
    "export",
]

_RISK_KEYWORDS = [
    "risk",
    "uncertain",
    "slowdown",
    "litigation",
    "contingent liability",
    "regulatory",
    "commodity price",
    "foreign exchange",
]

_GOVERNANCE_KEYWORDS = [
    "auditor",
    "qualified opinion",
    "related party",
    "pledge",
    "resignation",
    "independent director",
]


class RuleCommentaryExtractor:
    def extract_management_commentary(self, text: str, limit: int = 12) -> list[str]:
        return _sentences_with_keywords(text, _COMMENTARY_KEYWORDS, limit)

    def extract_risk_factors(self, text: str, limit: int = 12) -> list[str]:
        return _sentences_with_keywords(text, _RISK_KEYWORDS, limit)

    def extract_governance_flags(self, text: str, limit: int = 12) -> list[str]:
        return _sentences_with_keywords(text, _GOVERNANCE_KEYWORDS, limit)


def _sentences_with_keywords(text: str, keywords: list[str], limit: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    lowered_keywords = [k.lower() for k in keywords]
    out: list[str] = []
    for sentence in sentences:
        low = sentence.lower()
        if any(keyword in low for keyword in lowered_keywords):
            out.append(sentence[:500])
        if len(out) >= limit:
            break
    return out
