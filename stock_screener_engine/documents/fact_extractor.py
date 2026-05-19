"""Lightweight structured fact extraction from financial documents."""

from __future__ import annotations

import re

from stock_screener_engine.documents.schemas import DocumentFact


_FACT_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("financial", "revenue", re.compile(r"\b(revenue|income from operations)\b[^.\n]{0,80}?([0-9,.]+)\s*(crore|cr|million|mn|billion|bn)?", re.I)),
    ("financial", "ebitda", re.compile(r"\bebitda\b[^.\n]{0,80}?([0-9,.]+)\s*(crore|cr|million|mn|billion|bn)?", re.I)),
    ("financial", "pat", re.compile(r"\b(PAT|profit after tax|net profit)\b[^.\n]{0,80}?([0-9,.]+)\s*(crore|cr|million|mn|billion|bn)?", re.I)),
    ("margin", "margin", re.compile(r"\b(ebitda margin|pat margin|gross margin|operating margin)\b[^.\n]{0,80}?([0-9,.]+)\s*%", re.I)),
    ("growth", "growth", re.compile(r"\b(growth|grew|increased|declined)\b[^.\n]{0,80}?([0-9,.]+)\s*%", re.I)),
    ("debt", "debt", re.compile(r"\b(debt|borrowings|net debt)\b[^.\n]{0,80}?([0-9,.]+)\s*(crore|cr|million|mn|billion|bn)?", re.I)),
    ("cash_flow", "cash flow", re.compile(r"\b(cash flow|operating cash flow|free cash flow)\b[^.\n]{0,80}?([0-9,.]+)\s*(crore|cr|million|mn|billion|bn)?", re.I)),
    ("capital_allocation", "capex", re.compile(r"\b(capex|capital expenditure)\b[^.\n]{0,80}?([0-9,.]+)\s*(crore|cr|million|mn|billion|bn)?", re.I)),
    ("ownership", "shareholding", re.compile(r"\b(promoter|fii|dii|institutional)\b[^.\n]{0,80}?([0-9,.]+)\s*%", re.I)),
]

_GOVERNANCE_KEYWORDS = {
    "related party": "related-party transaction reference",
    "qualified opinion": "qualified audit opinion reference",
    "emphasis of matter": "auditor emphasis-of-matter reference",
    "pledge": "pledge reference",
    "litigation": "litigation reference",
}

_CORPORATE_ACTION_KEYWORDS = {
    "dividend": "dividend reference",
    "buyback": "buyback reference",
    "bonus": "bonus issue reference",
    "split": "stock split reference",
    "merger": "merger reference",
    "demerger": "demerger reference",
}


class RuleFinancialFactExtractor:
    def extract(self, text: str, max_facts: int = 80) -> list[DocumentFact]:
        facts: list[DocumentFact] = []
        for kind, label, pattern in _FACT_PATTERNS:
            for match in pattern.finditer(text):
                value, unit = _value_and_unit(match)
                facts.append(
                    DocumentFact(
                        kind=kind,
                        label=label,
                        value=value,
                        unit=unit,
                        confidence=0.55,
                        evidence=_evidence(text, match.start(), match.end()),
                    )
                )
                if len(facts) >= max_facts:
                    return facts

        lowered = text.lower()
        for keyword, label in _GOVERNANCE_KEYWORDS.items():
            idx = lowered.find(keyword)
            if idx >= 0:
                facts.append(
                    DocumentFact(
                        kind="governance",
                        label=label,
                        value="mentioned",
                        confidence=0.5,
                        evidence=_evidence(text, idx, idx + len(keyword)),
                    )
                )

        for keyword, label in _CORPORATE_ACTION_KEYWORDS.items():
            idx = lowered.find(keyword)
            if idx >= 0:
                facts.append(
                    DocumentFact(
                        kind="corporate_action",
                        label=label,
                        value="mentioned",
                        confidence=0.5,
                        evidence=_evidence(text, idx, idx + len(keyword)),
                    )
                )

        return facts[:max_facts]


def _clean_number(value: str) -> str:
    return value.replace(",", "").strip()


def _value_and_unit(match: re.Match[str]) -> tuple[str, str]:
    groups = [g for g in match.groups() if g]
    value = ""
    unit = ""
    for group in groups:
        cleaned = _clean_number(group)
        if re.fullmatch(r"[0-9.]+", cleaned):
            value = cleaned
        elif group.lower() in {"crore", "cr", "million", "mn", "billion", "bn"}:
            unit = group
    return value or "mentioned", unit


def _evidence(text: str, start: int, end: int) -> str:
    left = max(0, start - 120)
    right = min(len(text), end + 120)
    return " ".join(text[left:right].split())[:300]
