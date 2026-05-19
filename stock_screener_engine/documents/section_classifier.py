"""Rule-based financial document section detection."""

from __future__ import annotations

import re

from stock_screener_engine.documents.schemas import DocumentSection


_SECTION_KEYWORDS: dict[str, list[str]] = {
    "business_overview": ["business overview", "our business", "company overview"],
    "management_discussion": ["management discussion", "md&a", "management commentary"],
    "financial_statements": ["standalone financial", "consolidated financial", "statement of profit"],
    "segment_analysis": ["segment", "segment-wise", "business segment"],
    "risk_factors": ["risk factor", "risks and concerns", "principal risks"],
    "governance": ["corporate governance", "related party", "auditor", "audit"],
    "capital_allocation": ["capital expenditure", "capex", "dividend", "buyback"],
    "outlook": ["outlook", "future prospects", "demand environment"],
}


class RuleSectionClassifier:
    def classify(self, text: str) -> dict[str, DocumentSection]:
        lowered = text.lower()
        sections: dict[str, DocumentSection] = {}
        for name, keywords in _SECTION_KEYWORDS.items():
            matches = [m.start() for kw in keywords for m in re.finditer(re.escape(kw), lowered)]
            if not matches:
                continue
            start = min(matches)
            end = min(len(text), start + 5_000)
            preview = " ".join(text[start:end].split())[:500]
            sections[name] = DocumentSection(
                name=name,
                start_char=start,
                end_char=end,
                text_preview=preview,
                confidence=0.65,
            )
        return sections
