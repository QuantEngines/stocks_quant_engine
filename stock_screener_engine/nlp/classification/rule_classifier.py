"""Rule-based document classifier with explicit upgrade path for ML/LLM models."""

from __future__ import annotations

from stock_screener_engine.nlp.schemas.events import DocumentCategory, NormalizedDocument

_CATEGORY_RULES: list[tuple[DocumentCategory, tuple[str, ...]]] = [
    (DocumentCategory.EARNINGS_RELATED, ("earnings", "quarterly", "results", "ebitda", "guidance")),
    (DocumentCategory.CORPORATE_ACTION, ("order", "capex", "acquisition", "merger", "stake", "dividend")),
    (DocumentCategory.MANAGEMENT_COMMENTARY, ("management", "conference call", "commentary", "outlook")),
    (DocumentCategory.MACRO_SECTOR, ("macro", "inflation", "rate", "policy", "sector")),
    (DocumentCategory.GENERAL_NEWS, ("news", "update", "announced", "reports")),
]


class RuleDocumentClassifier:
    def classify(self, doc: NormalizedDocument) -> DocumentCategory:
        text = f"{doc.title} {doc.body_text}".lower()
        for category, keywords in _CATEGORY_RULES:
            if any(k in text for k in keywords):
                return category
        return DocumentCategory.UNKNOWN
