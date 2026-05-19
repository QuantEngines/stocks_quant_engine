"""Structured schemas for financial document intelligence."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date


@dataclass(frozen=True)
class ExtractedTable:
    title: str
    rows: list[dict[str, str]]
    confidence: float = 0.0


@dataclass(frozen=True)
class DocumentSection:
    name: str
    start_char: int
    end_char: int
    text_preview: str
    confidence: float


@dataclass(frozen=True)
class DocumentFact:
    kind: str
    label: str
    value: str | float
    unit: str = ""
    period: str | None = None
    confidence: float = 0.0
    evidence: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FinancialDocument:
    document_id: str
    symbol: str
    company_name: str
    document_type: str
    source: str
    publication_date: date | None
    fiscal_period: str | None
    title: str
    extracted_text: str
    extracted_tables: list[ExtractedTable] = field(default_factory=list)
    section_map: dict[str, DocumentSection] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["publication_date"] = self.publication_date.isoformat() if self.publication_date else None
        return payload


@dataclass(frozen=True)
class DocumentIntelligenceResult:
    document: FinancialDocument
    facts: list[DocumentFact]
    management_commentary: list[str]
    risk_factors: list[str]
    governance_flags: list[str]
    extraction_warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        payload = self.document.to_dict()
        payload.update(
            {
                "facts": [fact.to_dict() for fact in self.facts],
                "management_commentary": list(self.management_commentary),
                "risk_factors": list(self.risk_factors),
                "governance_flags": list(self.governance_flags),
                "extraction_warnings": list(self.extraction_warnings),
            }
        )
        return payload
