"""Document intelligence pipeline for local financial reports."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from stock_screener_engine.config.settings import AppSettings
from stock_screener_engine.documents.commentary_extractor import RuleCommentaryExtractor
from stock_screener_engine.documents.fact_extractor import RuleFinancialFactExtractor
from stock_screener_engine.documents.local_loader import LocalDocumentLoader
from stock_screener_engine.documents.schemas import DocumentIntelligenceResult
from stock_screener_engine.documents.section_classifier import RuleSectionClassifier
from stock_screener_engine.storage.local_files import LocalFileStorage


class DocumentIntelligencePipeline:
    """Parse a local financial document into structured research facts."""

    def __init__(
        self,
        settings: AppSettings,
        loader: LocalDocumentLoader | None = None,
        section_classifier: RuleSectionClassifier | None = None,
        fact_extractor: RuleFinancialFactExtractor | None = None,
        commentary_extractor: RuleCommentaryExtractor | None = None,
    ) -> None:
        self.settings = settings
        self.file_store = LocalFileStorage(settings.storage.root_dir)
        self.loader = loader or LocalDocumentLoader()
        self.section_classifier = section_classifier or RuleSectionClassifier()
        self.fact_extractor = fact_extractor or RuleFinancialFactExtractor()
        self.commentary_extractor = commentary_extractor or RuleCommentaryExtractor()

    def run(
        self,
        symbol: str,
        file_path: str,
        company_name: str | None = None,
        document_type: str = "unknown",
        publication_date: date | None = None,
        fiscal_period: str | None = None,
    ) -> DocumentIntelligenceResult:
        loaded = self.loader.load(
            file_path=file_path,
            symbol=symbol,
            company_name=company_name,
            document_type=document_type,
            publication_date=publication_date,
            fiscal_period=fiscal_period,
        )
        document = loaded.document
        warnings = list(loaded.warnings)
        max_chars = getattr(self.settings.documents, "max_text_chars", 250_000)
        if max_chars > 0 and len(document.extracted_text) > max_chars:
            document = replace(document, extracted_text=document.extracted_text[:max_chars])
            warnings.append(f"Document text truncated to configured max_text_chars={max_chars}.")
        min_quality = getattr(self.settings.documents, "min_quality_score", 0.30)
        if document.quality_score < min_quality:
            warnings.append(
                f"Document quality score {document.quality_score:.2f} below configured minimum {min_quality:.2f}."
            )
        section_map = self.section_classifier.classify(document.extracted_text)
        document = replace(document, section_map=section_map)
        facts = self.fact_extractor.extract(document.extracted_text)
        commentary = self.commentary_extractor.extract_management_commentary(document.extracted_text)
        risks = self.commentary_extractor.extract_risk_factors(document.extracted_text)
        governance = self.commentary_extractor.extract_governance_flags(document.extracted_text)

        result = DocumentIntelligenceResult(
            document=document,
            facts=facts,
            management_commentary=commentary,
            risk_factors=risks,
            governance_flags=governance,
            extraction_warnings=warnings,
        )
        self.file_store.save_json(
            result.to_dict(),
            filename=f"{document.document_id}.json",
            subdir="documents",
        )
        return result
