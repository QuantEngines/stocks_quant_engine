"""Composable NLP pipeline: ingestion -> preprocessing -> classify -> extract -> sentiment -> feature aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from stock_screener_engine.llm.extraction.document_classifier import LLMDocumentClassifier
from stock_screener_engine.llm.extraction.event_extractor import LLMEventExtractor
from stock_screener_engine.llm.extraction.management_tone_extractor import LLMManagementToneExtractor
from stock_screener_engine.llm.extraction.sentiment_extractor import LLMSentimentExtractor
from stock_screener_engine.nlp.event_engine.audit import LowConfidenceAuditSink
from stock_screener_engine.nlp.classification.rule_classifier import RuleDocumentClassifier
from stock_screener_engine.nlp.event_engine.aggregation import EventFeatureAggregator
from stock_screener_engine.nlp.extraction.rule_event_extractor import RuleEventExtractor
from stock_screener_engine.nlp.ingestion.document_ingestor import TextDocumentIngestor
from stock_screener_engine.nlp.preprocessing.chunking import chunk_document
from stock_screener_engine.nlp.preprocessing.cleaning import preprocess_documents
from stock_screener_engine.nlp.schemas.events import DocumentAnalysis, TextFeatureSet
from stock_screener_engine.nlp.sentiment.rule_sentiment import RuleSentimentEngine


@dataclass
class TextIntelligencePipeline:
    ingestor: TextDocumentIngestor
    classifier: RuleDocumentClassifier = field(default_factory=RuleDocumentClassifier)
    extractor: RuleEventExtractor = field(default_factory=RuleEventExtractor)
    sentiment: RuleSentimentEngine = field(default_factory=RuleSentimentEngine)
    aggregator: EventFeatureAggregator = field(default_factory=EventFeatureAggregator)
    enable_sentiment: bool = True
    enable_event_extraction: bool = True
    llm_enabled: bool = False
    llm_min_confidence: float = 0.55
    llm_fallback_to_rules: bool = True
    llm_provider_name: str = "heuristic"
    llm_model_name: str = "heuristic-finance-v1"
    audit_low_confidence: bool = False
    audit_sink: LowConfidenceAuditSink | None = None
    llm_classifier: LLMDocumentClassifier | None = None
    llm_event_extractor: LLMEventExtractor | None = None
    llm_sentiment_extractor: LLMSentimentExtractor | None = None
    llm_management_tone_extractor: LLMManagementToneExtractor | None = None

    def run(self, symbols: list[str], as_of: datetime, lookback_days: int = 30) -> TextFeatureSet:
        raw_docs = self.ingestor.ingest(symbols=symbols, lookback_days=lookback_days)
        docs = preprocess_documents(raw_docs)

        analyses: list[DocumentAnalysis] = []
        for doc in docs:
            chunks = chunk_document(doc)
            category = self.classifier.classify(doc)
            llm_conf = 0.0
            if self.llm_enabled and self.llm_classifier is not None:
                llm_category, conf = self.llm_classifier.classify(doc)
                llm_conf = conf
                if conf >= self.llm_min_confidence:
                    category = llm_category
                else:
                    self._audit_low_confidence(doc=doc, task="classification", confidence=conf, used_fallback=True)

            base_decay = 1.0 / max(1.0, len(chunks))
            entities = {
                "company": [doc.symbol],
                "keywords": doc.metadata.get("entities_keywords", "").split(",") if doc.metadata.get("entities_keywords") else [],
            }

            event = None
            if self.llm_enabled and self.enable_event_extraction and self.llm_event_extractor is not None:
                llm_event = self.llm_event_extractor.extract(doc, entities=entities, time_decay_factor=base_decay)
                if llm_event is not None and llm_event.confidence >= self.llm_min_confidence:
                    event = llm_event
                elif llm_event is not None:
                    self._audit_low_confidence(
                        doc=doc,
                        task="event_extraction",
                        confidence=llm_event.confidence,
                        used_fallback=self.llm_fallback_to_rules,
                    )

            if event is None:
                if not self.llm_enabled or self.llm_fallback_to_rules:
                    event = self.extractor.extract(doc, time_decay_factor=base_decay, entities=entities)
                else:
                    event = self.extractor.extract(doc, time_decay_factor=base_decay, entities={"company": [doc.symbol], "keywords": []})

            sentiments = self.sentiment.analyze(doc) if self.enable_sentiment else []
            balance_sheet_sentiment = 0.0
            if self.llm_enabled and self.enable_sentiment and self.llm_sentiment_extractor is not None:
                llm_sents, bs = self.llm_sentiment_extractor.extract(doc)
                llm_conf_sent = max((s.confidence for s in llm_sents), default=0.0)
                if llm_conf_sent >= self.llm_min_confidence:
                    sentiments = llm_sents
                    llm_conf = max(llm_conf, llm_conf_sent)
                else:
                    self._audit_low_confidence(
                        doc=doc,
                        task="sentiment",
                        confidence=llm_conf_sent,
                        used_fallback=True,
                    )
                balance_sheet_sentiment = bs

            management_tone = 0.0
            transcript_quality = 0.0
            if self.llm_enabled and self.llm_management_tone_extractor is not None and doc.source.value in {"transcript", "interview"}:
                management_tone, transcript_quality, _, tone_conf = self.llm_management_tone_extractor.extract(doc)
                if tone_conf < self.llm_min_confidence:
                    self._audit_low_confidence(
                        doc=doc,
                        task="management_tone",
                        confidence=tone_conf,
                        used_fallback=False,
                    )

            analyses.append(
                DocumentAnalysis(
                    document_id=doc.id,
                    symbol=doc.symbol,
                    category=category,
                    event=event,
                    sentiments=sentiments,
                    management_tone_score=management_tone,
                    transcript_quality_signal=transcript_quality,
                    balance_sheet_sentiment=balance_sheet_sentiment,
                    llm_confidence=llm_conf,
                )
            )

        return self.aggregator.aggregate(analyses, as_of=as_of)

    def _audit_low_confidence(self, doc: object, task: str, confidence: float, used_fallback: bool) -> None:
        if not self.audit_low_confidence or self.audit_sink is None:
            return
        self.audit_sink.write(
            task=task,
            symbol=getattr(doc, "symbol", ""),
            document_id=getattr(doc, "id", ""),
            confidence=confidence,
            threshold=self.llm_min_confidence,
            provider=self.llm_provider_name,
            model=self.llm_model_name,
            used_fallback=used_fallback,
            title=str(getattr(doc, "title", "")),
            source=str(getattr(getattr(doc, "source", ""), "value", "")),
        )
