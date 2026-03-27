"""Text preprocessing: cleaning, dedup, timestamp normalization, and entity tagging."""

from __future__ import annotations

import re
from datetime import datetime

from stock_screener_engine.nlp.schemas.events import NormalizedDocument

_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    cleaned = _SPACE_RE.sub(" ", text.replace("\u00a0", " ")).strip()
    return cleaned


def normalize_timestamp(value: datetime) -> datetime:
    return value.replace(microsecond=0)


def deduplicate_documents(documents: list[NormalizedDocument]) -> list[NormalizedDocument]:
    seen: set[tuple[str, str, str]] = set()
    out: list[NormalizedDocument] = []
    for doc in sorted(documents, key=lambda d: (d.symbol, d.timestamp, d.id)):
        key = (doc.symbol, doc.title.lower().strip(), doc.body_text.lower().strip())
        if key in seen:
            continue
        seen.add(key)
        out.append(doc)
    return out


def tag_entities(doc: NormalizedDocument, symbol_aliases: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    aliases = (symbol_aliases or {}).get(doc.symbol, [])
    tokens = [doc.symbol] + aliases
    text = f"{doc.title} {doc.body_text}".lower()
    matched = [t for t in tokens if t and t.lower() in text]
    words = [w for w in re.findall(r"[A-Za-z0-9_]+", text) if len(w) > 4]
    return {
        "company": sorted(set(matched)) or [doc.symbol],
        "keywords": sorted(set(words[:20])),
    }


def preprocess_documents(
    documents: list[NormalizedDocument], symbol_aliases: dict[str, list[str]] | None = None
) -> list[NormalizedDocument]:
    normalized: list[NormalizedDocument] = []
    for doc in documents:
        title = clean_text(doc.title)
        body = clean_text(doc.body_text)
        ts = normalize_timestamp(doc.timestamp)
        md = dict(doc.metadata)
        tagged = tag_entities(doc, symbol_aliases=symbol_aliases)
        md["entities_company"] = ",".join(tagged.get("company", []))
        md["entities_keywords"] = ",".join(tagged.get("keywords", []))
        normalized.append(
            NormalizedDocument(
                id=doc.id,
                source=doc.source,
                timestamp=ts,
                symbol=doc.symbol,
                title=title,
                body_text=body,
                metadata=md,
            )
        )
    return deduplicate_documents(normalized)
