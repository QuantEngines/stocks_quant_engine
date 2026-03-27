"""Chunk long documents for downstream rule processing."""

from __future__ import annotations

from stock_screener_engine.nlp.schemas.events import NormalizedDocument


def chunk_document(doc: NormalizedDocument, max_chars: int = 1200) -> list[str]:
    text = doc.body_text.strip()
    if not text:
        return [doc.title]
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunks.append(text[start:end])
        start = end
    return chunks
