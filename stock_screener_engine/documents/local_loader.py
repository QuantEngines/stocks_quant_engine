"""Local text/PDF document loading with optional PDF dependencies."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from stock_screener_engine.documents.schemas import FinancialDocument


@dataclass(frozen=True)
class LoadedDocument:
    document: FinancialDocument
    warnings: list[str]


class LocalDocumentLoader:
    def load(
        self,
        file_path: str,
        symbol: str,
        company_name: str | None = None,
        document_type: str = "unknown",
        publication_date: date | None = None,
        fiscal_period: str | None = None,
    ) -> LoadedDocument:
        path = Path(file_path)
        warnings: list[str] = []
        if not path.exists():
            raise FileNotFoundError(f"document not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text, pdf_warnings = self._read_pdf(path)
            warnings.extend(pdf_warnings)
        else:
            text = path.read_text(encoding="utf-8", errors="replace")

        quality_score = _quality_score(text=text, warnings=warnings)
        document = FinancialDocument(
            document_id=_document_id(path, symbol),
            symbol=symbol.strip().upper(),
            company_name=company_name or symbol.strip().upper(),
            document_type=document_type,
            source=str(path),
            publication_date=publication_date,
            fiscal_period=fiscal_period,
            title=path.name,
            extracted_text=text,
            metadata={"file_name": path.name, "file_suffix": suffix},
            quality_score=quality_score,
        )
        return LoadedDocument(document=document, warnings=warnings)

    def _read_pdf(self, path: Path) -> tuple[str, list[str]]:
        warnings: list[str] = []
        reader_cls: Any
        try:
            from pypdf import PdfReader as _PypdfReader  # type: ignore[import-not-found]
            reader_cls = _PypdfReader
        except ModuleNotFoundError:
            try:
                from PyPDF2 import PdfReader as _PyPDF2Reader  # type: ignore[import-not-found]
                reader_cls = _PyPDF2Reader
            except ModuleNotFoundError:
                return "", ["PDF parser unavailable; install pypdf or PyPDF2 for text extraction."]

        try:
            reader = reader_cls(str(path))
            chunks: list[str] = []
            for idx, page in enumerate(reader.pages):
                try:
                    chunks.append(page.extract_text() or "")
                except Exception as exc:  # pragma: no cover - parser/page specific
                    warnings.append(f"Could not extract page {idx + 1}: {exc}")
            text = "\n\n".join(chunks).strip()
            if not text:
                warnings.append("PDF text extraction returned no text; OCR may be required.")
            return text, warnings
        except Exception as exc:  # pragma: no cover - parser specific
            return "", [f"PDF parser failed: {exc}"]


def _document_id(path: Path, symbol: str) -> str:
    seed = f"{symbol.upper()}:{path.resolve()}:{path.stat().st_mtime_ns}".encode("utf-8")
    return hashlib.sha1(seed).hexdigest()[:16]


def _quality_score(text: str, warnings: list[str]) -> float:
    if not text.strip():
        return 0.0
    length_score = min(1.0, len(text) / 20_000.0)
    warning_penalty = min(0.5, len(warnings) * 0.15)
    return round(max(0.0, 0.35 + 0.65 * length_score - warning_penalty), 3)
