"""Compatibility stub that now delegates to :mod:`core.rule_extraction`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .rule_extraction import DocumentLoader, RuleExtractionSettings, TextChunker


@dataclass
class IngestConfig:
    chunk_size: int = 800
    chunk_overlap: int = 100


class DocumentIngester:
    """Fast ingestion that reuses the shared loader + chunker."""

    def __init__(self, config: IngestConfig | None = None) -> None:
        settings = RuleExtractionSettings()
        config = config or IngestConfig(
            chunk_size=settings.max_chunk_tokens,
            chunk_overlap=settings.chunk_overlap_tokens,
        )
        self.config = config
        self.loader = DocumentLoader()
        self.chunker = TextChunker()

    def load_pdf(self, pdf_path: str) -> str:
        text, _ = self.loader._load_pdf(Path(pdf_path))  # type: ignore[attr-defined]
        return text

    def extract_text_from_file(self, file_path: str) -> str:
        text, _ = self.loader._load_text(Path(file_path))
        return text

    def extract_metadata(self, text: str, filename: str) -> Dict[str, int]:  # pragma: no cover - legacy API
        return {
            "source_file": filename,
            "char_count": len(text),
            "word_count": len(text.split()),
        }

    def chunk_text_smart(self, text: str, chunk_size: int | None = None, overlap: int | None = None) -> List[str]:
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        return self.chunker.split(text, window=chunk_size, overlap=overlap)

    def process_document(self, file_path: str) -> Tuple[str, dict, List[str]]:
        payload = self.loader._load_sync(Path(file_path))
        chunks = self.chunk_text_smart(
            payload.text,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        return payload.text, payload.metadata, chunks

    def batch_process(self, file_paths: List[str], max_workers: int | None = None):  # pragma: no cover - legacy path
        return [self.process_document(path) for path in file_paths]