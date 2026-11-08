"""Orchestrator: detect file type and dispatch extractor."""
from pathlib import Path
from typing import List

from app.ingest.extractors import extract_txt, extract_docx, extract_xlsx, extract_pdf


def parse_file(path: str) -> tuple[str, List[dict], dict]:
    """Return (full_text, tables, meta)."""
    ext = Path(path).suffix.lower()
    if ext in {".txt", ".md", ".markdown", ".json", ".csv"}:
        return extract_txt(path)
    if ext == ".docx":
        return extract_docx(path)
    if ext in {".xlsx", ".xls"}:
        return extract_xlsx(path)
    if ext == ".pdf":
        return extract_pdf(path)
    return extract_txt(path)
