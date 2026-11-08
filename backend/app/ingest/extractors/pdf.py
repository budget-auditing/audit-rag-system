"""PDF extractor using pdfplumber."""
from typing import List


def extract_pdf(path: str) -> tuple[str, List[dict], dict]:
    import pdfplumber

    all_texts = []
    tables = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            all_texts.append(txt)
            page_tables = page.extract_tables() or []
            for t in page_tables:
                if isinstance(t, list) and len(t) > 0:
                    tables.append({"page": i, "rows": t})
    text = "\n".join(all_texts)
    return text, tables, {"pages": len(pdf.pages)}
