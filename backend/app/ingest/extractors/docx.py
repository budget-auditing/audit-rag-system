"""DOCX extractor using python-docx."""
from typing import List


def extract_docx(path: str) -> tuple[str, List[dict], dict]:
    import docx

    doc = docx.Document(path)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    tables = []
    for i, table in enumerate(doc.tables):
        rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        tables.append({"index": i, "rows": rows, "caption": None})
    text = "\n".join(paragraphs)
    return text, tables, {"paragraphs": len(paragraphs), "tables": len(tables)}
