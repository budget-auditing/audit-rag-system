"""TXT / MD extractor."""
from pathlib import Path
from typing import List


def extract_txt(path: str) -> tuple[str, List[dict], dict]:
    """Return (full_text, tables=[], meta)."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return text, [], {"pages": text.count("\f") + 1}
