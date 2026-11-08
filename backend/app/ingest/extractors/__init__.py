"""Text extractors."""
from .txt import extract_txt
from .docx import extract_docx
from .xlsx import extract_xlsx
from .pdf import extract_pdf

__all__ = ["extract_txt", "extract_docx", "extract_xlsx", "extract_pdf"]
