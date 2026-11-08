"""XLSX/XLS extractor using openpyxl (xlsx) or xlld (xls).

- unmerge and fill merged cells so budget tables keep their headers
- render each sheet as Markdown for better downstream text retrieval
- still return native table rows for structured chunking
"""
from typing import List, Tuple


def _load_xls_with_xlrd(path: str):
    """Convert old .xls workbook into an in-memory openpyxl workbook."""
    import xlrd
    import openpyxl

    rb = xlrd.open_workbook(path, formatting_info=False)
    wb = openpyxl.Workbook()
    # remove default sheet
    wb.remove(wb.active)

    for sheet_idx in range(rb.nsheets):
        rs = rb.sheet_by_index(sheet_idx)
        ws = wb.create_sheet(title=rs.name)
        for r in range(rs.nrows):
            row_values = []
            for c in range(rs.ncols):
                val = rs.cell_value(r, c)
                # Convert xlrd float dates to ISO strings when possible
                if isinstance(val, float):
                    try:
                        from xlrd import xldate_as_tuple
                        tup = xldate_as_tuple(val, rb.datemode)
                        val = f"{tup[0]:04d}-{tup[1]:02d}-{tup[2]:02d}"
                    except Exception:
                        pass
                row_values.append(val)
            ws.append(row_values)
    return wb


def _load_and_explode_merged_cells(path: str):
    import openpyxl

    if str(path).lower().endswith(".xls"):
        wb = _load_xls_with_xlrd(path)
    else:
        wb = openpyxl.load_workbook(path, data_only=True)
        for ws in wb.worksheets:
            # copy list because unmerge changes the collection
            for rng in list(ws.merged_cells.ranges):
                ws.unmerge_cells(str(rng))
                tl_cell = ws.cell(rng.min_row, rng.min_col)
                for r in range(rng.min_row, rng.max_row + 1):
                    for c in range(rng.min_col, rng.max_col + 1):
                        ws.cell(r, c).value = tl_cell.value
    return wb


def _sheet_to_markdown(ws) -> Tuple[str, List[List[str]]]:
    """Render an openpyxl worksheet as Markdown table text and row values."""
    rows: List[List[str]] = []
    for row in ws.iter_rows(values_only=True):
        rows.append([str(cell).strip() if cell is not None else "" for cell in row])

    # Drop fully-empty leading/trailing rows
    while rows and all(not c for c in rows[0]):
        rows.pop(0)
    while rows and all(not c for c in rows[-1]):
        rows.pop()

    if not rows:
        return "", []

    # Use first non-empty row as header; if every cell is empty fallback to generic headers
    header = rows[0]
    body = rows[1:]

    def _escape_cell(cell: str) -> str:
        # Escape pipe to avoid breaking markdown table syntax
        return cell.replace("|", "\\|")

    md = ["| " + " | ".join(_escape_cell(c) or " ") + " |" for c in header]
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        # Extend or truncate to match header length
        cells = (r + [""] * len(header))[: len(header)]
        md.append("| " + " | ".join(_escape_cell(c) for c in cells) + " |")
    return "\n".join(md), rows


def extract_xlsx(path: str) -> tuple[str, List[dict], dict]:
    """Return (full_text, tables, meta).

    full_text is a Markdown-flavored concatenation of all sheets, which improves
    downstream embedding/retrieval while tables still carries native rows.
    """
    wb = _load_and_explode_merged_cells(path)
    all_tables = []
    md_parts: List[str] = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        md, rows = _sheet_to_markdown(ws)
        if rows:
            all_tables.append({"sheet": sheet_name, "rows": rows})
            md_parts.append(f"## 工作表: {sheet_name}\n")
            md_parts.append(md)
            md_parts.append("")

    text = "\n".join(md_parts).strip()
    return text, all_tables, {"sheets": wb.sheetnames}
