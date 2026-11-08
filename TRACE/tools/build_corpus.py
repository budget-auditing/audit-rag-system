#! /usr/bin/env python3
"""Convert a directory of source documents into Markdown under ``./corpus``.

This step only performs ``document -> Markdown`` conversion; it does not build
chunks or vector indices. The conversion rules are kept in sync with
``build_index.py``:

- **PDF**  - text extracted page-by-page with PyMuPDF and concatenated into a
  single ``.md`` file per PDF.
- **DOCX** - paragraph text extracted via ``python-docx``.
- **TXT / MD** - read directly.
- **XLS / XLSX** - converted with MarkItDown. For ``.xlsx`` we first explode
  merged cells and emit one Markdown file per sheet.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import tempfile
from typing import Iterable, List, Optional, Set

import tqdm
import fitz  # PyMuPDF
from docx import Document
from openpyxl import Workbook, load_workbook
from markitdown import MarkItDown


GENERAL_TEXT_EXTS = frozenset(["txt", "md", "markdown", "docx", "pdf"])
EXCEL_EXTS = frozenset(["xls", "xlsx"])


def _sanitize_filename(name: str) -> str:
    # Replace characters that are illegal on Windows for cross-platform safety.
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip().strip(".")
    return name or "untitled"


def _shorten_if_too_long(stem: str, max_len: int = 200) -> str:
    """Avoid file-name lengths that exceed common filesystem limits.

    Filenames are usually capped at 255 bytes. This helper keeps the original
    stem when possible and only truncates + appends a short hash when too long.
    """
    max_bytes = max_len  # Interpret as bytes for safety with non-ASCII names.
    b = stem.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return stem

    h = hashlib.md5(b).hexdigest()[:10]
    tail = "__" + h
    tail_b = tail.encode("utf-8", errors="ignore")
    keep_bytes = max(1, max_bytes - len(tail_b))

    # Binary-search for the largest prefix whose utf-8 length <= keep_bytes.
    lo, hi = 0, len(stem)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(stem[:mid].encode("utf-8", errors="ignore")) <= keep_bytes:
            lo = mid
        else:
            hi = mid - 1

    return stem[:lo] + tail


def _make_out_path(
    path_in: str,
    out_dir: str,
    *,
    suffix: str = "",
    force_ext: str = ".md",
) -> str:
    # Only use the original base name (without directories) to keep paths short.
    # Collisions are resolved by _ensure_unique_path via a __dupN suffix.
    src_stem = pathlib.Path(path_in).stem
    stem = _shorten_if_too_long(src_stem + suffix, max_len=240)
    out_name = stem + force_ext
    return os.path.join(out_dir, out_name)


def _ensure_unique_path(path_out: str) -> str:
    if not os.path.exists(path_out):
        return path_out
    base, ext = os.path.splitext(path_out)
    i = 2
    while True:
        cand = f"{base}__dup{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


def _write_text(path_out: str, text: str):
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    with open(path_out, "w", encoding="utf-8") as f:
        f.write(text or "")


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_md(path: str) -> str:
    return load_txt(path)


def load_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])


def load_pdf_text(path: str) -> str:
    parts: List[str] = []
    with fitz.open(path) as doc:
        # PyMuPDF Document is iterable at runtime but not always declared as
        # Iterable in its type stubs; use page_count + load_page explicitly.
        for i in range(doc.page_count):
            page = doc.load_page(i)
            raw = page.get_text("text")
            txt = raw if isinstance(raw, str) else str(raw or "")
            if txt.strip():
                parts.append(f"## Page {i + 1}\n\n{txt.strip()}")
    return "\n\n---\n\n".join(parts)


def convert_excel_to_md(md: MarkItDown, path_in: str) -> str:
    res = md.convert(path_in)
    return res.text_content or ""


def convert_xlsx_to_md_files(md: MarkItDown, path_in: str, out_dir: str):
    """Explode merged cells, then emit one Markdown file per sheet.

    Kept in sync with ``build_index.py``.
    """
    wb = None
    try:
        wb = load_workbook(path_in)
    except Exception as e:
        print(f"[WARN] load_workbook failed: {path_in}: {e}")
        return

    # Explode merged cells so every cell has its own value.
    for ws in wb.worksheets:
        for rng in list(ws.merged_cells.ranges):  # copy so we can mutate while iterating
            ws.unmerge_cells(str(rng))  # unmerge first; otherwise cells are read-only
            tl_cell = ws[rng.min_row][rng.min_col - 1]  # top-left anchor cell
            for r in range(rng.min_row, rng.max_row + 1):
                for c in range(rng.min_col, rng.max_col + 1):
                    ws.cell(r, c).value = tl_cell.value

    sheet_id_for_invalid_name = 0
    with tempfile.TemporaryDirectory(prefix="corpus_xlsx_") as tmp_dir:
        for sheet_idx, ws in enumerate(wb.worksheets, start=1):
            new_wb = Workbook()
            new_ws = new_wb.active
            if new_ws is None:
                new_ws = new_wb.create_sheet()
            new_ws.title = ws.title
            # Copy with values_only=True to avoid type-stub issues with MergedCell etc.
            for r_idx, row in enumerate(ws.iter_rows(values_only=True), start=1):
                for c_idx, value in enumerate(row, start=1):
                    new_ws.cell(row=r_idx, column=c_idx, value=value)

            safe_sheet_name = re.sub(r'[\\/:*?"<>|]', "_", ws.title.strip()) or "Sheet"
            if safe_sheet_name == "Sheet":
                safe_sheet_name = f"Sheet_{sheet_id_for_invalid_name}"
                sheet_id_for_invalid_name += 1

            tmp_xlsx = os.path.join(tmp_dir, f"sheet_{sheet_idx:03d}_{safe_sheet_name}.xlsx")
            try:
                new_wb.save(tmp_xlsx)
            except Exception as e:
                print(f"[WARN] failed to write temp xlsx: {path_in} / {ws.title}: {e}")
                continue

            try:
                md_text = convert_excel_to_md(md, tmp_xlsx)
            except Exception as e:
                print(f"[WARN] Excel conversion failed: {path_in} / {ws.title}: {e}")
                continue

            out_path = _make_out_path(
                path_in,
                out_dir,
                suffix=f"__{_sanitize_filename(safe_sheet_name)}",
                force_ext=".md",
            )
            out_path = _ensure_unique_path(out_path)
            _write_text(out_path, md_text)


def convert_xls_to_md_file(md: MarkItDown, path_in: str, out_dir: str):
    try:
        md_text = convert_excel_to_md(md, path_in)
    except Exception as e:
        print(f"[WARN] Excel conversion failed: {path_in}: {e}")
        return
    out_path = _make_out_path(path_in, out_dir, force_ext=".md")
    out_path = _ensure_unique_path(out_path)
    _write_text(out_path, md_text)


def convert_general_file_to_md_files(path_in: str, out_dir: str):
    low = path_in.lower()
    if low.endswith(".pdf"):
        try:
            txt = load_pdf_text(path_in)
        except Exception as e:
            print(f"[WARN] PDF read failed: {path_in}: {e}")
            return
        if not (txt or "").strip():
            return
        out_path = _make_out_path(path_in, out_dir, force_ext=".md")
        out_path = _ensure_unique_path(out_path)
        _write_text(out_path, txt)
        return

    if low.endswith(".docx"):
        try:
            txt = load_docx(path_in)
        except Exception as e:
            print(f"[WARN] DOCX read failed: {path_in}: {e}")
            return
        if not (txt or "").strip():
            return
        out_path = _make_out_path(path_in, out_dir, force_ext=".md")
        out_path = _ensure_unique_path(out_path)
        _write_text(out_path, txt)
        return

    if low.endswith(".txt"):
        try:
            txt = load_txt(path_in)
        except Exception as e:
            print(f"[WARN] TXT read failed: {path_in}: {e}")
            return
        if not (txt or "").strip():
            return
        out_path = _make_out_path(path_in, out_dir, force_ext=".md")
        out_path = _ensure_unique_path(out_path)
        _write_text(out_path, txt)
        return

    if low.endswith(".md") or low.endswith(".markdown"):
        try:
            txt = load_md(path_in)
        except Exception as e:
            print(f"[WARN] Markdown read failed: {path_in}: {e}")
            return
        if not (txt or "").strip():
            return
        out_path = _make_out_path(path_in, out_dir, force_ext=".md")
        out_path = _ensure_unique_path(out_path)
        _write_text(out_path, txt)
        return


def iter_files(input_dir: str, *, skip_dir_names: Set[str]) -> Iterable[str]:
    for root, dirs, files in os.walk(input_dir):
        # Filter in-place to avoid descending into ignored directories.
        dirs[:] = [d for d in dirs if d not in skip_dir_names and not d.startswith(".")]
        for fn in files:
            if fn.startswith(".~") or fn.startswith("~$"):
                # Excel lock files
                continue
            yield os.path.join(root, fn)


def _sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_corpus(input_dir: str, output_dir: str, *, skip_dir_names: Optional[Set[str]] = None):
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    if skip_dir_names is None:
        # Skip the output directory, caches, and the excel_mds dir used by build_index.py.
        skip_dir_names = {"corpus", "__pycache__", "index", "excel_mds"}

    os.makedirs(output_dir, exist_ok=True)

    md = MarkItDown(enable_plugins=False)
    stats = {"files_seen": 0, "files_converted": 0}
    seen_hashes: Set[str] = set()

    pbar = tqdm.tqdm(unit="file", desc="Building corpus")
    for path_in in iter_files(input_dir, skip_dir_names=skip_dir_names):
        stats["files_seen"] += 1
        pbar.update(1)

        # Deduplicate by content hash.
        try:
            digest = _sha256_file(path_in)
        except Exception as e:
            print(f"[WARN] failed to read for deduplication: {path_in}: {e}")
            continue
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        ext = pathlib.Path(path_in).suffix.lower().lstrip(".")
        if not ext:
            continue

        try:
            if ext in EXCEL_EXTS:
                stats["files_converted"] += 1
                if ext == "xlsx":
                    convert_xlsx_to_md_files(md, path_in, output_dir)
                else:
                    convert_xls_to_md_file(md, path_in, output_dir)
                continue

            if ext in GENERAL_TEXT_EXTS:
                stats["files_converted"] += 1
                convert_general_file_to_md_files(path_in, output_dir)
                continue
        except Exception as e:
            print(f"[WARN] conversion error: {path_in}: {e}")

    pbar.close()
    print(
        f"[build_corpus] done: seen {stats['files_seen']} files, "
        f"converted {stats['files_converted']}"
    )
    print(f"[build_corpus] output_dir={output_dir}")


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="Convert a directory of documents into Markdown under ./corpus (no chunking or indexing)"
    )
    parser.add_argument(
        "--input-dir",
        default="data/train_data",
        help="input directory (default: data/train_data)",
    )
    parser.add_argument(
        "--output-dir",
        default="corpus",
        help="output directory (default: corpus)",
    )
    parser.add_argument(
        "--skip-dirs",
        default="corpus,__pycache__,index,excel_mds",
        help="directory names to skip (comma-separated; default: corpus,__pycache__,index,excel_mds)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    skip = {s.strip() for s in (args.skip_dirs or "").split(",") if s.strip()}
    build_corpus(args.input_dir, args.output_dir, skip_dir_names=skip)
