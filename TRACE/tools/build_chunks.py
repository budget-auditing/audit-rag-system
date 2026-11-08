#! /usr/bin/env python3
"""Chunk documents into token-bounded windows and persist them as JSON.

This module does not build a vector index; it only produces the chunk file
that later stages embed.
"""

import os
import re
import json
import pathlib
from typing import List, Dict

import tiktoken
import tqdm
import fitz  # PyMuPDF
from docx import Document
from openpyxl import load_workbook, Workbook
from markitdown import MarkItDown


# ============ Configuration ============
CHUNK_TOKENS = 400  # ~400 tokens per chunk
CHUNK_OVERLAP = 80  # 80-token overlap between adjacent chunks

GENERAL_TEXT_EXTS = frozenset(["txt", "md", "markdown", "docx", "pdf"])
EXCEL_EXTS = frozenset(["xls", "xlsx"])


# ============ Tokenizer ============
def get_tokenizer(model_name: str = "gpt-4o-mini"):
    """Return a tiktoken tokenizer for chunk sizing."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


enc = get_tokenizer()


def tokenize(s: str) -> List[int]:
    return enc.encode(s, disallowed_special=())


def detokenize(tokens: List[int]) -> str:
    return enc.decode(tokens)


# ============ Chunking ============
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences or lines.

    Prefers line splitting (which suits CSV / table rows); when a single line
    is too long it falls back to sentence splitting.
    """
    lines = text.split("\n")
    sentences = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # If the line is short (likely a CSV row or table row), keep it as one unit.
        line_tokens = len(tokenize(line))
        if line_tokens <= CHUNK_TOKENS:
            sentences.append(line)
        else:
            # Over-long line: split by sentence punctuation (both Chinese and English)
            # while preserving delimiters via regex look-behind.
            parts = re.split(r'(?<=[。！？.!?])\s*', line)
            for part in parts:
                part = part.strip()
                if part:
                    sentences.append(part)
    
    return sentences


def chunk_by_sentences(
    text: str, target_tokens: int = CHUNK_TOKENS, overlap_sentences: int = 2
) -> List[str]:
    """Build chunks from whole sentences / lines.

    Args:
        text: Input text.
        target_tokens: Soft token budget. Chunks may exceed it to preserve sentence
            integrity.
        overlap_sentences: Number of sentences to overlap between adjacent chunks.

    Returns:
        The list of chunk strings.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sent in sentences:
        sent_tokens = len(tokenize(sent))
        
        # Accept the sentence if the current chunk is empty or adding it does not
        # exceed the soft budget. We allow overshoot for sentence integrity.
        if not current_chunk or current_tokens + sent_tokens <= target_tokens * 1.5:
            current_chunk.append(sent)
            current_tokens += sent_tokens
        else:
            # Current chunk is full: flush it and start a new one.
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            
            # Seed the new chunk with the last `overlap_sentences` sentences.
            if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                overlap_part = current_chunk[-overlap_sentences:]
                current_chunk = overlap_part + [sent]
                current_tokens = sum(len(tokenize(s)) for s in current_chunk)
            else:
                current_chunk = [sent]
                current_tokens = sent_tokens
    
    # Flush any trailing partial chunk.
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks


def chunk_by_tokens(
    text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Chunk by whole sentences / lines (alias preserved for callers).

    Each chunk contains full sentences / rows and may exceed ``chunk_tokens``.
    """
    # Convert token overlap to sentence overlap (~20-30 tokens per sentence).
    overlap_sentences = max(1, overlap // 30)
    return chunk_by_sentences(text, target_tokens=chunk_tokens, overlap_sentences=overlap_sentences)


# ============ Document loaders ============
def load_txt(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"text": text, "source": path, "page": None}]


def load_md(path: str) -> List[Dict]:
    return load_txt(path)


def load_docx(path: str) -> List[Dict]:
    doc = Document(path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return [{"text": text, "source": path, "page": None}]


def load_pdf(path: str) -> List[Dict]:
    """Load a PDF page-by-page (safe for large files)."""
    out = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            txt = page.get_text("text") or ""
            if txt.strip():
                out.append({"text": txt, "source": path, "page": i + 1})
    return out


def load_docs_from_folder(folder: str) -> List[Dict]:
    """Recursively load every supported document under ``folder``."""
    docs = []
    for root, _, files in os.walk(folder):
        for fn in files:
            path = os.path.join(root, fn)
            low = fn.lower()
            try:
                if low.endswith(".pdf"):
                    docs += load_pdf(path)
                elif low.endswith(".docx"):
                    docs += load_docx(path)
                elif low.endswith(".txt"):
                    docs += load_txt(path)
                elif low.endswith(".md"):
                    docs += load_md(path)
            except Exception as e:
                print(f"[WARN] read failed: {path}: {e}")
    return docs


# ============ Excel preprocessing ============
def convert_excel_to_md(path_in: str, path_out: str):
    """Convert a single Excel file to Markdown."""
    md = MarkItDown(enable_plugins=False)
    res = md.convert(path_in)
    with open(path_out, "w", encoding="utf-8") as f:
        f.write(res.text_content)


def preprocess_xlsx(path_in: str, dir_out: str):
    """Preprocess a ``.xlsx`` file: explode merged cells, emit one Markdown per sheet."""
    file_name = pathlib.Path(path_in).name
    wb = None
    try:
        wb = load_workbook(path_in)
    except Exception as e:
        print(f"[WARN] load_workbook failed: {path_in}: {e}")
        return

    # Explode merged cells.
    for ws in wb.worksheets:
        for rng in list(ws.merged_cells.ranges):
            ws.unmerge_cells(str(rng))
            tl_cell = ws[rng.min_row][rng.min_col - 1]
            for r in range(rng.min_row, rng.max_row + 1):
                for c in range(rng.min_col, rng.max_col + 1):
                    ws.cell(r, c).value = tl_cell.value

    sheet_id_for_invalid_name = 0
    for ws in wb.worksheets:
        new_wb = Workbook()
        new_ws = new_wb.active
        new_ws.title = ws.title
        for row in ws.iter_rows():
            for cell in row:
                new_ws.cell(row=cell.row, column=cell.col_idx, value=cell.value)
        safe_sheet_name = re.sub(r'[\\/:*?"<>|]', "_", ws.title.strip()) or "Sheet"
        if safe_sheet_name == "Sheet":
            safe_sheet_name = f"Sheet_{sheet_id_for_invalid_name}"
            sheet_id_for_invalid_name += 1
        out_name = file_name + "_" + safe_sheet_name + ".xlsx"
        new_wb.save(os.path.join(dir_out, out_name))
        convert_excel_to_md(
            os.path.join(dir_out, out_name), os.path.join(dir_out, out_name + ".md")
        )


def preprocess_xls(path_in: str, dir_out: str):
    """Preprocess a legacy ``.xls`` file by converting directly to Markdown."""
    file_name = pathlib.Path(path_in).name
    convert_excel_to_md(path_in, os.path.join(dir_out, file_name + ".md"))


def preprocess_all_excels(input_dir: str, output_dir: str):
    """Preprocess every Excel file under ``input_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    excel_path_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Skip Excel lock files.
            if file.startswith(".~") or file.startswith("~$"):
                continue
            file_path = os.path.join(root, file)
            if file.lower().split(".")[-1] in EXCEL_EXTS:
                excel_path_list.append(file_path)
    for file_path in tqdm.tqdm(excel_path_list, desc="Preprocessing Excels"):
        ext = file_path.lower().split(".")[-1]
        if ext == "xlsx":
            preprocess_xlsx(file_path, output_dir)
        elif ext == "xls":
            preprocess_xls(file_path, output_dir)


# ============ Chunking + persistence ============
def build_chunks(
    docs: List[Dict], chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """Chunk each document in ``docs`` and return the flattened chunk list."""
    chunks = []
    chunk_id = 0
    for d in tqdm.tqdm(docs, desc="Chunking documents"):
        text_chunks = chunk_by_tokens(d["text"], chunk_tokens, overlap)
        for ch in text_chunks:
            if ch.strip():
                chunks.append({
                    "id": chunk_id,
                    "text": ch,
                    "source": d.get("source"),
                    "page": d.get("page"),
                })
                chunk_id += 1
    return chunks


def save_chunks(chunks: List[Dict], output_path: str):
    """Persist chunks to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)
    print(f"[build_chunks] saved {len(chunks)} chunks to {output_path}")


# ============ Entrypoint ============
if __name__ == "__main__":
    # Configure paths (override by editing this script or wrapping it).
    data_dir = "data/train_data/EXCEL"
    excel_mds_dir = "data/train_data/excel_mds"
    output_json = "data/chunk.json"

    # Step 1: preprocess Excel files
    print("=" * 50)
    print("Step 1: preprocess Excel files")
    print("=" * 50)
    preprocess_all_excels(data_dir, excel_mds_dir)

    # Step 2: load documents
    print("\n" + "=" * 50)
    print("Step 2: load documents")
    print("=" * 50)
    folder = "data/train_data"
    docs = load_docs_from_folder(folder)
    print(f"loaded {len(docs)} document fragments")

    # Step 3: chunk
    print("\n" + "=" * 50)
    print("Step 3: chunk documents")
    print("=" * 50)
    chunks = build_chunks(docs, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP)

    # Step 4: save
    print("\n" + "=" * 50)
    print("Step 4: save chunks")
    print("=" * 50)
    save_chunks(chunks, output_json)

    print("\n[build_chunks] done.")
