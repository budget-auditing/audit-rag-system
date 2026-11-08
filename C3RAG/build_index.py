#!/usr/bin/env python3

import pathlib
import os
import re
from openpyxl import load_workbook, Workbook
from markitdown import MarkItDown
import tqdm
import json
from typing import List, Dict
import faiss
import numpy as np
import fitz
from docx import Document
import tiktoken
from openai import OpenAI

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 80
BATCH_SIZE = 64

GENERAL_TEXT_EXTS = frozenset(["txt", "md", "markdown", "docx", "pdf"])
EXCEL_EXTS = frozenset(["xls", "xlsx"])

def get_tokenizer(model_name: str = os.environ.get("CHAT_MODEL", "gpt-4o-mini")):

    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")

enc = get_tokenizer()

def tokenize(s: str) -> List[int]:
    return enc.encode(s, disallowed_special=())

def detokenize(tokens: List[int]) -> str:
    return enc.decode(tokens)

def chunk_by_tokens(
    text: str, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP
) -> List[str]:
    toks = tokenize(text)
    n = len(toks)
    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_tokens, n)
        chunk = detokenize(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap, start + 1)
    return chunks

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

    out = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc):
            txt = page.get_text("text") or ""
            if txt.strip():
                out.append({"text": txt, "source": path, "page": i + 1})
    return out

def load_docs_from_folder(folder: str) -> List[Dict]:
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
                print(f"[WARN] Failed to read: {path}: {e}")
    return docs

def embed_texts(texts: List[str], client: OpenAI) -> np.ndarray:

    vecs = []
    for i in tqdm.tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)

        vecs.extend([d.embedding for d in resp.data])
    arr = np.array(vecs, dtype="float32")

    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr

class RAGStore:
    def __init__(self, client: OpenAI):
        self.index = None
        self.meta: List[Dict] = []
        self.dim = None
        self.client = client

    def build(self, docs: List[Dict], chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP):
        texts, metas = [], []
        for d in tqdm.tqdm(docs, desc="Building index"):
            chunks = chunk_by_tokens(d["text"], chunk_tokens, overlap)
            for ch in chunks:
                if ch.strip():
                    texts.append(ch)
                    metas.append({
                        "source": d.get("source"),
                        "page": d.get("page"),
                        "doc_id": d.get("doc_id"),
                        "chunk_id": d.get("chunk_id"),
                    })
        if not texts:
            raise ValueError("No indexable text found")
        vectors = embed_texts(texts, self.client)
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)
        self.meta = [{"text": t, **m} for t, m in zip(texts, metas)]
        print(f"Index built: {len(self.meta)} chunks, dimension {self.dim}")

    def save(self, dirpath: str):
        os.makedirs(dirpath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dirpath, "index.faiss"))
        with open(os.path.join(dirpath, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)
        print(f"Index saved to {dirpath}")

    def load(self, dirpath: str):
        self.index = faiss.read_index(os.path.join(dirpath, "index.faiss"))
        with open(os.path.join(dirpath, "meta.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.dim = self.index.d
        print(f"Index loaded: {len(self.meta)} chunks, dimension {self.dim}")

    def search(self, query: str, k: int = 6) -> List[Dict]:
        qvec = embed_texts([query], self.client)
        scores, idxs = self.index.search(qvec, k)
        results = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            m = self.meta[int(i)]
            results.append({"score": float(score), **m})
        return results

def convert_excel_to_md(path_in: str, path_out):
    md = MarkItDown(enable_plugins=False)
    res = md.convert(path_in)
    with open(path_out, "w", encoding="utf-8") as f:
        f.write(res.text_content)

def preprocess_xlsx(path_in: str, dir_out: str):
    file_name = pathlib.Path(path_in).name
    wb = None
    try:
        wb = load_workbook(path_in)
    except Exception as e:
        print(f"[WARN] Failed to read: {path_in}: {e}")
        return
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
    file_name = pathlib.Path(path_in).name
    convert_excel_to_md(path_in, os.path.join(dir_out, file_name + ".md"))

def preprocess_all_excels(input_dir: str, output_dir: str):
    excel_path_list = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file.lower().split(".")[-1] in EXCEL_EXTS:
                excel_path_list.append(file_path)
    for file_path in tqdm.tqdm(excel_path_list, desc="Preprocessing Excels"):
        if file_path.lower().split(".")[-1] == "xlsx":
            preprocess_xlsx(file_path, output_dir)
        elif file_path.lower().split(".")[-1] == "xls":
            preprocess_xls(file_path, output_dir)

if __name__ == "__main__":
    data_dir = "data/train_data/files"
    excel_mds_dir = "data/train_data/excel_mds"
    index_dir = "data/index"

    preprocess_all_excels(data_dir, excel_mds_dir)

    client = OpenAI(
        base_url=os.environ.get("EMBED_URL", "http://localhost:8081/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    )

    folder = "data/train_data"
    docs = load_docs_from_folder(folder)
    store = RAGStore(client)
    store.build(docs, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP)
    store.save(index_dir)
