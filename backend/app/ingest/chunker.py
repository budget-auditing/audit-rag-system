"""Chunker: split parsed document into standard_chunks."""
import json
import re
import hashlib
from typing import List, Optional


HEADING_RE = re.compile(r"^(第[一二三四五六七八九十\d]+[章节]|\d+[\.、])\s*(.+)$")


def make_chunk_id(doc_id: str, text: str) -> str:
    return hashlib.sha256(f"{doc_id}:{text[:120]}".encode()).hexdigest()[:16]


def chunk_text(text: str, doc_id: str, doc_name: str, page_hint: Optional[int] = None) -> List[dict]:
    """Split text into paragraph chunks, preserving headings as path fragments."""
    chunks = []
    current_path: List[str] = [doc_name]
    lines = text.splitlines()
    buffer: List[str] = []

    def flush():
        if not buffer:
            return
        body = "\n".join(buffer).strip()
        if len(body) < 5:
            buffer.clear()
            return
        cid = make_chunk_id(doc_id, body)
        chunks.append({
            "chunk_id": cid,
            "doc_id": doc_id,
            "text": body,
            "chunk_type": "paragraph",
            "path_json": json.dumps(current_path, ensure_ascii=False),
            "table_context_json": "null",
            "source_location_json": json.dumps({"page": page_hint or 1}, ensure_ascii=False),
            "meta_json": json.dumps({}, ensure_ascii=False),
        })
        buffer.clear()

    for line in lines:
        line = line.strip()
        if not line:
            flush()
            continue
        m = HEADING_RE.match(line)
        if m:
            flush()
            # Simple path management: replace last if same level else append
            current_path = [doc_name, line]
        buffer.append(line)

    flush()
    return chunks


def chunk_tables(tables: List[dict], doc_id: str, doc_name: str) -> List[dict]:
    """Turn extracted tables into chunks."""
    chunks = []
    for t in tables:
        rows = t.get("rows", [])
        if not rows:
            continue
        # first row as caption if present
        caption = t.get("sheet") or t.get("caption") or "表格"
        header = rows[0] if len(rows) > 0 else []
        body = "\n".join(["\t".join(map(str, row)) for row in rows])
        cid = make_chunk_id(doc_id, body)
        chunks.append({
            "chunk_id": cid,
            "doc_id": doc_id,
            "text": body,
            "chunk_type": "table",
            "path_json": json.dumps([doc_name, caption], ensure_ascii=False),
            "table_context_json": json.dumps({
                "header": header,
                "rows_count": len(rows),
            }, ensure_ascii=False),
            "source_location_json": json.dumps({"page": t.get("page", 1), "sheet": t.get("sheet")}, ensure_ascii=False),
            "meta_json": json.dumps({}, ensure_ascii=False),
        })
    return chunks


def build_standard_chunks(doc_id: str, doc_name: str, text: str, tables: List[dict]) -> List[dict]:
    chunks = chunk_text(text, doc_id, doc_name) + chunk_tables(tables, doc_id, doc_name)
    # chunk_id hashes only doc_id + text[:120], so chunks sharing the same leading
    # text (repeated table fragments, garbled rotated tables) collide and violate
    # the chunk_id primary key. Disambiguate collisions with a positional salt.
    seen: set = set()
    for i, c in enumerate(chunks):
        cid = c["chunk_id"]
        salt = i
        while cid in seen:
            cid = hashlib.sha256(f"{doc_id}:{c['text'][:120]}:{salt}".encode()).hexdigest()[:16]
            salt += 1
        c["chunk_id"] = cid
        seen.add(cid)
    return chunks
