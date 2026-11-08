"""Export standard_chunks to shared data/chunks/ for C³-RAG and TRACE."""
import json
from pathlib import Path
from typing import Any, Dict
from sqlalchemy.orm import Session
from app.config import settings
from app.models import StandardChunk

EXPORT_DIR = Path(settings.data_dir) / "chunks"


def _safe_json_loads(raw: Any) -> Any:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _extract_year(text: str) -> str:
    import re
    m = re.search(r"20[0-9]{2}", text)
    return m.group(0) if m else ""


def _chunk_to_c3rag(chunk: StandardChunk) -> dict:
    loc = _safe_json_loads(chunk.source_location_json)
    return {
        "text": chunk.text or "",
        "source": loc.get("doc_name") or chunk.doc_id,
        "page": loc.get("page"),
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
    }


def _chunk_to_trace(chunk: StandardChunk) -> dict:
    meta = _safe_json_loads(chunk.meta_json)
    path = _safe_json_loads(chunk.path_json) or []
    section = " > ".join(path[1:]) if len(path) > 1 else (path[0] if path else "")
    return {
        "evidence_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "pid": chunk.chunk_id,
        "year": meta.get("year") or _extract_year(chunk.text or "") or "",
        "budget_caliber": meta.get("caliber") or meta.get("budget_caliber") or "",
        "budget_level": meta.get("level") or meta.get("budget_level") or "",
        "section": section,
        "text": chunk.text or "",
        "source_path": chunk.chunk_id,
    }


def _msg_doc(chunk: StandardChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "text": chunk.text or "",
        "path": _safe_json_loads(chunk.path_json),
        "type": chunk.chunk_type,
    }


def export_chunks(db: Session) -> Dict[str, str]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = db.query(StandardChunk).all()
    c3rag_path = EXPORT_DIR / "c3rag.jsonl"
    trace_path = EXPORT_DIR / "trace.paragraphs.jsonl"
    msg_path = EXPORT_DIR / "msg.jsonl"
    with open(c3rag_path, "w", encoding="utf-8") as cf, \
         open(trace_path, "w", encoding="utf-8") as tf, \
         open(msg_path, "w", encoding="utf-8") as mf:
        for chunk in chunks:
            cf.write(json.dumps(_chunk_to_c3rag(chunk), ensure_ascii=False) + "\n")
            tf.write(json.dumps(_chunk_to_trace(chunk), ensure_ascii=False) + "\n")
            mf.write(json.dumps(_msg_doc(chunk), ensure_ascii=False) + "\n")
    return {
        "c3rag": str(c3rag_path),
        "trace": str(trace_path),
        "msg": str(msg_path),
        "count": len(chunks),
    }
