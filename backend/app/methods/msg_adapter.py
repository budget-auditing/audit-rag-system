"""MSG-RAG adapter (in-process, backend built-in)."""
from typing import Any, List

from app.msg.graph import msg_graph


def _find_chunk_id(text: str) -> str:
    """Best-effort lookup of chunk_id by evidence text (structure fallback)."""
    try:
        from app.db import SessionLocal
        from app.models import StandardChunk
        target = (text or "").strip()
        if not target:
            return ""
        with SessionLocal() as db:
            for row in db.query(StandardChunk).all():
                if target in (row.text or "") or (row.text or "") in target:
                    return row.chunk_id
    except Exception:
        pass
    return ""


class MSGAdapter:
    route = "MSG_RAG"

    def answer(self, query: str, doc_scope: List[str] | None = None) -> dict:
        result = msg_graph.query(query, top_k=5, doc_scope=doc_scope)
        facts = result.get("facts", [])
        structures = result.get("structures", [])

        evidence = []
        for f in facts[:5]:
            span = f.get("support_span", "")
            evidence.append({
                "source_doc": "",
                "page": _page_from_json(f.get("source_location_json")),
                "span": span,
                "text": span,
                "path": " > ".join(_json_path(f.get("path_json"))),
                "value": f.get("value"),
                "unit": f.get("unit"),
                "chunk_id": f.get("chunk_id") or _find_chunk_id(span),
            })
        for s in structures[:5]:
            local = s.get("local_text", "")
            evidence.append({
                "source_doc": "",
                "page": _page_from_json(s.get("source_location_json")),
                "span": local,
                "text": local,
                "path": " > ".join(_json_path(s.get("path_json"))),
                "chunk_id": s.get("chunk_id") or _find_chunk_id(local),
            })

        return {
            "answer": {
                "summary": f"检索到 {len(facts)} 条事实节点，{len(structures)} 个结构节点。",
                "generated_answer": result.get("generated_answer", ""),
                "facts": facts,
                "structures": structures,
            },
            "evidence": evidence,
            "warnings": result.get("warnings", []),
            "metadata": {
                "index_version": str(msg_graph.status().get("edges", "")),
            },
        }


def _json_path(raw: Any) -> List[str]:
    import json
    if not raw:
        return []
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
        return val if isinstance(val, list) else []
    except Exception:
        return []


def _page_from_json(raw: Any) -> int | None:
    import json
    if not raw:
        return None
    try:
        val = json.loads(raw) if isinstance(raw, str) else raw
        return val.get("page") if isinstance(val, dict) else None
    except Exception:
        return None


msg_adapter = MSGAdapter()
