"""Helpers to persist method-specific evidence to dedicated tables."""
import json
import re
from typing import Optional

from sqlalchemy.orm import Session

from app.models import InstitutionalEvidence, RiskPassage, StandardChunk


def _year(text: str) -> str:
    m = re.search(r"20[0-9]{2}", text)
    return m.group(0) if m else ""


def _find_chunk(text: str, db: Session) -> Optional[StandardChunk]:
    target = (text or "").strip()
    if not target:
        return None
    rows = db.query(StandardChunk).limit(200).all()
    for row in rows:
        if target in (row.text or "") or (row.text or "") in target:
            return row
    return None


def persist_c3rag_evidence(query_id: str, evidence: list[dict], db: Session) -> None:
    for idx, item in enumerate(evidence):
        text = item.get("text") or item.get("span") or ""
        chunk = _find_chunk(text, db)
        source = item.get("source_doc") or ""
        page = item.get("page")
        source_loc = {"source": source, "page": page}
        ev = InstitutionalEvidence(
            evidence_id=f"{query_id}_ev_{idx}",
            chunk_id=chunk.chunk_id if chunk else "",
            text=text,
            year=_year(text),
            department="",
            metric="",
            span=text,
            meta_json=json.dumps({"score": item.get("score"), "route": "C3_RAG"}, ensure_ascii=False),
            source_location_json=json.dumps(source_loc, ensure_ascii=False),
        )
        db.add(ev)


def persist_trace_evidence(query_id: str, evidence: list[dict], db: Session) -> None:
    for idx, item in enumerate(evidence):
        text = item.get("text") or item.get("span") or ""
        chunk = _find_chunk(text, db)
        source = item.get("source_doc") or ""
        page = item.get("page")
        source_loc = {"source": source, "page": page, "role": item.get("role", "Observation")}
        rp = RiskPassage(
            passage_id=f"{query_id}_rp_{idx}",
            chunk_id=chunk.chunk_id if chunk else "",
            text=text,
            year=_year(text),
            statistical_basis="",
            risk_features_json=json.dumps({"role": item.get("role", "Observation"), "score": item.get("score")}, ensure_ascii=False),
            source_location_json=json.dumps(source_loc, ensure_ascii=False),
        )
        db.add(rp)
