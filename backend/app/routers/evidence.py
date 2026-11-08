"""Evidence router — query log retrieval & source traceback."""
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import QueryLog, StandardChunk
from app.schemas import EvidenceResponse, EvidenceSource

router = APIRouter(tags=["evidence"])


@router.get("/evidence/{query_id}", response_model=EvidenceResponse)
def get_evidence_by_query(query_id: str, db: Session = Depends(get_db)):
    """Retrieve stored evidence / answer envelope for a query_id."""
    log = db.query(QueryLog).filter(QueryLog.query_id == query_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Query log not found")
    return EvidenceResponse(
        query_id=log.query_id,
        query=log.query,
        route=log.route,
        confidence=log.confidence,
        answer=json.loads(log.answer_json) if log.answer_json else {},
        evidence=json.loads(log.evidence_json) if log.evidence_json else [],
        warnings=json.loads(log.warnings_json) if log.warnings_json else [],
        metadata={"time_cost_sec": log.time_cost},
    )


@router.get("/evidence/source/{evidence_id}", response_model=EvidenceSource)
def get_evidence_source(evidence_id: str, db: Session = Depends(get_db)):
    """Resolve evidence_id to its source chunk (V1: evidence_id == chunk_id)."""
    chunk = db.query(StandardChunk).filter(StandardChunk.chunk_id == evidence_id).first()
    if not chunk:
        raise HTTPException(status_code=404, detail="Source chunk not found")
    return EvidenceSource(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        source_location=json.loads(chunk.source_location_json) if chunk.source_location_json else None,
    )
