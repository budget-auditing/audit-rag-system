"""Pydantic schemas — unified envelope."""
from typing import List, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared /util
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str


# ---------------------------------------------------------------------------
# Unified envelope (all three routes return this outer shape)
# ---------------------------------------------------------------------------

class UnifiedAnswer(BaseModel):
    query_id: str
    query: str
    query_type: str
    route: str
    confidence: float
    answer: dict
    evidence: List[dict]
    warnings: List[str]
    metadata: dict


class QueryRequest(BaseModel):
    query: str
    doc_scope: Optional[List[str]] = None
    force_route: Optional[str] = None
    # User-led mode selection. Route value: MSG_RAG | C3_RAG | TRACE.
    selected_mode: Optional[str] = None


class ModeHintRequest(BaseModel):
    query: str


class ModeHintResponse(BaseModel):
    suggested_mode: str  # route value: MSG_RAG | C3_RAG | TRACE
    scores: dict  # {信息查询, 推断核验, 风险识别}
    reason: str


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class EvidenceSource(BaseModel):
    chunk_id: str
    doc_id: Optional[str] = None
    text: Optional[str] = None
    source_location: Optional[dict] = None


class EvidenceResponse(BaseModel):
    query_id: str
    query: str
    route: str
    confidence: float
    answer: dict
    evidence: List[dict]
    warnings: List[str]
    metadata: dict


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

class QueryReportRequest(BaseModel):
    query_id: str
    format: Optional[str] = "markdown"


class QueryReportResponse(BaseModel):
    content: str
    format: str


class AuditDraftRequest(BaseModel):
    query_ids: List[str]
    format: Optional[str] = "markdown"


class AuditDraftItem(BaseModel):
    query_id: str
    query: str
    route: str
    summary: str
    evidence_count: int


class AuditDraftResponse(BaseModel):
    content: str
    format: str


class EvidenceListRequest(BaseModel):
    query_ids: List[str]
    format: Optional[str] = "markdown"  # markdown | csv


class EvidenceListResponse(BaseModel):
    content: str
    format: str


# ---------------------------------------------------------------------------
# Document / Index placeholders
# ---------------------------------------------------------------------------

class DocumentUpload(BaseModel):
    doc_name: str


class DocumentOut(BaseModel):
    doc_id: str
    doc_name: str
    doc_type: Optional[str] = None
    year: Optional[str] = None
    department: Optional[str] = None
    parse_status: Optional[str] = None
    index_status: Optional[str] = None


class IndexStatus(BaseModel):
    msg: dict
    c3rag: dict
    trace: dict
    corpus: Optional[dict] = None
    last_update: Optional[str] = None
