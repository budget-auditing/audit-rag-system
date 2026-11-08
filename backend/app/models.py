"""SQLAlchemy models for 7 tables."""
from sqlalchemy import Column, Integer, Float, String, Text

from app.db import Base


class Document(Base):
    __tablename__ = "documents"
    doc_id = Column(String, primary_key=True)
    doc_name = Column(String)
    doc_type = Column(String)
    year = Column(String)
    department = Column(String)
    file_path = Column(String)
    upload_time = Column(String)
    parse_status = Column(String)
    parse_seconds = Column(Float)  # wall-clock parse+chunk time of the most recent run
    index_status = Column(Text)  # JSON: {"c3rag":"built|none","trace":"built|none","msg":"n/a"}


class StandardChunk(Base):
    __tablename__ = "standard_chunks"
    chunk_id = Column(String, primary_key=True)
    doc_id = Column(String)
    text = Column(Text)
    chunk_type = Column(String)  # type is a reserved keyword
    path_json = Column(Text)
    table_context_json = Column(Text)
    source_location_json = Column(Text)
    meta_json = Column(Text)


class StructureNode(Base):
    __tablename__ = "structure_nodes"
    node_id = Column(String, primary_key=True)
    doc_id = Column(String)
    title = Column(String)
    path_json = Column(Text)
    level = Column(Integer)
    parent_id = Column(String)
    scope_json = Column(Text)
    local_text = Column(Text)
    summary = Column(Text)
    source_location_json = Column(Text)


class FactNode(Base):
    __tablename__ = "fact_nodes"
    fact_id = Column(String, primary_key=True)
    doc_id = Column(String)
    chunk_id = Column(String)
    subject = Column(String)
    subject_desc = Column(Text)
    predicate = Column(String)
    predicate_desc = Column(Text)
    object = Column(String)
    object_desc = Column(Text)
    value = Column(String)
    unit = Column(String)
    support_span = Column(Text)
    anchored_section_id = Column(String)
    source_location_json = Column(Text)


class FactEdge(Base):
    __tablename__ = "fact_edges"
    edge_id = Column(String, primary_key=True)
    source_fact_id = Column(String)
    target_fact_id = Column(String)
    edge_type = Column(String)  # E_ff


class InstitutionalEvidence(Base):
    __tablename__ = "institutional_evidence"
    evidence_id = Column(String, primary_key=True)
    chunk_id = Column(String)
    text = Column(Text)
    year = Column(String)
    department = Column(String)
    metric = Column(String)
    span = Column(Text)
    meta_json = Column(Text)
    source_location_json = Column(Text)


class RiskPassage(Base):
    __tablename__ = "risk_passages"
    passage_id = Column(String, primary_key=True)
    chunk_id = Column(String)
    text = Column(Text)
    year = Column(String)
    statistical_basis = Column(String)
    risk_features_json = Column(Text)
    source_location_json = Column(Text)


class TaskLog(Base):
    """Persisted activity log for the dashboard 最近任务 list.

    Unlike the in-memory jobs registry (index builds only, lost on restart),
    this table records all three user-visible task types and survives restarts.
    """
    __tablename__ = "task_logs"
    task_id = Column(String, primary_key=True)
    task_type = Column(String)  # index_build | qa | export
    target = Column(Text)       # human label: index target / query text / export kind
    status = Column(String)     # building | ready | failed | done
    detail = Column(Text)       # phase / route / extra info
    created_at = Column(String)
    finished_at = Column(String)


class QueryLog(Base):
    __tablename__ = "query_logs"
    query_id = Column(String, primary_key=True)
    query = Column(Text)
    query_type = Column(String)
    route = Column(String)
    confidence = Column(Float)
    answer_json = Column(Text)
    evidence_json = Column(Text)
    warnings_json = Column(Text)
    time_cost = Column(Float)
    created_at = Column(String)
    selected_mode = Column(String)  # user-chosen mode (route value)
    rule_hint = Column(Text)  # JSON: rule-based suggestion snapshot
