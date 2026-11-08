"""Dashboard overview and system status endpoints."""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app import jobs
from app.config import settings
from app.db import get_db
from app.msg.graph import msg_graph
from app.models import (
    Document,
    QueryLog,
    TaskLog,
    StructureNode,
    FactNode,
    FactEdge,
    InstitutionalEvidence,
    RiskPassage,
)
from app.tasklog import TASK_TYPE_CN

router = APIRouter(tags=["stats"])


def _probe(base_url: str, path: str = "/health", timeout: float = 3.0) -> bool:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/{path.lstrip('/')}", timeout=timeout)
        return resp.status_code < 500
    except Exception:
        return False


@router.get("/stats/overview", summary="Dashboard overview")
def stats_overview(db: Session = Depends(get_db)):
    total_docs = db.query(Document).count()
    parsed_docs = db.query(Document).filter(Document.parse_status == "parsed").count()
    total_queries = db.query(QueryLog).count()

    msg = msg_graph.status()
    structure_nodes = msg.get("structure_count") or db.query(StructureNode).count()
    fact_nodes = msg.get("count") or db.query(FactNode).count()
    edges = msg.get("edges") or db.query(FactEdge).count()

    # 最近任务: persisted across restarts, covering 索引构建/问答执行/报告导出.
    task_rows = (
        db.query(TaskLog).order_by(TaskLog.created_at.desc()).limit(10).all()
    )
    recent_tasks = [
        {
            "task_id": t.task_id,
            "task_type": t.task_type,
            "task_type_cn": TASK_TYPE_CN.get(t.task_type, t.task_type),
            "target": t.target,
            "status": t.status,
            "detail": t.detail,
            "created_at": t.created_at,
            "finished_at": t.finished_at,
        }
        for t in task_rows
    ]
    return {
        "documents": {
            "total": total_docs,
            "parsed": parsed_docs,
        },
        "indexes": {
            "msg": {
                "built": bool(fact_nodes),
                "structure_nodes": structure_nodes,
                "fact_nodes": fact_nodes,
                "edges": edges,
            },
            "institutional_evidence": db.query(InstitutionalEvidence).count(),
            "risk_passages": db.query(RiskPassage).count(),
        },
        "queries": {"total": total_queries},
        "recent_tasks": recent_tasks,
        "recent_jobs": jobs.list_jobs()[:10],  # live build progress (in-memory)
    }


@router.get("/system/status", summary="System status page")
def system_status(db: Session = Depends(get_db)):
    # database probe
    try:
        db.query(Document).limit(1).all()
        database = "connected"
    except Exception:
        database = "disconnected"

    structure_nodes = db.query(StructureNode).count()
    fact_nodes = db.query(FactNode).count()
    edges = db.query(FactEdge).count()
    graph_loaded = (structure_nodes + fact_nodes) > 0

    c3rag_up = _probe(settings.c3rag_service_url)
    trace_up = _probe(settings.trace_service_url)
    chat_up = _probe(settings.chat_url)
    embed_up = _probe(settings.embed_url)
    ces_ckpt = Path(settings.data_dir).resolve().parent / "C3RAG" / "models" / "cces_model.pt"
    ces_loaded = c3rag_up or os.path.exists(ces_ckpt)

    components: Dict[str, Any] = {
        "database": {"label": "关系数据库 (SQLite)", "status": database},
        "graph_store": {
            "label": "事实图谱 (SQLite + 内存 NetworkX)",
            "status": "loaded" if graph_loaded else "not_loaded",
            "nodes": structure_nodes + fact_nodes,
            "edges": edges,
        },
        "faiss": {
            "label": "向量索引 (FAISS)",
            "status": "ready" if (c3rag_up or trace_up or graph_loaded) else "not_ready",
        },
        "local_model": {
            "label": "对话模型",
            "status": "connected" if chat_up else "disconnected",
            "name": settings.chat_model,
        },
        "embedding_model": {
            "label": "嵌入模型",
            "status": "connected" if embed_up else "disconnected",
            "name": settings.embed_model,
        },
        "ces_model": {"label": "CES 一致性模型", "status": "loaded" if ces_loaded else "not_loaded"},
    }
    return {"components": components, "checked_at": datetime.now().isoformat()}