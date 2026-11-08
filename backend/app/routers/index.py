"""Index build / status router (async background builds + dispatch to micro-services)."""
import json
import threading
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import jobs, tasklog
from app.config import settings
from app.db import get_db, SessionLocal
from app.models import Document, StandardChunk
from app.msg.graph import msg_graph
from app.ingest.export import export_chunks
from app.schemas import IndexStatus

router = APIRouter(tags=["index"])


def _get_chunks(db: Session):
    return db.query(StandardChunk).all()


def _set_document_index_status(db: Session, updates: dict):
    now = datetime.now().isoformat()
    for doc in db.query(Document).all():
        existing = {}
        if doc.index_status:
            try:
                existing = json.loads(doc.index_status)
            except Exception:
                existing = {}
        existing.update(updates)
        existing["last_update"] = now
        doc.index_status = json.dumps(existing, ensure_ascii=False)
    db.commit()


def _service_post(base_url: str, path: str, payload: dict, timeout: float = 3600.0) -> dict:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    resp = httpx.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _service_get(base_url: str, path: str, timeout: float = 10.0) -> dict:
    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _export_paths(db: Session) -> dict:
    try:
        return export_chunks(db)
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
# Background workers. Each opens its own DB session (the request session is    #
# already closed by the time the thread runs) and reports progress to the job. #
# --------------------------------------------------------------------------- #

def _build_msg(db: Session, job_id: str) -> dict:
    chunks = _get_chunks(db)

    def _progress(cur, total, phase):
        jobs.set_progress(job_id, cur, total, phase=f"MSG-{phase}")

    counts = msg_graph.build(chunks, progress=_progress)
    jobs.update_job(job_id, phase="MSG-持久化")
    msg_graph.persist(db)
    _set_document_index_status(db, {"msg": "built"})
    return counts


def _build_c3rag(db: Session, job_id: str) -> dict:
    jobs.update_job(job_id, phase="C³-RAG-嵌入+建索引")
    paths = _export_paths(db)
    res = _service_post(settings.c3rag_service_url, "/index/build", {"chunks_path": paths.get("c3rag")})
    _set_document_index_status(db, {"c3rag": "built"})
    return {"c3rag": res, "export": paths}


def _build_trace(db: Session, job_id: str) -> dict:
    jobs.update_job(job_id, phase="TRACE-嵌入+建索引")
    paths = _export_paths(db)
    res = _service_post(settings.trace_service_url, "/index/build", {"corpus_file": paths.get("trace")})
    _set_document_index_status(db, {"trace": "built"})
    return {"trace": res, "export": paths}


_WORKERS = {
    "msg": [_build_msg],
    "c3rag": [_build_c3rag],
    "trace": [_build_trace],
    "all": [_build_msg, _build_c3rag, _build_trace],
}


_TARGET_CN = {
    "all": "全部索引",
    "msg": "事实图谱(MSG-RAG)",
    "c3rag": "结论核验证据(C³-RAG)",
    "trace": "风险段落(TRACE)",
}


def _run_job(job_id: str, target: str):
    jobs.update_job(job_id, status=jobs.BUILDING, started_at=datetime.now().isoformat())
    db = SessionLocal()
    task_id = tasklog.record_task(
        db, "index_build", _TARGET_CN.get(target, target), "building", detail="构建中", finished=False
    )
    try:
        if not _get_chunks(db):
            jobs.update_job(
                job_id, status=jobs.FAILED, error="暂无已解析文档，请先上传并解析文档。",
                finished_at=datetime.now().isoformat(),
            )
            tasklog.update_task(db, task_id, "failed", detail="暂无已解析文档")
            return
        result = {}
        for worker in _WORKERS[target]:
            part = worker(db, job_id)
            result.update(part)
        jobs.update_job(
            job_id, status=jobs.READY, phase="完成", result=result,
            message="索引构建完成", finished_at=datetime.now().isoformat(),
        )
        tasklog.update_task(db, task_id, "ready", detail="构建完成")
    except Exception as exc:  # noqa: BLE001 — surface any failure to the job record
        jobs.update_job(
            job_id, status=jobs.FAILED, error=str(exc),
            finished_at=datetime.now().isoformat(),
        )
        tasklog.update_task(db, task_id, "failed", detail=str(exc))
    finally:
        db.close()


def _kick_off(target: str, db: Session) -> dict:
    if not _get_chunks(db):
        return {"built": False, "message": "暂无已解析文档，请先上传文档。", "job": None}
    active = jobs.active_job()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail={"message": "已有索引构建任务进行中，请等待其完成。", "job": active},
        )
    job = jobs.create_job(target)
    thread = threading.Thread(target=_run_job, args=(job["job_id"], target), daemon=True)
    thread.start()
    return {"built": False, "queued": True, "message": "索引构建已在后台启动", "job": job}


@router.post("/index/build/all", summary="Build all indexes (async)")
def build_all(db: Session = Depends(get_db)):
    return _kick_off("all", db)


@router.post("/index/build/fact-graph", summary="Build MSG-RAG fact-graph only (async)")
def build_msg(db: Session = Depends(get_db)):
    return _kick_off("msg", db)


@router.post("/index/build/institutional", summary="Build C³-RAG institutional index only (async)")
def build_c3rag(db: Session = Depends(get_db)):
    return _kick_off("c3rag", db)


@router.post("/index/build/risk", summary="Build TRACE risk index only (async)")
def build_trace(db: Session = Depends(get_db)):
    return _kick_off("trace", db)


@router.get("/index/build/status", summary="Poll build job progress")
def build_status(job_id: Optional[str] = Query(default=None)):
    job = jobs.get_job(job_id) if job_id else jobs.latest_job()
    if job is None:
        return {"job": None, "active": False}
    return {"job": job, "active": job["status"] in (jobs.QUEUED, jobs.BUILDING)}


@router.get("/index/status", response_model=IndexStatus, summary="Get index status")
def index_status(db: Session = Depends(get_db)):
    msg = msg_graph.status()
    try:
        c3rag = _service_get(settings.c3rag_service_url, "/index/status")
    except Exception as exc:
        c3rag = {"ready": False, "error": str(exc)}
    try:
        trace = _service_get(settings.trace_service_url, "/index/status")
    except Exception as exc:
        trace = {"ready": False, "error": str(exc)}

    # The C³-RAG/TRACE micro-services report their document count as ``size``;
    # expose it uniformly as ``count`` so consumers need not special-case.
    for svc in (c3rag, trace):
        if "count" not in svc and "size" in svc:
            svc["count"] = svc["size"]

    parse_seconds = db.query(func.coalesce(func.sum(Document.parse_seconds), 0.0)).scalar() or 0.0
    corpus = {
        "chunks_total": db.query(StandardChunk).count(),
        "tables_total": db.query(StandardChunk).filter(StandardChunk.chunk_type == "table").count(),
        "parse_seconds": round(float(parse_seconds), 1),
    }

    return IndexStatus(
        msg=msg,
        c3rag=c3rag,
        trace=trace,
        corpus=corpus,
        last_update=datetime.now().isoformat(),
    )
