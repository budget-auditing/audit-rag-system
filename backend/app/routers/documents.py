"""Documents router — upload, list, parse, delete."""
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Document as DocumentModel, StandardChunk
from app.schemas import DocumentOut
from app.ingest.parser import parse_file
from app.ingest.chunker import build_standard_chunks
from app.ingest.export import export_chunks

router = APIRouter(tags=["documents"])

UPLOAD_DIR = Path(__file__).resolve().parents[2] / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _guess_doc_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    mapping = {
        ".pdf": "pdf",
        ".docx": "word",
        ".xlsx": "excel",
        ".xls": "excel",
        ".txt": "text",
        ".md": "text",
        ".csv": "text",
    }
    return mapping.get(ext, "unknown")


@router.post("/documents/upload")
def upload_document(
    file: UploadFile = File(...),
    year: str = Form(""),
    department: str = Form(""),
    db: Session = Depends(get_db),
):
    doc_id = str(uuid.uuid4())
    filename = file.filename or "unnamed"
    save_path = UPLOAD_DIR / f"{doc_id}_{filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    doc = DocumentModel(
        doc_id=doc_id,
        doc_name=filename,
        doc_type=_guess_doc_type(filename),
        year=year or "",
        department=department or "",
        file_path=str(save_path),
        upload_time=datetime.now().isoformat(),
        parse_status="uploaded",
        index_status='{"c3rag":"none","trace":"none","msg":"none"}',
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"message": "uploaded", "doc_id": doc_id, "filename": filename}


@router.get("/documents", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    rows = db.query(DocumentModel).order_by(DocumentModel.upload_time.desc()).all()
    result = []
    for r in rows:
        result.append(DocumentOut(
            doc_id=r.doc_id,
            doc_name=r.doc_name,
            doc_type=r.doc_type,
            year=r.year,
            department=r.department,
            parse_status=r.parse_status,
            index_status=r.index_status,
        ))
    return result


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc.doc_id,
        "doc_name": doc.doc_name,
        "doc_type": doc.doc_type,
        "parse_status": doc.parse_status,
        "index_status": doc.index_status,
        "upload_time": doc.upload_time,
    }


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    # delete file
    if doc.file_path and os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    # delete chunks
    db.query(StandardChunk).filter(StandardChunk.doc_id == doc_id).delete()
    db.delete(doc)
    db.commit()
    return {"deleted": doc_id}


@router.post("/documents/parse-all")
def parse_all_documents(db: Session = Depends(get_db)):
    """Parse every document that is not yet successfully parsed.

    Each document is committed individually so a single failure does not
    abort the rest; chunk export runs once at the end.
    """
    pending = [
        d for d in db.query(DocumentModel).all() if d.parse_status != "parsed"
    ]
    results = []
    parsed = 0
    for doc in pending:
        doc_id = doc.doc_id
        if not doc.file_path or not os.path.exists(doc.file_path):
            doc.parse_status = "error: file not found on disk"
            db.commit()
            results.append({"doc_id": doc_id, "status": "error", "error": "file not found on disk"})
            continue
        try:
            started = time.perf_counter()
            text, tables, meta = parse_file(doc.file_path)
            chunks = build_standard_chunks(doc_id, doc.doc_name, text, tables)
            db.query(StandardChunk).filter(StandardChunk.doc_id == doc_id).delete()
            for c in chunks:
                db.add(StandardChunk(**c))
            doc.parse_status = "parsed"
            doc.parse_seconds = round(time.perf_counter() - started, 3)
            db.commit()
            parsed += 1
            results.append({"doc_id": doc_id, "status": "parsed", "chunks": len(chunks)})
        except Exception as e:  # noqa: BLE001 — record per-doc failure, keep going
            db.rollback()
            doc = db.query(DocumentModel).filter(DocumentModel.doc_id == doc_id).first()
            if doc:
                doc.parse_status = f"error: {e}"
                db.commit()
            results.append({"doc_id": doc_id, "status": "error", "error": str(e)})

    if parsed:
        try:
            export_result = export_chunks(db)
        except Exception as exc:
            export_result = {"error": str(exc)}
    else:
        export_result = {"skipped": "no document parsed"}

    return {
        "total": len(pending),
        "parsed": parsed,
        "failed": len(pending) - parsed,
        "results": results,
        "export": export_result,
    }


@router.post("/documents/{doc_id}/parse")
def parse_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.doc_id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    try:
        started = time.perf_counter()
        text, tables, meta = parse_file(doc.file_path)
        chunks = build_standard_chunks(doc_id, doc.doc_name, text, tables)
        # upsert chunks: delete old then insert new (simplistic)
        db.query(StandardChunk).filter(StandardChunk.doc_id == doc_id).delete()
        for c in chunks:
            db.add(StandardChunk(**c))
        doc.parse_status = "parsed"
        doc.parse_seconds = round(time.perf_counter() - started, 3)
        try:
            export_result = export_chunks(db)
        except Exception as exc:
            export_result = {"error": str(exc)}
        db.commit()
        return {
            "doc_id": doc_id,
            "status": "parsed",
            "chunks": len(chunks),
            "meta": meta,
            "export": export_result,
        }
    except Exception as e:
        # The failed parse leaves the session in a PendingRollbackError state;
        # roll back first so the error-status write itself can commit.
        db.rollback()
        doc = db.query(DocumentModel).filter(DocumentModel.doc_id == doc_id).first()
        if doc:
            doc.parse_status = f"error: {e}"
            db.commit()
        raise HTTPException(status_code=500, detail=str(e))
