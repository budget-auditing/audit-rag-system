"""In-memory job registry for long-running index builds.

Single-process uvicorn assumption: a module-level dict guarded by a lock is
enough. Jobs are not persisted across restarts (a restart means no build is
running anyway). Status lifecycle: queued -> building -> ready | failed.
"""
import threading
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}

QUEUED = "queued"
BUILDING = "building"
READY = "ready"
FAILED = "failed"

_ACTIVE = {QUEUED, BUILDING}


def create_job(target: str) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "target": target,            # all | msg | c3rag | trace
        "status": QUEUED,
        "phase": "",                 # human-readable current step
        "progress": {"current": 0, "total": 0},
        "message": "",
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "finished_at": None,
    }
    with _LOCK:
        _JOBS[job_id] = job
    return dict(job)


def update_job(job_id: str, **fields: Any) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        progress = fields.pop("progress", None)
        if progress is not None:
            job["progress"] = {
                "current": int(progress.get("current", 0)),
                "total": int(progress.get("total", 0)),
            }
        job.update(fields)


def set_progress(job_id: str, current: int, total: int, phase: Optional[str] = None) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job["progress"] = {"current": int(current), "total": int(total)}
        if phase is not None:
            job["phase"] = phase


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def latest_job() -> Optional[Dict[str, Any]]:
    with _LOCK:
        if not _JOBS:
            return None
        job = max(_JOBS.values(), key=lambda j: j["created_at"])
        return dict(job)


def list_jobs() -> List[Dict[str, Any]]:
    with _LOCK:
        return [dict(j) for j in sorted(_JOBS.values(), key=lambda j: j["created_at"], reverse=True)]


def active_job() -> Optional[Dict[str, Any]]:
    with _LOCK:
        for j in sorted(_JOBS.values(), key=lambda j: j["created_at"], reverse=True):
            if j["status"] in _ACTIVE:
                return dict(j)
    return None
