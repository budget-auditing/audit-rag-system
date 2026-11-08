"""Helpers for the persisted TaskLog activity feed (dashboard 最近任务).

Records the three user-visible task types (index_build / qa / export) to the
``task_logs`` table so the dashboard survives restarts, unlike the in-memory
``app.jobs`` registry which only tracks live index-build progress.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.models import TaskLog

TASK_TYPE_CN = {
    "index_build": "索引构建",
    "qa": "问答执行",
    "export": "报告导出",
}


def record_task(
    db: Session,
    task_type: str,
    target: str,
    status: str,
    detail: str = "",
    finished: bool = True,
) -> str:
    """Insert a task row. Returns the new task_id.

    For one-shot tasks (qa/export) leave ``finished=True``; for long-running
    builds pass ``finished=False`` then call :func:`update_task` when done.
    """
    now = datetime.now().isoformat()
    task_id = uuid.uuid4().hex[:12]
    row = TaskLog(
        task_id=task_id,
        task_type=task_type,
        target=(target or "")[:200],
        status=status,
        detail=(detail or "")[:200],
        created_at=now,
        finished_at=now if finished else None,
    )
    db.add(row)
    db.commit()
    return task_id


def update_task(db: Session, task_id: str, status: str, detail: Optional[str] = None) -> None:
    row = db.query(TaskLog).filter(TaskLog.task_id == task_id).first()
    if row is None:
        return
    row.status = status
    if detail is not None:
        row.detail = detail[:200]
    row.finished_at = datetime.now().isoformat()
    db.commit()
