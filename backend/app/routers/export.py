"""Export router — query report & audit draft generation."""
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import tasklog
from app.db import get_db
from app.models import QueryLog
from app.schemas import (
    QueryReportRequest,
    QueryReportResponse,
    AuditDraftRequest,
    AuditDraftResponse,
    EvidenceListRequest,
    EvidenceListResponse,
)

router = APIRouter(tags=["export"])


def _render_answer(answer: dict, route: str) -> str:
    """Render answer as human-readable Markdown instead of JSON dump."""
    lines: list[str] = []

    if route == "MSG_RAG":
        summary = answer.get("summary", "")
        generated = answer.get("generated_answer", "")
        if summary:
            lines.append(summary)
        if generated and generated != summary:
            lines.append(generated)
        if answer.get("facts"):
            lines.append("\n### 事实节点")
            for f in answer["facts"]:
                subj = f.get("subject", "")
                pred = f.get("predicate", "")
                obj = f.get("object", "")
                val = f.get("value", "")
                unit = f.get("unit", "")
                line = f"- **  {subj} → {pred} → {obj}: {val} {unit}".strip()
                lines.append(line)

    elif route == "C3_RAG":
        if answer.get("generated_answer"):
            lines.append(answer["generated_answer"])
        if answer.get("consistency_result"):
            cc = answer["consistency_result"]
            lines.append(
                f"\n- 口径一致: {cc.get('caliber_consistent', False)} | 因果完整: {cc.get('causal_complete', False)}"
            )
            labels = cc.get("conflict_labels", [])
            if labels:
                lines.append("- 冲突标签: " + ", ".join(labels))
        if answer.get("conflicts"):
            lines.append("\n### 冲突详情")
            for c in answer["conflicts"]:
                lines.append(f"- **{c.get('type', '')}**: {c.get('message', '')}")

    elif route == "TRACE" or route == "HIDDEN_RISK":
        if answer.get("observation"):
            lines.append("**Observation**\n" + answer["observation"])
        if answer.get("cause"):
            lines.append("**Cause**\n" + answer["cause"])
        if answer.get("impact"):
            lines.append("**Impact**\n" + answer["impact"])
        if answer.get("abstain"):
            lines.append("\n> 系统弃权：证据不足。")

    return "\n".join(lines) if lines else "（无答案内容）"


def _render_evidence(idx: int, ev: dict) -> str:
    """Render a single evidence item as Markdown."""
    md = f"\n### 证据 {idx}\n"
    if ev.get("role"):
        md += f"- **角色**: {ev['role']}\n"
    if ev.get("chunk_id"):
        md += f"- **chunk_id**: {ev['chunk_id']}\n"
    if ev.get("source_doc"):
        md += f"- **来源**: {ev['source_doc']}\n"
    if ev.get("page"):
        md += f"- **页码**: {ev['page']}\n"
    if ev.get("score") is not None:
        md += f"- **得分**: {ev['score']:.4f}\n"
    text = ev.get("text") or ev.get("span") or ""
    if text:
        md += f"\n> {text.replace(chr(10), ' ')}\n"
    return md


def _log_to_markdown(log: QueryLog) -> str:
    answer = json.loads(log.answer_json) if log.answer_json else {}
    evidence = json.loads(log.evidence_json) if log.evidence_json else []
    warnings = json.loads(log.warnings_json) if log.warnings_json else []

    md = f"""# Query Report — {log.query_id}

## Query
{log.query}

## Routing
- **Route**: {log.route}
- **Confidence**: {log.confidence}
- **Cost**: {log.time_cost}s

## Answer
{_render_answer(answer, log.route)}

## Evidence ({len(evidence)} items)
"""
    for idx, ev in enumerate(evidence, start=1):
        md += _render_evidence(idx, ev)

    if warnings:
        md += "\n## Warnings\n"
        for w in warnings:
            md += f"- {w}\n"
    else:
        md += "\n## Warnings\nNone\n"

    return md


@router.post("/export/query-report", response_model=QueryReportResponse)
def export_query_report(req: QueryReportRequest, db: Session = Depends(get_db)):
    """Export a single query log as Markdown."""
    log = db.query(QueryLog).filter(QueryLog.query_id == req.query_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Query log not found")
    content = _log_to_markdown(log)
    tasklog.record_task(db, "export", "查询报告", "done", detail=req.format or "markdown")
    return QueryReportResponse(
        content=content,
        format=req.format or "markdown",
    )


def _evidence_location(ev: dict) -> tuple[str, str, str]:
    """Extract (title_path, page/table, span) from an evidence item."""
    path = ev.get("path") or ev.get("title_path") or ""
    if isinstance(path, list):
        path = " / ".join(str(p) for p in path)
    loc = ev.get("source_location") or {}
    page = ""
    if isinstance(loc, dict):
        page = str(loc.get("page") or loc.get("table") or "")
    if not page:
        page = str(ev.get("page") or "")
    span = ev.get("span") or ev.get("text") or ""
    span = str(span).replace("\n", " ").strip()
    return str(path), page, span


@router.post("/export/evidence-list", response_model=EvidenceListResponse)
def export_evidence_list(req: EvidenceListRequest, db: Session = Depends(get_db)):
    """Export a flat evidence list (span + 标题路径 + 页/表位置)."""
    logs = db.query(QueryLog).filter(QueryLog.query_id.in_(req.query_ids)).all()
    found = {log.query_id for log in logs}
    missing = [qid for qid in req.query_ids if qid not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f"Query logs not found: {missing}")

    fmt = (req.format or "markdown").lower()
    rows: list[tuple] = []
    for log in logs:
        evidence = json.loads(log.evidence_json) if log.evidence_json else []
        for ev in evidence:
            path, page, span = _evidence_location(ev)
            rows.append((log.query_id, ev.get("chunk_id", ""), path, page, span))

    if fmt == "csv":
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["query_id", "chunk_id", "标题路径", "页/表位置", "证据片段"])
        writer.writerows(rows)
        tasklog.record_task(db, "export", "证据清单", "done", detail=f"csv · {len(rows)}条")
        return EvidenceListResponse(content=buf.getvalue(), format="csv")

    md = ["# 证据清单\n", "| 查询ID | chunk_id | 标题路径 | 页/表位置 | 证据片段 |", "|---|---|---|---|---|"]
    for qid, cid, path, page, span in rows:
        safe = span.replace("|", "\\|")
        md.append(f"| {qid} | {cid} | {path} | {page} | {safe} |")
    tasklog.record_task(db, "export", "证据清单", "done", detail=f"markdown · {len(rows)}条")
    return EvidenceListResponse(content="\n".join(md), format="markdown")


@router.post("/export/audit-draft", response_model=AuditDraftResponse)
def export_audit_draft(req: AuditDraftRequest, db: Session = Depends(get_db)):
    """Export an audit draft aggregating multiple queries."""
    logs = db.query(QueryLog).filter(QueryLog.query_id.in_(req.query_ids)).all()
    found_ids = {log.query_id for log in logs}
    missing = [qid for qid in req.query_ids if qid not in found_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"Query logs not found: {missing}")

    md_parts = ["# Audit Draft\n"]
    for log in logs:
        answer = json.loads(log.answer_json) if log.answer_json else {}
        evidence = json.loads(log.evidence_json) if log.evidence_json else []
        md_parts.append(
            f"## {log.query_id}\n"
            f"- **Query**: {log.query}\n"
            f"- **Route**: {log.route}\n"
            f"- **Summary**: {answer.get('summary', '')}\n"
            f"- **Answer**: {_render_answer(answer, log.route)}\n"
            f"- **Evidence count**: {len(evidence)}\n\n"
        )

    tasklog.record_task(db, "export", "审计底稿", "done", detail=f"{len(logs)}条查询")
    return AuditDraftResponse(
        content="\n".join(md_parts),
        format=req.format or "markdown",
    )
