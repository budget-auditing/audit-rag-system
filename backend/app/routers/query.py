"""Query router with adapter dispatch."""
import json
import re
import time
import uuid
from typing import Any, Dict, List, Tuple, Optional

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app import tasklog
from app.config import settings
from app.db import get_db
from app.methods import msg_adapter, c3rag_adapter, trace_adapter
from app.models import QueryLog
from app.schemas import QueryRequest, UnifiedAnswer, ModeHintRequest, ModeHintResponse

router = APIRouter(tags=["query"])

# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

# Route priority: hidden_risk > institutional_consistency > fact_numeric (fallback).
# Weights are chosen so that mixed queries carrying risk/institutional signals
# still route to the higher-priority specialised route.
_KEYWORDS = {
    "hidden_risk": {
        "core": [
            "风险",
            "可持续性",
            "隐患",
            "执行偏差",
            "虚报",
            "瞒报",
            "违规",
            "不合规",
            "隐性债务",
            "债务风险",
            "财政可持续",
            "偿债压力",
            "还本付息",
        ],
        "weight": 1.5,
    },
    "institutional_consistency": {
        "core": [
            "口径",
            "一致",
            "同比",
            "可比",
            "因果",
            "制度",
            "规范",
            "标准",
            "口径变更",
            "统计口径",
            "部门口径",
            "核算口径",
            "差异原因",
        ],
        "weight": 1.2,
    },
    "fact_numeric": {
        "core": [
            "多少",
            "金额",
            "预算",
            "决算",
            "支出",
            "收入",
            "比例",
            "数量",
            "总额",
            "合计",
            "增速",
            "比上年增长",
        ],
        "weight": 1.0,
    },
}
_PRIORITY = ["hidden_risk", "institutional_consistency", "fact_numeric"]


# qtype ↔ route value ↔ Chinese mode label
_QTYPE_TO_ROUTE = {
    "fact_numeric": "MSG_RAG",
    "institutional_consistency": "C3_RAG",
    "hidden_risk": "TRACE",
}
_ROUTE_TO_QTYPE = {v: k for k, v in _QTYPE_TO_ROUTE.items()}
_QTYPE_TO_MODE_CN = {
    "fact_numeric": "信息查询",
    "institutional_consistency": "结论核验",
    "hidden_risk": "风险识别",
}


def _rule_scores(query: str) -> Dict[str, float]:
    return {
        qtype: sum(cfg["weight"] for w in cfg["core"] if w in query)
        for qtype, cfg in _KEYWORDS.items()
    }


def _rule_classify(query: str) -> Tuple[str, float]:
    scores = _rule_scores(query)
    total = sum(scores.values())
    if total == 0:
        return "fact_numeric", 0.0
    # tie-breaker by route priority (specialised routes win over fallback)
    best = max(_PRIORITY, key=lambda t: (scores[t], -_PRIORITY.index(t)))
    confidence = round(scores[best] / total, 3)
    # boost confidence if it is clearly the dominant signal
    second = sorted(scores.values(), reverse=True)[1]
    if scores[best] >= second + 1.5:
        confidence = min(1.0, round(confidence + 0.1, 3))
    return best, confidence


def _llm_classify(query: str) -> Tuple[str, float]:
    try:
        from openai import OpenAI  # type: ignore[import-untyped]
    except Exception:
        raise RuntimeError("openai not installed")

    client = OpenAI(base_url=settings.chat_url, api_key=settings.openai_api_key)
    prompt = (
        "你是一位审计文档查询分类助手。请将以下用户查询分类为以下三类之一：\n"
        "- fact_numeric: 事实或数值查询，例如‘某项目预算金额是多少’、‘具体数量’等。\n"
        "- institutional_consistency: 制度一致性查询，例如‘口径是否一致’、‘因果关系’等。\n"
        "- hidden_risk: 隐藏风险查询，例如‘可持续性风险’、‘执行偏差’等。\n\n"
        "请只输出 JSON 格式，不要任何其他内容：\n"
        '{"type": "fact_numeric", "confidence": 0.85}\n\n'
        f"用户查询：{query}"
    )
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content or "{}"
    # extract json block
    match = re.search(r'\{[^}]+\}', raw)
    if not match:
        raise ValueError("no JSON obj in LLM output")
    body = json.loads(match.group())
    qtype = body.get("type", "fact_numeric")
    confidence = float(body.get("confidence", 0.5))
    return qtype, confidence


def classify_query(query_text: str) -> Tuple[str, float]:
    """Try LLM; fallback to rules."""
    try:
        qtype, confidence = _llm_classify(query_text)
        if confidence < 0.3:
            # low confidence from LLM → blend with rule score
            rtype, rconf = _rule_classify(query_text)
            if rconf > confidence:
                return rtype, round((confidence + rconf) / 2, 3)
        return qtype, confidence
    except Exception:
        return _rule_classify(query_text)


# ---------------------------------------------------------------------------
# CARC-style triplet decomposition (C³-RAG)
# ---------------------------------------------------------------------------

def _decompose_query(query: str) -> List[Dict[str, Optional[str]]]:
    """Decompose query into (time, department, metric) triplets.

    Falls back to a rule-based extractor when LLM is unavailable.
    """
    try:
        return _llm_decompose(query)
    except Exception:
        return _rule_decompose(query)


def _llm_decompose(query: str) -> List[Dict[str, Optional[str]]]:
    from openai import OpenAI  # type: ignore[import-untyped]

    client = OpenAI(base_url=settings.chat_url, api_key=settings.openai_api_key)
    prompt = (
        "你是一位审计查询分析助手。请将以下查询分解为一个或多个 (时间, 部门, 指标) 三元组，\n"
        "用于检索不同口径/年份的可比证据。如果字段未知，使用 null。\n\n"
        "查询：" + query + "\n\n"
        "只返回 JSON 列表，例如：\n"
        '[{"time":"2024","department":"","metric":"科学技术支出"}]\\n'
    )
    resp = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content or ""
    match = re.search(r'\[[^\]]*\]', raw)
    if not match:
        return _rule_decompose(query)
    try:
        data = json.loads(match.group())
        if isinstance(data, list) and data:
            return data
    except Exception:
        pass
    return _rule_decompose(query)


def _rule_decompose(query: str) -> List[Dict[str, Optional[str]]]:
    time_match = re.search(r"(20\d{2})\s*年?", query)
    time = time_match.group(1) if time_match else None
    metric = None
    for m in ["科学技术支出", "预算", "收入", "支出", "决算"]:
        if m in query:
            metric = m
            break
    return [{"time": time, "department": None, "metric": metric or query}]


def _triplet_to_query(triplet: Dict[str, Optional[str]]) -> str:
    parts = []
    if triplet.get("time"):
        parts.append(f"{triplet['time']}年")
    if triplet.get("department"):
        parts.append(triplet["department"])
    if triplet.get("metric"):
        parts.append(triplet["metric"])
    return " ".join(parts) if parts else ""


def _map_route(qtype: str) -> str:
    return {
        "fact_numeric": "MSG_RAG",
        "institutional_consistency": "C3_RAG",
        "hidden_risk": "TRACE",
    }.get(qtype, "MSG_RAG")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/qa/mode-hint", response_model=ModeHintResponse)
def mode_hint(req: ModeHintRequest):
    """Rule-based mode suggestion for user-led selection.

    Only advises; does NOT run retrieval. The user remains in control of the
    final mode via the frontend confirm buttons.
    """
    scores = _rule_scores(req.query)
    total = sum(scores.values())
    if total == 0:
        best = "fact_numeric"
        reason = "未检测到明显的制度一致性或风险信号，建议使用信息查询模式。"
    else:
        best = max(_PRIORITY, key=lambda t: (scores[t], -_PRIORITY.index(t)))
        hit = [w for w in _KEYWORDS[best]["core"] if w in req.query]
        reason = (
            f"命中{_QTYPE_TO_MODE_CN[best]}相关关键词：{'、'.join(hit[:5])}。"
            if hit
            else f"依据规则打分，建议使用{_QTYPE_TO_MODE_CN[best]}模式。"
        )
    cn_scores = {_QTYPE_TO_MODE_CN[q]: round(s, 2) for q, s in scores.items()}
    return ModeHintResponse(
        suggested_mode=_QTYPE_TO_ROUTE[best],
        scores=cn_scores,
        reason=reason,
    )


@router.post("/query/answer", response_model=UnifiedAnswer)
def answer_query(req: QueryRequest, db: Session = Depends(get_db)):
    start = time.time()
    query_id = uuid.uuid4().hex[:16]

    # User-led mode selection takes precedence: no auto-classification when the
    # user has explicitly chosen a mode. force_route is an alias of selected_mode.
    chosen = req.selected_mode or req.force_route
    rule_hint_snapshot: Dict[str, Any] = {}
    if chosen in _ROUTE_TO_QTYPE:
        qtype = _ROUTE_TO_QTYPE[chosen]
        confidence = 1.0
        # Record what the rule engine would have suggested, for the query log.
        rule_hint_snapshot = {
            "suggested_mode": _QTYPE_TO_ROUTE[_rule_classify(req.query)[0]],
            "scores": {
                _QTYPE_TO_MODE_CN[q]: round(s, 2)
                for q, s in _rule_scores(req.query).items()
            },
        }
    else:
        qtype, confidence = classify_query(req.query)

    route = _map_route(qtype)

    # ---------------------------------------------------------------
    # Dispatch via adapters (parallel methods, no pipeline chaining)
    # ---------------------------------------------------------------
    adapter = {
        "MSG_RAG": msg_adapter,
        "C3_RAG": c3rag_adapter,
        "TRACE": trace_adapter,
    }.get(route, msg_adapter)

    try:
        routed = adapter.answer(req.query, doc_scope=req.doc_scope)
    except Exception as exc:
        # fallback to built-in simple adapters for resilience
        if route == "MSG_RAG":
            from app.msg.graph import msg_graph

            result = msg_graph.query(req.query, top_k=5, doc_scope=req.doc_scope)
            routed = {
                "answer": {
                    "summary": f"检索到 {len(result['facts'])} 条事实节点，{len(result['structures'])} 个结构节点。",
                    "generated_answer": result.get("generated_answer", ""),
                    "facts": result["facts"],
                    "structures": result["structures"],
                },
                "evidence": result["facts"] + result["structures"],
                "warnings": result["warnings"] + [str(exc)],
            }
        else:
            routed = {
                "answer": {"summary": "服务暂时不可用，返回 abstain。"},
                "evidence": [],
                "warnings": [str(exc)],
            }

    answer_body = routed["answer"]
    evidence = routed["evidence"]
    warnings = routed["warnings"]
    adapter_meta = routed.get("metadata", {}) or {}

    cost = round(time.time() - start, 3)

    # persist route-specific evidence tables
    if route == "C3_RAG":
        from app.routers._persist import persist_c3rag_evidence
        persist_c3rag_evidence(query_id, evidence, db)
    elif route == "TRACE":
        from app.routers._persist import persist_trace_evidence
        persist_trace_evidence(query_id, evidence, db)

    # persist log
    log = QueryLog(
        query_id=query_id,
        query=req.query,
        query_type=qtype,
        route=route,
        confidence=round(confidence, 3),
        answer_json=json.dumps(answer_body, ensure_ascii=False),
        evidence_json=json.dumps(evidence, ensure_ascii=False),
        warnings_json=json.dumps(warnings, ensure_ascii=False),
        time_cost=cost,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        selected_mode=chosen if chosen in _ROUTE_TO_QTYPE else None,
        rule_hint=json.dumps(rule_hint_snapshot, ensure_ascii=False) if rule_hint_snapshot else None,
    )
    db.add(log)
    db.commit()

    tasklog.record_task(
        db, "qa", req.query, "done",
        detail=f"{_QTYPE_TO_MODE_CN.get(qtype, route)} · {cost}s",
    )

    return UnifiedAnswer(
        query_id=query_id,
        query=req.query,
        query_type=qtype,
        route=route,
        confidence=round(confidence, 3),
        answer=answer_body,
        evidence=evidence,  # type: ignore[arg-type]
        warnings=warnings,
        metadata={
            "time_cost_sec": cost,
            "selected_mode": chosen if chosen in _ROUTE_TO_QTYPE else None,
            "mode_cn": _QTYPE_TO_MODE_CN.get(qtype),
            "rule_hint": rule_hint_snapshot or None,
            **adapter_meta,
        },
    )


@router.get("/query/history", response_model=List[UnifiedAnswer])
def list_history(db: Session = Depends(get_db)):
    """List all query logs (newest first)."""
    logs = db.query(QueryLog).order_by(QueryLog.created_at.desc()).all()
    return [
        UnifiedAnswer(
            query_id=log.query_id,
            query=log.query,
            query_type=log.query_type,
            route=log.route,
            confidence=log.confidence,
            answer=json.loads(log.answer_json) if log.answer_json else {},
            evidence=json.loads(log.evidence_json) if log.evidence_json else [],
            warnings=json.loads(log.warnings_json) if log.warnings_json else [],
            metadata={
                "time_cost_sec": log.time_cost,
                "selected_mode": log.selected_mode,
                "mode_cn": _QTYPE_TO_MODE_CN.get(log.query_type),
                "rule_hint": json.loads(log.rule_hint) if log.rule_hint else None,
            },
        )
        for log in logs
    ]


@router.get("/query/{query_id}", response_model=UnifiedAnswer)
def get_query(query_id: str, db: Session = Depends(get_db)):
    """Retrieve a previously stored query result."""
    log = db.query(QueryLog).filter(QueryLog.query_id == query_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Query log not found")
    return UnifiedAnswer(
        query_id=log.query_id,
        query=log.query,
        query_type=log.query_type,
        route=log.route,
        confidence=log.confidence,
        answer=json.loads(log.answer_json) if log.answer_json else {},
        evidence=json.loads(log.evidence_json) if log.evidence_json else [],
        warnings=json.loads(log.warnings_json) if log.warnings_json else [],
        metadata={
            "time_cost_sec": log.time_cost,
            "selected_mode": log.selected_mode,
            "mode_cn": _QTYPE_TO_MODE_CN.get(log.query_type),
            "rule_hint": json.loads(log.rule_hint) if log.rule_hint else None,
        },
    )
