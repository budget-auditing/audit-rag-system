"""C³-RAG micro-service adapter.

Normalises C³-RAG output to the unified envelope and maps conflict labels to the
standard conflict label set.
"""
from typing import List
import httpx

from app.config import settings


def _strip_cot(text: str) -> str:
    """Remove chain-of-thought that reasoning models embed in `content`.

    Qwen-style models emit the full CoT ending with `</think>` before the real
    answer. Keep only the text after the last `</think>`.
    """
    if not text:
        return text
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[-1].strip()
        if tail:
            return tail
    return text.strip()


def _find_chunk_id(text: str) -> str:
    """Best-effort lookup of chunk_id by evidence text."""
    try:
        from app.db import SessionLocal
        from app.models import StandardChunk
        target = (text or "").strip()
        if not target:
            return ""
        with SessionLocal() as db:
            rows = db.query(StandardChunk).all()
            for row in rows:
                if target in (row.text or "") or (row.text or "") in target:
                    return row.chunk_id
    except Exception:
        pass
    return ""


class C3RAGAdapter:
    route = "C3_RAG"



    def __init__(self, base_url: str | None = None, timeout: float = 600.0):
        self.base_url = base_url or settings.c3rag_service_url
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"C³-RAG service error: {exc.response.text}") from exc
        except Exception as exc:
            raise RuntimeError(f"C³-RAG service unreachable: {exc}") from exc

    def answer(self, query: str, doc_scope: List[str] | None = None) -> dict:
        raw = self._post("/answer", {"query": query, "doc_scope": doc_scope, "k": 8})
        return self._normalise(raw, query)

    def _normalise(self, raw: dict, query: str) -> dict:
        evidence = self._normalise_evidence(raw.get("evidence", []))
        conflicts = self._normalise_conflicts(raw.get("evidence", []), query)
        score = float(raw.get("consistency_score", 0.0))

        return {
            "answer": {
                "summary": f"一致性评分 {score:.3f}，检索到 {len(evidence)} 条证据。",
                "generated_answer": _strip_cot(raw.get("answer", "")),
                "consistency_score": score,
                "consistency_result": {
                    "caliber_consistent": score >= 0.7,
                    "causal_complete": not any(
                        c["type"] == "CAUSAL_GAP" for c in conflicts
                    ),
                    "conflict_labels": [c["type"] for c in conflicts],
                },
                "conflicts": conflicts,
            },
            "evidence": evidence,
            "warnings": self._build_warnings(conflicts),
            "metadata": {
                "index_version": raw.get("index_version") or raw.get("built_at") or "",
            },
        }

    def _normalise_evidence(self, evidence: list[dict]) -> list[dict]:
        out = []
        for e in evidence:
            txt = e.get("text", "")
            item = {
                "text": txt,
                "span": txt,
                "source_doc": e.get("source") or "",
                "page": e.get("page"),
                "score": e.get("score"),
                "chunk_id": _find_chunk_id(txt),
            }
            out.append(item)
        return out

    def _normalise_conflicts(self, evidence: list[dict], query: str) -> list[dict]:
        conflicts: List[dict] = []
        years = sorted({e.get("year") for e in evidence if e.get("year")})
        if len(years) > 1:
            conflicts.append({
                "type": "YEAR_MISMATCH",
                "field": "year",
                "values": years,
                "message": f"证据涉及多个年份：{', '.join(years)}",
            })

        depts = sorted({e.get("source") for e in evidence if e.get("source")})[:3]
        if len(depts) > 1:
            conflicts.append({
                "type": "DEPT_MISMATCH",
                "field": "department",
                "values": depts,
                "message": f"证据来源不同：{', '.join(depts)}",
            })

        query_depts = self._extract_depts(query)
        for e in evidence:
            src = e.get("source", "")
            if src and query_depts and not any(d in src for d in query_depts):
                conflicts.append({
                    "type": "DEPT_MISMATCH",
                    "field": "department",
                    "message": f"证据来源 '{src}' 与查询部门不一致",
                })
                break

        metrics = sorted({e.get("metric") for e in evidence if e.get("metric")})
        if len(metrics) > 1:
            conflicts.append({
                "type": "METRIC_MISMATCH",
                "field": "metric",
                "values": metrics,
                "message": f"证据涉及不同指标：{', '.join(metrics)}",
            })

        if any("口径" in e.get("text", "") for e in evidence):
            conflicts.append({
                "type": "CALIBER_MISMATCH",
                "field": "caliber",
                "message": "证据中存在统计口径/资金范围变化",
            })

        if not conflicts:
            conflicts.append({
                "type": "EVIDENCE_INSUFFICIENT",
                "field": "general",
                "message": "未识别到明确冲突标签",
            })

        return conflicts

    def _extract_depts(self, text: str) -> list[str]:
        import re
        return re.findall(r"[\u4e00-\u9fa5]{2,}(?:厅|局|委|部|处|科|院|会|办)", text)

    def _build_warnings(self, conflicts: list[dict]) -> list[str]:
        if not conflicts:
            return []
        msgs = [f"检测到 {conflicts[0]['type']}: {conflicts[0].get('message', '')}"]
        return msgs


# singleton used by query router
c3rag_adapter = C3RAGAdapter(timeout=600.0)
