"""Minimal TRACE in-memory index."""
import json
from typing import Any, Dict, List


_RISK_TERMS = ["风险", "隐患", "偏差", "虚报", "瞒报", "违规", "不合规", "可持续性"]
_OBS_TERMS = ["发现", "观测", "数据显示", "支出", "收入"]
_CAUSE_TERMS = ["由于", "因为", "导致", "原因"]
_IMPACT_TERMS = ["影响", "后果", "威胁", "下降", "上升"]


class TRACEIndex:
    """Keyword + risk term tagging on standard chunks, returns OCI evidence."""

    def __init__(self):
        self.passages: List[Dict[str, Any]] = []

    def build(self, chunks: List[Any]) -> int:
        self.passages.clear()
        for chunk in chunks:
            text = (chunk.text or "").strip()
            if not text:
                continue
            meta = _safe_json(chunk.meta_json)
            risk_tags = _extract_risk_tags(text)
            passage = {
                "chunk_id": chunk.chunk_id,
                "text": text,
                "span": text[:400],
                "year": meta.get("year") or _guess_year(text),
                "statistical_basis": meta.get("statistical_basis") or "",
                "risk_features": risk_tags,
                "source_location_json": chunk.source_location_json,
            }
            self.passages.append(passage)
        return len(self.passages)

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        q = query_text.strip()
        scores = []
        for p in self.passages:
            score = _keyword_score(q, p.get("text", ""))
            if p["risk_features"]:
                score += 2  # slight boost for risk-bearing passages
            scores.append((score, p))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = [p for sc, p in scores[:top_k] if sc > 0]
        warnings: List[str] = []
        if not top:
            warnings.append("未检索到足够风险证据，回答 abstain")
        return {
            "evidence": top,
            "risk_count": len([p for p in top if p["risk_features"]]),
            "warnings": warnings,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "built": bool(self.passages),
            "count": len(self.passages),
        }


# singleton
trace_index = TRACEIndex()


def _safe_json(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _guess_year(text: str) -> str:
    import re
    m = re.search(r"20[12]\d", text)
    return m.group(0) if m else ""


def _extract_risk_tags(text: str) -> List[str]:
    tags = []
    for t in _RISK_TERMS:
        if t in text:
            tags.append(t)
    return tags


def _keyword_score(query: str, text: str) -> int:
    terms = set(query.split())
    return sum(1 for t in terms if t in text)
