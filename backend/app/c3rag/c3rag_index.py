"""C³-RAG in-memory index."""
import json
import re
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

from app.utils.embed import embed_texts
from app.c3rag.evidence_parser import EvidenceParser, check_caliber_consistency


class RAGStoreFAISS:
    """Dense vector retrieval backed by FAISS."""

    def __init__(self) -> None:
        self.index: Optional[faiss.IndexFlatIP] = None
        self.texts: List[str] = []
        self.metas: List[Dict[str, Any]] = []
        self.dim: Optional[int] = None

    def build(self, items: List[Dict[str, Any]]) -> int:
        self.texts = []
        self.metas = []
        vectors: List[np.ndarray] = []
        batch_texts: List[str] = []
        batch_metas: List[Dict[str, Any]] = []

        for item in items:
            text = item.get("text") or item.get("span") or ""
            text = text.strip()
            if not text:
                continue
            batch_texts.append(text)
            batch_metas.append(item)
            if len(batch_texts) >= 32:
                embs = embed_texts(batch_texts)
                self.texts.extend(batch_texts)
                self.metas.extend(batch_metas)
                if embs.size:
                    vectors.extend(embs)
                    self.dim = embs.shape[1]
                batch_texts, batch_metas = [], []

        if batch_texts:
            embs = embed_texts(batch_texts)
            self.texts.extend(batch_texts)
            self.metas.extend(batch_metas)
            if embs.size:
                vectors.extend(embs)
                self.dim = embs.shape[1]

        if vectors and self.dim:
            arr = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(arr)
            return len(self.texts)
        return 0

    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        if self.index is None or self.dim is None:
            return []
        qvec = embed_texts([query])
        if not qvec.size:
            return []
        norm = np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12
        qvec = qvec / norm
        scores, idxs = self.index.search(qvec.astype("float32"), min(k, len(self.texts)))
        results = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            results.append({
                "score": float(score),
                "text": self.texts[int(i)],
                **self.metas[int(i)],
            })
        return results


class C3RAGIndex:
    """Vector + keyword + metadata extraction on standard chunks."""

    def __init__(self):
        self.items: List[Dict[str, Any]] = []  # chunk-derived evidence
        self.vector_store = RAGStoreFAISS()
        self.parser = EvidenceParser()

    def build(self, chunks: List[Any]) -> int:
        """Build sparse + dense index from StandardChunk objects."""
        self.items.clear()
        for chunk in chunks:
            text = (chunk.text or "").strip()
            if not text:
                continue
            meta = _safe_json(chunk.meta_json)
            parsed = self.parser.parse_evidence(text)
            item = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": text,
                "year": meta.get("year") or parsed["t"] or _guess_year(text),
                "department": meta.get("department") or parsed["d"],
                "metric": meta.get("metric") or _guess_metric_from_metrics(parsed["m"]),
                "caliber_meta": parsed["meta"],
                "span": text[:400],
                "source_location_json": chunk.source_location_json,
                "_tuple": parsed,
            }
            self.items.append(item)
        self.vector_store.build(self.items)
        return len(self.items)

    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """Return top-k evidence + conflict detection."""
        q = query_text.strip()

        # 1) dense retrieval
        dense = self.vector_store.search(q, k=top_k * 4)

        # 2) keyword fallback/diversity: blend keyword high-scorers not already in dense
        keyword_scores = []
        for item in self.items:
            score = _keyword_score(q, item["text"])
            if score > 0:
                keyword_scores.append((score, item))
        keyword_scores.sort(key=lambda x: x[0], reverse=True)

        seen = {d["chunk_id"] for d in dense}
        merged = list(dense)
        for _, item in keyword_scores:
            if item["chunk_id"] not in seen:
                merged.append({**item, "score": float(item.get("score", 0))})
                seen.add(item["chunk_id"])
            if len(merged) >= top_k * 3:
                break

        # 3) consistency re-rank: penalize entries that introduce conflicts
        final = _rank_by_consistency(q, merged, top_k)

        warnings: List[str] = []
        conflicts = _detect_conflicts(final)
        caliber_summary = _caliber_summary(final)
        if conflicts:
            warnings.append(f"检测到 {len(conflicts)} 条口径/年份冲突")
        return {
            "evidence": final,
            "conflicts": conflicts,
            "caliber_summary": caliber_summary,
            "top_n": len(final),
        }

    def status(self) -> Dict[str, Any]:
        return {
            "built": bool(self.items),
            "count": len(self.items),
            "vector_count": len(self.vector_store.texts) if self.vector_store else 0,
        }


# global singleton
c3rag_index = C3RAGIndex()


def _safe_json(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


# naive heuristics ----------------------------------------------------------

_YEAR_RE = re.compile(r"20[12]\d")


def _guess_year(text: str) -> str:
    m = _YEAR_RE.search(text)
    return m.group(0) if m else ""


def _guess_metric_from_metrics(metrics: List[str]) -> str:
    if not metrics:
        return ""
    priority = {
        "budget": "预算",
        "execution": "执行",
        "expenditure": "支出",
        "revenue": "收入",
        "transfer": "转移支付",
        "debt": "债务",
        "performance": "绩效",
        "balance": "结余",
    }
    for key, label in priority.items():
        if key in metrics:
            return label
    return metrics[0]


def _keyword_score(query: str, text: str) -> float:
    """Weighted keyword overlap with synonym expansion for institutional terms."""
    _SYNONYMS = {
        "口径": ["口径", "统计口径", "核算口径", "填报口径", "可比口径"],
        "变更": ["变更", "调整", "变化", "不同", "差异", "不一致"],
        "一致": ["一致", "相同", "可比", "统一"],
        "部门": ["部门", "市局", "县局", "单位"],
        "支出": ["支出", "花费", "开支"],
        "收入": ["收入", "税收", "非税"],
    }
    query_terms = set(query.split())
    expanded: set[str] = set()
    for term in query_terms:
        expanded.add(term)
        for canon, syns in _SYNONYMS.items():
            if term in syns:
                expanded.update(syns)
    text = text.lower()
    hits = 0
    covered = set()
    for term in expanded:
        t = term.lower()
        if t in text:
            hits += 1
            covered.add(t)
    coverage_bonus = min(0.5, 0.1 * len(covered))
    return round((hits + coverage_bonus) ** 0.7 + 0.05, 3)


def _detect_conflicts(evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Combine classic mismatch flags + parser-based caliber consistency check."""
    conflicts: List[Dict[str, Any]] = []
    years = [e["year"] for e in evidence if e.get("year")]
    metrics = [e["metric"] for e in evidence if e.get("metric")]
    if len(set(years)) > 1:
        conflicts.append({"type": "YEAR_MISMATCH", "field": "year", "values": sorted(list(set(years)))})
    if len(set(metrics)) > 1:
        conflicts.append({"type": "METRIC_MISMATCH", "field": "metric", "values": sorted(list(set(metrics)))})

    # Caliber-level consistency via 5-tuple parser
    tuples = [e.get("_tuple") for e in evidence if e.get("_tuple")
              and (e.get("year") or e.get("department") or e.get("caliber_meta"))]
    if len(tuples) >= 2:
        check = check_caliber_consistency(tuples)
        if not check["consistent"]:
            seen_types = {c.get("field") for c in conflicts}
            ctype: str = ""
            for issue in check["issues"]:
                inferred = _infer_conflict_type(issue)
                # avoid duplicating a year conflict we already reported
                if inferred == "YEAR_MISMATCH" and "year" in seen_types:
                    continue
                if inferred != ctype:
                    conflicts.append({"type": inferred, "field": "caliber", "message": issue})
                    ctype = inferred
    return conflicts


def _infer_conflict_type(issue: str) -> str:
    issue = issue or ""
    if "年份" in issue or "年" in issue:
        return "YEAR_MISMATCH"
    if "部门" in issue or "单位" in issue:
        return "DEPT_MISMATCH"
    if "口径" in issue or "范围" in issue:
        return "CALIBER_SHIFT"
    if "合并" in issue or "单体" in issue:
        return "CALIBER_SHIFT"
    return "CAUSAL_GAP"


def _caliber_summary(evidence: List[Dict[str, Any]]) -> str:
    years = sorted({e["year"] for e in evidence if e.get("year")})
    metas = set()
    for e in evidence:
        metas.update(e.get("caliber_meta") or [])
    parts = []
    if years:
        parts.append(f"涉及年份: {', '.join(years)}")
    if metas:
        parts.append(f"口径注记: {', '.join(sorted(metas))}")
    return "；".join(parts) if parts else ""


def _rank_by_consistency(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Re-rank candidates: preference for dense score + query keyword overlap."""
    scored = []
    for cand in candidates:
        dense = cand.get("score", 0)
        kw = _keyword_score(query, cand["text"])
        # penalize if candidate has no year/metric for comparison questions
        has_meta = bool(cand.get("year") or cand.get("metric"))
        bonus = 0.02 if has_meta else 0
        scored.append((dense + kw + bonus, cand))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [cand for _, cand in scored[:top_k]]
