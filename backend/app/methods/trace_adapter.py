"""TRACE micro-service adapter.

Maps TRACE OCI/abstain/evidence_chain output to the unified envelope.
"""
from typing import List

from app.config import settings
import httpx


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


class TRACEAdapter:
    route = "TRACE"

    def __init__(self, base_url: str | None = None, timeout: float = 600.0):
        self.base_url = base_url or settings.trace_service_url
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        try:
            resp = httpx.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"TRACE service error: {exc.response.text}") from exc
        except Exception as exc:
            raise RuntimeError(f"TRACE service unreachable: {exc}") from exc

    def answer(self, query: str, doc_scope: List[str] | None = None) -> dict:
        raw = self._post("/answer", {"query": query, "doc_scope": doc_scope, "k": 8})
        return self._normalise(raw)

    def _normalise(self, raw: dict) -> dict:
        oci = raw.get("oci") or {}
        abstain = bool(raw.get("abstain"))
        chain = raw.get("evidence_chain", [])
        verification = raw.get("verification") or {}

        answer = {
            "summary": "",
            "observation": " ".join(self._claims(oci.get("Observation", []))),
            "cause": " ".join(self._claims(oci.get("Cause", []))),
            "impact": " ".join(self._claims(oci.get("Impact", []))),
            "abstain": abstain,
            "verification": {
                "claim_supported": verification.get("claim_supported"),
                "attribute_consistent": verification.get("attribute_consistent"),
                "single_passage_supported": verification.get("single_passage_supported"),
                "abstain": abstain,
            },
        }
        if not answer["observation"] and not answer["cause"] and not answer["impact"]:
            answer["summary"] = "证据不足，系统 abstain。"

        evidence = []
        for item in chain:
            role = item.get("role", "Observation")
            txt = item.get("text", "")
            evidence.append({
                "role": role,
                "text": txt,
                "span": txt,
                "source_doc": item.get("source") or item.get("source_path") or "",
                "page": item.get("page"),
                "score": item.get("score"),
                "chunk_id": _find_chunk_id(txt),
            })

        warnings = []
        if abstain:
            warnings.append("证据不足，返回保守结论或 abstain。")
        if not verification.get("claim_supported", True):
            warnings.append("部分 OCI claim 缺乏证据支持。")

        return {
            "answer": answer,
            "evidence": evidence,
            "warnings": warnings,
            "metadata": {
                "index_version": raw.get("index_version") or raw.get("built_at") or "",
            },
        }

    def _claims(self, items: list) -> list[str]:
        out = []
        for it in items:
            if isinstance(it, dict):
                claim = it.get("claim") or ""
                if claim:
                    out.append(str(claim))
            elif isinstance(it, str):
                out.append(it)
        return out


# singleton
trace_adapter = TRACEAdapter()
