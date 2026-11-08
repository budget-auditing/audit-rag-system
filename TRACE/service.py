#!/usr/bin/env python3
"""TRACE micro-service (:8102)."""
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))

from fastapi import FastAPI
from pydantic import BaseModel
import faiss

import build_index as trace_build_index
from trace_rag.retrieval import embed_texts, search as trace_search, RiskMeasureCache
from trace_rag.evidence import build_evidence_chain
from trace_rag.generate import align_claims, call_chat_api

app = FastAPI(title="TRACE Service", version="0.1.0")

PORT = int(os.getenv("PORT", "8102"))
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8081/v1/embeddings")
CHAT_URL = os.getenv("CHAT_URL", "http://localhost:8000/v1/chat/completions")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
INDEX_DIR = os.getenv("INDEX_DIR", str(Path(__file__).resolve().parents[1] / "data" / "index" / "trace"))

_index: faiss.Index | None = None
_meta: List[Dict] = []
_info: Dict = {}
_built_at: str | None = None
_risk_cache: RiskMeasureCache | None = None


def _load_index() -> bool:
    global _index, _meta, _info, _built_at
    if not Path(INDEX_DIR).exists():
        return False
    faiss_path = os.path.join(INDEX_DIR, "index.faiss")
    meta_path = os.path.join(INDEX_DIR, "meta.jsonl")
    info_path = os.path.join(INDEX_DIR, "index.info.json")
    if not os.path.exists(faiss_path):
        return False
    _index = faiss.read_index(faiss_path)
    _meta = []
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    _meta.append(json.loads(line))
    _info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            _info = json.load(f)
    _built_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    return True


def _ensure_index():
    if _index is None and not _load_index():
        raise RuntimeError("Index not ready; call /index/build first")


def _ensure_risk_cache():
    global _risk_cache
    if _risk_cache is None:
        _risk_cache = RiskMeasureCache(
            server_url=EMBED_URL,
            batch_size=64,
            cache_size=max(len(_meta) + 1000, 20000),
            token_source="contextual",
        )


class BuildRequest(BaseModel):
    corpus_file: str
    doc_scope: List[str] | None = None


class AnswerRequest(BaseModel):
    query: str
    doc_scope: List[str] | None = None
    k: int = 8


@app.get("/health")
def health():
    ready = _index is not None or _load_index()
    return {
        "status": "ok",
        "route": "TRACE",
        "index_ready": ready,
        "index_dir": INDEX_DIR,
    }


@app.post("/index/build")
def index_build(req: BuildRequest):
    t0 = time.time()
    os.makedirs(INDEX_DIR, exist_ok=True)
    trace_build_index.build_index(
        corpus_file=req.corpus_file,
        output_dir=INDEX_DIR,
        add_meta_tokens=False,
        limit=None,
        batch_size=64,
        server_url=EMBED_URL,
        server_urls=None,
        workers=1,
        min_split_batch=16,
        clean_noise=False,
        inject_dept=False,
    )
    _load_index()
    elapsed = time.time() - t0
    return {
        "built": True,
        "n_chunks": len(_meta),
        "dim": _index.d if _index else None,
        "elapsed": elapsed,
    }


@app.get("/index/status")
def index_status():
    ready = _index is not None or _load_index()
    return {
        "ready": ready,
        "size": len(_meta),
        "dim": _index.d if _index else None,
        "built_at": _built_at,
    }


@app.post("/answer")
def answer(req: AnswerRequest):
    _ensure_index()
    _ensure_risk_cache()
    qvec = embed_texts(EMBED_URL, [req.query], batch_size=64)
    search_result = trace_search(
        index=_index,
        meta=_meta,
        query_vec=qvec,
        query_text=req.query,
        top_sem_k=50,
        top_k=req.k,
        lambda_ot=0.3,
        risk_mode="token",
        lambda_risk=0.1,
        lambda_num=0.1,
        risk_cache=_risk_cache,
    )
    hits = search_result.meta_indices
    candidates = [_meta[i] for i in hits]

    if not candidates:
        return {
            "oci": {"Observation": [], "Cause": [], "Impact": []},
            "abstain": True,
            "evidence_chain": [],
            "query": req.query,
        }

    o, c, i, score, cov, link, chain_abstain, candidate_pools = build_evidence_chain(
        index=_index,
        meta=_meta,
        query=req.query,
        top_m=8,
        top_sem_k=50,
        beam_width=10,
        eta=0.3,
        risk_mode="token",
        risk_matrix=None,
        risk_nums=None,
        risk_cache=_risk_cache,
        ot_cost=None,
        ot_kernel=None,
        ot_epsilon=0.05,
        ot_iters=50,
        use_ot=False,
        lambda_ot=0.0,
        lambda_risk=0.1,
        lambda_num=0.1,
        score_norm="minmax",
        use_role=True,
        use_coverage=True,
        delta_c=0.45,
        server_url=EMBED_URL,
        batch_size=64,
    )

    oci_input = {
        "Observation": [{"claim": o.get("text", "")}],
        "Cause": [{"claim": c.get("text", "")}],
        "Impact": [{"claim": i.get("text", "")}],
    }
    oci_input["Observation"] = [x for x in oci_input["Observation"] if x["claim"]]
    oci_input["Cause"] = [x for x in oci_input["Cause"] if x["claim"]]
    oci_input["Impact"] = [x for x in oci_input["Impact"] if x["claim"]]

    evidence_for_chain = [o, c, i]
    evidence_for_chain = [x for x in evidence_for_chain if x.get("text")]

    if not any(oci_input.values()):
        return {
            "oci": oci_input,
            "abstain": True,
            "evidence_chain": evidence_for_chain,
            "query": req.query,
            "coverage": cov,
            "link": link,
        }

    try:
        oci_json, _ = call_chat_api(
            req.query,
            evidence_for_chain,
            model=CHAT_MODEL,
            max_retries=2,
        )
        if not any(oci_json.values()):
            oci_json = oci_input
    except Exception:
        oci_json = oci_input

    verification = align_claims(
        oci=oci_json,
        evidence=evidence_for_chain,
        server_url=EMBED_URL,
        lambda_attr=0.7,
        delta_a=0.2,
        delta_v=0.2,
        use_local_suppression=True,
    )

    chain = []
    for role, cand in [("Observation", o), ("Cause", c), ("Impact", i)]:
        if cand.get("text"):
            chain.append({"role": role, **cand})

    return {
        "oci": verification.get("prediction_oci", oci_json),
        "abstain": bool(verification.get("abstain_flag") or chain_abstain),
        "evidence_chain": chain,
        "verification": {
            "claim_supported": verification.get("local_metrics", {}).get("supported_claims", 0) > 0,
            "attribute_consistent": verification.get("local_metrics", {}).get("qac_supported_claims", 0) > 0,
            "abstain": bool(verification.get("abstain_flag")),
        },
        "query": req.query,
        "coverage": cov,
        "link": link,
    }


if __name__ == "__main__":
    import uvicorn
    _load_index()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
