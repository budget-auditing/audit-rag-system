#!/usr/bin/env python3
"""C³-RAG micro-service (:8101)."""
import os
import json
import time
from pathlib import Path
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# C³-RAG modules live beside this file
import build_index
import pipeline

app = FastAPI(title="C³-RAG Service", version="0.1.0")

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8101"))
EMBED_URL = os.getenv("EMBED_URL", "http://localhost:8081/v1")
CHAT_URL = os.getenv("CHAT_URL", "http://localhost:8000/v1")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
INDEX_DIR = os.getenv("INDEX_DIR", str(Path(__file__).resolve().parents[1] / "data" / "index" / "c3rag"))
CCES_MODEL_PATH = os.getenv("CCES_MODEL_PATH", str(Path(__file__).resolve().parent / "models" / "cces_model.pt"))
ENCODER_NAME = os.getenv("ENCODER_NAME", "nghuyong/ernie-3.0-base-zh")
API_KEY = os.getenv("OPENAI_API_KEY", "sk-placeholder")

# Two clients: embed endpoint may differ from chat endpoint
embed_client = OpenAI(base_url=EMBED_URL, api_key=API_KEY)
chat_client = OpenAI(base_url=CHAT_URL, api_key=API_KEY)

# Shared state
rag_store: build_index.RAGStore | None = None
_c3rag_pipeline: pipeline.C3RAGPipeline | None = None
_index_ready = False
_built_at: str | None = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_index() -> bool:
    global rag_store, _index_ready, _built_at
    if not os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        return False
    rag_store = build_index.RAGStore(embed_client)
    rag_store.load(INDEX_DIR)
    _index_ready = True
    _built_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    return True


def _get_pipeline() -> pipeline.C3RAGPipeline:
    global _c3rag_pipeline
    if _c3rag_pipeline is None:
        if rag_store is None and not _load_index():
            raise RuntimeError("Index not ready; call /index/build first")
        _c3rag_pipeline = pipeline.load_c3rag_pipeline(
            index_dir=INDEX_DIR,
            cces_model_path=CCES_MODEL_PATH,
            encoder_name=ENCODER_NAME,
            llm_client=chat_client,
            embed_client=embed_client,
        )
    return _c3rag_pipeline


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BuildRequest(BaseModel):
    chunks_path: str  # path to JSON/JSONL or folder
    doc_scope: List[str] | None = None


class AnswerRequest(BaseModel):
    query: str
    doc_scope: List[str] | None = None
    k: int = 8


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "route": "C3_RAG",
        "index_ready": _index_ready,
        "index_dir": INDEX_DIR,
    }


@app.post("/index/build")
def index_build(req: BuildRequest):
    global rag_store, _index_ready, _built_at
    t0 = time.time()
    p = Path(req.chunks_path)
    if p.is_dir():
        docs = build_index.load_docs_from_folder(str(p))
    else:
        docs = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                docs.append(json.loads(line))
    rag_store = build_index.RAGStore(embed_client)
    rag_store.build(docs)
    os.makedirs(INDEX_DIR, exist_ok=True)
    rag_store.save(INDEX_DIR)
    _index_ready = True
    _built_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    elapsed = time.time() - t0
    return {
        "built": True,
        "n_chunks": len(rag_store.meta),
        "dim": rag_store.dim,
        "elapsed": elapsed,
    }


@app.get("/index/status")
def index_status():
    if not _index_ready and not _load_index():
        return {"ready": False, "size": 0, "dim": None, "built_at": None}
    return {
        "ready": True,
        "size": len(rag_store.meta) if rag_store else 0,
        "dim": rag_store.dim if rag_store else None,
        "built_at": _built_at,
    }


@app.post("/answer")
def answer(req: AnswerRequest):
    c3rag_pipe = _get_pipeline()
    result = c3rag_pipe.run(req.query, k=req.k, doc_scope=req.doc_scope)
    return {
        "answer": result["answer"],
        "consistency_score": result["consistency_score"],
        "evidence": result["evidence"],
        "evidence_count": result["evidence_count"],
    }


if __name__ == "__main__":
    import uvicorn
    _load_index()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
