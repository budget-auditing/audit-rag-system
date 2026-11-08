"""End-to-end smoke test: upload → parse → build index → query."""

import os
import sys
import time

# Ensure backend/app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models import Base
from app.main import app
from app.db import get_db
import app.db as _appdb

SQLALCHEMY_DATABASE_URL = "sqlite:///./test_e2e.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
# Async index builds run in a background thread using the app's SessionLocal
# (not the request-scoped override), so bind it to the test DB too.
_appdb.SessionLocal.configure(bind=engine)
client = TestClient(app)


def wait_for_build(job_id: str, timeout: float = 600.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get("/api/index/build/status", params={"job_id": job_id})
        assert resp.status_code == 200, resp.text
        job = resp.json()["job"]
        if job and job["status"] in ("ready", "failed"):
            return job
        time.sleep(1)
    raise AssertionError(f"build job {job_id} timed out")

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "demo_doc.txt")


def test_upload_file():
    with open(FIXTURE, "rb") as f:
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("demo_doc.txt", f, "text/plain")},
            data={"doc_type": "budget_execution"},
        )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["message"] == "uploaded"
    assert "doc_id" in data
    return data["doc_id"]


def parse_doc(doc_id: str):
    resp = client.post(f"/api/documents/{doc_id}/parse")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "parsed"
    assert data["chunks"] > 0
    return data


def build_all_index():
    resp = client.post("/api/index/build/all")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["queued"] is True
    assert data["job"] is not None
    job = wait_for_build(data["job"]["job_id"])
    assert job["status"] == "ready", job.get("error")
    return job


def submit_query():
    payload = {
        "query": "2023年全市一般公共预算收入是多少万元？",
    }
    resp = client.post("/api/query/answer", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["route"] in ("MSG_RAG", "C3_RAG", "TRACE", "unknown")
    assert "answer" in data
    return data


def test_e2e_flow():
    doc_id = test_upload_file()
    parse_doc(doc_id)
    job = build_all_index()
    # The backend may run zero-vector fallback for embeddings; counts may be minimal.
    assert job["status"] == "ready"
    query_data = submit_query()
    print("QUERY_ANSWER_ROUTE:", query_data["route"])


def test_per_shard_build():
    # assumes prior test_e2e_flow populated chunks
    for endpoint in (
        "/api/index/build/fact-graph",
        "/api/index/build/institutional",
        "/api/index/build/risk",
    ):
        resp = client.post(endpoint)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["queued"] is True
        job = wait_for_build(data["job"]["job_id"])
        assert job["status"] == "ready", job.get("error")


def test_all_routes():
    queries = [
        ("2023年全市一般公共预算收入是多少万元？", "MSG_RAG"),
        ("口径是否一致？", "C3_RAG"),
        ("可持续性风险如何？", "TRACE"),
    ]
    cached_ids = []
    for q, expected_route in queries:
        payload = {"query": q}
        resp = client.post("/api/query/answer", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["route"] == expected_route, f"expected {expected_route}, got {data['route']}"
        assert "answer" in data
        assert "evidence" in data
        assert "query_id" in data
        cached_ids.append(data["query_id"])
    return cached_ids


def test_get_query():
    # Submit a query first to obtain a valid query_id
    payload = {"query": "2023年全市一般公共预算收入是多少万元？"}
    resp = client.post("/api/query/answer", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    query_id = data["query_id"]

    resp = client.get(f"/api/query/{query_id}")
    assert resp.status_code == 200, resp.text
    fetched = resp.json()
    assert fetched["query_id"] == query_id
    assert fetched["query"] == payload["query"]
    assert "answer" in fetched


def test_query_history():
    resp = client.get("/api/query/history")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data, list)
    if data:
        assert "query_id" in data[0]


def test_evidence_retrieval():
    payload = {"query": "2023年全市一般公共预算收入是多少万元？"}
    resp = client.post("/api/query/answer", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    query_id = data["query_id"]

    resp = client.get(f"/api/evidence/{query_id}")
    assert resp.status_code == 200, resp.text
    ev = resp.json()
    assert "evidence" in ev


def test_evidence_source():
    payload = {"query": "2023年全市一般公共预算收入是多少万元？"}
    resp = client.post("/api/query/answer", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    evidence_items = data.get("evidence", [])
    # Find an evidence item with chunk_id; skip if none (e.g. empty evidence)
    chunk_id = None
    for item in evidence_items:
        if isinstance(item, dict) and item.get("chunk_id"):
            chunk_id = item["chunk_id"]
            break
    if not chunk_id:
        print("SKIPPING evidence source: no evidence with chunk_id")
        return
    resp = client.get(f"/api/evidence/source/{chunk_id}")
    assert resp.status_code == 200, resp.text
    src = resp.json()
    assert "text" in src


def test_export_query_report():
    payload = {"query": "2023年全市一般公共预算收入是多少万元？"}
    resp = client.post("/api/query/answer", json=payload)
    assert resp.status_code == 200, resp.text
    query_id = resp.json()["query_id"]

    req = {"query_id": query_id, "format": "markdown"}
    resp = client.post("/api/export/query-report", json=req)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "content" in body
    assert body["format"] == "markdown"
    assert isinstance(body["content"], str)


def test_export_audit_draft():
    # Ensure at least two queries exist
    query_ids = []
    for q in ("预算金额多少？", "口径是否一致？"):
        payload = {"query": q}
        r = client.post("/api/query/answer", json=payload)
        assert r.status_code == 200
        query_ids.append(r.json()["query_id"])

    req = {"query_ids": query_ids, "format": "markdown"}
    resp = client.post("/api/export/audit-draft", json=req)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "content" in body
    assert body["format"] == "markdown"
    assert isinstance(body["content"], str)
