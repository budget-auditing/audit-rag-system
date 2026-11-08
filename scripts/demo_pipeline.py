#!/usr/bin/env python3
"""End-to-end demo pipeline for MSG_RAG/C3_RAG/TRACE over 3 specific audit docs."""

import json
from pathlib import Path

import requests

BASE = "http://localhost:8001"
FIXTURES = Path(__file__).resolve().parents[1] / "backend" / "tests" / "fixtures"

DEMO_DOC_NAMES = {
    "demo_budget_2024.txt",
    "demo_execution_comparison.txt",
    "demo_risk_2024.txt",
    "demo_doc.txt",
}


def cleanup_demo_docs():
    r = requests.get(f"{BASE}/api/documents")
    r.raise_for_status()
    print(f"[Cleanup] Scanning {len(r.json())} existing document(s)...")
    for doc in r.json():
        if doc.get("doc_name", "") in DEMO_DOC_NAMES:
            doc_id = doc["doc_id"]
            print(f"  Deleting old demo doc: {doc['doc_name']} -> {doc_id}")
            resp = requests.delete(f"{BASE}/api/documents/{doc_id}")
            if resp.status_code == 404:
                print(f"  Already deleted: {doc_id}")
            else:
                resp.raise_for_status()


def upload_and_parse(path: Path):
    # 1. upload
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/api/documents/upload", files={"file": (path.name, f)})
    r.raise_for_status()
    doc_id = r.json()["doc_id"]
    print(f"  Uploaded {path.name} -> {doc_id}")

    # 2. parse
    r2 = requests.post(f"{BASE}/api/documents/{doc_id}/parse")
    r2.raise_for_status()
    j = r2.json()
    print(f"  Parsed -> {j['chunks']} chunks")
    return doc_id


def build_indexes():
    r = requests.post(f"{BASE}/api/index/build/all")
    r.raise_for_status()
    print(f"  Build index -> {r.json()}")


def query(q: str, force_route: str = None) -> dict:
    payload = {"query": q}
    if force_route:
        payload["force_route"] = force_route
    r = requests.post(f"{BASE}/api/query/answer", json=payload)
    r.raise_for_status()
    return r.json()


def pretty_print(resp: dict):
    print(json.dumps(resp, ensure_ascii=False, indent=2))


def main():
    # 0. Remove leftover demo documents + chunks from previous runs
    cleanup_demo_docs()

    # 1. Upload 3 dedicated docs
    docs = [
        "demo_budget_2024.txt",
        "demo_execution_comparison.txt",
        "demo_risk_2024.txt",
    ]
    doc_ids = []
    for name in docs:
        print(f"\n[Upload & Parse] {name}")
        doc_ids.append(upload_and_parse(FIXTURES / name))

    # 2. Build all indexes
    print("\n[Build Indexes]")
    build_indexes()

    # 3. Run queries
    print("\n" + "=" * 60)
    print("[Query] MSG_RAG – factual numeric")
    print("=" * 60)
    q_msg = "2024年全市一般公共预算收入安排多少万元？税收收入占比多少？"
    resp_msg = query(q_msg)
    pretty_print(resp_msg)

    print("\n" + "=" * 60)
    print("[Query] C3_RAG – institutional consistency")
    print("=" * 60)
    q_c3rag = "2024年与2023年相比，科学技术支出的口径是否一致？是否存在口径变更或部门口径不一致？"
    resp_c3rag = query(q_c3rag)
    pretty_print(resp_c3rag)

    print("\n" + "=" * 60)
    print("[Query] TRACE – hidden risk")
    print("=" * 60)
    q_trace = "审计发现了哪些财政可持续性风险？可能造成什么影响？"
    resp_trace = query(q_trace)
    pretty_print(resp_trace)

    print("\n" + "=" * 60)
    print("[Query] Mixed – auto routing")
    print("=" * 60)
    q_mixed = "一般公共预算收入的增长能否覆盖政府债务还本付息支出？是否存在隐性债务风险？"
    resp_mixed = query(q_mixed)
    pretty_print(resp_mixed)

    print("\n✅ Demo pipeline complete. Check responses above.")

if __name__ == "__main__":
    main()
