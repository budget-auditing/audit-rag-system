#!/usr/bin/env python3
"""
Build augmented queries for TRACE evaluation (Strategies 1-3).

Strategy 1: Rewrite "neither hit" queries with document-specific risk signals
Strategy 2: Generate cross-table reconciliation queries from gold docs
Strategy 3: Generate multi-evidence OCI queries from multi-chunk docs

Usage:
    python -m trace_rag.utils.build_aug_queries --strategy s1,s2,s3 --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set
from urllib import request

import tqdm

CHAT_BASE_URL = os.getenv("CHAT_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct")

MAX_CHUNK_CHARS = 500
MIN_QUERY_CHARS = 100
MAX_QUERY_CHARS = 1500

S2_FOCUSES = [
    "重点关注：收支总表与财政拨款收支表之间的金额差异及其形成原因。",
    "重点关注：项目支出表与支出总表之间的口径差异，以及单位资金/结转资金的使用。",
    "重点关注：上年结转结余在本年的使用情况与年末结转的勾稽关系。",
]

S3_FOCUSES = [
    "重点关注：收支缺口的资金来源、弥补方式与可持续性风险。",
    "重点关注：非财政拨款资金的使用去向、合规性与口径差异。",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def read_jsonl(path: str) -> list:
    items = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def doc_prefix(evidence_id: str) -> str:
    return evidence_id.rsplit("-", 1)[0] if "-" in evidence_id else evidence_id


def call_llm(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "top_k": 20,
        "max_tokens": max_tokens,
        "presence_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(1, max_retries + 1):
        try:
            req = request.Request(
                CHAT_BASE_URL,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
            return content.strip()
        except Exception:
            if attempt == max_retries:
                return ""
            time.sleep(1.5 * attempt)
    return ""


def clean_query(text: str) -> str:
    """Remove common LLM preamble and formatting."""
    text = text.strip()
    text = re.sub(
        r"^(好的[，,。.]?\s*|以下是.*?[：:]\s*|改写后的查询[：:]\s*|"
        r"改写后[：:]\s*|问题[：:]\s*|审计问题[：:]\s*)",
        "", text,
    )
    text = re.sub(r"^\*\*.*?\*\*\s*", "", text)
    text = re.sub(r"^#+\s*", "", text)
    return text.strip()


def qc_query(text: str) -> bool:
    """Basic programmatic QC for a generated query."""
    if not text:
        return False
    if len(text) < MIN_QUERY_CHARS or len(text) > MAX_QUERY_CHARS:
        return False
    if not re.search(r"\d", text):
        return False
    if not re.search(r"(请|说明|核实|核查|解释|分析|是否|如何)", text):
        return False
    return True


def format_chunks(chunks: List[Dict], max_chars: int = MAX_CHUNK_CHARS) -> str:
    parts = []
    for c in chunks:
        eid = c.get("evidence_id", "")
        text = (c.get("text") or "")[:max_chars]
        parts.append(f"[{eid}]\n{text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Strategy 1: Rewrite "neither hit" queries
# ---------------------------------------------------------------------------

def build_s1_targets(
    splits_path: str,
    qrels: Dict[str, Set[str]],
    naive_path: str,
    trace_path: str,
    meta: Dict[str, Dict],
    k: int = 20,
) -> List[Dict]:
    queries = {
        item["row_id"]: item
        for item in read_jsonl(splits_path)
        if item.get("row_id")
    }
    naive = {
        item["row_id"]: item
        for item in read_jsonl(naive_path)
        if item.get("row_id")
    }
    trace = {
        item["row_id"]: item
        for item in read_jsonl(trace_path)
        if item.get("row_id")
    }

    targets = []
    for rid, q in queries.items():
        if rid not in qrels or rid not in naive or rid not in trace:
            continue
        gold = qrels[rid]
        naive_ids = set(naive[rid].get("topk_evidence_ids", [])[:k])
        trace_ids = set(trace[rid].get("topk_evidence_ids", [])[:k])
        if not (naive_ids & gold) and not (trace_ids & gold):
            gold_chunks = [meta[g] for g in sorted(gold) if g in meta]
            if gold_chunks:
                targets.append({
                    "source_row_id": rid,
                    "row_id": f"{rid}-s1",
                    "query": q.get("query", ""),
                    "gold_evidence_ids": sorted(gold),
                    "chunks": gold_chunks,
                })
    return targets


def s1_generate(target: Dict, model: str) -> Dict:
    system = "你是预算审计查询改写专家。"
    evidence = format_chunks(target["chunks"])
    user = (
        "给定一个审计查询和对应的证据文档，请改写查询使其包含更多来自证据的具体风险线索。\n\n"
        "要求：\n"
        "1. 从证据中提取 2-3 个具有区分度的数值（如具体金额、差额、比率），注入查询中\n"
        "2. 引用具体的表编号与科目名称（如'表03支出合计XX万元'）\n"
        "3. 明确指出跨表差额或口径异常\n"
        "4. 保持审计问题性质（以'请说明/请核实/请解释'结尾）\n"
        "5. 只输出改写后的查询，300-500字\n\n"
        f"原始查询：\n{target['query'][:600]}\n\n"
        f"证据文档：\n{evidence}\n"
    )
    text = clean_query(call_llm(system, user, model=model))
    return {
        "row_id": target["row_id"],
        "source_row_id": target["source_row_id"],
        "query": text,
        "gold_evidence_ids": target["gold_evidence_ids"],
        "strategy": "s1",
        "qc": qc_query(text),
    }


# ---------------------------------------------------------------------------
# Strategy 2: Cross-table reconciliation queries
# ---------------------------------------------------------------------------

def build_s2_targets(
    qrels: Dict[str, Set[str]],
    meta: Dict[str, Dict],
) -> List[Dict]:
    doc_gold: Dict[str, Set[str]] = defaultdict(set)
    for golds in qrels.values():
        prefix = doc_prefix(next(iter(golds)))
        doc_gold[prefix].update(golds)

    doc_chunks: Dict[str, List[Dict]] = defaultdict(list)
    for eid, m in meta.items():
        doc_chunks[doc_prefix(eid)].append(m)

    targets = []
    for prefix in sorted(doc_gold.keys()):
        chunks = doc_chunks.get(prefix, [])
        if len(chunks) < 2:
            continue
        gold_eids = sorted(doc_gold[prefix])
        for i, focus in enumerate(S2_FOCUSES):
            targets.append({
                "row_id": f"{prefix}-s2-{i + 1}",
                "source_doc": prefix,
                "gold_evidence_ids": gold_eids,
                "chunks": chunks,
                "focus": focus,
                "seq": i + 1,
            })
    return targets


def s2_generate(target: Dict, model: str) -> Dict:
    system = "你是预算审计数据标注员。"
    evidence = format_chunks(target["chunks"])
    user = (
        "给定某单位多张预算表的内容，请生成一个'跨表口径核对'类审计问题。\n\n"
        "要求：\n"
        "1. 引用至少两张表的具体数值，指出表间金额差异\n"
        "2. 要求解释差异原因（如结转资金、单位资金、口径范围）\n"
        "3. 包含具体金额（万元），300-500字\n"
        "4. 以'请说明/请核实/请解释'结尾\n"
        "5. 只输出问题文本\n\n"
        f"{target['focus']}\n\n"
        f"表格内容：\n{evidence}\n"
    )
    text = clean_query(call_llm(system, user, model=model))
    return {
        "row_id": target["row_id"],
        "source_doc": target["source_doc"],
        "query": text,
        "gold_evidence_ids": target["gold_evidence_ids"],
        "strategy": "s2",
        "seq": target["seq"],
        "qc": qc_query(text),
    }


# ---------------------------------------------------------------------------
# Strategy 3: Multi-evidence OCI queries
# ---------------------------------------------------------------------------

def build_s3_targets(
    qrels: Dict[str, Set[str]],
    meta: Dict[str, Dict],
) -> List[Dict]:
    doc_gold: Dict[str, Set[str]] = defaultdict(set)
    for golds in qrels.values():
        prefix = doc_prefix(next(iter(golds)))
        doc_gold[prefix].update(golds)

    doc_chunks: Dict[str, List[Dict]] = defaultdict(list)
    for eid, m in meta.items():
        doc_chunks[doc_prefix(eid)].append(m)

    targets = []
    for prefix in sorted(doc_gold.keys()):
        gold_eids = doc_gold[prefix]
        if len(gold_eids) < 3:
            continue
        chunks = doc_chunks.get(prefix, [])
        if len(chunks) < 3:
            continue
        gold_eids_sorted = sorted(gold_eids)
        for i, focus in enumerate(S3_FOCUSES):
            targets.append({
                "row_id": f"{prefix}-s3-{i + 1}",
                "source_doc": prefix,
                "gold_evidence_ids": gold_eids_sorted,
                "chunks": chunks,
                "focus": focus,
                "seq": i + 1,
            })
    return targets


def s3_generate(target: Dict, model: str) -> Dict:
    system = "你是预算审计数据标注员。"
    evidence = format_chunks(target["chunks"])
    user = (
        "给定某单位预算文档的多个段落，请生成一个需要综合多段证据才能回答的审计问题。"
        "答案应涵盖现象(Observation)、原因(Cause)、影响(Impact)三个维度。\n\n"
        "要求：\n"
        "1. 问题需要至少3个不同段落的信息才能完整回答\n"
        "2. 包含具体数值和表编号\n"
        "3. 300-500字\n"
        "4. 以'请说明/请核实'结尾\n"
        "5. 只输出问题文本\n\n"
        f"{target['focus']}\n\n"
        f"文档段落：\n{evidence}\n"
    )
    text = clean_query(call_llm(system, user, model=model))
    return {
        "row_id": target["row_id"],
        "source_doc": target["source_doc"],
        "query": text,
        "gold_evidence_ids": target["gold_evidence_ids"],
        "strategy": "s3",
        "seq": target["seq"],
        "qc": qc_query(text),
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_strategy(
    targets: List[Dict],
    generate_fn,
    output_path: str,
    workers: int,
    model: str,
) -> List[Dict]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    done_ids: Set[str] = set()
    results: List[Dict] = []
    if os.path.exists(output_path):
        for item in read_jsonl(output_path):
            done_ids.add(item.get("row_id"))
            results.append(item)

    pending = [t for t in targets if t["row_id"] not in done_ids]
    if not pending:
        print(f"  All {len(targets)} items already done, skipping")
        return results

    print(f"  {len(done_ids)} done, {len(pending)} pending")

    write_lock = threading.Lock()
    new_results: List[Dict] = []

    with open(output_path, "a", encoding="utf-8") as f:
        if workers <= 1:
            for target in tqdm.tqdm(pending, desc="  Generate"):
                result = generate_fn(target, model)
                new_results.append(result)
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(generate_fn, t, model): t for t in pending}
                for future in tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc="  Generate"
                ):
                    result = future.result()
                    new_results.append(result)
                    with write_lock:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()

    results.extend(new_results)
    passed = sum(1 for r in new_results if r.get("qc"))
    total = max(len(new_results), 1)
    print(f"  Generated: {len(new_results)}, QC passed: {passed} ({passed / total * 100:.0f}%)")

    return results


def merge_outputs(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    original_qrels_path: str,
    original_splits_path: str,
    qrels_output: str,
    splits_output: str,
):
    """Merge strategy outputs (QC-passed only) with originals."""
    original_qrels = read_jsonl(original_qrels_path)
    original_splits = read_jsonl(original_splits_path)

    aug_items: List[Dict] = []
    for path in [s1_path, s2_path, s3_path]:
        if not os.path.exists(path):
            continue
        for item in read_jsonl(path):
            if item.get("qc"):
                aug_items.append(item)

    os.makedirs(os.path.dirname(qrels_output), exist_ok=True)
    with open(qrels_output, "w", encoding="utf-8") as f:
        for item in original_qrels:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        for item in aug_items:
            f.write(json.dumps({
                "row_id": item["row_id"],
                "gold_evidence_ids": item["gold_evidence_ids"],
            }, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(splits_output), exist_ok=True)
    with open(splits_output, "w", encoding="utf-8") as f:
        for item in original_splits:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        for item in aug_items:
            f.write(json.dumps({
                "row_id": item["row_id"],
                "qid": item["row_id"],
                "query": item["query"],
            }, ensure_ascii=False) + "\n")

    by_strategy = defaultdict(int)
    for item in aug_items:
        by_strategy[item.get("strategy", "?")] += 1

    print(f"  Original: {len(original_splits)} queries, {len(original_qrels)} qrels")
    print(f"  Augmented (QC passed): {len(aug_items)} total", end="")
    if by_strategy:
        parts = [f"{k}={v}" for k, v in sorted(by_strategy.items())]
        print(f" ({', '.join(parts)})")
    else:
        print()
    print(f"  -> {qrels_output}")
    print(f"  -> {splits_output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build augmented queries (Strategies 1-3)"
    )
    parser.add_argument(
        "--strategy", default="s1,s2,s3",
        help="Comma-separated strategies to run (default: s1,s2,s3)",
    )
    parser.add_argument("--index-dir", default="runs/index")
    parser.add_argument("--splits-dir", default="datasets-public/audit-ociqa-demo/splits")
    parser.add_argument("--split", default="test")
    parser.add_argument("--qrels", default="datasets-public/audit-ociqa-demo/gold.qrels.jsonl")
    parser.add_argument("--naive", default="runs/audit-ociqa/qwen/retrieval/naive-dense-rag.jsonl")
    parser.add_argument("--trace", default="runs/audit-ociqa/qwen/retrieval/trace.jsonl")
    parser.add_argument("--output-dir", default="runs/audit-ociqa/aug")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--k", type=int, default=20, help="Top-K for neither-hit detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategies = [s.strip() for s in args.strategy.split(",") if s.strip()]

    print("Loading index meta...")
    meta: Dict[str, Dict] = {}
    for item in read_jsonl(os.path.join(args.index_dir, "meta.jsonl")):
        eid = item.get("evidence_id")
        if eid:
            meta[eid] = item
    print(f"  {len(meta)} evidence chunks")

    print("Loading qrels...")
    qrels: Dict[str, Set[str]] = {}
    for item in read_jsonl(args.qrels):
        rid = item.get("row_id")
        gold = set(item.get("gold_evidence_ids") or [])
        if rid and gold:
            qrels[rid] = gold
    print(f"  {len(qrels)} queries with gold evidence")

    splits_path = os.path.join(args.splits_dir, f"{args.split}.jsonl")
    s1_path = os.path.join(args.output_dir, "qa.aug.s1.jsonl")
    s2_path = os.path.join(args.output_dir, "qa.aug.s2.jsonl")
    s3_path = os.path.join(args.output_dir, "qa.aug.s3.jsonl")

    if "s1" in strategies:
        print("\n=== Strategy 1: Rewrite 'neither hit' queries ===")
        targets = build_s1_targets(
            splits_path, qrels, args.naive, args.trace, meta, args.k,
        )
        print(f"  Targets: {len(targets)}")
        if targets:
            run_strategy(targets, s1_generate, s1_path, args.workers, args.model)

    if "s2" in strategies:
        print("\n=== Strategy 2: Cross-table reconciliation queries ===")
        targets = build_s2_targets(qrels, meta)
        print(f"  Targets: {len(targets)}")
        if targets:
            run_strategy(targets, s2_generate, s2_path, args.workers, args.model)

    if "s3" in strategies:
        print("\n=== Strategy 3: Multi-evidence OCI queries ===")
        targets = build_s3_targets(qrels, meta)
        print(f"  Targets: {len(targets)}")
        if targets:
            run_strategy(targets, s3_generate, s3_path, args.workers, args.model)

    print("\n=== Merging outputs ===")
    merge_outputs(
        s1_path, s2_path, s3_path,
        args.qrels,
        splits_path,
        os.path.join(args.output_dir, "gold.qrels.aug.jsonl"),
        os.path.join(args.splits_dir, f"{args.split}.aug.jsonl"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
