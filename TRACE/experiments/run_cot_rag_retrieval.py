#! /usr/bin/env python3
"""
Run CoT-RAG retrieval (dense recall + ColBERT-style rerank).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple

import numpy as np
import tqdm

from trace_rag.utils.experiment_layout import DEFAULT_AUDIT_SPLITS_DIR, validate_non_legacy_output_path
from trace_rag.retrieval import (
    DEFAULT_BATCH_SIZE,
    EMBED_SERVER_URL,
    embed_texts,
    embed_tokens_contextual,
    load_index,
    load_queries,
    meta_text_for_risk,
    normalize_meta_item,
    rerank_with_cross_encoder,
)


DEFAULT_DATASET_TAG = "audit-ociqa"
DEFAULT_MODEL_TAG = "shared"
DEFAULT_METHOD_LABEL = "CoT-RAG"
DEFAULT_TOP_SEM_K = 50
DEFAULT_RERANK_BATCH_SIZE = 8


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def default_output_path(dataset_tag: str, model_tag: str, method: str) -> str:
    filename = f"{normalize_tag(method, 'cot-rag')}.jsonl"
    return os.path.join("runs", dataset_tag, model_tag, "retrieval", filename)


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def file_stat_signature(path: str) -> Dict | None:
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def split_signature(splits_dir: str, split: str) -> Dict[str, Dict | None]:
    if split == "all":
        filenames = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    else:
        filenames = [f"{split}.jsonl"]
    return {
        filename: file_stat_signature(os.path.join(splits_dir, filename))
        for filename in filenames
    }


def index_signature(index_dir: str) -> Dict[str, Dict | None]:
    return {
        filename: file_stat_signature(os.path.join(index_dir, filename))
        for filename in ("index.faiss", "meta.jsonl")
    }


def build_run_config(args: argparse.Namespace) -> Dict:
    return {
        "index_dir": os.path.abspath(args.index_dir),
        "index_signature": index_signature(args.index_dir),
        "splits_dir": os.path.abspath(args.splits_dir),
        "split": args.split,
        "split_signature": split_signature(args.splits_dir, args.split),
        "method": normalize_tag(args.method, "cot-rag"),
        "method_label": args.method,
        "top_k": args.top_k,
        "top_sem_k": args.top_sem_k,
        "batch_size": args.batch_size,
        "rerank_batch_size": args.rerank_batch_size,
        "server_url": args.server_url,
        "reranker": "colbert-maxsim",
    }


def late_interaction_score(query_tokens: np.ndarray, doc_tokens: np.ndarray) -> float:
    if query_tokens.size == 0 or doc_tokens.size == 0:
        return 0.0
    scores = query_tokens @ doc_tokens.T
    return float(scores.max(axis=1).mean())


def coarse_search(
    index,
    meta: List[Dict],
    query_vec: np.ndarray,
    top_sem_k: int,
    exclude_doc_prefix: str | None = None,
) -> Tuple[List[int], List[str], List[float]]:
    fetch_k = top_sem_k * 3 if exclude_doc_prefix else top_sem_k
    fetch_k = min(fetch_k, index.ntotal)
    scores, idxs = index.search(query_vec.reshape(1, -1), fetch_k)
    raw_idx_list = [int(idx) for idx in idxs[0] if idx != -1]
    raw_sem_scores = [float(score) for score in scores[0][: len(raw_idx_list)]]
    if exclude_doc_prefix:
        filtered = [
            (idx, s)
            for idx, s in zip(raw_idx_list, raw_sem_scores)
            if not meta[idx].get("evidence_id", "").startswith(exclude_doc_prefix)
        ]
        idx_list = [idx for idx, _ in filtered[:top_sem_k]]
        sem_scores = [s for _, s in filtered[:top_sem_k]]
    else:
        idx_list = raw_idx_list
        sem_scores = raw_sem_scores
    evidence_ids = [meta[idx].get("evidence_id") for idx in idx_list]
    return idx_list, evidence_ids, sem_scores


def rerank_candidates(
    server_url: str,
    query_text: str,
    candidate_items: List[Dict],
    rerank_batch_size: int,
) -> List[float]:
    if not candidate_items:
        return []
    texts = [query_text, *[meta_text_for_risk(item) for item in candidate_items]]
    token_vecs = embed_tokens_contextual(
        server_url,
        texts,
        batch_size=max(1, rerank_batch_size),
    )
    query_tokens = token_vecs[0]
    doc_tokens = token_vecs[1:]
    return [late_interaction_score(query_tokens, doc_vecs) for doc_vecs in doc_tokens]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoT-RAG retrieval (dense recall + ColBERT rerank)")
    parser.add_argument("--index-dir", default="runs/index", help="FAISS index directory")
    parser.add_argument("--splits-dir", default=DEFAULT_AUDIT_SPLITS_DIR, help="Splits directory")
    parser.add_argument(
        "--dataset-tag",
        default=DEFAULT_DATASET_TAG,
        help="Dataset tag for default output path",
    )
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help="Model tag for default output path when --output is omitted",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to run (e.g. train, dev, test, test-mini, all)",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-K final retrieval")
    parser.add_argument(
        "--top-sem-k",
        type=int,
        default=DEFAULT_TOP_SEM_K,
        help="Top-K semantic candidates before rerank",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Dense embedding batch size")
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=DEFAULT_RERANK_BATCH_SIZE,
        help="Batch size for contextual token rerank encoding",
    )
    parser.add_argument(
        "--server-url",
        default=EMBED_SERVER_URL,
        help="Local embedding server URL (also used for /v1/colbert)",
    )
    parser.add_argument("--method", default=DEFAULT_METHOD_LABEL, help="Method display name")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: runs/<dataset>/<model>/retrieval/cot-rag.jsonl)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output by skipping finished row_id",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries (after resume filtering)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers")
    parser.add_argument(
        "--exclude-doc-prefix",
        default=None,
        help="Exclude evidence IDs starting with this prefix (e.g. 'A2023-<doc_id>')",
    )
    parser.add_argument(
        "--reranker-url",
        default=None,
        help="Reranker server URL(s) for cross-encoder reranking (overrides ColBERT rerank)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_sem_k < args.top_k:
        raise ValueError("top_sem_k must be >= top_k")

    output_path = args.output or default_output_path(
        args.dataset_tag,
        normalize_tag(args.model_tag, DEFAULT_MODEL_TAG),
        args.method,
    )
    validate_non_legacy_output_path(output_path, label="CoT-RAG retrieval output")
    run_config = build_run_config(args)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    index, raw_meta = load_index(args.index_dir)
    meta = [normalize_meta_item(dict(item)) for item in raw_meta]
    queries = load_queries(args.splits_dir, args.split)

    done_ids = set()
    existing_config = None
    if args.resume and os.path.exists(output_path):
        for item in read_jsonl(output_path):
            if existing_config is None:
                existing_config = item.get("config")
            row_id = item.get("row_id")
            if row_id:
                done_ids.add(row_id)
    if done_ids and existing_config is None:
        raise ValueError("output file exists but has no `config` field; pick a different --output path")
    if existing_config and existing_config != run_config:
        raise ValueError("output file exists with an inconsistent config; pick a different --output path")
    if done_ids:
        queries = [query for query in queries if query.get("row_id") not in done_ids]
        print(f"Resume enabled: skip {len(done_ids)} rows")
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        print(f"[retrieval] {args.method} done -> {output_path}")
        print("[retrieval] no pending queries; skipping this run")
        return

    query_texts = [query.get("query", "") for query in queries]
    query_vecs = embed_texts(args.server_url, query_texts, batch_size=max(1, args.batch_size))

    exclude_doc_prefix = getattr(args, "exclude_doc_prefix", None)

    reranker_url = getattr(args, "reranker_url", None)

    def _process(payload: Tuple[Dict, np.ndarray]) -> Dict:
        query_row, query_vec = payload
        query_text = query_row.get("query", "")
        idx_list, coarse_ids, coarse_scores = coarse_search(
            index, meta, query_vec, args.top_sem_k, exclude_doc_prefix=exclude_doc_prefix,
        )
        candidate_items = [meta[idx] for idx in idx_list]
        if reranker_url:
            doc_texts = [meta_text_for_risk(item) for item in candidate_items]
            ce_ranked = rerank_with_cross_encoder(reranker_url, query_text, doc_texts, top_k=args.top_k)
            n = len(coarse_ids)
            final_ids = [coarse_ids[orig_idx] for orig_idx, _ in ce_ranked if orig_idx < n]
            final_scores = [score for orig_idx, score in ce_ranked if orig_idx < n]
            return {
                "row_id": query_row.get("row_id"),
                "qid": query_row.get("qid"),
                "query": query_text,
                "method": args.method,
                "retrieval_query": query_text,
                "coarse_topk_evidence_ids": coarse_ids,
                "coarse_scores": coarse_scores,
                "topk_evidence_ids": final_ids,
                "scores": final_scores,
                "config": run_config,
            }
        rerank_scores = rerank_candidates(
            args.server_url,
            query_text=query_text,
            candidate_items=candidate_items,
            rerank_batch_size=args.rerank_batch_size,
        )
        ranked = sorted(
            [
                {
                    "idx": idx,
                    "evidence_id": meta[idx].get("evidence_id"),
                    "coarse_score": coarse_score,
                    "rerank_score": rerank_score,
                }
                for idx, coarse_score, rerank_score in zip(idx_list, coarse_scores, rerank_scores)
            ],
            key=lambda item: (item["rerank_score"], item["coarse_score"]),
            reverse=True,
        )
        final_ranked = ranked[: args.top_k]
        return {
            "row_id": query_row.get("row_id"),
            "qid": query_row.get("qid"),
            "query": query_text,
            "method": args.method,
            "retrieval_query": query_text,
            "coarse_topk_evidence_ids": coarse_ids,
            "coarse_scores": coarse_scores,
            "topk_evidence_ids": [item["evidence_id"] for item in final_ranked if item.get("evidence_id")],
            "scores": [float(item["rerank_score"]) for item in final_ranked],
            "rerank_scores": [float(item["rerank_score"]) for item in ranked],
            "config": run_config,
        }

    mode = "a" if args.resume else "w"
    workers = max(1, args.workers)
    with open(output_path, mode, encoding="utf-8") as f:
        if workers <= 1:
            for payload in tqdm.tqdm(zip(queries, query_vecs), total=len(queries), desc=args.method):
                f.write(json.dumps(_process(payload), ensure_ascii=False) + "\n")
                f.flush()
        else:
            write_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_process, payload): payload[0]
                    for payload in zip(queries, query_vecs)
                }
                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=args.method):
                    with write_lock:
                        f.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                        f.flush()

    print(f"[retrieval] {args.method} done -> {output_path}")


if __name__ == "__main__":
    main()
