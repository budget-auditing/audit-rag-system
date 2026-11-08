#! /usr/bin/env python3
"""Baseline retrievers: BM25 / Naive Dense / HyDE."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple
from urllib import request

import faiss
import numpy as np
import tqdm

from trace_rag.utils.experiment_layout import DEFAULT_AUDIT_SPLITS_DIR, validate_non_legacy_output_path
from trace_rag.retrieval import (
    embed_texts as _shared_embed_texts,
    load_queries as _shared_load_queries,
    meta_text_for_risk,
    normalize_meta_item,
    rerank_with_cross_encoder,
)


DEFAULT_BATCH_SIZE = 64
DEFAULT_DATASET_TAG = "audit-ociqa"
DEFAULT_MODEL_TAG = "shared"
EMBED_SERVER_URL = os.getenv(
    "EMBED_SERVER_URL",
    "http://localhost:8081/v1/embeddings",
)

CHAT_BASE_URL = os.getenv("CHAT_URL", "http://localhost:8000/v1/chat/completions")
DEFAULT_HYDE_MODEL = os.getenv(
    "AUDIT_OCIQA_MODEL_HYDE",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
)
METHOD_LABELS = {
    "bm25": "BM25-RAG",
    "naive": "Naive Dense RAG",
    "hyde": "HyDE-RAG",
}


def _read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _embed_texts(
    server_url: str,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    return _shared_embed_texts(server_url, texts, batch_size)


def _normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def _default_output_path(dataset_tag: str, model_tag: str, method: str) -> str:
    filename = f"{_normalize_tag(METHOD_LABELS.get(method, method), method)}.jsonl"
    return os.path.join("runs", dataset_tag, model_tag, "retrieval", filename)


def _file_stat_signature(path: str) -> Dict | None:
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _split_signature(splits_dir: str, split: str) -> Dict[str, Dict | None]:
    if split == "all":
        filenames = ["train.jsonl", "dev.jsonl", "test.jsonl"]
    else:
        filenames = [f"{split}.jsonl"]
    return {
        filename: _file_stat_signature(os.path.join(splits_dir, filename))
        for filename in filenames
    }


def _index_signature(index_dir: str) -> Dict[str, Dict | None]:
    return {
        filename: _file_stat_signature(os.path.join(index_dir, filename))
        for filename in ("index.faiss", "meta.jsonl")
    }


def _build_run_config(args: argparse.Namespace) -> Dict:
    dense_method = args.method in {"naive", "hyde"}
    cfg: Dict = {
        "index_dir": os.path.abspath(args.index_dir),
        "index_signature": _index_signature(args.index_dir),
        "splits_dir": os.path.abspath(args.splits_dir),
        "split": args.split,
        "split_signature": _split_signature(args.splits_dir, args.split),
        "method": args.method,
        "method_label": METHOD_LABELS.get(args.method, args.method),
        "top_k": args.top_k,
        "batch_size": args.batch_size if dense_method else None,
        "server_url": args.server_url if dense_method else None,
        "hyde_model": args.model if args.method == "hyde" else None,
    }
    exclude_doc_prefix = getattr(args, "exclude_doc_prefix", None)
    if exclude_doc_prefix:
        cfg["exclude_doc_prefix"] = exclude_doc_prefix
    return cfg


def _detect_lang(text: str) -> str:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk > len(text) * 0.1 else "en"


def _call_chat_api(
    query: str,
    model: str,
    max_retries: int = 3,
    dataset_tag: str = DEFAULT_DATASET_TAG,
) -> str:
    lang = _detect_lang(query)
    if lang == "zh":
        if dataset_tag == "finqa":
            hyde_prompt = (
                "请基于以下金融问答生成一段适合检索的“假设性答案/摘要”，"
                "用于扩大检索召回。要求：简洁、包含关键实体/指标/年份，"
                "不编造具体数值。\n\n问题："
                f"{query}"
            )
        else:
            hyde_prompt = (
                "请基于以下审计问题生成一段适合检索的“假设性答案/摘要”，"
                "用于扩大检索召回。要求：简洁、包含关键实体/指标/年份，"
                "不编造具体数值。\n\n问题："
                f"{query}"
            )
    else:
        hyde_prompt = (
            "Generate a short hypothetical answer/summary for the following "
            "financial question, suitable for retrieval augmentation. "
            "Be concise, include key entities/metrics/years, "
            "do not fabricate specific numbers.\n\n"
            f"Question: {query}"
        )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": hyde_prompt,
            }
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "max_tokens": 512,
        "presence_penalty": 1.5,
        
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
            with request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
                content = data["choices"][0]["message"]["content"]
                return content.strip()
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(f"HyDE generation failed after {max_retries} retries.") from exc
            time.sleep(1.5 * attempt)
    return ""


def _cpu_to_multi_gpu(index: faiss.Index) -> faiss.Index:
    """Shard a CPU FAISS index across all visible GPUs (select them with CUDA_VISIBLE_DEVICES)."""
    usable = list(range(faiss.get_num_gpus()))
    if not usable:
        print("[faiss-gpu] no visible GPUs; falling back to CPU")
        return index
    resources = [faiss.StandardGpuResources() for _ in usable]
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    gpu_index = faiss.index_cpu_to_gpu_multiple_py(resources, index, co, usable)
    print(f"[faiss-gpu] index loaded onto GPUs {usable} (shard={co.shard})")
    return gpu_index


def _load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    meta_path = os.path.join(index_dir, "meta.jsonl")
    meta = list(_read_jsonl(meta_path))
    if index.ntotal != len(meta):
        raise ValueError(
            f"index vector count does not match meta: {index.ntotal} vs {len(meta)}"
        )
    index = _cpu_to_multi_gpu(index)
    return index, meta


def _load_queries(splits_dir: str, split: str) -> List[Dict]:
    return _shared_load_queries(splits_dir, split)


def _search(
    index: faiss.Index,
    meta: List[Dict],
    query_vecs: np.ndarray,
    top_k: int,
    exclude_doc_prefix: str | None = None,
) -> Tuple[List[List[str]], List[List[float]], List[List[int]]]:
    fetch_k = top_k * 3 if exclude_doc_prefix else top_k
    fetch_k = min(fetch_k, index.ntotal)
    scores, idxs = index.search(query_vecs, fetch_k)
    all_ids: List[List[str]] = []
    all_scores: List[List[float]] = []
    all_meta_idxs: List[List[int]] = []
    for row_idxs, row_scores in zip(idxs, scores):
        ids: List[str] = []
        scs: List[float] = []
        midxs: List[int] = []
        for i, s in zip(row_idxs, row_scores):
            if i == -1:
                continue
            eid = meta[int(i)].get("evidence_id", "")
            if exclude_doc_prefix and eid.startswith(exclude_doc_prefix):
                continue
            ids.append(eid)
            scs.append(float(s))
            midxs.append(int(i))
            if len(ids) >= top_k:
                break
        all_ids.append(ids)
        all_scores.append(scs)
        all_meta_idxs.append(midxs)
    return all_ids, all_scores, all_meta_idxs


def _rerank_results(
    reranker_url: str,
    meta: List[Dict],
    query_text: str,
    coarse_ids: List[str],
    coarse_scores: List[float],
    coarse_meta_idxs: List[int],
    top_k: int,
) -> Tuple[List[str], List[float]]:
    """Rerank coarse retrieval results using cross-encoder reranker."""
    if not coarse_ids:
        return [], []
    doc_texts = [meta_text_for_risk(meta[idx]) for idx in coarse_meta_idxs]
    ranked = rerank_with_cross_encoder(reranker_url, query_text, doc_texts, top_k=top_k)
    n = len(coarse_ids)
    final_ids = [coarse_ids[orig_idx] for orig_idx, _ in ranked if orig_idx < n]
    final_scores = [score for orig_idx, score in ranked if orig_idx < n]
    return final_ids, final_scores


def _tokenize(text: str) -> List[str]:
    return [token.group(0).lower() for token in re.finditer(r"[\u4e00-\u9fff]|[a-zA-Z0-9_.%-]+", text or "")]


def _build_bm25(meta: List[Dict]) -> Tuple[Dict[int, str], Dict[str, List[Tuple[int, int]]], Dict[int, int], Dict[str, float], float]:
    doc_ids: Dict[int, str] = {}
    postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    doc_lens: Dict[int, int] = {}
    doc_freq: Counter[str] = Counter()
    for doc_idx, raw_item in enumerate(meta):
        item = normalize_meta_item(dict(raw_item))
        evidence_id = item.get("evidence_id")
        if not evidence_id:
            continue
        text = meta_text_for_risk(item)
        tokens = _tokenize(text)
        doc_ids[doc_idx] = evidence_id
        doc_lens[doc_idx] = len(tokens)
        if not tokens:
            continue
        tf = Counter(tokens)
        for term, freq in tf.items():
            postings[term].append((doc_idx, freq))
            doc_freq[term] += 1
    num_docs = max(len(doc_ids), 1)
    avgdl = sum(doc_lens.values()) / max(len(doc_lens), 1)
    idf = {
        term: float(np.log(1.0 + (num_docs - df + 0.5) / (df + 0.5)))
        for term, df in doc_freq.items()
    }
    return doc_ids, postings, doc_lens, idf, avgdl


def run_bm25(
    meta: List[Dict],
    queries: List[Dict],
    top_k: int,
    output_path: str,
    run_config: Dict,
    mode: str = "w",
    exclude_doc_prefix: str | None = None,
):
    k1 = 1.5
    b = 0.75
    doc_ids, postings, doc_lens, idf, avgdl = _build_bm25(meta)
    with open(output_path, mode, encoding="utf-8") as f:
        for q in tqdm.tqdm(queries, desc="BM25"):
            query = q.get("query_with_meta") or q.get("query", "")
            scores: Dict[int, float] = defaultdict(float)
            for term in _tokenize(query):
                if term not in postings:
                    continue
                term_idf = idf.get(term, 0.0)
                for doc_idx, tf in postings[term]:
                    dl = doc_lens.get(doc_idx, 0)
                    denom = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1e-6))
                    scores[doc_idx] += term_idf * (tf * (k1 + 1.0)) / max(denom, 1e-6)
            ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
            evidence_ids: List[str] = []
            evidence_scores: List[float] = []
            for idx, score in ranked:
                if idx not in doc_ids:
                    continue
                eid = doc_ids[idx]
                if exclude_doc_prefix and eid.startswith(exclude_doc_prefix):
                    continue
                evidence_ids.append(eid)
                evidence_scores.append(float(score))
                if len(evidence_ids) >= top_k:
                    break
            f.write(
                json.dumps(
                    {
                        "row_id": q.get("row_id"),
                        "qid": q.get("qid"),
                        "query": q.get("query"),
                        "method": METHOD_LABELS["bm25"],
                        "topk_evidence_ids": evidence_ids,
                        "scores": evidence_scores,
                        "config": run_config,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_naive(
    server_url: str,
    index: faiss.Index,
    meta: List[Dict],
    queries: List[Dict],
    top_k: int,
    batch_size: int,
    output_path: str,
    run_config: Dict,
    mode: str = "w",
    exclude_doc_prefix: str | None = None,
    reranker_url: str | None = None,
    coarse_top_k: int | None = None,
):
    retrieval_k = coarse_top_k if reranker_url and coarse_top_k else top_k
    query_texts = [q.get("query", "") for q in queries]
    vecs = _embed_texts(server_url, query_texts, batch_size)
    all_ids, all_scores, all_meta_idxs = _search(index, meta, vecs, retrieval_k, exclude_doc_prefix=exclude_doc_prefix)

    with open(output_path, mode, encoding="utf-8") as f:
        for q, ids, scores, midxs in tqdm.tqdm(
            zip(queries, all_ids, all_scores, all_meta_idxs), total=len(queries), desc="Naive+Rerank"
        ):
            if reranker_url:
                ids, scores = _rerank_results(
                    reranker_url, meta, q.get("query", ""), ids, scores, midxs, top_k
                )
            f.write(
                json.dumps(
                    {
                        "row_id": q.get("row_id"),
                        "qid": q.get("qid"),
                        "query": q.get("query"),
                        "method": METHOD_LABELS["naive"],
                        "topk_evidence_ids": ids,
                        "scores": scores,
                        "config": run_config,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_hyde(
    server_url: str,
    index: faiss.Index,
    meta: List[Dict],
    queries: List[Dict],
    top_k: int,
    model: str,
    output_path: str,
    run_config: Dict,
    dataset_tag: str,
    workers: int = 1,
    mode: str = "w",
    exclude_doc_prefix: str | None = None,
    reranker_url: str | None = None,
    coarse_top_k: int | None = None,
):
    retrieval_k = coarse_top_k if reranker_url and coarse_top_k else top_k

    def _process(q: Dict) -> Dict:
        query = q.get("query", "")
        hypo = _call_chat_api(query, model=model, dataset_tag=dataset_tag)
        vecs = _embed_texts(server_url, [hypo], batch_size=1)
        all_ids, all_scores, all_meta_idxs = _search(index, meta, vecs, retrieval_k, exclude_doc_prefix=exclude_doc_prefix)
        ids, scores = all_ids[0], all_scores[0]
        if reranker_url:
            ids, scores = _rerank_results(
                reranker_url, meta, query, ids, scores, all_meta_idxs[0], top_k
            )
        return {
            "row_id": q.get("row_id"),
            "qid": q.get("qid"),
            "query": query,
            "method": METHOD_LABELS["hyde"],
            "hyde": hypo,
            "topk_evidence_ids": ids,
            "scores": scores,
            "config": run_config,
        }

    with open(output_path, mode, encoding="utf-8") as f:
        if max(1, workers) <= 1:
            for q in tqdm.tqdm(queries, desc="HyDE"):
                f.write(json.dumps(_process(q), ensure_ascii=False) + "\n")
                f.flush()
        else:
            write_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_process, q): q for q in queries}
                for future in tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc="HyDE"
                ):
                    with write_lock:
                        f.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                        f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval baselines (BM25 / naive / hyde)")
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
    parser.add_argument(
        "--method",
        default="naive",
        choices=["bm25", "naive", "hyde"],
        help="Baseline method",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_HYDE_MODEL,
        help="LLM model name used by HyDE query generation",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-K retrieval")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--server-url",
        default=EMBED_SERVER_URL,
        help="Local embedding server URL",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: runs/<dataset>/<model>/retrieval/<method>.jsonl)",
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
        help="Reranker server URL(s) for two-stage retrieval (e.g. http://localhost:8085/v1/rerank,...)",
    )
    parser.add_argument(
        "--coarse-top-k",
        type=int,
        default=100,
        help="Number of coarse candidates before reranking (default: 100)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or _default_output_path(args.dataset_tag, args.model_tag, args.method)
    validate_non_legacy_output_path(output_path, label="retrieval output")
    run_config = _build_run_config(args)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    index, meta = _load_index(args.index_dir)
    queries = _load_queries(args.splits_dir, args.split)
    done_ids = set()
    existing_config = None
    if args.resume and os.path.exists(output_path):
        for item in _read_jsonl(output_path):
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
        queries = [q for q in queries if q.get("row_id") not in done_ids]
        print(f"Resume enabled: skip {len(done_ids)} rows")
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        print(
            f"[retrieval] {METHOD_LABELS.get(args.method, args.method)} done -> {output_path}"
        )
        print("[retrieval] no pending queries; skipping this run")
        return

    mode = "a" if args.resume else "w"

    exclude_doc_prefix = args.exclude_doc_prefix

    if args.method == "bm25":
        run_bm25(
            meta=meta,
            queries=queries,
            top_k=args.top_k,
            output_path=output_path,
            run_config=run_config,
            mode=mode,
            exclude_doc_prefix=exclude_doc_prefix,
        )
    elif args.method == "naive":
        run_naive(
            server_url=args.server_url,
            index=index,
            meta=meta,
            queries=queries,
            top_k=args.top_k,
            batch_size=args.batch_size,
            output_path=output_path,
            run_config=run_config,
            mode=mode,
            exclude_doc_prefix=exclude_doc_prefix,
            reranker_url=args.reranker_url,
            coarse_top_k=args.coarse_top_k,
        )
    else:
        run_hyde(
            server_url=args.server_url,
            index=index,
            meta=meta,
            queries=queries,
            top_k=args.top_k,
            model=args.model,
            output_path=output_path,
            run_config=run_config,
            dataset_tag=args.dataset_tag,
            workers=args.workers,
            mode=mode,
            exclude_doc_prefix=exclude_doc_prefix,
            reranker_url=args.reranker_url,
            coarse_top_k=args.coarse_top_k,
        )

    print(
        f"[retrieval] {METHOD_LABELS.get(args.method, args.method)} done -> {output_path}"
    )


if __name__ == "__main__":
    main()
