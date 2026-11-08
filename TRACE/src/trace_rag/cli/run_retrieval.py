#! /usr/bin/env python3
"""
TRACE retrieval runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tqdm

from trace_rag.utils.experiment_layout import DEFAULT_AUDIT_SPLITS_DIR, validate_non_legacy_output_path
from trace_rag.retrieval import (
    DEFAULT_BATCH_SIZE,
    EMBED_SERVER_URL,
    DEFAULT_LAMBDA_OT,
    DEFAULT_OT_EPS,
    DEFAULT_OT_ITERS,
    DEFAULT_RISK_CACHE_SIZE,
    DEFAULT_RISK_MAX_TOKENS,
    DEFAULT_RISK_MODE,
    DEFAULT_RISK_TEMPERATURE,
    DEFAULT_RISK_TOKEN_SOURCE,
    DEFAULT_RISK_TOP_TOKENS,
    DEFAULT_SCORE_NORM,
    RiskMeasureCache,
    SearchResult,
    embed_texts,
    load_ot_cache,
    load_index,
    load_queries,
    load_risk_matrix_cache,
    meta_text_for_risk,
    rerank_with_cross_encoder,
    risk_terms_signature,
    search,
)


DEFAULT_DATASET_TAG = "audit-ociqa"
DEFAULT_MODEL_TAG = "shared"


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def default_output_path(dataset_tag: str, model_tag: str, method: str) -> str:
    filename = f"{normalize_tag(method, 'trace')}.jsonl"
    return os.path.join("runs", dataset_tag, model_tag, "retrieval", filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE retrieval (semantic coarse search + rerank)")
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
    parser.add_argument("--top-k", type=int, default=20, help="Top-K final results")
    parser.add_argument("--top-sem-k", type=int, default=100, help="Top-K semantic candidates")
    parser.add_argument(
        "--lambda-risk",
        type=float,
        default=0.0,
        help="Proxy risk score weight (0 disables the proxy term)",
    )
    parser.add_argument("--lambda-risk-proxy", type=float, default=None, help="Proxy risk weight")
    parser.add_argument(
        "--lambda-ot",
        type=float,
        default=None,
        help="OT score weight (default: DEFAULT_LAMBDA_OT when enabled)",
    )
    parser.add_argument("--ot-epsilon", type=float, default=DEFAULT_OT_EPS, help="OT epsilon")
    parser.add_argument("--ot-iters", type=int, default=DEFAULT_OT_ITERS, help="OT iterations")
    parser.add_argument("--ot-top-k", type=int, default=None, help="OT scoring candidates")
    parser.add_argument("--no-ot", action="store_true", help="Disable OT scoring")
    parser.add_argument(
        "--risk-mode",
        default=DEFAULT_RISK_MODE,
        choices=["token", "proxy"],
        help="Risk measure mode",
    )
    parser.add_argument(
        "--risk-top-tokens",
        type=int,
        default=DEFAULT_RISK_TOP_TOKENS,
        help="Top risk tokens per text",
    )
    parser.add_argument(
        "--risk-max-tokens",
        type=int,
        default=DEFAULT_RISK_MAX_TOKENS,
        help="Max tokens per text before truncation",
    )
    parser.add_argument(
        "--risk-cache-size",
        type=int,
        default=DEFAULT_RISK_CACHE_SIZE,
        help="Risk measure cache size",
    )
    parser.add_argument(
        "--risk-temperature",
        type=float,
        default=DEFAULT_RISK_TEMPERATURE,
        help="Risk attention temperature",
    )
    parser.add_argument(
        "--risk-token-source",
        default=DEFAULT_RISK_TOKEN_SOURCE,
        choices=["contextual", "word"],
        help="Token source for risk measures (contextual=ColBERT, word=independent)",
    )
    parser.add_argument("--lambda-num", type=float, default=0.0, help="Numeric score weight")
    parser.add_argument(
        "--score-norm",
        default=DEFAULT_SCORE_NORM,
        choices=["minmax", "none"],
        help="Score normalization",
    )
    parser.add_argument(
        "--pref-weights", default=None, help="Preference weights JSON path"
    )
    parser.add_argument("--no-pref", action="store_true", help="Ignore preference weights")
    parser.add_argument(
        "--no-pref-lambda",
        action="store_true",
        help="Do not override lambda_ot from pref weights",
    )
    parser.add_argument(
        "--strict-no-ot",
        action="store_true",
        help="When --no-ot, disable proxy risk fallback for strict ablation",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--server-url", default=EMBED_SERVER_URL, help="Local embedding server URL")
    parser.add_argument("--method", default=None, help="Method name override")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for retrieval")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries (after resume filtering)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: runs/<dataset>/<model>/retrieval/<method>.jsonl)",
    )
    parser.add_argument(
        "--exclude-doc-prefix",
        default=None,
        help="Exclude evidence IDs starting with this prefix (e.g. 'A2023-<doc_id>')",
    )
    parser.add_argument(
        "--reranker-url",
        default=None,
        help="Reranker server URL(s) for cross-encoder reranking after TRACE scoring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_sem_k < args.top_k:
        raise ValueError("top_sem_k must be >= top_k")
    if args.lambda_risk < 0 or args.lambda_num < 0:
        raise ValueError("lambda_risk and lambda_num must be non-negative")
    if args.lambda_ot is not None and args.lambda_ot < 0:
        raise ValueError("lambda_ot must be non-negative")
    if args.lambda_risk_proxy is not None and args.lambda_risk_proxy < 0:
        raise ValueError("lambda_risk_proxy must be non-negative")

    pref_config = None
    risk_anchor = None
    user_set_lambda_ot = args.lambda_ot is not None
    if args.pref_weights and not args.no_pref:
        with open(args.pref_weights, "r", encoding="utf-8") as f:
            pref_config = json.load(f)
        if (
            "lambda_ot" in pref_config
            and not args.no_pref_lambda
            and not user_set_lambda_ot
        ):
            args.lambda_ot = float(pref_config["lambda_ot"])
        if "ot_epsilon" in pref_config:
            args.ot_epsilon = float(pref_config["ot_epsilon"])
        if "ot_iters" in pref_config:
            args.ot_iters = int(pref_config["ot_iters"])
        if "risk_mode" in pref_config:
            args.risk_mode = pref_config["risk_mode"]
        if "risk_top_tokens" in pref_config:
            args.risk_top_tokens = int(pref_config["risk_top_tokens"])
        if "risk_max_tokens" in pref_config:
            args.risk_max_tokens = int(pref_config["risk_max_tokens"])
        if "risk_cache_size" in pref_config:
            args.risk_cache_size = int(pref_config["risk_cache_size"])
        if "risk_temperature" in pref_config:
            args.risk_temperature = float(pref_config["risk_temperature"])
        if "risk_token_source" in pref_config:
            args.risk_token_source = pref_config["risk_token_source"]
        if "risk_anchor" in pref_config:
            risk_anchor = np.asarray(pref_config["risk_anchor"], dtype="float32")

    use_ot = not args.no_ot
    lambda_ot = (
        args.lambda_ot
        if args.lambda_ot is not None
        else (DEFAULT_LAMBDA_OT if use_ot else 0.0)
    )
    lambda_risk_effective = (
        args.lambda_risk_proxy
        if args.lambda_risk_proxy is not None
        else (0.0 if args.strict_no_ot and not use_ot else args.lambda_risk)
    )
    if args.method:
        method = args.method
    else:
        method = "TRACE"
        if args.risk_mode == "proxy":
            method = f"{method}-proxy"
        if not use_ot:
            method = f"{method}-no-ot"
    output_path = args.output or default_output_path(
        dataset_tag=args.dataset_tag,
        model_tag=normalize_tag(args.model_tag, DEFAULT_MODEL_TAG),
        method=method,
    )
    validate_non_legacy_output_path(output_path, label="TRACE retrieval output")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ot_top_k = args.ot_top_k if args.ot_top_k is not None else args.top_k

    index, meta = load_index(args.index_dir)
    risk_matrix = None
    risk_nums = None
    if args.risk_mode == "proxy":
        risk_matrix, risk_nums = load_risk_matrix_cache(args.index_dir, meta)

    risk_cache = None
    if args.risk_mode == "token":
        risk_cache = RiskMeasureCache(
            server_url=args.server_url,
            batch_size=args.batch_size,
            top_tokens=args.risk_top_tokens,
            max_tokens=args.risk_max_tokens,
            temperature=args.risk_temperature,
            cache_size=args.risk_cache_size,
            risk_anchor=risk_anchor,
            token_source=args.risk_token_source,
        )
        needed_size = len(meta) + 1000
        if risk_cache.cache_size < needed_size:
            risk_cache.cache_size = needed_size
        print(f"[Prefill] Pre-warming risk measures for {len(meta)} docs...")
        doc_texts = [meta_text_for_risk(m) for m in meta]
        risk_cache.batch_prefill(doc_texts, desc="Prefill docs")

    ot_cost = None
    ot_kernel = None
    if use_ot and args.risk_mode == "proxy":
        ot_cost, ot_kernel = load_ot_cache(
            args.index_dir, args.server_url, args.batch_size, args.ot_epsilon
        )

    queries = load_queries(args.splits_dir, args.split)
    terms_hash = risk_terms_signature()

    run_config = {
        "index_dir": args.index_dir,
        "split": args.split,
        "top_k": args.top_k,
        "top_sem_k": args.top_sem_k,
        "lambda_risk": args.lambda_risk,
        "lambda_risk_proxy": args.lambda_risk_proxy,
        "lambda_risk_effective": lambda_risk_effective,
        "lambda_ot": lambda_ot,
        "lambda_num": args.lambda_num,
        "ot_epsilon": args.ot_epsilon,
        "ot_iters": args.ot_iters,
        "ot_top_k": ot_top_k,
        "use_ot": use_ot,
        "score_norm": args.score_norm,
        "risk_mode": args.risk_mode,
        "risk_top_tokens": args.risk_top_tokens,
        "risk_max_tokens": args.risk_max_tokens,
        "risk_cache_size": args.risk_cache_size,
        "risk_temperature": args.risk_temperature,
        "risk_token_source": args.risk_token_source,
        "max_queries": args.max_queries,
        "pref_weights": args.pref_weights if args.pref_weights and not args.no_pref else None,
        "no_pref_lambda": args.no_pref_lambda,
        "strict_no_ot": args.strict_no_ot,
        "server_url": args.server_url,
        "terms_hash": terms_hash,
        "method": method,
        "exclude_doc_prefix": getattr(args, "exclude_doc_prefix", None),
    }

    done_ids = set()
    existing_config = None
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if existing_config is None:
                    existing_config = item.get("config")
                done_ids.add(item.get("row_id"))
    if done_ids and existing_config is None:
        raise ValueError("output file exists but has no `config` field; pick a different --output path")
    if existing_config and existing_config != run_config:
        raise ValueError("output file exists with an inconsistent config; pick a different --output path")
    if done_ids:
        print(f"[retrieval] {len(done_ids)} entries already done; skipping")
        queries = [q for q in queries if q.get("row_id") not in done_ids]
    if args.max_queries is not None:
        queries = queries[: args.max_queries]
    if not queries:
        print(f"[retrieval] TRACE retrieval done -> {output_path}")
        print("[retrieval] no pending queries; skipping this run")
        return

    query_texts = [q.get("query_with_meta") or q.get("query", "") for q in queries]
    query_vecs = embed_texts(args.server_url, query_texts, args.batch_size)
    if len(query_vecs) != len(queries):
        raise ValueError("query_vecs count does not match queries")

    items = [
        (q, q_vec) for q, q_vec in zip(queries, query_vecs)
        if q.get("row_id") not in done_ids
    ]

    reranker_url = getattr(args, "reranker_url", None)

    def _process(q_and_vec):
        q, q_vec = q_and_vec
        q_text_raw = q.get("query", "")
        q_text = q.get("query_with_meta") or q_text_raw
        result: SearchResult = search(
            index=index,
            meta=meta,
            query_vec=q_vec.reshape(1, -1),
            query_text=q_text,
            top_sem_k=args.top_sem_k,
            top_k=args.top_k,
            lambda_ot=lambda_ot,
            risk_mode=args.risk_mode,
            lambda_risk=lambda_risk_effective,
            lambda_num=args.lambda_num,
            risk_matrix=risk_matrix,
            risk_nums=risk_nums,
            risk_cache=risk_cache,
            ot_cost=ot_cost,
            ot_kernel=ot_kernel,
            ot_epsilon=args.ot_epsilon,
            ot_iters=args.ot_iters,
            ot_top_k=ot_top_k,
            score_norm=args.score_norm,
            exclude_doc_prefix=getattr(args, "exclude_doc_prefix", None),
        )
        final_ids = result.evidence_ids
        final_scores = result.combined_scores
        if reranker_url and result.coarse_evidence_ids:
            coarse_eids = result.coarse_evidence_ids
            eid_to_midx = {}
            for midx, m in enumerate(meta):
                eid = m.get("evidence_id")
                if eid and eid not in eid_to_midx:
                    eid_to_midx[eid] = midx
            matched_eids = []
            matched_meta_idxs = []
            for eid in coarse_eids:
                if eid in eid_to_midx:
                    matched_eids.append(eid)
                    matched_meta_idxs.append(eid_to_midx[eid])
            doc_texts = [meta_text_for_risk(meta[midx]) for midx in matched_meta_idxs]
            ce_ranked = rerank_with_cross_encoder(reranker_url, q_text_raw, doc_texts, top_k=args.top_k)
            n = len(matched_eids)
            final_ids = [matched_eids[orig_idx] for orig_idx, _ in ce_ranked if orig_idx < n]
            final_scores = [score for orig_idx, score in ce_ranked if orig_idx < n]
        return {
            "row_id": q.get("row_id"),
            "qid": q.get("qid"),
            "query": q_text_raw,
            "method": method,
            "coarse_topk_evidence_ids": result.coarse_evidence_ids,
            "coarse_sem_scores": result.coarse_sem_scores,
            "topk_evidence_ids": final_ids,
            "scores": final_scores,
            "fused_scores": result.combined_scores,
            "sem_scores": result.sem_scores,
            "ot_scores": result.ot_scores,
            "risk_scores": result.risk_scores,
            "num_scores": result.num_scores,
            "s_sem": result.sem_scores,
            "s_ot": result.ot_scores,
            "s_risk": result.risk_scores,
            "s_num": result.num_scores,
            "s_sem_norm": result.sem_norm_scores,
            "s_ot_norm": result.ot_norm_scores,
            "s_risk_norm": result.risk_norm_scores,
            "s_num_norm": result.num_norm_scores,
            "lambda_risk": args.lambda_risk,
            "lambda_risk_proxy": args.lambda_risk_proxy,
            "lambda_risk_effective": lambda_risk_effective,
            "lambda_ot": lambda_ot,
            "lambda_num": args.lambda_num,
            "top_sem_k": args.top_sem_k,
            "top_k": args.top_k,
            "config": run_config,
        }

    workers = max(1, args.workers)
    write_lock = threading.Lock()
    with open(output_path, "a", encoding="utf-8") as f:
        if workers <= 1:
            for item in tqdm.tqdm(items, desc="TRACE"):
                out = _process(item)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process, it): it for it in items}
                for future in tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc="TRACE"
                ):
                    out = future.result()
                    with write_lock:
                        f.write(json.dumps(out, ensure_ascii=False) + "\n")
                        f.flush()

    print(f"[retrieval] TRACE retrieval done -> {output_path}")


if __name__ == "__main__":
    main()
