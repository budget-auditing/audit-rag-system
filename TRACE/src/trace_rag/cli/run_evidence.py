#! /usr/bin/env python3
"""
TRACE evidence chain runner.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import tqdm

from trace_rag.utils.experiment_layout import DEFAULT_AUDIT_SPLITS_DIR, validate_non_legacy_output_path
from trace_rag.evidence import (
    DEFAULT_DELTA_C,
    build_chains_for_queries,
    build_chains_with_resources,
    prepare_chain_resources,
)
from trace_rag.retrieval import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_SCORE_NORM,
    DEFAULT_OT_EPS,
    DEFAULT_OT_ITERS,
    DEFAULT_LAMBDA_OT,
    DEFAULT_RISK_CACHE_SIZE,
    DEFAULT_RISK_MAX_TOKENS,
    DEFAULT_RISK_MODE,
    DEFAULT_RISK_TEMPERATURE,
    DEFAULT_RISK_TOKEN_SOURCE,
    DEFAULT_RISK_TOP_TOKENS,
    EMBED_SERVER_URL,
    load_queries,
    risk_terms_signature,
)


DEFAULT_DATASET_TAG = "audit-ociqa"
DEFAULT_MODEL_TAG = "shared"
_SHARED_CHAIN_RESOURCES: Dict[str, Any] | None = None


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def default_output_path(dataset_tag: str, model_tag: str, method: str) -> str:
    filename = f"{normalize_tag(method, 'trace')}.chains.jsonl"
    return os.path.join("runs", dataset_tag, model_tag, "evidence", filename)


def _default_evidence_workers() -> int:
    env = os.getenv("EVIDENCE_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except Exception:
            pass
    cores = max(1, os.cpu_count() or 1)
    if cores <= 4:
        return cores
    # Default to ~75% cores with a practical upper bound.
    return max(4, min(48, int(round(cores * 0.75))))


def _chunk_queries(queries: List[Dict], chunk_size: int) -> List[List[Dict]]:
    size = max(1, int(chunk_size))
    return [queries[i : i + size] for i in range(0, len(queries), size)]


def _resolve_chunk_size(num_queries: int, workers: int, requested_chunk_size: int) -> int:
    if requested_chunk_size and requested_chunk_size > 0:
        return max(1, int(requested_chunk_size))
    # Aim for at least ~2 chunks per worker so CPU stays busy
    target_chunks = max(1, workers * 2)
    auto_size = (max(1, num_queries) + target_chunks - 1) // target_chunks
    return max(8, min(32, auto_size))


def _build_chunk(task: Tuple[int, List[Dict], Dict]) -> Tuple[int, List[Dict]]:
    chunk_idx, chunk_queries, kwargs = task
    if _SHARED_CHAIN_RESOURCES is not None:
        chunk_outputs = build_chains_with_resources(chunk_queries, _SHARED_CHAIN_RESOURCES)
    else:
        chunk_outputs = build_chains_for_queries(
            queries=chunk_queries,
            prefill_risk_cache=False,
            precompute_role_vectors=False,
            **kwargs,
        )
    return chunk_idx, chunk_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TRACE evidence chains")
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
    parser.add_argument("--top-m", type=int, default=50, help="Top-M candidates per role")
    parser.add_argument(
        "--top-sem-k",
        type=int,
        default=None,
        help="Top-K semantic pool before OT rerank",
    )
    parser.add_argument("--beam-width", type=int, default=10, help="Beam width")
    parser.add_argument("--eta", type=float, default=0.3, help="Link weight")
    parser.add_argument("--delta-c", type=float, default=DEFAULT_DELTA_C, help="Evidence chain fallback threshold")
    parser.add_argument(
        "--lambda-ot",
        type=float,
        default=None,
        help="OT weight for pool rerank (default: DEFAULT_LAMBDA_OT when enabled)",
    )
    parser.add_argument("--lambda-risk", type=float, default=0.0, help="Risk-term score weight")
    parser.add_argument("--lambda-num", type=float, default=0.0, help="Numeric score weight")
    parser.add_argument(
        "--score-norm",
        default=DEFAULT_SCORE_NORM,
        choices=["minmax", "none"],
        help="Score normalization for role candidate pools",
    )
    parser.add_argument("--ot-epsilon", type=float, default=DEFAULT_OT_EPS, help="OT epsilon")
    parser.add_argument("--ot-iters", type=int, default=DEFAULT_OT_ITERS, help="OT iterations")
    parser.add_argument("--no-ot", action="store_true", help="Disable OT coverage/link")
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
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage optimization")
    parser.add_argument("--no-role", action="store_true", help="Disable role conditioning")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--server-url", default=EMBED_SERVER_URL, help="Local embedding server URL")
    parser.add_argument("--method", default=None, help="Method name override")
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
        "--output",
        default=None,
        help="Output JSONL path (default: runs/<dataset>/<model>/evidence/<method>.chains.jsonl)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries (after resume filtering)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_evidence_workers(),
        help="Parallel workers for evidence query chunks",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Queries per chunk when workers > 1 (0 = auto)",
    )
    parser.add_argument(
        "--flat-topk",
        type=int,
        default=0,
        help="If >0, add top-K candidates as extra_evidence (bypass OCI-only structure)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_queries(args.splits_dir, args.split)

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
            risk_anchor = pref_config["risk_anchor"]

    use_ot = not args.no_ot
    lambda_ot = (
        args.lambda_ot
        if args.lambda_ot is not None
        else (DEFAULT_LAMBDA_OT if use_ot else 0.0)
    )
    use_role = not args.no_role
    use_coverage = not args.no_coverage
    top_sem_k = None if args.top_sem_k in (None, 0) else args.top_sem_k
    if top_sem_k is not None and top_sem_k < args.top_m:
        top_sem_k = args.top_m
    if args.method:
        method = args.method
    else:
        method = "TRACE"
        if args.risk_mode == "proxy":
            method = f"{method}-proxy"
        if not use_ot:
            method = f"{method}-no-ot"
        if not use_role:
            method = f"{method}-no-role"
        if not use_coverage:
            method = f"{method}-no-coverage"
    output_path = args.output or default_output_path(
        dataset_tag=args.dataset_tag,
        model_tag=normalize_tag(args.model_tag, DEFAULT_MODEL_TAG),
        method=method,
    )
    validate_non_legacy_output_path(output_path, label="TRACE evidence output")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    run_config = {
        "index_dir": args.index_dir,
        "top_m": args.top_m,
        "top_sem_k": top_sem_k,
        "beam_width": args.beam_width,
        "eta": args.eta,
        "ot_epsilon": args.ot_epsilon,
        "ot_iters": args.ot_iters,
        "use_ot": use_ot,
        "risk_mode": args.risk_mode,
        "risk_top_tokens": args.risk_top_tokens,
        "risk_max_tokens": args.risk_max_tokens,
        "risk_cache_size": args.risk_cache_size,
        "risk_temperature": args.risk_temperature,
        "risk_token_source": args.risk_token_source,
        "lambda_risk": args.lambda_risk,
        "lambda_num": args.lambda_num,
        "score_norm": args.score_norm,
        "use_role": use_role,
        "use_coverage": use_coverage,
        "lambda_ot": lambda_ot,
        "delta_c": args.delta_c,
        "server_url": args.server_url,
        "terms_hash": risk_terms_signature(),
        "method": method,
        "pref_weights": args.pref_weights if args.pref_weights and not args.no_pref else None,
        "no_pref_lambda": args.no_pref_lambda,
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
        print(f"[evidence] {len(done_ids)} entries already done; skipping")
        queries = [q for q in queries if q.get("row_id") not in done_ids]
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    build_kwargs = dict(
        index_dir=args.index_dir,
        top_m=args.top_m,
        top_sem_k=top_sem_k,
        beam_width=args.beam_width,
        eta=args.eta,
        ot_epsilon=args.ot_epsilon,
        ot_iters=args.ot_iters,
        use_ot=use_ot,
        risk_mode=args.risk_mode,
        risk_top_tokens=args.risk_top_tokens,
        risk_max_tokens=args.risk_max_tokens,
        risk_cache_size=args.risk_cache_size,
        risk_temperature=args.risk_temperature,
        token_source=args.risk_token_source,
        risk_anchor=risk_anchor,
        lambda_risk=args.lambda_risk,
        lambda_num=args.lambda_num,
        score_norm=args.score_norm,
        use_role=use_role,
        use_coverage=use_coverage,
        lambda_ot=lambda_ot,
        delta_c=args.delta_c,
        server_url=args.server_url,
        batch_size=args.batch_size,
        method=method,
        pref_weights=args.pref_weights if args.pref_weights and not args.no_pref else None,
        flat_topk=getattr(args, "flat_topk", 0),
    )
    workers = max(1, int(args.workers))
    if workers <= 1 or len(queries) <= 1:
        resources = prepare_chain_resources(
            queries=queries,
            prefill_risk_cache=True,
            precompute_role_vectors=True,
            **build_kwargs,
        )
        results = build_chains_with_resources(queries, resources)
    else:
        chunk_size = _resolve_chunk_size(len(queries), workers, args.chunk_size)
        chunks = _chunk_queries(queries, chunk_size)
        max_workers = min(workers, len(chunks))
        print(
            f"[evidence] parallel chain build: workers={max_workers}, chunks={len(chunks)}, "
            f"chunk_size={chunk_size}"
        )
        shared_resources = prepare_chain_resources(
            queries=queries,
            prefill_risk_cache=True,
            precompute_role_vectors=True,
            **build_kwargs,
        )
        chunk_results: Dict[int, List[Dict]] = {}
        global _SHARED_CHAIN_RESOURCES
        _SHARED_CHAIN_RESOURCES = shared_resources
        try:
            try:
                fork_ctx = mp.get_context("fork")
            except ValueError:  # pragma: no cover - non-posix fallback
                fork_ctx = None
            executor_kwargs = {"max_workers": max_workers}
            if fork_ctx is not None:
                executor_kwargs["mp_context"] = fork_ctx
            with ProcessPoolExecutor(**executor_kwargs) as pool:
                futures = {
                    pool.submit(_build_chunk, (idx, chunk, build_kwargs)): idx
                    for idx, chunk in enumerate(chunks)
                }
                for future in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Build chunks",
                ):
                    chunk_idx, outputs = future.result()
                    chunk_results[chunk_idx] = outputs
        finally:
            _SHARED_CHAIN_RESOURCES = None
        results = []
        for idx in range(len(chunks)):
            results.extend(chunk_results.get(idx, []))

    with open(output_path, "a", encoding="utf-8") as f:
        for item in tqdm.tqdm(results, desc="Write chains"):
            item["method"] = method
            if "config" in item and isinstance(item["config"], dict):
                item["config"]["method"] = method
            else:
                item["config"] = run_config
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[evidence] chains done -> {output_path}")


if __name__ == "__main__":
    main()
