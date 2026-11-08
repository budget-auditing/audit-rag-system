#! /usr/bin/env python3
"""
TRACE evidence chain construction (role-conditioned pools + coverage + beam search).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import tqdm

from .retrieval import (
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
    RISK_TERMS,
    embed_texts,
    load_index,
    load_risk_matrix_cache,
    load_ot_cache,
    measure_mean,
    meta_text_for_risk,
    merge_measures,
    normalize_weights,
    risk_terms_signature,
    risk_vector,
    RiskMeasureCache,
    search,
    sinkhorn_distance,
    sinkhorn_distance_measure,
)


DEFAULT_DELTA_C = 0.55


def role_condition_query(role: str, query: str) -> str:
    return f"[{role}] {query}".strip()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _ensure_2d(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype="float32")
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"query_vec must be a 1-D or 2-D array, got shape={arr.shape}")
    return arr


def _top_m_candidates(
    index: faiss.Index,
    meta: List[Dict],
    query_vec: np.ndarray,
    top_m: int,
    query_text: str | None = None,
    top_sem_k: int | None = None,
    risk_mode: str = DEFAULT_RISK_MODE,
    risk_matrix: np.ndarray | None = None,
    risk_nums: np.ndarray | None = None,
    risk_cache: RiskMeasureCache | None = None,
    ot_cost: np.ndarray | None = None,
    ot_kernel: np.ndarray | None = None,
    ot_epsilon: float = DEFAULT_OT_EPS,
    ot_iters: int = DEFAULT_OT_ITERS,
    use_ot: bool = True,
    lambda_ot: float = DEFAULT_LAMBDA_OT,
    lambda_risk: float = 0.0,
    lambda_num: float = 0.0,
    score_norm: str = DEFAULT_SCORE_NORM,
) -> List[Dict]:
    query_vec = _ensure_2d(query_vec)
    top_sem_k = max(top_m, top_sem_k or top_m)
    result = search(
        index=index,
        meta=meta,
        query_vec=query_vec,
        query_text=query_text or "",
        top_sem_k=top_sem_k,
        top_k=top_m,
        lambda_ot=(lambda_ot if use_ot else 0.0),
        risk_mode=risk_mode,
        lambda_risk=lambda_risk,
        lambda_num=lambda_num,
        risk_matrix=risk_matrix,
        risk_nums=risk_nums,
        risk_cache=risk_cache,
        ot_cost=ot_cost,
        ot_kernel=ot_kernel,
        ot_epsilon=ot_epsilon,
        ot_iters=ot_iters,
        score_norm=score_norm,
    )
    if not result.meta_indices:
        return []

    out = []
    for rank_idx, idx in enumerate(result.meta_indices):
        item = meta[idx]
        out.append(
            {
                "idx": int(idx),
                "evidence_id": item.get("evidence_id"),
                "text": item.get("text"),
                "text_with_meta": meta_text_for_risk(item),
                "year": item.get("year"),
                "caliber": item.get("caliber"),
                "score": float(result.combined_scores[rank_idx]),
                "s_sem": float(result.sem_scores[rank_idx]),
                "s_ot": float(result.ot_scores[rank_idx]),
                "s_risk": float(result.risk_scores[rank_idx]),
                "s_num": float(result.num_scores[rank_idx]),
                "s_sem_norm": float(result.sem_norm_scores[rank_idx]),
                "s_ot_norm": float(result.ot_norm_scores[rank_idx]),
                "s_risk_norm": float(result.risk_norm_scores[rank_idx]),
                "s_num_norm": float(result.num_norm_scores[rank_idx]),
            }
        )
    return out


def load_risk_matrix(index_dir: str, meta: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    return load_risk_matrix_cache(index_dir, meta)


def build_evidence_chain(
    index: faiss.Index,
    meta: List[Dict],
    query: str,
    top_m: int,
    top_sem_k: int | None,
    beam_width: int,
    eta: float,
    risk_mode: str,
    risk_matrix: np.ndarray | None,
    risk_nums: np.ndarray | None,
    risk_cache: RiskMeasureCache | None,
    ot_cost: np.ndarray | None,
    ot_kernel: np.ndarray | None,
    ot_epsilon: float,
    ot_iters: int,
    use_ot: bool,
    lambda_ot: float,
    lambda_risk: float,
    lambda_num: float,
    score_norm: str,
    use_role: bool,
    use_coverage: bool,
    delta_c: float,
    server_url: str,
    batch_size: int,
    role_vecs: Dict[str, np.ndarray] | None = None,
) -> Tuple[Dict, Dict, Dict, float, float, float, bool, Dict[str, List[Dict]]]:
    if risk_mode == "proxy":
        q_risk = risk_vector(query, RISK_TERMS)
        q_measure = None
    elif risk_mode == "token":
        if risk_cache is None:
            raise ValueError("risk_mode='token' requires `risk_cache`")
        q_risk = None
        q_measure = risk_cache.measure(query)
    else:
        raise ValueError(f"unknown risk_mode: {risk_mode}")

    if use_role:
        role_queries = {
            "O": role_condition_query("O", query),
            "C": role_condition_query("C", query),
            "I": role_condition_query("I", query),
        }
    else:
        role_queries = {"O": query, "C": query, "I": query}
    if role_vecs is not None:
        role_map = role_vecs
    else:
        _role_vecs = embed_texts(server_url, list(role_queries.values()), batch_size)
        role_map = dict(zip(role_queries.keys(), _role_vecs))

    def _fetch_candidates(role_key: str) -> List[Dict]:
        return _top_m_candidates(
            index,
            meta,
            role_map[role_key],
            top_m,
            query_text=role_queries[role_key],
            top_sem_k=top_sem_k,
            risk_mode=risk_mode,
            risk_matrix=risk_matrix,
            risk_nums=risk_nums,
            risk_cache=risk_cache,
            ot_cost=ot_cost,
            ot_kernel=ot_kernel,
            ot_epsilon=ot_epsilon,
            ot_iters=ot_iters,
            use_ot=use_ot,
            lambda_ot=lambda_ot,
            lambda_risk=lambda_risk,
            lambda_num=lambda_num,
            score_norm=score_norm,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_o = pool.submit(_fetch_candidates, "O")
        fut_c = pool.submit(_fetch_candidates, "C")
        fut_i = pool.submit(_fetch_candidates, "I")
        cand_o = fut_o.result()
        cand_c = fut_c.result()
        cand_i = fut_i.result()
    candidate_pools = {"O": cand_o, "C": cand_c, "I": cand_i}

    if not use_coverage:
        def pick_first(cands: List[Dict]) -> Dict:
            if cands:
                return cands[0]
            return {
                "idx": -1,
                "evidence_id": None,
                "text": "",
                "text_with_meta": "",
                "year": None,
                "caliber": None,
                "score": 0.0,
            }

        o = pick_first(cand_o)
        c = pick_first(cand_c)
        i = pick_first(cand_i)
        return o, c, i, 0.0, 0.0, 0.0, False, candidate_pools

    if risk_mode == "proxy":
        if risk_matrix is None:
            raise ValueError("risk_mode='proxy' requires `risk_matrix`")
        measure_cache = {c["idx"]: risk_matrix[c["idx"]] for c in cand_o + cand_c + cand_i}
    else:
        measure_cache = {
            c["idx"]: risk_cache.measure(c.get("text_with_meta") or c.get("text") or "")
            for c in cand_o + cand_c + cand_i
        }
    enforce_unique = len({c["idx"] for c in cand_o + cand_c + cand_i}) >= 3
    pair_ot_cache: Dict[Tuple[int, int], float] = {}

    def risk_for(cand: Dict):
        return measure_cache[cand["idx"]]

    def repr_vec(cand: Dict) -> np.ndarray:
        if risk_mode == "proxy":
            return risk_for(cand)
        return measure_mean(risk_for(cand))

    def merge_risk(cands: List[Dict]):
        scores = np.array([c.get("score", 0.0) for c in cands], dtype="float32")
        if scores.size:
            scores = np.exp(scores - float(np.max(scores)))
        weights = normalize_weights(scores) if scores.size else np.array([], dtype="float32")
        if weights.size != len(cands):
            weights = np.ones(len(cands), dtype="float32") / max(len(cands), 1)
        if risk_mode == "proxy":
            vec = np.zeros_like(q_risk, dtype="float32")
            for w, c in zip(weights, cands):
                vec += float(w) * risk_for(c)
            return normalize_weights(vec)
        measures = [risk_for(c) for c in cands]
        return merge_measures(measures, weights)

    def coverage_score(cands: List[Dict]) -> float:
        merged = merge_risk(cands)
        if risk_mode == "proxy":
            if use_ot and ot_cost is not None and ot_kernel is not None:
                dist = sinkhorn_distance(q_risk, merged, ot_cost, ot_kernel, ot_iters)
                return math.exp(-dist)
            return _cosine(q_risk, merged)
        if use_ot:
            dist = sinkhorn_distance_measure(q_measure, merged, ot_epsilon, ot_iters)
            return math.exp(-dist)
        return _cosine(measure_mean(q_measure), measure_mean(merged))

    def ot_pair(a: Dict, b: Dict) -> float:
        key = (a["idx"], b["idx"])
        if key not in pair_ot_cache:
            if risk_mode == "proxy":
                value = sinkhorn_distance(
                    risk_for(a), risk_for(b), ot_cost, ot_kernel, ot_iters
                )
            else:
                value = sinkhorn_distance_measure(
                    risk_for(a), risk_for(b), ot_epsilon, ot_iters
                )
            pair_ot_cache[key] = value
            pair_ot_cache[(b["idx"], a["idx"])] = value
        return pair_ot_cache[key]

    def build_beams(unique_required: bool) -> List[Tuple[float, Dict, Dict, Dict, float, float]]:
        pair_beams: List[Tuple[float, Dict, Dict, float]] = []
        for o in cand_o:
            o_vec = repr_vec(o)
            for c in cand_c:
                if unique_required and o["idx"] == c["idx"]:
                    continue
                c_vec = repr_vec(c)
                cov = coverage_score([o, c])
                if use_ot:
                    link = math.exp(-ot_pair(o, c))
                else:
                    link = _cosine(o_vec, c_vec)
                pair_beams.append((cov + eta * link, o, c, link))

        pair_beams.sort(key=lambda x: x[0], reverse=True)
        pair_beams = pair_beams[: max(1, beam_width)]

        beams: List[Tuple[float, Dict, Dict, Dict, float, float]] = []
        for _, o, c, link_oc in pair_beams:
            c_vec = repr_vec(c)
            for i in cand_i:
                if unique_required and i["idx"] in (o["idx"], c["idx"]):
                    continue
                i_vec = repr_vec(i)
                cov = coverage_score([o, c, i])
                if use_ot:
                    link = link_oc + math.exp(-ot_pair(c, i))
                else:
                    link = link_oc + _cosine(c_vec, i_vec)
                beams.append((cov + eta * link, o, c, i, cov, link))
        return beams

    beams = build_beams(enforce_unique)
    if not beams and enforce_unique:
        beams = build_beams(False)
    if not beams:
        raise ValueError("failed to assemble any evidence-chain candidates")

    beams.sort(key=lambda x: x[0], reverse=True)
    top = beams[0]
    abstain_flag = bool(delta_c > 0.0 and top[0] < delta_c)
    return top[1], top[2], top[3], top[0], top[4], top[5], abstain_flag, candidate_pools


def prepare_chain_resources(
    queries: List[Dict],
    index_dir: str,
    top_m: int,
    top_sem_k: int | None = None,
    beam_width: int = 10,
    eta: float = 0.3,
    ot_epsilon: float = DEFAULT_OT_EPS,
    ot_iters: int = DEFAULT_OT_ITERS,
    use_ot: bool = True,
    risk_mode: str = DEFAULT_RISK_MODE,
    risk_top_tokens: int = DEFAULT_RISK_TOP_TOKENS,
    risk_max_tokens: int = DEFAULT_RISK_MAX_TOKENS,
    risk_cache_size: int = DEFAULT_RISK_CACHE_SIZE,
    risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
    risk_anchor: np.ndarray | None = None,
    lambda_risk: float = 0.0,
    lambda_num: float = 0.0,
    score_norm: str = DEFAULT_SCORE_NORM,
    use_role: bool = True,
    use_coverage: bool = True,
    lambda_ot: float = DEFAULT_LAMBDA_OT,
    delta_c: float = DEFAULT_DELTA_C,
    server_url: str = EMBED_SERVER_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    method: str = "TRACE",
    pref_weights: str | None = None,
    prefill_risk_cache: bool = True,
    precompute_role_vectors: bool = True,
    token_source: str = DEFAULT_RISK_TOKEN_SOURCE,
    flat_topk: int = 0,
) -> Dict[str, Any]:
    index, meta = load_index(index_dir)
    if top_sem_k is None:
        top_sem_k = max(top_m * 3, top_m)
    risk_matrix = None
    risk_nums = None
    risk_cache = None
    ot_cost = None
    ot_kernel = None
    if risk_mode == "proxy":
        risk_matrix, risk_nums = load_risk_matrix(index_dir, meta)
        if use_ot:
            ot_cost, ot_kernel = load_ot_cache(index_dir, server_url, batch_size, ot_epsilon)
        if use_ot and (ot_cost is None or ot_kernel is None):
            print("[WARN] OT resources are missing; falling back to cosine scoring")
            use_ot = False
    else:
        risk_cache = RiskMeasureCache(
            server_url=server_url,
            batch_size=batch_size,
            top_tokens=risk_top_tokens,
            max_tokens=risk_max_tokens,
            temperature=risk_temperature,
            cache_size=risk_cache_size,
            risk_anchor=risk_anchor,
            token_source=token_source,
        )
    if use_ot and risk_mode == "token" and ot_epsilon <= 0:
        raise ValueError("ot_epsilon must be positive")

    # ── Phase 1: Pre-warm risk measure cache ──
    if prefill_risk_cache and risk_mode == "token" and risk_cache is not None:
        needed_size = len(meta) + len(queries) * 4 + 1000
        if risk_cache.cache_size < needed_size:
            risk_cache.cache_size = needed_size
        print(f"[Phase 1/3] Pre-warming risk measures (cache_size={risk_cache.cache_size})")
        doc_texts = [meta_text_for_risk(m) for m in meta]
        risk_cache.batch_prefill(doc_texts, desc="Prefill docs")
        query_texts_for_cache: List[str] = []
        for q in queries:
            qt = q.get("query_with_meta") or q.get("query", "")
            query_texts_for_cache.append(qt)
            if use_role:
                query_texts_for_cache.append(role_condition_query("O", qt))
                query_texts_for_cache.append(role_condition_query("C", qt))
                query_texts_for_cache.append(role_condition_query("I", qt))
            else:
                query_texts_for_cache.extend([qt, qt, qt])
        risk_cache.batch_prefill(query_texts_for_cache, desc="Prefill queries")
    elif risk_mode == "proxy":
        print("[Phase 1/3] Proxy mode, skip batch prefill")
    elif not prefill_risk_cache:
        print("[Phase 1/3] Skip risk cache prefill by config")
    else:
        print("[Phase 1/3] No risk cache, skip batch prefill")

    # ── Phase 2: Pre-compute role query embeddings ──
    precomputed_role_vecs_by_row: Dict[str, Dict[str, np.ndarray]] = {}
    if precompute_role_vectors:
        print(f"[Phase 2/3] Pre-computing role query embeddings ({len(queries)} queries)")
        all_role_texts: List[str] = []
        for q in queries:
            qt = q.get("query_with_meta") or q.get("query", "")
            if use_role:
                all_role_texts.extend([
                    role_condition_query("O", qt),
                    role_condition_query("C", qt),
                    role_condition_query("I", qt),
                ])
            else:
                all_role_texts.extend([qt, qt, qt])
        all_role_vecs = embed_texts(server_url, all_role_texts, batch_size)
        for qi in range(len(queries)):
            base = qi * 3
            row_id = queries[qi].get("row_id")
            role_vecs = {
                "O": all_role_vecs[base],
                "C": all_role_vecs[base + 1],
                "I": all_role_vecs[base + 2],
            }
            if row_id is not None:
                precomputed_role_vecs_by_row[str(row_id)] = role_vecs
    else:
        print("[Phase 2/3] Skip role query precompute; use per-query embedding")

    # ── Phase 3: Build evidence chains ──
    print(f"[Phase 3/3] Building evidence chains ({len(queries)} queries)")
    terms_hash = risk_terms_signature()
    run_config = {
        "index_dir": index_dir,
        "top_m": top_m,
        "top_sem_k": top_sem_k,
        "beam_width": beam_width,
        "eta": eta,
        "ot_epsilon": ot_epsilon,
        "ot_iters": ot_iters,
        "use_ot": use_ot,
        "risk_mode": risk_mode,
        "risk_top_tokens": risk_top_tokens,
        "risk_max_tokens": risk_max_tokens,
        "risk_cache_size": risk_cache_size,
        "risk_temperature": risk_temperature,
        "lambda_risk": lambda_risk,
        "lambda_num": lambda_num,
        "score_norm": score_norm,
        "use_role": use_role,
        "use_coverage": use_coverage,
        "lambda_ot": lambda_ot,
        "delta_c": delta_c,
        "server_url": server_url,
        "terms_hash": terms_hash,
        "method": method,
        "pref_weights": pref_weights,
    }
    return {
        "index": index,
        "meta": meta,
        "top_m": top_m,
        "top_sem_k": top_sem_k,
        "beam_width": beam_width,
        "eta": eta,
        "ot_epsilon": ot_epsilon,
        "ot_iters": ot_iters,
        "use_ot": use_ot,
        "risk_mode": risk_mode,
        "risk_matrix": risk_matrix,
        "risk_nums": risk_nums,
        "risk_cache": risk_cache,
        "lambda_risk": lambda_risk,
        "lambda_num": lambda_num,
        "score_norm": score_norm,
        "use_role": use_role,
        "use_coverage": use_coverage,
        "lambda_ot": lambda_ot,
        "delta_c": delta_c,
        "server_url": server_url,
        "batch_size": batch_size,
        "method": method,
        "run_config": run_config,
        "ot_cost": ot_cost,
        "ot_kernel": ot_kernel,
        "precomputed_role_vecs_by_row": precomputed_role_vecs_by_row,
        "flat_topk": flat_topk,
    }


def build_chains_with_resources(
    queries: List[Dict],
    resources: Dict[str, Any],
) -> List[Dict]:
    index = resources["index"]
    meta = resources["meta"]
    top_m = resources["top_m"]
    top_sem_k = resources["top_sem_k"]
    beam_width = resources["beam_width"]
    eta = resources["eta"]
    ot_epsilon = resources["ot_epsilon"]
    ot_iters = resources["ot_iters"]
    use_ot = resources["use_ot"]
    risk_mode = resources["risk_mode"]
    risk_matrix = resources["risk_matrix"]
    risk_nums = resources["risk_nums"]
    risk_cache = resources["risk_cache"]
    lambda_risk = resources["lambda_risk"]
    lambda_num = resources["lambda_num"]
    score_norm = resources["score_norm"]
    use_role = resources["use_role"]
    use_coverage = resources["use_coverage"]
    lambda_ot = resources["lambda_ot"]
    delta_c = resources["delta_c"]
    server_url = resources["server_url"]
    batch_size = resources["batch_size"]
    method = resources["method"]
    run_config = resources["run_config"]
    ot_cost = resources["ot_cost"]
    ot_kernel = resources["ot_kernel"]
    precomputed_role_vecs_by_row = resources.get("precomputed_role_vecs_by_row") or {}
    flat_topk = resources.get("flat_topk", 0)

    outputs = []
    for q in tqdm.tqdm(queries, desc="Build chains"):
        query_raw = q.get("query", "")
        query = q.get("query_with_meta") or query_raw
        if use_role:
            role_queries = {
                "O": role_condition_query("O", query),
                "C": role_condition_query("C", query),
                "I": role_condition_query("I", query),
            }
        else:
            role_queries = {"O": query, "C": query, "I": query}
        error = None
        fallback = False
        row_id = q.get("row_id")
        pre_role = None
        if row_id is not None:
            pre_role = precomputed_role_vecs_by_row.get(str(row_id))
        try:
            o, c, i, score, cov, link, chain_abstain_flag, candidate_pools = build_evidence_chain(
                index=index,
                meta=meta,
                query=query,
                top_m=top_m,
                top_sem_k=top_sem_k,
                beam_width=beam_width,
                eta=eta,
                risk_mode=risk_mode,
                risk_matrix=risk_matrix,
                risk_nums=risk_nums,
                risk_cache=risk_cache,
                ot_cost=ot_cost,
                ot_kernel=ot_kernel,
                ot_epsilon=ot_epsilon,
                ot_iters=ot_iters,
                use_ot=use_ot,
                lambda_ot=lambda_ot,
                lambda_risk=lambda_risk,
                lambda_num=lambda_num,
                score_norm=score_norm,
                use_role=use_role,
                use_coverage=use_coverage,
                delta_c=delta_c,
                server_url=server_url,
                batch_size=batch_size,
                role_vecs=pre_role,
            )
        except Exception as exc:
            error = str(exc)
            fallback = True
            role_map = pre_role
            if role_map is None:
                role_vecs = embed_texts(server_url, list(role_queries.values()), batch_size)
                role_map = dict(zip(role_queries.keys(), role_vecs))

            def pick(role_key: str) -> Dict:
                cands = _top_m_candidates(
                    index,
                    meta,
                    role_map[role_key],
                    1,
                    query_text=role_queries[role_key],
                    top_sem_k=top_sem_k,
                    risk_mode=risk_mode,
                    risk_matrix=risk_matrix,
                    risk_nums=risk_nums,
                    risk_cache=risk_cache,
                    ot_cost=ot_cost,
                    ot_kernel=ot_kernel,
                    ot_epsilon=ot_epsilon,
                    ot_iters=ot_iters,
                    use_ot=use_ot,
                    lambda_ot=lambda_ot,
                    lambda_risk=lambda_risk,
                    lambda_num=lambda_num,
                    score_norm=score_norm,
                )
                if cands:
                    return cands[0]
                return {
                    "idx": -1,
                    "evidence_id": None,
                    "text": "",
                    "text_with_meta": "",
                    "year": None,
                    "caliber": None,
                    "score": 0.0,
                }

            with ThreadPoolExecutor(max_workers=3) as pool:
                fut_o = pool.submit(pick, "O")
                fut_c = pool.submit(pick, "C")
                fut_i = pool.submit(pick, "I")
                o = fut_o.result()
                c = fut_c.result()
                i = fut_i.result()
            candidate_pools = {"O": [o], "C": [c], "I": [i]}
            score = 0.0
            cov = 0.0
            link = 0.0
            chain_abstain_flag = True
        def _ev_dict(cand: Dict) -> Dict:
            return {
                "evidence_id": cand["evidence_id"],
                "text": cand["text"],
                "text_with_meta": cand.get("text_with_meta") or cand.get("text") or "",
                "year": cand.get("year"),
                "caliber": cand.get("caliber"),
            }

        extra_evidence: List[Dict] = []
        if flat_topk > 0:
            seen = {o["evidence_id"], c["evidence_id"], i["evidence_id"]}
            all_cands = []
            for pool in candidate_pools.values():
                for cand in pool:
                    eid = cand.get("evidence_id")
                    if eid and eid not in seen:
                        seen.add(eid)
                        all_cands.append(cand)
            all_cands.sort(key=lambda x: x.get("score", 0), reverse=True)
            extra_evidence = [_ev_dict(c) for c in all_cands[:flat_topk - 3]]

        out_item = {
            "row_id": q.get("row_id"),
            "qid": q.get("qid"),
            "query": query_raw,
            "method": method,
            "role_queries": role_queries,
            "evidence": {
                "O": _ev_dict(o),
                "C": _ev_dict(c),
                "I": _ev_dict(i),
            },
            "candidate_pools": candidate_pools,
            "score": score,
            "coverage": cov,
            "link": link,
            "J_evidence": score,
            "Cov": cov,
            "coherence": link,
            "delta_c": delta_c,
            "chain_abstain_flag": chain_abstain_flag,
            "top_m": top_m,
            "beam_width": beam_width,
            "eta": eta,
            "config": run_config,
            "fallback": fallback,
            "error": error,
        }
        if extra_evidence:
            out_item["extra_evidence"] = extra_evidence
        outputs.append(out_item)
    return outputs


def build_chains_for_queries(
    queries: List[Dict],
    index_dir: str,
    top_m: int,
    top_sem_k: int | None = None,
    beam_width: int = 10,
    eta: float = 0.3,
    ot_epsilon: float = DEFAULT_OT_EPS,
    ot_iters: int = DEFAULT_OT_ITERS,
    use_ot: bool = True,
    risk_mode: str = DEFAULT_RISK_MODE,
    risk_top_tokens: int = DEFAULT_RISK_TOP_TOKENS,
    risk_max_tokens: int = DEFAULT_RISK_MAX_TOKENS,
    risk_cache_size: int = DEFAULT_RISK_CACHE_SIZE,
    risk_temperature: float = DEFAULT_RISK_TEMPERATURE,
    risk_anchor: np.ndarray | None = None,
    lambda_risk: float = 0.0,
    lambda_num: float = 0.0,
    score_norm: str = DEFAULT_SCORE_NORM,
    use_role: bool = True,
    use_coverage: bool = True,
    lambda_ot: float = DEFAULT_LAMBDA_OT,
    delta_c: float = DEFAULT_DELTA_C,
    server_url: str = EMBED_SERVER_URL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    method: str = "TRACE",
    pref_weights: str | None = None,
    prefill_risk_cache: bool = True,
    precompute_role_vectors: bool = True,
    token_source: str = DEFAULT_RISK_TOKEN_SOURCE,
) -> List[Dict]:
    resources = prepare_chain_resources(
        queries=queries,
        index_dir=index_dir,
        top_m=top_m,
        top_sem_k=top_sem_k,
        beam_width=beam_width,
        eta=eta,
        ot_epsilon=ot_epsilon,
        ot_iters=ot_iters,
        use_ot=use_ot,
        risk_mode=risk_mode,
        risk_top_tokens=risk_top_tokens,
        risk_max_tokens=risk_max_tokens,
        risk_cache_size=risk_cache_size,
        risk_temperature=risk_temperature,
        risk_anchor=risk_anchor,
        lambda_risk=lambda_risk,
        lambda_num=lambda_num,
        score_norm=score_norm,
        use_role=use_role,
        use_coverage=use_coverage,
        lambda_ot=lambda_ot,
        delta_c=delta_c,
        server_url=server_url,
        batch_size=batch_size,
        method=method,
        pref_weights=pref_weights,
        prefill_risk_cache=prefill_risk_cache,
        precompute_role_vectors=precompute_role_vectors,
        token_source=token_source,
    )
    return build_chains_with_resources(queries, resources)
