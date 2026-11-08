#! /usr/bin/env python3
"""
TRACE preference training: jointly optimize w_r (risk anchor) and λ_ot.

Minimises the preference loss L_pref = −log σ(β (Δ_θ − Δ_0)), where Δ is the
score margin between a positive and a hard-negative document and Δ_0 is the
margin under the frozen reference parameters θ_0 = (w_r_init, λ_ot_init).

  - Multiple preference pairs per query (hard-negative mining).
  - w_r and λ_ot are optimised jointly in a single training loop.
  - Uses ColBERT contextual token vectors only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import torch
import tqdm

from .retrieval import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_OT_EPS,
    DEFAULT_OT_ITERS,
    DEFAULT_RISK_CACHE_SIZE,
    DEFAULT_RISK_MAX_TOKENS,
    DEFAULT_RISK_TEMPERATURE,
    DEFAULT_RISK_TOP_TOKENS,
    EMBED_SERVER_URL,
    RiskMeasure,
    _score_risk_measure_contextual,
    build_risk_anchor,
    embed_texts,
    embed_tokens_contextual,
    load_index,
    load_queries,
    meta_text_for_risk,
    read_jsonl,
    risk_terms_signature,
)


# ── Helpers ───────────────────────────────────────────────────────


def _load_qrels(path: str, allowed_ids: Set[str]) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = {}
    for item in read_jsonl(path):
        row_id = item.get("row_id")
        if not row_id or row_id not in allowed_ids:
            continue
        gold = set(item.get("gold_evidence_ids") or [])
        if gold:
            qrels[row_id] = gold
    return qrels


def _sigmoid(x: float) -> float:
    if x > 500.0:
        return 1.0
    if x < -500.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _load_pref_weights(path: str) -> Dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


# ── ColBERT Vector Cache ─────────────────────────────────────────


class _ColBERTVecCache:
    """Cache contextual (ColBERT) token vectors per text, anchor-independent.

    Stores encoder output H_x = f_θ(x).  Risk-measure weights
    a_i = softmax(w_r^⊤ h_i / T) are computed on-the-fly with the current
    anchor so that gradient updates take immediate effect.
    """

    def __init__(self, server_url: str, batch_size: int) -> None:
        self.server_url = server_url
        self.batch_size = batch_size
        self._store: Dict[str, np.ndarray] = {}

    def _key(self, text: str) -> str:
        return (text or "").strip()

    def get_vecs(self, text: str) -> np.ndarray:
        """Return (n_tokens, dim) ColBERT token vectors for *text*."""
        key = self._key(text)
        if key in self._store:
            return self._store[key]
        vecs_list = embed_tokens_contextual(
            self.server_url, [key or ""], self.batch_size,
        )
        if vecs_list and vecs_list[0].ndim == 2 and vecs_list[0].shape[0] > 0:
            vecs = vecs_list[0]
        else:
            vecs = embed_texts(self.server_url, [key or ""], self.batch_size)
        self._store[key] = vecs
        return vecs

    def make_measure(
        self, text: str, anchor: np.ndarray, temperature: float
    ) -> RiskMeasure:
        """Build risk measure μ(x) with current anchor (no top-k truncation)."""
        vecs = self.get_vecs(text)
        return _score_risk_measure_contextual(vecs, anchor, temperature, top_tokens=0)

    def batch_prefill(self, texts: List[str]) -> None:
        """Batch-embed all texts via ColBERT endpoint."""
        uncached = [t for t in texts if self._key(t) not in self._store]
        if not uncached:
            return
        colbert_batch = min(self.batch_size, 16)
        for start in range(0, len(uncached), colbert_batch):
            chunk = uncached[start : start + colbert_batch]
            vecs_list = embed_tokens_contextual(
                self.server_url, chunk, colbert_batch,
            )
            for text, token_vecs in zip(chunk, vecs_list):
                key = self._key(text)
                if token_vecs.ndim == 2 and token_vecs.shape[0] > 0:
                    self._store[key] = token_vecs
                else:
                    fallback = embed_texts(
                        self.server_url, [text or ""], self.batch_size,
                    )
                    self._store[key] = fallback


# ── Preference Pair Construction ─────────────────────────────────


class PrefPair(NamedTuple):
    query_text: str
    pos_text: str
    neg_text: str
    delta_sem: float


def _build_preference_pairs(
    queries: List[Dict],
    query_vecs: np.ndarray,
    index,
    meta: List[Dict],
    qrels: Dict[str, Set[str]],
    top_sem_k: int,
    max_neg_per_pos: int = 3,
    max_pairs_per_query: int = 10,
) -> List[PrefPair]:
    """Build hard-negative preference pairs.

    For each query:
      1. Retrieve top_sem_k candidates by semantic similarity.
      2. Split into gold (d†) and non-gold sets.
      3. Sort non-gold by descending semantic score (hardest negatives first).
      4. Pair each gold with up to *max_neg_per_pos* hardest negatives.
      5. Cap total pairs per query at *max_pairs_per_query*.
    """
    pairs: List[PrefPair] = []
    skipped = 0
    for q, q_vec in tqdm.tqdm(
        zip(queries, query_vecs), total=len(queries), desc="Building pref pairs"
    ):
        row_id = q.get("row_id")
        if not row_id or row_id not in qrels:
            continue
        gold_ids = qrels[row_id]
        if not gold_ids:
            continue

        scores, idxs = index.search(q_vec.reshape(1, -1), top_sem_k)
        idx_list = [int(i) for i in idxs[0] if i != -1]
        sem_scores = [float(s) for s in scores[0][: len(idx_list)]]
        if not idx_list:
            continue

        gold_entries: List[Tuple[int, float]] = []
        neg_entries: List[Tuple[int, float]] = []
        for i, idx in enumerate(idx_list):
            ev_id = meta[idx].get("evidence_id")
            if ev_id in gold_ids:
                gold_entries.append((i, sem_scores[i]))
            else:
                neg_entries.append((i, sem_scores[i]))

        if not gold_entries or not neg_entries:
            skipped += 1
            continue

        neg_entries.sort(key=lambda x: x[1], reverse=True)
        query_text = q.get("query_with_meta") or q.get("query", "")
        query_pairs: List[PrefPair] = []
        for pos_rank, pos_sem in gold_entries:
            for neg_rank, neg_sem in neg_entries[:max_neg_per_pos]:
                query_pairs.append(PrefPair(
                    query_text=query_text,
                    pos_text=meta_text_for_risk(meta[idx_list[pos_rank]]),
                    neg_text=meta_text_for_risk(meta[idx_list[neg_rank]]),
                    delta_sem=pos_sem - neg_sem,
                ))
            if len(query_pairs) >= max_pairs_per_query:
                break
        pairs.extend(query_pairs[:max_pairs_per_query])

    if skipped:
        print(f"  Skipped {skipped} queries (no gold or no neg in top-{top_sem_k})")
    print(f"  Total preference pairs: {len(pairs)}")
    return pairs


# ── GPU-accelerated OT (torch) ───────────────────────────────────

_OT_DEVICE: Optional[torch.device] = None


def _get_ot_device() -> torch.device:
    """Return the first visible GPU (select it with CUDA_VISIBLE_DEVICES)."""
    global _OT_DEVICE
    if _OT_DEVICE is not None:
        return _OT_DEVICE
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError(
            f"[train_pref] no visible GPU (cuda.available={torch.cuda.is_available()})"
        )
    _OT_DEVICE = torch.device("cuda:0")
    try:
        torch.zeros(1, device=_OT_DEVICE)
    except Exception as e:
        raise RuntimeError(f"[train_pref] GPU cuda:0 unavailable: {e}") from e
    print(f"[train_pref] OT device: cuda:0 ({torch.cuda.get_device_name(0)}, visible_gpus={n_gpus})")
    return _OT_DEVICE


def _normalize_weights_t(w: torch.Tensor) -> torch.Tensor:
    s = w.sum()
    if s > 1e-12:
        return w / s
    return torch.ones_like(w) / max(w.numel(), 1)


def _sinkhorn_gpu(
    a_w: torch.Tensor,
    b_w: torch.Tensor,
    a_sup: torch.Tensor,
    b_sup: torch.Tensor,
    epsilon: float,
    n_iters: int,
    cost_mode: str = "l2sq",
    return_dual: bool = False,
):
    """Sinkhorn OT on GPU. Inputs are torch tensors already on device."""
    if a_sup.numel() == 0 or b_sup.numel() == 0:
        one = torch.ones(1, device=a_w.device, dtype=torch.float32)
        if return_dual:
            return 0.0, one, one
        return 0.0

    a_w = _normalize_weights_t(a_w)
    b_w = _normalize_weights_t(b_w)

    # Cost matrix
    if cost_mode == "l2sq":
        diff = a_sup.unsqueeze(1) - b_sup.unsqueeze(0)  # (m, n, d)
        cost = (diff * diff).sum(dim=-1)                  # (m, n)
    else:
        sim = torch.clamp(a_sup @ b_sup.T, -1.0, 1.0)
        cost = torch.clamp(1.0 - sim, min=0.0)

    kernel = torch.exp(-cost / max(epsilon, 1e-6))

    u = torch.ones_like(a_w)
    v = torch.ones_like(b_w)
    eps = 1e-9
    for _ in range(max(1, n_iters)):
        u = a_w / (kernel @ v + eps)
        v = b_w / (kernel.T @ u + eps)

    transport = (u.unsqueeze(1) * kernel) * v.unsqueeze(0)
    dist = float((transport * cost).sum().item())

    if return_dual:
        return dist, u, v
    return dist


def _make_measure_gpu(
    vecs_np: np.ndarray,
    anchor_t: torch.Tensor,
    temperature: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build (weights, support) on GPU from cached numpy vecs + current anchor."""
    if vecs_np.ndim != 2 or vecs_np.shape[0] == 0:
        dim = int(vecs_np.shape[1]) if vecs_np.ndim == 2 else 1
        w = torch.ones(1, device=device, dtype=torch.float32)
        s = torch.zeros(1, dim, device=device, dtype=torch.float32)
        return w, s
    vecs_t = torch.from_numpy(vecs_np).to(device=device, dtype=torch.float32)
    scores = vecs_t @ anchor_t
    if temperature > 1e-6:
        scores = scores / temperature
    weights = torch.softmax(scores, dim=0)
    weights = _normalize_weights_t(weights)
    return weights, vecs_t


def _ot_marginal_anchor_grad_gpu(
    dual_potentials: torch.Tensor,
    weights: torch.Tensor,
    vecs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """∂OT/∂w_r contribution from one marginal via softmax Jacobian.

    By the envelope theorem for Sinkhorn OT:
        ∂OT/∂a_i = f_i = ε·log(u_i)       (Sinkhorn dual potential)

    Softmax Jacobian (a_i = softmax(w_r^⊤ h_i / T)):
        ∂a_i/∂w_r = a_i · (h_i/T − Σ_j a_j h_j/T)

    Returns  Σ_i f_i · a_i · (h_i/T − mean/T).
    """
    n = weights.numel()
    if n <= 1 or vecs.numel() == 0:
        dim = int(vecs.shape[1]) if vecs.ndim == 2 else 1
        return torch.zeros(dim, device=vecs.device, dtype=torch.float32)

    inv_t = 1.0 / max(temperature, 1e-6)
    scaled = vecs[:n] * inv_t
    weighted_mean = (weights.unsqueeze(1) * scaled).sum(dim=0)
    deviations = scaled - weighted_mean.unsqueeze(0)
    coeffs = dual_potentials[:n] * weights
    return (coeffs.unsqueeze(1) * deviations).sum(dim=0)


# ── Joint Training ───────────────────────────────────────────────


def _train_joint(
    pairs: List[PrefPair],
    tvc: _ColBERTVecCache,
    anchor_init: np.ndarray,
    lambda_ot_init: float,
    temperature: float,
    ot_epsilon: float,
    ot_iters: int,
    epochs: int,
    lr_anchor: float,
    lr_lambda: float,
    beta: float,
    seed: int,
) -> Tuple[np.ndarray, float]:
    """Jointly optimise w_r and λ_ot by minimising L_pref.

    GPU-accelerated version: all Sinkhorn OT and gradient computations
    run on GPU via torch.  ColBERT vecs stay cached as numpy and are
    transferred to GPU on demand.
    """
    from .retrieval import DEFAULT_OT_COST_MODE

    device = _get_ot_device()

    anchor_np = np.asarray(anchor_init, dtype="float32").copy()
    if anchor_np.ndim != 1:
        raise ValueError(f"risk_anchor must be 1-D, got shape={anchor_np.shape}")
    anchor_t = torch.from_numpy(anchor_np).to(device)
    lambda_ot = max(lambda_ot_init, 0.0)

    ref_anchor_t = anchor_t.clone()
    ref_lambda = float(lambda_ot)

    # Cache reference model measures (frozen θ_0) on GPU: text -> (weights, support)
    ref_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _ref_measure_gpu(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (text or "").strip()
        if key not in ref_cache:
            vecs_np = tvc.get_vecs(text)
            ref_cache[key] = _make_measure_gpu(vecs_np, ref_anchor_t, temperature, device)
        return ref_cache[key]

    cost_mode = DEFAULT_OT_COST_MODE

    rng = random.Random(seed)
    for epoch in range(1, epochs + 1):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        total_loss = 0.0
        n_updates = 0
        t0 = time.time()

        for pi in tqdm.tqdm(order, desc=f"Epoch {epoch}/{epochs}", leave=False):
            pair = pairs[pi]

            # Current-model measures on GPU (w_r changes each step)
            q_vecs = tvc.get_vecs(pair.query_text)
            pos_vecs = tvc.get_vecs(pair.pos_text)
            neg_vecs = tvc.get_vecs(pair.neg_text)

            q_w, q_s = _make_measure_gpu(q_vecs, anchor_t, temperature, device)
            pos_w, pos_s = _make_measure_gpu(pos_vecs, anchor_t, temperature, device)
            neg_w, neg_s = _make_measure_gpu(neg_vecs, anchor_t, temperature, device)

            # Current OT scores with dual variables
            pos_dist, pos_u, pos_v = _sinkhorn_gpu(
                q_w, pos_w, q_s, pos_s, ot_epsilon, ot_iters, cost_mode, return_dual=True,
            )
            neg_dist, neg_u, neg_v = _sinkhorn_gpu(
                q_w, neg_w, q_s, neg_s, ot_epsilon, ot_iters, cost_mode, return_dual=True,
            )
            delta_ot = -pos_dist + neg_dist  # s_ot(q,d†) − s_ot(q,d⊥)

            # Reference OT scores (frozen θ_0)
            rq_w, rq_s = _ref_measure_gpu(pair.query_text)
            rpos_w, rpos_s = _ref_measure_gpu(pair.pos_text)
            rneg_w, rneg_s = _ref_measure_gpu(pair.neg_text)
            pos_ot_ref = -_sinkhorn_gpu(
                rq_w, rpos_w, rq_s, rpos_s, ot_epsilon, ot_iters, cost_mode,
            )
            neg_ot_ref = -_sinkhorn_gpu(
                rq_w, rneg_w, rq_s, rneg_s, ot_epsilon, ot_iters, cost_mode,
            )
            delta_ot_ref = pos_ot_ref - neg_ot_ref

            # L_pref = −log σ(β (Δ_θ − Δ_0))
            delta_theta = pair.delta_sem + lambda_ot * delta_ot
            delta_0 = pair.delta_sem + ref_lambda * delta_ot_ref
            z = beta * (delta_theta - delta_0)
            loss = math.log1p(math.exp(-z))
            total_loss += loss

            sig_z = _sigmoid(z)
            one_minus_sig = 1.0 - sig_z

            # ── Update λ_ot ──
            grad_lambda = -beta * delta_ot * one_minus_sig
            lambda_ot -= lr_lambda * grad_lambda
            if lambda_ot < 0.0:
                lambda_ot = 0.0

            # ── Update w_r ──
            loss_weight = one_minus_sig * beta * lambda_ot
            if abs(loss_weight) < 1e-12:
                n_updates += 1
                continue

            pos_dual_q = ot_epsilon * torch.log(torch.clamp(pos_u, min=1e-30))
            pos_dual_d = ot_epsilon * torch.log(torch.clamp(pos_v, min=1e-30))
            grad_ot_pos = (
                _ot_marginal_anchor_grad_gpu(pos_dual_q, q_w, q_s, temperature)
                + _ot_marginal_anchor_grad_gpu(pos_dual_d, pos_w, pos_s, temperature)
            )

            neg_dual_q = ot_epsilon * torch.log(torch.clamp(neg_u, min=1e-30))
            neg_dual_d = ot_epsilon * torch.log(torch.clamp(neg_v, min=1e-30))
            grad_ot_neg = (
                _ot_marginal_anchor_grad_gpu(neg_dual_q, q_w, q_s, temperature)
                + _ot_marginal_anchor_grad_gpu(neg_dual_d, neg_w, neg_s, temperature)
            )

            grad_anchor = loss_weight * (grad_ot_pos - grad_ot_neg)
            anchor_t = anchor_t - lr_anchor * grad_anchor
            anchor_t = anchor_t / (torch.norm(anchor_t) + 1e-12)
            n_updates += 1

        elapsed = time.time() - t0
        anchor_norm = float(torch.norm(anchor_t).item())
        avg_loss = total_loss / max(n_updates, 1)
        print(
            f"[Epoch {epoch}/{epochs}] loss={avg_loss:.4f}  "
            f"λ_ot={lambda_ot:.4f}  ‖w_r‖={anchor_norm:.4f}  "
            f"time={elapsed:.1f}s"
        )

    return anchor_t.cpu().numpy(), lambda_ot


# ── CLI ──────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRACE preference training (joint w_r + λ_ot, ColBERT mode)"
    )
    parser.add_argument("--index-dir", default="runs/index")
    parser.add_argument("--splits-dir", default="datasets-public/audit-ociqa-demo/splits")
    parser.add_argument(
        "--split", default="train", choices=["train", "dev", "test", "all"],
    )
    parser.add_argument(
        "--qrels", default="datasets-public/audit-ociqa-demo/gold.qrels.jsonl",
    )
    parser.add_argument("--top-sem-k", type=int, default=100)
    parser.add_argument(
        "--max-neg-per-pos", type=int, default=3,
        help="Hard negatives per positive evidence",
    )
    parser.add_argument(
        "--max-pairs-per-query", type=int, default=10,
        help="Max preference pairs per query",
    )
    parser.add_argument("--ot-epsilon", type=float, default=DEFAULT_OT_EPS)
    parser.add_argument("--ot-iters", type=int, default=DEFAULT_OT_ITERS)
    parser.add_argument("--risk-top-tokens", type=int, default=DEFAULT_RISK_TOP_TOKENS)
    parser.add_argument("--risk-max-tokens", type=int, default=DEFAULT_RISK_MAX_TOKENS)
    parser.add_argument("--risk-cache-size", type=int, default=DEFAULT_RISK_CACHE_SIZE)
    parser.add_argument("--risk-temperature", type=float, default=DEFAULT_RISK_TEMPERATURE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--server-url", default=EMBED_SERVER_URL)
    parser.add_argument("--lambda-ot-init", type=float, default=0.3)
    parser.add_argument(
        "--ref-weights", default=None,
        help="Reference weights JSON (warm-start anchor and lambda_ot)",
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--lr-anchor", type=float, default=0.05,
        help="Learning rate for w_r",
    )
    parser.add_argument(
        "--lr-lambda", type=float, default=0.1,
        help="Learning rate for λ_ot",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument(
        "--output", default="runs/audit-ociqa/pref/trace.pref.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # torch must create its GPU context before faiss-gpu does, otherwise
    # torch fails to initialise afterwards.
    _get_ot_device()

    ref_config = _load_pref_weights(args.ref_weights) if args.ref_weights else {}
    ref_anchor: Optional[np.ndarray] = None
    if ref_config:
        if "risk_anchor" in ref_config:
            ref_anchor = np.asarray(ref_config["risk_anchor"], dtype="float32")
        if "lambda_ot" in ref_config:
            args.lambda_ot_init = float(ref_config["lambda_ot"])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    index, meta = load_index(args.index_dir)

    queries = load_queries(args.splits_dir, args.split)
    if args.max_queries:
        queries = queries[: args.max_queries]
    allowed_ids = {q.get("row_id") for q in queries if q.get("row_id")}
    qrels = _load_qrels(args.qrels, allowed_ids)

    query_texts = [q.get("query_with_meta") or q.get("query", "") for q in queries]
    query_vecs = embed_texts(args.server_url, query_texts, args.batch_size)
    if len(query_vecs) != len(queries):
        raise ValueError("query_vecs count mismatch")

    # ── Build preference pairs (multi-pair, hard-negative) ──
    pairs = _build_preference_pairs(
        queries=queries,
        query_vecs=query_vecs,
        index=index,
        meta=meta,
        qrels=qrels,
        top_sem_k=args.top_sem_k,
        max_neg_per_pos=args.max_neg_per_pos,
        max_pairs_per_query=args.max_pairs_per_query,
    )
    if not pairs:
        raise ValueError(
            "No valid preference pairs generated. "
            "Check qrels or retrieval params."
        )

    # ── Pre-embed all texts via ColBERT ──
    tvc = _ColBERTVecCache(args.server_url, args.batch_size)
    all_texts: Set[str] = set()
    for p in pairs:
        all_texts.update([p.query_text, p.pos_text, p.neg_text])
    print(f"Pre-embedding {len(all_texts)} unique texts (ColBERT)...")
    tvc.batch_prefill(list(all_texts))

    # ── Initialise anchor ──
    anchor_init = (
        ref_anchor
        if ref_anchor is not None
        else build_risk_anchor(args.server_url, args.batch_size)
    )

    # ── Joint training of w_r and λ_ot ──
    print(
        f"Joint training: {len(pairs)} pairs, "
        f"{args.epochs} epochs, "
        f"lr_anchor={args.lr_anchor}, lr_lambda={args.lr_lambda}"
    )
    anchor, lambda_ot = _train_joint(
        pairs=pairs,
        tvc=tvc,
        anchor_init=anchor_init,
        lambda_ot_init=args.lambda_ot_init,
        temperature=args.risk_temperature,
        ot_epsilon=args.ot_epsilon,
        ot_iters=args.ot_iters,
        epochs=args.epochs,
        lr_anchor=args.lr_anchor,
        lr_lambda=args.lr_lambda,
        beta=args.beta,
        seed=args.seed,
    )

    # ── Save (output format compatible with run_retrieval / run_evidence) ──
    output = {
        "lambda_ot": lambda_ot,
        "risk_anchor": anchor.tolist(),
        "risk_mode": "token",
        "risk_top_tokens": args.risk_top_tokens,
        "risk_max_tokens": args.risk_max_tokens,
        "risk_cache_size": args.risk_cache_size,
        "risk_temperature": args.risk_temperature,
        "ot_epsilon": args.ot_epsilon,
        "ot_iters": args.ot_iters,
        "beta": args.beta,
        "epochs": args.epochs,
        "lr_anchor": args.lr_anchor,
        "lr_lambda": args.lr_lambda,
        "top_sem_k": args.top_sem_k,
        "max_neg_per_pos": args.max_neg_per_pos,
        "max_pairs_per_query": args.max_pairs_per_query,
        "pairs": len(pairs),
        "split": args.split,
        "qrels": args.qrels,
        "index_dir": args.index_dir,
        "ref_weights": args.ref_weights,
        "terms_hash": risk_terms_signature(),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False, indent=2))

    print(f"Preference weights saved -> {args.output}")


if __name__ == "__main__":
    main()
