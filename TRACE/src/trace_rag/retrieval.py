#! /usr/bin/env python3
"""
TRACE retrieval: semantic coarse search + risk-aware rerank.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterable, List, NamedTuple, Optional, Set, Tuple
from urllib import request

import faiss
import numpy as np
import tqdm


DEFAULT_BATCH_SIZE = 64
EMBED_SERVER_URL = os.getenv(
    "EMBED_SERVER_URL",
    "http://localhost:8081/v1/embeddings",
)
DEFAULT_OT_EPS = 0.05
DEFAULT_OT_ITERS = 50
DEFAULT_RISK_MODE = "token"
DEFAULT_RISK_TOP_TOKENS = 32
DEFAULT_RISK_MAX_TOKENS = 256
DEFAULT_RISK_CACHE_SIZE = 20000
DEFAULT_RISK_TEMPERATURE = 1.0
DEFAULT_SCORE_NORM = "minmax"
DEFAULT_LAMBDA_OT = 0.3
DEFAULT_OT_COST_MODE = "l2sq"
DEFAULT_RISK_TOKEN_SOURCE = os.getenv("RISK_TOKEN_SOURCE", "contextual")
DEFAULT_EMBED_TIMEOUT = int(os.getenv("EMBED_REQUEST_TIMEOUT", "60"))
DEFAULT_EMBED_MAX_INFLIGHT = max(
    1, int(os.getenv("EMBED_MAX_INFLIGHT_PER_ENDPOINT", "4"))
)
DEFAULT_EMBED_WORKER_CAP = max(1, int(os.getenv("EMBED_MAX_WORKERS", "64")))
EMBED_SHOW_PROGRESS = os.getenv("EMBED_SHOW_PROGRESS", "0") == "1"
RISK_TEXT_MODE = "text_with_meta"


RISK_TERMS = [
    "异常",
    "缺口",
    "偏差",
    "同比",
    "环比",
    "下降",
    "上升",
    "减少",
    "增加",
    "超支",
    "不足",
    "结转",
    "结余",
    "未落实",
    "未细化",
    "风险",
    "违规",
    "问题",
    "滞后",
    "拖欠",
    "挤占",
    "挪用",
    "亏损",
    "赤字",
    "欠收",
    "收支缺口",
    "口径",
    "可比",
    "调整",
    "并入",
    "剔除",
    "勾稽",
    "差额",
    "不一致",
    "对应关系",
    "资金来源",
    "财政拨款",
    "单位资金",
    "其他收入",
    "结转结余",
    # English financial risk terms
    "decrease",
    "increase",
    "decline",
    "growth",
    "loss",
    "deficit",
    "impairment",
    "write-off",
    "writedown",
    "overdue",
    "default",
    "variance",
    "discrepancy",
    "anomaly",
    "deviation",
    "shortfall",
    "surplus",
    "year-over-year",
    "yoy",
    "quarter-over-quarter",
    "qoq",
    "restructuring",
    "restatement",
    "adjustment",
    "provision",
    "contingency",
    "depreciation",
    "amortization",
    "accrual",
    "deferred",
    "receivable",
    "payable",
    "revenue",
    "expense",
    "liability",
    "obligation",
    "outstanding",
    "maturity",
]

RISK_STOP_TERMS = {
    "问题",
    "风险",
    "增加",
    "减少",
    "上升",
    "下降",
    "同比",
    "环比",
    # English common terms (lower weight)
    "increase",
    "decrease",
    "growth",
    "decline",
    "revenue",
    "expense",
}
RISK_TERM_WEIGHTS = {term: 0.5 for term in RISK_STOP_TERMS}
RISK_TERM_WEIGHT_VEC = np.array(
    [RISK_TERM_WEIGHTS.get(term, 1.0) for term in RISK_TERMS], dtype="float32"
)
RISK_TERM_INDEX = {term: idx for idx, term in enumerate(RISK_TERMS)}
RISK_TERM_PATTERN = re.compile(
    "|".join(sorted((re.escape(term) for term in RISK_TERMS), key=len, reverse=True))
)

NUM_REGEX = re.compile(r"-?\d+(?:\.\d+)?%?")
TAG_PATTERN = re.compile(r"\[(YEAR=\d{4}|CAL=[^\]]+)\]")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?%?|[%‰]|[\u4e00-\u9fff]+")
YEAR_QUERY_PATTERN = re.compile(r"(20\d{2})")

CALIBER_PATTERNS = (
    (r"一般公共预算", "general_public_budget"),
    (r"政府性基金", "government_fund_budget"),
    (r"国有资本经营预算", "state_capital_budget"),
    (r"社会保险基金", "social_security_fund_budget"),
    # English financial categories
    (r"(?i)operating\s+(?:income|expense|revenue)", "operating"),
    (r"(?i)net\s+(?:income|revenue|earnings|loss)", "net_income"),
    (r"(?i)gross\s+(?:profit|margin|revenue)", "gross_profit"),
    (r"(?i)total\s+(?:assets?|liabilit(?:y|ies)|equity|revenue|debt)", "balance_sheet"),
    (r"(?i)cash\s+flow", "cash_flow"),
    (r"(?i)(?:interest|tax)\s+(?:expense|income|rate)", "interest_tax"),
    (r"(?i)shareholders?\W+equity", "equity"),
    (r"(?i)(?:long|short)[\s-]+term\s+debt", "debt"),
    (r"(?i)capital\s+expenditure", "capex"),
    (r"(?i)(?:earnings|income)\s+per\s+share", "eps"),
)

_RISK_ANCHOR_CACHE: Dict[str, np.ndarray] = {}
_RISK_ANCHOR_LOCK = threading.Lock()
_EMBED_DISPATCHERS: Dict[Tuple[str, ...], "_EmbeddingDispatcher"] = {}
_EMBED_DISPATCHER_LOCK = threading.Lock()
_EMBED_LOGGED_CONFIGS: Set[Tuple[Tuple[str, ...], int, int, int]] = set()


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_meta_item(item: Dict) -> Dict:
    if item.get("caliber") is None and item.get("budget_caliber"):
        item["caliber"] = item.get("budget_caliber")
    if item.get("level") is None and item.get("budget_level"):
        item["level"] = item.get("budget_level")
    raw_text = item.get("text") or ""
    base_text = item.get("text_with_meta") or item.get("text_for_embedding") or raw_text
    year = _normalize_year(item.get("year")) or _extract_year_from_text(base_text or raw_text)
    calibers = _normalize_calibers(item.get("caliber") or item.get("budget_caliber"))
    if not calibers:
        calibers = _extract_calibers_from_text(base_text or raw_text)

    # Keep existing metadata when available, otherwise backfill from text.
    if item.get("year") is None and year is not None:
        item["year"] = year
    if item.get("caliber") is None and calibers:
        item["caliber"] = calibers[0]

    # Ensure corpus text carries [YEAR]/[CAL] tags for risk-aware alignment.
    item["text_with_meta"] = _format_query_with_meta(base_text, year, calibers)
    if not item.get("text_for_embedding"):
        item["text_for_embedding"] = item["text_with_meta"]
    return item


def meta_text_for_risk(item: Dict) -> str:
    return item.get("text_with_meta") or item.get("text_for_embedding") or item.get("text") or ""


def meta_text_for_numeric(item: Dict) -> str:
    return item.get("text") or ""


def _normalize_year(value: object) -> int | None:
    try:
        year = int(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if 1900 <= year <= 2100:
        return year
    return None


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_calibers(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [str(v).strip() for v in value if v is not None]
    else:
        items = [str(value).strip()]
    items = [v for v in items if v and v.lower() != "unknown"]
    return _dedupe(items)


def _extract_year_from_text(text: str) -> int | None:
    if not text:
        return None
    match = YEAR_QUERY_PATTERN.search(text)
    if not match:
        return None
    return _normalize_year(match.group(1))


def _extract_calibers_from_text(text: str) -> List[str]:
    if not text:
        return []
    hits: List[str] = []
    for pattern, label in CALIBER_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(label)
    return _dedupe(hits)


def _format_query_with_meta(query: str, year: int | None, calibers: List[str]) -> str:
    text = (query or "").strip()
    if not text:
        return text
    if TAG_PATTERN.search(text):
        return text
    parts: List[str] = []
    if year is not None:
        parts.append(f"[YEAR={year}]")
    if calibers:
        parts.append(f"[CAL={';'.join(calibers)}]")
    prefix = " ".join(parts).strip()
    if not prefix:
        return text
    return f"{prefix}\n{text}"


def _resolve_meta_path(splits_dir: str, rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    if os.path.isabs(rel_path):
        return rel_path
    root = os.path.abspath(os.path.join(splits_dir, "..", ".."))
    candidate = os.path.join(root, rel_path)
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(rel_path):
        return os.path.abspath(rel_path)
    return candidate


def _load_query_meta_map(splits_dir: str) -> Dict[str, Dict]:
    meta_path = os.path.join(splits_dir, "meta.jsonl")
    if not os.path.exists(meta_path):
        return {}
    meta_info = None
    for item in read_jsonl(meta_path):
        meta_info = item
        break
    if not meta_info:
        return {}
    source = meta_info.get("source") or {}
    qa_path = _resolve_meta_path(splits_dir, source.get("qa_file"))
    if not qa_path or not os.path.exists(qa_path):
        return {}
    meta_map: Dict[str, Dict] = {}
    for item in read_jsonl(qa_path):
        key = item.get("row_id") or item.get("qid")
        if key:
            meta_map[str(key)] = item
    return meta_map


class RiskMeasure(NamedTuple):
    weights: np.ndarray
    support: np.ndarray


def _is_cjk(token: str) -> bool:
    return bool(token) and all("\u4e00" <= ch <= "\u9fff" for ch in token)


def _split_cjk(token: str, chunk_size: int = 3) -> List[str]:
    return [token[i : i + chunk_size] for i in range(0, len(token), chunk_size)]


def _softmax(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    shifted = scores - float(np.max(scores))
    exp_vals = np.exp(shifted)
    return exp_vals / (np.sum(exp_vals) + 1e-12)


def tokenize_text(
    text: str, max_tokens: int = DEFAULT_RISK_MAX_TOKENS
) -> Tuple[List[str], np.ndarray]:
    if not text:
        return [], np.array([], dtype="float32")
    tokens: List[str] = []
    for match in TAG_PATTERN.finditer(text):
        tokens.append(match.group(0))
    for match in RISK_TERM_PATTERN.finditer(text):
        tokens.append(match.group(0))
    for match in TOKEN_PATTERN.finditer(text):
        tok = match.group(0).strip()
        if not tok:
            continue
        if _is_cjk(tok) and len(tok) > 4:
            tokens.extend(_split_cjk(tok, 3))
        else:
            tokens.append(tok)

    counts: "OrderedDict[str, int]" = OrderedDict()
    for tok in tokens:
        if tok in counts:
            counts[tok] += 1
            continue
        if len(counts) >= max_tokens:
            continue
        counts[tok] = 1
    uniq = list(counts.keys())
    freqs = np.array(list(counts.values()), dtype="float32")
    return uniq, freqs


def _normalize_anchor(anchor: np.ndarray) -> np.ndarray:
    arr = np.asarray(anchor, dtype="float32")
    norm = np.linalg.norm(arr) + 1e-12
    return (arr / norm).astype("float32")


def build_risk_anchor(server_url: str, batch_size: int) -> np.ndarray:
    canonical_url = _canonical_server_url(server_url)
    cache_key = f"{canonical_url}:{risk_terms_signature()}"
    with _RISK_ANCHOR_LOCK:
        if cache_key in _RISK_ANCHOR_CACHE:
            return _RISK_ANCHOR_CACHE[cache_key]
    term_vecs = embed_texts(canonical_url, RISK_TERMS, batch_size)
    weights = np.asarray(RISK_TERM_WEIGHT_VEC, dtype="float32")
    anchor = np.average(term_vecs, axis=0, weights=weights)
    anchor = _normalize_anchor(anchor)
    with _RISK_ANCHOR_LOCK:
        _RISK_ANCHOR_CACHE[cache_key] = anchor
    return anchor


def _score_risk_measure(
    tokens: List[str],
    freqs: np.ndarray,
    vecs: np.ndarray,
    anchor: np.ndarray,
    temperature: float,
    top_tokens: int,
    use_term_bias: bool = False,
    use_freq_boost: bool = False,
) -> RiskMeasure:
    """Compute RiskMeasure from pre-embedded token vectors.

    With default parameters this computes the risk attention of
    docs/method.md (Stage 1):
        a_i = softmax(w_r^T h_i / T)

    The optional term_bias / freq_boost heuristics are off by default.
    """
    scores = vecs @ anchor
    if temperature and temperature > 1e-6:
        scores = scores / float(temperature)
    if use_term_bias and RISK_TERM_WEIGHTS:
        term_bias = np.array(
            [float(RISK_TERM_WEIGHTS.get(tok, 1.0)) for tok in tokens], dtype="float32"
        )
        scores = scores * term_bias
    if use_freq_boost and freqs.size == scores.size:
        scores = scores * (1.0 + np.log1p(freqs))
    weights = _softmax(scores)
    if top_tokens and weights.size > top_tokens:
        idx = np.argsort(weights)[::-1][:top_tokens]
        weights = weights[idx]
        vecs = vecs[idx]
    weights = normalize_weights(weights)
    return RiskMeasure(weights=weights, support=vecs)


def build_risk_measure(
    text: str,
    server_url: str,
    batch_size: int,
    top_tokens: int = DEFAULT_RISK_TOP_TOKENS,
    max_tokens: int = DEFAULT_RISK_MAX_TOKENS,
    temperature: float = DEFAULT_RISK_TEMPERATURE,
    risk_anchor: Optional[np.ndarray] = None,
    token_source: str = DEFAULT_RISK_TOKEN_SOURCE,
) -> RiskMeasure:
    anchor = risk_anchor if risk_anchor is not None else build_risk_anchor(server_url, batch_size)

    if token_source == "contextual":
        global _COLBERT_FALLBACK_WARNED
        try:
            token_vecs_list = embed_tokens_contextual(server_url, [text or ""], batch_size)
            if token_vecs_list and token_vecs_list[0].size > 0:
                return _score_risk_measure_contextual(
                    token_vecs_list[0], anchor, temperature, top_tokens
                )
        except Exception:
            if not _COLBERT_FALLBACK_WARNED:
                print("[WARN] ColBERT endpoint unavailable, falling back to word-level tokens")
                _COLBERT_FALLBACK_WARNED = True

    tokens, freqs = tokenize_text(text, max_tokens=max_tokens)
    if not tokens:
        vec = embed_texts(server_url, [text or ""], batch_size)
        return RiskMeasure(weights=np.array([1.0], dtype="float32"), support=vec)

    vecs = embed_texts(server_url, tokens, batch_size)
    return _score_risk_measure(tokens, freqs, vecs, anchor, temperature, top_tokens)


def measure_mean(measure: RiskMeasure) -> np.ndarray:
    if measure.support.size == 0:
        dim = int(measure.support.shape[1]) if measure.support.ndim == 2 else 1
        return np.zeros((dim,), dtype="float32")
    mean = np.sum(measure.support * measure.weights[:, None], axis=0)
    norm = np.linalg.norm(mean) + 1e-12
    return (mean / norm).astype("float32")


def merge_measures(measures: List[RiskMeasure], weights: np.ndarray) -> RiskMeasure:
    if not measures:
        return RiskMeasure(weights=np.array([], dtype="float32"), support=np.zeros((0, 1)))
    if weights.size != len(measures):
        weights = np.ones(len(measures), dtype="float32") / max(len(measures), 1)
    support_list = []
    weight_list = []
    for w, m in zip(weights, measures):
        if m.support.size == 0:
            continue
        support_list.append(m.support)
        weight_list.append(m.weights * float(w))
    if not support_list:
        return RiskMeasure(weights=np.array([], dtype="float32"), support=np.zeros((0, 1)))
    support = np.vstack(support_list)
    merged_weights = normalize_weights(np.concatenate(weight_list))
    return RiskMeasure(weights=merged_weights, support=support)


def sinkhorn_distance_measure(
    a: RiskMeasure, b: RiskMeasure, epsilon: float, n_iters: int = DEFAULT_OT_ITERS,
    return_dual: bool = False,
):
    if a.support.size == 0 or b.support.size == 0:
        if return_dual:
            return 0.0, np.ones(1, dtype="float32"), np.ones(1, dtype="float32")
        return 0.0
    if DEFAULT_OT_COST_MODE == "l2sq":
        diff = a.support[:, None, :] - b.support[None, :, :]
        cost = np.sum(diff * diff, axis=-1).astype("float32")
    else:
        sim = np.clip(a.support @ b.support.T, -1.0, 1.0)
        cost = np.maximum(1.0 - sim, 0.0).astype("float32")
    kernel = np.exp(-cost / max(epsilon, 1e-6)).astype("float32")
    return sinkhorn_distance(a.weights, b.weights, cost, kernel, n_iters, return_dual=return_dual)


class RiskMeasureCache:
    def __init__(
        self,
        server_url: str = EMBED_SERVER_URL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        top_tokens: int = DEFAULT_RISK_TOP_TOKENS,
        max_tokens: int = DEFAULT_RISK_MAX_TOKENS,
        temperature: float = DEFAULT_RISK_TEMPERATURE,
        cache_size: int = DEFAULT_RISK_CACHE_SIZE,
        risk_anchor: Optional[np.ndarray] = None,
        token_source: str = DEFAULT_RISK_TOKEN_SOURCE,
    ) -> None:
        self.server_url = server_url
        self.batch_size = batch_size
        self.top_tokens = top_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache_size = cache_size
        self.token_source = token_source
        self.cache: "OrderedDict[str, RiskMeasure]" = OrderedDict()
        self._lock = threading.Lock()
        self.anchor = (
            _normalize_anchor(risk_anchor)
            if risk_anchor is not None
            else build_risk_anchor(server_url, batch_size)
        )

    def _key(self, text: str) -> str:
        raw = (text or "").strip().encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def batch_prefill(self, texts: List[str], desc: str = "Prefill") -> None:
        """Pre-compute risk measures for many texts with batched embedding."""
        uncached: List[Tuple[int, str]] = []
        for i, text in enumerate(texts):
            if self._key(text) not in self.cache:
                uncached.append((i, text))
        if not uncached:
            print(f"  {desc}: all {len(texts)} cached, skip")
            return

        if self.token_source == "contextual":
            self._batch_prefill_contextual(uncached, desc)
            return

        # ── Word-level token mode ──
        tokenized: List[Tuple[List[str], np.ndarray]] = []
        all_tokens: List[str] = []
        slices: List[Tuple[int, int]] = []
        empty_indices: List[int] = []

        for j, (_, text) in enumerate(uncached):
            tokens, freqs = tokenize_text(text, max_tokens=self.max_tokens)
            tokenized.append((tokens, freqs))
            if not tokens:
                empty_indices.append(j)
                slices.append((0, 0))
            else:
                start = len(all_tokens)
                all_tokens.extend(tokens)
                slices.append((start, start + len(tokens)))

        print(f"  {desc}: {len(uncached)} texts, {len(all_tokens)} tokens to embed")
        all_vecs = (
            embed_texts(self.server_url, all_tokens, self.batch_size)
            if all_tokens
            else np.zeros((0, 1), dtype="float32")
        )

        empty_vecs = None
        if empty_indices:
            empty_texts = [uncached[ei][1] or "" for ei in empty_indices]
            empty_vecs = embed_texts(self.server_url, empty_texts, self.batch_size)

        empty_map = {ei: k for k, ei in enumerate(empty_indices)}
        for j, (_, text) in enumerate(tqdm.tqdm(uncached, desc=f"  {desc}")):
            tokens, freqs = tokenized[j]
            if not tokens:
                ev_idx = empty_map[j]
                vec = empty_vecs[ev_idx : ev_idx + 1]
                measure = RiskMeasure(
                    weights=np.array([1.0], dtype="float32"), support=vec
                )
            else:
                start, end = slices[j]
                vecs = all_vecs[start:end].copy()
                measure = _score_risk_measure(
                    tokens, freqs, vecs, self.anchor,
                    self.temperature, self.top_tokens,
                )
            key = self._key(text)
            with self._lock:
                self.cache[key] = measure
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)

    def _batch_prefill_contextual(
        self, uncached: List[Tuple[int, str]], desc: str
    ) -> None:
        uncached_texts = [text for _, text in uncached]
        print(f"  {desc}: {len(uncached_texts)} texts (contextual ColBERT)")
        colbert_batch = min(self.batch_size, 16)
        all_token_vecs: List[np.ndarray] = []
        total_batches = max(1, math.ceil(len(uncached_texts) / max(colbert_batch, 1)))
        for batch_idx in range(total_batches):
            start = batch_idx * colbert_batch
            end = min(len(uncached_texts), start + colbert_batch)
            batch_texts = uncached_texts[start:end]
            batch_vecs = embed_tokens_contextual(
                self.server_url, batch_texts, colbert_batch,
            )
            all_token_vecs.extend(batch_vecs)
            if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == total_batches:
                print(
                    f"  {desc}: batch {batch_idx + 1}/{total_batches} "
                    f"(texts={end}/{len(uncached_texts)})"
                )
        for (_, text), token_vecs in zip(uncached, all_token_vecs):
            if token_vecs.ndim == 2 and token_vecs.shape[0] > 0:
                measure = _score_risk_measure_contextual(
                    token_vecs, self.anchor, self.temperature, self.top_tokens,
                )
            else:
                vec = embed_texts(self.server_url, [text or ""], self.batch_size)
                measure = RiskMeasure(
                    weights=np.array([1.0], dtype="float32"), support=vec,
                )
            key = self._key(text)
            with self._lock:
                self.cache[key] = measure
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)

    def measure(self, text: str) -> RiskMeasure:
        key = self._key(text)
        with self._lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
                return cached
        measure = build_risk_measure(
            text=text,
            server_url=self.server_url,
            batch_size=self.batch_size,
            top_tokens=self.top_tokens,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            risk_anchor=self.anchor,
            token_source=self.token_source,
        )
        with self._lock:
            self.cache[key] = measure
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return measure


def _split_server_urls(server_url: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"[,\s]+", (server_url or "").strip()) if p.strip()]
    if not parts:
        raise ValueError("`server_url` must not be empty")
    seen = set()
    urls: List[str] = []
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        urls.append(part)
    return urls


def _canonical_server_url(server_url: str) -> str:
    return ",".join(sorted(_split_server_urls(server_url)))


class _EmbeddingDispatcher:
    def __init__(
        self,
        server_urls: List[str],
        timeout: int = DEFAULT_EMBED_TIMEOUT,
        max_inflight: int = DEFAULT_EMBED_MAX_INFLIGHT,
    ) -> None:
        self.server_urls = tuple(server_urls)
        self.timeout = max(1, int(timeout))
        self.max_inflight = max(1, int(max_inflight))
        self._slots = [threading.Semaphore(self.max_inflight) for _ in self.server_urls]
        self._inflight = [0 for _ in self.server_urls]
        self._lock = threading.Lock()
        self._rr_counter = 0

    def _acquire_endpoint(self, hint: int) -> int:
        size = len(self.server_urls)
        order = [int((hint + i) % size) for i in range(size)]
        while True:
            for idx in order:
                if self._slots[idx].acquire(blocking=False):
                    with self._lock:
                        self._inflight[idx] += 1
                    return idx
            with self._lock:
                idx = min(order, key=lambda i: self._inflight[i])
            self._slots[idx].acquire()
            with self._lock:
                self._inflight[idx] += 1
            return idx

    def _release_endpoint(self, idx: int) -> None:
        with self._lock:
            self._inflight[idx] = max(0, self._inflight[idx] - 1)
        self._slots[idx].release()

    def embed_batch(
        self,
        batch: List[str],
        start_idx: int,
        *,
        batch_size: int,
        max_retries: int = 3,
    ) -> List[List[float]]:
        with self._lock:
            self._rr_counter += 1
            rr = self._rr_counter
        base_hint = (rr + start_idx // max(batch_size, 1)) % len(self.server_urls)
        attempt_budget = max(max_retries, len(self.server_urls))
        for attempt in range(attempt_budget):
            endpoint_idx = self._acquire_endpoint(base_hint + attempt)
            endpoint = self.server_urls[endpoint_idx]
            try:
                payload = {"input": batch}
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = request.Request(
                    endpoint,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                data_list = data.get("data") or []
                if len(data_list) != len(batch):
                    raise RuntimeError(
                        f"embedding count mismatch: got {len(data_list)} expected {len(batch)} "
                        f"(endpoint={endpoint})"
                    )
                return [item["embedding"] for item in data_list]
            except Exception as exc:
                if attempt == attempt_budget - 1:
                    raise RuntimeError(
                        f"Embedding batch failed after {attempt_budget} attempts "
                        f"(start_idx={start_idx}, size={len(batch)}, "
                        f"endpoints={list(self.server_urls)})."
                    ) from exc
                time.sleep(0.3 * (attempt + 1))
            finally:
                self._release_endpoint(endpoint_idx)
        return []


def _get_embed_dispatcher(server_url: str) -> Tuple[List[str], _EmbeddingDispatcher]:
    server_urls = _split_server_urls(server_url)
    key = tuple(server_urls)
    with _EMBED_DISPATCHER_LOCK:
        dispatcher = _EMBED_DISPATCHERS.get(key)
        if dispatcher is None:
            dispatcher = _EmbeddingDispatcher(server_urls)
            _EMBED_DISPATCHERS[key] = dispatcher
    return server_urls, dispatcher


def embed_texts(server_url: str, texts: List[str], batch_size: int, *, max_workers: int = 2) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    server_urls, dispatcher = _get_embed_dispatcher(server_url)
    vecs: List[List[float]] = []

    def _embed_batch(batch: List[str], start_idx: int, max_retries: int = 3) -> List[List[float]]:
        return dispatcher.embed_batch(
            batch,
            start_idx,
            batch_size=batch_size,
            max_retries=max_retries,
        )

    def _embed_with_split(batch: List[str], start_idx: int) -> List[List[float]]:
        try:
            return _embed_batch(batch, start_idx)
        except Exception:
            if len(batch) <= 1:
                raise
            mid = len(batch) // 2
            left = _embed_with_split(batch[:mid], start_idx)
            right = _embed_with_split(batch[mid:], start_idx + mid)
            return left + right

    batches = [(texts[i : i + batch_size], i) for i in range(0, len(texts), batch_size)]

    def _process_batch(args):
        return _embed_with_split(args[0], args[1])

    target_workers = len(server_urls) * dispatcher.max_inflight
    workers = min(
        len(batches),
        DEFAULT_EMBED_WORKER_CAP,
        max(max_workers, target_workers),
    )
    log_key = (tuple(server_urls), batch_size, workers, dispatcher.max_inflight)
    with _EMBED_DISPATCHER_LOCK:
        if log_key not in _EMBED_LOGGED_CONFIGS:
            print(
                f"[EmbeddingLB] endpoints={len(server_urls)} workers={workers} "
                f"batch_size={batch_size} inflight/endpoint={dispatcher.max_inflight}"
            )
            _EMBED_LOGGED_CONFIGS.add(log_key)

    if len(batches) > 1 and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            iterator = pool.map(_process_batch, batches)
            if EMBED_SHOW_PROGRESS:
                iterator = tqdm.tqdm(iterator, total=len(batches), desc="Embedding")
            for result in iterator:
                vecs.extend(result)
    else:
        iterator = batches
        if EMBED_SHOW_PROGRESS:
            iterator = tqdm.tqdm(batches, desc="Embedding")
        for batch_texts, start_idx in iterator:
            vecs.extend(_embed_with_split(batch_texts, start_idx))

    arr = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


# ── Cross-encoder reranker ────────────────────────────────────────


def _split_reranker_urls(server_url: str) -> List[str]:
    urls = [u.strip() for u in server_url.split(",") if u.strip()]
    result = []
    for u in urls:
        if "/v1/rerank" not in u:
            u = u.rstrip("/") + "/v1/rerank"
        result.append(u)
    return result


def rerank_with_cross_encoder(
    reranker_url: str,
    query: str,
    documents: List[str],
    top_k: int | None = None,
) -> List[Tuple[int, float]]:
    """Call cross-encoder reranker API and return (original_index, score) sorted by score desc."""
    if not documents:
        return []
    urls = _split_reranker_urls(reranker_url)
    all_results: List[Tuple[int, float]] = []
    batch_size = 256
    for batch_start in range(0, len(documents), batch_size):
        batch_docs = documents[batch_start : batch_start + batch_size]
        url = urls[batch_start // batch_size % len(urls)]
        payload = json.dumps({
            "model": "reranker",
            "query": query,
            "documents": batch_docs,
        }, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        for attempt in range(3):
            try:
                with request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.0 * (attempt + 1))
        for item in data.get("results", []):
            orig_idx = batch_start + item["index"]
            all_results.append((orig_idx, float(item["relevance_score"])))
    all_results.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        all_results = all_results[:top_k]
    return all_results


# ── Contextual (ColBERT) token embeddings ─────────────────────────

_COLBERT_FALLBACK_WARNED = False


def _to_colbert_url(url: str) -> str:
    url = url.strip()
    if url.endswith("/v1/embeddings"):
        return url[: -len("/v1/embeddings")] + "/v1/colbert"
    if url.endswith("/embeddings"):
        return url[: -len("/embeddings")] + "/colbert"
    return url + "/v1/colbert"


def _colbert_request(
    url: str, texts: List[str], timeout: int = 120
) -> List[np.ndarray]:
    payload = {"input": texts}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
    results: List[np.ndarray] = []
    for item in sorted(data.get("data", []), key=lambda x: x.get("index", 0)):
        vecs = np.array(item["colbert_vecs"], dtype="float32")
        if vecs.size == 0:
            results.append(vecs)
            continue
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        results.append((vecs / norms).astype("float32"))
    return results


def embed_tokens_contextual(
    server_url: str,
    texts: List[str],
    batch_size: int,
    timeout: int = DEFAULT_EMBED_TIMEOUT,
    max_retries: int = 3,
) -> List[np.ndarray]:
    """Get contextual token vectors via ColBERT endpoint.

    Returns one (num_tokens, dim) array per input text.
    """
    if not texts:
        return []
    urls = [_to_colbert_url(u) for u in _split_server_urls(server_url)]
    results: List[np.ndarray] = []
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    for bi, batch in enumerate(batches):
        url = urls[bi % len(urls)]
        for attempt in range(max_retries):
            try:
                results.extend(_colbert_request(url, batch, timeout))
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
    return results


def _score_risk_measure_contextual(
    token_vecs: np.ndarray,
    anchor: np.ndarray,
    temperature: float,
    top_tokens: int,
) -> RiskMeasure:
    """Build RiskMeasure from contextual encoder token vectors.

    Risk attention of docs/method.md (Stage 1): a_i = softmax(w_r^T h_i / T).
    No term bias or frequency boosting — w_r learns to attend via preference training.
    """
    if token_vecs.ndim != 2 or token_vecs.shape[0] == 0:
        dim = int(token_vecs.shape[1]) if token_vecs.ndim == 2 else 1
        return RiskMeasure(
            weights=np.array([1.0], dtype="float32"),
            support=np.zeros((1, dim), dtype="float32"),
        )
    scores = token_vecs @ anchor
    if temperature > 1e-6:
        scores = scores / float(temperature)
    weights = _softmax(scores)
    if top_tokens and weights.size > top_tokens:
        idx = np.argsort(weights)[::-1][:top_tokens]
        weights = weights[idx]
        token_vecs = token_vecs[idx]
    weights = normalize_weights(weights)
    return RiskMeasure(weights=weights, support=token_vecs)


def risk_terms_signature() -> str:
    payload = {
        "terms": RISK_TERMS,
        "weights": [float(RISK_TERM_WEIGHTS.get(term, 1.0)) for term in RISK_TERMS],
        "norm": "l1",
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def risk_term_counts(text: str) -> np.ndarray:
    counts = np.zeros(len(RISK_TERMS), dtype="float32")
    if not text:
        return counts
    for match in RISK_TERM_PATTERN.finditer(text):
        idx = RISK_TERM_INDEX.get(match.group(0))
        if idx is not None:
            counts[idx] += 1.0
    return counts


def normalize_weights(vec: np.ndarray) -> np.ndarray:
    total = float(np.sum(vec))
    if total < 1e-8:
        return np.ones_like(vec, dtype="float32") / max(len(vec), 1)
    return (vec / total).astype("float32")


def risk_weights(text: str, terms: List[str]) -> np.ndarray:
    if terms == RISK_TERMS:
        counts = risk_term_counts(text)
        return np.log1p(counts) * RISK_TERM_WEIGHT_VEC
    counts = [text.count(t) for t in terms]
    return np.log1p(np.array(counts, dtype="float32"))


def risk_vector(text: str, terms: List[str]) -> np.ndarray:
    weights = risk_weights(text, terms)
    return normalize_weights(weights)


def numeric_count(text: str) -> float:
    return float(len(NUM_REGEX.findall(text)))


def build_risk_matrix(meta: List[Dict], terms: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    vectors = []
    nums = []
    for item in tqdm.tqdm(meta, desc="Risk vectors"):
        text = meta_text_for_risk(item)
        vectors.append(risk_vector(text, terms))
        nums.append(math.log1p(numeric_count(meta_text_for_numeric(item))))
    return np.stack(vectors, axis=0), np.array(nums, dtype="float32")


def load_risk_matrix_cache(index_dir: str, meta: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    risk_cache_path = os.path.join(index_dir, "risk_matrix.npz")
    terms_hash = risk_terms_signature()
    if os.path.exists(risk_cache_path):
        cache = np.load(risk_cache_path)
        cache_terms_hash = cache["terms_hash"].item() if "terms_hash" in cache.files else None
        cache_text_mode = cache["text_mode"].item() if "text_mode" in cache.files else None
        risk_matrix = cache["risk_matrix"] if "risk_matrix" in cache.files else None
        risk_nums = cache["risk_nums"] if "risk_nums" in cache.files else None
        if (
            risk_matrix is not None
            and cache_terms_hash == terms_hash
            and risk_matrix.shape[0] == len(meta)
            and risk_matrix.shape[1] == len(RISK_TERMS)
            and cache_text_mode == RISK_TEXT_MODE
            and risk_nums is not None
            and risk_nums.shape[0] == len(meta)
        ):
            return risk_matrix, risk_nums

    risk_matrix, risk_nums = build_risk_matrix(meta, RISK_TERMS)
    np.savez(
        risk_cache_path,
        risk_matrix=risk_matrix,
        risk_nums=risk_nums,
        terms_hash=terms_hash,
        text_mode=RISK_TEXT_MODE,
    )
    return risk_matrix, risk_nums


def load_ot_cache(
    index_dir: str,
    server_url: str,
    batch_size: int,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    canonical_url = _canonical_server_url(server_url)
    cache_path = os.path.join(index_dir, "risk_ot_cache.npz")
    terms_hash = risk_terms_signature()
    if os.path.exists(cache_path):
        cache = np.load(cache_path)
        cache_hash = cache["terms_hash"].item() if "terms_hash" in cache.files else None
        cache_eps = float(cache["epsilon"]) if "epsilon" in cache.files else None
        cache_url = cache["server_url"].item() if "server_url" in cache.files else None
        cache_mode = (
            cache["cost_mode"].item()
            if "cost_mode" in cache.files
            else "cosine"
        )
        if (
            cache_hash == terms_hash
            and cache_eps == epsilon
            and cache_url == canonical_url
            and cache_mode == DEFAULT_OT_COST_MODE
        ):
            return cache["cost"], cache["kernel"]

    term_vecs = embed_texts(canonical_url, RISK_TERMS, batch_size)
    if DEFAULT_OT_COST_MODE == "l2sq":
        diff = term_vecs[:, None, :] - term_vecs[None, :, :]
        cost = np.sum(diff * diff, axis=-1).astype("float32")
    else:
        sim = np.clip(term_vecs @ term_vecs.T, -1.0, 1.0)
        cost = np.maximum((1.0 - sim).astype("float32"), 0.0)
    kernel = np.exp(-cost / max(epsilon, 1e-6)).astype("float32")
    np.savez(
        cache_path,
        cost=cost,
        kernel=kernel,
        epsilon=np.array(epsilon, dtype="float32"),
        terms_hash=np.array(terms_hash),
        server_url=np.array(canonical_url),
        cost_mode=np.array(DEFAULT_OT_COST_MODE),
    )
    return cost, kernel


def sinkhorn_distance(
    a: np.ndarray,
    b: np.ndarray,
    cost: np.ndarray,
    kernel: np.ndarray,
    n_iters: int = DEFAULT_OT_ITERS,
    return_dual: bool = False,
):
    a = normalize_weights(np.asarray(a, dtype="float32"))
    b = normalize_weights(np.asarray(b, dtype="float32"))
    u = np.ones_like(a)
    v = np.ones_like(b)
    eps = 1e-9
    for _ in range(max(1, n_iters)):
        u = a / (kernel @ v + eps)
        v = b / (kernel.T @ u + eps)
    transport = (u[:, None] * kernel) * v[None, :]
    dist = float(np.sum(transport * cost))
    if return_dual:
        return dist, u, v
    return dist


def _cpu_to_multi_gpu(index: faiss.Index) -> faiss.Index:
    """Shard a CPU FAISS index across all visible GPUs (select them with CUDA_VISIBLE_DEVICES)."""
    usable = list(range(faiss.get_num_gpus()))
    if not usable:
        print("[faiss-gpu] no visible GPUs; falling back to CPU")
        return index
    resources = [faiss.StandardGpuResources() for _ in usable]
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True  # shard the index across GPUs
    gpu_index = faiss.index_cpu_to_gpu_multiple_py(resources, index, co, usable)
    print(f"[faiss-gpu] index loaded onto GPUs {usable} (shard={co.shard})")
    return gpu_index


def load_index(index_dir: str) -> Tuple[faiss.Index, List[Dict]]:
    index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
    meta_path = os.path.join(index_dir, "meta.jsonl")
    meta = list(read_jsonl(meta_path))
    for item in meta:
        normalize_meta_item(item)
    if index.ntotal != len(meta):
        raise ValueError(
            f"index vector count does not match meta: {index.ntotal} vs {len(meta)}"
        )
    index = _cpu_to_multi_gpu(index)
    return index, meta


def load_queries(splits_dir: str, split: str) -> List[Dict]:
    if split == "all":
        paths = [
            os.path.join(splits_dir, "train.jsonl"),
            os.path.join(splits_dir, "dev.jsonl"),
            os.path.join(splits_dir, "test.jsonl"),
        ]
    else:
        paths = [os.path.join(splits_dir, f"{split}.jsonl")]

    queries: List[Dict] = []
    for path in paths:
        queries.extend(list(read_jsonl(path)))
    meta_map = _load_query_meta_map(splits_dir)
    for q in queries:
        key = q.get("row_id") or q.get("qid")
        meta = meta_map.get(str(key)) if key is not None else None
        query_text = q.get("query") or ""
        if not query_text and meta:
            query_text = meta.get("query") or ""
            if query_text:
                q["query"] = query_text
        year = _normalize_year(q.get("year") or (meta.get("year") if meta else None))
        if year is None:
            year = _extract_year_from_text(query_text)
        calibers = _normalize_calibers(q.get("caliber") or q.get("budget_caliber"))
        if not calibers:
            calibers = _extract_calibers_from_text(query_text)
        q["query_with_meta"] = _format_query_with_meta(query_text, year, calibers)
    return queries


class SearchResult(NamedTuple):
    coarse_meta_indices: List[int]
    coarse_evidence_ids: List[str]
    coarse_sem_scores: List[float]
    meta_indices: List[int]
    evidence_ids: List[str]
    combined_scores: List[float]
    sem_scores: List[float]
    ot_scores: List[float]
    risk_scores: List[float]
    num_scores: List[float]
    sem_norm_scores: List[float]
    ot_norm_scores: List[float]
    risk_norm_scores: List[float]
    num_norm_scores: List[float]


def search(
    index: faiss.Index,
    meta: List[Dict],
    query_vec: np.ndarray,
    query_text: str,
    top_sem_k: int,
    top_k: int,
    lambda_ot: float,
    risk_mode: str = DEFAULT_RISK_MODE,
    lambda_risk: float = 0.0,
    lambda_num: float = 0.0,
    risk_matrix: np.ndarray | None = None,
    risk_nums: np.ndarray | None = None,
    risk_cache: RiskMeasureCache | None = None,
    ot_cost: np.ndarray | None = None,
    ot_kernel: np.ndarray | None = None,
    ot_epsilon: float = DEFAULT_OT_EPS,
    ot_iters: int = DEFAULT_OT_ITERS,
    ot_top_k: int | None = None,
    score_norm: str = DEFAULT_SCORE_NORM,
    exclude_doc_prefix: str | None = None,
) -> SearchResult:
    if top_sem_k < top_k:
        raise ValueError("top_sem_k must be >= top_k")
    fetch_k = top_sem_k * 3 if exclude_doc_prefix else top_sem_k
    fetch_k = min(fetch_k, index.ntotal)
    scores, idxs = index.search(query_vec, fetch_k)
    raw_idx_list = [int(i) for i in idxs[0] if i != -1]
    raw_sem_scores = [float(s) for s in scores[0][: len(raw_idx_list)]]
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
    if not idx_list:
        return SearchResult([], [], [], [], [], [], [], [], [], [], [], [], [], [])

    coarse_evidence_ids = [meta[idx].get("evidence_id") for idx in idx_list]

    ot_scores: List[float | None] = [None for _ in idx_list]
    risk_scores = [0.0 for _ in idx_list]
    num_scores = [0.0 for _ in idx_list]
    query_num = float(np.log1p(numeric_count(query_text))) if lambda_num > 0 else 0.0

    if risk_mode == "proxy":
        if risk_matrix is None or risk_nums is None:
            raise ValueError("risk_mode='proxy' requires `risk_matrix` and `risk_nums`")
        query_risk = risk_vector(query_text, RISK_TERMS)
        if lambda_risk > 0:
            risk_scores = (risk_matrix[idx_list] @ query_risk).tolist()
        if lambda_num > 0:
            num_scores = []
            for num_val in risk_nums[idx_list]:
                denom = max(query_num, float(num_val), 1e-6)
                num_scores.append(1.0 - abs(query_num - float(num_val)) / denom)
        if lambda_ot > 0 and ot_cost is not None and ot_kernel is not None:
            if ot_top_k is None or ot_top_k <= 0 or ot_top_k >= len(idx_list):
                for i, idx in enumerate(idx_list):
                    ot_scores[i] = -sinkhorn_distance(
                        query_risk, risk_matrix[idx], ot_cost, ot_kernel, ot_iters
                    )
            else:
                for i, idx in enumerate(idx_list[:ot_top_k]):
                    ot_scores[i] = -sinkhorn_distance(
                        query_risk, risk_matrix[idx], ot_cost, ot_kernel, ot_iters
                    )
    elif risk_mode == "token":
        if risk_cache is None:
            raise ValueError("risk_mode='token' requires `risk_cache`")
        query_measure = risk_cache.measure(query_text)
        query_mean = measure_mean(query_measure)
        doc_measures: List[RiskMeasure] = []
        if lambda_risk > 0 or lambda_ot > 0:
            for idx in idx_list:
                doc_measures.append(risk_cache.measure(meta_text_for_risk(meta[idx])))
        if lambda_risk > 0 and doc_measures:
            risk_scores = [
                float(np.dot(query_mean, measure_mean(m))) for m in doc_measures
            ]
        if lambda_num > 0:
            num_scores = []
            for idx in idx_list:
                num_val = numeric_count(meta_text_for_numeric(meta[idx]))
                denom = max(query_num, float(num_val), 1e-6)
                num_scores.append(1.0 - abs(query_num - float(num_val)) / denom)
        if lambda_ot > 0 and doc_measures:
            limit = (
                len(idx_list)
                if ot_top_k is None or ot_top_k <= 0 or ot_top_k >= len(idx_list)
                else ot_top_k
            )
            for i, doc_measure in enumerate(doc_measures[:limit]):
                ot_scores[i] = -sinkhorn_distance_measure(
                    query_measure, doc_measure, ot_epsilon, ot_iters
                )
    else:
        raise ValueError(f"unknown risk_mode: {risk_mode}")

    if score_norm == "minmax":
        sem_min, sem_max = min(sem_scores), max(sem_scores)
        risk_min, risk_max = min(risk_scores), max(risk_scores)
        num_min, num_max = min(num_scores), max(num_scores)
        computed_ot = [o for o in ot_scores if o is not None]
        ot_min = min(computed_ot) if computed_ot else 0.0
        ot_max = max(computed_ot) if computed_ot else 0.0

        def _minmax(x: float, lo: float, hi: float) -> float:
            if hi - lo < 1e-12:
                return 0.0
            return (x - lo) / (hi - lo)

        sem_vals = [_minmax(s, sem_min, sem_max) for s in sem_scores]
        risk_vals = [_minmax(r, risk_min, risk_max) for r in risk_scores]
        num_vals = [_minmax(n, num_min, num_max) for n in num_scores]
        if computed_ot and ot_max - ot_min < 1e-12:
            ot_vals = [1.0 if o is not None else 0.0 for o in ot_scores]
        else:
            ot_vals = [
                _minmax(o, ot_min, ot_max) if o is not None else 0.0 for o in ot_scores
            ]
    elif score_norm == "none":
        sem_vals = sem_scores
        risk_vals = risk_scores
        ot_vals = [o if o is not None else 0.0 for o in ot_scores]
        num_vals = num_scores
    else:
        raise ValueError(f"unknown score_norm: {score_norm}")

    combined = [
        s
        + (lambda_ot * o if lambda_ot > 0 else 0.0)
        + (lambda_risk * r if lambda_risk > 0 else 0.0)
        + (lambda_num * n if lambda_num > 0 else 0.0)
        for s, o, r, n in zip(sem_vals, ot_vals, risk_vals, num_vals)
    ]

    ranked = sorted(
        zip(
            idx_list,
            combined,
            sem_scores,
            ot_scores,
            risk_scores,
            num_scores,
            sem_vals,
            ot_vals,
            risk_vals,
            num_vals,
        ),
        key=lambda x: x[1],
        reverse=True,
    )
    ranked = ranked[:top_k]

    meta_indices = [i for i, *_ in ranked]
    evidence_ids = [meta[i].get("evidence_id") for i in meta_indices]
    combined_scores = [c for _, c, *_ in ranked]
    sem_out = [s for _, _, s, *_ in ranked]
    ot_out = [o if o is not None else 0.0 for _, _, _, o, *_ in ranked]
    risk_out = [r for _, _, _, _, r, *_ in ranked]
    num_out = [n for _, _, _, _, _, n, *_ in ranked]
    sem_norm_out = [s for _, _, _, _, _, _, s, *_ in ranked]
    ot_norm_out = [o for _, _, _, _, _, _, _, o, *_ in ranked]
    risk_norm_out = [r for _, _, _, _, _, _, _, _, r, _ in ranked]
    num_norm_out = [n for _, _, _, _, _, _, _, _, _, n in ranked]
    return SearchResult(
        coarse_meta_indices=idx_list,
        coarse_evidence_ids=coarse_evidence_ids,
        coarse_sem_scores=sem_scores,
        meta_indices=meta_indices,
        evidence_ids=evidence_ids,
        combined_scores=combined_scores,
        sem_scores=sem_out,
        ot_scores=ot_out,
        risk_scores=risk_out,
        num_scores=num_out,
        sem_norm_scores=sem_norm_out,
        ot_norm_scores=ot_norm_out,
        risk_norm_scores=risk_norm_out,
        num_norm_scores=num_norm_out,
    )

