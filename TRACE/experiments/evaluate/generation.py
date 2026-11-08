#! /usr/bin/env python3
"""
Evaluate TRACE generation with AC / QAC metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Tuple


from trace_rag.generate import consistency_score, extract_triples
from trace_rag.retrieval import embed_texts, meta_text_for_risk, normalize_meta_item
from trace_rag.utils.experiment_layout import validate_non_legacy_output_path


DEFAULT_DATASET_TAG = "audit-ociqa"
DEFAULT_MODEL_TAG = "shared"


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def default_pred_path(dataset_tag: str, model_tag: str, method: str) -> str:
    filename = f"{normalize_tag(method, 'trace')}.pred.jsonl"
    return os.path.join("runs", dataset_tag, model_tag, "generation", filename)


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_unique_predictions(path: str) -> Tuple[List[Dict], int]:
    rows_by_key: Dict[str, Dict] = {}
    duplicate_rows = 0
    line_idx = 0
    for item in read_jsonl(path):
        key = item.get("row_id") or item.get("qid")
        if key is None:
            key = f"__line__{line_idx}"
        key = str(key)
        if key in rows_by_key:
            duplicate_rows += 1
        rows_by_key[key] = item
        line_idx += 1
    return list(rows_by_key.values()), duplicate_rows


def load_meta(index_dir: str) -> Dict[str, Dict]:
    meta_path = os.path.join(index_dir, "meta.jsonl")
    meta = {}
    for raw_item in read_jsonl(meta_path):
        item = normalize_meta_item(dict(raw_item))
        ev_id = item.get("evidence_id")
        if not ev_id:
            continue
        caliber = item.get("caliber")
        if caliber is None:
            caliber = item.get("budget_caliber")
        meta[ev_id] = {
            "evidence_id": ev_id,
            "text": item.get("text") or "",
            "text_with_meta": meta_text_for_risk(item),
            "year": item.get("year"),
            "caliber": caliber,
        }
    return meta


def semantic_scores(claim: str, evidence: List[Dict], server_url: str) -> List[float]:
    if not evidence:
        return []
    texts = [
        claim,
        *[
            e.get("text_with_meta") or e.get("text_for_embedding") or e.get("text", "")
            for e in evidence
        ],
    ]
    vecs = embed_texts(server_url, texts, batch_size=16)
    claim_vec = vecs[0]
    ev_vecs = vecs[1:]
    scores = claim_vec @ ev_vecs.T
    return scores.tolist() if scores.size else []


def _claim_support_qac_cons(
    support_item: Dict,
    meta: Dict[str, Dict],
    server_url: str,
) -> float | None:
    if not support_item.get("has_numeric_attr"):
        return None

    cached = support_item.get("qac_cons")
    if cached is not None:
        return float(cached)

    claim = str(support_item.get("claim") or "").strip()
    if not claim:
        return None

    cites = support_item.get("cites") or []
    evidence = [meta[c] for c in cites if c in meta]
    if not evidence:
        best_evidence_id = support_item.get("best_evidence_id")
        if best_evidence_id and best_evidence_id in meta:
            evidence = [meta[best_evidence_id]]
    if evidence:
        claim_triples = extract_triples(claim)
        if not claim_triples:
            return None
        tri_scores = [
            consistency_score(
                claim,
                e,
                server_url=server_url,
                claim_triples=claim_triples,
            )
            for e in evidence
        ]
        if tri_scores:
            return max(float(score) for score in tri_scores)

    cons_val = support_item.get("Cons")
    return float(cons_val) if cons_val is not None else None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _collect_claims(oci: Dict) -> List[str]:
    sections = ("Observation", "Cause", "Impact")
    claims: List[str] = []
    for section in sections:
        items = oci.get(section, [])
        if isinstance(items, dict) or isinstance(items, str):
            items = [items]
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, dict):
                claim = item.get("claim")
            else:
                claim = item
            if claim is None:
                continue
            claim_str = str(claim).strip()
            if claim_str:
                claims.append(claim_str)
    return claims


def _oci_to_text(oci) -> str:
    if isinstance(oci, str):
        return oci.strip()
    return "\n".join(_collect_claims(oci))


def _iter_role_claim_items(oci: Dict, role_name: str) -> Iterable[Dict]:
    items = oci.get(role_name, []) if isinstance(oci, dict) else []
    if isinstance(items, dict):
        items = [items]
    elif isinstance(items, str):
        items = [{"claim": items}]
    elif not isinstance(items, list):
        items = []
    for item in items:
        if isinstance(item, dict):
            claim = str(item.get("claim") or "").strip()
            cites = item.get("cites") or []
            yield {"claim": claim, "cites": cites}
        else:
            claim = str(item or "").strip()
            yield {"claim": claim, "cites": []}


def _align_claim_with_evidence(
    claim: str,
    evidence: List[Dict],
    server_url: str,
    lambda_attr: float,
) -> float:
    if not claim or not evidence:
        return 0.0
    claim_triples = extract_triples(claim)
    has_triples = bool(claim_triples)
    tri_scores = []
    if has_triples:
        tri_scores = [
            consistency_score(
                claim,
                e,
                server_url=server_url,
                claim_triples=claim_triples,
            )
            for e in evidence
        ]
    sem_scores = semantic_scores(claim, evidence, server_url)
    if has_triples and sem_scores:
        align_scores = [
            lambda_attr * tri_score + (1.0 - lambda_attr) * sem_scores[idx]
            for idx, tri_score in enumerate(tri_scores)
        ]
        return max((float(score) for score in align_scores), default=0.0)
    if sem_scores:
        return max(float(score) for score in sem_scores)
    if tri_scores:
        return max(float(score) for score in tri_scores)
    return 0.0


def _char_f1(pred: str, gold: str) -> float:
    pred_norm = _normalize_text(pred)
    gold_norm = _normalize_text(gold)
    if not pred_norm or not gold_norm:
        return 0.0
    pred_cnt = Counter(pred_norm)
    gold_cnt = Counter(gold_norm)
    overlap = sum((pred_cnt & gold_cnt).values())
    if overlap <= 0:
        return 0.0
    precision = overlap / max(len(pred_norm), 1)
    recall = overlap / max(len(gold_norm), 1)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ch in a:
        curr = [0]
        for j, ch_b in enumerate(b, start=1):
            if ch == ch_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def _rouge_l(pred: str, gold: str, max_chars: int = 2000) -> float:
    pred_norm = _normalize_text(pred)
    gold_norm = _normalize_text(gold)
    if not pred_norm or not gold_norm:
        return 0.0
    pred_seq = list(pred_norm[:max_chars])
    gold_seq = list(gold_norm[:max_chars])
    lcs = _lcs_length(pred_seq, gold_seq)
    if lcs <= 0:
        return 0.0
    precision = lcs / max(len(pred_seq), 1)
    recall = lcs / max(len(gold_seq), 1)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def _load_gold_oci(path: str) -> Dict[str, object]:
    gold_map: Dict[str, object] = {}
    for item in read_jsonl(path):
        key = item.get("row_id") or item.get("qid")
        if not key:
            continue
        gold = item.get("gold_oci") or {}
        gold_map[str(key)] = gold
    return gold_map


def _gold_roles(gold_oci: object) -> List[str]:
    if not isinstance(gold_oci, dict):
        return []
    roles = []
    for role_name in ("Observation", "Cause", "Impact"):
        claims = _collect_claims({role_name: gold_oci.get(role_name, [])})
        if claims:
            roles.append(role_name)
    return roles


def _load_qrels_evidence(qrels_path: str, meta: Dict[str, Dict]) -> Dict[str, List[Dict]]:
    """Load qrels and resolve evidence IDs to meta dicts for gold-evidence AC."""
    qrels_evidence: Dict[str, List[Dict]] = {}
    for item in read_jsonl(qrels_path):
        row_id = item.get("row_id")
        if not row_id:
            continue
        gold_ids = item.get("gold_evidence_ids") or []
        evidence = [meta[eid] for eid in gold_ids if eid in meta]
        if evidence:
            qrels_evidence[str(row_id)] = evidence
    return qrels_evidence


def evaluate_generation_metrics(
    pred_path: str,
    index_dir: str,
    server_url: str,
    lambda_attr: float,
    ac_threshold: float,
    qac_threshold: float,
    gold_path: str | None = None,
    qrels_path: str | None = None,
    max_rouge_chars: int = 2000,
) -> Dict[str, float]:
    meta = load_meta(index_dir)
    pred_rows, duplicate_pred_rows = load_unique_predictions(pred_path)
    qrels_evidence = (
        _load_qrels_evidence(qrels_path, meta)
        if qrels_path and os.path.exists(qrels_path)
        else {}
    )
    total_claims = 0
    supported_claims = 0
    total_claims_gold = 0
    supported_claims_gold = 0
    total_numeric = 0
    supported_numeric = 0
    f1_sum = 0.0
    rouge_sum = 0.0
    paired = 0
    missing_gold = 0
    ocr_sum = 0.0
    ocr_paired = 0

    gold_map = _load_gold_oci(gold_path) if gold_path and os.path.exists(gold_path) else {}

    for item in pred_rows:
        oci = item.get("prediction_oci") or {}
        claim_support = item.get("claim_support") or {}
        has_claim_support = any(
            claim_support.get(role_name)
            for role_name in ("Observation", "Cause", "Impact")
        )
        fallback_evidence_ids = item.get("used_evidence_ids") or []
        row_key = str(item.get("row_id") or item.get("qid") or "")
        gold_ev = qrels_evidence.get(row_key, []) if qrels_evidence else []
        if has_claim_support:
            for section in ("Observation", "Cause", "Impact"):
                for support_item in claim_support.get(section, []) or []:
                    claim = support_item.get("claim", "")
                    if not claim:
                        continue
                    total_claims += 1
                    cites = support_item.get("cites") or []
                    evidence = [meta[c] for c in cites if c in meta]
                    if not evidence:
                        best_eid = support_item.get("best_evidence_id")
                        if best_eid and best_eid in meta:
                            evidence = [meta[best_eid]]
                    align = _align_claim_with_evidence(
                        claim=claim,
                        evidence=evidence,
                        server_url=server_url,
                        lambda_attr=lambda_attr,
                    )
                    if align >= ac_threshold:
                        supported_claims += 1
                    if gold_ev:
                        total_claims_gold += 1
                        align_gold = _align_claim_with_evidence(
                            claim=claim,
                            evidence=gold_ev,
                            server_url=server_url,
                            lambda_attr=lambda_attr,
                        )
                        if align_gold >= ac_threshold:
                            supported_claims_gold += 1
                    if support_item.get("has_numeric_attr"):
                        total_numeric += 1
                        cons_val = _claim_support_qac_cons(support_item, meta, server_url)
                        if cons_val is not None and float(cons_val) >= qac_threshold:
                            supported_numeric += 1
            key = item.get("row_id") or item.get("qid")
            if gold_map:
                gold_oci = gold_map.get(str(key)) if key is not None else None
                if gold_oci is None:
                    missing_gold += 1
                else:
                    pred_text = _oci_to_text(oci)
                    gold_text = _oci_to_text(gold_oci)
                    f1_sum += _char_f1(pred_text, gold_text)
                    rouge_sum += _rouge_l(pred_text, gold_text, max_chars=max_rouge_chars)
                    paired += 1
                    gold_roles = _gold_roles(gold_oci)
                    if gold_roles:
                        supported_roles = 0
                        for role_name in gold_roles:
                            for support_item in claim_support.get(role_name, []) or []:
                                claim_text = support_item.get("claim", "")
                                if not claim_text:
                                    continue
                                role_cites = support_item.get("cites") or []
                                role_evidence = [meta[c] for c in role_cites if c in meta]
                                if not role_evidence:
                                    best_eid = support_item.get("best_evidence_id")
                                    if best_eid and best_eid in meta:
                                        role_evidence = [meta[best_eid]]
                                role_align = _align_claim_with_evidence(
                                    claim=claim_text,
                                    evidence=role_evidence,
                                    server_url=server_url,
                                    lambda_attr=lambda_attr,
                                )
                                if role_align >= ac_threshold:
                                    supported_roles += 1
                                    break
                        ocr_sum += supported_roles / len(gold_roles)
                        ocr_paired += 1
            continue
        for section in ("Observation", "Cause", "Impact"):
            for claim_item in oci.get(section, []) or []:
                claim = claim_item.get("claim", "") if isinstance(claim_item, dict) else ""
                if not claim:
                    continue
                total_claims += 1
                cites = claim_item.get("cites") if isinstance(claim_item, dict) else []
                evidence_ids = cites or fallback_evidence_ids
                evidence = [meta[c] for c in evidence_ids if c in meta]
                if not evidence:
                    continue

                claim_triples = extract_triples(claim)
                has_triples = bool(claim_triples)
                tri_scores = [
                    consistency_score(
                        claim,
                        e,
                        server_url=server_url,
                        claim_triples=claim_triples,
                    )
                    for e in evidence
                ]
                sem_scores = semantic_scores(claim, evidence, server_url)
                sem_max = max(sem_scores) if sem_scores else 0.0

                if has_triples:
                    align_scores = [
                        lambda_attr * t + (1 - lambda_attr) * sem_scores[i]
                        for i, t in enumerate(tri_scores)
                    ] if sem_scores else []
                    align = max(align_scores) if align_scores else 0.0
                    if align >= ac_threshold:
                        supported_claims += 1
                else:
                    if sem_max >= ac_threshold:
                        supported_claims += 1

                if gold_ev:
                    total_claims_gold += 1
                    align_gold = _align_claim_with_evidence(
                        claim=claim,
                        evidence=gold_ev,
                        server_url=server_url,
                        lambda_attr=lambda_attr,
                    )
                    if align_gold >= ac_threshold:
                        supported_claims_gold += 1

                if has_triples:
                    total_numeric += 1
                    if tri_scores and max(tri_scores) >= qac_threshold:
                        supported_numeric += 1

        if gold_map:
            key = item.get("row_id") or item.get("qid")
            gold_oci = gold_map.get(str(key)) if key is not None else None
            if gold_oci is None:
                missing_gold += 1
            else:
                pred_text = _oci_to_text(oci)
                gold_text = _oci_to_text(gold_oci)
                f1_sum += _char_f1(pred_text, gold_text)
                rouge_sum += _rouge_l(pred_text, gold_text, max_chars=max_rouge_chars)
                paired += 1
                gold_roles = _gold_roles(gold_oci)
                if gold_roles:
                    supported_roles = 0
                    for role_name in gold_roles:
                        role_supported = False
                        for claim_item in _iter_role_claim_items(oci, role_name):
                            claim = claim_item.get("claim") or ""
                            evidence_ids = claim_item.get("cites") or fallback_evidence_ids
                            evidence = [meta[c] for c in evidence_ids if c in meta]
                            align = _align_claim_with_evidence(
                                claim=claim,
                                evidence=evidence,
                                server_url=server_url,
                                lambda_attr=lambda_attr,
                            )
                            if align >= ac_threshold:
                                role_supported = True
                                break
                        if role_supported:
                            supported_roles += 1
                    ocr_sum += supported_roles / len(gold_roles)
                    ocr_paired += 1

    ac = supported_claims / total_claims if total_claims else 0.0
    qac = supported_numeric / total_numeric if total_numeric else 0.0
    ac_gold = supported_claims_gold / total_claims_gold if total_claims_gold else None
    metrics = {
        "ac": ac,
        "qac": qac,
        "claims": total_claims,
        "numeric_claims": total_numeric,
        "duplicate_pred_rows": duplicate_pred_rows,
        "evaluated_rows": len(pred_rows),
    }
    if ac_gold is not None:
        metrics["ac_gold"] = ac_gold
        metrics["claims_gold"] = total_claims_gold
    if gold_map:
        char_f1 = f1_sum / paired if paired else 0.0
        metrics["char_f1"] = char_f1
        metrics["f1"] = char_f1
        metrics["rouge_l"] = rouge_sum / paired if paired else 0.0
        metrics["ocr"] = ocr_sum / ocr_paired if ocr_paired else 0.0
        metrics["paired"] = paired
        metrics["missing_gold"] = missing_gold
        metrics["ocr_paired"] = ocr_paired
        metrics["max_rouge_chars"] = max_rouge_chars
    return metrics


def evaluate_generation(
    pred_path: str,
    index_dir: str,
    server_url: str,
    lambda_attr: float,
    ac_threshold: float,
    qac_threshold: float,
) -> Tuple[float, float, int, int]:
    metrics = evaluate_generation_metrics(
        pred_path=pred_path,
        index_dir=index_dir,
        server_url=server_url,
        lambda_attr=lambda_attr,
        ac_threshold=ac_threshold,
        qac_threshold=qac_threshold,
    )
    return (
        float(metrics.get("ac", 0.0)),
        float(metrics.get("qac", 0.0)),
        int(metrics.get("claims", 0)),
        int(metrics.get("numeric_claims", 0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TRACE generation (AC/QAC)")
    parser.add_argument(
        "--dataset-tag",
        default=DEFAULT_DATASET_TAG,
        help="Dataset tag for default prediction path",
    )
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        help="Model tag for default prediction path when --pred is omitted",
    )
    parser.add_argument(
        "--pred",
        default=None,
        help="Prediction JSONL path (default: runs/<dataset>/<model>/generation/<method>.pred.jsonl)",
    )
    parser.add_argument("--method", default="TRACE", help="Method name for default prediction path")
    parser.add_argument("--index-dir", default="runs/index", help="Index directory")
    parser.add_argument(
        "--server-url",
        default=os.getenv(
            "EMBED_SERVER_URL",
            "http://localhost:8081/v1/embeddings",
        ),
        help="Embedding server URL",
    )
    parser.add_argument("--lambda-attr", type=float, default=0.7, help="Attribute consistency weight")
    parser.add_argument("--ac-threshold", type=float, default=0.2, help="AC threshold")
    parser.add_argument("--qac-threshold", type=float, default=0.2, help="QAC threshold")
    parser.add_argument("--lambda-tri", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ncc-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--gold",
        default="datasets-public/audit-ociqa-demo/qa.gold.jsonl",
        help="Gold JSONL path (for F1/ROUGE-L)",
    )
    parser.add_argument(
        "--qrels",
        default=None,
        help="Qrels JSONL path (for ac_gold — alignment against gold evidence)",
    )
    parser.add_argument(
        "--max-rouge-chars",
        type=int,
        default=2000,
        help="Max chars per text for ROUGE-L",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_path = args.pred or default_pred_path(
        args.dataset_tag,
        normalize_tag(args.model_tag, DEFAULT_MODEL_TAG),
        args.method,
    )
    lambda_attr = args.lambda_tri if args.lambda_tri is not None else args.lambda_attr
    qac_threshold = (
        args.ncc_threshold if args.ncc_threshold is not None else args.qac_threshold
    )
    metrics = evaluate_generation_metrics(
        pred_path=pred_path,
        index_dir=args.index_dir,
        server_url=args.server_url,
        lambda_attr=lambda_attr,
        ac_threshold=args.ac_threshold,
        qac_threshold=qac_threshold,
        gold_path=args.gold,
        qrels_path=args.qrels,
        max_rouge_chars=args.max_rouge_chars,
    )
    print(f"AC: {metrics['ac']:.4f} (claims={metrics['claims']})")
    if "ac_gold" in metrics:
        print(f"AC_gold: {metrics['ac_gold']:.4f} (claims_gold={metrics['claims_gold']})")
    print(f"QAC: {metrics['qac']:.4f} (numeric_claims={metrics['numeric_claims']})")
    duplicate_pred_rows = int(metrics.get("duplicate_pred_rows", 0))
    if duplicate_pred_rows > 0:
        print(
            f"[WARN] found {duplicate_pred_rows} duplicate prediction rows; "
            "deduplicated by row_id/qid (keeping the last occurrence)"
        )
    if "f1" in metrics:
        print(f"Char-F1: {metrics['char_f1']:.4f} (paired={metrics['paired']})")
        print(f"ROUGE-L: {metrics['rouge_l']:.4f}")
        print(f"OCR: {metrics['ocr']:.4f} (paired={metrics['ocr_paired']})")
        missing_gold = int(metrics.get("missing_gold", 0))
        if missing_gold > 0:
            print(
                "[WARN] some prediction rows are missing gold_oci; "
                f"Char-F1 / ROUGE-L / OCR are computed over paired={metrics['paired']} only "
                f"(missing_gold={missing_gold})"
            )
    if args.output:
        validate_non_legacy_output_path(args.output, label="generation eval output")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
