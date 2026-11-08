#! /usr/bin/env python3
"""
Evaluate retrieval results (Recall@K, nDCG@K, RR@K).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Set

from trace_rag.utils.experiment_layout import DEFAULT_AUDIT_SPLITS_DIR, validate_non_legacy_output_path

DEFAULT_RESULTS = []
DEFAULT_SPLITS_DIR = DEFAULT_AUDIT_SPLITS_DIR
DEFAULT_QRELS = "datasets-public/audit-ociqa-demo/gold.qrels.jsonl"
DEFAULT_GOLD = "datasets-public/audit-ociqa-demo/qa.gold.jsonl"
DEFAULT_K_LIST = [5, 10, 20]


def _read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_split_ids(splits_dir: str, split: str) -> Set[str]:
    if split == "all":
        paths = [
            os.path.join(splits_dir, "train.jsonl"),
            os.path.join(splits_dir, "dev.jsonl"),
            os.path.join(splits_dir, "test.jsonl"),
        ]
    else:
        paths = [os.path.join(splits_dir, f"{split}.jsonl")]
    row_ids: Set[str] = set()
    for path in paths:
        for item in _read_jsonl(path):
            row_id = item.get("row_id")
            if row_id:
                row_ids.add(row_id)
    return row_ids


def _load_qrels(path: str, allowed_ids: Set[str]) -> Dict[str, Set[str]]:
    qrels: Dict[str, Set[str]] = {}
    for item in _read_jsonl(path):
        row_id = item.get("row_id")
        if not row_id or row_id not in allowed_ids:
            continue
        gold = set(item.get("gold_evidence_ids") or [])
        if gold:
            qrels[row_id] = gold
    return qrels


def _load_role_evidence_map(path: str, allowed_ids: Set[str]) -> Dict[str, Dict[str, Set[str]]]:
    role_map: Dict[str, Dict[str, Set[str]]] = {}
    for item in _read_jsonl(path):
        row_id = item.get("row_id")
        if not row_id or row_id not in allowed_ids:
            continue
        gold_oci = item.get("gold_oci") or {}
        row_roles: Dict[str, Set[str]] = {}
        for role_name in ("Observation", "Cause", "Impact"):
            raw_items = gold_oci.get(role_name, [])
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            elif isinstance(raw_items, str):
                raw_items = [{"claim": raw_items}]
            elif not isinstance(raw_items, list):
                raw_items = []
            cites: Set[str] = set()
            has_claim = False
            for claim_item in raw_items:
                if isinstance(claim_item, dict):
                    claim = str(claim_item.get("claim") or "").strip()
                    cite_list = claim_item.get("cites") or []
                else:
                    claim = str(claim_item or "").strip()
                    cite_list = []
                if claim:
                    has_claim = True
                for evidence_id in cite_list:
                    if evidence_id:
                        cites.add(str(evidence_id))
            if has_claim:
                row_roles[role_name] = cites
        if row_roles:
            role_map[row_id] = row_roles
    return role_map


def _load_results(path: str, allowed_ids: Set[str]) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    for item in _read_jsonl(path):
        row_id = item.get("row_id")
        if not row_id or row_id not in allowed_ids:
            continue
        results[row_id] = item.get("topk_evidence_ids") or []
    return results


def _dcg(hits: List[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(hits, start=1):
        if rel:
            score += 1.0 / math.log2(idx + 1)
    return score


def _ndcg_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = [1 if doc_id in gold else 0 for doc_id in pred[:k]]
    dcg = _dcg(hits)
    ideal_hits = [1] * min(len(gold), k)
    idcg = _dcg(ideal_hits)
    return dcg / idcg if idcg > 0 else 0.0


def _hit_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    """Binary hit rate: 1 if any gold doc appears in pred[:k], else 0."""
    if not gold:
        return 0.0
    return 1.0 if any(doc_id in gold for doc_id in pred[:k]) else 0.0


def _recall_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    """Fractional recall: |pred[:k] ∩ gold| / |gold|."""
    if not gold:
        return 0.0
    hits = sum(1 for doc_id in pred[:k] if doc_id in gold)
    return hits / len(gold)


def _rr_at_k(pred: List[str], role_gold: Dict[str, Set[str]], k: int) -> float:
    if not role_gold:
        return 0.0
    pred_k = pred[:k]
    covered_roles = 0
    for evidence_ids in role_gold.values():
        if not evidence_ids:
            continue
        if any(doc_id in evidence_ids for doc_id in pred_k):
            covered_roles += 1
    return covered_roles / max(len(role_gold), 1)


def _infer_name(result_path: str) -> str:
    for item in _read_jsonl(result_path):
        method = item.get("method")
        if method:
            return str(method)
        break
    return Path(result_path).stem


def _parse_list(raw: str, cast_fn) -> List:
    items = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast_fn(part))
    return items


def _expand_results(results: List[str]) -> List[str]:
    items: List[str] = []
    for entry in results:
        items.extend([part.strip() for part in entry.split(",") if part.strip()])
    return items


def evaluate_results(
    result_path: str,
    qrels: Dict[str, Set[str]],
    role_gold: Dict[str, Dict[str, Set[str]]],
    k_list: List[int],
) -> Dict[str, float]:
    allowed_ids = set(qrels.keys())
    results = _load_results(result_path, allowed_ids)

    metrics = {}
    for k in k_list:
        hits = []
        recalls = []
        ndcgs = []
        role_recalls = []
        for row_id, gold in qrels.items():
            pred = results.get(row_id, [])
            hits.append(_hit_at_k(pred, gold, k))
            recalls.append(_recall_at_k(pred, gold, k))
            ndcgs.append(_ndcg_at_k(pred, gold, k))
            if row_id in role_gold:
                role_recalls.append(_rr_at_k(pred, role_gold[row_id], k))
        metrics[f"Hit@{k}"] = sum(hits) / len(hits) if hits else 0.0
        metrics[f"Recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        metrics[f"nDCG@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        metrics[f"RR@{k}"] = sum(role_recalls) / len(role_recalls) if role_recalls else 0.0
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval results (Recall@K, nDCG@K, RR@K)")
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help="Result JSONL paths (space or comma separated)",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory to scan for *.jsonl results",
    )
    parser.add_argument(
        "--splits-dir", default=DEFAULT_SPLITS_DIR, help="Splits directory"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to evaluate (e.g. train, dev, test, test-mini, all)",
    )
    parser.add_argument("--qrels", default=DEFAULT_QRELS, help="Qrels JSONL path")
    parser.add_argument(
        "--gold",
        default=DEFAULT_GOLD,
        help="Gold QA JSONL path used to derive role evidence map for RR@K",
    )
    parser.add_argument(
        "--k-list",
        default="5,10,20",
        help="Recall/nDCG K list (comma separated)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path for metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = _expand_results(args.results) if args.results else []
    if args.results_dir:
        for path in sorted(Path(args.results_dir).glob("*.jsonl")):
            results.append(str(path))
    if not results:
        results = list(DEFAULT_RESULTS)

    k_list = _parse_list(args.k_list, int) or DEFAULT_K_LIST
    split_ids = _load_split_ids(args.splits_dir, args.split)
    qrels = _load_qrels(args.qrels, split_ids)
    if not qrels:
        raise ValueError("no qrels were loaded; check the --split and --qrels paths")
    role_gold = _load_role_evidence_map(args.gold, split_ids)

    all_metrics = {}
    for path in results:
        name = _infer_name(path)
        metrics = evaluate_results(path, qrels, role_gold, k_list)
        all_metrics[name] = metrics
        print(f"[{name}] {path}")
        for k in k_list:
            print(f"  Hit@{k}: {metrics[f'Hit@{k}']:.4f}")
            print(f"  Recall@{k}: {metrics[f'Recall@{k}']:.4f}")
            print(f"  nDCG@{k}: {metrics[f'nDCG@{k}']:.4f}")
            print(f"  RR@{k}: {metrics[f'RR@{k}']:.4f}")

    if args.output:
        validate_non_legacy_output_path(args.output, label="retrieval eval output")
        payload = {
            "results": results,
            "split": args.split,
            "k_list": k_list,
            "metrics": all_metrics,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
