#! /usr/bin/env python3
"""
Evaluate robustness stress test outputs, including VAR and PHR.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from typing import Dict, Iterable, List

from experiments.evaluate.generation import evaluate_generation_metrics
from trace_rag.utils.phase3 import read_jsonl, write_json


def _iter_claim_support(item: Dict) -> Iterable[Dict]:
    claim_support = item.get("claim_support") or {}
    for section in ("Observation", "Cause", "Impact"):
        for support_item in claim_support.get(section, []) or []:
            yield support_item


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).replace(",", "")


def _tokens_for_perturbation(stress_item: Dict) -> List[str]:
    tokens = []
    for key in ("perturbed_value_text", "perturbed_unit_text", "perturbed_year_text", "perturbed_boundary_text"):
        value = stress_item.get(key)
        if value:
            tokens.append(_normalize(value))
    return [token for token in tokens if token]


def _claim_mentions_perturbation(claim: str, stress_item: Dict) -> bool:
    claim_norm = _normalize(claim)
    if not claim_norm:
        return False
    return any(token and token in claim_norm for token in _tokens_for_perturbation(stress_item))


def evaluate_stress_outputs(
    pred_path: str,
    stress_dataset_path: str,
    index_dir: str,
    server_url: str,
    gold_path: str,
    lambda_attr: float,
    ac_threshold: float,
    qac_threshold: float,
) -> Dict:
    metrics = evaluate_generation_metrics(
        pred_path=pred_path,
        index_dir=index_dir,
        server_url=server_url,
        lambda_attr=lambda_attr,
        ac_threshold=ac_threshold,
        qac_threshold=qac_threshold,
        gold_path=gold_path,
    )
    stress_map = {
        str(item.get("row_id")): item
        for item in read_jsonl(stress_dataset_path)
        if item.get("row_id")
    }

    total = 0
    var_count = 0
    phr_count = 0
    by_type = defaultdict(lambda: {"count": 0, "VAR": 0, "PHR": 0})

    for item in read_jsonl(pred_path):
        row_id = str(item.get("row_id"))
        if row_id not in stress_map:
            continue
        stress_item = stress_map[row_id]
        perturb_type = stress_item.get("perturb_type") or "unknown"
        total += 1
        by_type[perturb_type]["count"] += 1

        prediction_oci = item.get("prediction_oci") or {}
        abstain_flag = bool(item.get("abstain_flag"))
        if not abstain_flag:
            abstain_flag = not any(prediction_oci.get(section) for section in ("Observation", "Cause", "Impact"))
        if abstain_flag:
            var_count += 1
            by_type[perturb_type]["VAR"] += 1
            continue

        phr_hit = False
        for support_item in _iter_claim_support(item):
            claim = support_item.get("claim") or ""
            if not _claim_mentions_perturbation(claim, stress_item):
                continue
            supported = bool(support_item.get("supported"))
            qac_pass = bool(support_item.get("qac_pass"))
            has_numeric_attr = bool(support_item.get("has_numeric_attr"))
            if (not supported) or (has_numeric_attr and not qac_pass):
                phr_hit = True
                break
        if phr_hit:
            phr_count += 1
            by_type[perturb_type]["PHR"] += 1

    metrics["var"] = (var_count / total) if total else 0.0
    metrics["phr"] = (phr_count / total) if total else 0.0
    metrics["stress_rows"] = total
    metrics["by_perturb_type"] = {
        key: {
            "count": value["count"],
            "VAR": (value["VAR"] / value["count"]) if value["count"] else 0.0,
            "PHR": (value["PHR"] / value["count"]) if value["count"] else 0.0,
        }
        for key, value in sorted(by_type.items())
    }
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stress test outputs")
    parser.add_argument("--pred", required=True, help="Stress prediction JSONL path")
    parser.add_argument("--stress-dataset", required=True, help="Stress dataset JSONL path")
    parser.add_argument("--index-dir", default="runs/index")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8081/v1/embeddings",
    )
    parser.add_argument("--gold", default="datasets-public/audit-ociqa-demo/qa.gold.jsonl")
    parser.add_argument("--lambda-attr", type=float, default=0.7)
    parser.add_argument("--ac-threshold", type=float, default=0.2)
    parser.add_argument("--qac-threshold", type=float, default=0.2)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_stress_outputs(
        pred_path=args.pred,
        stress_dataset_path=args.stress_dataset,
        index_dir=args.index_dir,
        server_url=args.server_url,
        gold_path=args.gold,
        lambda_attr=args.lambda_attr,
        ac_threshold=args.ac_threshold,
        qac_threshold=args.qac_threshold,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        write_json(args.output, metrics)


if __name__ == "__main__":
    main()
