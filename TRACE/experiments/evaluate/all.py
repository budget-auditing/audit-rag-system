#! /usr/bin/env python3
"""
Unified evaluation entry for retrieval and generation results.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from trace_rag.utils.experiment_layout import DEFAULT_AUDIT_SPLITS_DIR, validate_non_legacy_output_path
from experiments.evaluate.generation import evaluate_generation_metrics
from experiments.evaluate.retrieval import (
    _infer_name,
    _load_qrels,
    _load_role_evidence_map,
    _load_split_ids,
    _parse_list,
    evaluate_results,
)


def _expand_paths(raw: List[str]) -> List[str]:
    items: List[str] = []
    for entry in raw:
        items.extend([part.strip() for part in entry.split(",") if part.strip()])
    return items


def _infer_generation_name(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            method = item.get("method")
            if method:
                return str(method)
            break
    return Path(path).stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified evaluation for retrieval/generation")
    parser.add_argument("--retrieval", nargs="*", default=None, help="Retrieval result JSONL paths")
    parser.add_argument("--generation", nargs="*", default=None, help="Generation result JSONL paths")
    parser.add_argument(
        "--splits-dir",
        default=DEFAULT_AUDIT_SPLITS_DIR,
        help="Splits directory for retrieval eval",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to evaluate (e.g. train, dev, test, test-mini, all)",
    )
    parser.add_argument(
        "--qrels",
        default="datasets-public/audit-ociqa-demo/gold.qrels.jsonl",
        help="Qrels JSONL path",
    )
    parser.add_argument(
        "--k-list",
        default="5,10,20",
        help="Recall/nDCG K list (comma separated)",
    )
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
    parser.add_argument(
        "--gold",
        default="datasets-public/audit-ociqa-demo/qa.gold.jsonl",
        help="Gold JSONL path (for F1/ROUGE-L)",
    )
    parser.add_argument(
        "--max-rouge-chars",
        type=int,
        default=2000,
        help="Max chars per text for ROUGE-L",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: Dict[str, Dict] = {}

    if args.retrieval:
        retrieval_paths = _expand_paths(args.retrieval)
        split_ids = _load_split_ids(args.splits_dir, args.split)
        qrels = _load_qrels(args.qrels, split_ids)
        role_gold = _load_role_evidence_map(args.gold, split_ids)
        if not qrels:
            raise ValueError("no qrels were loaded; check the --split and --qrels paths")
        k_list = _parse_list(args.k_list, int)
        for path in retrieval_paths:
            name = _infer_name(path)
            metrics = evaluate_results(path, qrels, role_gold, k_list)
            results.setdefault("retrieval", {})[name] = metrics

    if args.generation:
        generation_paths = _expand_paths(args.generation)
        for path in generation_paths:
            name = _infer_generation_name(path)
            metrics = evaluate_generation_metrics(
                pred_path=path,
                index_dir=args.index_dir,
                server_url=args.server_url,
                lambda_attr=args.lambda_attr,
                ac_threshold=args.ac_threshold,
                qac_threshold=args.qac_threshold,
                gold_path=args.gold,
                max_rouge_chars=args.max_rouge_chars,
            )
            results.setdefault("generation", {})[name] = metrics

    if not results:
        raise ValueError("no retrieval or generation result paths were provided")

    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.output:
        validate_non_legacy_output_path(args.output, label="unified eval output")
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
