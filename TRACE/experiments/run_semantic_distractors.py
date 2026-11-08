#! /usr/bin/env python3
"""
Run semantic distractor robustness experiments.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from experiments.evaluate.generation import evaluate_generation_metrics
from trace_rag.utils.phase3 import (
    chain_evidence_list,
    load_gold_map,
    load_meta,
    normalize_tag,
    read_jsonl,
    stage3_root,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _parse_int_list(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _infer_method(rows: List[Dict], fallback: str) -> str:
    for row in rows:
        method = row.get("method")
        if method:
            return str(method)
    return fallback


def _infer_candidates(item: Dict) -> List[str]:
    coarse = item.get("coarse_topk_evidence_ids") or []
    if coarse:
        return [str(eid) for eid in coarse if eid]
    topk = item.get("topk_evidence_ids") or []
    return [str(eid) for eid in topk if eid]


def _generator_cmd(
    generator: str,
    chains_path: str,
    pred_path: str,
    method: str,
    args: argparse.Namespace,
) -> List[str]:
    if generator == "trace":
        return [
            sys.executable,
            "-m",
            "trace_rag.cli.run_generate",
            "--chains",
            chains_path,
            "--output",
            pred_path,
            "--model",
            args.model,
            "--method",
            method,
            "--workers",
            str(args.workers),
            "--lambda-attr",
            str(args.lambda_attr),
            "--delta-a",
            str(args.delta_a),
            "--delta-v",
            str(args.delta_v),
            "--max-retries",
            str(args.max_retries),
            "--server-url",
            args.server_url,
        ]
    if generator == "cot-rag":
        return [
            sys.executable,
            "-m",
            "experiments.run_cot_rag_generation",
            "--chains",
            chains_path,
            "--output",
            pred_path,
            "--model",
            args.model,
            "--method",
            method,
            "--workers",
            str(args.workers),
            "--max-retries",
            str(args.max_retries),
        ]
    return [
        sys.executable,
        "-m",
        "experiments.run_generation_baselines",
        "--chains",
        chains_path,
        "--output",
        pred_path,
        "--model",
        args.model,
        "--method",
        method,
        "--workers",
        str(args.workers),
        "--max-retries",
        str(args.max_retries),
        "--prompt-style",
        "rag",
    ]


def _run(cmd: List[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic distractor robustness experiments")
    parser.add_argument("--dataset-tag", default="audit-ociqa")
    parser.add_argument("--model-tag", default="qwen")
    parser.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--chains", required=True, help="Base chains JSONL path")
    parser.add_argument(
        "--retrieval",
        default="runs/audit-ociqa/qwen/retrieval/trace.jsonl",
        help="Retrieval JSONL used to source distractor candidates",
    )
    parser.add_argument("--index-dir", default="runs/index")
    parser.add_argument("--gold", default="datasets-public/audit-ociqa-demo/qa.gold.jsonl")
    parser.add_argument("--noise-levels", default="0,5,10,15")
    parser.add_argument("--generator", choices=["trace", "baseline-rag", "cot-rag"], default="trace")
    parser.add_argument("--method", default=None, help="Display method label override")
    parser.add_argument("--output-dir", default=None, help="Stage-3 output root for this method")
    parser.add_argument(
        "--server-url",
        default=os.getenv(
            "EMBED_SERVER_URL",
            "http://localhost:8081/v1/embeddings",
        ),
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--lambda-attr", type=float, default=0.7)
    parser.add_argument("--delta-a", type=float, default=0.2)
    parser.add_argument("--delta-v", type=float, default=0.2)
    parser.add_argument("--ac-threshold", type=float, default=0.2)
    parser.add_argument("--qac-threshold", type=float, default=0.2)
    parser.add_argument("--run-generate", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise_levels = _parse_int_list(args.noise_levels)
    base_rows_map = {}
    for item in read_jsonl(args.chains):
        row_id = item.get("row_id")
        if row_id is None:
            continue
        base_rows_map[str(row_id)] = item
    base_rows = list(base_rows_map.values())
    base_method = args.method or _infer_method(base_rows, "TRACE")
    method_tag = normalize_tag(base_method, "method")
    output_dir = args.output_dir or os.path.join(
        stage3_root(args.dataset_tag, args.model_tag),
        "semantic_distractors",
        method_tag,
    )
    os.makedirs(output_dir, exist_ok=True)

    gold_map = load_gold_map(args.gold)
    meta = load_meta(args.index_dir)
    retrieval_rows = {
        str(item.get("row_id")): _infer_candidates(item)
        for item in read_jsonl(args.retrieval)
        if item.get("row_id")
    }

    chain_paths: Dict[int, str] = {}
    coverage_rows = []
    for noise in noise_levels:
        noisy_rows = []
        missing_rows = 0
        for row in base_rows:
            row_id = str(row.get("row_id"))
            base_eids = {
                evidence.get("evidence_id")
                for evidence in chain_evidence_list(row)
                if evidence.get("evidence_id")
            }
            gold_eids = set((gold_map.get(row_id) or {}).get("gold_evidence_ids") or [])
            distractors = []
            for evidence_id in retrieval_rows.get(row_id, []):
                if evidence_id in base_eids or evidence_id in gold_eids:
                    continue
                if evidence_id not in meta:
                    continue
                distractors.append(dict(meta[evidence_id]))
                if len(distractors) >= noise:
                    break
            if len(distractors) < noise:
                missing_rows += 1
            noisy_row = dict(row)
            noisy_row["extra_evidence"] = distractors
            noisy_row["noise_level"] = noise
            noisy_row["distractor_evidence_ids"] = [item["evidence_id"] for item in distractors]
            noisy_rows.append(noisy_row)
        chain_path = os.path.join(output_dir, f"noise-{noise:02d}.chains.jsonl")
        write_jsonl(chain_path, noisy_rows)
        chain_paths[noise] = chain_path
        coverage_rows.append(
            {
                "noise_level": noise,
                "rows": len(noisy_rows),
                "rows_missing_full_distractors": missing_rows,
            }
        )

    write_json(os.path.join(output_dir, "chain_coverage.json"), {"rows": coverage_rows})

    summary_rows = []
    for noise in noise_levels:
        pred_path = os.path.join(output_dir, f"noise-{noise:02d}.pred.jsonl")
        eval_path = os.path.join(output_dir, f"noise-{noise:02d}.eval.json")
        method_label = f"{base_method}-noise-{noise:02d}"
        if args.run_generate:
            _run(_generator_cmd(args.generator, chain_paths[noise], pred_path, method_label, args))
        if args.evaluate:
            metrics = evaluate_generation_metrics(
                pred_path=pred_path,
                index_dir=args.index_dir,
                server_url=args.server_url,
                lambda_attr=args.lambda_attr,
                ac_threshold=args.ac_threshold,
                qac_threshold=args.qac_threshold,
                gold_path=args.gold,
            )
            metrics["noise_level"] = noise
            write_json(eval_path, metrics)
            summary_rows.append(
                {
                    "noise_level": noise,
                    "AC": metrics.get("ac", 0.0),
                    "QAC": metrics.get("qac", 0.0),
                    "OCR": metrics.get("ocr", 0.0),
                    "Char-F1": metrics.get("char_f1", 0.0),
                    "ROUGE-L": metrics.get("rouge_l", 0.0),
                }
            )

    if summary_rows:
        summary_json = os.path.join(output_dir, "summary.json")
        summary_csv = os.path.join(output_dir, "summary.csv")
        write_json(summary_json, {"method": base_method, "rows": summary_rows})
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["noise_level", "AC", "QAC", "OCR", "Char-F1", "ROUGE-L"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
