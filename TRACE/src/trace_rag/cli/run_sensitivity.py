#! /usr/bin/env python3
"""
Run TRACE sensitivity sweeps.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Set

ROOT_DIR = Path(__file__).resolve().parents[3]


def _parse_list(raw: str, cast_fn):
    if raw is None:
        return []
    items = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast_fn(part))
    return items


def _format_float(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


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
            score += 1.0 / (math.log2(idx + 1))
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
    if not gold:
        return 0.0
    return 1.0 if any(doc_id in gold for doc_id in pred[:k]) else 0.0


def _recall_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    found = sum(1 for doc_id in pred[:k] if doc_id in gold)
    return found / len(gold)


def _rr_at_k(pred: List[str], gold: Set[str], k: int) -> float:
    for idx, doc_id in enumerate(pred[:k]):
        if doc_id in gold:
            return 1.0 / (idx + 1)
    return 0.0


def _evaluate_retrieval(result_path: str, qrels: Dict[str, Set[str]], k_list: List[int]) -> Dict:
    allowed_ids = set(qrels.keys())
    results = _load_results(result_path, allowed_ids)
    metrics = {}
    for k in k_list:
        hits = []
        recalls = []
        ndcgs = []
        rrs = []
        for row_id, gold in qrels.items():
            pred = results.get(row_id, [])
            hits.append(_hit_at_k(pred, gold, k))
            recalls.append(_recall_at_k(pred, gold, k))
            ndcgs.append(_ndcg_at_k(pred, gold, k))
            rrs.append(_rr_at_k(pred, gold, k))
        metrics[f"Hit@{k}"] = sum(hits) / len(hits) if hits else 0.0
        metrics[f"Recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        metrics[f"nDCG@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        metrics[f"RR@{k}"] = sum(rrs) / len(rrs) if rrs else 0.0
    return metrics


def _ensure_pythonpath(env: Dict[str, str], root: Path) -> Dict[str, str]:
    """Ensure the installed trace_rag package resolves correctly.

    If the repository is consumed via ``pip install -e .``, no extra tweaks are
    required. Otherwise we prepend ``<root>/src`` to ``PYTHONPATH`` so that
    subprocesses can locate the package without installation.
    """
    src_path = str(root / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_path}:{current}" if current else src_path
    return env


def _run(cmd: List[str], env: Dict[str, str], dry_run: bool) -> bool:
    print("[RUN]", " ".join(cmd))
    if dry_run:
        return True
    return subprocess.run(cmd, env=env).returncode == 0


def _split_server_urls(server_url: str) -> List[str]:
    return [part.strip() for part in (server_url or "").split(",") if part.strip()]


def _default_worker_count(server_url: str, *, per_endpoint: int, floor: int, cap: int) -> int:
    endpoint_count = max(1, len(_split_server_urls(server_url)))
    cpu_count = max(1, os.cpu_count() or 1)
    return max(floor, min(cap, endpoint_count * per_endpoint, cpu_count))


def _load_summary_run_ids(path: Path) -> Set[str]:
    run_ids: Set[str] = set()
    if not path.exists():
        return run_ids
    for item in _read_jsonl(str(path)):
        run_id = item.get("run_id")
        if run_id:
            run_ids.add(run_id)
    return run_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE sensitivity sweeps")
    parser.add_argument("--index-dir", default="runs/index", help="FAISS index directory")
    parser.add_argument("--splits-dir", default="datasets-public/audit-ociqa-demo/splits", help="Splits directory")
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to run (e.g. train, dev, test, test-mini, all)",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/audit-ociqa/shared/sensitivity",
        help="Output directory for sweep results",
    )
    parser.add_argument(
        "--shared-cache-dir",
        default=None,
        help="Optional shared retrieval/evidence/generation cache root across sweeps",
    )
    parser.add_argument("--lambda-ot", default="0.3", help="Comma list for lambda_ot")
    parser.add_argument("--risk-top-tokens", default="32", help="Comma list for m")
    parser.add_argument("--k-values", default="20", help="Comma list for K")
    parser.add_argument("--lambda-risk", type=float, default=0.1, help="Retrieval lambda_risk")
    parser.add_argument("--lambda-num", type=float, default=0.1, help="Numeric score weight")
    parser.add_argument("--top-sem-k", type=int, default=None, help="Override top_sem_k")
    parser.add_argument(
        "--top-sem-k-multiplier",
        type=int,
        default=5,
        help="top_sem_k multiplier for retrieval",
    )
    parser.add_argument(
        "--evidence-top-sem-k-multiplier",
        type=int,
        default=3,
        help="top_sem_k multiplier for evidence",
    )
    parser.add_argument("--pref-weights", default=None, help="Preference weights JSON path")
    parser.add_argument("--no-pref", action="store_true", help="Ignore preference weights")
    parser.add_argument("--run-generate", action="store_true", help="Run generation step")
    parser.add_argument("--eval-retrieval", action="store_true", help="Evaluate retrieval outputs")
    parser.add_argument("--eval-generation", action="store_true", help="Evaluate generation outputs")
    parser.add_argument(
        "--k-list",
        default="5,10,20",
        help="Recall/nDCG K list for retrieval eval",
    )
    parser.add_argument(
        "--qrels",
        default="datasets-public/audit-ociqa-demo/gold.qrels.jsonl",
        help="Qrels JSONL path",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct", help="LLM model name")
    parser.add_argument(
        "--server-url",
        default=os.getenv(
            "EMBED_SERVER_URL",
            "http://localhost:8081/v1/embeddings",
        ),
    )
    parser.add_argument("--lambda-attr", type=float, default=0.7, help="Attribute consistency weight")
    parser.add_argument(
        "--lambda-attr-values",
        default=None,
        help="Comma list for lambda_attr (defaults to --lambda-attr)",
    )
    parser.add_argument("--delta-a", type=float, default=0.2, help="Alignment threshold")
    parser.add_argument("--ac-threshold", type=float, default=0.2, help="AC threshold")
    parser.add_argument("--qac-threshold", type=float, default=0.2, help="QAC threshold")
    parser.add_argument("--beam-widths", default="10", help="Comma list for beam width B")
    parser.add_argument("--lambda-tri", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ncc-threshold", type=float, default=None, help=argparse.SUPPRESS)
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
    parser.add_argument("--max-retries", type=int, default=3, help="Chat API max retries")
    parser.add_argument("--method-prefix", default="TRACE", help="Method name prefix")
    parser.add_argument(
        "--exclude-doc-prefix",
        default=None,
        help="Exclude evidence IDs starting with this prefix (passed to retrieval runner)",
    )
    parser.add_argument("--resume", action="store_true", help="Skip existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--retrieval-workers", type=int, default=None, help="Workers for retrieval stage")
    parser.add_argument("--evidence-workers", type=int, default=None, help="Workers for evidence stage")
    parser.add_argument("--evidence-chunk-size", type=int, default=0, help="Chunk size for evidence stage (0 = auto)")
    parser.add_argument("--generation-workers", type=int, default=None, help="Workers for generation stage")
    parser.add_argument("--max-parallel", type=int, default=None, help="Max parallel sweep configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = _ensure_pythonpath(os.environ.copy(), ROOT_DIR)
    endpoint_count = max(1, len(_split_server_urls(args.server_url)))
    if args.retrieval_workers is None:
        args.retrieval_workers = _default_worker_count(
            args.server_url, per_endpoint=4, floor=8, cap=24
        )
    if args.evidence_workers is None:
        args.evidence_workers = _default_worker_count(
            args.server_url, per_endpoint=4, floor=8, cap=24
        )
    if args.generation_workers is None:
        args.generation_workers = _default_worker_count(
            args.server_url, per_endpoint=6, floor=12, cap=32
        )
    if args.max_parallel is None:
        args.max_parallel = max(1, min(2, endpoint_count))
    print(
        "[parallelism] "
        f"endpoints={endpoint_count} max_parallel={args.max_parallel} "
        f"retrieval_workers={args.retrieval_workers} "
        f"evidence_workers={args.evidence_workers} "
        f"generation_workers={args.generation_workers} "
        f"evidence_chunk_size={args.evidence_chunk_size or 'auto'}"
    )
    lambda_attr = args.lambda_tri if args.lambda_tri is not None else args.lambda_attr
    delta_a = args.threshold if args.threshold is not None else args.delta_a
    qac_threshold = args.ncc_threshold if args.ncc_threshold is not None else args.qac_threshold

    lambda_ots = _parse_list(args.lambda_ot, float)
    risk_top_tokens = _parse_list(args.risk_top_tokens, int)
    k_values = _parse_list(args.k_values, int)
    k_list = _parse_list(args.k_list, int)
    lambda_attrs = _parse_list(args.lambda_attr_values, float) if args.lambda_attr_values else [lambda_attr]
    beam_widths = _parse_list(args.beam_widths, int)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_root = Path(args.shared_cache_dir) if args.shared_cache_dir else output_dir
    retrieval_dir = stage_root / "retrieval"
    evidence_dir = stage_root / "evidence"
    generation_dir = stage_root / "generation"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    generation_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.jsonl"
    if summary_path.exists() and not args.resume:
        summary_path.unlink()
    completed_run_ids = _load_summary_run_ids(summary_path) if args.resume else set()

    qrels = None
    if args.eval_retrieval:
        split_ids = _load_split_ids(args.splits_dir, args.split)
        qrels = _load_qrels(args.qrels, split_ids)

    def _run_one_config(config_tuple):
        lambda_ot, m, k, attr, beam_width = config_tuple
        run_id = f"ot{_format_float(lambda_ot)}_m{m}_k{k}_attr{_format_float(attr)}_b{beam_width}"
        if args.resume and run_id in completed_run_ids:
            return None

        top_sem_k = args.top_sem_k or max(k * args.top_sem_k_multiplier, k)
        evidence_top_sem_k = args.top_sem_k or max(k * args.evidence_top_sem_k_multiplier, k)
        pref_tag = "nopref" if args.no_pref else ("pref" if args.pref_weights else "plain")
        retrieval_id = (
            f"s{args.split}_ot{_format_float(lambda_ot)}_m{m}_k{k}_sem{top_sem_k}_"
            f"lr{_format_float(args.lambda_risk)}_ln{_format_float(args.lambda_num)}_{pref_tag}"
        )
        evidence_id = (
            f"{retrieval_id}_esem{evidence_top_sem_k}_b{beam_width}"
        )
        generation_id = (
            f"{evidence_id}_attr{_format_float(attr)}_da{_format_float(delta_a)}_"
            f"qac{_format_float(qac_threshold)}"
        )
        retrieval_method = f"{args.method_prefix}-retr-{retrieval_id}"
        evidence_method = f"{args.method_prefix}-evi-{evidence_id}"
        generation_method = f"{args.method_prefix}-{run_id}"
        retrieval_path = retrieval_dir / f"{retrieval_id}.jsonl"
        evidence_path = evidence_dir / f"{evidence_id}.jsonl"
        generation_path = generation_dir / f"{generation_id}.jsonl"

        status = {"retrieval": "skipped", "evidence": "skipped", "generation": "skipped"}
        metrics = {"retrieval": None, "generation": None}

        retrieval_cmd = [
            sys.executable,
            "-m",
            "trace_rag.cli.run_retrieval",
            "--index-dir",
            args.index_dir,
            "--splits-dir",
            args.splits_dir,
            "--split",
            args.split,
            "--top-k",
            str(k),
            "--top-sem-k",
            str(top_sem_k),
            "--lambda-ot",
            str(lambda_ot),
            "--lambda-risk",
            str(args.lambda_risk),
            "--lambda-num",
            str(args.lambda_num),
            "--risk-top-tokens",
            str(m),
            "--method",
            retrieval_method,
            "--workers",
            str(args.retrieval_workers),
            "--output",
            str(retrieval_path),
        ]
        if args.pref_weights and not args.no_pref:
            retrieval_cmd.extend(["--pref-weights", args.pref_weights])
        if args.no_pref:
            retrieval_cmd.append("--no-pref")
        if getattr(args, "exclude_doc_prefix", None):
            retrieval_cmd.extend(["--exclude-doc-prefix", args.exclude_doc_prefix])

        if not (args.resume and retrieval_path.exists()):
            status["retrieval"] = "ran" if _run(retrieval_cmd, env, args.dry_run) else "failed"

        evidence_cmd = [
            sys.executable,
            "-m",
            "trace_rag.cli.run_evidence",
            "--index-dir",
            args.index_dir,
            "--splits-dir",
            args.splits_dir,
            "--split",
            args.split,
            "--top-m",
            str(k),
            "--top-sem-k",
            str(evidence_top_sem_k),
            "--beam-width",
            str(beam_width),
            "--lambda-ot",
            str(lambda_ot),
            "--lambda-risk",
            str(args.lambda_risk),
            "--lambda-num",
            str(args.lambda_num),
            "--risk-top-tokens",
            str(m),
            "--method",
            evidence_method,
            "--workers",
            str(args.evidence_workers),
            "--chunk-size",
            str(args.evidence_chunk_size),
            "--output",
            str(evidence_path),
        ]
        if args.pref_weights and not args.no_pref:
            evidence_cmd.extend(["--pref-weights", args.pref_weights])
        if args.no_pref:
            evidence_cmd.append("--no-pref")

        if not (args.resume and evidence_path.exists()):
            status["evidence"] = "ran" if _run(evidence_cmd, env, args.dry_run) else "failed"

        if args.run_generate:
            generate_cmd = [
                sys.executable,
                "-m",
                "trace_rag.cli.run_generate",
                "--chains",
                str(evidence_path),
                "--output",
                str(generation_path),
                "--model",
                args.model,
                "--server-url",
                args.server_url,
                "--lambda-attr",
                str(attr),
                "--delta-a",
                str(delta_a),
                "--max-retries",
                str(args.max_retries),
                "--method",
                generation_method,
                "--workers",
                str(args.generation_workers),
            ]
            if not (args.resume and generation_path.exists()):
                status["generation"] = (
                    "ran" if _run(generate_cmd, env, args.dry_run) else "failed"
                )

        if args.eval_retrieval and qrels and retrieval_path.exists():
            metrics["retrieval"] = _evaluate_retrieval(
                str(retrieval_path), qrels, k_list
            )

        if args.eval_generation and generation_path.exists():
            from experiments.evaluate.generation import evaluate_generation_metrics

            metrics["generation"] = evaluate_generation_metrics(
                pred_path=str(generation_path),
                index_dir=args.index_dir,
                server_url=args.server_url,
                lambda_attr=attr,
                ac_threshold=args.ac_threshold,
                qac_threshold=qac_threshold,
                gold_path=args.gold,
                max_rouge_chars=args.max_rouge_chars,
            )

        return {
            "run_id": run_id,
            "params": {
                "lambda_ot": lambda_ot,
                "risk_top_tokens": m,
                "k": k,
                "top_sem_k": top_sem_k,
                "evidence_top_sem_k": evidence_top_sem_k,
                "lambda_risk": args.lambda_risk,
                "lambda_num": args.lambda_num,
                "lambda_attr": attr,
                "beam_width": beam_width,
                "delta_a": delta_a,
                "qac_threshold": qac_threshold,
                "method": generation_method,
                "gold": args.gold,
                "max_rouge_chars": args.max_rouge_chars,
            },
            "paths": {
                "retrieval": str(retrieval_path),
                "evidence": str(evidence_path),
                "generation": str(generation_path),
            },
            "status": status,
            "metrics": metrics,
        }

    configs = list(itertools.product(lambda_ots, risk_top_tokens, k_values, lambda_attrs, beam_widths))
    max_parallel = max(1, args.max_parallel)
    write_lock = threading.Lock()
    summary_file = open(summary_path, "a", encoding="utf-8")
    try:
        if max_parallel <= 1:
            for config in configs:
                record = _run_one_config(config)
                if record is None:
                    continue
                summary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                summary_file.flush()
        else:
            with ThreadPoolExecutor(max_workers=max_parallel) as pool:
                futures = {pool.submit(_run_one_config, c): c for c in configs}
                for future in as_completed(futures):
                    record = future.result()
                    if record is None:
                        continue
                    with write_lock:
                        summary_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        summary_file.flush()
    finally:
        summary_file.close()


if __name__ == "__main__":
    main()
