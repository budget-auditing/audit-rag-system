#!/usr/bin/env python3
"""
Run FinQA experiments with the full method matrix.

Supported methods:
    - BM25-RAG
    - Naive Dense RAG
    - HyDE-RAG
    - CoT-RAG
    - TRACE (train-free)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

FINQA_DATA = "datasets/finqa"
CORPUS_FILE = f"{FINQA_DATA}/corpus.paragraphs.jsonl"
SPLITS_DIR = f"{FINQA_DATA}/splits"
QRELS = f"{FINQA_DATA}/gold.qrels.jsonl"
GOLD = f"{FINQA_DATA}/qa_gold.jsonl"
INDEX_DIR = "runs/finqa/index"
BASE_OUTDIR = "runs/finqa"
SPLIT = "test"
TOP_K = 10
TOP_SEM_K = 50
TOP_M = 10
BEAM_WIDTH = 10
ETA = 0.3
LAMBDA_OT = 0.3
# FinQA keeps TRACE's OT / numeric structure, but avoids reusing the
# audit-specific risk-term prior as a default setting.
LAMBDA_RISK = 0.0
LAMBDA_NUM = 0.1
LAMBDA_ATTR = 0.7
DELTA_A = 0.2
DELTA_V = 0.2
WORKERS_GEN = int(os.getenv("WORKERS_GEN", "8"))
WORKERS_RET = int(os.getenv("WORKERS_RET", "8"))
WORKERS_EVI = int(os.getenv("WORKERS_EVI", "8"))

EMBED_URL = os.getenv(
    "EMBED_SERVER_URL",
    "http://localhost:8081/v1/embeddings",
)
MODEL = os.getenv(
    "CHAT_MODEL",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
)

METHOD_SPECS = {
    "bm25-rag": {
        "label": "BM25-RAG",
        "type": "baseline-rag",
        "retrieval_method": "bm25",
        "baseline_prompt_style": "rag",
        "evidence_top_n": 1,
    },
    "naive-dense-rag": {
        "label": "Naive Dense RAG",
        "type": "baseline-rag",
        "retrieval_method": "naive",
        "baseline_prompt_style": "rag",
        "evidence_top_n": 1,
    },
    "hyde-rag": {
        "label": "HyDE-RAG",
        "type": "baseline-rag",
        "retrieval_method": "hyde",
        "baseline_prompt_style": "rag",
        "evidence_top_n": 1,
    },
    "cot-rag": {
        "label": "CoT-RAG",
        "type": "cot-rag",
        "evidence_top_n": 3,
    },
    "trace": {
        "label": "TRACE",
        "type": "trace",
    },
}
DEFAULT_METHODS = list(METHOD_SPECS.keys())


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def infer_model_tag(model_name: str) -> str:
    return normalize_tag(model_name, "model")


def parse_csv(raw: str | None, valid: Sequence[str] | None = None) -> List[str]:
    values = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not values:
        return []
    if valid is not None:
        invalid = [value for value in values if value not in valid]
        if invalid:
            raise ValueError(f"Unknown options: {invalid}; available: {list(valid)}")
    return values


def py(*args: str) -> List[str]:
    return [sys.executable, "-u", *args]


def run(cmd: List[str], log_path: Path, label: str) -> None:
    os.makedirs(log_path.parent, exist_ok=True)
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"  log: {log_path}")
    print(f"  cmd: {' '.join(cmd[:10])}...")
    print(f"{'=' * 72}")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=1,
            universal_newlines=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        proc.wait()

    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (rc={proc.returncode})"
    print(f"  [{status}] {label} ({elapsed:.0f}s)")
    with open(log_path, "a", encoding="utf-8") as log_f:
        log_f.write(f"\n--- {status} elapsed={elapsed:.0f}s ---\n")
    if not ok:
        raise RuntimeError(f"{label} failed")


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown_table(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    os.makedirs(path.parent, exist_ok=True)
    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def format_metric(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def paper_method_label(method_id: str) -> str:
    if method_id == "trace":
        return "TRACE (train-free)"
    return method_label(method_id)


def write_paper_tables(summary: Dict, method_ids: Sequence[str], output_dir: Path) -> None:
    main_fieldnames = ["method", "F1", "ROUGE-L"]
    diagnostic_fieldnames = ["method", "numeric_consistency"]
    main_rows: List[Dict[str, str]] = []
    diagnostic_rows: List[Dict[str, str]] = []

    methods_payload = summary.get("methods", {})
    for method_id in method_ids:
        method_payload = methods_payload.get(method_id, {})
        generation_metrics = method_payload.get("generation", {})
        paper_label = paper_method_label(method_id)
        main_rows.append(
            {
                "method": paper_label,
                "F1": format_metric(generation_metrics.get("f1")),
                "ROUGE-L": format_metric(generation_metrics.get("rouge_l")),
            }
        )
        diagnostic_rows.append(
            {
                "method": paper_label,
                "numeric_consistency": format_metric(generation_metrics.get("numeric_consistency")),
            }
        )

    write_csv(output_dir / "main_table.csv", main_fieldnames, main_rows)
    write_markdown_table(output_dir / "main_table.md", main_fieldnames, main_rows)
    write_csv(output_dir / "diagnostic_table.csv", diagnostic_fieldnames, diagnostic_rows)
    write_markdown_table(output_dir / "diagnostic_table.md", diagnostic_fieldnames, diagnostic_rows)


def method_slug(method_id: str) -> str:
    return normalize_tag(METHOD_SPECS[method_id]["label"], method_id)


def method_label(method_id: str) -> str:
    return METHOD_SPECS[method_id]["label"]


def ensure_dirs(outdir: Path) -> None:
    for subdir in ("retrieval", "evidence", "generation", "eval", "logs"):
        (outdir / subdir).mkdir(parents=True, exist_ok=True)


def retrieval_path(outdir: Path, method_id: str) -> Path:
    return outdir / "retrieval" / f"{method_slug(method_id)}.jsonl"


def evidence_path(outdir: Path, method_id: str) -> Path:
    return outdir / "evidence" / f"{method_slug(method_id)}.chains.jsonl"


def generation_path(outdir: Path, method_id: str) -> Path:
    return outdir / "generation" / f"{method_slug(method_id)}.pred.jsonl"


def retrieval_eval_path(outdir: Path, method_id: str) -> Path:
    return outdir / "eval" / f"retrieval.{method_slug(method_id)}.json"


def generation_eval_path(outdir: Path, method_id: str) -> Path:
    return outdir / "eval" / f"generation.{method_slug(method_id)}.json"


def _with_max_queries(cmd: List[str], max_queries: int | None) -> List[str]:
    if max_queries is None:
        return cmd
    return [*cmd, "--max-queries", str(max_queries)]


def stage_convert(args: argparse.Namespace, log_dir: Path) -> None:
    corpus_path = ROOT / CORPUS_FILE
    if corpus_path.exists():
        print(f"[SKIP] {corpus_path} already exists")
        return
    run(
        py(
            "-m", "trace_rag.utils.finqa_convert",
            "--finqa-dir", args.finqa_dir,
            "--output-dir", args.data_dir,
        ),
        log_dir / "00.convert.log",
        "[0] Convert FinQA data",
    )


def stage_build_index(args: argparse.Namespace, log_dir: Path) -> None:
    index_dir = ROOT / args.index_dir
    if (index_dir / "index.faiss").exists() and (index_dir / "meta.jsonl").exists():
        print(f"[SKIP] index already exists at {index_dir}")
        return
    run(
        py(
            "build_index.py",
            "--corpus-file", args.corpus_file,
            "--output-dir", args.index_dir,
            "--no-meta-tokens",
            "--server-urls", EMBED_URL,
        ),
        log_dir / "01.build_index.log",
        "[1] Build FinQA index",
    )


def stage_retrieval(args: argparse.Namespace, outdir: Path) -> None:
    common = [
        "--splits-dir", args.splits_dir,
        "--split", args.split,
        "--index-dir", args.index_dir,
        "--top-k", str(args.top_k),
        "--server-url", EMBED_URL,
    ]
    for idx, method_id in enumerate(args.methods, start=1):
        spec = METHOD_SPECS[method_id]
        log_path = args.resolved_logdir / f"{idx + 1:02d}.retrieval.{method_slug(method_id)}.log"
        output = retrieval_path(outdir, method_id)
        if spec["type"] == "baseline-rag":
            cmd = [
                *py("-m", "experiments.run_retrieval_baselines"),
                *common,
                "--dataset-tag", "finqa",
                "--model-tag", args.resolved_model_tag,
                "--method", spec["retrieval_method"],
                "--workers", str(args.workers_ret),
                "--resume",
                "--output", str(output),
            ]
            if method_id == "hyde-rag":
                cmd.extend(["--model", args.model])
        elif spec["type"] == "cot-rag":
            cmd = [
                *py("-m", "experiments.run_cot_rag_retrieval"),
                *common,
                "--dataset-tag", "finqa",
                "--model-tag", args.resolved_model_tag,
                "--top-sem-k", str(args.top_sem_k),
                "--workers", str(args.workers_ret),
                "--resume",
                "--method", spec["label"],
                "--output", str(output),
            ]
        else:
            cmd = [
                *py("-m", "trace_rag.cli.run_retrieval"),
                *common,
                "--dataset-tag", "finqa",
                "--model-tag", args.resolved_model_tag,
                "--top-sem-k", str(args.top_sem_k),
                "--lambda-ot", str(args.lambda_ot),
                "--lambda-risk", str(args.lambda_risk),
                "--lambda-num", str(args.lambda_num),
                "--workers", str(args.workers_ret),
                "--no-pref",
                "--method", spec["label"],
                "--output", str(output),
            ]
        run(
            _with_max_queries(cmd, args.max_queries),
            log_path,
            f"[{idx + 1}] Retrieval: {spec['label']}",
        )


def stage_evidence(args: argparse.Namespace, outdir: Path) -> None:
    idx = 1
    for method_id in args.methods:
        spec = METHOD_SPECS[method_id]
        log_path = args.resolved_logdir / f"{idx + 6:02d}.evidence.{method_slug(method_id)}.log"
        if spec["type"] in {"baseline-rag", "cot-rag"}:
            cmd = [
                *py("-m", "trace_rag.utils.build_baseline_chains"),
                "--retrieval", str(retrieval_path(outdir, method_id)),
                "--index-dir", args.index_dir,
                "--method", spec["label"],
                "--top-n", str(spec["evidence_top_n"]),
                "--flat-topk", str(args.top_k),
                "--output", str(evidence_path(outdir, method_id)),
            ]
        else:
            cmd = [
                *py("-m", "trace_rag.cli.run_evidence"),
                "--splits-dir", args.splits_dir,
                "--split", args.split,
                "--index-dir", args.index_dir,
                "--server-url", EMBED_URL,
                "--dataset-tag", "finqa",
                "--model-tag", args.resolved_model_tag,
                "--top-m", str(args.top_m),
                "--top-sem-k", str(args.top_sem_k),
                "--beam-width", str(args.beam_width),
                "--eta", str(args.eta),
                "--lambda-ot", str(args.lambda_ot),
                "--lambda-risk", str(args.lambda_risk),
                "--lambda-num", str(args.lambda_num),
                "--workers", str(args.workers_evi),
                "--no-pref",
                "--flat-topk", str(args.top_k),
                "--method", spec["label"],
                "--output", str(evidence_path(outdir, method_id)),
            ]
        run(cmd, log_path, f"[{idx + 6}] Evidence: {spec['label']}")
        idx += 1


def stage_generation(args: argparse.Namespace, outdir: Path) -> None:
    idx = 1
    for method_id in args.methods:
        spec = METHOD_SPECS[method_id]
        output = generation_path(outdir, method_id)
        log_path = args.resolved_logdir / f"{idx + 11:02d}.generation.{method_slug(method_id)}.log"
        if spec["type"] == "baseline-rag":
            cmd = [
                *py("-m", "experiments.run_generation_baselines"),
                "--dataset-tag", "finqa",
                "--model-tag", args.resolved_model_tag,
                "--chains", str(evidence_path(outdir, method_id)),
                "--model", args.model,
                "--prompt-style", spec["baseline_prompt_style"],
                "--workers", str(args.workers_gen),
                "--method", spec["label"],
                "--output", str(output),
            ]
        elif spec["type"] == "cot-rag":
            cmd = [
                *py("-m", "experiments.run_cot_rag_generation"),
                "--dataset-tag", "finqa",
                "--model-tag", args.resolved_model_tag,
                "--chains", str(evidence_path(outdir, method_id)),
                "--model", args.model,
                "--workers", str(args.workers_gen),
                "--method", spec["label"],
                "--output", str(output),
            ]
        else:
            cmd = [
                *py("-m", "trace_rag.cli.run_generate"),
                "--dataset-tag", "finqa",
                "--model-tag", args.resolved_model_tag,
                "--chains", str(evidence_path(outdir, method_id)),
                "--output", str(output),
                "--model", args.model,
                "--server-url", EMBED_URL,
                "--lambda-attr", str(args.lambda_attr),
                "--delta-a", str(args.delta_a),
                "--delta-v", str(args.delta_v),
                "--workers", str(args.workers_gen),
                "--method", spec["label"],
            ]
        run(cmd, log_path, f"[{idx + 11}] Generation: {spec['label']}")
        idx += 1


def stage_eval(args: argparse.Namespace, outdir: Path) -> None:
    summary = {
        "model": args.model,
        "model_tag": args.resolved_model_tag,
        "split": args.split,
        "methods": {},
    }
    for idx, method_id in enumerate(args.methods, start=1):
        spec = METHOD_SPECS[method_id]
        summary["methods"][method_id] = {"label": spec["label"]}

        ret_output = retrieval_path(outdir, method_id)
        if ret_output.exists():
            ret_eval = retrieval_eval_path(outdir, method_id)
            run(
                [
                    *py("-m", "experiments.evaluate.finqa"),
                    "--retrieval", str(ret_output),
                    "--qrels", args.qrels,
                    "--k-list", args.k_list,
                    "--output", str(ret_eval),
                ],
                args.resolved_logdir / f"{idx + 16:02d}.eval.retrieval.{method_slug(method_id)}.log",
                f"[{idx + 16}] Eval retrieval: {spec['label']}",
            )
            summary["methods"][method_id]["retrieval"] = read_json(ret_eval).get("retrieval", {})

        gen_output = generation_path(outdir, method_id)
        if gen_output.exists():
            gen_eval = generation_eval_path(outdir, method_id)
            run(
                [
                    *py("-m", "experiments.evaluate.finqa"),
                    "--generation", str(gen_output),
                    "--gold", args.gold,
                    "--qrels", args.qrels,
                    "--output", str(gen_eval),
                ],
                args.resolved_logdir / f"{idx + 21:02d}.eval.generation.{method_slug(method_id)}.log",
                f"[{idx + 21}] Eval generation: {spec['label']}",
            )
            summary["methods"][method_id]["generation"] = read_json(gen_eval).get("generation", {})

    write_json(outdir / "eval" / "all.json", summary)
    write_paper_tables(summary, args.methods, outdir / "eval")
    print(f"Summary saved -> {outdir / 'eval' / 'all.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FinQA experiments")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["convert", "build_index", "retrieval", "evidence", "generation", "eval", "all"],
        help="Which stage to run",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods",
    )
    parser.add_argument("--split", default=SPLIT, choices=["train", "dev", "test", "test-mini", "test-pure-txt", "all"])
    parser.add_argument("--model", default=MODEL, help="Generation backbone model name")
    parser.add_argument("--model-tag", default=None, help="Optional model tag override")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: runs/finqa/<model-tag>)",
    )
    parser.add_argument(
        "--logdir",
        default=None,
        help="Log directory (default: <outdir>/logs)",
    )
    parser.add_argument("--data-dir", default=FINQA_DATA, help="Converted FinQA data directory")
    parser.add_argument("--finqa-dir", default="FinQA/dataset", help="Original FinQA dataset directory")
    parser.add_argument("--corpus-file", default=CORPUS_FILE, help="Corpus JSONL path")
    parser.add_argument("--splits-dir", default=SPLITS_DIR, help="Splits directory")
    parser.add_argument("--qrels", default=QRELS, help="Qrels JSONL path")
    parser.add_argument("--gold", default=GOLD, help="Gold answer JSONL path")
    parser.add_argument("--index-dir", default=INDEX_DIR, help="Index directory")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Final retrieval K")
    parser.add_argument("--top-sem-k", type=int, default=TOP_SEM_K, help="Semantic candidate K")
    parser.add_argument("--top-m", type=int, default=TOP_M, help="Evidence top-M")
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH, help="Evidence beam width")
    parser.add_argument("--eta", type=float, default=ETA, help="Evidence link weight")
    parser.add_argument("--lambda-ot", type=float, default=LAMBDA_OT, help="TRACE lambda_ot")
    parser.add_argument("--lambda-risk", type=float, default=LAMBDA_RISK, help="TRACE lambda_risk")
    parser.add_argument("--lambda-num", type=float, default=LAMBDA_NUM, help="TRACE lambda_num")
    parser.add_argument("--lambda-attr", type=float, default=LAMBDA_ATTR, help="TRACE lambda_attr")
    parser.add_argument("--delta-a", type=float, default=DELTA_A, help="AC threshold")
    parser.add_argument("--delta-v", type=float, default=DELTA_V, help="Numeric consistency threshold")
    parser.add_argument("--workers-gen", type=int, default=WORKERS_GEN, help="Generation workers")
    parser.add_argument("--workers-ret", type=int, default=WORKERS_RET, help="Retrieval workers")
    parser.add_argument("--workers-evi", type=int, default=WORKERS_EVI, help="Evidence workers")
    parser.add_argument("--k-list", default="1,3,5,10", help="Retrieval K list for evaluation")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query cap for smoke tests")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.methods = parse_csv(args.methods, METHOD_SPECS.keys())

    model_tag = args.model_tag or infer_model_tag(args.model)
    args.resolved_model_tag = model_tag
    outdir = Path(args.outdir or os.path.join(BASE_OUTDIR, model_tag))
    logdir = Path(args.logdir or outdir / "logs")
    args.resolved_logdir = logdir

    print(f"[Config] model={args.model}")
    print(f"[Config] model_tag={model_tag}")
    print(f"[Config] outdir={outdir}")
    print(f"[Config] logdir={logdir}")
    print(f"[Config] methods={args.methods}")

    ensure_dirs(outdir)
    logdir.mkdir(parents=True, exist_ok=True)

    stages = {
        "convert": lambda: stage_convert(args, logdir),
        "build_index": lambda: stage_build_index(args, logdir),
        "retrieval": lambda: stage_retrieval(args, outdir),
        "evidence": lambda: stage_evidence(args, outdir),
        "generation": lambda: stage_generation(args, outdir),
        "eval": lambda: stage_eval(args, outdir),
    }

    if args.stage == "all":
        for stage_name in ("convert", "build_index", "retrieval", "evidence", "generation", "eval"):
            stages[stage_name]()
    else:
        stages[args.stage]()

    print("\nDone.")


if __name__ == "__main__":
    main()
