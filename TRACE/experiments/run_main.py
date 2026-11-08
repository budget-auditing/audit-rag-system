#!/usr/bin/env python3
"""
Master runner for the main Audit-OCIQA experiments.

It supports:
1. method list + backbone list
2. split/qrels/gold consistency checks
3. retrieval / evidence / generation / evaluation orchestration
4. repeated generation runs with mean/std aggregation
"""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

from trace_rag.utils.experiment_layout import DEFAULT_AUDIT_SPLITS_DIR, validate_non_legacy_output_path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

DATASET_TAG = "audit-ociqa"
SPLITS_DIR = DEFAULT_AUDIT_SPLITS_DIR
INDEX_DIR = "runs/index"
QRELS = "datasets-public/audit-ociqa-demo/gold.qrels.jsonl"
GOLD = "datasets-public/audit-ociqa-demo/qa.gold.jsonl"
PREF_WEIGHTS = "runs/audit-ociqa/pref/trace.pref.json"
BASE_OUTDIR = "runs/audit-ociqa"
SPLIT = "test"
TOP_K = 20
TOP_SEM_K = 50
TOP_M = 10
BEAM_WIDTH = 10
ETA = 0.3
LAMBDA_OT = 0.3
LAMBDA_RISK = 0.1
LAMBDA_NUM = 0.1
LAMBDA_ATTR = 0.7
DELTA_A = 0.2
DELTA_V = 0.2
EXCLUDE_DOC_PREFIX = ""
NUM_REPEATS = 3
WORKERS_GEN = 32
WORKERS_RET = 32
WORKERS_EVI = 16
EMBED_URL = os.getenv(
    "EMBED_SERVER_URL",
    "http://localhost:8081/v1/embeddings",
)
RERANKER_URL = None
COARSE_TOP_K = 100

BACKBONE_REGISTRY = {
    "qwen": os.getenv("AUDIT_OCIQA_MODEL_QWEN", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
    "deepseek": os.getenv("AUDIT_OCIQA_MODEL_DEEPSEEK", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"),
    "llama": os.getenv("AUDIT_OCIQA_MODEL_LLAMA", "unsloth/Llama-3.3-70B-Instruct"),
}

METHOD_SPECS = {
    "closed-book": {"label": "Closed-book", "type": "closed-book"},
    "bm25-rag": {"label": "BM25-RAG", "type": "baseline-rag", "retrieval_method": "bm25"},
    "naive-dense-rag": {"label": "Naive Dense RAG", "type": "baseline-rag", "retrieval_method": "naive"},
    "hyde-rag": {"label": "HyDE-RAG", "type": "baseline-rag", "retrieval_method": "hyde"},
    "cot-rag": {"label": "CoT-RAG", "type": "cot-rag"},
    "trace": {"label": "TRACE", "type": "trace"},
}

DEFAULT_METHODS = list(METHOD_SPECS.keys())
DEFAULT_BACKBONES = ["deepseek"]


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_csv(raw: str | None, valid: Sequence[str] | None = None) -> List[str]:
    values = [part.strip() for part in (raw or "").split(",") if part.strip()]
    if not values:
        return []
    if valid is not None:
        invalid = [value for value in values if value not in valid]
        if invalid:
            raise ValueError(f"unknown option: {invalid}; valid values: {list(valid)}")
    return values


def method_slug(method_id: str) -> str:
    spec = METHOD_SPECS[method_id]
    return normalize_tag(spec["label"], method_id)


def method_label(method_id: str) -> str:
    return METHOD_SPECS[method_id]["label"]


def output_dir_for_backbone(backbone: str) -> Path:
    return ROOT / BASE_OUTDIR / backbone


def retrieval_path(outdir: Path, method_id: str) -> Path:
    return outdir / "retrieval" / f"{method_slug(method_id)}.jsonl"


def evidence_path(outdir: Path, method_id: str) -> Path:
    return outdir / "evidence" / f"{method_slug(method_id)}.chains.jsonl"


def generation_path(outdir: Path, repeat_idx: int, method_id: str) -> Path:
    return outdir / "generation" / f"repeat-{repeat_idx:02d}" / f"{method_slug(method_id)}.pred.jsonl"


def generation_eval_path(outdir: Path, repeat_idx: int, method_id: str) -> Path:
    return outdir / "eval" / "generation" / f"repeat-{repeat_idx:02d}" / f"{method_slug(method_id)}.json"


def py(*args: str) -> List[str]:
    return [sys.executable, "-u", *args]


def run(cmd: List[str], log_path: Path, label: str) -> bool:
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
    return ok


def ensure_dirs(outdir: Path) -> None:
    for subdir in ("retrieval", "evidence", "generation", "eval", "logs"):
        (outdir / subdir).mkdir(parents=True, exist_ok=True)


def load_gold_rows(gold_path: str) -> List[Dict]:
    return read_jsonl(str(ROOT / gold_path))


def consistency_report(
    splits_dir: str,
    qrels_path: str,
    gold_path: str,
    split: str,
) -> Dict:
    split_dir_path = ROOT / splits_dir
    split_names = ["train", "dev", "test"]
    # Allow custom split names (e.g. test-mini): make sure the file is loaded too.
    if split != "all" and split not in split_names:
        split_names = split_names + [split]
    qrels_check_splits = split_names if split == "all" else [split]
    split_rows: Dict[str, List[Dict]] = {}
    for split_name in split_names:
        split_file = split_dir_path / f"{split_name}.jsonl"
        if split_file.exists():
            split_rows[split_name] = read_jsonl(str(split_file))
        else:
            split_rows[split_name] = []
    split_ids = {
        split_name: {str(item.get("row_id")) for item in rows if item.get("row_id")}
        for split_name, rows in split_rows.items()
    }
    overlaps = {}
    for left in split_names:
        for right in split_names:
            if left >= right:
                continue
            overlaps[f"{left}_{right}"] = len(split_ids[left] & split_ids[right])

    gold_rows = load_gold_rows(gold_path)
    gold_by_row = {str(item["row_id"]): item for item in gold_rows if item.get("row_id")}
    qrels_rows = read_jsonl(str(ROOT / qrels_path))
    qrels_ids = {str(item.get("row_id")) for item in qrels_rows if item.get("row_id")}

    split_union = set().union(*split_ids.values()) if split_ids else set()
    qrels_expected = set().union(*(split_ids[name] for name in qrels_check_splits)) if split_ids else set()
    missing_in_gold = sorted(split_union - set(gold_by_row.keys()))
    missing_in_qrels = sorted(qrels_expected - qrels_ids)
    missing_in_qrels_all = sorted(split_union - qrels_ids)
    qrels_not_in_gold = sorted(qrels_ids - set(gold_by_row.keys()))

    doc_splits: Dict[str, set[str]] = {name: set() for name in split_names}
    for split_name, row_ids in split_ids.items():
        for row_id in row_ids:
            gold_item = gold_by_row.get(row_id) or {}
            doc_id = gold_item.get("doc_id")
            if doc_id:
                doc_splits[split_name].add(str(doc_id))
    doc_overlaps = {}
    for left in split_names:
        for right in split_names:
            if left >= right:
                continue
            doc_overlaps[f"{left}_{right}"] = len(doc_splits[left] & doc_splits[right])

    split_meta_path = split_dir_path / "meta.jsonl"
    split_meta = read_jsonl(str(split_meta_path))[0] if split_meta_path.exists() else {}
    source_info = split_meta.get("source", {}) if isinstance(split_meta, dict) else {}
    consistency_issues = []
    consistency_warnings = []
    if missing_in_gold:
        consistency_issues.append(f"split rows missing in gold: {len(missing_in_gold)}")
    if missing_in_qrels:
        consistency_issues.append(f"active split rows missing in qrels: {len(missing_in_qrels)}")
    if missing_in_qrels_all and split != "all":
        consistency_warnings.append(f"all split rows missing in qrels: {len(missing_in_qrels_all)}")
    if qrels_not_in_gold:
        consistency_issues.append(f"qrels rows not in gold: {len(qrels_not_in_gold)}")
    # Custom splits derived from a standard split (e.g. test-mini from test)
    # are expected to overlap; skip those pairs.
    _standard = {"train", "dev", "test"}
    def _is_subset_pair(pair: str) -> bool:
        left, right = pair.split("_", 1)
        return (left in _standard and right.startswith(left + "-")) or \
               (right in _standard and left.startswith(right + "-"))

    for pair_name, overlap_count in overlaps.items():
        if overlap_count and not _is_subset_pair(pair_name):
            consistency_issues.append(f"row overlap {pair_name}: {overlap_count}")
    for pair_name, overlap_count in doc_overlaps.items():
        if overlap_count and not _is_subset_pair(pair_name):
            consistency_issues.append(f"doc overlap {pair_name}: {overlap_count}")

    return {
        "active_split": split,
        "split_counts": {name: len(rows) for name, rows in split_rows.items()},
        "split_doc_counts": {name: len(doc_splits[name]) for name in split_names},
        "split_overlaps": overlaps,
        "doc_overlaps": doc_overlaps,
        "gold_total": len(gold_by_row),
        "qrels_total": len(qrels_ids),
        "missing_split_rows_in_gold": missing_in_gold[:100],
        "missing_active_split_rows_in_qrels": missing_in_qrels[:100],
        "missing_all_split_rows_in_qrels": missing_in_qrels_all[:100],
        "qrels_rows_not_in_gold": qrels_not_in_gold[:100],
        "source": {
            "splits_dir": splits_dir,
            "qrels": qrels_path,
            "gold": gold_path,
            "qrels_check_splits": qrels_check_splits,
            "split_meta": source_info,
        },
        "issues": consistency_issues,
        "warnings": consistency_warnings,
        "ok": not consistency_issues,
    }


def write_json(path: Path, payload: Dict) -> None:
    validate_non_legacy_output_path(str(path), label="json output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    validate_non_legacy_output_path(str(path), label="summary csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_markdown_table(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    validate_non_legacy_output_path(str(path), label="summary markdown")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    lines = [header, divider]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def format_float(value: float | int | None) -> str:
    return f"{float(value or 0.0):.4f}"


def format_mean_std(metric_payload: Dict[str, float]) -> str:
    return f"{metric_payload.get('mean', 0.0):.4f}±{metric_payload.get('std', 0.0):.4f}"


def exact_sign_flip_pvalue(deltas: List[float]) -> float | None:
    if not deltas:
        return None
    if len(deltas) > 12:
        return None
    observed = abs(sum(deltas) / len(deltas))
    total = 0
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        total += 1
        permuted = abs(sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas))
        if permuted >= observed - 1e-12:
            extreme += 1
    return extreme / total if total else None


def collect_method_run_metrics(
    method_labels: List[str],
    generation_results: Dict[str, Dict[str, Dict]],
    metric_names: List[str],
) -> Dict[str, Dict[str, List[float]]]:
    method_runs = {
        label: {metric_name: [] for metric_name in metric_names}
        for label in method_labels
    }
    for repeat_key in sorted(generation_results.keys()):
        repeat_payload = generation_results[repeat_key]
        for label in method_labels:
            metrics = repeat_payload.get(label)
            if not metrics:
                continue
            for metric_name in metric_names:
                if metric_name in metrics:
                    method_runs[label][metric_name].append(float(metrics.get(metric_name, 0.0)))
    return method_runs


def build_statistical_summary(
    method_labels: List[str],
    generation_results: Dict[str, Dict[str, Dict]],
    metric_names: List[str],
) -> Dict:
    method_runs = collect_method_run_metrics(method_labels, generation_results, metric_names)
    trace_label = "TRACE"
    if trace_label not in method_runs:
        return {"test": "exact_sign_flip", "notes": ["TRACE results unavailable"], "trace_vs_baselines": {}}
    comparisons = {}
    notes = []
    trace_runs = method_runs[trace_label]
    for method_label in method_labels:
        if method_label == trace_label:
            continue
        metric_results = {}
        for metric_name in metric_names:
            trace_values = trace_runs.get(metric_name, [])
            baseline_values = method_runs.get(method_label, {}).get(metric_name, [])
            pair_count = min(len(trace_values), len(baseline_values))
            deltas = [
                trace_values[idx] - baseline_values[idx]
                for idx in range(pair_count)
            ]
            p_value = exact_sign_flip_pvalue(deltas) if pair_count >= 2 else None
            if pair_count >= 2 and p_value is None:
                notes.append(f"{metric_name} paired runs exceed exact-test limit")
            metric_results[metric_name] = {
                "trace_mean": float(statistics.fmean(trace_values)) if trace_values else 0.0,
                "baseline_mean": float(statistics.fmean(baseline_values)) if baseline_values else 0.0,
                "mean_delta": float(statistics.fmean(deltas)) if deltas else 0.0,
                "n_pairs": pair_count,
                "p_value": p_value,
                "significant_at_0.05": bool(p_value is not None and p_value < 0.05),
            }
        comparisons[method_label] = metric_results
    return {
        "test": "exact_sign_flip",
        "notes": sorted(set(notes)),
        "trace_vs_baselines": comparisons,
    }


def summary_table_fieldnames(k_list: List[int]) -> List[str]:
    fields = ["backbone", "model_name", "method"]
    for k in k_list:
        fields.extend([f"Hit@{k}", f"Recall@{k}", f"nDCG@{k}", f"RR@{k}"])
    for metric_name in ("ROUGE-L", "Char-F1", "AC", "AC_gold", "QAC", "OCR"):
        fields.extend([f"{metric_name}_mean", f"{metric_name}_std", f"{metric_name}_mean±std"])
    return fields


def summary_table_rows(summaries: List[Dict], k_list: List[int]) -> List[Dict[str, str]]:
    rows = []
    metric_key_map = {
        "ROUGE-L": "rouge_l",
        "Char-F1": "char_f1",
        "AC": "ac",
        "AC_gold": "ac_gold",
        "QAC": "qac",
        "OCR": "ocr",
    }
    for summary in summaries:
        retrieval_metrics = summary.get("retrieval", {}).get("metrics", {})
        generation_summary = summary.get("generation", {}).get("summary", {})
        for method_name in summary.get("methods", []):
            row = {
                "backbone": summary.get("backbone", ""),
                "model_name": summary.get("model_name", ""),
                "method": method_name,
            }
            for k in k_list:
                metrics = retrieval_metrics.get(method_name, {})
                row[f"Hit@{k}"] = format_float(metrics.get(f"Hit@{k}", 0.0))
                row[f"Recall@{k}"] = format_float(metrics.get(f"Recall@{k}", 0.0))
                row[f"nDCG@{k}"] = format_float(metrics.get(f"nDCG@{k}", 0.0))
                row[f"RR@{k}"] = format_float(metrics.get(f"RR@{k}", 0.0))
            gen_metrics = generation_summary.get(method_name, {})
            for display_name, metric_key in metric_key_map.items():
                payload = gen_metrics.get(metric_key, {})
                row[f"{display_name}_mean"] = format_float(payload.get("mean", 0.0))
                row[f"{display_name}_std"] = format_float(payload.get("std", 0.0))
                row[f"{display_name}_mean±std"] = format_mean_std(payload)
            rows.append(row)
    return rows


def write_summary_tables(summaries: List[Dict], output_dir: Path, basename: str) -> None:
    if not summaries:
        return
    k_union = sorted({
        int(k)
        for summary in summaries
        for k in summary.get("retrieval", {}).get("k_list", [])
    })
    fieldnames = summary_table_fieldnames(k_union)
    rows = summary_table_rows(summaries, k_union)
    write_csv(output_dir / f"{basename}.csv", fieldnames, rows)
    write_markdown_table(output_dir / f"{basename}.md", fieldnames, rows)


def load_existing_summaries(backbones: List[str]) -> List[Dict]:
    summaries = []
    for backbone in backbones:
        summary_path = output_dir_for_backbone(backbone) / "eval" / "all.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"missing summary for backbone={backbone}: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))
    return summaries


def required_retrieval_methods(methods: List[str]) -> List[str]:
    needed = set()
    for method_id in methods:
        spec = METHOD_SPECS[method_id]
        retrieval_method = spec.get("retrieval_method")
        if method_id == "trace":
            needed.add("trace")
        elif method_id == "cot-rag":
            needed.add("cot-rag")
        elif retrieval_method:
            needed.add(method_id)
    ordered = [method_id for method_id in DEFAULT_METHODS if method_id in needed]
    return ordered


def run_retrieval_stage(args: argparse.Namespace, backbone: str, model_name: str, outdir: Path) -> None:
    print(f"\n{'█' * 72}\n  STAGE: RETRIEVAL ({backbone})\n{'█' * 72}")
    reranker_url = getattr(args, "reranker_url", None)
    coarse_top_k = getattr(args, "coarse_top_k", COARSE_TOP_K)
    common = [
        "--splits-dir", args.splits_dir,
        "--split", args.split,
        "--index-dir", args.index_dir,
        "--top-k", str(args.top_k),
        "--server-url", EMBED_URL,
    ]
    exclude_doc_prefix = getattr(args, "exclude_doc_prefix", None)
    if exclude_doc_prefix:
        common.extend(["--exclude-doc-prefix", exclude_doc_prefix])
    methods_to_run = required_retrieval_methods(args.methods)
    for idx, method_id in enumerate(methods_to_run, start=1):
        log_path = outdir / "logs" / f"{idx:02d}.retrieval.{method_slug(method_id)}.log"
        output = retrieval_path(outdir, method_id)
        if method_id == "trace":
            cmd = [
                *py("-m", "trace_rag.cli.run_retrieval"),
                *common,
                "--dataset-tag", DATASET_TAG,
                "--model-tag", backbone,
                "--top-sem-k", str(args.top_sem_k),
                "--lambda-ot", str(args.lambda_ot),
                "--lambda-risk", str(args.lambda_risk),
                "--lambda-num", str(args.lambda_num),
                "--workers", str(args.workers_ret),
                "--method", method_label("trace"),
                "--output", str(output),
            ]
            pref_path = ROOT / args.pref_weights
            if pref_path.exists():
                cmd.extend(["--pref-weights", str(pref_path), "--no-pref-lambda"])
            if reranker_url:
                cmd.extend(["--reranker-url", reranker_url])
            label = f"[{idx}] Retrieval: {method_label('trace')}"
        elif method_id == "cot-rag":
            cmd = [
                *py("-m", "experiments.run_cot_rag_retrieval"),
                *common,
                "--dataset-tag", DATASET_TAG,
                "--model-tag", backbone,
                "--top-sem-k", str(args.top_sem_k),
                "--batch-size", "64",
                "--workers", str(args.workers_ret),
                "--method", method_label(method_id),
                "--resume",
                "--output", str(output),
            ]
            if reranker_url:
                cmd.extend(["--reranker-url", reranker_url])
            label = f"[{idx}] Retrieval: {method_label(method_id)}"
        else:
            retrieval_method = METHOD_SPECS[method_id]["retrieval_method"]
            cmd = [
                *py("-m", "experiments.run_retrieval_baselines"),
                *common,
                "--dataset-tag", DATASET_TAG,
                "--model-tag", backbone,
                "--model", model_name,
                "--method", retrieval_method,
                "--workers", str(args.workers_ret),
                "--coarse-top-k", str(coarse_top_k),
                "--resume",
                "--output", str(output),
            ]
            if reranker_url:
                cmd.extend(["--reranker-url", reranker_url])
            label = f"[{idx}] Retrieval: {method_label(method_id)}"
        if not run(cmd, log_path, label):
            raise RuntimeError(f"retrieval failed: {method_id}")


def run_evidence_stage(args: argparse.Namespace, backbone: str, outdir: Path) -> None:
    print(f"\n{'█' * 72}\n  STAGE: EVIDENCE ({backbone})\n{'█' * 72}")
    idx = 1
    for method_id in args.methods:
        spec = METHOD_SPECS[method_id]
        log_path = outdir / "logs" / f"{idx:02d}.evidence.{method_slug(method_id)}.log"
        if spec["type"] == "closed-book":
            idx += 1
            continue
        if spec["type"] in {"baseline-rag", "cot-rag"}:
            cmd = [
                *py("-m", "trace_rag.utils.build_baseline_chains"),
                "--retrieval", str(retrieval_path(outdir, method_id)),
                "--index-dir", args.index_dir,
                "--method", method_label(method_id),
                "--top-n", "3" if method_id == "cot-rag" else "1",
                "--output", str(evidence_path(outdir, method_id)),
            ]
        else:
            cmd = [
                *py("-m", "trace_rag.cli.run_evidence"),
                "--splits-dir", args.splits_dir,
                "--split", args.split,
                "--index-dir", args.index_dir,
                "--server-url", EMBED_URL,
                "--dataset-tag", DATASET_TAG,
                "--model-tag", backbone,
                "--top-m", str(args.top_m),
                "--top-sem-k", str(args.top_sem_k),
                "--beam-width", str(args.beam_width),
                "--eta", str(args.eta),
                "--lambda-ot", str(args.lambda_ot),
                "--lambda-risk", str(args.lambda_risk),
                "--lambda-num", str(args.lambda_num),
                "--workers", str(args.workers_evi),
                "--method", method_label(method_id),
                "--output", str(evidence_path(outdir, method_id)),
            ]
            pref_path = ROOT / args.pref_weights
            if pref_path.exists():
                cmd.extend(["--pref-weights", str(pref_path), "--no-pref-lambda"])
        if not run(cmd, log_path, f"[{idx}] Evidence: {method_label(method_id)}"):
            raise RuntimeError(f"evidence failed: {method_id}")
        idx += 1


def run_generation_stage(args: argparse.Namespace, backbone: str, model_name: str, outdir: Path) -> None:
    print(f"\n{'█' * 72}\n  STAGE: GENERATION ({backbone})\n{'█' * 72}")
    idx = 1
    for repeat_idx in range(1, args.num_repeats + 1):
        for method_id in args.methods:
            spec = METHOD_SPECS[method_id]
            output = generation_path(outdir, repeat_idx, method_id)
            log_path = outdir / "logs" / f"{idx:02d}.generation.r{repeat_idx:02d}.{method_slug(method_id)}.log"
            if spec["type"] == "closed-book":
                cmd = [
                    *py("-m", "experiments.run_generation_baselines"),
                    "--dataset-tag", DATASET_TAG,
                    "--model-tag", backbone,
                    "--splits-dir", args.splits_dir,
                    "--split", args.split,
                    "--model", model_name,
                    "--prompt-style", "closed-book",
                    "--workers", str(args.workers_gen),
                    "--method", method_label(method_id),
                    "--output", str(output),
                ]
            elif spec["type"] in {"baseline-rag", "cot-rag"}:
                if spec["type"] == "cot-rag":
                    cmd = [
                        *py("-m", "experiments.run_cot_rag_generation"),
                        "--dataset-tag", DATASET_TAG,
                        "--model-tag", backbone,
                        "--chains", str(evidence_path(outdir, method_id)),
                        "--model", model_name,
                        "--workers", str(args.workers_gen),
                        "--method", method_label(method_id),
                        "--output", str(output),
                    ]
                else:
                    cmd = [
                        *py("-m", "experiments.run_generation_baselines"),
                        "--dataset-tag", DATASET_TAG,
                        "--model-tag", backbone,
                        "--chains", str(evidence_path(outdir, method_id)),
                        "--model", model_name,
                        "--prompt-style", "rag",
                        "--workers", str(args.workers_gen),
                        "--method", method_label(method_id),
                        "--output", str(output),
                    ]
            else:
                cmd = [
                    *py("-m", "trace_rag.cli.run_generate"),
                    "--dataset-tag", DATASET_TAG,
                    "--model-tag", backbone,
                    "--chains", str(evidence_path(outdir, method_id)),
                    "--model", model_name,
                    "--server-url", EMBED_URL,
                    "--lambda-attr", str(args.lambda_attr),
                    "--delta-a", str(args.delta_a),
                    "--delta-v", str(args.delta_v),
                    "--workers", str(args.workers_gen),
                    "--method", method_label(method_id),
                    "--output", str(output),
                ]
            if not run(cmd, log_path, f"[{idx}] Generation r{repeat_idx}: {method_label(method_id)}"):
                raise RuntimeError(f"generation failed: {method_id} repeat={repeat_idx}")
            idx += 1


def run_eval_stage(args: argparse.Namespace, backbone: str, outdir: Path) -> Dict:
    print(f"\n{'█' * 72}\n  STAGE: EVAL ({backbone})\n{'█' * 72}")
    retrieval_files = [
        str(retrieval_path(outdir, method_id))
        for method_id in required_retrieval_methods(args.methods)
        if retrieval_path(outdir, method_id).exists()
    ]
    retrieval_eval_path = outdir / "eval" / "retrieval.json"
    if retrieval_files:
        cmd = [
            *py("-m", "experiments.evaluate.retrieval"),
            "--results", *retrieval_files,
            "--splits-dir", args.splits_dir,
            "--split", args.split,
            "--qrels", args.qrels,
            "--gold", args.gold,
            "--k-list", ",".join(str(value) for value in args.k_list),
            "--output", str(retrieval_eval_path),
        ]
        log_path = outdir / "logs" / "90.eval.retrieval.log"
        if not run(cmd, log_path, f"Eval retrieval: {backbone}"):
            raise RuntimeError("retrieval evaluation failed")

    generation_results: Dict[str, Dict[str, Dict]] = {}
    for repeat_idx in range(1, args.num_repeats + 1):
        repeat_key = f"repeat-{repeat_idx:02d}"
        generation_results[repeat_key] = {}
        for method_id in args.methods:
            pred_path = generation_path(outdir, repeat_idx, method_id)
            if not pred_path.exists():
                continue
            output = generation_eval_path(outdir, repeat_idx, method_id)
            cmd = [
                *py("-m", "experiments.evaluate.generation"),
                "--pred", str(pred_path),
                "--index-dir", args.index_dir,
                "--server-url", EMBED_URL,
                "--lambda-attr", str(args.lambda_attr),
                "--ac-threshold", str(args.delta_a),
                "--qac-threshold", str(args.delta_v),
                "--gold", args.gold,
                "--qrels", args.qrels,
                "--output", str(output),
            ]
            log_path = outdir / "logs" / f"91.eval.gen.r{repeat_idx:02d}.{method_slug(method_id)}.log"
            if not run(cmd, log_path, f"Eval generation r{repeat_idx}: {method_label(method_id)}"):
                raise RuntimeError(f"generation evaluation failed: {method_id} repeat={repeat_idx}")
            with open(output, "r", encoding="utf-8") as f:
                generation_results[repeat_key][method_label(method_id)] = json.load(f)

    retrieval_metrics = {}
    if retrieval_eval_path.exists():
        with open(retrieval_eval_path, "r", encoding="utf-8") as f:
            retrieval_payload = json.load(f)
        retrieval_metrics = retrieval_payload.get("metrics", {})

    summary = aggregate_summary(args, backbone, retrieval_metrics, generation_results)
    summary_path = outdir / "eval" / "all.json"
    write_json(summary_path, summary)
    write_summary_tables([summary], outdir / "eval", "main_table")
    print_summary(summary)
    return summary


def aggregate_metric(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "runs": 0}
    mean = float(statistics.fmean(values))
    std = float(statistics.stdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "runs": len(values)}


def aggregate_summary(
    args: argparse.Namespace,
    backbone: str,
    retrieval_metrics: Dict[str, Dict[str, float]],
    generation_results: Dict[str, Dict[str, Dict]],
) -> Dict:
    consistency = consistency_report(args.splits_dir, args.qrels, args.gold, args.split)
    generation_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    metric_names = ["rouge_l", "char_f1", "ac", "ac_gold", "qac", "ocr"]
    method_labels = [method_label(method_id) for method_id in args.methods]
    for method_id in args.methods:
        label = method_label(method_id)
        per_run = []
        for repeat_payload in generation_results.values():
            if label in repeat_payload:
                per_run.append(repeat_payload[label])
        generation_summary[label] = {
            metric_name: aggregate_metric(
                [float(run_payload.get(metric_name, 0.0)) for run_payload in per_run]
            )
            for metric_name in metric_names
        }
    statistics_summary = build_statistical_summary(
        method_labels=method_labels,
        generation_results=generation_results,
        metric_names=metric_names,
    )
    return {
        "dataset": DATASET_TAG,
        "backbone": backbone,
        "model_name": BACKBONE_REGISTRY[backbone],
        "split": args.split,
        "methods": method_labels,
        "num_repeats": args.num_repeats,
        "protocol": {
            "top_k": args.top_k,
            "top_sem_k": args.top_sem_k,
            "top_m": args.top_m,
            "beam_width": args.beam_width,
            "eta": args.eta,
            "lambda_ot": args.lambda_ot,
            "lambda_risk": args.lambda_risk,
            "lambda_num": args.lambda_num,
            "lambda_attr": args.lambda_attr,
            "delta_a": args.delta_a,
            "delta_v": args.delta_v,
            "splits_dir": args.splits_dir,
            "qrels": args.qrels,
            "gold": args.gold,
        },
        "consistency": consistency,
        "retrieval": {
            "k_list": args.k_list,
            "metrics": retrieval_metrics,
        },
        "generation": {
            "per_run": generation_results,
            "summary": generation_summary,
        },
        "statistics": statistics_summary,
    }


def print_summary(summary: Dict) -> None:
    print("\n" + "=" * 72)
    print(f"  RESULTS SUMMARY ({summary['backbone']})")
    print("=" * 72)
    consistency = summary.get("consistency", {})
    warnings = consistency.get("warnings", [])
    if warnings:
        print(f"[Consistency Warning] {'; '.join(warnings)}")
    ret = summary.get("retrieval", {}).get("metrics", {})
    if ret:
        k_list = [int(k) for k in summary.get("retrieval", {}).get("k_list", [])]
        preferred_k = int(summary.get("protocol", {}).get("top_k", 0) or 0)
        if preferred_k not in k_list and k_list:
            preferred_k = k_list[0]
        print("\n## Retrieval")
        print(
            f"{'Method':<20} "
            f"{f'Hit@{preferred_k}':>8} "
            f"{f'R@{preferred_k}':>8} "
            f"{f'nDCG@{preferred_k}':>10} "
            f"{f'RR@{preferred_k}':>8}"
        )
        print("-" * 62)
        for method_name in summary.get("methods", []):
            metrics = ret.get(method_name)
            if not metrics:
                continue
            print(
                f"{method_name:<20} "
                f"{metrics.get(f'Hit@{preferred_k}', 0.0):>8.4f} "
                f"{metrics.get(f'Recall@{preferred_k}', 0.0):>8.4f} "
                f"{metrics.get(f'nDCG@{preferred_k}', 0.0):>10.4f} "
                f"{metrics.get(f'RR@{preferred_k}', 0.0):>8.4f}"
            )
    gen = summary.get("generation", {}).get("summary", {})
    if gen:
        print("\n## Generation (mean±std)")
        print(f"{'Method':<20} {'ROUGE-L':>10} {'Char-F1':>10} {'AC':>10} {'AC_gold':>10} {'QAC':>10} {'OCR':>10}")
        print("-" * 88)
        for method_name in summary.get("methods", []):
            metrics = gen.get(method_name)
            if not metrics:
                continue
            def fmt(metric_name: str) -> str:
                payload = metrics.get(metric_name, {})
                mean = payload.get("mean")
                if mean is None:
                    return "-"
                return f"{mean:.4f}±{payload.get('std', 0.0):.4f}"
            print(
                f"{method_name:<20} "
                f"{fmt('rouge_l'):>10} "
                f"{fmt('char_f1'):>10} "
                f"{fmt('ac'):>10} "
                f"{fmt('ac_gold'):>10} "
                f"{fmt('qac'):>10} "
                f"{fmt('ocr'):>10}"
            )
    stats = summary.get("statistics", {}).get("trace_vs_baselines", {})
    if stats:
        print("\n## TRACE Significance")
        for baseline_name, metric_payload in stats.items():
            ac_result = metric_payload.get("ac", {})
            qac_result = metric_payload.get("qac", {})
            if ac_result:
                ac_p = ac_result.get("p_value")
                qac_p = qac_result.get("p_value")
                ac_text = f"{ac_p:.4f}" if ac_p is not None else "n/a"
                qac_text = f"{qac_p:.4f}" if qac_p is not None else "n/a"
                print(f"{baseline_name:<20} AC p={ac_text}  QAC p={qac_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main Audit-OCIQA experiments")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["consistency", "retrieval", "evidence", "generation", "eval", "aggregate", "all"],
        help="Which stage to run",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=f"Comma-separated methods from: {', '.join(DEFAULT_METHODS)}",
    )
    parser.add_argument(
        "--backbones",
        default=",".join(DEFAULT_BACKBONES),
        help=f"Comma-separated backbones from: {', '.join(BACKBONE_REGISTRY.keys())}",
    )
    parser.add_argument("--splits-dir", default=SPLITS_DIR, help="Split directory")
    parser.add_argument("--qrels", default=QRELS, help="Qrels JSONL path")
    parser.add_argument("--gold", default=GOLD, help="Gold QA JSONL path")
    parser.add_argument("--index-dir", default=INDEX_DIR, help="Index directory")
    parser.add_argument("--pref-weights", default=PREF_WEIGHTS, help="TRACE preference weights JSON path")
    parser.add_argument("--split", default=SPLIT, help="Which split to run (e.g. train, dev, test, test-mini, all)")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Final retrieval K")
    parser.add_argument("--top-sem-k", type=int, default=TOP_SEM_K, help="Semantic candidate K")
    parser.add_argument("--top-m", type=int, default=TOP_M, help="Evidence candidate M")
    parser.add_argument("--beam-width", type=int, default=BEAM_WIDTH, help="Evidence beam width")
    parser.add_argument("--eta", type=float, default=ETA, help="Evidence link weight")
    parser.add_argument("--lambda-ot", type=float, default=LAMBDA_OT, help="TRACE lambda_ot")
    parser.add_argument("--lambda-risk", type=float, default=LAMBDA_RISK, help="TRACE lambda_risk")
    parser.add_argument("--lambda-num", type=float, default=LAMBDA_NUM, help="TRACE lambda_num")
    parser.add_argument("--lambda-attr", type=float, default=LAMBDA_ATTR, help="TRACE lambda_attr")
    parser.add_argument("--delta-a", type=float, default=DELTA_A, help="AC/OCR threshold")
    parser.add_argument("--delta-v", type=float, default=DELTA_V, help="QAC threshold")
    parser.add_argument("--exclude-doc-prefix", default=EXCLUDE_DOC_PREFIX,
                        help="Exclude evidence IDs starting with this prefix (e.g. 'A2023-<doc_id>')")
    parser.add_argument("--num-repeats", type=int, default=NUM_REPEATS, help="Repeated generation runs")
    parser.add_argument("--workers-gen", type=int, default=WORKERS_GEN, help="Generation workers")
    parser.add_argument("--workers-ret", type=int, default=WORKERS_RET, help="Retrieval workers")
    parser.add_argument("--workers-evi", type=int, default=WORKERS_EVI, help="Evidence workers")
    parser.add_argument("--reranker-url", default=None, help="Reranker server URL(s)")
    parser.add_argument("--coarse-top-k", type=int, default=COARSE_TOP_K, help="Coarse retrieval K before rerank")
    parser.add_argument("--k-list", default="5,10,20", help="K list for retrieval evaluation")
    parser.add_argument(
        "--skip-consistency-check",
        action="store_true",
        help="Skip split/qrels/gold consistency gate before running non-aggregate stages",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.methods = parse_csv(args.methods, METHOD_SPECS.keys())
    args.backbones = parse_csv(args.backbones, BACKBONE_REGISTRY.keys())
    args.k_list = [int(part) for part in parse_csv(args.k_list) or ["5", "10", "20"]]
    if args.top_k not in args.k_list:
        args.k_list = sorted(set([*args.k_list, args.top_k]))
    if not args.methods:
        raise ValueError("`methods` must not be empty")
    if not args.backbones:
        raise ValueError("`backbones` must not be empty")
    if args.num_repeats < 1:
        raise ValueError("num_repeats must be >= 1")
    if args.stage == "aggregate":
        summaries = load_existing_summaries(args.backbones)
        write_summary_tables(summaries, ROOT / BASE_OUTDIR / "summary", "stage2_main_table")
        print(f"\n[Aggregate] wrote combined tables for backbones={args.backbones}")
        print("\n✓ Done.")
        return

    summaries: List[Dict] = []
    for backbone in args.backbones:
        model_name = BACKBONE_REGISTRY[backbone]
        outdir = output_dir_for_backbone(backbone)
        ensure_dirs(outdir)
        print(f"\n[Config] backbone={backbone}")
        print(f"[Config] model_name={model_name}")
        print(f"[Config] outdir={outdir}")
        print(f"[Config] methods={args.methods}")

        need_consistency_gate = args.stage == "consistency" or not args.skip_consistency_check
        if args.stage != "aggregate" and need_consistency_gate:
            report = consistency_report(args.splits_dir, args.qrels, args.gold, args.split)
            write_json(outdir / "eval" / "data_consistency.json", report)
            print(f"[Consistency] ok={report['ok']} split_counts={report['split_counts']}")
            if report.get("warnings"):
                print(f"[Consistency] warnings={report['warnings']}")
            if not report["ok"]:
                issues = "; ".join(report.get("issues", [])) or "unknown consistency issue"
                raise RuntimeError(f"data consistency check failed: {issues}")
            if args.stage == "consistency":
                continue
        if args.stage in {"retrieval", "all"}:
            run_retrieval_stage(args, backbone, model_name, outdir)
            if args.stage == "retrieval":
                continue
        if args.stage in {"evidence", "all"}:
            run_evidence_stage(args, backbone, outdir)
            if args.stage == "evidence":
                continue
        if args.stage in {"generation", "all"}:
            run_generation_stage(args, backbone, model_name, outdir)
            if args.stage == "generation":
                continue
        if args.stage in {"eval", "all"}:
            summaries.append(run_eval_stage(args, backbone, outdir))

    if summaries:
        write_summary_tables(summaries, ROOT / BASE_OUTDIR / "summary", "stage2_main_table")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
