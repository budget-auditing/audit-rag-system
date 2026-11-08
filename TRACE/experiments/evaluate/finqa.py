#!/usr/bin/env python3
"""
Evaluate FinQA experiments.

Metrics:
  - Retrieval: Recall@K / nDCG@K / RR@K
  - Generation: EM / F1 / ROUGE-L / numeric consistency / answer coverage
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple


NUM_REGEX = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?%?")
YES_NO_REGEX = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
TOKEN_REGEX = re.compile(r"[\u4e00-\u9fff]|[a-z0-9]+(?:[._%-][a-z0-9]+)*", re.IGNORECASE)
DIAGNOSTIC_NOTES = {
    "numeric_consistency": "FinQA-only numeric diagnostic; not Audit-OCIQA QAC."
}


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def parse_k_list(raw: str | None) -> List[int]:
    values = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values or [1, 3, 5, 10, 20]


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower().replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in TOKEN_REGEX.finditer(normalize_text(text))]


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for token in a:
        curr = [0]
        for idx, token_b in enumerate(b, start=1):
            if token == token_b:
                curr.append(prev[idx - 1] + 1)
            else:
                curr.append(max(prev[idx], curr[idx - 1]))
        prev = curr
    return prev[-1]


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = 0
    gold_counts: Dict[str, int] = {}
    for token in gold_tokens:
        gold_counts[token] = gold_counts.get(token, 0) + 1
    for token in pred_tokens:
        if gold_counts.get(token, 0) > 0:
            common += 1
            gold_counts[token] -= 1
    if common <= 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def rouge_l(pred: str, gold: str) -> float:
    pred_tokens = tokenize(pred)
    gold_tokens = tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    if lcs <= 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def extract_numbers(text: str) -> List[Tuple[str, float]]:
    matches: List[Tuple[str, float]] = []
    for match in NUM_REGEX.finditer(text or ""):
        raw = match.group(0)
        normalized = raw.replace(",", "")
        is_pct = normalized.endswith("%")
        normalized = normalized.rstrip("%")
        try:
            value = float(normalized)
        except ValueError:
            continue
        if is_pct:
            value /= 100.0
        matches.append((raw, value))
    return matches


def coerce_numeric(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    numbers = extract_numbers(text)
    if numbers:
        return numbers[-1][1]
    try:
        return float(text)
    except ValueError:
        return None


def number_match(pred_num: float, gold_num: float, tol: float = 1e-4) -> bool:
    if gold_num == 0:
        return abs(pred_num) < tol
    rel_err = abs(pred_num - gold_num) / max(abs(gold_num), 1e-10)
    if rel_err < tol:
        return True
    if abs(gold_num) > 1e-10:
        rel_100 = abs(pred_num - gold_num * 100) / max(abs(gold_num * 100), 1e-10)
        rel_div100 = abs(pred_num * 100 - gold_num) / max(abs(gold_num), 1e-10)
        if rel_100 < tol or rel_div100 < tol:
            return True
    return False


def collect_claim_texts(oci: Dict) -> List[str]:
    claims: List[str] = []
    for section in ("Observation", "Cause", "Impact"):
        items = oci.get(section, []) if isinstance(oci, dict) else []
        if isinstance(items, dict):
            items = [items]
        elif isinstance(items, str):
            items = [items]
        elif not isinstance(items, list):
            items = []
        for item in items:
            if isinstance(item, dict):
                claim = str(item.get("claim") or "").strip()
            else:
                claim = str(item or "").strip()
            if claim:
                claims.append(claim)
    return claims


def gold_answer_text(gold_item: Dict) -> str:
    answer = str(gold_item.get("answer") or "").strip()
    if answer:
        return answer
    exe_ans = gold_item.get("exe_ans")
    return "" if exe_ans is None else str(exe_ans).strip()


def _try_dsl_execution(raw_output: str) -> Optional[float]:
    """Try to extract and execute a DSL program from LLM output."""
    try:
        from trace_rag.utils.finqa_dsl import extract_program_from_llm, execute_program
    except ImportError:
        return None
    steps = extract_program_from_llm(raw_output)
    if not steps:
        return None
    return execute_program(steps)


def extract_prediction_answer(item: Dict, gold_text: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Returns (pred_text, pred_num_direct, pred_num_dsl)."""
    oci = item.get("prediction_oci") or item.get("oci_raw") or {}
    claim_texts = collect_claim_texts(oci)
    raw_output = str(item.get("raw_model_output") or "").strip()
    candidates = [*claim_texts]
    if raw_output:
        candidates.append(raw_output)
    if not candidates:
        return None, None, None

    dsl_result = _try_dsl_execution(raw_output)

    gold_norm = normalize_text(gold_text)
    joined = "\n".join(candidates)
    if gold_norm in {"yes", "no", "true", "false"}:
        matches = YES_NO_REGEX.findall(joined)
        if matches:
            yes_val = 1.0 if matches[-1].lower() in ("yes", "true") else 0.0
            return matches[-1].lower(), yes_val, dsl_result

    numeric_matches: List[Tuple[str, float]] = []
    for text in candidates:
        numeric_matches.extend(extract_numbers(text))
    if numeric_matches:
        return numeric_matches[-1][0], numeric_matches[-1][1], dsl_result

    if claim_texts:
        return claim_texts[0], None, dsl_result

    first_nonempty = next((line.strip() for line in raw_output.splitlines() if line.strip()), None)
    if first_nonempty:
        return first_nonempty, None, dsl_result
    return None, None, dsl_result


def evaluate_retrieval(
    retrieval_path: str,
    qrels_path: str,
    k_list: Sequence[int],
) -> Dict[str, float]:
    qrels_data = read_jsonl(qrels_path)
    qrels = {}
    for item in qrels_data:
        row_id = item.get("row_id")
        gold = set(item.get("gold_evidence_ids") or [])
        if row_id and gold:
            qrels[str(row_id)] = gold

    results = {}
    for item in read_jsonl(retrieval_path):
        row_id = item.get("row_id")
        if row_id:
            results[str(row_id)] = item.get("topk_evidence_ids", [])

    common_rids = sorted(set(qrels.keys()) & set(results.keys()))
    metrics: Dict[str, float] = {}
    for k in k_list:
        recalls = []
        ndcgs = []
        rrs = []
        for row_id in common_rids:
            gold = qrels[row_id]
            pred = results.get(row_id, [])[:k]
            hit = 1.0 if any(evidence_id in gold for evidence_id in pred) else 0.0
            recalls.append(hit)

            hits = [1 if evidence_id in gold else 0 for evidence_id in pred]
            dcg = sum(1.0 / math.log2(idx + 2) for idx, match in enumerate(hits) if match)
            ideal = [1] * min(len(gold), k)
            idcg = sum(1.0 / math.log2(idx + 2) for idx in range(len(ideal)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

            rr = 0.0
            for idx, evidence_id in enumerate(pred, start=1):
                if evidence_id in gold:
                    rr = 1.0 / idx
                    break
            rrs.append(rr)

        metrics[f"Recall@{k}"] = sum(recalls) / len(recalls) if recalls else 0.0
        metrics[f"nDCG@{k}"] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
        metrics[f"RR@{k}"] = sum(rrs) / len(rrs) if rrs else 0.0
    metrics["n_queries"] = len(common_rids)
    return metrics


def evaluate_generation(
    generation_path: str,
    gold_path: str,
    tol: float = 1e-4,
) -> Dict[str, float]:
    gold_map = {}
    for item in read_jsonl(gold_path):
        row_id = item.get("row_id")
        if row_id:
            gold_map[str(row_id)] = item

    total = 0
    answer_found = 0
    em_sum = 0.0
    f1_sum = 0.0
    rouge_sum = 0.0
    numeric_total = 0
    numeric_correct_direct = 0
    numeric_correct_dsl = 0
    dsl_parsed = 0

    for item in read_jsonl(generation_path):
        row_id = item.get("row_id")
        gold_item = gold_map.get(str(row_id))
        if not gold_item:
            continue

        total += 1
        gold_text = gold_answer_text(gold_item)
        pred_text, pred_num, dsl_num = extract_prediction_answer(item, gold_text)
        if pred_text is not None:
            answer_found += 1
            em_sum += exact_match(pred_text, gold_text)
            f1_sum += token_f1(pred_text, gold_text)
            rouge_sum += rouge_l(pred_text, gold_text)

        gold_num = coerce_numeric(gold_item.get("exe_ans"))
        if gold_num is not None:
            numeric_total += 1
            if pred_num is None and pred_text is not None:
                pred_num = coerce_numeric(pred_text)
            if pred_num is not None and number_match(pred_num, gold_num, tol):
                numeric_correct_direct += 1
            if dsl_num is not None:
                dsl_parsed += 1
                if number_match(dsl_num, gold_num, tol):
                    numeric_correct_dsl += 1

    best_correct = 0
    for item in read_jsonl(generation_path):
        row_id = item.get("row_id")
        gold_item = gold_map.get(str(row_id))
        if not gold_item:
            continue
        gold_num = coerce_numeric(gold_item.get("exe_ans"))
        if gold_num is None:
            continue
        gold_text = gold_answer_text(gold_item)
        _, pred_num, dsl_num = extract_prediction_answer(item, gold_text)
        if pred_num is None:
            raw = str(item.get("raw_model_output") or "")
            pred_num = coerce_numeric(raw)
        hit_direct = pred_num is not None and number_match(pred_num, gold_num, tol)
        hit_dsl = dsl_num is not None and number_match(dsl_num, gold_num, tol)
        if hit_direct or hit_dsl:
            best_correct += 1

    exe_acc_best = (best_correct / numeric_total) if numeric_total else 0.0
    exe_acc_dsl = (numeric_correct_dsl / numeric_total) if numeric_total else 0.0
    exe_acc_direct = (numeric_correct_direct / numeric_total) if numeric_total else 0.0

    metrics = {
        "em": em_sum / total if total else 0.0,
        "f1": f1_sum / total if total else 0.0,
        "rouge_l": rouge_sum / total if total else 0.0,
        "exe_accuracy": exe_acc_best,
        "exe_accuracy_dsl": exe_acc_dsl,
        "exe_accuracy_direct": exe_acc_direct,
        "dsl_parse_rate": (dsl_parsed / numeric_total) if numeric_total else 0.0,
        "numeric_consistency": exe_acc_best,
        "numeric_accuracy": exe_acc_best,
        "answer_coverage": (answer_found / total) if total else 0.0,
        "total": total,
        "answer_found": answer_found,
        "numeric_total": numeric_total,
        "numeric_correct_dsl": numeric_correct_dsl,
        "numeric_correct_direct": numeric_correct_direct,
        "numeric_correct": best_correct,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FinQA experiments")
    parser.add_argument("--retrieval", default=None, help="Retrieval JSONL path")
    parser.add_argument("--generation", default=None, help="Generation JSONL path")
    parser.add_argument("--gold", default="datasets/finqa/qa_gold.jsonl", help="Gold answer JSONL")
    parser.add_argument("--qrels", default="datasets/finqa/gold.qrels.jsonl", help="Qrels JSONL")
    parser.add_argument("--k-list", default="1,3,5,10,20", help="Comma-separated K list for retrieval metrics")
    parser.add_argument("--tol", type=float, default=1e-4, help="Numerical tolerance")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    all_metrics: Dict[str, Dict] = {}
    all_metrics["meta"] = {"diagnostics": DIAGNOSTIC_NOTES}

    if args.retrieval and os.path.exists(args.retrieval):
        print("=== Retrieval Evaluation ===")
        ret_metrics = evaluate_retrieval(args.retrieval, args.qrels, parse_k_list(args.k_list))
        all_metrics["retrieval"] = ret_metrics
        for key, value in sorted(ret_metrics.items()):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if args.generation and os.path.exists(args.generation):
        print("\n=== Generation Evaluation ===")
        gen_metrics = evaluate_generation(args.generation, args.gold, args.tol)
        all_metrics["generation"] = gen_metrics
        for key, value in sorted(gen_metrics.items()):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved -> {args.output}")


if __name__ == "__main__":
    main()
