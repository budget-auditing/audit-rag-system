#! /usr/bin/env python3
"""
Run analysis utilities: oracle bounds, efficiency, and failure modes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
from typing import Dict, Iterable, List, Set, Tuple

from experiments.evaluate.generation import evaluate_generation_metrics
from experiments.evaluate.stress import evaluate_stress_outputs
from trace_rag.utils.phase3 import (
    load_gold_map,
    load_meta,
    read_jsonl,
    stage3_root,
    write_json,
    write_jsonl,
)
from trace_rag.generate import align_claims, extract_triples, normalize_caliber


UNIT_ALIASES = {
    "%": "percent",
    "％": "percent",
    "percent": "percent",
    "basispoints": "basis_points",
    "bps": "basis_points",
    "trillion": "trillion",
    "billion": "billion",
    "million": "million",
    "thousand": "thousand",
    "hundred": "hundred",
    "万亿元": "ten_trillion_yuan",
    "千亿元": "hundred_billion_yuan",
    "百亿元": "ten_billion_yuan",
    "十亿元": "billion_yuan",
    "亿元": "hundred_million_yuan",
    "千万元": "ten_million_yuan",
    "百万元": "million_yuan",
    "十万元": "hundred_thousand_yuan",
    "万元": "ten_thousand_yuan",
    "元": "yuan",
    "‰": "permille",
    "万": "ten_thousand",
    "亿": "hundred_million",
}

UNIT_SCALES = {
    "percent": 0.01,
    "permille": 0.001,
    "basis_points": 0.0001,
    "hundred": 100.0,
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
    "trillion": 1_000_000_000_000.0,
    "yuan": 1.0,
    "hundred_thousand_yuan": 100_000.0,
    "million_yuan": 1_000_000.0,
    "ten_million_yuan": 10_000_000.0,
    "ten_thousand_yuan": 10_000.0,
    "hundred_million_yuan": 100_000_000.0,
    "billion_yuan": 1_000_000_000.0,
    "ten_billion_yuan": 10_000_000_000.0,
    "hundred_billion_yuan": 100_000_000_000.0,
    "ten_trillion_yuan": 1_000_000_000_000.0,
    "ten_thousand": 10_000.0,
    "hundred_million": 100_000_000.0,
}


def _collect_claims(oci: Dict, role_name: str) -> List[Dict]:
    items = oci.get(role_name, []) if isinstance(oci, dict) else []
    if isinstance(items, dict):
        items = [items]
    elif isinstance(items, str):
        items = [{"claim": items}]
    elif not isinstance(items, list):
        items = []
    results = []
    for item in items:
        if isinstance(item, dict):
            claim = str(item.get("claim") or "").strip()
            cites = item.get("cites") or []
        else:
            claim = str(item or "").strip()
            cites = []
        if claim:
            results.append({"claim": claim, "cites": cites})
    return results


def _prediction_roles(oci: Dict) -> Set[str]:
    roles = set()
    for role_name in ("Observation", "Cause", "Impact"):
        if _collect_claims(oci, role_name):
            roles.add(role_name)
    return roles


def _parse_elapsed_seconds(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    matches = re.findall(r"elapsed=(\d+)s", text)
    if not matches:
        return None
    values = [float(item) for item in matches]
    nonzero = [item for item in values if item > 0]
    return max(nonzero) if nonzero else max(values)


def _safe_fmean(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _unique_rows(path: str) -> List[Dict]:
    rows = {}
    for item in read_jsonl(path):
        row_id = item.get("row_id")
        if row_id is None:
            continue
        rows[str(row_id)] = item
    return list(rows.values())


def _retrieval_hits(path: str, gold_map: Dict[str, Dict]) -> Dict[str, bool]:
    hits = {}
    for item in _unique_rows(path):
        row_id = str(item.get("row_id"))
        gold_ids = set((gold_map.get(row_id) or {}).get("gold_evidence_ids") or [])
        pred_ids = set(item.get("topk_evidence_ids") or [])
        hits[row_id] = bool(gold_ids & pred_ids)
    return hits


def _write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    key = re.sub(r"\s+", "", str(unit).strip().lower())
    return UNIT_ALIASES.get(key, key or None)


def _unit_scale(unit: str | None) -> float:
    normalized = _normalize_unit(unit)
    return UNIT_SCALES.get(normalized, 1.0)


def _scaled_value(triple: Dict) -> float | None:
    value = triple.get("value")
    if value is None:
        return None
    try:
        return float(value) * _unit_scale(triple.get("unit"))
    except Exception:
        return None


def _numbers_close(left: float | None, right: float | None, rel_tol: float = 0.05, abs_tol: float = 1e-6) -> bool:
    if left is None or right is None:
        return False
    if abs(left - right) <= abs_tol:
        return True
    denom = max(abs(left), abs(right), 1e-10)
    return abs(left - right) / denom <= rel_tol


def _caliber_mismatch(left: str | None, right: str | None) -> bool:
    left_norm = normalize_caliber(left)
    right_norm = normalize_caliber(right)
    return bool(left_norm and right_norm and left_norm != right_norm)


def _best_evidence_triple(claim_triple: Dict, evidence_triples: List[Dict]) -> Dict | None:
    if not evidence_triples:
        return None

    def _score(evidence_triple: Dict) -> Tuple[float, float]:
        score = 0.0
        claim_year = claim_triple.get("year")
        evidence_year = evidence_triple.get("year")
        if claim_year is None or evidence_year is None:
            score += 0.5
        elif claim_year == evidence_year:
            score += 3.0

        claim_caliber = normalize_caliber(claim_triple.get("caliber"))
        evidence_caliber = normalize_caliber(evidence_triple.get("caliber"))
        if claim_caliber is None or evidence_caliber is None:
            score += 0.5
        elif claim_caliber == evidence_caliber:
            score += 3.0

        claim_unit = _normalize_unit(claim_triple.get("unit"))
        evidence_unit = _normalize_unit(evidence_triple.get("unit"))
        if claim_unit is None or evidence_unit is None:
            score += 0.5
        elif claim_unit == evidence_unit:
            score += 2.0

        claim_value = _scaled_value(claim_triple)
        evidence_value = _scaled_value(evidence_triple)
        if _numbers_close(claim_value, evidence_value):
            score += 4.0
        elif claim_value is not None and evidence_value is not None:
            score -= abs(claim_value - evidence_value) / max(abs(claim_value), abs(evidence_value), 1.0)
        return score, -abs((claim_value or 0.0) - (evidence_value or 0.0))

    return max(evidence_triples, key=_score)


def _attribute_mismatch_breakdown(support_item: Dict, meta: Dict[str, Dict]) -> Dict[str, bool]:
    flags = {
        "numeric_mismatch": False,
        "unit_mismatch": False,
        "year_mismatch": False,
        "reporting_scope_mismatch": False,
    }
    if not bool(support_item.get("has_numeric_attr")) or bool(support_item.get("qac_pass")):
        return flags

    claim = str(support_item.get("claim") or "").strip()
    if not claim:
        return flags

    best_evidence_id = support_item.get("best_evidence_id")
    evidence_meta = meta.get(best_evidence_id, {}) if best_evidence_id else {}
    evidence_text = (
        support_item.get("best_evidence_text")
        or evidence_meta.get("text_with_meta")
        or evidence_meta.get("text")
        or ""
    )
    claim_triples = extract_triples(claim)
    evidence_triples = extract_triples(
        evidence_text,
        caliber_hint=evidence_meta.get("caliber"),
        year_hint=evidence_meta.get("year"),
    )

    for claim_triple in claim_triples:
        evidence_triple = _best_evidence_triple(claim_triple, evidence_triples)
        if evidence_triple is None:
            continue

        claim_year = claim_triple.get("year")
        evidence_year = evidence_triple.get("year")
        if claim_year is not None and evidence_year is not None and claim_year != evidence_year:
            flags["year_mismatch"] = True

        if _caliber_mismatch(claim_triple.get("caliber"), evidence_triple.get("caliber")):
            flags["reporting_scope_mismatch"] = True

        claim_unit = _normalize_unit(claim_triple.get("unit"))
        evidence_unit = _normalize_unit(evidence_triple.get("unit"))
        claim_value = _scaled_value(claim_triple)
        evidence_value = _scaled_value(evidence_triple)
        if claim_unit and evidence_unit and claim_unit != evidence_unit:
            flags["unit_mismatch"] = True
            if not _numbers_close(claim_value, evidence_value):
                flags["numeric_mismatch"] = True
        elif not _numbers_close(claim_value, evidence_value):
            flags["numeric_mismatch"] = True

    if not any(flags.values()):
        claim_years = {triple.get("year") for triple in claim_triples if triple.get("year") is not None}
        evidence_year = evidence_meta.get("year")
        if claim_years and evidence_year is not None and any(year != evidence_year for year in claim_years):
            flags["year_mismatch"] = True

        claim_calibers = {
            normalize_caliber(triple.get("caliber"))
            for triple in claim_triples
            if triple.get("caliber")
        }
        evidence_caliber = normalize_caliber(evidence_meta.get("caliber"))
        if claim_calibers and evidence_caliber and any(caliber != evidence_caliber for caliber in claim_calibers):
            flags["reporting_scope_mismatch"] = True

    if not any(flags.values()):
        flags["numeric_mismatch"] = True
    return flags


def build_oracle_chain_predictions(
    gold_map: Dict[str, Dict],
    active_ids: Set[str],
    meta: Dict[str, Dict],
    server_url: str,
    lambda_attr: float,
    delta_a: float,
    delta_v: float,
) -> List[Dict]:
    rows = []
    for row_id in sorted(active_ids):
        gold_row = gold_map[row_id]
        gold_oci = copy.deepcopy(gold_row.get("gold_oci") or {})
        cited_ids = []
        evidence = []
        for role_name in ("Observation", "Cause", "Impact"):
            for claim_item in _collect_claims(gold_oci, role_name):
                for evidence_id in claim_item.get("cites") or []:
                    if evidence_id in meta and evidence_id not in cited_ids:
                        cited_ids.append(evidence_id)
                        evidence.append(dict(meta[evidence_id]))
        aligned = align_claims(
            gold_oci,
            evidence,
            server_url,
            lambda_attr=lambda_attr,
            delta_a=delta_a,
            delta_v=delta_v,
            use_local_suppression=False,
        )
        rows.append(
            {
                "row_id": row_id,
                "qid": gold_row.get("qid"),
                "method": "Oracle chain",
                "model": "gold",
                "used_evidence_ids": cited_ids,
                "oci_raw": gold_oci,
                "raw_model_output": json.dumps(gold_oci, ensure_ascii=False),
                "prediction_O": aligned["prediction_O"],
                "prediction_C": aligned["prediction_C"],
                "prediction_I": aligned["prediction_I"],
                "prediction_oci": aligned["prediction_oci"],
                "claim_support": aligned["claim_support"],
                "suppressed_claims": [],
                "abstain_flag": False,
                "abstain_reason": None,
                "local_metrics": aligned["local_metrics"],
                "config": {"method": "Oracle chain"},
            }
        )
    return rows


def build_oracle_evidence_chains(
    gold_map: Dict[str, Dict],
    active_ids: Set[str],
    meta: Dict[str, Dict],
) -> List[Dict]:
    empty = {"evidence_id": None, "text": "", "text_with_meta": "", "year": None, "caliber": None}
    rows = []
    for row_id in sorted(active_ids):
        gold_row = gold_map[row_id]
        role_evidence = {}
        all_gold = gold_row.get("gold_evidence_ids") or []
        for role_name, role_key in (("Observation", "O"), ("Cause", "C"), ("Impact", "I")):
            chosen = None
            for claim_item in _collect_claims(gold_row.get("gold_oci") or {}, role_name):
                for evidence_id in claim_item.get("cites") or []:
                    if evidence_id in meta:
                        chosen = dict(meta[evidence_id])
                        break
                if chosen:
                    break
            if chosen is None and all_gold:
                fallback = next((dict(meta[eid]) for eid in all_gold if eid in meta), None)
                chosen = fallback or dict(empty)
            role_evidence[role_key] = chosen or dict(empty)
        rows.append(
            {
                "row_id": row_id,
                "qid": gold_row.get("qid"),
                "query": gold_row.get("query", ""),
                "method": "Oracle evidence",
                "evidence": role_evidence,
                "score": 1.0,
                "coverage": 1.0,
                "link": 1.0,
                "config": {"method": "Oracle evidence", "source": "gold_oci"},
            }
        )
    return rows


def analyze_failures(
    pred_path: str,
    retrieval_path: str,
    gold_map: Dict[str, Dict],
    meta: Dict[str, Dict],
    stress_pred_path: str | None,
    stress_dataset_path: str | None,
) -> Dict:
    retrieval_hit = _retrieval_hits(retrieval_path, gold_map)
    rows = []
    counts = {
        "sparse_clue_miss": 0,
        "role_missing": 0,
        "attribute_mismatch": 0,
        "numeric_mismatch": 0,
        "unit_mismatch": 0,
        "year_mismatch": 0,
        "reporting_scope_mismatch": 0,
        "should_abstain_but_did_not": 0,
        "over_conservative_false_abstention": 0,
    }
    for item in _unique_rows(pred_path):
        row_id = str(item.get("row_id"))
        gold_row = gold_map.get(row_id)
        if not gold_row:
            continue
        gold_roles = _prediction_roles(gold_row.get("gold_oci") or {})
        pred_roles = _prediction_roles(item.get("prediction_oci") or {})
        sparse_clue_miss = not retrieval_hit.get(row_id, False)
        role_missing = bool(gold_roles - pred_roles)
        mismatch_flags = {
            "numeric_mismatch": False,
            "unit_mismatch": False,
            "year_mismatch": False,
            "reporting_scope_mismatch": False,
        }
        for support in (item.get("claim_support") or {}).values():
            for support_item in (support or []):
                support_flags = _attribute_mismatch_breakdown(support_item, meta)
                for key in mismatch_flags:
                    mismatch_flags[key] = mismatch_flags[key] or support_flags[key]
        attribute_mismatch = any(mismatch_flags.values())
        abstain_flag = bool(item.get("abstain_flag"))
        if not abstain_flag:
            abstain_flag = not pred_roles
        over_conservative = abstain_flag and retrieval_hit.get(row_id, False) and bool(gold_roles)
        row = {
            "row_id": row_id,
            "sparse_clue_miss": sparse_clue_miss,
            "role_missing": role_missing,
            "attribute_mismatch": attribute_mismatch,
            "numeric_mismatch": mismatch_flags["numeric_mismatch"],
            "unit_mismatch": mismatch_flags["unit_mismatch"],
            "year_mismatch": mismatch_flags["year_mismatch"],
            "reporting_scope_mismatch": mismatch_flags["reporting_scope_mismatch"],
            "should_abstain_but_did_not": False,
            "over_conservative_false_abstention": over_conservative,
        }
        for key in counts:
            counts[key] += int(bool(row[key]))
        rows.append(row)

    if stress_pred_path and stress_dataset_path and os.path.exists(stress_pred_path) and os.path.exists(stress_dataset_path):
        stress_dataset = {
            str(item.get("row_id")): item
            for item in read_jsonl(stress_dataset_path)
            if item.get("row_id")
        }
        phr_rows = set()
        for item in _unique_rows(stress_pred_path):
            row_id = str(item.get("row_id"))
            if row_id not in stress_dataset:
                continue
            abstain_flag = bool(item.get("abstain_flag"))
            if abstain_flag:
                continue
            for section in ("Observation", "Cause", "Impact"):
                for support_item in (item.get("claim_support") or {}).get(section, []) or []:
                    if bool(support_item.get("supported")) and (
                        not bool(support_item.get("has_numeric_attr")) or bool(support_item.get("qac_pass"))
                    ):
                        continue
                    phr_rows.add(row_id)
                    break
                if row_id in phr_rows:
                    break
        for row in rows:
            if row["row_id"] in phr_rows:
                row["should_abstain_but_did_not"] = True
        counts["should_abstain_but_did_not"] = len(phr_rows)
    return {"counts": counts, "rows": rows}


def analyze_efficiency(log_dir: str, row_count: int) -> Dict:
    trace_retrieval = _parse_elapsed_seconds(os.path.join(log_dir, "05.retrieval.trace.log"))
    naive_retrieval = _parse_elapsed_seconds(os.path.join(log_dir, "02.retrieval.naive-dense-rag.log"))
    trace_generation = _safe_fmean(
        [
            _parse_elapsed_seconds(os.path.join(log_dir, "06.generation.r01.trace.log")),
            _parse_elapsed_seconds(os.path.join(log_dir, "12.generation.r02.trace.log")),
            _parse_elapsed_seconds(os.path.join(log_dir, "18.generation.r03.trace.log")),
        ]
    )
    naive_generation = _safe_fmean(
        [
            _parse_elapsed_seconds(os.path.join(log_dir, "03.generation.r01.naive-dense-rag.log")),
            _parse_elapsed_seconds(os.path.join(log_dir, "09.generation.r02.naive-dense-rag.log")),
            _parse_elapsed_seconds(os.path.join(log_dir, "15.generation.r03.naive-dense-rag.log")),
        ]
    )
    trace_evidence = _parse_elapsed_seconds(os.path.join(log_dir, "06.evidence.trace.log"))
    naive_evidence = _parse_elapsed_seconds(os.path.join(log_dir, "03.evidence.naive-dense-rag.log"))
    return {
        "row_count": row_count,
        "trace": {
            "retrieval_total_seconds": trace_retrieval,
            "retrieval_avg_seconds": (trace_retrieval / row_count) if trace_retrieval else None,
            "evidence_total_seconds": trace_evidence,
            "evidence_avg_seconds": (trace_evidence / row_count) if trace_evidence else None,
            "generation_avg_total_seconds": trace_generation,
            "generation_avg_seconds": (trace_generation / row_count) if trace_generation else None,
        },
        "naive_dense_rag": {
            "retrieval_total_seconds": naive_retrieval,
            "retrieval_avg_seconds": (naive_retrieval / row_count) if naive_retrieval else None,
            "evidence_total_seconds": naive_evidence,
            "evidence_avg_seconds": (naive_evidence / row_count) if naive_evidence else None,
            "generation_avg_total_seconds": naive_generation,
            "generation_avg_seconds": (naive_generation / row_count) if naive_generation else None,
        },
        "relative_overhead_vs_naive_dense_rag": {
            "retrieval_ratio": (trace_retrieval / naive_retrieval) if trace_retrieval and naive_retrieval else None,
            "generation_ratio": (trace_generation / naive_generation) if trace_generation and naive_generation else None,
        },
        "training_free_note": "TRACE keeps zero task-specific training cost during robustness evaluation.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the analysis bundle")
    parser.add_argument("--dataset-tag", default="audit-ociqa")
    parser.add_argument("--model-tag", default="qwen")
    parser.add_argument("--trace-pred", default="runs/audit-ociqa/qwen/generation/repeat-01/trace.pred.jsonl")
    parser.add_argument("--trace-chains", default="runs/audit-ociqa/qwen/evidence/trace.chains.jsonl")
    parser.add_argument("--trace-retrieval", default="runs/audit-ociqa/qwen/retrieval/trace.jsonl")
    parser.add_argument("--stress-pred", default="runs/audit-ociqa/qwen/stage3/stress/trace/stress.pred.jsonl")
    parser.add_argument("--stress-dataset", default="runs/audit-ociqa/qwen/stage3/stress/trace/stress_dataset.jsonl")
    parser.add_argument("--gold", default="datasets-public/audit-ociqa-demo/qa.gold.jsonl")
    parser.add_argument("--index-dir", default="runs/index")
    parser.add_argument("--log-dir", default="runs/audit-ociqa/qwen/logs")
    parser.add_argument(
        "--server-url",
        default="http://localhost:8081/v1/embeddings",
    )
    parser.add_argument("--lambda-attr", type=float, default=0.7)
    parser.add_argument("--delta-a", type=float, default=0.2)
    parser.add_argument("--delta-v", type=float, default=0.2)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(
        stage3_root(args.dataset_tag, args.model_tag),
        "analysis",
    )
    os.makedirs(output_dir, exist_ok=True)

    gold_map = load_gold_map(args.gold)
    meta = load_meta(args.index_dir)
    active_ids = {str(item.get("row_id")) for item in read_jsonl(args.trace_pred) if item.get("row_id")}

    oracle_chain_rows = build_oracle_chain_predictions(
        gold_map,
        active_ids,
        meta,
        args.server_url,
        args.lambda_attr,
        args.delta_a,
        args.delta_v,
    )
    oracle_chain_path = os.path.join(output_dir, "oracle_chain.pred.jsonl")
    write_jsonl(oracle_chain_path, oracle_chain_rows)
    oracle_chain_eval = evaluate_generation_metrics(
        pred_path=oracle_chain_path,
        index_dir=args.index_dir,
        server_url=args.server_url,
        lambda_attr=args.lambda_attr,
        ac_threshold=args.delta_a,
        qac_threshold=args.delta_v,
        gold_path=args.gold,
    )
    write_json(os.path.join(output_dir, "oracle_chain.eval.json"), oracle_chain_eval)

    oracle_evidence_rows = build_oracle_evidence_chains(gold_map, active_ids, meta)
    oracle_evidence_path = os.path.join(output_dir, "oracle_evidence.chains.jsonl")
    write_jsonl(oracle_evidence_path, oracle_evidence_rows)

    failure = analyze_failures(
        pred_path=args.trace_pred,
        retrieval_path=args.trace_retrieval,
        gold_map=gold_map,
        meta=meta,
        stress_pred_path=args.stress_pred,
        stress_dataset_path=args.stress_dataset,
    )
    write_json(os.path.join(output_dir, "failure_modes.json"), failure["counts"])
    _write_csv(os.path.join(output_dir, "failure_modes.csv"), failure["rows"])

    efficiency = analyze_efficiency(args.log_dir, len(active_ids))
    write_json(os.path.join(output_dir, "efficiency.json"), efficiency)

    summary = {
        "oracle_chain": oracle_chain_eval,
        "oracle_evidence_chains_path": oracle_evidence_path,
        "failure_modes": failure["counts"],
        "efficiency": efficiency,
    }
    oracle_evidence_pred_path = os.path.join(output_dir, "oracle_evidence.pred.jsonl")
    if os.path.exists(oracle_evidence_pred_path):
        oracle_evidence_eval = evaluate_generation_metrics(
            pred_path=oracle_evidence_pred_path,
            index_dir=args.index_dir,
            server_url=args.server_url,
            lambda_attr=args.lambda_attr,
            ac_threshold=args.delta_a,
            qac_threshold=args.delta_v,
            gold_path=args.gold,
        )
        write_json(os.path.join(output_dir, "oracle_evidence.eval.json"), oracle_evidence_eval)
        summary["oracle_evidence"] = oracle_evidence_eval
    if os.path.exists(args.stress_pred) and os.path.exists(args.stress_dataset):
        stress_eval = evaluate_stress_outputs(
            pred_path=args.stress_pred,
            stress_dataset_path=args.stress_dataset,
            index_dir=args.index_dir,
            server_url=args.server_url,
            gold_path=args.gold,
            lambda_attr=args.lambda_attr,
            ac_threshold=args.delta_a,
            qac_threshold=args.delta_v,
        )
        summary["stress"] = stress_eval
    write_json(os.path.join(output_dir, "summary.json"), summary)


if __name__ == "__main__":
    main()
