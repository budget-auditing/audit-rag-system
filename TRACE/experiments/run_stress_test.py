#! /usr/bin/env python3
"""
Build and run attribute perturbation stress tests.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import tqdm

from experiments.evaluate.stress import evaluate_stress_outputs
from trace_rag.utils.phase3 import (
    chain_evidence_list,
    load_gold_map,
    normalize_tag,
    read_jsonl,
    stage3_root,
    write_json,
    write_jsonl,
)
from trace_rag.generate import align_claims, call_chat_api

NUM_REGEX = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
CALIBER_SWAP = {
    "一般公共预算": "政府性基金预算",
    "政府性基金预算": "一般公共预算",
    "财政拨款": "单位资金",
    "单位资金": "财政拨款",
    "上年结转结余": "其他收入",
    "其他收入": "上年结转结余",
}
UNIT_SWAP = {
    "万元": "亿元",
    "亿元": "万元",
    "元": "万元",
    "%": "个百分点",
    "％": "个百分点",
    "个百分点": "%",
}


def _format_number_like(token: str, value: float) -> str:
    token_no_comma = token.replace(",", "")
    decimals = 0
    if "." in token_no_comma:
        decimals = len(token_no_comma.split(".", 1)[1])
    fmt = f"{{:,.{decimals}f}}" if "," in token else f"{{:.{decimals}f}}"
    rendered = fmt.format(value)
    if decimals == 0:
        rendered = rendered.split(".", 1)[0]
    return rendered


def _perturb_value_token(token: str) -> str:
    raw = float(token.replace(",", ""))
    if raw == 0:
        new_value = 1.0
    else:
        delta = max(abs(raw) * 0.15, 1.0)
        new_value = raw + delta if raw > 0 else raw - delta
    return _format_number_like(token, new_value)


def _select_value_token(text: str, value: float) -> Optional[str]:
    for match in NUM_REGEX.finditer(text):
        token = match.group(0)
        try:
            parsed = float(token.replace(",", ""))
        except ValueError:
            continue
        if abs(parsed - float(value)) <= max(1e-6, abs(value) * 1e-6):
            return token
    return None


def _first_available_boundary(text: str) -> Optional[Dict[str, str]]:
    for src, dst in CALIBER_SWAP.items():
        if src in text:
            return {"original": src, "perturbed": dst}
    return None


def _first_available_unit(text: str, unit: str | None) -> Optional[Dict[str, str]]:
    ordered = [unit] if unit else []
    ordered.extend([item for item in UNIT_SWAP.keys() if item != unit])
    for token in ordered:
        if not token:
            continue
        if token in text and token in UNIT_SWAP:
            return {"original": token, "perturbed": UNIT_SWAP[token]}
    return None


def _build_perturbation(row_id: str, gold_row: Dict, base_row: Dict) -> Optional[Dict]:
    evidence_map = {
        item.get("evidence_id"): item
        for item in chain_evidence_list(base_row)
        if item.get("evidence_id")
    }
    triples = gold_row.get("triples") or []
    chooser = sum(ord(ch) for ch in row_id)
    candidates = []
    for triple in triples:
        cites = triple.get("cites") or []
        for evidence_id in cites:
            if evidence_id not in evidence_map:
                continue
            evidence = evidence_map[evidence_id]
            text = evidence.get("text") or ""
            value_token = _select_value_token(text, float(triple.get("value")))
            if value_token:
                candidates.append(
                    {
                        "target_evidence_id": evidence_id,
                        "perturb_type": "value",
                        "original_value_text": value_token,
                        "perturbed_value_text": _perturb_value_token(value_token),
                    }
                )
            year = triple.get("year")
            if year and str(year) in text:
                candidates.append(
                    {
                        "target_evidence_id": evidence_id,
                        "perturb_type": "year",
                        "original_year_text": str(year),
                        "perturbed_year_text": str(int(year) + 1),
                    }
                )
            unit_swap = _first_available_unit(text, triple.get("unit"))
            if unit_swap:
                candidates.append(
                    {
                        "target_evidence_id": evidence_id,
                        "perturb_type": "unit",
                        "original_unit_text": unit_swap["original"],
                        "perturbed_unit_text": unit_swap["perturbed"],
                    }
                )
            boundary_swap = _first_available_boundary(text)
            if boundary_swap:
                candidates.append(
                    {
                        "target_evidence_id": evidence_id,
                        "perturb_type": "boundary",
                        "original_boundary_text": boundary_swap["original"],
                        "perturbed_boundary_text": boundary_swap["perturbed"],
                    }
                )
    if not candidates:
        return None
    return candidates[chooser % len(candidates)]


def _replace_once(text: str, original: str, perturbed: str) -> str:
    return text.replace(original, perturbed, 1) if original and perturbed else text


def _apply_perturbation(row: Dict, perturbation: Dict) -> Dict:
    target_id = perturbation["target_evidence_id"]
    updated = copy.deepcopy(row)
    sections = ["O", "C", "I"]
    candidates = sections + [f"extra:{idx}" for idx, _ in enumerate(updated.get("extra_evidence") or [])]
    for locator in candidates:
        if locator.startswith("extra:"):
            idx = int(locator.split(":", 1)[1])
            evidence = updated.get("extra_evidence", [])[idx]
        else:
            evidence = updated.get("evidence", {}).get(locator, {})
        if evidence.get("evidence_id") != target_id:
            continue
        for prefix in ("value", "unit", "year", "boundary"):
            original_key = f"original_{prefix}_text"
            perturbed_key = f"perturbed_{prefix}_text"
            if perturbation.get(original_key) and perturbation.get(perturbed_key):
                evidence["text"] = _replace_once(
                    evidence.get("text") or "",
                    perturbation[original_key],
                    perturbation[perturbed_key],
                )
                evidence["text_with_meta"] = _replace_once(
                    evidence.get("text_with_meta") or "",
                    perturbation[original_key],
                    perturbation[perturbed_key],
                )
        if perturbation.get("perturb_type") == "year" and perturbation.get("perturbed_year_text"):
            try:
                evidence["year"] = int(perturbation["perturbed_year_text"])
            except ValueError:
                pass
        if perturbation.get("perturb_type") == "boundary" and perturbation.get("perturbed_boundary_text"):
            evidence["caliber"] = perturbation["perturbed_boundary_text"]
        break
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run attribute perturbation stress tests")
    parser.add_argument("--dataset-tag", default="audit-ociqa")
    parser.add_argument("--model-tag", default="qwen")
    parser.add_argument("--model", default="Qwen/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--chains", default="runs/audit-ociqa/qwen/evidence/trace.chains.jsonl")
    parser.add_argument("--gold", default="datasets-public/audit-ociqa-demo/qa.gold.jsonl")
    parser.add_argument("--index-dir", default="runs/index")
    parser.add_argument("--method", default="TRACE")
    parser.add_argument("--output-dir", default=None)
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
    parser.add_argument("--no-local-suppression", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    method_tag = normalize_tag(args.method, "trace")
    output_dir = args.output_dir or os.path.join(
        stage3_root(args.dataset_tag, args.model_tag),
        "stress",
        method_tag,
    )
    os.makedirs(output_dir, exist_ok=True)

    base_rows_map = {}
    for item in read_jsonl(args.chains):
        row_id = item.get("row_id")
        if row_id is None:
            continue
        base_rows_map[str(row_id)] = item
    base_rows = list(base_rows_map.values())
    gold_map = load_gold_map(args.gold)
    dataset_rows = []
    stress_rows = []
    for row in base_rows:
        if row.get("chain_abstain_flag"):
            continue
        row_id = str(row.get("row_id"))
        gold_row = gold_map.get(row_id)
        if not gold_row:
            continue
        perturbation = _build_perturbation(row_id, gold_row, row)
        if not perturbation:
            continue
        stress_row = _apply_perturbation(row, perturbation)
        dataset_row = {
            "row_id": row_id,
            "query": row.get("query"),
            "original_evidence_ids": [
                item.get("evidence_id")
                for item in chain_evidence_list(row)
                if item.get("evidence_id")
            ],
            "perturb_type": perturbation["perturb_type"],
            "target_evidence_id": perturbation["target_evidence_id"],
            "perturbed_evidence": {
                "evidence_id": perturbation["target_evidence_id"],
                "text": next(
                    (
                        item.get("text")
                        for item in chain_evidence_list(stress_row)
                        if item.get("evidence_id") == perturbation["target_evidence_id"]
                    ),
                    "",
                ),
            },
            "gold_answer": gold_row.get("gold_oci"),
            "gold_oci": gold_row.get("gold_oci"),
            **perturbation,
        }
        dataset_rows.append(dataset_row)
        stress_rows.append((row, stress_row, dataset_row))
        if args.max_rows and len(stress_rows) >= args.max_rows:
            break

    dataset_path = os.path.join(output_dir, "stress_dataset.jsonl")
    pred_path = os.path.join(output_dir, "stress.pred.jsonl")
    eval_path = os.path.join(output_dir, "stress.eval.json")
    write_jsonl(dataset_path, dataset_rows)

    if not args.skip_generate:
        done_ids = set()
        if os.path.exists(pred_path):
            for item in read_jsonl(pred_path):
                if item.get("row_id"):
                    done_ids.add(str(item["row_id"]))

        run_config = {
            "model": args.model,
            "method": f"{args.method}-stress",
            "chains": args.chains,
            "stress_dataset": dataset_path,
            "lambda_attr": args.lambda_attr,
            "delta_a": args.delta_a,
            "delta_v": args.delta_v,
            "use_local_suppression": not args.no_local_suppression,
        }

        def _process(bundle):
            original_row, perturbed_row, dataset_row = bundle
            row_id = str(dataset_row["row_id"])
            input_evidence = chain_evidence_list(perturbed_row)
            original_evidence = chain_evidence_list(original_row)
            oci_raw, raw_output = call_chat_api(
                dataset_row.get("query") or "",
                input_evidence,
                args.model,
                max_retries=args.max_retries,
            )
            aligned_input = align_claims(
                oci_raw,
                input_evidence,
                args.server_url,
                lambda_attr=args.lambda_attr,
                delta_a=args.delta_a,
                delta_v=args.delta_v,
                use_local_suppression=not args.no_local_suppression,
            )
            aligned_original = align_claims(
                oci_raw,
                original_evidence,
                args.server_url,
                lambda_attr=args.lambda_attr,
                delta_a=args.delta_a,
                delta_v=args.delta_v,
                use_local_suppression=False,
            )
            return {
                "row_id": row_id,
                "qid": original_row.get("qid"),
                "method": f"{args.method}-stress",
                "model": args.model,
                "used_evidence_ids": [
                    item.get("evidence_id")
                    for item in input_evidence
                    if item.get("evidence_id")
                ],
                "oci_raw": oci_raw,
                "raw_model_output": raw_output,
                "prediction_O": aligned_input["prediction_O"],
                "prediction_C": aligned_input["prediction_C"],
                "prediction_I": aligned_input["prediction_I"],
                "prediction_oci": aligned_input["prediction_oci"],
                "claim_support": aligned_original["claim_support"],
                "claim_support_input": aligned_input["claim_support"],
                "suppressed_claims": aligned_input["suppressed_claims"],
                "abstain_flag": aligned_input["abstain_flag"],
                "abstain_reason": aligned_input["abstain_reason"],
                "local_metrics": aligned_original["local_metrics"],
                "local_metrics_input": aligned_input["local_metrics"],
                "stress_meta": dataset_row,
                "config": run_config,
            }

        pending = [bundle for bundle in stress_rows if str(bundle[2]["row_id"]) not in done_ids]
        write_lock = threading.Lock()
        with open(pred_path, "a", encoding="utf-8") as f_out:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                futures = {pool.submit(_process, bundle): bundle for bundle in pending}
                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Stress"):
                    result = future.result()
                    with write_lock:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()

    metrics = evaluate_stress_outputs(
        pred_path=pred_path,
        stress_dataset_path=dataset_path,
        index_dir=args.index_dir,
        server_url=args.server_url,
        gold_path=args.gold,
        lambda_attr=args.lambda_attr,
        ac_threshold=args.ac_threshold,
        qac_threshold=args.qac_threshold,
    )
    write_json(eval_path, metrics)


if __name__ == "__main__":
    main()
