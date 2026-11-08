#! /usr/bin/env python3
"""
TRACE generation runner.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

from trace_rag.utils.experiment_layout import validate_non_legacy_output_path
from trace_rag.generate import (
    DEFAULT_MODEL,
    DEFAULT_SERVER_URL,
    DEFAULT_DELTA_A,
    DEFAULT_DELTA_V,
    DEFAULT_LAMBDA_ATTR,
    DEFAULT_MAX_CITES,
    DEFAULT_MULTI_CITE_DELTA,
    align_claims,
    call_chat_api,
)


DEFAULT_DATASET_TAG = "audit-ociqa"


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def default_stage_path(dataset_tag: str, model_tag: str, stage: str, filename: str) -> str:
    return os.path.join("runs", dataset_tag, model_tag, stage, filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE generation (OCI + claim verification)")
    parser.add_argument(
        "--dataset-tag",
        default=DEFAULT_DATASET_TAG,
        help="Dataset tag for default input/output paths",
    )
    parser.add_argument(
        "--model-tag",
        default=None,
        help="Model tag for default input/output paths (default: inferred from --model)",
    )
    parser.add_argument(
        "--chains",
        default=None,
        help="Evidence chain JSONL path (default: runs/<dataset>/<model>/evidence/<method>.chains.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: runs/<dataset>/<model>/generation/<method>.pred.jsonl)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Embedding server URL")
    parser.add_argument("--lambda-attr", type=float, default=DEFAULT_LAMBDA_ATTR, help="Attribute consistency weight")
    parser.add_argument("--delta-a", type=float, default=DEFAULT_DELTA_A, help="Claim support threshold")
    parser.add_argument("--delta-v", type=float, default=DEFAULT_DELTA_V, help="QAC consistency threshold")
    parser.add_argument("--thresholds", default=None, help="JSON map of model->threshold")
    parser.add_argument("--max-retries", type=int, default=3, help="Chat API max retries")
    parser.add_argument("--no-attribute-consistency", action="store_true", help="Disable attribute consistency scoring")
    parser.add_argument("--no-local-suppression", action="store_true", help="Disable local unsupported-claim suppression")
    parser.add_argument("--lambda-tri", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-triple", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--multi-cite-delta",
        type=float,
        default=DEFAULT_MULTI_CITE_DELTA,
        help="Multi-cite score delta",
    )
    parser.add_argument("--max-cites", type=int, default=DEFAULT_MAX_CITES, help="Max cites per claim")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers for generation")
    parser.add_argument("--method", default=None, help="Method name override")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_tag = normalize_tag(args.model_tag or args.model, "model")
    if args.method:
        method_tag = normalize_tag(args.method, "trace")
    elif args.chains and args.chains.endswith(".chains.jsonl"):
        method_tag = normalize_tag(
            os.path.basename(args.chains)[: -len(".chains.jsonl")],
            "trace",
        )
    else:
        method_tag = "trace"
    chains_path = args.chains or default_stage_path(
        args.dataset_tag,
        model_tag,
        "evidence",
        f"{method_tag}.chains.jsonl",
    )
    output_path = args.output or default_stage_path(
        args.dataset_tag,
        model_tag,
        "generation",
        f"{method_tag}.pred.jsonl",
    )
    validate_non_legacy_output_path(output_path, label="TRACE generation output")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    thresholds_map = None
    if args.thresholds:
        with open(args.thresholds, "r", encoding="utf-8") as f:
            thresholds_map = json.load(f)
    lambda_attr = args.lambda_attr
    if args.lambda_tri is not None:
        lambda_attr = args.lambda_tri
    delta_a = args.delta_a
    if args.threshold is not None:
        delta_a = args.threshold
    if thresholds_map and args.model in thresholds_map:
        delta_a = float(thresholds_map[args.model])
    use_attribute_consistency = not (args.no_attribute_consistency or args.no_triple)
    use_local_suppression = not args.no_local_suppression

    run_config = {
        "dataset_tag": args.dataset_tag,
        "model": args.model,
        "lambda_attr": lambda_attr,
        "delta_a": delta_a,
        "delta_v": args.delta_v,
        "max_retries": args.max_retries,
        "server_url": args.server_url,
        "chains": chains_path,
        "method": args.method,
        "thresholds": args.thresholds,
        "use_attribute_consistency": use_attribute_consistency,
        "use_local_suppression": use_local_suppression,
        "multi_cite_delta": args.multi_cite_delta,
        "max_cites": args.max_cites,
    }

    done_ids = set()
    existing_config = None
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if existing_config is None:
                    existing_config = item.get("config")
                done_ids.add(item.get("row_id"))
    if existing_config and args.method is None:
        run_config["method"] = existing_config.get("method")
    if done_ids and existing_config is None:
        raise ValueError("output file exists but has no `config` field; pick a different --output path")
    if existing_config and existing_config != run_config:
        raise ValueError("output file exists with an inconsistent config; pick a different --output path")
    if done_ids:
        print(f"[generate] {len(done_ids)} entries already done; skipping")

    items = []
    with open(chains_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item.get("row_id") not in done_ids:
                items.append(item)

    def _process(item):
        evidence = [
            item["evidence"]["O"],
            item["evidence"]["C"],
            item["evidence"]["I"],
        ]
        extra_evidence = item.get("extra_evidence") or []
        if extra_evidence:
            evidence.extend(extra_evidence)
        method = args.method or item.get("method") or "TRACE"
        if item.get("chain_abstain_flag"):
            oci_raw = {"Observation": [], "Cause": [], "Impact": []}
            raw_output = ""
            aligned = {
                "prediction_O": [],
                "prediction_C": [],
                "prediction_I": [],
                "prediction_oci": {"Observation": [], "Cause": [], "Impact": []},
                "claim_support": {"Observation": [], "Cause": [], "Impact": []},
                "suppressed_claims": [],
                "abstain_flag": True,
                "abstain_reason": "delta_c",
                "local_metrics": {
                    "total_claims": 0,
                    "supported_claims": 0,
                    "numeric_claims": 0,
                    "qac_supported_claims": 0,
                    "suppressed_claims": 0,
                },
            }
        else:
            oci_raw, raw_output = call_chat_api(
                item.get("query", ""),
                evidence,
                args.model,
                max_retries=args.max_retries,
                dataset_tag=args.dataset_tag,
            )
            if args.dataset_tag == "finqa":
                aligned = {
                    "prediction_O": [],
                    "prediction_C": [],
                    "prediction_I": [],
                    "prediction_oci": {"Observation": [], "Cause": [], "Impact": []},
                    "claim_support": {},
                    "suppressed_claims": [],
                    "abstain_flag": False,
                    "abstain_reason": None,
                    "local_metrics": {
                        "total_claims": 0,
                        "supported_claims": 0,
                        "numeric_claims": 0,
                        "qac_supported_claims": 0,
                        "suppressed_claims": 0,
                    },
                }
            else:
                aligned = align_claims(
                    oci_raw,
                    evidence,
                    args.server_url,
                    lambda_attr=lambda_attr,
                    delta_a=delta_a,
                    delta_v=args.delta_v,
                    use_attribute_consistency=use_attribute_consistency,
                    use_local_suppression=use_local_suppression,
                    multi_cite_delta=args.multi_cite_delta,
                    max_cites=args.max_cites,
                )
        row_config = dict(run_config)
        row_config["method"] = method
        return {
            "row_id": item.get("row_id"),
            "qid": item.get("qid"),
            "method": method,
            "model": args.model,
            "used_evidence_ids": [e["evidence_id"] for e in evidence],
            "oci_raw": oci_raw,
            "raw_model_output": raw_output,
            "prediction_O": aligned["prediction_O"],
            "prediction_C": aligned["prediction_C"],
            "prediction_I": aligned["prediction_I"],
            "prediction_oci": aligned["prediction_oci"],
            "claim_support": aligned["claim_support"],
            "suppressed_claims": aligned["suppressed_claims"],
            "abstain_flag": aligned["abstain_flag"],
            "abstain_reason": aligned["abstain_reason"],
            "local_metrics": aligned["local_metrics"],
            "config": row_config,
        }

    workers = max(1, args.workers)
    write_lock = threading.Lock()
    with open(output_path, "a", encoding="utf-8") as f_out:
        if workers <= 1:
            for item in tqdm.tqdm(items, desc="Generate"):
                out = _process(item)
                f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process, it): it for it in items}
                for future in tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc="Generate"
                ):
                    out = future.result()
                    with write_lock:
                        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
                        f_out.flush()

    print(f"[generate] done -> {output_path}")


if __name__ == "__main__":
    main()
