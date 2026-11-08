#! /usr/bin/env python3
"""
Run baseline generation methods.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Tuple
from urllib import request

import tqdm

from trace_rag.generate import CHAT_BASE_URL, DEFAULT_MODEL, parse_oci_json
from trace_rag.retrieval import load_queries
from trace_rag.utils.experiment_layout import DEFAULT_AUDIT_SPLITS_DIR, validate_non_legacy_output_path


DEFAULT_DATASET_TAG = "audit-ociqa"
EMPTY_OCI = {"Observation": [], "Cause": [], "Impact": []}


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def default_output_path(dataset_tag: str, model_tag: str, method: str) -> str:
    filename = f"{normalize_tag(method, 'baseline')}.pred.jsonl"
    return os.path.join("runs", dataset_tag, model_tag, "generation", filename)


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def detect_lang(text: str) -> str:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk > len(text) * 0.1 else "en"


def unique_evidence(evidence: List[Dict]) -> List[Dict]:
    seen = set()
    results = []
    for item in evidence:
        evidence_id = item.get("evidence_id")
        key = evidence_id or json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


def build_prompt(query: str, evidence: List[Dict], prompt_style: str, dataset_tag: str) -> str:
    evidence_text = "\n\n".join(
        [f"[{item.get('evidence_id') or 'N/A'}]\n{item.get('text') or ''}" for item in evidence]
    ).strip()
    lang = detect_lang(query + evidence_text[:500])
    if dataset_tag == "finqa":
        from trace_rag.utils.finqa_dsl import format_dsl_prompt_block
        dsl_block = format_dsl_prompt_block()
        if prompt_style == "closed-book":
            return (
                f"{dsl_block}\n"
                f"Q: {query}\nProgram:"
            )
        return (
            f"{dsl_block}\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Q: {query}\nProgram:"
        )
    if prompt_style == "closed-book":
        if lang == "zh":
            return (
                "你是预算审计专家。请基于你已有的知识直接回答问题，并输出 OCI 结构 "
                "（Observation/Cause/Impact）的 JSON。若无法确定，请保持克制，不要编造具体数值或证据。\n"
                "输出格式："
                "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}\n\n"
                f"问题：{query}\n"
            )
        return (
            "You are a financial audit expert. Answer the question directly from prior knowledge "
            "and output an OCI JSON (Observation/Cause/Impact). If uncertain, stay conservative "
            "and do not fabricate numbers or evidence.\n"
            "Output format: "
            "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}\n\n"
            f"Question: {query}\n"
        )
    if prompt_style == "cot-rag":
        if lang == "zh":
            return (
                "你是预算审计专家。请先在内部逐步推理，再仅输出最终 OCI JSON，不要展示推理过程。"
                "回答必须严格依据给定证据，不要编造数值、年份或口径。\n"
                "输出格式："
                "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}\n\n"
                f"问题：{query}\n\n证据：\n{evidence_text}\n"
            )
        return (
            "You are a financial audit expert. Reason step by step internally, then output only the final "
            "OCI JSON without showing the reasoning. Use only the provided evidence and do not fabricate "
            "numbers, years, or reporting scope.\n"
            "Output format: "
            "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}\n\n"
            f"Question: {query}\n\nEvidence:\n{evidence_text}\n"
        )
    if lang == "zh":
        return (
            "你是预算审计专家。请严格依据证据回答问题，输出 OCI 结构（Observation/Cause/Impact）的 JSON。"
            "不要编造数值与口径，不要输出引用。\n"
            "输出格式："
            "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}\n\n"
            f"问题：{query}\n\n证据：\n{evidence_text}\n"
        )
    return (
        "You are a financial audit expert. Answer the question strictly based on the provided evidence and "
        "output an OCI JSON (Observation/Cause/Impact). Do not fabricate numbers or citations.\n"
        "Output format: "
        "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}\n\n"
        f"Question: {query}\n\nEvidence:\n{evidence_text}\n"
    )


def call_chat_api(
    query: str,
    evidence: List[Dict],
    model: str,
    prompt_style: str,
    max_retries: int,
    dataset_tag: str,
) -> Tuple[Dict, str]:
    prompt = build_prompt(query, evidence, prompt_style, dataset_tag)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "top_p": 0.9,
        "top_k": 20,
        "max_tokens": 2048,
        "presence_penalty": 1.0,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        CHAT_BASE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(1, max_retries + 1):
        try:
            with request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            return parse_oci_json(content), content
        except Exception:
            if attempt == max_retries:
                return EMPTY_OCI, ""
            time.sleep(1.5 * attempt)
    return EMPTY_OCI, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline generation methods")
    parser.add_argument("--dataset-tag", default=DEFAULT_DATASET_TAG, help="Dataset tag for default output path")
    parser.add_argument("--model-tag", default=None, help="Model tag for default output path")
    parser.add_argument(
        "--splits-dir",
        default=DEFAULT_AUDIT_SPLITS_DIR,
        help="Splits directory for closed-book mode",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Which split to run when --chains is omitted (e.g. train, dev, test, test-mini, all)",
    )
    parser.add_argument("--chains", default=None, help="Evidence chain JSONL path for RAG-style methods")
    parser.add_argument("--output", default=None, help="Output prediction JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument(
        "--prompt-style",
        default="rag",
        choices=["closed-book", "rag", "cot-rag"],
        help="Baseline prompt style",
    )
    parser.add_argument("--method", default=None, help="Method display name")
    parser.add_argument("--max-retries", type=int, default=3, help="Chat API max retries")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers for generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_tag = normalize_tag(args.model_tag or args.model, "model")
    if args.method:
        method = args.method
    elif args.prompt_style == "closed-book":
        method = "Closed-book"
    elif args.prompt_style == "cot-rag":
        method = "CoT-RAG"
    else:
        method = "RAG"
    output_path = args.output or default_output_path(args.dataset_tag, model_tag, method)
    validate_non_legacy_output_path(output_path, label="generation output")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    run_config = {
        "dataset_tag": args.dataset_tag,
        "model": args.model,
        "method": method,
        "prompt_style": args.prompt_style,
        "chains": args.chains,
        "split": args.split,
    }

    done_ids = set()
    existing_config = None
    if os.path.exists(output_path):
        for item in read_jsonl(output_path):
            if existing_config is None:
                existing_config = item.get("config")
            row_id = item.get("row_id")
            if row_id:
                done_ids.add(row_id)
    if done_ids and existing_config is None:
        raise ValueError("output file exists but has no `config` field; pick a different --output path")
    if existing_config and existing_config != run_config:
        raise ValueError("output file exists with an inconsistent config; pick a different --output path")

    items = []
    if args.chains and args.prompt_style != "closed-book":
        for row in read_jsonl(args.chains):
            row_id = row.get("row_id")
            if row_id in done_ids:
                continue
            evidence = unique_evidence(
                [
                    row.get("evidence", {}).get("O", {}),
                    row.get("evidence", {}).get("C", {}),
                    row.get("evidence", {}).get("I", {}),
                    *(row.get("extra_evidence") or []),
                ]
            )
            items.append(
                {
                    "row_id": row_id,
                    "qid": row.get("qid"),
                    "query": row.get("query", ""),
                    "evidence": evidence,
                }
            )
    else:
        for row in load_queries(args.splits_dir, args.split):
            row_id = row.get("row_id")
            if row_id in done_ids:
                continue
            items.append(
                {
                    "row_id": row_id,
                    "qid": row.get("qid"),
                    "query": row.get("query", ""),
                    "evidence": [],
                }
            )

    def _process(row: Dict) -> Dict:
        evidence = [] if args.prompt_style == "closed-book" else row["evidence"]
        oci_raw, raw_output = call_chat_api(
            query=row["query"],
            evidence=evidence,
            model=args.model,
            prompt_style=args.prompt_style,
            max_retries=args.max_retries,
            dataset_tag=args.dataset_tag,
        )
        prediction_oci = oci_raw if isinstance(oci_raw, dict) else EMPTY_OCI
        return {
            "row_id": row["row_id"],
            "qid": row["qid"],
            "method": method,
            "model": args.model,
            "used_evidence_ids": [item.get("evidence_id") for item in evidence if item.get("evidence_id")],
            "oci_raw": prediction_oci,
            "raw_model_output": raw_output,
            "prediction_O": prediction_oci.get("Observation", []),
            "prediction_C": prediction_oci.get("Cause", []),
            "prediction_I": prediction_oci.get("Impact", []),
            "prediction_oci": prediction_oci,
            "claim_support": {},
            "suppressed_claims": [],
            "abstain_flag": False,
            "abstain_reason": None,
            "local_metrics": {
                "total_claims": sum(
                    len(prediction_oci.get(role_name, []) or [])
                    for role_name in ("Observation", "Cause", "Impact")
                ),
                "supported_claims": 0,
                "numeric_claims": 0,
                "qac_supported_claims": 0,
                "suppressed_claims": 0,
            },
            "config": run_config,
        }

    workers = max(1, args.workers)
    write_lock = threading.Lock()
    with open(output_path, "a", encoding="utf-8") as f:
        if workers <= 1:
            for row in tqdm.tqdm(items, desc=method):
                output_row = _process(row)
                f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                f.flush()
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process, row): row for row in items}
                for future in tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc=method
                ):
                    output_row = future.result()
                    with write_lock:
                        f.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                        f.flush()

    print(f"[generation] done -> {output_path}")


if __name__ == "__main__":
    main()
