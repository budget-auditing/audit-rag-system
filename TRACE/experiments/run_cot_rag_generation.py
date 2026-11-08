#! /usr/bin/env python3
"""
Run CoT-RAG generation with a dataset-specific prompt.
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
from trace_rag.utils.experiment_layout import validate_non_legacy_output_path


DEFAULT_DATASET_TAG = "audit-ociqa"
DEFAULT_METHOD_LABEL = "CoT-RAG"
EMPTY_OCI = {"Observation": [], "Cause": [], "Impact": []}


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def default_output_path(dataset_tag: str, model_tag: str, method: str) -> str:
    filename = f"{normalize_tag(method, 'cot-rag')}.pred.jsonl"
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


def build_prompt(query: str, evidence: List[Dict], dataset_tag: str) -> str:
    evidence_blocks = []
    for idx, item in enumerate(evidence, start=1):
        evidence_id = item.get("evidence_id") or f"doc-{idx}"
        evidence_blocks.append(f"Document {idx} [{evidence_id}]\n{item.get('text') or ''}")
    evidence_text = "\n\n".join(evidence_blocks).strip()
    lang = detect_lang(query + evidence_text[:500])
    if dataset_tag == "finqa":
        from trace_rag.utils.finqa_dsl import format_dsl_prompt_block
        dsl_block = format_dsl_prompt_block()
        return (
            f"{dsl_block}\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Q: {query}\nProgram:"
        )
    if lang == "zh":
        return (
            "你是预算审计专家。请先基于检索到的证据进行逐步分析，再给出最终结论。\n"
            "分析时请围绕以下三点组织：1) 关键现象；2) 潜在成因；3) 可能影响。\n"
            "请严格依据证据，不要编造数值、年份、单位或统计口径。\n"
            "输出要求：\n"
            "1) 先给出简短的推理分析；\n"
            "2) 最后单独输出一个 JSON 对象，结构必须是 "
            "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]};\n"
            "3) JSON 中不要包含引用字段，且除了该 JSON 外不要再输出额外的 JSON。\n\n"
            f"问题：{query}\n\n证据：\n{evidence_text}\n"
        )
    return (
        "You are a financial audit expert. First reason over the retrieved evidence step by step, "
        "then produce the final answer.\n"
        "Organize the reasoning around: 1) key observations, 2) likely causes, and 3) possible impacts.\n"
        "Use only the provided evidence. Do not fabricate numbers, years, units, or reporting scope.\n"
        "Output requirements:\n"
        "1) Provide a short reasoning section first.\n"
        "2) End with exactly one JSON object in the format "
        "{\"Observation\":[{\"claim\":...}],\"Cause\":[{\"claim\":...}],\"Impact\":[{\"claim\":...}]}\n"
        "3) Do not include citations inside the JSON.\n\n"
        f"Question: {query}\n\nEvidence:\n{evidence_text}\n"
    )


def call_chat_api(
    query: str,
    evidence: List[Dict],
    model: str,
    max_retries: int,
    dataset_tag: str,
) -> Tuple[Dict, str]:
    prompt = build_prompt(query, evidence, dataset_tag)
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
    parser = argparse.ArgumentParser(description="Run CoT-RAG generation with RAG-driven CoT prompting")
    parser.add_argument("--dataset-tag", default=DEFAULT_DATASET_TAG, help="Dataset tag for default output path")
    parser.add_argument("--model-tag", default=None, help="Model tag for default output path")
    parser.add_argument("--chains", required=True, help="Evidence chain JSONL path")
    parser.add_argument("--output", default=None, help="Output prediction JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument("--method", default=DEFAULT_METHOD_LABEL, help="Method display name")
    parser.add_argument("--max-retries", type=int, default=3, help="Chat API max retries")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent workers for generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_tag = normalize_tag(args.model_tag or args.model, "model")
    output_path = args.output or default_output_path(args.dataset_tag, model_tag, args.method)
    validate_non_legacy_output_path(output_path, label="CoT-RAG generation output")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    run_config = {
        "dataset_tag": args.dataset_tag,
        "model": args.model,
        "method": args.method,
        "chains": args.chains,
        "prompt_style": "rag-driven-cot",
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

    def _process(row: Dict) -> Dict:
        oci_raw, raw_output = call_chat_api(
            query=row["query"],
            evidence=row["evidence"],
            model=args.model,
            max_retries=args.max_retries,
            dataset_tag=args.dataset_tag,
        )
        prediction_oci = oci_raw if isinstance(oci_raw, dict) else EMPTY_OCI
        return {
            "row_id": row["row_id"],
            "qid": row["qid"],
            "method": args.method,
            "model": args.model,
            "used_evidence_ids": [
                item.get("evidence_id") for item in row["evidence"] if item.get("evidence_id")
            ],
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
    with open(output_path, "a", encoding="utf-8") as f:
        if workers <= 1:
            for row in tqdm.tqdm(items, desc=args.method):
                f.write(json.dumps(_process(row), ensure_ascii=False) + "\n")
                f.flush()
        else:
            write_lock = threading.Lock()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_process, row): row for row in items}
                for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc=args.method):
                    with write_lock:
                        f.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                        f.flush()

    print(f"[generation] done -> {output_path}")


if __name__ == "__main__":
    main()
