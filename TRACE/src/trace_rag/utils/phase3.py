from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List

from trace_rag.retrieval import meta_text_for_risk, normalize_meta_item
from trace_rag.utils.experiment_layout import validate_non_legacy_output_path


def normalize_tag(value: str | None, fallback: str) -> str:
    normalized = (value or "").strip().lower().replace("/", "--")
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-.")
    return normalized or fallback


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: str, payload: Dict) -> None:
    validate_non_legacy_output_path(path, label="json output")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict]) -> None:
    validate_non_legacy_output_path(path, label="jsonl output")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_meta(index_dir: str) -> Dict[str, Dict]:
    meta = {}
    meta_path = os.path.join(index_dir, "meta.jsonl")
    for raw_item in read_jsonl(meta_path):
        item = normalize_meta_item(dict(raw_item))
        evidence_id = item.get("evidence_id")
        if not evidence_id:
            continue
        caliber = item.get("caliber")
        if caliber is None:
            caliber = item.get("budget_caliber")
        meta[evidence_id] = {
            "evidence_id": evidence_id,
            "doc_id": item.get("doc_id"),
            "text": item.get("text") or "",
            "text_with_meta": item.get("text_with_meta") or meta_text_for_risk(item),
            "year": item.get("year"),
            "caliber": caliber,
            "level": item.get("level"),
            "source_path": item.get("source_path"),
        }
    return meta


def load_gold_map(gold_path: str) -> Dict[str, Dict]:
    return {
        str(item.get("row_id")): item
        for item in read_jsonl(gold_path)
        if item.get("row_id")
    }


def unique_evidence(items: List[Dict]) -> List[Dict]:
    seen = set()
    results = []
    for item in items:
        if not item:
            continue
        evidence_id = item.get("evidence_id")
        key = evidence_id or json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        results.append(item)
    return results


def chain_evidence_list(row: Dict) -> List[Dict]:
    evidence = [
        row.get("evidence", {}).get("O", {}),
        row.get("evidence", {}).get("C", {}),
        row.get("evidence", {}).get("I", {}),
    ]
    evidence.extend(row.get("extra_evidence") or [])
    return unique_evidence(evidence)


def stage3_root(dataset_tag: str, model_tag: str) -> str:
    return str(Path("runs") / dataset_tag / model_tag / "stage3")
