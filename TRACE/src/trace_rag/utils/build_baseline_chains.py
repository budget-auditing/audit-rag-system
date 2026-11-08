#!/usr/bin/env python3
"""
Convert retrieval top-K results into evidence chain format for baselines.

For each query, takes top-1 retrieval result and assigns the same evidence
to O/C/I roles (baseline: no role conditioning, no coverage optimization).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List

from trace_rag.retrieval import meta_text_for_risk, normalize_meta_item
from trace_rag.utils.experiment_layout import validate_non_legacy_output_path


def read_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_meta(index_dir: str) -> Dict[str, Dict]:
    meta = {}
    for raw_item in read_jsonl(os.path.join(index_dir, "meta.jsonl")):
        item = normalize_meta_item(dict(raw_item))
        eid = item.get("evidence_id")
        if eid:
            meta[eid] = {
                "evidence_id": eid,
                "text": item.get("text", ""),
                "text_with_meta": meta_text_for_risk(item),
                "year": item.get("year"),
                "caliber": item.get("caliber") or item.get("budget_caliber"),
            }
    return meta


def sha1_file(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_stat_signature(path: str) -> Dict | None:
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_run_config(args: argparse.Namespace, retrieval_rows: List[Dict]) -> Dict:
    source_config = retrieval_rows[0].get("config") if retrieval_rows else None
    return {
        "retrieval": os.path.abspath(args.retrieval),
        "retrieval_sha1": sha1_file(args.retrieval),
        "retrieval_config": source_config,
        "index_dir": os.path.abspath(args.index_dir),
        "index_signature": file_stat_signature(os.path.join(args.index_dir, "meta.jsonl")),
        "method": args.method,
        "top_n": args.top_n,
    }


def main():
    parser = argparse.ArgumentParser(description="Build baseline evidence chains from retrieval")
    parser.add_argument("--retrieval", required=True, help="Retrieval JSONL path")
    parser.add_argument("--index-dir", default="runs/index", help="Index directory")
    parser.add_argument("--output", required=True, help="Output chains JSONL path")
    parser.add_argument("--method", default="baseline-top1", help="Method name")
    parser.add_argument("--top-n", type=int, default=1, help="Use top-N evidence (1=same for O/C/I)")
    parser.add_argument("--flat-topk", type=int, default=0,
                        help="If >0, bypass OCI structure and put all top-K as extra_evidence")
    args = parser.parse_args()

    validate_non_legacy_output_path(args.output, label="baseline evidence output")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    meta = load_meta(args.index_dir)
    retrieval = read_jsonl(args.retrieval)
    run_config = build_run_config(args, retrieval)

    empty_evidence = {
        "evidence_id": None,
        "text": "",
        "text_with_meta": "",
        "year": None,
        "caliber": None,
    }

    done_ids = set()
    existing_config = None
    if os.path.exists(args.output):
        for item in read_jsonl(args.output):
            if existing_config is None:
                existing_config = item.get("config")
            done_ids.add(item.get("row_id"))
    if done_ids and existing_config is None:
        raise ValueError(
            "output file exists but has no `config` field; pick a different --output path"
        )
    if existing_config and existing_config != run_config:
        raise ValueError(
            "output file exists with an inconsistent config; pick a different --output path"
        )

    with open(args.output, "a", encoding="utf-8") as f:
        for item in retrieval:
            rid = item.get("row_id")
            if rid in done_ids:
                continue
            top_ids = item.get("topk_evidence_ids", [])

            if args.flat_topk > 0:
                all_ev = []
                seen = set()
                for eid in top_ids[:args.flat_topk]:
                    if eid and eid not in seen:
                        seen.add(eid)
                        all_ev.append(meta.get(eid, empty_evidence))
                o_ev = all_ev[0] if len(all_ev) > 0 else empty_evidence
                c_ev = all_ev[1] if len(all_ev) > 1 else empty_evidence
                i_ev = all_ev[2] if len(all_ev) > 2 else empty_evidence
                extra = all_ev[3:] if len(all_ev) > 3 else []
            elif args.top_n == 1:
                eid = top_ids[0] if top_ids else None
                ev = meta.get(eid, empty_evidence) if eid else empty_evidence
                o_ev = c_ev = i_ev = ev
                extra = []
            else:
                o_ev = meta.get(top_ids[0], empty_evidence) if len(top_ids) > 0 else empty_evidence
                c_ev = meta.get(top_ids[1], empty_evidence) if len(top_ids) > 1 else o_ev
                i_ev = meta.get(top_ids[2], empty_evidence) if len(top_ids) > 2 else o_ev
                extra = []

            chain = {
                "row_id": rid,
                "qid": item.get("qid"),
                "query": item.get("query", ""),
                "method": args.method,
                "evidence": {"O": o_ev, "C": c_ev, "I": i_ev},
                "score": 0.0,
                "coverage": 0.0,
                "link": 0.0,
                "config": run_config,
            }
            if extra:
                chain["extra_evidence"] = extra
            f.write(json.dumps(chain, ensure_ascii=False) + "\n")

    total = len(retrieval) - len(done_ids)
    print(f"Built {total} chains -> {args.output}")


if __name__ == "__main__":
    main()
