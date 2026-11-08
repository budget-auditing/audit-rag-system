#!/usr/bin/env python3
"""
Convert FinQA dataset into TRACE pipeline format.

Produces:
  - corpus.paragraphs.jsonl   (one evidence paragraph per line)
  - {train,dev,test}.jsonl     (query splits)
  - gold.qrels.jsonl           (gold evidence labels)
  - qa_gold.jsonl              (gold answers for numerical evaluation)

Usage:
    python -m trace_rag.utils.finqa_convert \
        --finqa-dir FinQA/dataset \
        --output-dir datasets/finqa
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple


def linearize_table_row(header: List[str], row: List[str]) -> str:
    """Linearize a table row using the header as context.

    Follows the FinQA convention: "the {row_label} of {col} is {value} ;".
    """
    if not row:
        return ""
    row_label = row[0].strip()
    parts = []
    for col_idx in range(1, min(len(row), len(header))):
        col_name = header[col_idx].strip()
        cell = row[col_idx].strip()
        if cell and cell != "-" and cell != "—":
            parts.append(f"the {row_label} of {col_name} is {cell}")
    if not parts:
        return row_label
    return " ; ".join(parts) + " ;"


def extract_provenance(filename: str) -> Tuple[str, str]:
    """Extract company ticker and fiscal year from FinQA filename.

    Example: 'ETR/2016/page_23.pdf' -> ('ETR', '2016')
    """
    parts = filename.split("/")
    ticker = parts[0] if parts else ""
    year = parts[1] if len(parts) >= 2 and len(parts[1]) == 4 and parts[1].isdigit() else ""
    return ticker, year


def process_sample(
    sample: Dict, split: str
) -> Tuple[List[Dict], Dict, Dict, Dict]:
    """Process one FinQA sample.

    Returns:
        corpus_items: list of evidence paragraphs
        query: query dict for split file
        qrel: qrel dict with gold evidence ids
        gold_answer: gold answer dict for evaluation
    """
    sample_id = sample["id"]
    filename = sample.get("filename", "")
    ticker, year = extract_provenance(filename)
    provenance = f"{ticker}, {year} Annual Report. " if ticker and year else ""

    pre_text = sample.get("pre_text") or []
    post_text = sample.get("post_text") or []
    table = sample.get("table") or []
    qa = sample.get("qa", {})

    corpus_items: List[Dict] = []

    # Text paragraphs from pre_text
    for idx, text in enumerate(pre_text):
        text = text.strip()
        if not text:
            continue
        evidence_id = f"{filename}::text_{idx}"
        corpus_items.append({
            "evidence_id": evidence_id,
            "doc_id": filename,
            "pid": f"text_{idx}",
            "text": f"{provenance}{text}",
            "section": "pre_text",
            "source_path": filename,
        })

    # Table rows (linearized)
    # FinQA gold_inds can reference table_0 when the first row is data
    # (not a header). We include all rows, using row[0] as first column
    # even for the header row.
    header = table[0] if table else []
    for idx, row in enumerate(table):
        if idx == 0:
            # The first row is usually the header; linearize it as-is
            # so gold_inds referencing table_0 can find it.
            linearized = " | ".join(cell.strip() for cell in row if cell.strip())
        else:
            linearized = linearize_table_row(header, row)
        if not linearized:
            continue
        evidence_id = f"{filename}::table_{idx}"
        corpus_items.append({
            "evidence_id": evidence_id,
            "doc_id": filename,
            "pid": f"table_{idx}",
            "text": f"{provenance}{linearized}",
            "section": "table",
            "source_path": filename,
        })

    # Text paragraphs from post_text
    for idx, text in enumerate(post_text):
        text = text.strip()
        if not text:
            continue
        text_idx = len(pre_text) + idx
        evidence_id = f"{filename}::text_{text_idx}"
        corpus_items.append({
            "evidence_id": evidence_id,
            "doc_id": filename,
            "pid": f"text_{text_idx}",
            "text": f"{provenance}{text}",
            "section": "post_text",
            "source_path": filename,
        })

    # Query (augmented with document provenance for retrieval alignment)
    question = qa.get("question", "")
    augmented_q = (
        f"In {ticker}'s {year} annual report, {question}"
        if ticker and year else question
    )
    query = {
        "row_id": sample_id,
        "qid": sample_id,
        "query": augmented_q,
        "original_query": question,
        "company_ticker": ticker,
        "year": year,
        "split": split,
    }

    # QRels: map FinQA gold_inds to evidence_ids
    gold_inds = qa.get("gold_inds", {})
    gold_evidence_ids = []
    for key in gold_inds:
        evidence_id = f"{filename}::{key}"
        gold_evidence_ids.append(evidence_id)

    qrel = {
        "row_id": sample_id,
        "gold_evidence_ids": gold_evidence_ids,
    }

    # Gold answer for numerical evaluation
    gold_answer = {
        "row_id": sample_id,
        "question": question,
        "answer": qa.get("answer", ""),
        "exe_ans": qa.get("exe_ans"),
        "program": qa.get("program", ""),
        "program_re": qa.get("program_re", ""),
        "gold_inds": gold_inds,
    }

    return corpus_items, query, qrel, gold_answer


def convert_split(
    data: List[Dict], split: str
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    all_corpus = []
    all_queries = []
    all_qrels = []
    all_gold = []

    for sample in data:
        corpus_items, query, qrel, gold_answer = process_sample(sample, split)
        all_corpus.extend(corpus_items)
        all_queries.append(query)
        all_qrels.append(qrel)
        all_gold.append(gold_answer)

    return all_corpus, all_queries, all_qrels, all_gold


def write_jsonl(path: str, items: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  {path}: {len(items)} items")


def main():
    parser = argparse.ArgumentParser(description="Convert FinQA to TRACE format")
    parser.add_argument(
        "--finqa-dir",
        default="FinQA/dataset",
        help="FinQA dataset directory containing train/dev/test.json",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/finqa",
        help="Output directory for converted data",
    )
    args = parser.parse_args()

    splits_dir = os.path.join(args.output_dir, "splits")
    os.makedirs(splits_dir, exist_ok=True)

    all_corpus: List[Dict] = []
    all_qrels: List[Dict] = []
    all_gold: List[Dict] = []
    split_stats = {}

    for split in ["train", "dev", "test"]:
        path = os.path.join(args.finqa_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"[SKIP] {path} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Processing {split}: {len(data)} samples")

        corpus, queries, qrels, gold = convert_split(data, split)
        all_corpus.extend(corpus)
        all_qrels.extend(qrels)
        all_gold.extend(gold)

        write_jsonl(os.path.join(splits_dir, f"{split}.jsonl"), queries)
        split_stats[split] = {"samples": len(data), "evidence": len(corpus)}

    # Deduplicate corpus (same doc may appear in multiple splits — unlikely
    # for FinQA but safe to handle)
    seen_ids = set()
    unique_corpus = []
    for item in all_corpus:
        eid = item["evidence_id"]
        if eid in seen_ids:
            continue
        seen_ids.add(eid)
        unique_corpus.append(item)

    write_jsonl(
        os.path.join(args.output_dir, "corpus.paragraphs.jsonl"), unique_corpus
    )
    write_jsonl(os.path.join(args.output_dir, "gold.qrels.jsonl"), all_qrels)
    write_jsonl(os.path.join(args.output_dir, "qa_gold.jsonl"), all_gold)

    # Write splits meta (needed by load_queries)
    meta = {
        "source": {
            "dataset": "FinQA",
            "qa_file": None,
        },
        "splits": split_stats,
    }
    write_jsonl(os.path.join(splits_dir, "meta.jsonl"), [meta])

    # Summary
    print("\nConversion complete:")
    print(f"  Corpus: {len(unique_corpus)} evidence paragraphs")
    print(f"  Qrels: {len(all_qrels)} queries with gold evidence")
    for split, stats in split_stats.items():
        print(f"  {split}: {stats['samples']} queries, {stats['evidence']} evidence items")


if __name__ == "__main__":
    main()
