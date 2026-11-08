#! /usr/bin/env python3
"""Build the FAISS vector index for a TRACE corpus.

Reads ``datasets/<dataset>/corpus.paragraphs.jsonl`` and writes
``index.faiss``, ``meta.jsonl`` and ``index.info.json`` to the output
directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re as _re
import time
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import request

import numpy as np
import faiss
import tqdm


DEFAULT_BATCH_SIZE = 64
MAX_EMBED_CHARS = 120000
EMBED_SERVER_URL = os.getenv("EMBED_URL", "http://localhost:8081/v1/embeddings")


def _split_server_urls(raw: str | None) -> List[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def _read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _build_dept_map(corpus_file: str) -> Dict[str, str]:
    """Extract organisation names from each document cover page (``pid=1``).

    Returns a mapping ``doc_id -> department_name``.
    """
    dept_map: Dict[str, str] = {}
    _SKIP = {"Unnamed: 0", "---", "NaN", "## 封面", ""}
    for item in _read_jsonl(corpus_file):
        if item.get("pid") != 1:
            continue
        doc_id = item.get("doc_id", "")
        if doc_id in dept_map:
            continue
        dept = _extract_dept_from_cover(item.get("text", ""), _SKIP)
        if not dept:
            dept = _extract_dept_from_path(item.get("source_path", ""))
        if dept:
            dept_map[doc_id] = dept
    return dept_map


def _extract_dept_from_cover(text: str, skip_set: set) -> Optional[str]:
    """Pull the organisation name out of a cover-page Markdown table."""
    for line in text.split("\n"):
        cell = line.strip().strip("|").strip()
        if cell in skip_set:
            continue
        if "预算" in cell or "封面" in cell:
            continue
        if _re.match(r"^\d{4}", cell):
            continue
        if cell:
            return cell
    return None


def _extract_dept_from_path(source_path: str) -> Optional[str]:
    """Fallback: derive an organisation name from ``source_path``."""
    m = _re.search(r"corpus/\d+(.*?)_", source_path)
    if m:
        return m.group(1)
    basename = os.path.basename(source_path)
    name, _ = os.path.splitext(basename)
    if name and name != source_path:
        return name
    return None


def _clean_markdown_noise(text: str) -> str:
    """Strip common Markdown-table noise before feeding text to the encoder."""
    text = _re.sub(r"Unnamed:\s*\d+", "", text)
    text = _re.sub(r"(?<=\|)\s*NaN\s*(?=\|)", " ", text)
    text = _re.sub(r"^\s*NaN\s*$", "", text, flags=_re.MULTILINE)
    text = _re.sub(r"^\|[\s|]*\|$", "", text, flags=_re.MULTILINE)
    text = _re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_text(
    item: Dict,
    add_meta_tokens: bool,
    dept_map: Optional[Dict[str, str]] = None,
    clean_noise: bool = False,
) -> str:
    text = item.get("text") or ""
    if clean_noise:
        text = _clean_markdown_noise(text)
    if not add_meta_tokens:
        return text.strip()

    year = item.get("year")
    caliber = item.get("budget_caliber")
    level = item.get("budget_level")

    meta = []
    if year:
        meta.append(f"[YEAR={year}]")
    if caliber and str(caliber).lower() != "unknown":
        meta.append(f"[CAL={caliber}]")
    if level and str(level).lower() != "unknown":
        meta.append(f"[LEVEL={level}]")

    # Inject organisation-name context (kept in Chinese so that the
    # multilingual encoder can still recognise audit terminology).
    if dept_map:
        doc_id = item.get("doc_id", "")
        dept = dept_map.get(doc_id)
        if dept:
            meta.append(f"[部门：{dept}]")

    meta_prefix = " ".join(meta).strip()
    if not meta_prefix:
        return text.strip()
    return f"{meta_prefix}\n{text.strip()}"


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool, int]:
    original_len = len(text)
    if original_len <= max_chars:
        return text, False, original_len
    return text[:max_chars], True, original_len


def _request_embeddings(server_url: str, batch: List[str]) -> List[List[float]]:
    payload = {"input": batch}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        server_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)
    data_list = data.get("data") or []
    if len(data_list) != len(batch):
        raise RuntimeError(
            f"embedding count mismatch: got {len(data_list)} expected {len(batch)}"
        )
    return [item["embedding"] for item in data_list]


def _is_payload_too_large_error(exc: Exception) -> bool:
    s = str(exc).lower()
    keys = [
        "413",
        "payload too large",
        "request entity too large",
        "maximum context length",
        "context length exceeded",
        "input too long",
    ]
    return any(k in s for k in keys)


def _embed_batch_retry(
    *,
    server_url: str,
    batch: List[str],
    start_idx: int,
    max_retries: int = 4,  # more whole-batch retries before falling back to splitting
) -> List[List[float]]:
    for attempt in range(1, max_retries + 1):
        try:
            return _request_embeddings(server_url, batch)
        except Exception as exc:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Embedding batch failed after {max_retries} retries "
                    f"(url={server_url}, start_idx={start_idx}, size={len(batch)})."
                ) from exc
            time.sleep(1.2 * attempt)
    return []


def _embed_with_split(
    *,
    server_url: str,
    batch: List[str],
    start_idx: int,
    max_retries: int = 4,
    min_split_batch: int = 16,  # floor on sub-batch size to avoid tiny requests
) -> List[List[float]]:
    try:
        return _embed_batch_retry(
            server_url=server_url,
            batch=batch,
            start_idx=start_idx,
            max_retries=max_retries,
        )
    except Exception as exc:
        # Only split on payload-too-large failures; transient network errors
        # should not fragment the batch into many tiny requests.
        if not _is_payload_too_large_error(exc):
            raise

        # Keep each half above the minimum sub-batch size.
        if len(batch) < min_split_batch * 2:
            raise

        mid = len(batch) // 2
        left = _embed_with_split(
            server_url=server_url,
            batch=batch[:mid],
            start_idx=start_idx,
            max_retries=max_retries,
            min_split_batch=min_split_batch,
        )
        right = _embed_with_split(
            server_url=server_url,
            batch=batch[mid:],
            start_idx=start_idx + mid,
            max_retries=max_retries,
            min_split_batch=min_split_batch,
        )
        return left + right


def _embed_task(
    task: Tuple[int, str, List[str], int, int],
) -> Tuple[int, List[List[float]]]:
    start_idx, server_url, batch, max_retries, min_split_batch = task
    vecs = _embed_with_split(
        server_url=server_url,
        batch=batch,
        start_idx=start_idx,
        max_retries=max_retries,
        min_split_batch=min_split_batch,
    )
    return start_idx, vecs


def _embed_texts(
    server_urls: List[str],
    texts: List[str],
    batch_size: int,
    workers: int,
    max_retries: int = 4,
    min_split_batch: int = 16,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    urls = [u.strip() for u in (server_urls or []) if u and u.strip()]
    if not urls:
        raise ValueError("`server_urls` must not be empty")

    tasks: List[Tuple[int, str, List[str], int, int]] = []
    starts = list(range(0, len(texts), batch_size))
    for i, st in enumerate(starts):
        url = urls[i % len(urls)]
        tasks.append(
            (st, url, texts[st : st + batch_size], max_retries, min_split_batch)
        )

    out_by_start: Dict[int, List[List[float]]] = {}

    actual_workers = max(1, min(int(workers), len(tasks)))
    if actual_workers == 1:
        for task in tqdm.tqdm(tasks, desc="Embedding"):
            st, vecs = _embed_task(task)
            out_by_start[st] = vecs
    else:
        # Fan out across multi-core CPUs and multiple embedding endpoints.
        with cf.ProcessPoolExecutor(max_workers=actual_workers) as ex:
            futures = [ex.submit(_embed_task, t) for t in tasks]
            for fut in tqdm.tqdm(cf.as_completed(futures), total=len(futures), desc="Embedding"):
                st, vecs = fut.result()
                out_by_start[st] = vecs

    vecs: List[List[float]] = []
    for st in sorted(out_by_start.keys()):
        vecs.extend(out_by_start[st])

    arr = np.array(vecs, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / norms


def build_index(
    corpus_file: str,
    output_dir: str,
    add_meta_tokens: bool,
    limit: Optional[int],
    batch_size: int,
    server_url: str,
    server_urls: Optional[List[str]] = None,
    workers: int = 1,
    min_split_batch: int = 16,
    clean_noise: bool = False,
    inject_dept: bool = False,
):
    dept_map: Optional[Dict[str, str]] = None
    if inject_dept:
        dept_map = _build_dept_map(corpus_file)
        print(f"[build_index] department map: {len(dept_map)} documents")

    texts: List[str] = []
    meta: List[Dict] = []
    for idx, item in enumerate(tqdm.tqdm(_read_jsonl(corpus_file), desc="Loading corpus")):
        if limit is not None and idx >= limit:
            break
        text_for_embedding = _format_text(
            item, add_meta_tokens, dept_map=dept_map, clean_noise=clean_noise,
        )
        text_for_embedding, truncated, original_len = _truncate_text(
            text_for_embedding, MAX_EMBED_CHARS
        )
        if not text_for_embedding:
            continue
        texts.append(text_for_embedding)

        # Always record the department in metadata, even if not injected into embeddings.
        doc_id = item.get("doc_id", "")
        department = dept_map.get(doc_id, "") if dept_map else ""

        meta.append(
            {
                "evidence_id": item.get("evidence_id"),
                "doc_id": doc_id,
                "pid": item.get("pid"),
                "year": item.get("year"),
                "department": department,
                "budget_caliber": item.get("budget_caliber"),
                "budget_level": item.get("budget_level"),
                "section": item.get("section"),
                "text": item.get("text"),
                "text_for_embedding": text_for_embedding,
                "text_truncated": truncated,
                "text_length": original_len,
                "source_path": item.get("source_path"),
            }
        )

    if not texts:
        raise ValueError("no indexable text found in the corpus")

    urls: List[str] = []
    for item in server_urls or []:
        urls.extend(_split_server_urls(item))
    if not urls:
        urls = _split_server_urls(server_url)
    vectors = _embed_texts(
        server_urls=urls,
        texts=texts,
        batch_size=batch_size,
        workers=workers,
        max_retries=4,
        min_split_batch=min_split_batch,
    )
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

    meta_path = os.path.join(output_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for item in meta:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    info = {
        "corpus_file": corpus_file,
        "output_dir": output_dir,
        "embedding_model": "BAAI/bge-m3",
        "count": len(meta),
        "dim": dim,
        "add_meta_tokens": add_meta_tokens,
        "clean_noise": clean_noise,
        "inject_dept": inject_dept,
        "dept_count": len(dept_map) if dept_map else 0,
        "batch_size": batch_size,
        "max_embed_chars": MAX_EMBED_CHARS,
        "embedding_server_urls": urls,
        "workers": workers,
        "min_split_batch": min_split_batch,
    }
    with open(os.path.join(output_dir, "index.info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print(
        f"[build_index] done: {len(meta)} entries, dim={dim}, output_dir={output_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vector index for Audit-OCIQA corpus")
    parser.add_argument(
        "--corpus-file",
        default="datasets-public/audit-ociqa-demo/corpus.paragraphs.jsonl",
        help="JSONL corpus file path",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/index",
        help="Index output directory",
    )
    parser.add_argument(
        "--add-meta-tokens",
        action="store_true",
        help="Prefix [YEAR]/[CAL]/[LEVEL] tokens to text before embedding",
    )
    parser.add_argument(
        "--no-meta-tokens",
        action="store_true",
        help="Do not prefix metadata tokens before embedding",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of rows for quick test",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--server-url",
        default=EMBED_SERVER_URL,
        help="Local embedding server URL (single endpoint fallback)",
    )
    parser.add_argument(
        "--server-urls",
        default="",
        help="comma-separated list of embedding server URLs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1))),
        help="number of worker processes (>= number of endpoints is recommended)",
    )
    parser.add_argument(
        "--min-split-batch",
        type=int,
        default=16,
        help="minimum sub-batch size when splitting on payload-too-large",
    )
    parser.add_argument(
        "--clean-noise",
        action="store_true",
        help="strip Markdown-table noise (Unnamed/NaN, blank rows) before embedding",
    )
    parser.add_argument(
        "--inject-dept",
        action="store_true",
        help="extract organisation names from cover pages and inject them as a prefix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    add_meta = args.add_meta_tokens and not args.no_meta_tokens
    if not args.add_meta_tokens and not args.no_meta_tokens:
        add_meta = True

    urls = _split_server_urls(args.server_urls)
    if not urls:
        urls = _split_server_urls(args.server_url)

    build_index(
        corpus_file=args.corpus_file,
        output_dir=args.output_dir,
        add_meta_tokens=add_meta,
        limit=args.limit,
        batch_size=args.batch_size,
        server_url=args.server_url,
        server_urls=urls,
        workers=args.workers,
        min_split_batch=args.min_split_batch,
        clean_noise=args.clean_noise,
        inject_dept=args.inject_dept,
    )


if __name__ == "__main__":
    main()

