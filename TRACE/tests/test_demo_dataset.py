import json
from pathlib import Path

from experiments.evaluate import retrieval as retrieval_eval
from tools.make_demo_dataset import DEFAULT_OUTPUT_DIR, build

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / DEFAULT_OUTPUT_DIR
SHIPPED_FILES = {
    "corpus": "corpus.paragraphs.jsonl",
    "qrels": "gold.qrels.jsonl",
    "gold": "qa.gold.jsonl",
    "queries": "splits/test.jsonl",
}


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_shipped_demo_matches_generator():
    data = build(seed=2026)
    for key, name in SHIPPED_FILES.items():
        assert _read_jsonl(DEMO_DIR / name) == data[key], name


def test_gold_citations_resolve_to_corpus():
    corpus_ids = {p["evidence_id"] for p in _read_jsonl(DEMO_DIR / SHIPPED_FILES["corpus"])}
    for row in _read_jsonl(DEMO_DIR / SHIPPED_FILES["gold"]):
        cited = {eid for claims in row["gold_oci"].values() for c in claims for eid in c["cites"]}
        assert cited == set(row["gold_evidence_ids"])
        assert cited <= corpus_ids


def test_oracle_retrieval_scores_perfectly(tmp_path):
    split_ids = retrieval_eval._load_split_ids(str(ROOT / retrieval_eval.DEFAULT_SPLITS_DIR), "test")
    qrels = retrieval_eval._load_qrels(str(ROOT / retrieval_eval.DEFAULT_QRELS), split_ids)
    role_gold = retrieval_eval._load_role_evidence_map(str(ROOT / retrieval_eval.DEFAULT_GOLD), split_ids)
    assert qrels and set(qrels) == split_ids

    results = tmp_path / "oracle.jsonl"
    results.write_text(
        "".join(
            json.dumps({"row_id": row_id, "topk_evidence_ids": sorted(gold)}) + "\n"
            for row_id, gold in qrels.items()
        ),
        encoding="utf-8",
    )
    metrics = retrieval_eval.evaluate_results(str(results), qrels, role_gold, [5])
    for name in ("Hit@5", "Recall@5", "nDCG@5", "RR@5"):
        assert metrics[name] == 1.0, name
