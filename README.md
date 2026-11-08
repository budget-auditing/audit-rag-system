# Audit RAG Platform

A question-answering platform for government budget auditing that routes each
question to one of three retrieval-augmented generation methods:

| Mode (UI) | Method | Purpose | Code |
|---|---|---|---|
| 信息查询 (fact lookup) | MSG-RAG | Facts and figures, via a multi-layer semantic graph | `backend/app/msg/` |
| 结论核验 (conclusion check) | C³-RAG | Caliber comparability and causal completeness | [`C3RAG/`](C3RAG/) |
| 风险识别 (risk discovery) | TRACE | Latent risks with Observation–Cause–Impact evidence chains | [`TRACE/`](TRACE/) |

The user picks the mode in the UI; a rule-based hint suggests one.

## Repository layout

```
.
├── backend/    FastAPI app: document ingestion, MSG-RAG, routing, export (port 8001)
├── frontend/   Vue 3 + Element Plus web UI (port 5173)
├── C3RAG/      C³-RAG method and its HTTP service (port 8101)
├── TRACE/      TRACE method, experiments, embedding server, synthetic demo data,
│               HTTP service (port 8102); method notes in TRACE/docs/
├── scripts/    start_all.sh, smoke_test.sh, demo_pipeline.py
└── data/       Runtime data (indexes, database); not tracked
```

## Data

The Audit-OCIQA benchmark used in the papers is **not** publicly released.
`TRACE/datasets-public/audit-ociqa-demo/` contains a fully synthetic dataset
with the same schema, and `backend/tests/fixtures/` contains synthetic budget
documents for the end-to-end demo. See
[`TRACE/docs/datasets.md`](TRACE/docs/datasets.md).

## Prerequisites

- Python 3.11+ (the backend's pinned dependencies need it)
- Node.js 18+ and pnpm
- An OpenAI-compatible chat endpoint at `http://localhost:8000/v1`
  (e.g. a local vLLM server, or any hosted API)
- The BGE-M3 embedding server from `TRACE/tools/embed_server.py` on port 8081.
  It serves both dense vectors (`/v1/embeddings`) and ColBERT token vectors
  (`/v1/colbert`); TRACE needs the latter, so a generic embedding API is not
  enough.

## Setup

Each Python component has its own virtual environment, which
`scripts/start_all.sh` expects at `<component>/.venv`:

```bash
python -m venv backend/.venv && backend/.venv/bin/pip install -r backend/requirements.txt
python -m venv C3RAG/.venv   && C3RAG/.venv/bin/pip install -r C3RAG/requirements.txt
python -m venv TRACE/.venv   && TRACE/.venv/bin/pip install -e './TRACE[service,embed-server]'

cp backend/.env.example backend/.env    # set CHAT_MODEL and endpoints here
cd frontend && pnpm install && cd ..
```

C³-RAG's conclusion check needs a trained CCES checkpoint at
`C3RAG/models/cces_model.pt`, which is not distributed; see *Training C³-RAG*
below.

## Running

```bash
scripts/start_all.sh            # embedding server, C³-RAG, TRACE, backend; logs in logs/
scripts/smoke_test.sh           # health checks
cd frontend && pnpm dev         # http://localhost:5173
```

`start_all.sh` reads `CHAT_MODEL` from `backend/.env` and passes the endpoints
to each service. With all services up, `python scripts/demo_pipeline.py`
uploads the synthetic fixtures, builds every index, and runs one query per
mode.

| Service | Port |
|---|---|
| Chat LLM (external) | 8000 |
| Backend API | 8001 |
| Embedding server | 8081 |
| C³-RAG service | 8101 |
| TRACE service | 8102 |
| Frontend (dev) | 5173 |

The embedding server reads `EMBED_MODEL`, `EMBED_DEVICE` (e.g. `cuda:0`;
unset = automatic), `EMBED_FP16` (`0` to disable), `HOST`, and `PORT`. FAISS
search and TRACE preference training use all visible GPUs; restrict them with
`CUDA_VISIBLE_DEVICES`.

## TRACE

Command-line tools (installed with the `TRACE` package) default to the
synthetic demo data and write under `TRACE/runs/audit-ociqa/<model-tag>/`.
Run them from `TRACE/`:

```bash
bash scripts/build_index.sh          # FAISS index over the demo corpus (runs/index/)
trace-retrieve                       # risk-aware retrieval
trace-evidence                       # OCI evidence chains
trace-generate --model <chat-model>  # OCI generation with claim verification (needs CHAT_URL)
```

Full experiment runs:

```bash
bash scripts/run_main_experiments.sh   # Audit-OCIQA pipeline + evaluation, on the demo by default
bash scripts/run_finqa.sh              # FinQA; see docs/datasets.md for data preparation
```

Results on the demo are **not** comparable to the paper. The hyper-parameters
of the main experiments are the constants at the top of
`TRACE/experiments/run_main.py`; each CLI lists its defaults under `--help`.
Optional extras: `train` (preference learning, `python -m trace_rag.train_pref`;
needs a dataset with a train split), `data-build` (corpus construction from raw
documents), `dev` (tests and lint).

Further reading: [`TRACE/docs/method.md`](TRACE/docs/method.md) (method
summary), [`TRACE/docs/datasets.md`](TRACE/docs/datasets.md) (schemas),
[`TRACE/case_studies.md`](TRACE/case_studies.md) (illustrative cases).

## Training C³-RAG

Run from `C3RAG/` with its environment active. The scripts talk to the same
OpenAI-compatible endpoints as the service:

| Variable | Default | Purpose |
|---|---|---|
| `EMBED_URL` | `http://localhost:8081/v1` | Embedding base URL |
| `CHAT_URL` | `http://localhost:8000/v1` | Chat base URL |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat model name |
| `OPENAI_API_KEY` | `sk-placeholder` | API key for both endpoints |
| `ENCODER_NAME` | `nghuyong/ernie-3.0-base-zh` | CCES encoder; must match the one used for training |

1. `python build_index.py` indexes documents under `data/train_data/`
   (Excel files under `data/train_data/files/`) into `data/index/`.
2. Put contrastive training data at `data/train_data/cces_training_data.jsonl`
   and `data/train_data/cces_validation_data.jsonl`. Each line holds a
   `query`, its `positive_evidence`, and `negative_evidences` of four types
   (caliber shift, causal gap, polarity reversal, random). The generation
   script used in the paper is not included.
3. `python train.py` trains CCES (InfoNCE loss, `nghuyong/ernie-3.0-base-zh`
   encoder) and writes `models/cces_model.pt`.

## Tests

```bash
cd TRACE && .venv/bin/pip install -e '.[dev]' && .venv/bin/python -m pytest -q && .venv/bin/ruff check .
```

The TRACE tests run offline. `backend/tests/test_e2e_smoke.py` is an
end-to-end test: it needs the chat, embedding, C³-RAG, and TRACE services from
`scripts/start_all.sh` to be running.

```bash
cd backend && .venv/bin/python -m pytest -q
```

## License

MIT; see [`LICENSE`](LICENSE). `C3RAG/` and `TRACE/` carry their own MIT
license files.
