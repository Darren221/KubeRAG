# KubeRAG

Hybrid-search Retrieval-Augmented Generation over the Kubernetes documentation. Dense (ChromaDB) and sparse (BM25) retrieval are fused via Reciprocal Rank Fusion, reranked by an LLM judge, and answered with cited grounding plus per-citation verification.

> **TODO:** Add a demo video link here once recorded (target: under 4 minutes).

## Results

Evaluated against a hand-written 50-question golden set covering four question types (lookup, multi-hop, unanswerable, ambiguous). Same eval suite, same corpus, two chunking strategies:

| Metric              | Fixed chunking | Recursive chunking | Δ |
|---------------------|---------------:|-------------------:|------:|
| Answer correctness  | TBD            | TBD                | TBD   |
| Faithfulness        | TBD            | TBD                | TBD   |
| Retrieval recall@10 | TBD            | TBD                | TBD   |
| Citation accuracy   | TBD            | TBD                | TBD   |

> **TODO:** Replace TBD cells with the values from `eval/results/chunking_compare_*.md` after running `python -m kuberag.eval.compare_chunking`.

All four metrics are LLM-as-judge or set-overlap, computed by a separate, cheaper model (`gpt-4o-mini`) than the answering model (`gpt-4o`).

## Capabilities

**Hybrid retrieval.** Dense (semantic) and sparse (keyword) retrieval run in parallel and are fused via Reciprocal Rank Fusion. The combination catches both paraphrased queries and exact-term matches — flag names, function signatures, error codes — that dense retrieval alone misses.

**LLM-as-judge reranker.** A single API call ranks the top-20 fused candidates by relevance and returns the top-5. Single-call rather than per-candidate keeps the reranker within the latency budget.

**Per-citation verification.** Each `[n]` marker in a generated answer is checked against its cited chunk by a secondary LLM. The aggregate supported / unsupported ratio drives the citation-accuracy metric.

**Composite confidence scoring.** Each answer carries a 0–1 composite score derived from three signals: retrieval confidence, citation coverage, and answer completeness. The per-dimension breakdown is preserved for the dashboard.

**Threshold-based refusal.** When composite confidence falls below a configurable threshold, the system returns a structured response listing the documents it searched, rather than fabricating an answer.

**Four-metric evaluation framework.** Answer correctness, faithfulness, retrieval recall@k, and citation accuracy. Metrics are orthogonal — a system can score well on three and badly on the fourth, with each failure mode implicating a different subsystem. The chunking comparison in the results table above is the controlled experiment that motivates the recursive-over-fixed chunker choice.

## Quickstart

```bash
# 1. Add your OpenAI API key
cp .env.example .env
echo "OPENAI_API_KEY=sk-..." >> .env

# 2. Boot the stack. First-boot seeds ~28 curated Kubernetes docs (~2 min, ~$0.03).
docker compose up -d

# 3. Open the dashboard
open http://localhost:8501

# 4. Or hit the API directly
curl -X POST http://localhost:8000/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How does a readiness probe affect Service endpoints?"}'
```

Interactive API docs live at <http://localhost:8000/docs>.

## Architecture

The system splits into eight layers, each independently testable:

```text
              ┌─────────────────────────────────────────────────┐
              │   Streamlit dashboard  /  HTTP client / curl    │
              └────────────────────┬────────────────────────────┘
                                   │ HTTP
              ┌────────────────────▼────────────────────────────┐
              │   FastAPI service (/v1/ask, /v1/health, ...)    │
              └────────────────────┬────────────────────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
       ▼                           ▼                           ▼
┌──────────────┐         ┌────────────────────┐      ┌─────────────────┐
│   Retrieval  │         │     Generation     │      │      Eval       │
│              │         │                    │      │                 │
│  Dense  ─┐   │         │   Grounded prompt  │      │  Golden set     │
│  Sparse ─┼─► RRF ─► Rerank ─► gpt-4o ─► Cited      │  Correctness    │
│          │   │         │      answer         │      │  Faithfulness   │
│          │   │         │         │           │      │  Recall@k       │
│          │   │         │         ▼           │      │  Citation acc.  │
│          │   │         │   Citation verify   │      │                 │
│          │   │         │   Confidence score  │      │  Chunking       │
│          │   │         │   "I don't know"    │      │   comparator    │
└──────────┴───┘         └────────────────────┘      └─────────────────┘
       ▲
       │
┌──────┴──────────────────────────────────────────────────────────────┐
│  Indexes (persisted to a shared volume)                              │
│    ChromaDB (dense, cosine)    BM25 (sparse, keyword)                │
└──────────────────────────────────────────────────────────────────────┘
       ▲
       │  ingest
┌──────┴──────────────────────────────────────────────────────────────┐
│  Pipeline: load → chunk (fixed | recursive) → embed → dedupe        │
│  → atomic write to both stores → integrity check                    │
└─────────────────────────────────────────────────────────────────────┘
```

Each layer holds one responsibility. The integrity check at the bottom of the ingest pipeline — `chroma.count() == bm25.count()` — is the single line that keeps hybrid retrieval correct over time.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem standard for AI work |
| API | FastAPI | Async-native, OpenAPI by construction, dependency injection |
| Dashboard | Streamlit | Pure-Python, fastest path to a demo surface |
| Embeddings | OpenAI `text-embedding-3-small` | Cost-effective, high quality, pinned for reproducibility |
| Vector store | ChromaDB | File-backed, no separate service for local development |
| Sparse search | `rank_bm25` (BM25Okapi) | Still the strongest sparse baseline; no neural net needed |
| Generation | OpenAI `gpt-4o` | Strong grounding and citation behavior |
| Judge / reranker / verifier | OpenAI `gpt-4o-mini` | ~30× cheaper than gpt-4o; sufficient for narrow yes/no judgments |
| Chunking | LangChain text splitters | For the recursive structure-aware splitter |
| Validation | Pydantic v2 | Settings, request/response models, structured LLM outputs |
| Dependency manager | `uv` | Fast, deterministic lockfile |
| Container | Multi-stage Docker | 608 MB runtime image; non-root user |

All model IDs are pinned (no `latest` aliases) so eval numbers stay comparable across runs.

## Detailed usage

### Ingest your own corpus

```bash
# Index a local directory
docker exec kuberag-api python -m kuberag.ingest --path /data/raw/my-docs

# Or pull from the kubernetes/website repo (full version, not the seed subset)
docker exec kuberag-api python -m kuberag.ingest --source k8s
```

Supported extensions: `.md`, `.markdown`, `.html`, `.htm`, `.txt`, `.pdf`.

The pipeline:

1. Loads each file into a uniform `Document` (HTML strips scripts/styles, PDFs extract text via `pypdf`).
2. Chunks via the chosen strategy (`fixed` for baseline, `recursive` for markdown-aware).
3. Embeds with content-hash caching — re-ingest of unchanged docs costs nothing.
4. Deduplicates by cosine-similarity threshold.
5. Writes to ChromaDB *and* BM25 atomically; fails loudly if the indexes drift.

### Run the eval suite

```bash
# Smoke test against 5 questions (~$0.09, ~30 sec)
docker exec kuberag-api python -m kuberag.eval.run --limit 5

# Full 50-question eval (~$0.87, ~3-5 min)
docker exec kuberag-api python -m kuberag.eval.run

# Compare chunking strategies (requires both corpora pre-indexed)
docker exec kuberag-api python -m kuberag.eval.compare_chunking
```

Reports land in `eval/results/{timestamp}.{json,md}` and `eval/results/chunking_compare_{timestamp}.{md}`. The Markdown reports are designed for skim → drill: headline scores at the top, by-type breakdown, per-entry detail.

The eval cache is keyed by `(question, corpus_version, config_hash)` — re-running with the same inputs is free, only changes hit the API.

### Dashboard

```bash
open http://localhost:8501
```

Features:

- Question input with `k` and `top_n` sliders.
- Cited answer with clickable `[n]` badges that scroll to the corresponding retrieved chunk.
- Confidence breakdown bars (retrieval / citation / completeness / composite).
- Per-chunk provenance line: final rank, RRF score, original dense rank, original sparse rank. Visually demonstrates when sparse retrieval surfaced a chunk dense missed.
- **Side-by-side hybrid vs. dense-only toggle.** Asks the same question with both retrieval modes and renders the answers + chunks panels side by side. This is the easiest way to see hybrid retrieval working on a rare-term question.

### Local development without Docker

```bash
uv sync --extra dev --extra dashboard
cp .env.example .env  # then add OPENAI_API_KEY

# API
uv run uvicorn kuberag.api:create_app --factory --reload

# Dashboard
uv run streamlit run dashboard/app.py
```

Run quality gates:

```bash
make check   # ruff + mypy strict + pytest (skips network/eval markers)
```

## Project structure

```text
src/kuberag/
├── api.py              FastAPI app factory + lifespan + routes
├── api_models.py       Request models
├── config.py           Pydantic Settings (env-driven)
├── ingest/             Loaders, chunkers, embedder, dedupe, pipeline, CLI
├── retrieval/          Dense, sparse, RRF fusion, LLM reranker, orchestrator
├── generation/         Prompts, generator, citation parser+verifier,
│                       confidence scorer, refusal orchestrator
├── stores/             ChromaDB + BM25 wrappers, shared Hit model
└── eval/               Golden loader, four metrics, cache, runner, comparator

dashboard/
├── app.py              Streamlit script
├── api_client.py       Typed httpx client over /v1/ask
└── components.py       Citation linkifier, chunks panel, confidence bars

scripts/
└── seed.py             First-boot index population (~28 curated docs)

docker/
├── Dockerfile             API image (multi-stage)
├── Dockerfile.dashboard   Streamlit image
└── docker-compose.yml     api + dashboard + seed services

tests/
├── unit/         Pure-logic tests (sub-second, every commit)
└── integration/  Filesystem / mocked-LLM tests (~10s, every commit)
```

> **TODO:** Add screenshots — at minimum: (1) dashboard with cited answer, (2) side-by-side hybrid-vs-dense view, (3) confidence breakdown bars. Save under `docs/screenshots/` and link here.

## Configuration reference

All settings come from environment variables (or a `.env` file at the repo root). Key knobs:

| Variable | Default | What it controls |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI key |
| `KUBERAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `KUBERAG_GENERATION_MODEL` | `gpt-4o` | Answering model |
| `KUBERAG_JUDGE_MODEL` | `gpt-4o-mini` | Reranker + verifier + metrics |
| `KUBERAG_CHUNK_SIZE` | `800` | Characters per chunk |
| `KUBERAG_CHUNK_OVERLAP` | `120` | Overlap between adjacent chunks |
| `KUBERAG_RETRIEVAL_K` | `10` | Per-retriever candidate count |
| `KUBERAG_RERANK_TOP_N` | `5` | Final reranked count |
| `KUBERAG_RRF_DENSE_WEIGHT` | `0.7` | Dense weight in RRF (1.0 = dense only) |
| `KUBERAG_DEDUPE_THRESHOLD` | `0.95` | Cosine similarity above which to skip a chunk |
| `KUBERAG_CONFIDENCE_THRESHOLD` | `0.4` | Composite confidence floor for grounded answers |

See `.env.example` for the complete list.

## License

> **TODO:** Add a LICENSE file. MIT is a reasonable default for portfolio projects.
