# memory-service

A long-term memory service for an LLM agent. Ingests conversation turns,
extracts structured facts (v0.3+), and answers recall queries via a
hybrid retrieval pipeline (v0.4+). Designed against the Higgsfield AI
Engineering Challenge (`task.md`).

> **Status:** v0.5 - submission ready. Hybrid retrieval (BM25 + vector
> + RRF) + Jina Reranker v3 + bi-temporal supersession. See
> `CHANGELOG.md` for the full iteration history and `plan.md` for design
> + debates.
>
> **Eval results** (LLM-as-judge, 3-class verdict yes/partial/no):
> - LongMemEval-S cleaned, N=12 stratified across 4 categories: **0.75**
> - Synthetic fixture (17 probes, 6 categories): **0.82** - this number
>   is fixture-overfit and should not be compared against the public
>   dataset score; see [Honest disclosures](#honest-disclosures) below.

---

## Quick start

```bash
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY=sk-or-v1-...

docker compose up -d
until curl -sf http://localhost:8080/health; do sleep 1; done
```

The service binds to **port 8080** and persists to the named Docker
volume `memory-db-data`. `docker compose down && docker compose up -d`
preserves all data.

To run the test suites against the running service:

```bash
pip install -r requirements.txt
pytest -m contract                  # 11 shape & resilience tests
pytest -m persistence               # restart-survival (needs docker CLI)
pytest -m concurrent                # cross-user / cross-session isolation
pytest -m memeval -s                # synthetic recall-quality fixture
pytest -m longmemeval -s            # real LongMemEval-S, ~$1, ~25 min
```

Smoke test (the exact ТЗ §7 example):

```bash
bash scripts/smoke.sh
```

Override the target with `MEMORY_SERVICE_URL=http://...:8080 pytest`.

## Stack

Default: **OpenRouter** as the unified gateway for embedding + extraction
(one key, two models). The reranker uses Jina's direct API since
OpenRouter does not currently expose Jina v3.

| Layer | Default model | Endpoint |
|---|---|---|
| Embedding | `google/gemini-embedding-2-preview` via OpenRouter (1536d, MRL truncate) | `POST /api/v1/embeddings` |
| Extraction LLM | `openai/gpt-5.4-mini` via OpenRouter (strict JSON Schema, v0.3+) | `POST /api/v1/chat/completions` |
| Reranker | `jina-reranker-v3` direct (v0.4+, BEIR nDCG-10 ~62) | `POST https://api.jina.ai/v1/rerank` |

| Layer | Choice | Notes |
|---|---|---|
| API | FastAPI + uvicorn | Synchronous `/turns` (no async queues). |
| Storage | PostgreSQL 16 + pgvector + tsvector | Single engine: ACID for supersession, hybrid retrieval, recursive CTEs for graph traversal. |
| Persistence | Named volume `memory-db-data` | Survives `docker compose down/up`. |
| Auth | Optional `MEMORY_AUTH_TOKEN` bearer | Off by default. |

## Provider alternatives

Set per-layer `*_PROVIDER=direct` and provide the matching key, or list
the model in the same field — the service routes accordingly.

```env
# Direct Gemini (Google AI Studio key)
EMBEDDING_PROVIDER=direct
EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_KEY=...

# Direct Jina reranker (much better than Cohere on BEIR; needs separate key)
RERANKER_PROVIDER=direct
RERANKER_MODEL=jina-reranker-v3
JINA_API_KEY=...

# Direct Cohere rerank (skip OpenRouter)
RERANKER_PROVIDER=direct
RERANKER_MODEL=rerank-v3.5
COHERE_API_KEY=...

# Direct OpenAI for extraction
EXTRACTION_PROVIDER=direct
EXTRACTION_MODEL=gpt-4.1-mini
OPENAI_API_KEY=...

# Anthropic Claude for extraction
EXTRACTION_PROVIDER=direct
EXTRACTION_MODEL=claude-haiku-4-5
ANTHROPIC_API_KEY=...
```

## Self-host fallback (no API keys, GPU available)

If you'd rather not call out at all, point the service at a local
inference server (TEI, vLLM, Ollama) exposing OpenAI-compatible endpoints
and override the base URL:

```env
EMBEDDING_PROVIDER=direct
EMBEDDING_MODEL=codefuse-ai/F2LLM-v2-4B          # MTEB top-2 in 4B class
OPENAI_API_KEY=local                             # any non-empty token
OPENROUTER_BASE_URL=http://host.docker.internal:8001/v1

RERANKER_PROVIDER=direct
RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B          # or jinaai/jina-reranker-v3 weights
```

These models require a GPU container — without it the eval's 60s timeout
on `/turns` will trip on the first batch. We do **not** ship a
`docker-compose.gpu.yml` by default; you'd add one with the relevant
NVIDIA runtime annotation pointing at TEI / vLLM.

## Architecture (high level)

```
POST /turns    →  episodic_turn (append-only) + extract → memory (bi-temporal)
POST /recall   →  hybrid (BM25 + vector) → RRF → reranker → tiered context
POST /search   →  same retrieval, structured response
GET  /users/.. →  inspect structured memories with supersession chain
DELETE ..      →  hard delete (eval cleanup)
```

See `plan.md` for the architecture diagram and the rationale behind each
choice (storage, embedding, reranker, supersession, contradiction
handling).

## Endpoints

All endpoints accept an optional `Authorization: Bearer <token>` header.
If `MEMORY_AUTH_TOKEN` is set in `.env`, all non-`/health` endpoints
require it.

### `GET /health`
Returns `200 {"status":"ok"}` when the service can reach Postgres.

### `POST /turns`
Persist a completed turn. **Synchronous**: extraction (v0.3+) and
indexing run before the response. ТЗ allows up to a 60s timeout.

```json
{
  "session_id": "smoke-1",
  "user_id": "user-1",
  "messages": [
    {"role": "user", "content": "I just moved to Berlin from NYC."},
    {"role": "assistant", "content": "How are you settling in?"}
  ],
  "timestamp": "2026-05-01T10:30:00Z",
  "metadata": {}
}
```

Returns `201 {"id": "<uuid>"}`.

### `POST /recall`
Returns the formatted context for the agent's next turn. Tiered priority
(v0.4): stable identity facts → query-relevant memories → recent turns.
Respects `max_tokens` via tiktoken.

### `POST /search`
Same retrieval, structured response — for explicit agent search calls.

### `GET /users/{user_id}/memories`
Inspect the structured memory store. Returns active and superseded rows
with the supersession chain (v0.3+).

### `DELETE /sessions/{session_id}` and `DELETE /users/{user_id}`
Hard delete. Used by the eval between scenarios. Returns 204.

## Failure modes

| Condition | Behaviour |
|---|---|
| Cold session / unknown user | `/recall` returns empty `context` and `citations`. Never errors. |
| Malformed JSON | 422 with details. Never 5xx. |
| Unicode oddity in payload | Persisted verbatim, indexed via tsvector. |
| Missing API keys | Service starts, logs the missing provider. `/turns` writes a turn without an embedding (it can still be retrieved via plain text scan in v0.4). |
| Postgres unreachable | `/health` returns 503. Other endpoints likewise. |
| Restart | Named volume preserves all data. |
| Concurrent sessions, same user | Memories are user-scoped, not session-scoped (v0.3+ documented). Sessions don't bleed across users. |

## Honest disclosures

Submitting a memory system that benchmaxxes its own fixture is easy and
worthless. A few things to keep in mind when reading the numbers:

- **Synthetic fixture (0.82) is fixture-overfit.** All thresholds in
  `src/services/assembler.py` (`MEMORY_DENSE_GATE=0.68`,
  `EPISODIC_DENSE_GATE=0.55`) were tuned against my own probes in
  `fixtures/probes.json`. The synthetic number is a sanity check on the
  pipeline, not a generalisation claim.
- **Real LongMemEval-S (0.75) is the number that matters.** That is
  scored with an independent LLM-judge against the public dataset
  (`xiaowu0162/longmemeval-cleaned`) over 12 stratified questions
  (knowledge_update, multi_session, single_session, temporal). Wall clock
  ~25 min, cost ~$1. Confidence interval at N=12 is wide (~±15pp); the
  v0.5 submission baseline runs at N=40.
- **"Multi-hop" probes in the synthetic fixture are not real multi-hop.**
  They are single-hop multi-fact (one retrieval surfacing several facts
  about one entity). True multi-hop reasoning across an entity graph is
  out of scope for this build.
- **Abstention is measured only on the synthetic fixture.** The
  LongMemEval-cleaned subset has very few abstention questions, so the
  abstention metric is mostly diagnostic.
- **The cosine gate is what actually drives abstention.** The Jina v3
  reranker is used for ordering only - its logit-like scores are too
  compressed on personal-fact corpora to make a clean gate. Raw Gemini
  Embedding 2 cosine top-1 has a cleaner separation between legitimate
  recalls (≥0.65) and abstention queries (≤0.55).

## Repo layout

```
.
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── plan.md             ← full design + debates (read first)
├── CHANGELOG.md        ← iteration history with metrics per version
├── README.md
├── src/
│   ├── main.py
│   ├── config.py
│   ├── db.py
│   ├── auth.py
│   ├── schemas.py
│   ├── api/            ← one router per endpoint
│   ├── services/       ← embedding, openrouter, tokens (extraction
│   │                     and reranker land in v0.3/v0.4)
│   └── migrations/     ← *.sql, applied at startup
├── scripts/
│   └── smoke.sh             ← exact task.md §7 smoke test
├── tests/
│   ├── conftest.py
│   ├── test_contract.py     ← 11 shape & resilience tests
│   ├── test_persistence.py  ← restart-survival (docker compose stop+start)
│   ├── test_concurrent.py   ← cross-user / cross-session isolation
│   ├── test_memeval.py      ← synthetic recall-quality fixture
│   ├── test_longmemeval.py  ← real LongMemEval-S cleaned eval
│   └── fixtures/
│       ├── memeval_baseline.json
│       └── longmemeval_baseline.json
└── fixtures/
    ├── conversations.json   ← 2 users, 9 sessions, supersession arc
    ├── probes.json          ← 17 probes × 6 categories
    └── longmemeval/         ← downloaded by src/eval/loader.py (gitignored)
```
