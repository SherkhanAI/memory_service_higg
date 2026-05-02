# memory-service

A long-term memory service for an LLM agent. Ingests conversation turns,
extracts structured facts (v0.3+), and answers recall queries via a
hybrid retrieval pipeline (v0.4+). Designed against the Higgsfield AI
Engineering Challenge (`task.md`).

> **Status:** v0.2 — Gemini Embedding via OpenRouter + vanilla-cosine
> baseline. Self-eval overall **0.71**. Extraction and reranking land in
> v0.3/v0.4. See `CHANGELOG.md` for iteration history and `plan.md` for
> the full design + debates.

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

To run the contract tests and self-eval against the running service:

```bash
pip install -r requirements.txt
pytest -m contract                  # 11 shape & resilience tests
pytest -m memeval -s                # recall-quality fixture (needs API key)
```

Override the target with `MEMORY_SERVICE_URL=http://...:8080 pytest`.

## Stack

Default: **OpenRouter** as the unified gateway — one key, three models.

| Layer | Default model (via OpenRouter) | Endpoint |
|---|---|---|
| Embedding | `google/gemini-embedding-2-preview` (1536d, MRL truncate) | `POST /api/v1/embeddings` |
| Extraction LLM | `openai/gpt-5.4-mini` (strict JSON Schema, v0.3+) | `POST /api/v1/chat/completions` |
| Reranker | `cohere/rerank-4-fast` (v0.4+) | `POST /api/v1/rerank` |

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
├── tests/
│   ├── conftest.py
│   ├── test_contract.py     ← 11 shape & resilience tests
│   ├── test_memeval.py      ← LongMemEval-shaped self-eval
│   └── fixtures/
│       └── memeval_baseline.json
└── fixtures/
    ├── conversations.json   ← 2 users, 9 sessions, supersession arc
    └── probes.json          ← 16 probes × 6 categories
```
