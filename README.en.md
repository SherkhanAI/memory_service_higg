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
> - LongMemEval-S cleaned, **N=40** stratified (10 per category): **0.61**
>   (knowledge_update 0.80, multi_session 0.55, single_session 0.50,
>   temporal 0.56). 38/40 valid scores; 2 judge errors excluded.
> - LongMemEval-S cleaned, N=12 (sanity-check): **0.75**.
> - Synthetic fixture (17 probes, 6 categories): **0.82** - this number
>   is fixture-overfit and should not be compared against the public
>   dataset score; see [Honest disclosures](#honest-disclosures) below.
>
> Honest caveat: the N=40 run was partly degraded by ~60 `402 Payment
> Required` responses from OpenRouter near the end of the run when
> credits were exhausted, which silently turned several extraction calls
> into empty fact lists. The number is reported as-is rather than
> retried, since fixing the credits and re-running on the same seed
> would just be benchmaxxing.

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

## Backing store choice

**Postgres 16 + pgvector + tsvector**, single engine. Reasons:

- ACID transactions are load-bearing for bi-temporal supersession - a
  conflicting fact has to invalidate the old row and link the new one
  atomically, otherwise `/users/.../memories` returns inconsistent state
  during the gap.
- pgvector handles cosine search at our scale (10k-1M memories per user
  is well under HNSW's saturation point).
- tsvector + `ts_rank_cd` is a serviceable BM25-flavoured sparse search
  with no extra deployment surface.
- One container, one volume, one connection pool. No two-phase commit
  between a vector store and a relational store.

Schema: `episodic_turn` (append-only raw turns + embedding + tsv) and
`memory` (bi-temporal: `t_valid`, `t_invalid`, `t_created`,
`t_expired`, `superseded_by` self-FK, plus `mention_count` for
salience). Migrations apply at startup from `src/migrations/*.sql`.

## Extraction pipeline

`POST /turns` runs three stages synchronously inside the 60-second
budget:

1. **Embed** the raw turn text once (Gemini Embedding 2 via OpenRouter,
   1536d MRL-truncated and L2-renormalised).
2. **Persist** the episodic turn (raw text + embedding + tsv).
3. **Extract** structured facts with a single strict-JSON-schema call
   to `openai/gpt-5.4-mini`. Coreference is handled by passing the last
   2 prior turns of the same session as context. Each fact is `{type,
   predicate, object_text, stance, confidence, is_implicit, source_text}`
   keyed by a canonical predicate enum (~30 predicates: `name`,
   `employer`, `lives_in`, `pet`, `preference.*`, `event.*`, ...).
4. **Reconcile** each new fact against existing memories using a Mem0-style
   4-action decision (`ADD`, `UPDATE`, `SUPERSEDE`, `NOOP`) before
   writing - exclusive predicates supersede, multi-valued predicates
   accumulate, duplicates bump `mention_count`.

What we extract well: names, employer/role, location, pets, preferences,
opinions with stance, life events with date qualifiers.

What we miss: multi-hop reasoning across entities, fine-grained temporal
facts when the LLM paraphrases away the date, anything outside the
~30-predicate vocabulary (it falls into `other:*` and gets scored
purely on embedding match, no canonical reasoning).

## Recall strategy

`POST /recall` runs a hybrid retrieval -> rerank -> tiered assembly
pipeline:

1. **Hybrid retrieve** in 4 parallel SQL streams (memory dense, memory
   sparse, episodic dense, episodic sparse), fused with Reciprocal Rank
   Fusion (RRF, k=60). Sparse and dense are scale-incompatible, so
   rank-based fusion is the safe default.
2. **Rerank** the top-30 candidates with Jina Reranker v3. Memory facts
   are verbalised into natural-language sentences before they hit the
   cross-encoder ("employer: Notion" -> "The user currently works at
   Notion."); cross-encoders are trained on sentences, not key:value
   triples, and the verbalisation step alone moved a critical probe
   from ranked-second to ranked-first.
3. **Fetch stable identity facts** independently (predicate-filtered SQL,
   no retrieval) - these are the always-on user profile.
4. **Fetch the last few turns** of the current session for follow-up
   context, scoped by both `session_id` and `user_id`.
5. **Assemble** under `max_tokens` with three quotas: stable identity
   ~30%, query-relevant ~50%, recent context ~20% (overflow re-allocated
   to the next section). The order matches the priority defended below.
6. **Abstention gate**: if max(memory_dense_top1, episodic_dense_top1)
   raw cosine is below `MEMORY_DENSE_GATE` (0.68), return an empty
   context. The reranker is used for ordering, not gating - its
   logit-like scores are too compressed to threshold cleanly on
   personal-fact corpora.

**Priority under budget tightness**: stable identity facts win first,
relevant memories second, recent turns last. Reasoning: identity context
("works at Notion, allergic to shellfish") is needed for *every* turn,
even when the query has no clear retrieval target. Dropping it to make
room for one more recent turn produces a worse downstream answer.

## Fact evolution

Bi-temporal model with explicit supersession (Zep/Graphiti-flavoured):

- `t_valid` is when the fact became true in the world ("I just started
  at Notion" stamped from the turn timestamp).
- `t_invalid` is when it stopped being true (set when superseded).
- `t_created` / `t_expired` track the database-write time independently.
- `superseded_by` is a self-FK to the row that replaced this one.

Reconciliation runs an LLM-decided 4-action choice (Mem0):

| Action | When | Effect |
|---|---|---|
| `ADD` | New predicate or non-conflicting multi-valued | Insert active row |
| `UPDATE` | Same predicate, slightly different wording | Insert + invalidate predecessor, link via `superseded_by` |
| `SUPERSEDE` | Exclusive predicate, conflicting object | Same as UPDATE; semantic distinction |
| `NOOP` | Duplicate / already known | Bump `mention_count` only |

`/recall` filters `t_invalid IS NULL` so superseded rows are invisible
to the agent but visible to the inspector at `/users/{user_id}/memories`.

Opinion arcs (the harder variant in task.md): handled as repeated
`opinion_about` facts with explicit `stance` field (`positive` /
`negative` / `mixed` / `none`) and `t_valid` timestamps. The current
implementation surfaces the most recent stance in `/recall`; building a
trajectory summary ("started positive, drifted to mixed") is an
extraction-time task that's out of scope for this build.

## Tradeoffs

Optimised for:

- Honest measurement on a public benchmark (LongMemEval-S cleaned)
  rather than a tuned-to-impress synthetic number.
- Zero infrastructure surface beyond Postgres + the app container.
- Synchronous `/turns` so eval probes never observe stale state.
- Provider-agnostic routing (OpenRouter unified gateway with
  per-layer overrides).

Gave up:

- Throughput per `/turns` (one extraction LLM call + 4 SQL writes per
  turn -> ~2-4 s latency). Acceptable inside the 60 s budget; would
  not scale to thousands of writes per second without batching.
- Multi-hop reasoning across entities (no graph traversal step). All
  retrieval is single-hop hybrid over flat memories.
- True opinion-arc summarisation. Stances are stored, but not
  trajectory-summarised at recall time.
- Self-host with no API keys (provider keys are the default; the
  README documents how to point at a local TEI/vLLM endpoint, but
  there is no `docker-compose.gpu.yml` shipped).

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
