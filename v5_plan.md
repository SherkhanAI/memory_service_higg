# v0.5 Plan + LongMemEval Improvement Notes

**This file is the recovery anchor after memory compaction.** Read it
plus `plan.md`, `CHANGELOG.md`, and the two baseline JSONs to fully
restore context.

---

## Snapshot of current state (end of v0.4.5)

- **Stack**: Python 3.12 + FastAPI + Postgres 16 + pgvector
- **Models** (all via OpenRouter unified gateway, except Jina direct):
  - Embedding: `google/gemini-embedding-2-preview` (MRL 1536d)
  - Extraction: `openai/gpt-5.4-mini`
  - Reranker: `jina-reranker-v3` (direct API, **not** via OpenRouter)
- **Auth**: keys in `.env`
  - `OPENROUTER_API_KEY` (73 chars, present)
  - `JINA_API_KEY` (65 chars, present)
- **Tests passing**:
  - 11/11 contract (`pytest -m contract`)
  - 17/17 synthetic memeval (overall 0.82, see `tests/fixtures/memeval_baseline.json`)
  - 12/12 real LongMemEval-S (overall 0.75, see `tests/fixtures/longmemeval_baseline.json`)
- **Working dir**: `C:\Users\User\higgsfield`
- **Container state**: `docker compose up -d` runs `memory-db` + `memory-app` on port 8080. Volume `memory-db-data` persists data.
- **Health**: `curl -sf http://localhost:8080/health` returns 200.

### Numbers reference (real LongMemEval-S, N=12)

| Category         | Score | n   |
|------------------|-------|-----|
| knowledge_update | 1.00  | 3/3 |
| multi_session    | 0.67  | 3/3 |
| single_session   | 0.67  | 3/3 |
| temporal         | 0.67  | 3/3 |
| **overall**      | **0.75** | 12 |

Wall clock 22.5 min, ~$1 spent. CI on per-category accuracy is wide
(±32pp at n=3), so target N=10/cat for the submission baseline.

---

## v0.5 — Submission readiness checklist

### P1 — blockers (task.md hard requirements)

#### 1. `tests/test_persistence.py` — restart-survival (§5 + §7)
ТЗ explicitly: "data survives docker compose down && docker compose up".
Implementation:
```python
@pytest.mark.persistence
def test_restart_persistence(client):
    user = f"persist_{uuid4().hex[:8]}"
    # ingest 2 turns with extractable content
    client.post("/turns", json={...})
    # snapshot recall + memories
    before_recall = client.post("/recall", json={...}).json()
    before_memories = client.get(f"/users/{user}/memories").json()
    # restart container
    subprocess.run(["docker", "compose", "stop", "app"], check=True)
    subprocess.run(["docker", "compose", "start", "app"], check=True)
    wait_for_health(client)
    # re-snapshot
    after_recall = client.post("/recall", json={...}).json()
    after_memories = client.get(f"/users/{user}/memories").json()
    assert before_memories == after_memories
    assert before_recall["context"] == after_recall["context"]
    client.delete(f"/users/{user}")
```
Note: needs `docker` CLI access from pytest. Skip if `which docker` not found.

#### 2. `tests/test_concurrent.py` — cross-user / cross-session isolation (§5)
```python
@pytest.mark.concurrent
def test_concurrent_users_no_leak(client):
    # parallel ingest 2 users with different facts via threadpool
    # each user gets their own /turns and /recall
    # assert recall(user_a) does not include user_b facts
    # assert recall(user_b) does not include user_a facts
```
Use `concurrent.futures.ThreadPoolExecutor` with httpx.

#### 3. `/search` session-only bug fix (§3 contract)
**File**: `src/api/search.py:24-28`
```python
# Current (wrong):
if not req.user_id and not req.session_id:
    return SearchOut(results=[])
user_id = req.user_id or ""
if not user_id:
    return SearchOut(results=[])

# Should be: support session-only by querying episodic_turn directly
# without user_id filter when session_id is provided.
```
Implementation: pass `user_id=None` to `hybrid_search`; in retrieval,
if `user_id` is None but `session_id` set, scope to session in SQL.

### P2 — quality of life

#### 4. `scripts/smoke.sh` — exact ТЗ §7 example
```bash
#!/usr/bin/env bash
set -e
curl -sf http://localhost:8080/health | jq .
curl -X POST http://localhost:8080/turns -H 'Content-Type: application/json' \
    -d '{"session_id":"smoke-1","user_id":"user-1", ...}'
curl -X POST http://localhost:8080/recall ... # expects "Berlin"
curl http://localhost:8080/users/user-1/memories | jq .
```

#### 5. README final polish — **honest disclosures**
- "Synthetic memeval is fixture-overfit (0.82); real LongMemEval-S baseline 0.75"
- Threshold caveat: gates calibrated against my fixture, may need recalibration
- Multi-hop probes in synthetic are fake (single-hop multi-fact)
- abstention measured only on synthetic; LongMemEval-cleaned has ~5%

#### 6. Final LongMemEval N=10/cat run (40 q, ~75 min, ~$3)
Command:
```bash
docker compose up -d
LONGMEMEVAL_N_PER_CAT=10 LONGMEMEVAL_DISTRACTORS=8 \
  python -m pytest tests/test_longmemeval.py -m longmemeval -s
```
Output to `tests/fixtures/longmemeval_baseline.json`. Use that number
in the final CHANGELOG v0.5 entry.

### P3 — skip unless time

7. `docker-compose.gpu.yml` for self-host F2LLM/Qwen3-Reranker (mention only)
8. Rate-limit handling on 429s (currently retries only 5xx)

---

## LongMemEval improvement targets (for v0.5+)

The pattern in the v0.4.5 run: most failures were `partial` (0.5), not
`no` (0.0). System retrieves the right session/area but the extracted
memory is too abstract. **Fix extraction fidelity, not retrieval.**

### Concrete improvements ranked by ROI

#### High ROI (do these for v0.5)

**A. Verbatim source quotes**
- Current: extraction LLM emits paraphrased `source_text`
- Fix: enforce verbatim quote via prompt + post-validation
  ```python
  # In extraction.py post-processing:
  if fact["source_text"] not in turn_text:
      # Find nearest substring match; or drop fact
  ```
- Even better: integrate **LangExtract** (`pip install langextract`)
  for char-span grounded extraction. Replaces `chat_json` call in
  `src/services/extraction.py`.
- **Why ROI high**: judge counts "yes" only when context conveys exact
  gold answer. Paraphrased quotes lose dates/numbers/names → partial.

**B. Anthropic Contextual Retrieval prefix on `episodic_turn`**
- For each turn, generate 50-100 token prefix via LLM (prompt-cached):
  "This turn is from a session on YYYY-MM-DD where the user discussed
  X. Within it: ..."
- Prepend to `raw_text` BEFORE embedding.
- Reported gains: 49% reduction in retrieval failures (Anthropic blog).
- **Cost**: 1 extra prompt-cached LLM call per turn. Acceptable since
  extraction already runs sync.

**C. Object_qualifiers re-enabled**
- Currently `object_qualifiers={}` always (we dropped it for strict
  schema simplicity).
- Fix: extend `_FACT_ITEM` schema with optional `object_qualifiers`
  as `string` (JSON-encoded), parse server-side.
- Critical for: dates ("started Notion 2026-02"), counts, breeds.

**D. Lower memory_dense_gate from 0.68 to 0.60-0.62**
- Real-data probes don't always hit 0.68 cosine. Some legitimate
  recalls in LongMemEval scored 0.6-0.65.
- Won't break abstention because LongMemEval-cleaned has very few
  abstention probes anyway (only ~5% question_id ends in `_abs`).

#### Medium ROI

**E. Multi-pass extraction**
- Pass 1: identify atomic facts (current)
- Pass 2: for each fact, extract dates/numbers/named entities into
  `object_qualifiers`
- Doubles cost but boosts partial→yes significantly.

**F. Coreference window from 2 → 5 prior turns**
- File: `src/api/turns.py:_recent_session_text`
- Parameter: `limit=5`
- Catches longer multi-message coref chains.

**G. Temporal-aware verbalization**
- Add date suffix to memory text:
  `"The user currently works at Notion (as of 2026-02-18)."`
- Helps temporal queries which currently miss because date isn't in
  embedding text.
- File: `src/services/retrieval.py:_make_memory_text`

#### Low ROI / skip

**H. ColBERT / late-interaction**: too much infra for marginal gain.
**I. GraphRAG full**: indexing cost not worth it at 50 q scale.
**J. Fine-tune extractor**: out of scope for 2-day build.

### Expected v0.5 numbers
With improvements A+B+C: real LongMemEval overall 0.75 → ~0.85.
With A+B+C+D+E+F+G: ~0.88-0.92.

These are guesses; the ONLY way to know is to run N=10/cat after each
change and measure. Don't tune gates against synthetic.

---

## Recovery procedure after memory compaction

If you (Claude) are reading this after compaction:

1. **Read these files in order**:
   - `task.md` — original spec
   - `plan.md` — architecture + debates (long but indexable)
   - `CHANGELOG.md` — full iteration history v0.1 → v0.4.5
   - `v5_plan.md` — this file
   - `tests/fixtures/longmemeval_baseline.json` — last real numbers
   - `tests/fixtures/memeval_baseline.json` — synthetic numbers

2. **Verify environment**:
   - `docker compose ps` should show `memory-db` and `memory-app` Running
   - `curl -sf http://localhost:8080/health` returns 200
   - `awk -F= '/^OPENROUTER_API_KEY=/{print "ok=" (length($2)>0)}' .env`
   - `awk -F= '/^JINA_API_KEY=/{print "ok=" (length($2)>0)}' .env`

3. **Pick up from current task**: should be P1 #1 (`test_persistence.py`).

4. **Don't re-tune thresholds against synthetic** — the user explicitly
   warned against benchmaxxing. Always test changes on real LongMemEval
   subsample.

5. **Cost discipline**: each LongMemEval run N=12 is ~$1, ~25 min.
   N=40 is ~$3, ~75 min. Don't run more than necessary.

6. **Open subtasks**: nothing in-flight as of this writing — the
   v0.4.5 LongMemEval N=12 run finished and is already in CHANGELOG.

---

## Key file paths (don't lose these)

```
src/main.py                       FastAPI app + lifespan
src/config.py                     Settings (OpenRouter / Jina routing)
src/db.py                         psycopg pool + bootstrap extensions
src/auth.py                       optional bearer
src/schemas.py                    Pydantic models matching ТЗ contract
src/api/health.py                 GET /health
src/api/turns.py                  POST /turns (sync ingest pipeline)
src/api/recall.py                 POST /recall (hybrid + rerank + tiered)
src/api/search.py                 POST /search (BUG: session-only)
src/api/users.py                  GET memories, DELETE user
src/api/sessions.py               DELETE session
src/services/openrouter.py        shared httpx client for OpenRouter
src/services/embedding.py         Gemini embedding via OpenRouter (MRL truncate)
src/services/llm.py               chat_json wrapper (strict JSON Schema)
src/services/extraction.py        single-call structured extractor
src/services/predicates.py        canonical predicate enum + classification
src/services/reconciliation.py    Mem0 4-action + bi-temporal write
src/services/retrieval.py         hybrid 4-stream + RRF + NL verbalization
src/services/reranker.py          Jina v3 / OpenRouter rerank
src/services/assembler.py         tiered context + cosine gate
src/services/tokens.py            tiktoken budget helper
src/migrations/001_init.sql       Postgres schema (bi-temporal memory)
src/eval/loader.py                LongMemEval HF download + subsample
src/eval/judge.py                 LLM-as-judge for LongMemEval scoring
tests/test_contract.py            11 shape & resilience tests
tests/test_memeval.py             synthetic LongMemEval-shaped runner
tests/test_longmemeval.py         real LongMemEval-S runner
fixtures/conversations.json       synthetic users (Alex + Taylor)
fixtures/probes.json              17 synthetic probes
fixtures/longmemeval/             277MB real dataset cache (gitignored)
docker-compose.yml                pgvector/pgvector:pg16 + app
Dockerfile                        python:3.12-slim
requirements.txt                  pinned versions
plan.md                           architecture + debates
CHANGELOG.md                      v0.1 → v0.4.5 with metrics
README.md                         deploy guide + stack table
```

## Settings cheat-sheet

```python
# src/services/assembler.py
MEMORY_DENSE_GATE = 0.68      # raw cosine threshold for memory hits
EPISODIC_DENSE_GATE = 0.55    # raw cosine threshold for episodic hits
INCLUDE_FLOOR = -1.0           # don't filter on rerank, trust ordering
_STABLE_QUOTA = 0.30
_RELEVANT_QUOTA = 0.50
_RECENT_TURN_LIMIT = 4
```

```python
# src/services/predicates.py
STABLE_PREDICATES = {name, age, lives_in, employer, role, pet, ...}
EXCLUSIVE_PREDICATES = {name, lives_in, employer, role, ...}     # SUPERSEDE on conflict
MULTI_VALUED_PREDICATES = {skill, hobby, friend, lived_in, ...}  # ADD on different value
```

```python
# tests/test_longmemeval.py defaults (env-overridable)
LONGMEMEVAL_N_PER_CAT = 5     # questions per category
LONGMEMEVAL_DISTRACTORS = 8   # random non-evidence sessions per question
LONGMEMEVAL_SEED = 42
```
