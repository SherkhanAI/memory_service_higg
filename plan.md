# Memory Service — План реализации (v0.1 → v0.5)

> 2-day build, single Git repo, `docker compose up` без ручных шагов.
> ТЗ: `task.md` (Higgsfield AI Engineering Challenge).

---

## TL;DR — финальный стек (2025-2026 SOTA)

**Zep-shaped service с Mem0-shaped write path.** Append-only лог сырых turns + bi-temporal граф фактов поверх него + гибридный recall с реранкером + LongMemEval-style self-eval.

| Слой | Решение |
|---|---|
| Язык/фреймворк | Python 3.12 + FastAPI + uvicorn |
| Storage | **PostgreSQL 16 + pgvector + tsvector** (один движок) |
| Embedding (primary) | **Gemini `gemini-embedding-001`** (MRL 3072→1536d, task-aware) |
| Embedding (fallback) | F2LLM-v2-4B local (GPU) / `text-embedding-3-small` API — в README |
| Sparse | Postgres `tsvector` + `ts_rank_cd` |
| Fusion | RRF, k=60 |
| Reranker (primary) | **Jina `jina-reranker-v3`** API (sota: 61.94 nDCG-10 на BEIR) |
| Reranker (fallback) | Qwen3-Reranker-0.6B / jina-reranker-v3 weights local — в README |
| Extraction LLM | OpenAI `gpt-4.1-mini` (или Anthropic `claude-haiku-4-5`) с JSON Schema |
| Контекстуализация turns | Anthropic Contextual Retrieval prefix |
| Multi-hop | Лёгкий entity-index + 2-hop BFS, итеративный hop при низком top-1 score |
| Контрадикции | Mem0 4-action prompt (`ADD/UPDATE/SUPERSEDE/NOOP`) + bi-temporal edges |
| Scoring | `0.5·rerank + 0.3·recency_decay + 0.2·mention_count` |

---

## Архитектура

```
                        ┌──────────────────────────────────────────┐
       POST /turns ───► │ Ingest                                    │
                        │  1. validate JSON                         │
                        │  2. write episodic_turn (append-only)     │
                        │  3. contextualize turn (LLM, prompt-cache)│
                        │  4. extract candidates (LLM, JSON Schema) │
                        │  5. normalize predicates                  │
                        │  6. retrieve k-NN existing memories       │
                        │  7. reconcile (LLM ADD/UPDATE/SUPERSEDE)  │
                        │  8. write memory rows + edges + embed     │
                        └──────────────────────────────────────────┘
                                          │
                                          ▼
                        ┌──────────────────────────────────────────┐
                        │ Postgres                                  │
                        │  - episodic_turn (raw, append-only)       │
                        │  - memory (bi-temporal: t_valid/t_invalid)│
                        │  - entity (people/places/dates)           │
                        │  - memory_entity (M2M edge)               │
                        │  - HNSW vector index, GIN tsvector index  │
                        └──────────────────────────────────────────┘
                                          │
       POST /recall ───► ┌────────────────▼─────────────────────────┐
       POST /search      │ Recall                                    │
                        │  1. coreference-resolve query (last 5     │
                        │     turns of session)                     │
                        │  2. hybrid: dense top-50 + BM25 top-50    │
                        │     filter t_invalid IS NULL              │
                        │  3. RRF fuse → top-30                     │
                        │  4. Jina reranker v3 → top-K              │
                        │  5. multi-hop: if top-1 score < τ,        │
                        │     extract entities → 2-hop BFS → re-rank│
                        │  6. assemble context under max_tokens:    │
                        │     [stable user facts] →                 │
                        │     [query-relevant memories] →           │
                        │     [recent context]                      │
                        └──────────────────────────────────────────┘
```

---

## Schema (Postgres 16 + pgvector)

```sql
-- Сырой лог turns, append-only
CREATE TABLE episodic_turn (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id      TEXT NOT NULL,
  user_id         TEXT,
  messages        JSONB NOT NULL,
  ts              TIMESTAMPTZ NOT NULL,
  metadata        JSONB DEFAULT '{}'::jsonb,
  created_at      TIMESTAMPTZ DEFAULT now(),
  raw_text        TEXT,                       -- joined messages
  context_prefix  TEXT,                       -- Anthropic contextual retrieval prefix
  embedding       vector(1536),               -- gemini MRL@1536
  tsv             tsvector
);
CREATE INDEX ON episodic_turn (session_id);
CREATE INDEX ON episodic_turn (user_id, ts DESC);
CREATE INDEX ON episodic_turn USING GIN (tsv);
CREATE INDEX ON episodic_turn USING hnsw (embedding vector_cosine_ops);

-- Структурированные факты с bi-temporal моделью
CREATE TYPE memory_kind AS ENUM ('fact','preference','opinion','event');
CREATE TYPE memory_confidence AS ENUM ('low','med','high');

CREATE TABLE memory (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           TEXT NOT NULL,
  session_scope     TEXT,                     -- NULL = cross-session
  subject           TEXT NOT NULL DEFAULT 'user',
  predicate         TEXT NOT NULL,            -- canonical key из enum или 'other:*'
  object_text       TEXT NOT NULL,
  object_qualifiers JSONB DEFAULT '{}'::jsonb,
  kind              memory_kind NOT NULL,
  stance            TEXT,                     -- positive|negative|neutral|null
  confidence        memory_confidence NOT NULL DEFAULT 'med',
  is_implicit       BOOLEAN NOT NULL DEFAULT false,
  -- Bi-temporal
  t_valid           TIMESTAMPTZ NOT NULL,
  t_invalid         TIMESTAMPTZ,              -- NULL = active
  t_created         TIMESTAMPTZ NOT NULL DEFAULT now(),
  t_expired         TIMESTAMPTZ,
  -- Provenance
  source_turn_id    UUID REFERENCES episodic_turn(id) ON DELETE CASCADE,
  source_text       TEXT,
  -- Evolution links
  superseded_by     UUID REFERENCES memory(id),
  mention_count     INT NOT NULL DEFAULT 1,
  -- Indexing
  embedding         vector(1536),
  tsv               tsvector
);
CREATE INDEX ON memory (user_id, predicate, t_invalid);
CREATE INDEX ON memory (user_id, kind, t_invalid);
CREATE INDEX ON memory (source_turn_id);
CREATE INDEX ON memory USING GIN (tsv);
CREATE INDEX ON memory USING hnsw (embedding vector_cosine_ops);

-- Entities для multi-hop
CREATE TABLE entity (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         TEXT NOT NULL,
  type            TEXT NOT NULL,              -- person|place|org|pet|event|...
  name            TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  first_seen_at   TIMESTAMPTZ DEFAULT now(),
  embedding       vector(1536),
  UNIQUE (user_id, type, normalized_name)
);
CREATE INDEX ON entity (user_id, type);
CREATE INDEX ON entity USING hnsw (embedding vector_cosine_ops);

CREATE TABLE memory_entity (
  memory_id  UUID REFERENCES memory(id) ON DELETE CASCADE,
  entity_id  UUID REFERENCES entity(id) ON DELETE CASCADE,
  role       TEXT,                            -- subject|object|qualifier
  PRIMARY KEY (memory_id, entity_id, role)
);
CREATE INDEX ON memory_entity (entity_id);
```

---

## Канонические предикаты (~30)

```
Identity:        name, age, gender, lives_in, lived_in, born_in, nationality
Work:            employer, role, employer_past, employer_start
Education:       degree, school, field_of_study
Relationships:   spouse, partner, family.parent, family.child, family.sibling, friend
Pets:            pet
Preferences:     preference.food, preference.language, preference.tool,
                 preference.framework, preference.activity
Opinions:        opinion_about        (с stance: positive/negative/neutral)
Events:          event.travel, event.health, event.life_change
Skills:          skill, hobby
Other:           other:<free-form>
```

LLM на extraction-шаге выбирает из enum или ставит `other:<freeform>`. Embedding-нормализация подтаскивает близкие варианты к канону.

---

## План реализации (CHANGELOG-coupled milestones)

### **v0.1 — Skeleton** (2-3h)
- `docker-compose.yml`: app + `pgvector/pgvector:pg16`
- FastAPI каркас, все 7 эндпоинтов отвечают валидным JSON (заглушки)
- Init-миграция: схема выше
- `/health` ready после миграций
- Basic contract test: roundtrip `/turns` → `/recall` (вернёт пустой контекст)
- **CHANGELOG**: v0.1 baseline. Метрика: 0/0 (фикстуры нет).

### **v0.2 — Naive baseline** (3-4h)
- Сохраняем raw text в `memory.value`, embed Gemini `gemini-embedding-001` (output_dim=1536)
- `/recall` = vanilla cosine top-k → конкатенация текстов
- Self-eval fixture: 3 conversations × 5 категорий × 4 probes = 60 probes (LongMemEval-shaped)
- Runner: `pytest tests/test_memeval.py`
- **CHANGELOG**: v0.2 vanilla cosine. Ожидаемая accuracy ~0.30-0.40 — намеренный пол.

### **v0.3 — Real extraction + bi-temporal** (5-6h)
- LLM-extractor с JSON Schema: `explicit_facts[] + implicit_facts[]`
- Schema: `(subject, predicate, object_text, kind, stance, valid_at, confidence, is_implicit, source_turn_id, source_text)`
- Канонический список ~30 предикатов как enum + `other:*`
- При записи: top-5 retrieval по `(user_id, predicate)` → второй LLM-вызов tool-call: `{action: ADD|UPDATE|SUPERSEDE|NOOP, target_id, merged_object?}`
- SUPERSEDE проставляет `t_invalid = now()` старой записи + `superseded_by`
- `/users/{id}/memories` возвращает структурированные факты с историей
- **CHANGELOG**: v0.3 structured extraction + supersession. Ожидаемая supersession accuracy 0→0.7, recall 0.4→0.55.

### **v0.4 — Hybrid + rerank + context assembly** (4-5h)
- BM25 через Postgres `tsvector` + `ts_rank_cd`; параллельно dense; RRF k=60
- Cross-encoder reranker: Jina v3 API → top-5
- Anthropic-style contextual prefix для каждого turn перед embedding (prompt-cached LLM-вызов)
- Coreference-resolve запроса по последним 5 turns сессии (один лёгкий LLM-вызов)
- Сборка контекста под `max_tokens` через tiktoken: stable user facts → top-K relevant → recent
- Multi-hop: если top-1 reranker score < 0.5, extract entities → 2-hop BFS по `memory_entity` → re-rank
- **CHANGELOG**: v0.4 hybrid + rerank + budget-aware assembly. Ожидаемый recall 0.55→0.75, multi-hop 0.3→0.6.

### **v0.5 — Polish + docs** (2-3h)
- Resilience: 4xx на malformed JSON / unicode / oversized payload
- Concurrent session test (две сессии разных users — нет утечек)
- Restart-persistence test (down/up — `/recall` возвращает то же)
- README с диаграммой, защитой решений, failure modes, fallback гайдом для self-host (F2LLM, Qwen3-Reranker)
- `.env.example`, smoke test
- **CHANGELOG**: v0.5 hardening + docs.

---

## Self-Eval Fixture (LongMemEval-shaped)

3 conversations × 5 категорий × 4 probes = 60 probes.

| Категория | Спецификация |
|---|---|
| **Recall (IE)** | факт стейтнут единожды → ассерт через 5+ сессий |
| **Multi-hop** | "city of user with dog Biscuit" — chain через 2 факта |
| **Temporal** | "before joining Notion" — порядок |
| **Supersession** | Stripe→Notion, вернуть только Notion |
| **Abstention** | тема не упоминалась → пустой context, штраф за галлюцинацию |

Метрики:
- Per-category accuracy (LLM-judge через `gpt-4.1-mini`, бинарный)
- Recall@5 по ground-truth memory IDs
- Tokens-per-recall
- Abstention precision

Бейзлайн в `tests/fixtures/memeval_baseline.json`. CI fail при просадке >5pp.

---

## Дебаты (для интервью и README)

### 1. Backing store: **Postgres+pgvector** vs Qdrant vs SQLite+FTS

| | Postgres+pgvector | Qdrant | SQLite+FTS5 |
|---|---|---|---|
| Hybrid из коробки | tsvector + pgvector в одном WHERE | sparse+dense, fusion native | FTS5 + sqlite-vec |
| ACID для supersession | ✅ tx, FK, триггеры | слабее | ✅ |
| Recursive CTE для 2-hop BFS | ✅ | ❌ | ✅ |
| Single-image docker | + 1 контейнер | + 1 контейнер | 0 контейнеров |
| Скорость на ~10k векторов/user | OK с HNSW | лучше | OK |

**Выбор: Postgres.** Bi-temporal модель и supersession естественно ложатся на реляционную схему. Qdrant требовал бы отдельный Postgres для метаданных — два движка хуже одного.

### 2. Memory representation: free-form vs strict triples vs **hybrid**

- Free-form (Mem0): `value="User's favorite food is pasta"` — supersession через embedding-сходство, шумно.
- Strict triples (HippoRAG/ODKE+): `(user, employer, Stripe)` — детерминируемо, но негибко.
- **Hybrid (Zep)**: `(subject, predicate∈enum, object_text, qualifiers_jsonb)`. **Выбран.** Predicate из канонического списка → supersession через индекс; object — свободный текст.

### 3. Контрадикции: NLI vs **LLM judge** vs vector threshold

- NLI (DeBERTa-v3): дёшево, но ещё одна модель в контейнере.
- **LLM judge** (Mem0/Zep): дороже, но ловит нюанс ("работаю там" vs "интервьюируюсь") и возвращает action enum. Стоит ~$0.0001/вызов на mini.
- Vector threshold: только парафразы.

**Выбор: LLM judge.** Один батч-запрос на ≤5 кандидатов. NLI как Phase-2 (упомянем в README).

### 4. Bi-temporal vs append-only-with-pointer vs destructive

- Destructive: теряем историю → **дисквалифицирующее по ТЗ**.
- Append-only с `superseded_by`: просто, но "что было активно на дату X" требует развёртки.
- **Bi-temporal (Zep)**: `t_valid`, `t_invalid`, `t_created`, `t_expired`. **Выбран.** Запрос «активно сейчас» = `WHERE t_invalid IS NULL`; «активно на дату X» = `WHERE t_valid <= X AND (t_invalid IS NULL OR t_invalid > X)`. Стоит ~20 LOC.

### 5. Recall: vanilla vs **hybrid+RRF+rerank** vs HippoRAG/GraphRAG

- Vanilla cosine: ТЗ явно говорит "will not score well".
- **Hybrid BM25+dense+RRF + Jina v3 rerank**: канонический мейнстрим, +15-25% recall на keyword запросах. **Выбран.**
- HippoRAG/GraphRAG: PPR over KG, ~+20% F1 на multi-hop, но требует OpenIE-пайплайна — слишком тяжёлый для 2 дней. Замена: лёгкий entity-index + 2-hop BFS только когда top-1 score низкий.

### 6. Reranker: **Jina v3 API** vs Qwen3 vs LLM-as-reranker

- Jina v3: SOTA на BEIR (61.94 nDCG-10), 0.6B, 131K context, 64 docs за вызов. **Выбран как primary.**
- Qwen3-Reranker-0.6B/4B: open weights, для self-host. **В README как fallback.**
- bge-reranker-v2-m3: устарел.
- LLM rerank: дороже всего, не оправдано.

### 7. Embedding: **Gemini** vs OpenAI vs F2LLM local

- Gemini `gemini-embedding-001`: MRL (3072→1536→768), task-aware (`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`), 8192 tokens. **Выбран.**
- `text-embedding-3-small`: cost/quality sweet-spot, но без task-awareness и MRL менее гибкий. **Fallback в env-var.**
- F2LLM-v2-4B: лучшая открытая модель в ~4B классе на MTEB (#2 в 4B-bracket). **В README для self-host на GPU.**

### 8. Сборка контекста: **tiered priority** под бюджет

ТЗ требует "stable user facts → query-relevant → recent" с защитой.

**Выбор: tiered квоты на бюджет**:
1. **Stable identity** (`predicate ∈ {employer, lives_in, name, pet, family.*}`, `t_invalid IS NULL`) — fixed quota ~30%.
2. **Query-relevant** (top-K из реранкера) — ~50%.
3. **Recent** (последние N turns текущей сессии) — ~20%.

Каждый блок усекается tiktoken'ом по своей квоте. Логика объяснимая, защищаема в README.

### 9. Async vs **sync** extraction

ТЗ: "after `/turns` returns, memories must be immediately queryable". Это убивает async-очереди. Плюс 60s timeout.

**Выбор: полностью синхронно.** Никаких Celery/RQ.

### 10. Извлечение: **one-shot JSON Schema** vs multi-pass

- One-shot strict JSON Schema: дёшево, fast, валидно через structured outputs.
- Two-step (free-think → schema): +15-20% точности, удваивает стоимость.
- Multi-pass (extract → verify → normalize): хорошо, но 3 LLM-вызова на turn → риск 60s timeout.

**Выбор: one-shot с strict JSON Schema** на ингесте. 5-8 few-shot примеров (negation, opinion shift, ambiguous→skip). Если parse-error rate >5% — добавим second-pass валидатор.

---

## Что отбрасываем (защита в интервью)

| Идея | Почему пропускаем |
|---|---|
| HyDE | Галлюцинирует на personal facts. |
| ColBERT/ColPali | Operational overhead 5-10x хранилища. |
| Full GraphRAG | Indexing cost не оправдан на тысячах items/user. |
| MemoryBank Ebbinghaus decay | `t_invalid` субсумирует forgetting. |
| MemGPT agent-curated memory | ТЗ запрещает agent code; уступает Mem0/Zep на LongMemEval. |
| Fine-tuning extractor | 2 дня. |
| Reflection passes (Generative Agents) | Заменено на `mention_count` как cheap importance proxy. |
| NLI cross-encoder для контрадикций | LLM-judge достаточен; NLI в README как Phase-2. |
| bge-reranker-v2-m3 | Устарел, проигрывает Jina v3 на 5+pp. |

---

## Sources

- [Mem0 paper](https://arxiv.org/abs/2504.19413)
- [Zep/Graphiti paper](https://arxiv.org/abs/2501.13956)
- [LongMemEval](https://arxiv.org/abs/2410.10813)
- [LOCOMO](https://arxiv.org/abs/2402.17753)
- [HippoRAG 2](https://openreview.net/forum?id=LWH8yn4HS2)
- [A-MEM](https://arxiv.org/abs/2502.12110)
- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Gemini Embedding 2 announcement](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- [Jina Reranker v3](https://jina.ai/models/jina-reranker-v3/)
- [F2LLM-v2-4B](https://huggingface.co/codefuse-ai/F2LLM-v2-4B)
- [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
