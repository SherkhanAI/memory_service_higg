-- Memory service schema v0.1
-- Bi-temporal facts + append-only episodic log + entity index for multi-hop.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

DO $$ BEGIN
    CREATE TYPE memory_kind AS ENUM ('fact','preference','opinion','event');
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE memory_confidence AS ENUM ('low','med','high');
EXCEPTION WHEN duplicate_object THEN null; END $$;


CREATE TABLE IF NOT EXISTS episodic_turn (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      TEXT NOT NULL,
    user_id         TEXT,
    messages        JSONB NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    raw_text        TEXT NOT NULL DEFAULT '',
    context_prefix  TEXT,
    embedding       vector(1536),
    tsv             tsvector
);

CREATE INDEX IF NOT EXISTS episodic_turn_session_idx
    ON episodic_turn (session_id);
CREATE INDEX IF NOT EXISTS episodic_turn_user_ts_idx
    ON episodic_turn (user_id, ts DESC);
CREATE INDEX IF NOT EXISTS episodic_turn_tsv_idx
    ON episodic_turn USING GIN (tsv);
CREATE INDEX IF NOT EXISTS episodic_turn_embedding_idx
    ON episodic_turn USING hnsw (embedding vector_cosine_ops);


CREATE TABLE IF NOT EXISTS memory (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           TEXT NOT NULL,
    session_scope     TEXT,
    subject           TEXT NOT NULL DEFAULT 'user',
    predicate         TEXT NOT NULL,
    object_text       TEXT NOT NULL,
    object_qualifiers JSONB NOT NULL DEFAULT '{}'::jsonb,
    kind              memory_kind NOT NULL DEFAULT 'fact',
    stance            TEXT,
    confidence        memory_confidence NOT NULL DEFAULT 'med',
    is_implicit       BOOLEAN NOT NULL DEFAULT false,
    -- bi-temporal
    t_valid           TIMESTAMPTZ NOT NULL,
    t_invalid         TIMESTAMPTZ,
    t_created         TIMESTAMPTZ NOT NULL DEFAULT now(),
    t_expired         TIMESTAMPTZ,
    -- provenance
    source_turn_id    UUID REFERENCES episodic_turn(id) ON DELETE CASCADE,
    source_text       TEXT,
    -- evolution
    superseded_by     UUID REFERENCES memory(id) ON DELETE SET NULL,
    mention_count     INT NOT NULL DEFAULT 1,
    -- search
    embedding         vector(1536),
    tsv               tsvector
);

CREATE INDEX IF NOT EXISTS memory_user_predicate_idx
    ON memory (user_id, predicate, t_invalid);
CREATE INDEX IF NOT EXISTS memory_user_kind_idx
    ON memory (user_id, kind, t_invalid);
CREATE INDEX IF NOT EXISTS memory_source_idx
    ON memory (source_turn_id);
CREATE INDEX IF NOT EXISTS memory_session_scope_idx
    ON memory (user_id, session_scope, t_invalid);
CREATE INDEX IF NOT EXISTS memory_tsv_idx
    ON memory USING GIN (tsv);
CREATE INDEX IF NOT EXISTS memory_embedding_idx
    ON memory USING hnsw (embedding vector_cosine_ops);


CREATE TABLE IF NOT EXISTS entity (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         TEXT NOT NULL,
    type            TEXT NOT NULL,
    name            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    first_seen_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding       vector(1536),
    UNIQUE (user_id, type, normalized_name)
);
CREATE INDEX IF NOT EXISTS entity_user_type_idx ON entity (user_id, type);
CREATE INDEX IF NOT EXISTS entity_embedding_idx
    ON entity USING hnsw (embedding vector_cosine_ops);


CREATE TABLE IF NOT EXISTS memory_entity (
    memory_id  UUID NOT NULL REFERENCES memory(id) ON DELETE CASCADE,
    entity_id  UUID NOT NULL REFERENCES entity(id) ON DELETE CASCADE,
    role       TEXT NOT NULL DEFAULT 'object',
    PRIMARY KEY (memory_id, entity_id, role)
);
CREATE INDEX IF NOT EXISTS memory_entity_entity_idx
    ON memory_entity (entity_id);
