from __future__ import annotations

import logging
from pathlib import Path

import psycopg
from psycopg_pool import AsyncConnectionPool
from pgvector.psycopg import register_vector_async

from .config import settings

log = logging.getLogger(__name__)

_pool: AsyncConnectionPool | None = None


async def _bootstrap_extensions() -> None:
    """Ensure required Postgres extensions exist before opening the vector-aware pool.

    ``register_vector_async`` (called per pool connection) needs the ``vector``
    type to be already registered server-side, so we cannot rely on a migration
    inside the pool to create it — that's chicken-and-egg. Open a bare
    connection, create the extensions, close, then start the pool.
    """
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        async with conn.cursor() as cur:
            await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
            await cur.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')
        await conn.commit()
    log.info("postgres extensions ready")


async def _configure(conn) -> None:
    await register_vector_async(conn)


async def init_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is not None:
        return _pool
    await _bootstrap_extensions()
    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        configure=_configure,
        open=False,
    )
    await _pool.open()
    await _pool.wait()
    log.info("DB pool ready")
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised")
    return _pool


async def run_migrations() -> None:
    """Idempotent: applies every .sql file in src/migrations in lexical order."""
    migrations_dir = Path(__file__).parent / "migrations"
    files = sorted(p for p in migrations_dir.glob("*.sql"))
    if not files:
        log.warning("no migrations found")
        return
    async with pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    name TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT now()
                )
                """
            )
            await conn.commit()
            for f in files:
                await cur.execute(
                    "SELECT 1 FROM schema_migrations WHERE name = %s", (f.name,)
                )
                if await cur.fetchone():
                    continue
                log.info("applying migration %s", f.name)
                sql = f.read_text(encoding="utf-8")
                await cur.execute(sql)
                await cur.execute(
                    "INSERT INTO schema_migrations(name) VALUES (%s)", (f.name,)
                )
                await conn.commit()
