"""Neon Postgres schema management for Chainlit's data layer.

Chainlit persists users, threads, steps, elements and feedback itself, so we use
its schema rather than a parallel one of our own. That buys chat history, thread
resume and the feedback UI for free; the cost is that the shape is Chainlit's,
not ours.

The tables (`users`, `threads`, `steps`, `elements`, `feedbacks`) come verbatim
from https://docs.chainlit.io/data-layers/sqlalchemy. The quoted camelCase column
names are load-bearing -- Chainlit's queries reference them exactly, and
unquoted identifiers would be folded to lowercase by Postgres and stop matching.

What we would otherwise have lost -- token counts, cost, and the pinned config
that produced an answer -- is written into `steps.metadata` by the app, so
per-answer accounting survives the switch.

Usage:
    python src/db.py --init      # create Chainlit tables (safe to re-run)
    python src/db.py --check     # connectivity, tables, row counts, spend
    python src/db.py --drop      # destructive; asks for confirmation
"""

import argparse
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row

load_dotenv()

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("logs/db.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

CHAINLIT_TABLES = ("users", "threads", "steps", "elements", "feedbacks")
RETIRED_TABLES = ("feedback", "turns", "sessions")  # our own first-pass schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id"         UUID PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata"   JSONB NOT NULL,
    "createdAt"  TEXT
);

CREATE TABLE IF NOT EXISTS threads (
    "id"             UUID PRIMARY KEY,
    "createdAt"      TEXT,
    "name"           TEXT,
    "userId"         UUID,
    "userIdentifier" TEXT,
    "tags"           TEXT[],
    "metadata"       JSONB,
    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS steps (
    "id"            UUID PRIMARY KEY,
    "name"          TEXT NOT NULL,
    "type"          TEXT NOT NULL,
    "threadId"      UUID NOT NULL,
    "parentId"      UUID,
    "streaming"     BOOLEAN NOT NULL,
    "waitForAnswer" BOOLEAN,
    "isError"       BOOLEAN,
    "metadata"      JSONB,
    "tags"          TEXT[],
    "input"         TEXT,
    "output"        TEXT,
    "createdAt"     TEXT,
    "command"       TEXT,
    "start"         TEXT,
    "end"           TEXT,
    "generation"    JSONB,
    "showInput"     TEXT,
    "language"      TEXT,
    "indent"        INT,
    "defaultOpen"   BOOLEAN,
    "modes"         JSONB,
    -- NOT in the published DDL at docs.chainlit.io/data-layers/sqlalchemy, but
    -- chainlit 2.11.1 writes them. Without "autoCollapse" every step INSERT
    -- fails with UndefinedColumnError -- and it fails SILENTLY, because
    -- Step.send() persists via a fire-and-forget asyncio task that only logs a
    -- warning. The UI looks perfect while nothing is saved. Derived from
    -- chainlit.step.StepDict rather than from the docs.
    "autoCollapse"  BOOLEAN,
    "icon"          TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS elements (
    "id"          UUID PRIMARY KEY,
    "threadId"    UUID,
    "type"        TEXT,
    "url"         TEXT,
    "chainlitKey" TEXT,
    "name"        TEXT NOT NULL,
    "display"     TEXT,
    "objectKey"   TEXT,
    "size"        TEXT,
    "page"        INT,
    "language"    TEXT,
    "forId"       UUID,
    "mime"        TEXT,
    "props"       JSONB,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedbacks (
    "id"       UUID PRIMARY KEY,
    "forId"    UUID NOT NULL,
    "threadId" UUID NOT NULL,
    "value"    INT NOT NULL,
    "comment"  TEXT,
    FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
);

-- Ours, not Chainlit's: cost reporting scans step metadata by thread and time.
CREATE INDEX IF NOT EXISTS ix_steps_thread  ON steps ("threadId");
CREATE INDEX IF NOT EXISTS ix_steps_created ON steps ("createdAt" DESC);
"""


def sync_dsn() -> str:
    """Connection string for psycopg (schema management)."""
    dsn = os.environ.get("NEON_DB_STRING")
    if not dsn:
        raise RuntimeError("NEON_DB_STRING is not set. Add it to .env")
    return dsn


def async_dsn() -> str:
    """Connection string for SQLAlchemy + asyncpg, which Chainlit's data layer needs.

    asyncpg does not accept libpq's `sslmode` or `channel_binding` DSN
    parameters -- passing them through raises `unexpected keyword argument`. It
    wants `ssl=` instead, so translate rather than forward.
    """
    parts = urlsplit(sync_dsn())
    query = dict(parse_qsl(parts.query))
    sslmode = query.pop("sslmode", None)
    query.pop("channel_binding", None)
    if sslmode in (None, "require", "verify-ca", "verify-full", "prefer"):
        query["ssl"] = "require"
    return urlunsplit(("postgresql+asyncpg", parts.netloc, parts.path,
                       urlencode(query), ""))


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    with psycopg.connect(sync_dsn(), row_factory=dict_row) as conn:
        yield conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)
    logger.info("Chainlit schema applied: %s", ", ".join(CHAINLIT_TABLES))


def drop_db(include_retired: bool = True) -> None:
    with connect() as conn:
        conn.execute(
            'DROP TABLE IF EXISTS feedbacks, elements, steps, threads, users CASCADE;'
        )
        if include_retired:
            conn.execute("DROP TABLE IF EXISTS feedback, turns, sessions CASCADE;")
    logger.warning("Dropped Chainlit tables%s",
                   " and the retired first-pass tables" if include_retired else "")


def check() -> None:
    with connect() as conn:
        version = conn.execute("SELECT version()").fetchone()["version"]
        logger.info("Connected: %s", version.split(",")[0])

        rows = conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name
            """
        ).fetchall()
        present = {r["table_name"] for r in rows}

        for table in CHAINLIT_TABLES:
            if table not in present:
                logger.warning("  %-10s MISSING -- run --init", table)
                continue
            count = conn.execute(f'SELECT count(*) AS n FROM "{table}"').fetchone()["n"]
            logger.info("  %-10s %d rows", table, count)

        leftover = [t for t in RETIRED_TABLES if t in present]
        if leftover:
            logger.warning("Retired tables still present: %s "
                           "(drop them once you are sure nothing needs the data)",
                           ", ".join(leftover))

        if "steps" in present:
            # Cost lives in step metadata, written by the app rather than Chainlit.
            agg = conn.execute(
                """
                SELECT count(*) AS n,
                       coalesce(sum((metadata->>'cost_usd')::numeric), 0) AS cost,
                       coalesce(sum((metadata->>'input_tokens')::int), 0)  AS tin,
                       coalesce(sum((metadata->>'output_tokens')::int), 0) AS tout
                FROM steps
                WHERE metadata ? 'cost_usd'
                """
            ).fetchone()
            logger.info("Answered steps: %d | spend $%.6f | %s in / %s out tokens",
                        agg["n"], agg["cost"], f"{agg['tin']:,}", f"{agg['tout']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Neon schema management (Chainlit data layer)")
    parser.add_argument("--init", action="store_true", help="Create tables (safe to re-run)")
    parser.add_argument("--check", action="store_true", help="Connectivity and row counts")
    parser.add_argument("--drop", action="store_true", help="DROP all tables (destructive)")
    parser.add_argument("--dsn", action="store_true", help="Print the async DSN (masked)")
    args = parser.parse_args()

    if not any((args.init, args.check, args.drop, args.dsn)):
        parser.print_help()
        sys.exit(1)

    if args.drop:
        print("This DROPS the Chainlit tables and all chat history.")
        if input("Type 'drop' to confirm: ").strip() != "drop":
            print("Aborted.")
            sys.exit(1)
        drop_db()
    if args.init:
        init_db()
    if args.check:
        check()
    if args.dsn:
        dsn = async_dsn()
        head, _, tail = dsn.partition("@")
        logger.info("async DSN: %s://***@%s", head.split("://")[0], tail)


if __name__ == "__main__":
    main()
