"""Neon Postgres persistence: sessions, turns, feedback.

Mirrors the shape used in the Washington project, adapted from text-to-SQL to
document retrieval: the SQL/columns/rows/chart columns are replaced by
`retrieved` (which chunks were used and how well they matched).

Conventions, all deliberate:
  - timestamps are TIMESTAMPTZ (timezone-aware UTC), never naive
  - cost is NUMERIC(12,6) -- exact decimal, because float accumulates error over
    thousands of small charges
  - JSON-shaped fields are JSONB, so they can be queried and indexed
  - identity is the composite `(session_id, turn_idx)`

`config_snapshot` records the model, temperature and retrieval settings that
produced each answer. Without it a stored answer cannot be reproduced or fairly
compared against a later run.

Usage:
    python src/db.py --init      # create tables (safe to re-run)
    python src/db.py --check     # connectivity + row counts
    python src/db.py --drop      # destructive; asks for confirmation
"""

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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

TABLES = ("sessions", "turns", "feedback")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    label        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS turns (
    session_id       TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    turn_idx         INTEGER NOT NULL,
    role             TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT    NOT NULL,

    -- Which chunks the retriever surfaced, as
    -- [{chunk_id, post_id, url, title, distance}, ...]. Empty on user turns.
    -- Stored as a snapshot so history replays what was actually used even after
    -- the corpus is re-chunked.
    retrieved        JSONB   NOT NULL DEFAULT '[]'::jsonb,

    -- Per-LLM-call detail: [{model, role, input_tokens, output_tokens,
    -- cost_usd, latency_ms}, ...]. The columns below aggregate these.
    call_records     JSONB   NOT NULL DEFAULT '[]'::jsonb,

    -- The pinned settings that produced this answer, so a stored turn can be
    -- reproduced and compared against a later configuration.
    config_snapshot  JSONB,

    model            TEXT,
    input_tokens     INTEGER       NOT NULL DEFAULT 0,
    output_tokens    INTEGER       NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER       NOT NULL DEFAULT 0,
    cached_tokens    INTEGER       NOT NULL DEFAULT 0,
    cost_usd         NUMERIC(12,6) NOT NULL DEFAULT 0,
    latency_ms       INTEGER       NOT NULL DEFAULT 0,

    created_at       TIMESTAMPTZ   NOT NULL DEFAULT now(),

    PRIMARY KEY (session_id, turn_idx)
);

CREATE TABLE IF NOT EXISTS feedback (
    session_id  TEXT    NOT NULL,
    turn_idx    INTEGER NOT NULL,
    rating      SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, turn_idx),
    FOREIGN KEY (session_id, turn_idx)
        REFERENCES turns(session_id, turn_idx) ON DELETE CASCADE
);

-- Cost and volume reporting is time-bucketed, and negative feedback is the
-- first thing anyone looks for.
CREATE INDEX IF NOT EXISTS ix_turns_created  ON turns (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_turns_model    ON turns (model);
CREATE INDEX IF NOT EXISTS ix_feedback_rating ON feedback (rating);
"""


def connection_string() -> str:
    dsn = os.environ.get("NEON_DB_STRING")
    if not dsn:
        raise RuntimeError("NEON_DB_STRING is not set. Add it to .env")
    return dsn


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Open a connection. Commits on clean exit, rolls back on exception."""
    with psycopg.connect(connection_string(), row_factory=dict_row) as conn:
        yield conn


def init_db() -> None:
    """Create tables and indexes. Safe to re-run."""
    with connect() as conn:
        conn.execute(SCHEMA)
    logger.info("Schema applied: %s", ", ".join(TABLES))


def drop_db() -> None:
    """Drop every table. Destructive."""
    with connect() as conn:
        conn.execute("DROP TABLE IF EXISTS feedback, turns, sessions CASCADE;")
    logger.warning("Dropped tables: %s", ", ".join(TABLES))


def check() -> None:
    """Report server version, tables present, row counts and spend to date."""
    with connect() as conn:
        version = conn.execute("SELECT version()").fetchone()["version"]
        logger.info("Connected: %s", version.split(",")[0])

        rows = conn.execute(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' ORDER BY table_name
            """
        ).fetchall()
        present = [r["table_name"] for r in rows]
        logger.info("Tables in public: %s", present or "(none)")

        for table in TABLES:
            if table not in present:
                logger.warning("  %-10s MISSING -- run --init", table)
                continue
            count = conn.execute(f"SELECT count(*) AS n FROM {table}").fetchone()["n"]
            logger.info("  %-10s %d rows", table, count)

        if "turns" in present:
            agg = conn.execute(
                """
                SELECT coalesce(sum(cost_usd), 0)      AS cost,
                       coalesce(sum(input_tokens), 0)  AS tin,
                       coalesce(sum(output_tokens), 0) AS tout
                FROM turns
                """
            ).fetchone()
            logger.info("Spend to date: $%.6f  (%s in / %s out tokens)",
                        agg["cost"], f"{agg['tin']:,}", f"{agg['tout']:,}")


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def create_session(session_id: str, label: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions (session_id, label) VALUES (%s, %s)
            ON CONFLICT (session_id) DO NOTHING
            """,
            (session_id, label),
        )


def record_turn(
    session_id: str,
    turn_idx: int,
    role: str,
    content: str,
    retrieved: list[dict[str, Any]] | None = None,
    call_records: list[dict[str, Any]] | None = None,
    config_snapshot: dict[str, Any] | None = None,
    model: str | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_tokens: int = 0,
    cost_usd: float = 0.0,
    latency_ms: int = 0,
) -> None:
    """Insert one turn and touch the session's updated_at."""
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO turns (
                session_id, turn_idx, role, content, retrieved, call_records,
                config_snapshot, model, input_tokens, output_tokens,
                reasoning_tokens, cached_tokens, cost_usd, latency_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (session_id, turn_idx) DO NOTHING
            """,
            (
                session_id, turn_idx, role, content,
                json.dumps(retrieved or [], ensure_ascii=False),
                json.dumps(call_records or [], ensure_ascii=False),
                json.dumps(config_snapshot, ensure_ascii=False) if config_snapshot else None,
                model, input_tokens, output_tokens, reasoning_tokens,
                cached_tokens, cost_usd, latency_ms,
            ),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = now() WHERE session_id = %s",
            (session_id,),
        )


def record_feedback(session_id: str, turn_idx: int, rating: int,
                    comment: str | None = None) -> None:
    if rating not in (-1, 1):
        raise ValueError(f"rating must be -1 or 1, got {rating}")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO feedback (session_id, turn_idx, rating, comment)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id, turn_idx)
            DO UPDATE SET rating = EXCLUDED.rating,
                          comment = EXCLUDED.comment,
                          created_at = now()
            """,
            (session_id, turn_idx, rating, comment),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Neon schema management")
    parser.add_argument("--init", action="store_true", help="Create tables (safe to re-run)")
    parser.add_argument("--check", action="store_true", help="Connectivity and row counts")
    parser.add_argument("--drop", action="store_true", help="DROP all tables (destructive)")
    args = parser.parse_args()

    if not (args.init or args.check or args.drop):
        parser.print_help()
        sys.exit(1)

    if args.drop:
        print("This DROPS sessions, turns and feedback with all their data.")
        if input("Type 'drop' to confirm: ").strip() != "drop":
            print("Aborted.")
            sys.exit(1)
        drop_db()
    if args.init:
        init_db()
    if args.check:
        check()


if __name__ == "__main__":
    main()
