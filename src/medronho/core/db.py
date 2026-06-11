"""SQLite + sqlite-vec storage layer.

Similarity metric: cosine, declared once in the ``vec_chunks`` schema
(``distance_metric=cosine``). Vectors are stored as produced by the embedding
model; no normalization happens anywhere in the codebase.
"""

import json
import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path

# sqlite-vec ships no type stubs; only load() is used, inferred fine from source.
import sqlite_vec  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel


class IndexCompatibilityError(RuntimeError):
    """The on-disk index was built with different embedding settings."""


class SearchResult(BaseModel):
    """One chunk returned by a vector similarity search.

    Attributes:
        path: Note path relative to the vault root.
        heading_path: Markdown heading hierarchy the chunk lives under.
        position: 0-based chunk order within the note.
        text: Chunk content (as stored, without any embedding prefix).
        distance: Cosine distance to the query (lower is closer).
    """

    path: str
    heading_path: list[str]
    position: int
    text: str
    distance: float


def connect(db_path: Path) -> sqlite3.Connection:
    """Open the database, load the sqlite-vec extension and enable WAL."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema(conn: sqlite3.Connection, *, dim: int) -> None:
    """Create all tables if missing. ``dim`` fixes the vector column dimension."""
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                heading_path TEXT NOT NULL,
                position INTEGER NOT NULL,
                text TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                embed_model TEXT NOT NULL,
                embed_dim INTEGER NOT NULL
            )
            """
        )
        # Cosine is the one similarity metric of the project, declared here and
        # nowhere else. vec_chunks rowids mirror chunks.id (vec0 supports no FK).
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{dim}] distance_metric=cosine
            )
            """
        )


def drop_schema(conn: sqlite3.Connection) -> None:
    """Drop all medronho tables (used by ``medronho index --rebuild``)."""
    with conn:
        for table in ("vec_chunks", "chunks", "files", "index_meta"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")


def check_embedding_compat(conn: sqlite3.Connection, *, model: str, dim: int) -> None:
    """Lock the index to one (embedding model, dimension) pair.

    Records the pair on first use; on later runs raises
    :class:`IndexCompatibilityError` if current settings differ, so a model
    switch can never silently corrupt search quality.
    """
    row = conn.execute("SELECT embed_model, embed_dim FROM index_meta WHERE id = 1").fetchone()
    if row is None:
        with conn:
            conn.execute(
                "INSERT INTO index_meta (id, embed_model, embed_dim) VALUES (1, ?, ?)",
                (model, dim),
            )
        return
    stored_model, stored_dim = str(row[0]), int(row[1])
    if (stored_model, stored_dim) != (model, dim):
        raise IndexCompatibilityError(
            f"Index was built with {stored_model}/{stored_dim}, current settings are "
            f"{model}/{dim}. Re-index with: medronho index --rebuild"
        )


def serialize_vector(vector: Sequence[float]) -> bytes:
    """Pack a vector into the little-endian float32 blob sqlite-vec expects."""
    return struct.pack(f"<{len(vector)}f", *vector)


def search_chunks(
    conn: sqlite3.Connection, embedding: Sequence[float], *, k: int = 5
) -> list[SearchResult]:
    """Return the ``k`` nearest chunks to ``embedding`` by cosine distance."""
    rows = conn.execute(
        """
        SELECT c.path, c.heading_path, c.position, c.text, v.distance
        FROM (
            SELECT rowid, distance
            FROM vec_chunks
            WHERE embedding MATCH ? AND k = ?
        ) AS v
        JOIN chunks c ON c.id = v.rowid
        ORDER BY v.distance
        """,
        (serialize_vector(embedding), k),
    ).fetchall()
    return [
        SearchResult(
            path=str(row[0]),
            heading_path=json.loads(str(row[1])),
            position=int(row[2]),
            text=str(row[3]),
            distance=float(row[4]),
        )
        for row in rows
    ]
