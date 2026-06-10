"""Incremental, idempotent vault indexing.

Walks the vault, hashes every Markdown file, and (re-)embeds only what
changed. A malformed file is logged and skipped, never fatal. The index is
locked to one (embedding model, dimension) pair — see
:func:`secondbrain.core.db.check_embedding_compat`.
"""

import hashlib
import json
import logging
import sqlite3

from pydantic import BaseModel

from secondbrain.core.db import (
    check_embedding_compat,
    drop_schema,
    init_schema,
    serialize_vector,
)
from secondbrain.core.settings import Settings
from secondbrain.ingest.chunker import Chunk, chunk_markdown
from secondbrain.ingest.embedder import Embedder

logger = logging.getLogger(__name__)


class IndexStats(BaseModel):
    """Outcome of one indexing run.

    Attributes:
        scanned: Markdown files found in the vault.
        indexed: Files (re-)chunked and (re-)embedded.
        skipped: Files unchanged since the last run.
        removed: Files purged because they disappeared from the vault.
        errors: Files skipped because they could not be read or decoded.
    """

    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    errors: int = 0


def index_vault(
    conn: sqlite3.Connection,
    settings: Settings,
    embedder: Embedder,
    *,
    rebuild: bool = False,
) -> IndexStats:
    """Index the vault incrementally; with ``rebuild`` drop everything first."""
    if rebuild:
        drop_schema(conn)
    init_schema(conn, dim=settings.embed_dim)
    check_embedding_compat(conn, model=settings.llm_embed_model, dim=settings.embed_dim)

    stats = IndexStats()
    known: dict[str, str] = {
        str(path): str(content_hash)
        for path, content_hash in conn.execute("SELECT path, content_hash FROM files")
    }
    seen: set[str] = set()

    for md_file in sorted(settings.vault_path.rglob("*.md")):
        rel_path = md_file.relative_to(settings.vault_path).as_posix()
        seen.add(rel_path)
        stats.scanned += 1
        try:
            raw = md_file.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if known.get(rel_path) == digest:
                stats.skipped += 1
                continue
            _index_file(conn, embedder, rel_path, raw.decode("utf-8"), digest)
            stats.indexed += 1
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Skipping %s: %s", rel_path, exc)
            stats.errors += 1

    for rel_path in sorted(known.keys() - seen):
        with conn:
            _delete_file(conn, rel_path)
        stats.removed += 1
    return stats


def _index_file(
    conn: sqlite3.Connection,
    embedder: Embedder,
    rel_path: str,
    text: str,
    digest: str,
) -> None:
    """Replace one file's chunks and vectors atomically."""
    chunks = chunk_markdown(text)
    # The heading path is part of the embedded text (it carries real signal)
    # but stored fields stay clean of any embedding-time decoration.
    vectors = embedder.embed_documents([_embed_input(chunk) for chunk in chunks])
    with conn:
        _delete_file(conn, rel_path)
        conn.execute("INSERT INTO files (path, content_hash) VALUES (?, ?)", (rel_path, digest))
        for chunk, vector in zip(chunks, vectors, strict=True):
            cursor = conn.execute(
                "INSERT INTO chunks (path, heading_path, position, text) VALUES (?, ?, ?, ?)",
                (rel_path, json.dumps(chunk.heading_path), chunk.position, chunk.text),
            )
            conn.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                (cursor.lastrowid, serialize_vector(vector)),
            )


def _embed_input(chunk: Chunk) -> str:
    """Text actually sent to the embedder: heading context + chunk content."""
    if not chunk.heading_path:
        return chunk.text
    return f"{' > '.join(chunk.heading_path)}\n\n{chunk.text}"


def _delete_file(conn: sqlite3.Connection, rel_path: str) -> None:
    """Delete one file's rows from files, chunks and vec_chunks."""
    conn.execute(
        "DELETE FROM vec_chunks WHERE rowid IN (SELECT id FROM chunks WHERE path = ?)",
        (rel_path,),
    )
    conn.execute("DELETE FROM chunks WHERE path = ?", (rel_path,))
    conn.execute("DELETE FROM files WHERE path = ?", (rel_path,))
