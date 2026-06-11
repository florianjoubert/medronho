"""Tests for incremental indexing: idempotency, change detection, meta lock."""

import sqlite3
from pathlib import Path

import pytest

from medronho.core.db import IndexCompatibilityError, search_chunks
from medronho.core.settings import Settings
from medronho.ingest.indexer import index_vault

from .conftest import FakeEmbedder, make_settings


def _write_vault(vault: Path) -> None:
    (vault / "people").mkdir()
    (vault / "a.md").write_text("# Alpha topic\n\nEverything about alpha.\n")
    (vault / "b.md").write_text("# Bravo topic\n\nEverything about bravo.\n")
    (vault / "people" / "c.md").write_text("# Charlie\n\nNotes about charlie.\n")


def _chunk_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT count(*) FROM chunks").fetchone()[0])


def test_initial_index(
    conn: sqlite3.Connection, settings: Settings, vault: Path, fake_embedder: FakeEmbedder
) -> None:
    _write_vault(vault)
    stats = index_vault(conn, settings, fake_embedder)
    assert stats.scanned == 3
    assert stats.indexed == 3
    assert stats.skipped == 0
    assert _chunk_count(conn) == 3
    assert int(conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]) == 3


def test_reindex_is_idempotent(
    conn: sqlite3.Connection, settings: Settings, vault: Path, fake_embedder: FakeEmbedder
) -> None:
    _write_vault(vault)
    index_vault(conn, settings, fake_embedder)
    calls_after_first = fake_embedder.document_calls

    stats = index_vault(conn, settings, fake_embedder)

    assert stats.indexed == 0
    assert stats.skipped == 3
    assert fake_embedder.document_calls == calls_after_first  # zero new embed calls
    assert _chunk_count(conn) == 3


def test_modified_file_is_reindexed(
    conn: sqlite3.Connection, settings: Settings, vault: Path, fake_embedder: FakeEmbedder
) -> None:
    _write_vault(vault)
    index_vault(conn, settings, fake_embedder)
    (vault / "a.md").write_text("# Alpha topic\n\nNew alpha content entirely.\n")

    stats = index_vault(conn, settings, fake_embedder)

    assert stats.indexed == 1
    assert stats.skipped == 2
    row = conn.execute("SELECT text FROM chunks WHERE path = 'a.md'").fetchone()
    assert "New alpha content" in row[0]


def test_deleted_file_is_purged(
    conn: sqlite3.Connection, settings: Settings, vault: Path, fake_embedder: FakeEmbedder
) -> None:
    _write_vault(vault)
    index_vault(conn, settings, fake_embedder)
    (vault / "b.md").unlink()

    stats = index_vault(conn, settings, fake_embedder)

    assert stats.removed == 1
    assert conn.execute("SELECT count(*) FROM chunks WHERE path = 'b.md'").fetchone()[0] == 0
    assert _chunk_count(conn) == 2
    assert int(conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0]) == 2


def test_malformed_file_is_skipped_not_fatal(
    conn: sqlite3.Connection, settings: Settings, vault: Path, fake_embedder: FakeEmbedder
) -> None:
    _write_vault(vault)
    (vault / "broken.md").write_bytes(b"\xff\xfe invalid \xff utf-8")

    stats = index_vault(conn, settings, fake_embedder)

    assert stats.errors == 1
    assert stats.indexed == 3  # the valid files still made it


def test_search_ranking_is_provable(
    conn: sqlite3.Connection, settings: Settings, vault: Path, fake_embedder: FakeEmbedder
) -> None:
    # Each note carries exactly one vocabulary keyword (orthogonal vectors):
    # the note sharing the query's keyword is necessarily the nearest.
    _write_vault(vault)
    index_vault(conn, settings, fake_embedder)

    results = search_chunks(conn, fake_embedder.embed_query("tell me about bravo"), k=3)

    assert len(results) == 3
    assert results[0].path == "b.md"
    assert results[0].distance < results[1].distance


def test_meta_lock_blocks_model_change_and_rebuild_recovers(
    conn: sqlite3.Connection, settings: Settings, vault: Path, fake_embedder: FakeEmbedder
) -> None:
    _write_vault(vault)
    index_vault(conn, settings, fake_embedder)

    changed = make_settings(
        settings.vault_path, settings.db_path.parent, llm_embed_model="other-model"
    )
    with pytest.raises(IndexCompatibilityError, match="medronho index --rebuild"):
        index_vault(conn, changed, fake_embedder)

    stats = index_vault(conn, changed, fake_embedder, rebuild=True)
    assert stats.indexed == 3
    assert _chunk_count(conn) == 3
    row = conn.execute("SELECT embed_model FROM index_meta WHERE id = 1").fetchone()
    assert row[0] == "other-model"
