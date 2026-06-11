"""Tests for the SQLite + sqlite-vec layer: schema, KNN mechanics, meta lock."""

import json
import math
import sqlite3

import pytest

from medronho.core.db import (
    IndexCompatibilityError,
    check_embedding_compat,
    init_schema,
    search_chunks,
    serialize_vector,
)

DIM = 3


def _insert_chunk(
    conn: sqlite3.Connection, path: str, text: str, vector: list[float], position: int = 0
) -> None:
    cursor = conn.execute(
        "INSERT INTO chunks (path, heading_path, position, text) VALUES (?, ?, ?, ?)",
        (path, json.dumps(["H"]), position, text),
    )
    conn.execute(
        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
        (cursor.lastrowid, serialize_vector(vector)),
    )


def test_knn_mechanics_and_cosine_ranking(conn: sqlite3.Connection) -> None:
    init_schema(conn, dim=DIM)
    with conn:
        _insert_chunk(conn, "a.md", "about A", [1.0, 0.0, 0.0])
        _insert_chunk(conn, "b.md", "about B", [0.0, 1.0, 0.0])
        _insert_chunk(conn, "c.md", "about C", [0.0, 0.0, 1.0])

    results = search_chunks(conn, [1.0, 0.1, 0.0], k=2)

    assert len(results) == 2  # k is honored
    top = results[0]
    assert top.path == "a.md"  # cosine: collinear vector wins
    assert top.text == "about A"
    assert top.heading_path == ["H"]
    assert top.position == 0
    # Exact cosine distance, 1 - 1/sqrt(1.01): proves the metric is cosine
    # (under plain L2 the distance to [1,0,0] would be 0.1).
    assert math.isclose(top.distance, 1 - 1 / math.sqrt(1.01), rel_tol=1e-4)
    assert results[0].distance <= results[1].distance


def test_search_on_empty_index_returns_nothing(conn: sqlite3.Connection) -> None:
    init_schema(conn, dim=DIM)
    assert search_chunks(conn, [1.0, 0.0, 0.0], k=5) == []


def test_meta_lock_records_then_rejects_mismatch(conn: sqlite3.Connection) -> None:
    init_schema(conn, dim=DIM)
    check_embedding_compat(conn, model="nomic-embed-text", dim=DIM)
    # Same pair: fine.
    check_embedding_compat(conn, model="nomic-embed-text", dim=DIM)
    with pytest.raises(IndexCompatibilityError) as excinfo:
        check_embedding_compat(conn, model="other-model", dim=512)
    message = str(excinfo.value)
    assert f"nomic-embed-text/{DIM}" in message
    assert "other-model/512" in message
    assert "medronho index --rebuild" in message
