"""Shared fixtures: controllable fake embedder, settings and database factories."""

import math
import sqlite3
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from medronho.core.db import connect
from medronho.core.settings import Settings

# Each keyword maps to its own vector component (orthogonal base vectors), so
# similarity is controllable: a query sharing a keyword with exactly one chunk
# is provably nearest under cosine. The constant bias component guarantees no
# vector is ever null.
VOCAB: tuple[str, ...] = ("alpha", "bravo", "charlie", "delta")
FAKE_DIM = len(VOCAB) + 1
_BIAS = 0.25


class FakeEmbedder:
    """Deterministic embedder with controllable similarity and call counting."""

    def __init__(self) -> None:
        self.document_calls = 0
        self.documents_embedded = 0

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        components = [1.0 if keyword in lowered else 0.0 for keyword in VOCAB]
        components.append(_BIAS)
        norm = math.sqrt(sum(c * c for c in components))
        return [c / norm for c in components]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        self.documents_embedded += len(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    path = tmp_path / "vault"
    path.mkdir()
    return path


@pytest.fixture
def settings(vault: Path, tmp_path: Path) -> Settings:
    return make_settings(vault, tmp_path)


def make_settings(vault: Path, tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "vault_path": vault,
        "db_path": tmp_path / "test.db",
        "embed_dim": FAKE_DIM,
        **overrides,
    }
    # _env_file=None: tests must never pick up a developer's real .env.
    return Settings(_env_file=None, **values)  # pyright: ignore[reportCallIssue]


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    connection = connect(settings.db_path)
    yield connection
    connection.close()
