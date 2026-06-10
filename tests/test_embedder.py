"""Tests for OpenAIEmbedder: task prefixes and batching."""

from types import SimpleNamespace
from typing import cast

from openai import OpenAI

from secondbrain.ingest.embedder import OpenAIEmbedder


class _RecordingEmbeddingsAPI:
    """Stands in for client.embeddings; records every `create` call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | str]] = []

    def create(self, *, model: str, input: list[str] | str) -> SimpleNamespace:
        self.calls.append((model, input))
        n = len(input) if isinstance(input, list) else 1
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in range(n)])


def _embedder(batch_size: int = 64) -> tuple[OpenAIEmbedder, _RecordingEmbeddingsAPI]:
    api = _RecordingEmbeddingsAPI()
    # cast: a minimal structural stand-in for the OpenAI client is all the
    # embedder touches; a real client would require a live endpoint.
    client = cast(OpenAI, SimpleNamespace(embeddings=api))
    embedder = OpenAIEmbedder(
        client,
        model="test-model",
        document_prefix="search_document: ",
        query_prefix="search_query: ",
        batch_size=batch_size,
    )
    return embedder, api


def test_document_prefix_prepended() -> None:
    embedder, api = _embedder()
    embedder.embed_documents(["chunk one", "chunk two"])
    (model, sent) = api.calls[0]
    assert model == "test-model"
    assert sent == ["search_document: chunk one", "search_document: chunk two"]


def test_query_prefix_prepended() -> None:
    embedder, api = _embedder()
    vector = embedder.embed_query("what is alpha?")
    assert api.calls == [("test-model", "search_query: what is alpha?")]
    assert vector == [0.1, 0.2]


def test_empty_prefixes_supported() -> None:
    api = _RecordingEmbeddingsAPI()
    client = cast(OpenAI, SimpleNamespace(embeddings=api))  # cast: see _embedder
    embedder = OpenAIEmbedder(client, model="m", document_prefix="", query_prefix="")
    embedder.embed_documents(["raw text"])
    assert api.calls[0][1] == ["raw text"]


def test_batching_respects_batch_size() -> None:
    embedder, api = _embedder(batch_size=2)
    vectors = embedder.embed_documents([f"t{i}" for i in range(5)])
    assert [len(cast(list[str], sent)) for _, sent in api.calls] == [2, 2, 1]
    assert len(vectors) == 5
