"""Embedding backends behind a small protocol.

Task prefixes: nomic-embed-text (the default model) is trained with
instruction prefixes, and retrieval quality degrades measurably without them —
``"search_document: "`` for indexed content, ``"search_query: "`` for queries.
The pair is configurable (``EMBED_DOCUMENT_PREFIX`` / ``EMBED_QUERY_PREFIX``)
because other embedding models use different prefixes or none; set both to an
empty string in that case. Prefixes are an embedding-time concern only: they
are prepended here, right before the API call, and are never stored alongside
chunks.
"""

from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI

from secondbrain.core.llm import make_client
from secondbrain.core.settings import Settings

_BATCH_SIZE = 64


class Embedder(Protocol):
    """Anything that can embed documents for indexing and queries for search."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed indexable content (documents/chunks)."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (used by semantic search, Session 2+)."""
        ...


class OpenAIEmbedder:
    """Embedder backed by any OpenAI-compatible endpoint (Ollama by default).

    Applies the document/query task prefixes at call time; see module
    docstring for why they exist and why they are configurable.
    """

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        document_prefix: str,
        query_prefix: str,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        """Wrap ``client`` for embedding calls against ``model``."""
        self._client = client
        self._model = model
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._batch_size = batch_size

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAIEmbedder":
        """Build an embedder wired to the configured endpoint, model and prefixes."""
        return cls(
            make_client(settings),
            model=settings.llm_embed_model,
            document_prefix=settings.embed_document_prefix,
            query_prefix=settings.embed_query_prefix,
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents in batches, with the document prefix prepended."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=[f"{self._document_prefix}{text}" for text in batch],
            )
            vectors.extend(item.embedding for item in response.data)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query, with the query prefix prepended."""
        response = self._client.embeddings.create(
            model=self._model,
            input=f"{self._query_prefix}{text}",
        )
        return response.data[0].embedding
