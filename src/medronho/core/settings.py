"""Application settings, loaded from environment variables and an optional `.env` file.

Validation fails loudly on missing required values (`VAULT_PATH`, `DB_PATH`);
everything else defaults to a local Ollama setup.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for medronho.

    Attributes:
        vault_path: Directory containing the Markdown vault. Required.
        db_path: Location of the SQLite database file. Required.
        llm_base_url: Any OpenAI-compatible endpoint; defaults to local Ollama.
        llm_api_key: Ignored by Ollama, required by cloud providers.
        llm_chat_model: Chat / tool-calling model name.
        llm_embed_model: Embedding model name.
        embed_dim: Vector dimension of ``llm_embed_model``. The index is locked
            to ``(llm_embed_model, embed_dim)``; changing either requires
            ``medronho index --rebuild``.
        embed_document_prefix: Task prefix prepended to documents at embedding
            time (never stored). nomic-embed-text is trained with these
            prefixes; set to ``""`` for models that use none.
        embed_query_prefix: Task prefix prepended to search queries at
            embedding time.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vault_path: Path
    db_path: Path

    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_chat_model: str = "qwen3.6:35b-a3b"
    llm_embed_model: str = "nomic-embed-text"

    embed_dim: int = 768
    embed_document_prefix: str = "search_document: "
    embed_query_prefix: str = "search_query: "

    @field_validator("vault_path", "db_path", mode="after")
    @classmethod
    def _expand_user(cls, value: Path) -> Path:
        return value.expanduser()
