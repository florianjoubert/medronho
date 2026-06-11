"""Tests for settings loading and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from medronho.core.settings import Settings

from .conftest import make_settings


def test_missing_required_values_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VAULT_PATH", raising=False)
    monkeypatch.delenv("DB_PATH", raising=False)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    message = str(excinfo.value)
    assert "vault_path" in message
    assert "db_path" in message


def test_defaults_target_local_ollama(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, tmp_path)
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_embed_model == "nomic-embed-text"
    assert settings.embed_document_prefix == "search_document: "
    assert settings.embed_query_prefix == "search_query: "


def test_tilde_paths_expanded(tmp_path: Path) -> None:
    settings = make_settings(Path("~/somewhere"), tmp_path, db_path=Path("~/db/x.db"))
    assert "~" not in str(settings.vault_path)
    assert "~" not in str(settings.db_path)
