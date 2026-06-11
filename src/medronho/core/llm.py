"""Factory for the OpenAI-compatible inference client.

This is the single switch point between local (Ollama) and cloud inference:
both are reached through the same client, selected by ``LLM_BASE_URL``.
"""

from openai import OpenAI

from medronho.core.settings import Settings


def make_client(settings: Settings) -> OpenAI:
    """Build an OpenAI-compatible client from settings (Ollama by default)."""
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
