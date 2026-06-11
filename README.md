# medronho

*Medronho is the fruit of the strawberry tree that grows wild in the hills of
the Algarve, where this project was built. Locals distill it into something
stronger; this tool does the same with your notes.*

A local-first AI second brain: your notes live in a plain Markdown vault, your
structured facts in a SQLite database, and LLMs reach both through MCP servers.
Everything runs on your machine (Ollama); any OpenAI-compatible cloud API can be
substituted with two environment variables.

> Status: v0.x — Session 1 of the roadmap. The ingest pipeline (chunking →
> embeddings → sqlite-vec index) is functional; the MCP servers land next.

## How it works

1. **Vault** — a directory of Markdown files you own forever, git-versioned,
   outside this repo.
2. **Ingest** — `medronho index` walks the vault, chunks notes
   (heading-aware), embeds them with `nomic-embed-text` via Ollama, and upserts
   vectors into a sqlite-vec index. Incremental and idempotent: only changed
   files are re-embedded.
3. **MCP servers** (upcoming) — `mcp-notes` (semantic search, read, write) and
   `mcp-data` (read-only SQL + validated structured logging).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture.

## Quickstart

Requirements: Python ≥ 3.12, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com) running locally.

```sh
git clone https://github.com/florianjoubert/medronho
cd medronho
uv sync

# Pull the models (embeddings are required for indexing; chat comes later)
ollama pull nomic-embed-text
ollama pull qwen3.6:35b-a3b

# Configure: point VAULT_PATH at the bundled example vault to try it out
cp .env.example .env
# in .env: VAULT_PATH=./examples/vault  DB_PATH=./medronho.db

# Index the vault
uv run medronho index
```

Re-running `medronho index` on an unchanged vault is a no-op. If you change
the embedding model or dimension, the index refuses to run and tells you to
`medronho index --rebuild`.

## Development

```sh
uv sync
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
```

Tests and CI do **not** require Ollama — embeddings are faked at the test
boundary. The same checks run in GitHub Actions on every push.

## License

[MIT](LICENSE)
