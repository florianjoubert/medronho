# CLAUDE.md

## What this project is

A local-first "AI second brain": a personal knowledge and automation system that an
LLM can query through MCP servers. It runs entirely on the user's machine (Apple
Silicon, Ollama) with an optional escape hatch to cloud APIs.

Two goals, both first-class:

1. **Real daily use** by the author (notes memory, freelance pipeline tracking).
2. **A professional open-source showcase**: the repo must read as state-of-the-art
   Python (2026) — tooling, typing, tests, security, docs.

Never sacrifice goal 2 for a quick hack. Never sacrifice goal 1 by building
abstractions nobody asked for.

## Core design principle: generic engine, personal config

The engine is generic and configurable. Nothing in the codebase refers to the
author, his clients, or his data.

- The dynamic-context database schema is **declared in a YAML config file**, not
  hardcoded. The engine generates tables and logging tools from that declaration.
- Vault location, database path, model names: all come from settings (env vars /
  `.env`), never hardcoded.
- User data (the vault, the SQLite file, the personal YAML config) lives **outside
  the repo**. The repo ships an `examples/` config (a freelancer mini-CRM: contacts,
  opportunities, interactions, events) used by the quickstart and the tests.

## Architecture (4 layers)

```
Interfaces        Claude Code (dev/test client) → later: custom CLI (v2)
Inference         Ollama (default: qwen3.6:35b-a3b) | optional cloud API
                  → single OpenAI-compatible client, switched by base URL
MCP servers       mcp-notes (semantic search, read, write)
(Python/FastMCP)  mcp-data  (read-only SQL + validated structured logging)
                  mcp-skills (v2 — out of scope for now)
Sovereign data    Markdown vault (git-versioned) | SQLite + sqlite-vec index
```

Plus one transversal piece: `ingest/` — chunks modified notes, computes embeddings
(nomic-embed-text via Ollama), updates the sqlite-vec index. Idempotent, incremental
(hash-based change detection).

See `docs/ARCHITECTURE.md` for details.

## Locked technical decisions

Do not revisit these without an explicit discussion:

- **Python ≥ 3.12**, managed with **uv** (no pip, no poetry). Single `pyproject.toml`.
- **src/ layout**: code in `src/medronho/`, importable package.
- **ruff** (lint + format), **pyright strict** (type checking). All code fully typed.
- **pydantic v2** for data models; **pydantic-settings** for all configuration.
- **FastMCP** (official MCP Python SDK) for the servers, stdio transport.
- **SQLite** (stdlib `sqlite3`) + **sqlite-vec** for the vector index. One database
  file. No Postgres, no ChromaDB.
- **LLM access**: `openai` client pointed at an OpenAI-compatible base URL.
  Defaults to Ollama (`http://localhost:11434/v1`). Cloud is opt-in via env vars.
- **pytest** for tests; **GitHub Actions** CI (ruff + pyright + pytest) on every push.
- **pre-commit** hooks mirroring CI.
- License: **MIT**. Commits: **Conventional Commits** (`feat:`, `fix:`, `docs:`…).

## Code quality rules

- Type hints everywhere, including tests. `Any` requires a comment justifying it.
- Public functions and modules get docstrings (Google style). No comment noise on
  obvious code.
- Small, composable functions. No speculative abstraction: build for the current
  phase, design so v2 doesn't require rewrites.
- Tests are mandatory for the core: chunking, indexing, search ranking, path
  confinement, schema generation, write validation. UI/glue code may go untested.
- Errors: raise precise exceptions; MCP tools return structured, actionable error
  messages (the calling LLM must be able to self-correct).
- Dependencies: minimal and mainstream. Every new dependency must be justified.

## Security requirements (non-negotiable)

These are differentiating features of the project — treat them as such:

- **mcp-data is read-only by default.** The SQLite connection for the `query` tool
  is opened in read-only mode (`mode=ro` URI). Writes happen exclusively through
  dedicated logging tools whose inputs are validated by pydantic models generated
  from the YAML schema. No free-form SQL ever reaches a writable connection.
- **Vault confinement in mcp-notes.** Every path is resolved (`Path.resolve()`) and
  verified to be inside the vault root before any read or write. Path traversal
  attempts return an error. This is covered by tests.
- **Prompt injection awareness.** Note contents are untrusted data that flows into
  LLM context. No destructive tools are exposed (no delete, no shell). Document
  this threat model in `docs/SECURITY.md`.
- **No secrets in the repo.** `.env` is gitignored; `.env.example` documents every
  variable. Settings validation fails loudly on missing required values.

## Repository layout

```
medronho/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env.example
├── .github/workflows/ci.yml
├── src/medronho/
│   ├── core/          # settings, llm client, db connection
│   ├── ingest/        # chunker, embedder, indexer
│   └── servers/
│       ├── notes/     # mcp-notes
│       └── data/      # mcp-data (+ schema loader)
├── examples/          # example YAML schema + sample vault for quickstart/tests
├── tests/
└── docs/
    ├── ARCHITECTURE.md
    └── SECURITY.md
```

## Working agreement

- Work happens in ~half-day sessions, roughly weekly. **Every session must end
  with something that works and a commit.** Never leave a broken main branch
  between sessions.
- Versions are explicitly v0.x until the one-month usage checkpoint.
- README must always allow a stranger to install and try the project in under
  five minutes (`uv` install, an `init` command creating an example vault, one
  working search query).

## Roadmap

- **Session 1**: repo scaffolding (uv, ruff, pyright, pre-commit, CI), settings
  module, ingest pipeline (chunking → embeddings → sqlite-vec) with tests.
- **Session 2**: mcp-notes (search_notes, read_note, write_note), connected to
  Claude Code. First real usage.
- **Session 3**: YAML schema loader, table generation, mcp-data (read-only query
  + validated logging tools).
- **Session 4**: consolidation, README, SECURITY.md, quickstart polish, v0.1 tag.
- **Sessions 5-7**: hardening driven by real usage; usage checkpoint at one month.

Out of scope for v1 (do not build): custom agentic CLI, mcp-skills, any UI,
multi-user, remote hosting, non-Markdown ingestion.
