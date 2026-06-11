# Architecture

## Overview

This project is a local-first personal AI system built around a simple idea: the
LLM is a commodity; the durable value is the **context system** around it. That
system has three pillars:

1. **Static context** — a Markdown vault: everything worth remembering, in plain
   files, git-versioned, owned forever.
2. **Dynamic context** — a SQLite database of structured, changing facts
   (contacts, opportunities, interactions, events — or whatever the user declares).
3. **Capabilities** — tools the LLM can call to act on both (v1: search, read,
   write, query, log; v2: automations).

Everything runs on the user's machine. The LLM is served by Ollama; a cloud API
can be substituted by changing two environment variables.

## Layers

```
┌─────────────────────────────────────────────────────────┐
│ Interfaces                                              │
│   Claude Code (MCP client, dev & daily use in v1)       │
│   Custom agentic CLI (v2)                               │
└──────────────────────────┬──────────────────────────────┘
                           │ MCP (stdio)
┌──────────────────────────┴──────────────────────────────┐
│ MCP servers (Python, FastMCP)                           │
│   mcp-notes   search_notes / read_note / write_note     │
│   mcp-data    query (read-only) / log_* (validated)     │
└─────────┬───────────────────────────────┬───────────────┘
          │                               │
┌─────────┴─────────────┐   ┌─────────────┴───────────────┐
│ Markdown vault        │   │ SQLite database             │
│   plain .md files     │   │   user tables (from YAML)   │
│   git-versioned       │   │   + sqlite-vec index        │
│   outside the repo    │   │   outside the repo          │
└─────────▲─────────────┘   └─────────────▲───────────────┘
          │                               │
┌─────────┴───────────────────────────────┴───────────────┐
│ Ingest pipeline (indexer)                               │
│   watch/scan vault → chunk → embed → upsert into index  │
└─────────────────────────────────────────────────────────┘

Inference (used by ingest for embeddings, by interfaces for chat):
  Ollama — chat: qwen3.6:35b-a3b, embeddings: nomic-embed-text
  (any OpenAI-compatible endpoint via LLM_BASE_URL)
```

## Components

### Vault (static context)

A directory of Markdown files with optional YAML frontmatter. Conventions are
documented but minimally enforced: the indexer accepts any `.md` file. Suggested
top-level structure (shipped as `examples/vault/`):

```
vault/
├── inbox/        # unsorted captures — index everything, sort later
├── projects/
├── people/
├── reference/
└── journal/
```

Resilience argument: no proprietary format, no database required to read your
own notes. The vault outlives every tool, including this one.

### Ingest pipeline

`medronho.ingest` walks the vault, detects changed files (content hash stored
in SQLite), splits them into chunks (heading-aware, with overlap), requests
embeddings from the inference layer, and upserts vectors into sqlite-vec. It is:

- **Idempotent**: re-running on an unchanged vault is a no-op.
- **Incremental**: only changed files are re-embedded.
- **Tolerant**: a malformed file is logged and skipped, never fatal.

Run manually (`uv run medronho index`) or scheduled (launchd/cron).

### mcp-notes

FastMCP server exposing the vault:

- `search_notes(query, limit)` — embeds the query, runs vector similarity search,
  returns chunks with file path, heading context and score.
- `read_note(path)` — returns a full note.
- `write_note(path, content)` — creates or updates a note (vault-confined).

Security: all paths resolved and confined to the vault root (tested); contents
are treated as untrusted data (see `SECURITY.md`).

### mcp-data

FastMCP server exposing the dynamic-context database:

- `query(sql)` — executes SELECT statements on a **read-only** connection
  (SQLite `mode=ro`). Schema introspection is provided to the LLM via a
  `describe_schema` tool so it can write correct SQL.
- `log_<entity>(...)` — one structured write tool per entity declared in the
  YAML schema. Inputs validated by generated pydantic models. No free-form SQL
  on any writable connection.

### Schema-as-config

The user declares entities in YAML (example shipped in `examples/schema.yaml`):

```yaml
entities:
  contacts:
    fields:
      name: {type: str, required: true}
      company: {type: str}
      notes: {type: str}
  interactions:
    fields:
      contact: {type: ref, entity: contacts}
      kind: {type: enum, values: [call, email, meeting]}
      summary: {type: str, required: true}
      occurred_at: {type: datetime, default: now}
```

At startup, the engine generates: SQLite tables (with a migration-lite "create
if missing, never drop" policy), pydantic models, and the corresponding
`log_*` MCP tools. This is what makes the engine generic: the author's
freelancer CRM is just one possible YAML file, kept outside the repo.

### Inference layer

A single module (`medronho.core.llm`) builds an `openai.OpenAI` client from
settings:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama or any compatible API |
| `LLM_API_KEY` | `ollama` | ignored by Ollama, required by clouds |
| `LLM_CHAT_MODEL` | `qwen3.6:35b-a3b` | chat/tool-calling model |
| `LLM_EMBED_MODEL` | `nomic-embed-text` | embedding model |

Local-first, API-optional: sovereignty is having the choice.

## Data flow examples

**"What did I conclude about X?"** — client calls `search_notes("X")` →
mcp-notes embeds the query (Ollama) → sqlite-vec top-k → chunks returned to the
client LLM, which answers with citations to note paths.

**"Log this call with Y"** — client calls `log_interactions(contact=…, kind=call,
summary=…)` → pydantic validation → INSERT on the writable connection → confirmation.

**Weekly review** — client calls `query("SELECT … FROM interactions WHERE …")`
on the read-only connection, combines with `search_notes` results.

## Non-goals (v1)

Custom CLI, mcp-skills/automations, UI, multi-user, sync, remote hosting,
ingestion of non-Markdown sources. Revisit at the one-month usage checkpoint.
