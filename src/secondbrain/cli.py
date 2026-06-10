"""Command-line interface (`secondbrain`)."""

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from secondbrain.core.db import IndexCompatibilityError, connect
from secondbrain.core.settings import Settings
from secondbrain.ingest.embedder import OpenAIEmbedder
from secondbrain.ingest.indexer import index_vault


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``secondbrain`` command."""
    parser = argparse.ArgumentParser(
        prog="secondbrain",
        description="Local-first AI second brain.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser(
        "index", help="Index the Markdown vault into the vector store."
    )
    index_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the existing index and re-embed the whole vault.",
    )
    args = parser.parse_args(argv)
    if args.command == "index":
        return _cmd_index(rebuild=bool(args.rebuild))
    raise AssertionError("unreachable: subcommand is required")


def _cmd_index(*, rebuild: bool) -> int:
    """Run the indexer with settings from the environment."""
    try:
        # Values come from the environment / .env; pyright cannot see that.
        settings = Settings()  # pyright: ignore[reportCallIssue]
    except ValidationError as exc:
        print(f"Invalid configuration (check your .env):\n{exc}", file=sys.stderr)
        return 1
    if not settings.vault_path.is_dir():
        print(f"VAULT_PATH is not a directory: {settings.vault_path}", file=sys.stderr)
        return 1

    conn = connect(settings.db_path)
    try:
        stats = index_vault(conn, settings, OpenAIEmbedder.from_settings(settings), rebuild=rebuild)
    except IndexCompatibilityError as exc:
        print(exc, file=sys.stderr)
        return 1
    finally:
        conn.close()
    print(
        f"scanned={stats.scanned} indexed={stats.indexed} skipped={stats.skipped} "
        f"removed={stats.removed} errors={stats.errors}"
    )
    return 0
