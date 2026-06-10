# SQLite tips

## WAL mode

Write-Ahead Logging lets readers proceed while a writer is active. Enable it
once per database with `PRAGMA journal_mode=WAL`.

## One file is a feature

A single-file database is trivially easy to back up, version and move between
machines — a good fit for local-first tools.
