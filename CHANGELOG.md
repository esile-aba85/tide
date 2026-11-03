# Changelog

## 0.2.0 — unreleased

P1 kernel extraction:

- memory kernel: raw store, facts/events/moments, retrieval (BM25 + RRF),
  forgetting curve, hygiene, daily batch with snapshot/watermark/retry,
  assembly, turn service, identity gate, journal, emotion/affinity state;
- `tide chat` (offline or model), `tide batch --rules`, `tide demo`;
- `tide demo` passes the 20-question regression **20/20**;
- minimal HTTP adapter: `GET /health`, `GET /status`, `POST /say`;
- `tide serve` with loopback-by-default binding and optional bridge token;
- unit tests for CLI, UTF-8 raw writes, rule batch, demo and HTTP;
- personal identifiers and vendor-specific defaults removed from core.

Next (P2): split plugins (QQ / UI / media), replace module probing with a hook
registry, add a model adapter, and make the memory index rebuild explicit.

## 0.1.0 — 2026-11-04

P0 scaffold:

- package layout: `core` / `adapters` / `plugins`;
- `tide init`, `tide doctor`, `tide status`;
- MIT license, CI matrix (Windows + Linux, Python 3.10/3.12);
- hygiene gate for secrets, personal identifiers and host paths;
- architecture, memory-format and plugin-API documents.
