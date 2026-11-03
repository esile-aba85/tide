# Changelog

## 0.1.0 — unreleased

P0 scaffold:

- package layout: `core` / `adapters` / `plugins`;
- `tide init`, `tide doctor`, `tide status`;
- MIT license, CI matrix (Windows + Linux, Python 3.10/3.12);
- hygiene gate for secrets, personal identifiers and host paths;
- architecture, memory-format and plugin-API documents.

Next (P1): extract the memory kernel (raw store, facts, events, retrieval,
forgetting, daily batch, assembly, service) and make `tide demo` pass 20/20.
