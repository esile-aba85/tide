# Architecture

> Status: P1. The memory kernel is extracted and runs offline; plugins and a
> model adapter are next.

## Goal

A small, dependency-free runtime that lets an agent keep a long, inspectable
memory of one person across many conversations and channels.

The unit of truth is **a directory of files**, not a database and not a vector
store. Everything else is derived and can be rebuilt.

## Non-goals

GUI, chat-platform adapters, speech, singing, drawing, camera and Live2D.
Those are plugins or separate applications.

## Layers

```
channels/CLI · HTTP            (adapters)
        │
        ▼
Service · Assembler · Memory · Batch · Retrieve · Forget · State   (core)
        │
        ▼
files on disk                  (storage)
```

Dependency rule:

- `tidecore.core` imports only the standard library and `tidecore.core`.
- `tidecore.adapters` may import `tidecore.core`.
- `tidecore.plugins` may import `tidecore.core`.
- Core never imports adapters or plugins.

The rule is enforced in CI by import-linter.

## One turn

1. A channel delivers `Turn(sender, text, channel, at)`.
2. Identity gate decides `owner` / `member` / `stranger`.
3. Raw text is appended to the day file (append-only).
4. Assembly builds the prompt:
   - stable head: persona + core memory + boundaries + rules;
   - thread: recent messages;
   - dynamic tail: retrieved facts/events/raw lines + current state + time.
5. The model replies (or an offline placeholder does).
6. Actions (`draw`, `sing`, `body`, ...) are extracted and dispatched to plugins.
7. Her reply is written back to raw memory and to the outbox / channel.

## One day

1. After the configured hour, the batch checks the watermark.
2. Snapshot the whole memory directory.
3. Extract facts/events/summary with the model; fall back to rules offline.
4. Apply idempotently: close superseded facts, merge new ones, write overview,
   diary, handover, rebuild the index.
5. Run hygiene checks; write status.
6. Advance the watermark **only if every step succeeded**; otherwise schedule a
   retry without advancing it.

## Invariants

See the eight invariants in the README. They are the review checklist for every
change.

## What lives where

- `core/memory/` — raw files, fact/event schemas, backup and rollback.
- `core/batch/` — daily extraction, apply, diary, watermark.
- `core/retrieve.py` — bigram index, IDF coverage + BM25 + RRF.
- `core/forget.py` — Ebbinghaus recall/stability and archival.
- `core/assemble.py` — prompt layout and budgets.
- `core/service.py` — turn orchestration and hooks.
- `core/state/` — mood and affinity, deliberately outside memory.
