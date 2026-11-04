# Tide / 潮汐

**Local-first, file-based long-term memory runtime for companion agents.**

Tide is the memory kernel extracted from a long-running desktop companion project.
It treats a directory of plain files as the only source of truth, and derives
everything else — facts, events, index, daily summaries, handover — from it.

> **Status: P1 (2026-11).**
> The memory kernel is extracted and works offline: `init`, `doctor`, `status`,
> `chat`, `batch`, `demo`, `serve`. The three-day demo passes **20/20**.
> The HTTP adapter is deliberately minimal (single-threaded, loopback by default).

---

## Why another memory layer?

Most "AI memory" projects start with a vector database and end with a pile of
unverifiable chunks. Tide starts from the opposite side:

- **A file is a fact.** You can read, diff, back up and repair a companion's
  memory with `git`, a text editor and your eyes.
- **Forgetting is a feature.** Facts decay with an Ebbinghaus curve; being
  recalled strengthens them; stale ones are archived, not silently deleted.
- **A day is a transaction.** Once a day a batch turns raw conversations into
  facts/events/summaries, with a snapshot, a watermark and a retry path.
- **Privacy is a boundary, not a prompt.** What a group channel can see is
  decided while assembling the prompt, before the model is called.

## The eight invariants

1. **Raw text is the only truth.** Derived files are rebuildable.
2. **One brain, many channels.** Every raw line carries its channel tag.
3. **Privacy is enforced at assembly time**, never by asking the model nicely.
4. **Four rhythms:** seconds (raw) / minutes (state) / days (batch) / months (decay).
5. **State is not memory.** Mood and affinity live outside the fact table.
6. **The watermark advances only after a fully successful batch.**
7. **The model is optional.** Offline mode still exercises the whole memory path.
8. **One writer per data directory.**

## Install (development)

```bash
git clone <this-repo> tide
cd tide
python -m pip install -e ".[dev]"
```

Python 3.10+ and no runtime dependencies.

> The PyPI distribution is **`tidecore`** (the name `tide` is taken by an
> unrelated project). Both `tide` and `tidecore` console commands are installed.
> Tide is not published yet; install from source for now.

## Quickstart

```bash
tide init --dir ./her       # memory skeleton + persona template + config
tide doctor --dir ./her     # check Python, write access, layout
tide status --dir ./her     # what is in the memory directory
tide chat --dir ./her --offline     # terminal chat; no API key needed
tide demo --dir ./demo              # 3-day synthetic run + 20-question regression
tide serve --dir ./her --offline    # GET /health, POST /say
```

`chat` and `serve` default to offline mode. To use a real model, set
`TIDE_API_KEY` and put a model name in `config/llm.json`.

`tide batch` runs the daily consolidation:

```bash
tide batch --dir ./her --date 2026-11-04 --rules --force
```

## What is in the box

| Layer | Contents |
|---|---|
| `tidecore.core` | memory kernel, daily batch, retrieval, forgetting, assembly, state |
| `tidecore.adapters` | CLI, HTTP, file storage |
| `tidecore.plugins` | optional channel/feature packs (QQ, UI, voice, media) |

## What is deliberately out of scope

GUI, QQ, voice, singing, drawing, camera and Live2D are **not** part of the core.
They depend on third-party runtimes or non-redistributable assets, and they live
in separate plugin repositories.

## Documentation

- [开源清单 / What is included](开源清单.md)
- [架构（中文）](docs/架构.md)
- [Architecture](docs/architecture.md)
- [Memory format](docs/memory-format.md)
- [Plugin API](docs/plugin-api.md)
- [Security](SECURITY.md)

This repository is self-contained: it never contains personal data, API keys,
model weights or third-party assets. See the manifest for the exact boundary.

## License

MIT. See [LICENSE](LICENSE).
