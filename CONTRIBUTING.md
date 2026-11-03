# Contributing

Thanks for helping. Tide is deliberately small; the hard part is keeping the
memory model honest.

## Development

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m ruff check .
python scripts/check_hygiene.py
```

Python 3.10+. The core has no runtime dependencies; please keep it that way.

## Before you open a PR

- [ ] Tests pass on your platform.
- [ ] `ruff` is clean.
- [ ] `scripts/check_hygiene.py` is clean — no keys, personal names, owner IDs,
      QQ numbers or absolute host paths.
- [ ] The change does not break one of the eight invariants (README).
- [ ] If you touch the memory format, bump `schema:` and add a migration.
- [ ] New behaviour has a test with synthetic data, not a real memory directory.

## Scope

Good first contributions:

- retrieval and forgetting tests with synthetic corpora;
- a CLI or HTTP adapter improvement;
- documentation of the memory format;
- a plugin in a separate repository (QQ, UI, speech, drawing).

Out of scope for the core repository: GUI, proprietary models, Game/voice
assets, and anything that ships personal data.

## Privacy

Never commit a real `data/` directory, a `config/llm.json` with a key, or a
screenshot of a private conversation. Use the demo fixture when you need
realistic data.
