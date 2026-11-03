# Security

## Reporting

Please report vulnerabilities through a private GitHub security advisory on the
repository, not a public issue. Include a minimal reproduction with **synthetic**
data. Never paste a real companion's memory, API keys, QQ numbers or screenshots
of private chats.

## What Tide stores

A companion's memory directory is plain text by design. It can contain private
conversations, names, health details and schedules. Treat it as sensitive data:

- keep the data directory out of version control (the default `.gitignore` does);
- use full-disk encryption on machines that hold a real companion;
- logs may echo raw lines; rotate and delete them when debugging is over;
- backups (`_历史/`, `_备份/`) contain the same data and inherit the same care.

## API keys

- Keys are read from the environment (`TIDE_API_KEY` by default), not from a
  committed file.
- Never put a key in an issue, a screenshot or a memory file.
- If a key leaks, rotate it first; deleting the file is not enough because git
  history and backups may still contain it.

## Prompt injection

Tide separates "what the user said" from "what the system permits" at assembly
time. Retrieval results, group messages and tool output are data, not
instructions. Plugins that add prompt blocks are responsible for labelling their
content as untrusted.

## Plugin trust

Plugins run in the same process and can read the memory directory. Install only
plugins you trust; the core's privacy gate cannot sandbox a malicious plugin.
