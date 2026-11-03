# Plugin API

Tide core is channel- and vendor-agnostic. Everything optional is a plugin.

## Ports

```python
from typing import Protocol

class MemoryStore(Protocol):
    def append_raw(self, at, speaker, text, *, channel, session, group_id=None) -> None: ...
    def facts(self, *, visible_to: str, subject: str | None) -> list[Fact]: ...
    def write_current(self, text: str, at) -> None: ...
    def rebuild_index(self) -> None: ...
    def backup(self, tag: str) -> str: ...
    def rollback(self, name: str) -> None: ...

class LLM(Protocol):
    def reply(self, assembly: Assembly, *, offline: bool) -> Reply: ...

class Channel(Protocol):
    def name(self) -> str: ...              # "cli", "http", "qq", ...
    def deliver(self, turn: Turn, reply: Reply) -> None: ...

class StateStore(Protocol):
    def load(self) -> dict: ...
    def save(self, state: dict) -> None: ...
```

## Hooks

A plugin registers callbacks instead of being imported by the core:

```python
class Plugin(Protocol):
    def register(self, hooks: Hooks) -> None: ...

hooks.before_turn(fn)      # rewrite or enrich a Turn (identity, images, ...)
hooks.after_reply(fn)      # deliver to a channel, log, speak
hooks.action("draw", fn)   # handle <cmd>{"type":"draw"}</cmd>
hooks.heartbeat(fn)        # periodic autonomy (proactive, life, room)
hooks.prompt_block(fn)     # inject a block into the dynamic tail
```

Hook failures are isolated: one broken plugin must not take down a turn.

## Minimal channel plugin

```python
class TerminalChannel:
    def name(self) -> str:
        return "cli"

    def deliver(self, turn, reply) -> None:
        print(reply.text)


def register(hooks):
    hooks.after_reply(lambda turn, reply: TerminalChannel().deliver(turn, reply))
```

## Rules

1. Core never imports a plugin; a plugin never mutates core files directly.
2. A plugin gets capabilities through ports, not through globals.
3. Anything that touches the network, a GPU, a proprietary SDK or a large model
   belongs in a plugin package, not in `tidecore.core`.
4. Plugins must be optional: `pip install tidecore` works with zero plugins.
