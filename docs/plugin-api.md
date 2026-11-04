# Plugin API

Tide core is channel- and vendor-agnostic. Everything optional is a plugin.

## Ports (target seam, P2)

The core currently uses concrete classes. These protocols describe the seam we
are extracting in P2; they are not imported yet, so do not build against them
without checking the source.

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

class StateStore(Protocol):
    def load(self) -> dict: ...
    def save(self, state: dict) -> None: ...
```

## Hooks (implemented)

`tidecore.core.hooks.Hooks` is real. `Service.turn()` runs it around every turn:

```python
from tidecore.core.hooks import Hooks
from tidecore.core.service import Service

hooks = Hooks()
hooks.before_turn(lambda turn: print("in:", turn.text))       # may edit turn.text
hooks.after_reply(lambda turn, result: print("out:", result.reply))

service = Service(cfg, hooks=hooks)
service.turn("owner", "hello", channel="cli")
```

`Turn` carries `sender_id`, `text`, `session_type`, `group_id`, `channel`, `at`,
`nickname` and `images`.

A failing hook is skipped; the turn still runs. Hook order is registration order.

Planned for P2: `hooks.action("draw", fn)` for `<cmd>` actions, `hooks.heartbeat(fn)`
for periodic work and `hooks.prompt_block(fn)` for dynamic prompt blocks.

## Minimal channel plugin

```python
class TerminalChannel:
    def name(self) -> str:
        return "cli"

    def deliver(self, turn, result) -> None:
        print(result.reply)


def register(hooks):
    hooks.after_reply(lambda turn, result: TerminalChannel().deliver(turn, result))
```

A channel still drives the service itself (CLI and HTTP adapters show how);
there is no automatic discovery yet.

## Rules

1. Core never imports a plugin; a plugin never mutates core files directly.
2. A plugin gets capabilities through ports, not through globals.
3. Anything that touches the network, a GPU, a proprietary SDK or a large model
   belongs in a plugin package, not in `tidecore.core`.
4. Plugins must be optional: `pip install tidecore` works with zero plugins.
