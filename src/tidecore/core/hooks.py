"""Turn-level extension hooks.

Plugins get a :class:`Turn` before the core handles it and a result after. A
broken hook is isolated: it is skipped silently, because a plugin must never
take down a conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Turn:
    sender_id: str
    text: str
    session_type: str = "private"
    group_id: str | None = None
    channel: str = ""
    at: datetime = field(default_factory=datetime.now)
    nickname: str = ""
    images: list[str] = field(default_factory=list)


BeforeTurn = Callable[[Turn], None]
AfterReply = Callable[[Turn, Any], None]


class Hooks:
    """Ordered callback registry used by :meth:`Service.turn`."""

    def __init__(self) -> None:
        self._before: list[BeforeTurn] = []
        self._after: list[AfterReply] = []

    def before_turn(self, fn: BeforeTurn) -> BeforeTurn:
        self._before.append(fn)
        return fn

    def after_reply(self, fn: AfterReply) -> AfterReply:
        self._after.append(fn)
        return fn

    def run_before(self, turn: Turn) -> None:
        self._run(self._before, turn)

    def run_after(self, turn: Turn, result: Any) -> None:
        self._run(self._after, turn, result)

    @staticmethod
    def _run(fns: list[Callable[..., None]], *args: Any) -> None:
        for fn in list(fns):
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 a plugin must not break a turn
                continue
