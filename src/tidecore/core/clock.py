"""逻辑日：05:00–05:00。

他三点睡，按自然日切，凌晨那段要等二十小时才进记忆。
所以「2026-10-22.md」覆盖的是 22 日 05:00 到 23 日 05:00。
"""

from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta

LOGICAL_START_HOUR = 5


def logical_date(dt: datetime) -> date:
    if dt.hour < LOGICAL_START_HOUR:
        return (dt - timedelta(days=1)).date()
    return dt.date()


def logical_bounds(d: date) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time(LOGICAL_START_HOUR, 0))
    return start, start + timedelta(days=1)


def completed_logical_date(dt: datetime) -> date:
    """现在能整理的那个完整逻辑日。05:00 前后都取「昨天」。"""
    return logical_date(dt) - timedelta(days=1)


def is_after_midnight(dt: datetime, day: date) -> bool:
    """已过零点、但还属于上一个逻辑日的那一段。"""
    return dt.date() > day


def day_label(d: date) -> str:
    return d.strftime("%Y-%m-%d")


def short_day(d: date) -> str:
    return d.strftime("%m-%d")


def in_quiet_hours(dt: datetime) -> bool:
    return dt.hour < LOGICAL_START_HOUR


_PAST = re.compile(
    r"(上周|上星期|前几天|前阵子|上次|之前|以前|那件事|那事|后来|还记得|曾经|上个月|去年|"
    r"昨天|前天|当天|当时|那时候|后来呢|结果呢|怎么样了|什么时候|哪天|多久|第一次)"
)
_SELF = re.compile(
    r"(你昨天|你前天|你最近|你自己|你干嘛|你干了什么|你在想什么|你的日记|你的一天|"
    r"你这两天|你最近在|你上次说想|你有什么想)"
)
_TODAY = re.compile(r"(今天|今晚|今早|刚才|刚刚)")


def is_past_query(text: str) -> bool:
    return bool(_PAST.search(text or ""))


def is_self_query(text: str) -> bool:
    return bool(_SELF.search(text or ""))


def is_today_query(text: str) -> bool:
    return bool(_TODAY.search(text or ""))


def wants_recent_overview(gap_hours: float, text: str = "") -> bool:
    return gap_hours > 2 or is_past_query(text) or is_self_query(text)


def wants_events(gap_hours: float, text: str = "") -> bool:
    return gap_hours > 2 or is_past_query(text)


def wants_index(text: str) -> bool:
    return bool(
        re.search(
            r"(上周|上星期|前几天|前阵子|上个月|去年|月初|月底|之前|以前|那时候|什么时候|哪天|多久|第一次)",
            text or "",
        )
    )
