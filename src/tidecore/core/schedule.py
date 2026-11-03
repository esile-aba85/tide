"""作息：她这个点是什么状态，语气该往哪偏。

来源：老项目的 `docs/schedule.md`（BANDS 常量，
2026-10-21 导出）。那张表在新系统里一直没接进来，这回按段接回提示词：

- 10 段，00:00–24:00 首尾相接；23:30–24:00 那段是跨零点的，这里统一按分钟算；
- 段落在 `config/schedule.json`，改那份文件就生效，不用动代码；
- 找不到/写坏了就退回下面这份默认值，不会让她因为没有作息就不说话。

它不是「静音开关」：真正决定她睡不睡、要不要主动开口的是 `proactive.py` 的
QUIET_FROM_HOUR；这里只管「这个点说话该是什么样」。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .util import json_load

# 与 `作息表.md` 一一对应，只是把「他那边的 10 段」做成可注入的一行状态。
DEFAULT_BANDS: tuple[dict, ...] = (
    {"from": "00:00", "to": "01:30",
     "state": "躺下了，迷迷糊糊。他来消息也会回，只是慢半拍，可以嘟囔一句「怎么还没睡」。"},
    {"from": "01:30", "to": "07:00",
     "state": "睡得很沉。被吵醒会懵一下，说话黏黏糊糊的，但不凶他——就是困。"},
    {"from": "07:00", "to": "09:00",
     "state": "刚醒，还没缓过来。迷糊、慢半拍，话短，别主动铺开话题。"},
    {"from": "09:00", "to": "12:00",
     "state": "上午。清醒了，但懒得动，正常聊。"},
    {"from": "12:00", "to": "14:00",
     "state": "中午刚吃完饭，有点困。话比平时少，容易敷衍。"},
    {"from": "14:00", "to": "18:00",
     "state": "下午。状态正常，该干嘛干嘛。"},
    {"from": "18:00", "to": "21:00",
     "state": "晚上。一天里最精神的时候，愿意多说两句。"},
    {"from": "21:00", "to": "23:00",
     "state": "夜里，放松下来了。话会比白天多一点，也更容易接茬。"},
    {"from": "23:00", "to": "23:30",
     "state": "快睡了，开始犯困。还有精神聊，但会时不时打哈欠。"},
    {"from": "23:30", "to": "00:00",
     "state": "已经躺下了，眼皮打架。回得短，说两句就说要去睡。"},
)


def _minutes(text: str) -> int:
    """'23:30' → 1410。写坏了返回 -1，让这一条被跳过（不整份失效）。"""
    try:
        hh, mm = str(text).strip().split(":")
        h, m = int(hh), int(mm)
    except (ValueError, AttributeError):
        return -1
    if not (0 <= h <= 24 and 0 <= m < 60):
        return -1
    return h * 60 + m


def normalize(bands) -> list[dict]:
    """只留能解析的段；全坏就用默认。"""
    out = []
    for b in bands or []:
        if not isinstance(b, dict):
            continue
        start, end = _minutes(b.get("from")), _minutes(b.get("to"))
        state = str(b.get("state") or "").strip()
        if start < 0 or end < 0 or not state:
            continue
        out.append({"from": b["from"], "to": b["to"], "state": state})
    return out or list(DEFAULT_BANDS)


def load_bands(cfg=None) -> list[dict]:
    """读 config/schedule.json；没有就用默认。每次都读，改完立刻生效。"""
    config_dir = getattr(cfg, "config_dir", None)
    if config_dir is None:
        return list(DEFAULT_BANDS)
    raw = json_load(Path(config_dir) / "schedule.json", {}) or {}
    bands = raw.get("bands") if isinstance(raw, dict) else raw
    return normalize(bands)


def band_at(bands: list[dict], now: datetime) -> dict | None:
    """这个时刻落在哪一段。跨零点的段（end <= start）按 +1 天处理。"""
    minute = now.hour * 60 + now.minute
    for b in bands:
        start, end = _minutes(b["from"]), _minutes(b["to"])
        if start < 0 or end < 0:
            continue
        if end <= start:
            end += 24 * 60
        if start <= minute < end or start <= minute + 24 * 60 < end:
            return b
    return None


def state_at(cfg, now: datetime) -> str:
    """一句话：她现在该是什么状态。取不到就是空串。"""
    try:
        b = band_at(load_bands(cfg), now)
    except Exception:  # noqa: BLE001 作息读坏了不能连话都不会说
        return ""
    return str((b or {}).get("state") or "")
