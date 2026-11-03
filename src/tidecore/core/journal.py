"""写路径：每条消息，不调模型。

- 原文按来源分文件，只追加；
- 当天要点是草稿，日批后清空；
- 她自己的行动进 `我/今日-行动.md`；
- 群聊只记 @她 / 她说话前后 ±3 分钟 / 主人，其余每 30 分钟一行机械摘要；
- 边界由主人一句话即时写，不等日批。
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from . import clock
from .config import Config
from .identity import Gate
from .memory import Fact, Memory
from .util import clip, ensure_dir, json_dump, json_load, one_line, read_text, sanitize

HER_NAMES = ("小潮", "潮汐", "Tide")

SIGNAL_PATTERNS = {
    "承诺": re.compile(r"(答应|说好|约好|一定|保证|等我|回头|记得|别忘|下次|以后)"),
    "时间": re.compile(r"(今天|明天|后天|昨天|下周|上周|这周|月底|月初|下周[一二三四五六日天]|\d{1,2}[点号日])"),
    "人名": re.compile(r"(妈妈|爸爸|妈|爸|奶奶|爷爷|外婆|外公|同事|老板|朋友|医生|老师|[\u4e00-\u9fa5]{1,3}(?:叔|姨|哥|姐|弟|妹|总|老师))"),
    "情绪": re.compile(r"(累|难受|烦|开心|兴奋|难过|崩溃|生气|气死|哭|孤单|孤独|焦虑|紧张|害怕|心疼|委屈|睡不着|失眠|压力)"),
}
RELATION_WORDS = re.compile(r"(吵架|表白|道歉|再见|想你|喜欢你|讨厌你|别这样|生气|冷战|谢谢|对不起|离开|不要你|分手)")
PRIVACY_WORDS = re.compile(r"(别告诉|不要告诉|别说出去|这事保密|别跟别人说)")


def has_signal(text: str) -> str:
    for name, pat in SIGNAL_PATTERNS.items():
        if pat.search(text or ""):
            return name
    return ""


class Journal:
    def __init__(self, mem: Memory):
        self.mem = mem
        self.cfg: Config = mem.cfg
        # 群里每个号每天最多读写人物卡 6 次（卡上只留最好的 3 句）。
        # 不缓存的话，一百来号人每句话都读一遍文件，桥会被磁盘拖慢。
        self._people_seen: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------ 用户消息
    def record_user(
        self,
        gate: Gate,
        sender_id: str,
        sender_name: str,
        text: str,
        session_type: str,
        group_id: str | None,
        when: datetime,
        channel: str = "",
    ) -> dict:
        text = sanitize(text, 1200)
        if not text:
            return {"written": False}
        day = clock.logical_date(when)
        source = "private" if session_type == "private" else f"group:{group_id}"
        label = _speaker_label(self.cfg, gate, sender_name, session_type)

        info = {"written": False, "highlight": False, "boundary": False, "pending": False, "fact": False}

        # 群聊限流
        if session_type == "group" and group_id:
            written = self._record_group(gate, sender_id, label, text, group_id, when, day)
            info["written"] = written
        else:
            self.mem.append_raw(
                when, day, source, label, _privacy_filter(text), group_id=None, channel=channel
            )
            info["written"] = True

        # 群友分人记忆：不管这句有没有落进原文，都先在他的人物卡上记一笔。
        # 旁听的消息以前只进 30 分钟摘要，谁说的、说过什么，日批时已经糊成一片。
        if session_type == "group" and group_id and sender_id and gate.role != "owner":
            cache_key = (sender_name or sender_id, day.isoformat())
            used = self._people_seen.get(cache_key, 0)
            if used < 6:
                self._people_seen[cache_key] = used + 1
                try:
                    # 和原文用同一道隐私过滤：说了「别告诉别人」的内容不该在
                    # 人物卡（会被注入群/私聊提示词）里原样留着。
                    self.mem.upsert_person_seen(
                        sender_name,
                        sender_id,
                        self.cfg.group_name(group_id),
                        _privacy_filter(text),
                        when,
                        group_id=str(group_id),
                    )
                except Exception:  # noqa: BLE001 人物卡坏了不能拖垮收消息
                    pass
            if len(self._people_seen) > 5000:
                self._people_seen.clear()

        # 当天要点：命中信号，或攒够批量
        trigger = has_signal(text)
        count = self.mem.raw_count(day, source)
        batch = int(self.cfg.knobs.highlight_batch_private if session_type == "private" else self.cfg.knobs.highlight_batch_group)
        if written_flag(info) and (trigger or (batch and count and count % batch == 0)):
            prefix = "群里" if session_type == "group" else ""
            self.mem.append_highlight(day, f"{prefix}{label}：{text}", when)
            info["highlight"] = True

        # 关系待覆盖
        if gate.role == "owner" and RELATION_WORDS.search(text):
            self.mem.append_pending("关系", text, when)
            info["pending"] = True

        # 边界即时写：只对主人
        if gate.boundary_line and info["written"]:
            info["boundary"] = self.mem.append_boundary(gate.boundary_line)

        # 他人明确说「记住 X」：最多 转述 / 0.5 / 群 / 低
        if gate.low_fact and info["written"]:
            m = re.search(r"(?:记住|记一下|帮我记|别忘)[：:，,\s]*(.{1,80})", text)
            if m:
                fact = Fact(
                    subject=sender_name or sender_id,
                    topic="备注",
                    content=m.group(1).strip(),
                    speaker=sender_name or sender_id,
                    trust="路人",
                    source="转述",
                    confidence=0.5,
                    visibility="群",
                    importance="低",
                    noted=day.isoformat(),
                    group_id=str(group_id or ""),
                    origin=f"{when:%m-%d} 群{group_id or ''}",
                )
                self.mem.merge_fact(fact)
                info["fact"] = True

        # 最新消息时间（状态面板用）
        if info["written"]:
            self.mem.set_status({"最新消息": f"{when.strftime('%Y-%m-%d %H:%M')} {label}"})
        return info

    def _record_group(
        self,
        gate: Gate,
        sender_id: str,
        label: str,
        text: str,
        group_id: str,
        when: datetime,
        day,
    ) -> bool:
        state = self._load_hot(group_id)
        if state.get("day") != day.isoformat():
            state = {"day": day.isoformat(), "day_count": 0, "bucket": "", "count": 0, "words": {}, "recent": [], "her_at": ""}
        bucket = when.strftime("%H:") + f"{(when.minute // int(self.cfg.knobs.group_summary_minutes)) * int(self.cfg.knobs.group_summary_minutes):02d}"

        # 换桶：把上一桶写成一行机械摘要
        if state.get("bucket") and state["bucket"] != bucket and state.get("count"):
            self._flush_summary(state, when, group_id, day)

        if state.get("bucket") != bucket:
            state["bucket"] = bucket
            state["count"] = 0
            state["words"] = {}

        state["count"] = int(state.get("count", 0)) + 1
        state["day_count"] = int(state.get("day_count", 0)) + 1
        for w in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,8}", text)[:20]:
            state["words"][w] = int(state["words"].get(w, 0)) + 1

        recent = state.get("recent") or []
        recent.append(
            {
                "at": when.isoformat(timespec="seconds"),
                "label": label,
                "text": _privacy_filter(text),
                "written": False,
            }
        )
        # 只留最近 3 分钟 + 30 条
        cutoff = when - timedelta(minutes=int(self.cfg.knobs.group_keep_around_min))
        recent = [r for r in recent if _parse_dt(r.get("at")) and _parse_dt(r["at"]) >= cutoff][-30:]
        state["recent"] = recent

        priority = (
            gate.role == "owner"
            or any(n in text for n in HER_NAMES)
            or self._her_recent(state, when)
        )
        written = False
        if priority:
            # 先把 3 分钟内没写的补上；当前这条若还没写，再单独写
            self._flush_recent(state, when, group_id, day)
            if not recent[-1].get("written"):
                self.mem.append_raw(
                    when,
                    day,
                    f"group:{group_id}",
                    label,
                    recent[-1].get("text") or _privacy_filter(text),
                    group_id=group_id,
                )
                recent[-1]["written"] = True
            written = True
        else:
            state["recent"] = recent

        self._save_hot(group_id, state)
        return written

    def _flush_recent(self, state: dict, when: datetime, group_id: str, day) -> bool:
        wrote = False
        for r in state.get("recent") or []:
            if r.get("written"):
                continue
            at = _parse_dt(r.get("at"))
            if not at:
                continue
            if when - at > timedelta(minutes=int(self.cfg.knobs.group_keep_around_min)):
                r["written"] = True
                continue
            self.mem.append_raw(at, day, f"group:{group_id}", r.get("label") or "群友", r.get("text") or "", group_id=group_id)
            r["written"] = True
            wrote = True
        return wrote

    def _flush_summary(self, state: dict, when: datetime, group_id: str, day) -> None:
        words = sorted((state.get("words") or {}).items(), key=lambda kv: kv[1], reverse=True)[:5]
        top = "、".join(w for w, _ in words) or "闲聊"
        start = state.get("bucket") or ""
        self.mem.append_raw(
            when,
            day,
            f"group:{group_id}",
            "[群摘要]",
            f"{start}–{when.strftime('%H:%M')} 群 {state.get('count', 0)} 条，高频词：{top}",
            group_id=group_id,
        )

    def _her_recent(self, state: dict, when: datetime) -> bool:
        at = _parse_dt(state.get("her_at"))
        return bool(at and when - at <= timedelta(minutes=int(self.cfg.knobs.group_keep_around_min)))

    def note_her_spoke(self, session_type: str, group_id: str | None, when: datetime, day) -> None:
        if session_type != "group" or not group_id:
            return
        state = self._load_hot(group_id)
        if state.get("day") != day.isoformat():
            return
        state["her_at"] = when.isoformat(timespec="seconds")
        self._flush_recent(state, when, group_id, day)
        self._save_hot(group_id, state)

    def _hot_path(self, group_id: str):
        return self.cfg.hot_dir / f"group-{group_id}.json"

    def _load_hot(self, group_id: str) -> dict:
        ensure_dir(self.cfg.hot_dir)
        return json_load(self._hot_path(group_id), {}) or {}

    def _save_hot(self, group_id: str, state: dict) -> None:
        json_dump(self._hot_path(group_id), state)

    # ------------------------------------------------------------ 她的话 / 行动
    def record_her(
        self,
        session_type: str,
        group_id: str | None,
        text: str,
        when: datetime,
        day,
        channel: str = "",
    ) -> None:
        source = "private" if session_type == "private" else f"group:{group_id}"
        self.mem.append_raw(
            when, day, source, "我", _privacy_filter(text), group_id=group_id, channel=channel
        )

    def record_action(self, text: str, when: datetime) -> None:
        self.mem.append_action(when, text)

    # ------------------------------------------------------------ 身份类指令
    def apply_identity(self, target: str, sentence: str, when: datetime) -> bool:
        """只有主人能走到这里。改核心 ≤2 句，立即生效。"""
        if target == "core_her":
            body = f"我是一个住在电脑里的陪伴者。\n{sentence.strip()}\n\n## 群里能说的\n我是一个住在电脑里的陪伴者。"
            return self.mem.write_core("her", body, when.strftime("%Y-%m-%d"))
        if target == "core_me":
            name = self.cfg.owner_name
            body = (
                f"他叫{name}，是我的主人，也是唯一能让我做事的人。\n"
                f"{sentence.strip()}\n"
                "他不太喜欢被追问，情绪低的时候要少说两句。\n\n"
                "## 群里能说的\n他是我主人。别的我不在群里说。"
            )
            return self.mem.write_core("me", body, when.strftime("%Y-%m-%d"))
        return False


def written_flag(info: dict) -> bool:
    return bool(info.get("written"))


def _speaker_label(cfg: Config, gate: Gate, sender_name: str, session_type: str) -> str:
    if gate.role == "owner":
        return cfg.owner_display if session_type == "private" else (sender_name or cfg.owner_name)
    return (sender_name or "群友")[:12]


def _privacy_filter(text: str) -> str:
    if PRIVACY_WORDS.search(text or ""):
        return "（他让我别说出去，内容不记）"
    return text


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None
