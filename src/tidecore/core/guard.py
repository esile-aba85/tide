"""群聊熔断：防止她和另一个 AI 互相 @，无限烧 token。

规则（都在 config/knobs.json 里可调）：
- 同一个群，窗口内她回的条数到上限 → 熔断；
- 连续 N 条都在回同一个账号 → 熔断；
- 熔断后冷却期内，群里非主人的消息一律不接（不回、不调模型、不写记忆）；
- 主人始终放行；`/熔断 解除` 手动解。

状态落在 data/group_guard.json，重启不丢。
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class Decision:
    allowed: bool
    reason: str = ""
    retry_after: int = 0     # 秒
    tripped: bool = False    # 本次是否刚触发（用来只记一次日志）


class GroupGuard:
    def __init__(self, cfg):
        self.cfg = cfg
        root = Path(cfg.root)
        self.path = root / "data" / "group_guard.json"
        self.log_path = root / "data" / "group_guard.log"
        self._lock = threading.Lock()
        self.groups = self._load()

    # ---------------------------------------------------------- 配置
    @property
    def enabled(self) -> bool:
        return bool(self.cfg.knobs.group_breaker_enabled)

    def _knob(self, name: str, default):
        try:
            return getattr(self.cfg.knobs, name)
        except AttributeError:
            return default

    # ---------------------------------------------------------- 状态
    def _load(self) -> dict:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            groups = raw.get("groups") if isinstance(raw, dict) else None
            return groups if isinstance(groups, dict) else {}
        except Exception:  # noqa: BLE001 状态坏了就当没有
            return {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps({"groups": self.groups}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass

    def _group(self, gid: str) -> dict:
        st = self.groups.get(gid)
        if not isinstance(st, dict):
            st = {}
            self.groups[gid] = st
        st.setdefault("replies", [])
        st.setdefault("streak", 0)
        st.setdefault("last_sender", "")
        st.setdefault("last_at", "")
        st.setdefault("trips", [])
        st.setdefault("cooldown_until", "")
        st.setdefault("tripped_at", "")
        st.setdefault("reason", "")
        return st

    @staticmethod
    def _parse(raw) -> datetime | None:
        try:
            return datetime.fromisoformat(str(raw)) if raw else None
        except ValueError:
            return None

    def _prune(self, st: dict, now: datetime) -> None:
        seconds = int(self._knob("group_breaker_window_sec", 3))
        floor = now - timedelta(seconds=seconds)
        keep = []
        for item in st.get("replies") or []:
            try:
                at = datetime.fromisoformat(str(item[0]))
            except (ValueError, TypeError, IndexError):
                continue
            if at > floor:
                keep.append(item)
        st["replies"] = keep

    def _retry_after(self, st: dict, now: datetime) -> int:
        until = self._parse(st.get("cooldown_until"))
        return max(0, int((until - now).total_seconds())) if until else 0

    # ---------------------------------------------------------- 判定
    def check(self, group_id: str, sender_id: str, is_owner: bool = False,
              now: datetime | None = None) -> Decision:
        now = now or datetime.now()
        if not self.enabled or not group_id:
            return Decision(True)
        gid = str(group_id)
        sid = str(sender_id or "")
        with self._lock:
            st = self._group(gid)
            if is_owner:                      # 主人永远放行
                return Decision(True, "owner")
            until = self._parse(st.get("cooldown_until"))
            if until and now < until:
                return Decision(False, st.get("reason") or "熔断中", self._retry_after(st, now))
            if until:                          # 冷却结束：重新给一个窗口
                st["cooldown_until"] = ""
                st["streak"] = 0
                st["last_sender"] = ""
                st["reason"] = ""

            # 隔太久没回同一个人，就不算「连续」了
            gap_sec = int(self._knob("group_breaker_streak_gap_sec", 30))
            last_at = self._parse(st.get("last_at"))
            if last_at and (now - last_at) > timedelta(seconds=gap_sec):
                st["streak"] = 0
                st["last_sender"] = ""

            self._prune(st, now)
            n = len(st["replies"])
            window = self._knob("group_breaker_window_sec", 3)
            max_replies = int(self._knob("group_breaker_max_replies", 10))
            if max_replies > 0 and n >= max_replies:
                reason = f"{window} 秒内已回 {n} 条"
                self._trip(gid, st, now, reason)
                return Decision(False, st["reason"], self._retry_after(st, now), tripped=True)

            streak_max = int(self._knob("group_breaker_streak", 3))
            if streak_max > 0 and sid and st.get("last_sender") == sid \
                    and int(st.get("streak", 0)) >= streak_max:
                reason = f"连续 {st['streak']} 条只回同一个账号 {sid}"
                self._trip(gid, st, now, reason)
                return Decision(False, st["reason"], self._retry_after(st, now), tripped=True)

            self._save()
            return Decision(True)

    def note_reply(self, group_id: str, sender_id: str, now: datetime | None = None) -> None:
        """她真的回了一条（调过模型）之后记一笔。"""
        now = now or datetime.now()
        if not self.enabled or not group_id:
            return
        gid = str(group_id)
        sid = str(sender_id or "")
        with self._lock:
            st = self._group(gid)
            if st.get("last_sender") == sid:
                st["streak"] = int(st.get("streak", 0)) + 1
            else:
                st["streak"] = 1
                st["last_sender"] = sid
            st["last_at"] = now.isoformat(timespec="milliseconds")
            st["replies"].append([now.isoformat(timespec="milliseconds"), sid])
            self._prune(st, now)
            self._save()

    def _escalated_cooldown(self, st: dict, now: datetime, base: int) -> int:
        """半小时内已经熔断 ≥3 次：第 4 次起直接关 10 分钟。"""
        window_min = int(self._knob("group_breaker_repeat_window_min", 30))
        need = int(self._knob("group_breaker_repeat_trips", 3))
        floor = now - timedelta(minutes=window_min)
        recent = [t for t in (st.get("trips") or [])
                  if (self._parse(t) and self._parse(t) > floor)]
        if need > 0 and len(recent) >= need:
            return max(base, int(self._knob("group_breaker_repeat_cooldown_sec", 600)))
        return base

    def _trip(self, gid: str, st: dict, now: datetime, reason: str,
              base_seconds: int | None = None) -> None:
        base = int(base_seconds if base_seconds is not None
                   else self._knob("group_breaker_cooldown_sec", 2))
        seconds = self._escalated_cooldown(st, now, base)
        until = now + timedelta(seconds=seconds)
        st["cooldown_until"] = until.isoformat(timespec="milliseconds")
        st["tripped_at"] = now.isoformat(timespec="milliseconds")
        st["reason"] = reason
        trips = st.setdefault("trips", [])
        trips.append(now.isoformat(timespec="milliseconds"))
        window_min = int(self._knob("group_breaker_repeat_window_min", 30))
        floor = now - timedelta(minutes=window_min)
        st["trips"] = [t for t in trips if self._parse(t) and self._parse(t) > floor]
        self._save()
        self._log(f"群 {gid} 熔断：{reason}；冷却 {seconds} 秒（到 {until:%H:%M:%S}）")

    def _log(self, line: str) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")
        except OSError:
            pass

    # ---------------------------------------------------------- 管理
    def reset(self, group_id: str | None = None) -> None:
        with self._lock:
            if group_id:
                self.groups.pop(str(group_id), None)
            else:
                self.groups.clear()
            self._save()

    def status(self, group_id: str | None = None) -> str:
        with self._lock:
            gids = [str(group_id)] if group_id else list(self.groups)
            lines = []
            now = datetime.now()
            for gid in gids:
                st = self.groups.get(gid)
                if not isinstance(st, dict):
                    continue
                until = self._parse(st.get("cooldown_until"))
                cooling = until and now < until
                state = f"熔断到 {until:%H:%M:%S}" if cooling else "正常"
                floor = now - timedelta(minutes=int(self._knob("group_breaker_repeat_window_min", 30)))
                recent_trips = [t for t in (st.get("trips") or [])
                                if (self._parse(t) and self._parse(t) > floor)]
                lines.append(
                    f"群 {gid}：{state}；窗口内已回 {len(st.get('replies') or [])} 条，"
                    f"连续同一人 {st.get('streak', 0)} 条，半小时内熔断 {len(recent_trips)} 次"
                    + (f"；原因：{st.get('reason')}" if st.get("reason") else "")
                )
            return "\n".join(lines) or "没有熔断记录。"
