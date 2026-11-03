"""好感度：她对群里每个人的熟络程度。

不是记忆，是状态 —— 一个人在**她这里**攒下的印象是「跟他说话我开不开心」，
所以单独落 `data/affinity.json`，**不进事实/人物卡**：
一旦进了记忆，日批会把它当事实，她就会开始编「我跟他关系好」这种没发生过的事。

两件事：

1. 谁在给她好脸色，分数往上走；谁让她难堪，往下掉；没人来往了，慢慢回落到「不熟」。
2. 分数高 + 此刻心情好 → 群里没 @ 她、没叫她名字，她也有点概率接一句。
   这是「熟人插话」，不是「有问必答」：基准概率很低，还有三重闸（试的间隔、
   插话的间隔、每天上限）。
"""

from __future__ import annotations

import json
import math
import random
import threading
from datetime import datetime, timedelta
from pathlib import Path

NEUTRAL = 0.35          # 不熟的人：既不亲近也不讨厌
HALF_LIFE_DAYS = 14.0   # 半个月不联系，关系往回退一半
MAX_CHANCE = 0.5        # 再怎么熟，单次也不会超过一半

_LOCK = threading.RLock()


# ---------------------------------------------------------------- 小工具

def _dt(text) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def _knob(cfg, name: str, default):
    try:
        return getattr(cfg.knobs, name, default)
    except Exception:  # noqa: BLE001 测试桩可能没有 knobs
        return default


def path_for(cfg) -> Path:
    return Path(getattr(cfg, "root", ".")) / "data" / "affinity.json"


def _load(cfg) -> dict:
    try:
        raw = json.loads(path_for(cfg).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    out = {
        "people": raw.get("people") if isinstance(raw.get("people"), dict) else {},
        "groups": raw.get("groups") if isinstance(raw.get("groups"), dict) else {},
    }
    return out


def _save(cfg, data: dict) -> None:
    path = path_for(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _decay(score: float, at: datetime | None, now: datetime) -> float:
    """往「不熟」回。没人来往的关系不该永远停在高位。"""
    if not at:
        return score
    days = max(0.0, (now - at).total_seconds() / 86400.0)
    return NEUTRAL + (score - NEUTRAL) * 0.5 ** (days / HALF_LIFE_DAYS)


# ---------------------------------------------------------------- 读写

def score_of(cfg, person_id: str, now: datetime | None = None) -> float:
    """这个人在她这儿现在几分（0~1，0.35 = 不熟）。只读，不落盘。"""
    now = now or datetime.now()
    person = (_load(cfg).get("people") or {}).get(str(person_id)) or {}
    try:
        raw = float(person.get("score", NEUTRAL))
    except (TypeError, ValueError):
        raw = NEUTRAL
    if not math.isfinite(raw):
        raw = NEUTRAL
    return max(0.0, min(1.0, _decay(raw, _dt(person.get("last")), now)))


def bump(cfg, person_id: str, delta: float, name: str = "",
         now: datetime | None = None) -> float:
    """一件事发生：她对这人的印象动一点。返回新分数。"""
    now = now or datetime.now()
    pid = str(person_id or "")
    if not pid:
        return NEUTRAL
    with _LOCK:
        data = _load(cfg)
        people = data.setdefault("people", {})
        person = people.get(pid) if isinstance(people.get(pid), dict) else {}
        try:
            base = float(person.get("score", NEUTRAL))
        except (TypeError, ValueError):
            base = NEUTRAL
        if not math.isfinite(base):
            base = NEUTRAL
        score = _decay(base, _dt(person.get("last")), now)
        step = float(delta)
        if not math.isfinite(step):
            step = 0.0
        score = max(0.0, min(1.0, score + step))
        people[pid] = {
            "score": round(score, 4),
            "n": int(person.get("n") or 0) + 1,
            "name": str(name or person.get("name") or ""),
            "last": now.isoformat(timespec="seconds"),
        }
        _save(cfg, data)
        return score


def top(cfg, limit: int = 3, now: datetime | None = None) -> list[tuple[str, float]]:
    """最熟的几个人，给 `run.py status` 看的。"""
    now = now or datetime.now()
    rows = []
    for pid, person in (_load(cfg).get("people") or {}).items():
        try:
            raw = float((person or {}).get("score", NEUTRAL))
        except (TypeError, ValueError):
            continue
        score = _decay(raw, _dt((person or {}).get("last")), now)
        if score >= NEUTRAL + 0.05:
            name = str((person or {}).get("name") or pid)
            rows.append((name, round(score, 2)))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[:limit]


# ---------------------------------------------------------------- 情绪 → 好感

WELLBEING_SCALE = 0.004   # 情绪净变化 1 点 = 好感 0.004
LEARN_MAX = 0.06          # 一次最多加 0.06：几句好话刷不满
LEARN_MIN = -0.15         # 一次最多减 0.15：坏印象比好印象快


def learn(cfg, person_id: str, delta: float, name: str = "",
          now: datetime | None = None) -> float:
    """这个人让她的情绪净变多少 → 好感动多少。

    正数（开心/被哄/被夸）加分，负数（被凶/被冒犯）扣分，扣得比加狠。
    情绪的边际效用本来就递减（越夸越麻木），所以这里不用再压。
    """
    now = now or datetime.now()
    try:
        d = float(delta) * WELLBEING_SCALE
    except (TypeError, ValueError):
        return score_of(cfg, person_id, now)
    if not math.isfinite(d):
        return score_of(cfg, person_id, now)   # 坏值不动印象（NaN 以前会变成 +0.06）
    if abs(d) < 0.005:
        return score_of(cfg, person_id, now)   # 没什么起伏就别记
    return bump(cfg, person_id, max(LEARN_MIN, min(LEARN_MAX, d)), name, now)


# ---------------------------------------------------------------- 要不要理他

def reply_factor(cfg, person_id: str, now: datetime | None = None) -> float:
    """被 @ 的时候她有多愿意回（0.25~1.0）。

    0.35（不熟）及以上一律照常回 —— 陌生人问一句她还是会搭腔，那不是「好感低」；
    只有真被她讨厌（低于不熟）才会被晾着，最低 0.25。她不是在群里值班，
    但也不该没礼貌。
    """
    s = score_of(cfg, person_id, now)
    if s >= NEUTRAL:
        return 1.0
    return max(0.25, 1.0 - (NEUTRAL - s) / NEUTRAL * 0.75)


# ---------------------------------------------------------------- 要不要插一句

def should_jump_in(cfg, person_id: str, group_id: str | None,
                   now: datetime | None = None,
                   rng: random.Random | None = None) -> bool:
    """没 @、没叫名字，她要不要接这个人的话。

    四道闸，按便宜的排前面：开关 → 熟不熟 → 心情 → 节奏/上限 → 概率。
    """
    now = now or datetime.now()
    if not bool(_knob(cfg, "group_jump_in_enabled", True)):
        return False
    try:
        min_score = float(_knob(cfg, "group_jump_in_min_score", 0.50))
    except (TypeError, ValueError):
        min_score = 0.50
    try:
        base = float(_knob(cfg, "group_jump_in_chance", 0.06))
    except (TypeError, ValueError):
        base = 0.06

    score = score_of(cfg, person_id, now)
    if score < min_score:
        return False

    # 心情不好就不插话：难过/生气的时候她连自己的话都懒得说。
    try:
        from .emotion import mood_bias

        bias, _why = mood_bias(cfg, now)
    except Exception:  # noqa: BLE001 情绪读不出来就按中性算
        bias = 1.0
    if bias < 0.8:
        return False

    gid = str(group_id or "")
    with _LOCK:
        data = _load(cfg)
        groups = data.setdefault("groups", {})
        state = groups.get(gid) if isinstance(groups.get(gid), dict) else {}
        day = now.strftime("%Y-%m-%d")
        if str(state.get("day") or "") != day:
            state = {"day": day, "jumps": 0,
                     "try_at": state.get("try_at", ""), "last_jump": state.get("last_jump", "")}

        cap = int(_knob(cfg, "group_jump_in_daily_cap", 3) or 0)
        if cap <= 0 or int(state.get("jumps") or 0) >= cap:
            return False
        last_jump = _dt(state.get("last_jump"))
        gap_jump = float(_knob(cfg, "group_jump_in_min_gap_min", 15)) * 60.0
        if last_jump and (now - last_jump).total_seconds() < gap_jump:
            return False
        # 试的间隔：别每来一句就掷一次骰子（群消息量很大）
        last_try = _dt(state.get("try_at"))
        gap_try = float(_knob(cfg, "group_jump_in_try_min_minutes", 3)) * 60.0
        if last_try and (now - last_try).total_seconds() < gap_try:
            return False

        state["try_at"] = now.isoformat(timespec="seconds")
        groups[gid] = state
        _save(cfg, data)

        p = base * min(2.0, (score - min_score) * 4.0) * min(1.6, bias)
        p = max(0.0, min(MAX_CHANCE, p))
        if (rng or random).random() >= p:
            return False

        state["jumps"] = int(state.get("jumps") or 0) + 1
        state["last_jump"] = now.isoformat(timespec="seconds")
        groups[gid] = state
        _save(cfg, data)
        return True
