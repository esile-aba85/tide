"""遗忘曲线：记忆不是「有 / 没有」，是「还多容易想起来」。

艾宾浩斯那条：`R = e^(-Δt / S)`

- **R（可回忆度）**：Δt = 距上次想起（或记下）多少天。R 高 = 一提就想起。
- **S（稳定度，天）**：不是常数。每成功想起一次就变长（间隔重复），而且这次
  隔得越久还想起、涨得越多——这就是「间隔效应」。所以「最近没提过」不等于
  「忘了」，被反复用到的老事反而更牢。
- **重要度**：只影响起点 S，不影响衰减的形状。重要的事一开始就不容易忘。

这里只有纯计算（不碰文件、不查库）：
- 强度存在索引库 `hit` 表（`IndexDB.bump_hit` / `hit_stats`）；
- 淡到一定程度还要归档，在 `hygiene.fix()`；
- 进不进提示词，在 `retrieve.Retriever.recall_facts()`。
"""

from __future__ import annotations

from datetime import date
from math import exp

DEFAULT_STABILITY = 45.0   # 没被想起过的事，大约 45 天掉到 1/e
DEFAULT_GROWTH = 2.0       # 想起来一次，稳定度翻倍
MAX_STABILITY = 365.0      # 再牢的记忆也有上限：一年没碰，仍然开始变淡
UNKNOWN = 0.35             # 没有日期的老数据：跟 recency() 以前给的保底一样

# 重要度只缩放起点：高 ≈ 67 天，中 ≈ 45 天，低 ≈ 29 天。
_BY_IMPORTANCE = {"高": 1.5, "中": 1.0, "低": 0.65}


def stability0(days: float, importance: str = "中") -> float:
    """一条还没被想起过的事，初始稳定度多少天。"""
    base = float(days) if days and float(days) > 0 else DEFAULT_STABILITY
    return max(1.0, base * _BY_IMPORTANCE.get((importance or "中").strip(), 1.0))


def retrievability(stability: float, last: date | None, today: date) -> float:
    """今天这条记忆还剩多少「想得起来」。last = 上次想起（没有则记下的那天）。"""
    if last is None:
        return UNKNOWN
    s = float(stability) if stability and float(stability) > 0 else DEFAULT_STABILITY
    days = max(0, (today - last).days)
    return exp(-days / max(1.0, s))


def review(
    stability: float,
    last: date | None,
    today: date,
    base: float = DEFAULT_STABILITY,
    growth: float = DEFAULT_GROWTH,
    cap: float = MAX_STABILITY,
) -> float:
    """想起一次之后的稳定度。

    涨多少看「想起来之前有多淡」：今天刚提过、R≈1 的再提一次几乎不涨
    （集中重复没用）；隔了 45 天才想起来（R≈0.37）就涨得多一点。
    每天都被用到的事靠这一点点复利慢慢变牢，而不是一次顶满。
    """
    s = max(float(base), float(stability or 0.0))
    r = retrievability(s, last, today)
    # 保底 +12%：天天被用到的事也得慢慢变牢；R 越低，这次巩固越值钱。
    gain = 1.0 + max(0.12, (max(1.0, float(growth)) - 1.0) * (1.0 - r))
    return min(float(cap), s * gain)


def label(r: float) -> str:
    """给人看的档位（体检 / status 用）。"""
    if r >= 0.8:
        return "记得牢"
    if r >= 0.5:
        return "记得"
    if r >= 0.2:
        return "有点淡"
    if r >= 0.05:
        return "快忘了"
    return "基本忘了"
