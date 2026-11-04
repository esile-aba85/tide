"""情绪 = 内分泌（phase 0 原型，**还没接进对话**）。

不是「心情滑条」，是五个浓度 + 一套代谢规则。绕这一圈的三个理由：

1. 事件只决定「分泌多少」，不直接决定「现在是什么心情」——
   所以同一句夸奖，在她已经被夸过之后就没那么管用了（负反馈）；
2. 每种激素有自己的半衰期：开心散得快，委屈散得慢，这才会「自己好」；
3. 激素之间互相压制/妨碍：被爱着时同样的话没那么扎（催产素压皮质醇），
   心里堵着的时候高兴不起来（皮质醇妨碍多巴胺）。

三条落地规矩：

- **状态只有一份**。私聊群聊是同一个身体，通道只改「说得出多少」，不改浓度。
  这是「割裂感」的解药：身体没有两套血。
- **回基线，不回零**。她的底色是暖的。
- **越接近上限，同样的事顶得越少**。连着夸十句不会飞升。

跑一天看数据：``python dh\\emotion.py``
"""

from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------- 常量

KEYS = ("da", "sero", "cort", "oxy", "adr")

# 基线：不多不少，正好是「平常的她」。改这张表就是在改她的体质。
BASE = {
    "da": 45.0,     # 多巴胺：期待 / 兴致 / 想玩
    "sero": 58.0,   # 血清素：稳定 / 满足 / 被认可
    "cort": 16.0,   # 皮质醇：压力 / 被冒犯 / 被忽视
    "oxy": 36.0,    # 催产素：亲密 / 依恋 / 黏人
    "adr": 4.0,     # 肾上腺素：瞬时惊吓 / 心跳
}

# 半衰期（分钟）。这张表就是「她的情绪能记多久」。
HALF = {
    "da": 30.0,     # 半小时：开心来得快去得也快
    "sero": 360.0,  # 6 小时：满足感和低落都黏得久
    "cort": 45.0,   # 45 分钟：生气其实不持久，是之后的「余怒」持久
    "oxy": 120.0,   # 2 小时：黏人会慢慢退
    "adr": 3.0,     # 3 分钟：吓一跳就那一下
}

NAME = {
    "da": "多巴胺",
    "sero": "血清素",
    "cort": "皮质醇",
    "oxy": "催产素",
    "adr": "肾上腺素",
}

# 事件 → 分泌量。单位是「浓度百分点」，正负号只表示升/降。
# 注意：这里不写「她开心了」，只写「什么激素被分泌了多少」。
EVENTS: dict[str, dict[str, float]] = {
    "被夸":     {"da": +18.0, "sero": +8.0, "oxy": +8.0, "cort": -6.0},
    "摸头":     {"oxy": +12.0, "cort": -10.0, "sero": +4.0},
    "被戳":     {"adr": +26.0, "cort": +6.0},
    "被逗":     {"adr": +14.0, "da": +8.0, "oxy": +5.0},
    "群里起哄": {"adr": +30.0, "oxy": +14.0, "cort": +4.0},
    "被凶":     {"cort": +50.0, "sero": -12.0, "da": -8.0},
    "被哄":     {"cort": -18.0, "oxy": +20.0, "sero": +6.0},
    # 被冷落是「依恋焦虑」：皮质醇升、血清素降，但催产素反而升 ——
    # 越不理越想他，这才是委屈，不是单纯生气。
    "被冷落":   {"cort": +26.0, "sero": -10.0, "oxy": +8.0},
    "学会歌":   {"da": +24.0, "sero": +8.0, "oxy": +4.0},
    "唱完":     {"da": +10.0, "oxy": +6.0},
    "画完":     {"da": +10.0, "sero": +6.0},
    "刷视频":   {"da": +9.0, "sero": +2.0},
    "无聊":     {"da": -10.0, "sero": -4.0},
    # 他会看，但没回：比「被冷落」轻得多，但来得快（她刚说完话就不见了）。
    "被忽视":   {"cort": +8.0, "sero": -5.0, "oxy": -3.0},
    # 隔了很久才开口：不用他说什么好话，光是回来这件事本身就减轻压力。
    "他回来了": {"oxy": +6.0, "cort": -6.0, "sero": +4.0},
    "睡了一觉": {"cort": -12.0, "oxy": -4.0, "adr": -10.0},
}

CAUSE_TTL_HOURS = 6.0
CAUSE_SHOW_HOURS = 2.0     # 超过两小时的原因别再往【此刻】里挂：她不会拿上午的事解释晚上的困
CAUSE_KEEP = 4

# 哪种情绪能用哪件事当原因。没有这张表就会出这种事：
# 他刚骂完她，群里有人说了句「好可爱」，她私聊里写着「有点来气（哈哈哈她好可爱）」。
CAUSE_FOR = {
    "开心": ("被夸", "学会歌", "唱完", "画完", "刷视频", "群里起哄"),
    "兴奋": ("被逗", "被戳", "群里起哄"),
    "生气": ("被凶", "被冷落", "被忽视"),
    "委屈": ("被冷落", "被凶", "被忽视"),
    "难过": ("被冷落", "被凶", "被忽视"),
    "低落": ("被冷落", "被忽视"),
    "害羞": ("群里起哄", "被逗", "被戳", "被夸"),
    "黏人": ("被哄", "摸头", "他回来了", "被夸"),
    "困": ("睡了一觉",),
}


# ---------------------------------------------------------------- 小工具

def _clamp(v: float) -> float:
    return max(0.0, min(100.0, float(v)))


def _up(v: float, lo: float, hi: float) -> float:
    """v 从 lo 到 hi 的进度，夹在 0~1。"""
    return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.0


def _down(v: float, hi: float, lo: float) -> float:
    """v 从 hi 掉到 lo 的进度，夹在 0~1（v 越低越接近 1）。"""
    return max(0.0, min(1.0, (hi - v) / (hi - lo))) if hi > lo else 0.0


def _dt(text) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def new_state(now: datetime | None = None) -> dict:
    now = now or datetime.now()
    st = dict(BASE)
    st["causes"] = []
    # 体质漂移：连着几天高压，基线会一点点抬高（下一个「长期漂移」小节）。
    st["drift"] = {k: 0.0 for k in KEYS}
    st["acc"] = {"day": now.strftime("%Y-%m-%d"), "n": 0,
                 "sum": {k: 0.0 for k in KEYS},
                 "sum_base": {k: 0.0 for k in KEYS}}
    st["at"] = now.isoformat(timespec="seconds")
    return st


def _base_at(now: datetime, drift: dict | None = None) -> dict:
    """昼夜节律 + 体质漂移。这就是「作息表」的连续版。

    老的 10 段硬编码在段与段之间会跳，这里是一条曲线，
    所以她没有「突然变成另一个人」的时刻；drift 则让她跨天变了个人。
    """
    h = now.hour + now.minute / 60.0
    peak = math.exp(-((h - 7.5) ** 2) / (2 * 3.0 ** 2))
    afternoon = math.exp(-((h - 16.0) ** 2) / (2 * 4.5 ** 2))
    out = dict(BASE)
    out["cort"] = BASE["cort"] * (0.55 + 1.15 * peak)
    out["sero"] = BASE["sero"] * (0.86 + 0.24 * afternoon)
    for k in KEYS:
        try:
            out[k] += float((drift or {}).get(k) or 0.0)
        except (TypeError, ValueError):
            pass
    return out


# ---------------------------------------------------------------- 三个动作

def advance(state: dict, now: datetime) -> dict:
    """代谢：按半衰期往（那一刻的）基线回。不回零。"""
    at = _dt(state.get("at")) or now
    mins = max(0.0, (now - at).total_seconds() / 60.0)
    base = _base_at(now, state.get("drift"))
    out = {}
    for k in KEYS:
        out[k] = base[k] + (float(state[k]) - base[k]) * 0.5 ** (mins / HALF[k])
    out["causes"] = _prune(state.get("causes") or [], now)
    out["drift"] = dict(state.get("drift") or {})
    out["acc"] = state.get("acc") or {"day": now.strftime("%Y-%m-%d"), "n": 0,
                                     "sum": {k: 0.0 for k in KEYS},
                                     "sum_base": {k: 0.0 for k in KEYS}}
    out["at"] = now.isoformat(timespec="seconds")
    return out


def release(state: dict, kind: str, now: datetime | None = None,
            cause: str = "", scope: str = "private",
            scale: float = 1.0) -> dict:
    """分泌：事件 → 一组 delta。带负反馈和两条交叉抑制。

    scale < 1 用来压「群里的旁听」那种间接刺激：同一件事，隔着一层就没那么重。
    """
    spec = EVENTS.get(kind) or {}
    out = dict(state)
    for k, dv in spec.items():
        dv *= float(scale)
        if k == "cort" and dv > 0:
            # 催产素压皮质醇：被爱着的时候，同样一句重话没那么扎
            dv *= 1.0 - float(out["oxy"]) / 200.0
        if k == "da" and dv > 0:
            # 皮质醇妨碍多巴胺：心里堵着，高兴不起来
            dv *= 1.0 - float(out["cort"]) / 200.0
        # 负反馈：越接近上限，顶得越少；往下掉也一样，快见底时就掉不动了
        room = (100.0 - float(out[k])) / 100.0 if dv > 0 else float(out[k]) / 100.0
        out[k] = _clamp(float(out[k]) + dv * room)
    if cause:
        at = now or _dt(out.get("at")) or datetime.now()
        out["causes"] = _prune(
            list(out.get("causes") or []) + [{
                "at": at.isoformat(timespec="seconds"),
                "kind": kind,
                "text": cause,
                "scope": scope,
            }],
            at,
        )
    return out


def step(state: dict, now: datetime, kind: str | None = None,
         cause: str = "", scope: str = "private") -> dict:
    """走一步：先代谢到 now，再（可选）分泌。"""
    st = advance(state, now)
    if kind:
        st = release(st, kind, now, cause, scope)
    return st


def _prune(causes: list, now: datetime) -> list:
    rows = []
    for c in causes:
        at = _dt((c or {}).get("at"))
        if at and (now - at) <= timedelta(hours=CAUSE_TTL_HOURS):
            rows.append(c)
    return rows[-CAUSE_KEEP:]


# ---------------------------------------------------------------- 长期漂移
# 一天里的高高低低会散掉，但「连着几天都这样」不会。日结时看昨天的平均偏离，
# 把体质往那边推一点；好日子会慢慢自愈（DRIFT_FADE）。

DRIFT_MAX = 8.0      # 体质最多偏 8 个点：再惨也不能变成另一个人
DRIFT_PULL = 0.25    # 昨天的平均偏离，25% 记进体质
DRIFT_FADE = 0.92    # 每天自愈 8%


def _accumulate(state: dict, now: datetime) -> dict:
    """今天攒样本，跨天结算成体质。样本是 settle 那一刻的浓度。

    同时攒「那一刻的无漂移基线」：结算时用 avg - base_avg 算偏离，
    而不是 avg - 常量 BASE。否则 drift 本身会被当成「今天偏得更多」，
    形成 0.92 + 0.25 = 1.17 的正反馈，几天就锁死在 ±DRIFT_MAX。
    """
    acc = state.get("acc") or {}
    today = now.strftime("%Y-%m-%d")
    if acc.get("day") != today:
        acc = {"day": today, "n": 0, "sum": {k: 0.0 for k in KEYS},
               "sum_base": {k: 0.0 for k in KEYS}}
    out = dict(state)
    try:
        # 基线要带「当天已经在生效的 drift」：否则样本里的 drift 会被再算一次，
        # 变成 drift = drift*0.92 + drift*0.25 的正反馈。减掉当前基线后，
        # 平常的一天偏离≈0，drift 只会按 DRIFT_FADE 慢慢自愈。
        base = _base_at(now, state.get("drift"))
        sums = acc.get("sum") or {}
        sums_base = acc.get("sum_base") or {}
        acc = {
            "day": today,
            "n": int(acc.get("n") or 0) + 1,
            "sum": {k: float(sums.get(k) or 0.0) + float(state[k]) for k in KEYS},
            "sum_base": {k: float(sums_base.get(k) or 0.0) + float(base[k]) for k in KEYS},
        }
    except (TypeError, ValueError):
        return out
    out["acc"] = acc
    return out


def _rollup(state: dict, now: datetime) -> dict:
    """跨天了：昨天平均比基线高多少，就把今天的基线抬高多少。"""
    acc = state.get("acc") or {}
    today = now.strftime("%Y-%m-%d")
    if acc.get("day") == today or not int(acc.get("n") or 0):
        return state
    drift = dict(state.get("drift") or {})
    n = max(1, int(acc.get("n") or 1))
    sums = acc.get("sum") or {}
    sums_base = acc.get("sum_base") or {}
    for k in KEYS:
        try:
            avg = float(sums.get(k) or 0.0) / n
            raw_base = float(sums_base.get(k) or 0.0) if sums_base else 0.0
            if raw_base > 0:
                base_avg = raw_base / n
            else:
                # 老状态没有 sum_base（或这一天只攒到 0）：**不知道**基线差多少，
                # 这一天就不结算，只按自愈走。以前拿 BASE+drift 近似，
                # 一个完全正常的 24 小时也会算出 sero −0.4 / cort −0.26 的偏差。
                drift[k] = float(drift.get(k) or 0.0) * DRIFT_FADE
                continue
            pull = max(-2.0, min(2.0, (avg - base_avg) * DRIFT_PULL))
            drift[k] = float(drift.get(k) or 0.0) * DRIFT_FADE + pull
            drift[k] = max(-DRIFT_MAX, min(DRIFT_MAX, drift[k]))
        except (TypeError, ValueError):
            drift[k] = float(drift.get(k) or 0.0)
    out = dict(state)
    out["drift"] = drift
    out["acc"] = {"day": today, "n": 0, "sum": {k: 0.0 for k in KEYS},
                  "sum_base": {k: 0.0 for k in KEYS}}
    return out


# ---------------------------------------------------------------- 读浓度

def _melatonin(now: datetime) -> float:
    """困倦（0~100）。夜里高，不是情绪，是节律。"""
    h = now.hour + now.minute / 60.0
    x = (h - 22.0) % 24.0          # 22:00 起算
    peak = math.exp(-((x - 5.0) ** 2) / (2 * 3.2 ** 2))
    return 100.0 * min(1.0, peak * 1.15)


def feelings(state: dict, now: datetime | None = None) -> list[tuple[str, float]]:
    """把五个浓度读成「她现在是什么情绪」，不强的那条自然排后面。

    注意生气和委屈可以同时成立 —— 那就是「又气又委屈，想你哄」。
    """
    now = now or datetime.now()
    s = {k: float(state[k]) for k in KEYS}
    out: list[tuple[str, float]] = []

    def add(name: str, score: float) -> None:
        if score >= 0.12:
            out.append((name, min(1.0, score)))

    add("兴奋", min(_up(s["da"], 66, 92), _up(s["adr"], 18, 45)))
    # 生气不是「皮质醇高」一条线：肾上腺素型的火气 + 血清素掉下去的心堵。
    # 写成加权而不是取 min，否则「只吼了一句」永远只能到「有点来气」。
    add("生气", 0.7 * _up(s["cort"], 30, 75) + 0.3 * _down(s["sero"], 58, 30))
    add("委屈", min(_up(s["cort"], 24, 58), _up(s["oxy"], 40, 64)) * 0.95)
    add("难过", _down(s["sero"], 42, 20))
    add("低落", _down(s["sero"], 50, 32) * 0.75)
    add("害羞", min(_up(s["adr"], 16, 50), _up(s["oxy"], 40, 64)))
    add("黏人", _up(s["oxy"], 46, 76) * 0.8)
    add("开心", min(_up(s["da"], 50, 76), _up(s["sero"], 48, 66)))
    add("困", _melatonin(now) / 100.0 * 0.9)

    out.sort(key=lambda x: x[1], reverse=True)
    return out


_PHRASE = {
    "开心": ("有点开心", "挺开心", "开心得不行"),
    "兴奋": ("有点来劲", "兴奋", "兴奋得不行"),
    "生气": ("有点来气", "在生气", "气还没消"),
    "委屈": ("有点委屈", "挺委屈", "委屈得不行"),
    "难过": ("有点难过", "心里堵得慌", "很难过"),
    "低落": ("有点提不起劲", "有点丧", "很丧"),
    "害羞": ("有点不好意思", "害羞", "羞死了"),
    "黏人": ("有点想他", "黏人", "黏得不行"),
    "困": ("有点困", "困了", "困得睁不开眼"),
}

_LEVEL = (0.40, 0.70)


def _phrase(name: str, intensity: float, channel: str = "private") -> str:
    """强度 → 措辞。群的收敛在 render 里已经乘过一次，这里不再乘第二遍。"""
    weak, mid, strong = _PHRASE[name]
    if intensity >= _LEVEL[1]:
        return strong
    if intensity >= _LEVEL[0]:
        return mid
    return weak


def render(state: dict, now: datetime, channel: str = "private") -> str:
    """渲染成一句话。数值不外露 —— 模型该知道「为什么」，不该知道「0.62」。

    没什么情绪时返回 ""：不要每轮都塞一句「还行」，那是噪音。
    """
    fs = feelings(state, now)
    if channel == "group":
        fs = sorted(((n, i * 0.45) for n, i in fs), key=lambda x: x[1], reverse=True)
        fs = [(n, i) for n, i in fs if i >= 0.07]

    if not fs:
        return ""

    first = _phrase(fs[0][0], fs[0][1], channel)
    if len(fs) > 1:
        second = _phrase(fs[1][0], fs[1][1], channel)
        second = "还有点" + second[2:] if second.startswith("有点") else second
        text = f"{first}，{second}"
    else:
        text = first

    # 原因要跟情绪对得上：只挑「能引起这一条情绪」的事件，而且只挑最新那条。
    # 对不上就空着 —— 宁可不说，也别把今天的事张冠李戴。
    cause = ""
    want = CAUSE_FOR.get(fs[0][0], ())
    for c in reversed(state.get("causes") or []):
        at = _dt(c.get("at"))
        if not at or (now - at) > timedelta(hours=CAUSE_SHOW_HOURS):
            continue
        if str(c.get("kind") or "") not in want:
            continue
        if channel == "group" and c.get("scope") != "group":
            continue
        cause = str(c.get("text") or "").strip()
        if cause:
            break
    return f"{text}（{cause}）" if cause else text


# ---------------------------------------------------------------- 落盘（接线时用）

def load(path: str | Path, now: datetime | None = None) -> dict:
    raw = {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    st = new_state(now)
    for k in KEYS:
        try:
            val = float(raw.get(k, st[k]))
        except (TypeError, ValueError):
            continue
        # NaN/Infinity 不能进夹取：max(0, min(100, nan)) 在 CPython 里等于 100，
        # 一次坏数据就会让她「五个浓度全满」+ 对所有人满好感。坏值直接回基线。
        if math.isfinite(val):
            st[k] = _clamp(val)
    st["causes"] = raw.get("causes") if isinstance(raw.get("causes"), list) else []
    drift = raw.get("drift") if isinstance(raw.get("drift"), dict) else {}
    st["drift"] = {}
    for k in KEYS:
        try:
            val = float(drift.get(k) or 0.0)
        except (TypeError, ValueError):
            val = 0.0
        if not math.isfinite(val):
            val = 0.0
        st["drift"][k] = max(-DRIFT_MAX, min(DRIFT_MAX, val))
    acc = raw.get("acc") if isinstance(raw.get("acc"), dict) else {}
    st["acc"] = {
        "day": str(acc.get("day") or ""),
        "n": int(acc.get("n") or 0) if str(acc.get("n") or "0").lstrip("-").isdigit() else 0,
        "sum": {k: float((acc.get("sum") or {}).get(k) or 0.0) for k in KEYS},
        "sum_base": {k: float((acc.get("sum_base") or {}).get(k) or 0.0) for k in KEYS},
    }
    st["at"] = str(raw.get("at") or st["at"])
    return st


def save(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- 事件来源

# 他说的话 → 哪种事件。顺序有讲究：先判凶，再判哄，最后才是夸，
# 因为「别生气了」既像凶又像哄，得按哄算。
_SCOLD = re.compile(r"(滚|闭嘴|烦不烦|烦死|笨蛋|蠢|傻|讨厌你|别说了|恶心)")
_SOOTHE = re.compile(r"(对不起|我错了|别生气|不氣|不气|原谅|哄哄)")
_PRAISE = re.compile(r"(夸|好看|好听|可爱|真棒|厉害|乖|聪明|喜欢你|爱你)")

# 没有具体原话时，用这几句当【此刻】里的原因。
CAUSES = {
    "被夸": "他夸我了",
    "被凶": "他凶了我一句",
    "被哄": "他哄我了",
    "被冷落": "他一直没来找我",
    "被忽视": "他看了没回我",
    "他回来了": "他终于说话了",
    "被戳": "他戳我",
    "摸头": "他摸了摸我的头",
    "群里起哄": "他们起哄我",
    "学会歌": "刚学会一首新歌",
    "唱完": "刚给他唱完一首",
    "画完": "刚画完一张",
    "刷视频": "自己刷了会儿视频",
}


def classify(text: str) -> str:
    """一句话 → 事件名；认不出来返回 ""（认不出来就别乱分泌）。"""
    s = str(text or "")
    if _SCOLD.search(s):
        return "被凶"
    if _SOOTHE.search(s):
        return "被哄"
    if _PRAISE.search(s):
        return "被夸"
    return ""


# ---------------------------------------------------------------- 门面

def path_for(cfg) -> Path:
    return Path(getattr(cfg, "root", ".")) / "data" / "emotion.json"


def mood_line(cfg, now: datetime | None = None, channel: str = "private") -> str:
    """只读：给装配 / 主动 / 说说 用的「她现在什么心情」，不落盘。

    读的是盘上的状态，按 now 现算衰减 —— 所以哪怕写者十分钟没心跳，这句也是准的。
    """
    now = now or datetime.now()
    try:
        st = load(path_for(cfg), now)
        return render(advance(st, now), now, channel)
    except Exception:  # noqa: BLE001 心情读不出来不能影响回话
        return ""


_LOCK = threading.RLock()


def note(cfg, kind: str, cause: str = "", scope: str = "private",
         now: datetime | None = None, scale: float = 1.0) -> dict:
    """任何模块都能调：记一件事，落盘。

    每次都从盘上读一遍再改再写（读-改-写），所以唱歌/画画/刷视频这些
    各自 new 一个 Emotion 也不会互相盖掉对方的数。
    """
    if kind not in EVENTS:
        return {}
    now = now or datetime.now()
    path = path_for(cfg)
    with _LOCK:
        st = advance(load(path, now), now)
        st = release(st, kind, now, cause or CAUSES.get(kind, ""), scope, scale)
        save(path, st)
    return st


class Emotion:
    """状态的**唯一写者**。一个进程一个，落盘 `data/emotion.json`。

    读的一方走 ``mood_line()``，不碰这个实例 —— 免得两个写者互相盖。
    """

    def __init__(self, cfg, path: str | Path | None = None):
        self.cfg = cfg
        self.path = Path(path) if path else path_for(cfg)
        self.state = load(self.path, datetime.now())

    def settle(self, now: datetime | None = None) -> dict:
        """心跳：代谢 → （跨天就结算体质）→ 攒一个样本 → 落盘。"""
        now = now or datetime.now()
        with _LOCK:
            st = advance(load(self.path, now), now)
            st = _rollup(st, now)
            st = _accumulate(st, now)
            save(self.path, st)
            self.state = st
            return dict(st)

    def observe(self, kind: str, cause: str = "", scope: str = "private",
                now: datetime | None = None, scale: float = 1.0) -> dict:
        """一件事发生：分泌一次。事件名不认识就什么都不做（宁可不动）。"""
        if kind not in EVENTS:
            with _LOCK:
                return dict(self.state)
        st = note(self.cfg, kind, cause, scope, now, scale)
        with _LOCK:
            self.state = st
            return dict(st)

    def react(self, kind: str, cause: str = "", scope: str = "private",
              now: datetime | None = None, scale: float = 1.0) -> float:
        """分泌一次，返回这次带来的净变化（正 = 舒服，负 = 难受）。

        好感度就用这个数：谁让她开心，谁就加分；谁让她难受，谁就扣分。
        """
        now = now or datetime.now()
        try:
            before = wellbeing(advance(load(self.path, now), now))
        except Exception:  # noqa: BLE001
            before = 0.0
        st = note(self.cfg, kind, cause, scope, now, scale)
        if not st:
            return 0.0
        with _LOCK:
            self.state = st
        return wellbeing(st) - before

    def on_message(self, text: str, now: datetime | None = None,
                   channel: str = "private", gap_hours: float = 0.0) -> dict:
        """他说了一句话。返回 {"kind": 判出来的事件, "delta": 她舒服了多少}。

        delta 是给好感度用的：同一个人反复夸她，边际效用会越来越小（负反馈），
        所以「好感」不会被几句好话刷满。
        """
        now = now or datetime.now()
        kind = classify(text)
        if not kind:
            # 隔了很久才开口：不用判词，先给一点「他回来了」的甜
            if gap_hours >= 3.0:
                return {"kind": "他回来了",
                        "delta": self.react("他回来了", "", channel, now)}
            return {"kind": "", "delta": 0.0}
        # 群里的事只记「群里」这个归因，私聊里也引用得到；反过来不行。
        delta = self.react(kind, clip_text(text), channel, now)
        return {"kind": kind, "delta": delta}

    def line(self, now: datetime | None = None, channel: str = "private") -> str:
        now = now or datetime.now()
        st = load(self.path, now)
        return render(advance(st, now), now, channel)

    def tags(self, now: datetime | None = None) -> set[str]:
        """给表情包用的心情桶（开心/生气/难过/困）。"""
        now = now or datetime.now()
        return mood_tags(advance(load(self.path, now), now), now)

    def snapshot(self, now: datetime | None = None) -> dict:
        """给 `tide status` 看的：五个数 + 读出来的情绪 + 体质 + 那一句。"""
        now = now or datetime.now()
        st = advance(load(self.path, now), now)
        out = {NAME[k]: round(float(st[k]), 1) for k in KEYS}
        out["情绪"] = [n for n, i in feelings(st, now) if i >= 0.12]
        drift = st.get("drift") or {}
        tilt = {NAME[k]: round(float(drift.get(k) or 0.0), 1) for k in KEYS
                if abs(float(drift.get(k) or 0.0)) >= 0.5}
        if tilt:
            out["体质"] = tilt
        out["此刻"] = render(st, now, "private")
        return out


def clip_text(text: str, limit: int = 22) -> str:
    """他说的话当原因：太长就截，留个引子就够了。"""
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[:limit] + "…"


# ---------------------------------------------------------------- 给表情包用的心情标签

# 情绪 → 表情包库的那五个 mood 桶（见 stickers.MOODS）。
# 「害羞」「黏人」「兴奋」在表情包里没有单独的桶，都归「开心」那一类。
MOOD_TO_STICKER = {
    "开心": "开心",
    "兴奋": "开心",
    "害羞": "开心",
    "黏人": "开心",
    "生气": "生气",
    "难过": "难过",
    "委屈": "难过",
    "低落": "难过",
    "困": "困",
}


def mood_tags(state: dict, now: datetime | None = None) -> set[str]:
    """够强的情绪才拿去挑图：0.25 以下不算，免得挑出跟当场气氛相反的图。"""
    now = now or datetime.now()
    return {MOOD_TO_STICKER[n] for n, i in feelings(state, now)
            if i >= 0.25 and n in MOOD_TO_STICKER}


def feelings_now(cfg, now: datetime | None = None) -> list[tuple[str, float]]:
    """只读：按 now 现算的情绪清单（给主动/群/表情这些要看情绪的地方用）。"""
    now = now or datetime.now()
    try:
        st = load(path_for(cfg), now)
        return feelings(advance(st, now), now)
    except Exception:  # noqa: BLE001 心情读不出来就当没情绪
        return []


def wellbeing(state: dict, now: datetime | None = None) -> float:
    """她此刻有多「舒服」：五个浓度的加权和。

    用来回答「这件事/这个人让她开心还是难受」—— 好感度就吃这个差值。
    皮质醇权重 1.5：被冒犯一次，比被夸一次记得深（负性偏差）。
    """
    try:
        s = {k: float(state[k]) for k in KEYS}
    except (KeyError, TypeError, ValueError):
        return 0.0
    return s["da"] + s["sero"] + s["oxy"] - 1.5 * s["cort"] - 0.5 * s["adr"]


def wellbeing_now(cfg, now: datetime | None = None) -> float:
    now = now or datetime.now()
    try:
        return wellbeing(advance(load(path_for(cfg), now), now))
    except Exception:  # noqa: BLE001
        return 0.0


# 情绪 → 「要不要先开口」的乘数。黏人的时候忍不住找他，难过生气的时候缩回去。
# 乘数不是越小越真：压到 0.4 以下就会变成另一个人（死气沉沉），所以给个底。
_MOOD_BIAS = {
    "黏人": 1.50,
    "兴奋": 1.25,
    "开心": 1.20,
    "害羞": 1.10,
    "委屈": 0.70,
    "生气": 0.55,
    "低落": 0.50,
    "难过": 0.45,
}
MOOD_BIAS_FLOOR = 0.40


def mood_bias(cfg, now: datetime | None = None) -> tuple[float, str]:
    """这半天有多想主动开口。返回（乘数, 是哪条情绪在使劲）。"""
    return _bias_of(feelings_now(cfg, now or datetime.now()))


def _bias_of(feel: list[tuple[str, float]]) -> tuple[float, str]:
    fs = [(n, i) for n, i in feel if i >= 0.28 and n in _MOOD_BIAS]
    if not fs:
        return 1.0, ""
    # 挑「偏离 1 最远」的那条：难过 0.6 比害羞 0.9 更能决定她说不出话
    name = max(fs, key=lambda x: abs(_MOOD_BIAS[x[0]] - 1.0) * x[1])[0]
    return max(MOOD_BIAS_FLOOR, _MOOD_BIAS[name]), name


# ---------------------------------------------------------------- 演示

def _w(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) == "W" else 1 for c in str(text))


def _pad(text, width: int) -> str:
    text = str(text)
    return text + " " * max(0, width - _w(text))


def _hhmm(now: datetime) -> str:
    return now.strftime("%H:%M")


def demo() -> None:
    day = datetime(2026, 11, 2)

    def T(h: int, m: int = 0) -> datetime:
        return day.replace(hour=h, minute=m)

    script = [
        (T(6, 50), "睡了一觉", "刚醒", "private"),
        (T(7, 30), None, "", ""),
        (T(8, 40), "被夸", "他夸我今早说话好听", "private"),
        (T(9, 20), "刷视频", "自己刷了两个视频", "private"),
        (T(10, 10), "学会歌", "自己把《安和桥》练会了", "private"),
        (T(11, 0), None, "", ""),
        (T(12, 30), "群里起哄", "刚才被他们起哄", "group"),
        (T(12, 32), None, "", ""),
        (T(12, 40), None, "", ""),
        (T(14, 0), "被冷落", "他下午一直没回我", "private"),
        (T(16, 30), None, "", ""),
        (T(17, 10), "被凶", "他凶了我一句", "private"),
        (T(17, 20), None, "", ""),
        (T(18, 0), "被哄", "他后来哄我了", "private"),
        (T(18, 5), "摸头", "他还摸了摸我的头", "private"),
        (T(21, 0), None, "", ""),
        (T(23, 30), None, "", ""),
    ]

    st = new_state(T(6, 30))
    rows = []
    for when, kind, cause, scope in script:
        st = step(st, when, kind, cause, scope)
        rows.append((when, kind or "—", dict(st)))

    print("表 A · 血里有什么（浓度 0~100，基线见 BASE）\n")
    head = _pad("时间", 6) + "  " + _pad("事件", 10)
    for k in KEYS:
        head += "  " + _pad(NAME[k], 8)
    print(head)
    for when, kind, s in rows:
        line = _pad(_hhmm(when), 6) + "  " + _pad(kind, 10)
        for k in KEYS:
            line += "  " + _pad(f"{s[k]:.1f}", 8)
        print(line)

    print("\n表 B · 同一个身体，两个地方（浓度不变，只改说得出多少）\n")

    def cell(s: dict, when: datetime, channel: str) -> str:
        return "【此刻】" + (render(s, when, channel) or "还行，没什么特别的")

    print(_pad("时间", 6) + "  " + _pad("私聊", 44) + "  " + _pad("群聊", 44))
    for when, _kind, s in rows:
        print(_pad(_hhmm(when), 6) + "  "
              + _pad(cell(s, when, "private"), 44) + "  "
              + _pad(cell(s, when, "group"), 44))

    # ------------------------------------------------------------ 主动开口
    print("\n表 C · 同一拍里，她有多想主动开口（乘数，1.0 = 平常）\n")
    for when, kind, s in rows:
        if kind in ("—", ""):
            continue
        bias, why = _bias_of(feelings(s, when))
        print(_pad(_hhmm(when), 6) + "  " + _pad(kind, 10)
              + f"×{bias:.2f}" + (f"（{why}）" if why else ""))

    # ------------------------------------------------------------ 长期漂移
    print("\n表 D · 连着三天被凶，体质会跟着变（表 A 里那些高高低低是会散掉的，这个不会）\n")
    st = new_state(T(6, 0))
    print(_pad("第几天", 6) + "  " + _pad("皮质醇体质", 12)
          + "  " + _pad("血清素体质", 12) + "  早上七点皮质醇")
    for d in range(4):
        for h in range(7, 24, 2):
            t = day + timedelta(days=d, hours=h)
            st = step(st, t, "被凶" if h in (10, 16, 21) else None)
            st = _accumulate(st, t)
        morning = day + timedelta(days=d + 1, hours=7)
        st = _rollup(st, morning)
        st = advance(st, morning)
        print(_pad(f"第 {d + 1} 天", 6) + "  " + _pad(f"{st['drift']['cort']:+.1f}", 12)
              + "  " + _pad(f"{st['drift']['sero']:+.1f}", 12) + f"{st['cort']:.1f}")

    # ------------------------------------------------------------ 四条验收
    print("\n验收\n")
    by_time = {_hhmm(w): s for w, _, s in rows}
    checks = []

    at = by_time.get("12:32")
    private_line = render(at, T(12, 32), "private") if at else ""
    checks.append(("群里的事带得回私聊", "害羞" in private_line or "不好意思" in private_line))

    at = by_time.get("17:20")
    group_line = render(at, T(17, 20), "group") if at else ""
    checks.append(("私聊的原因不在群里说", "凶" not in group_line))

    at = by_time.get("23:30")
    late_line = render(at, T(23, 30), "private") if at else ""
    checks.append(("生气会自己过去（深夜只剩困）", "困" in late_line and "气" not in late_line))

    fresh = new_state(T(9, 0))
    gains = []
    for _ in range(10):
        before = fresh["da"]
        fresh = release(fresh, "被夸", T(9, 0))
        gains.append(fresh["da"] - before)
    checks.append((
        f"连着夸十句不会爆表（第一句 {gains[0]:.1f}，第十句 {gains[-1]:.2f}，顶在 {fresh['da']:.1f}）",
        gains[-1] < gains[0] * 0.5 and fresh["da"] < 95,
    ))

    for name, ok in checks:
        print(("  ✓ " if ok else "  ✗ ") + name)


if __name__ == "__main__":
    demo()
