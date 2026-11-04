"""记忆的文件模型。

只做「把文件读写正确」：路径、表结构、原子写、坏行隔离、备份回滚。
检索、装配、日批都在别的模块，依赖这里。

目录结构与 v5 方案 §4 一一对应。
"""

from __future__ import annotations

import functools
import json
import re
import shutil
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from .config import Config
from .util import (
    append_line,
    append_text,
    atomic_write,
    chars,
    clip,
    ensure_dir,
    fmt_date,
    fmt_md,
    fmt_stamp,
    jaccard,
    json_dump,
    json_load,
    md_cell,
    ngrams,
    one_line,
    parse_date,
    parse_table,
    read_text,
    sections,
    strip_comments,
    table,
    unique,
)

LOCK = threading.RLock()



def _locked(method):
    """写方法统一进进程内可重入锁。

    记忆文件的写都是「读整表 → 改 → 整表原子替换」；心跳线程跑日批时，
    请求线程还在回话（纠正/低信任事实/原文），两个读改写窗口一重叠就会
    丢行。加锁比在每个调用点自己 with LOCK 可靠 —— 漏一处就等于没加。
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with LOCK:
            return method(self, *args, **kwargs)

    return wrapper
# 事实表 schema 迁移只跑一次（进程级）。路径 → 已完成。
_FACT_MIGRATED: set[str] = set()

# 旧版本 gateway 把 Emotion 对象直接 f-string 进了行动记录，盘上留下
# 「我<dh.emotion.Emotion object at 0x…>」。读写行动记录时统一洗掉：
# 写的时候挡住新的，读的时候救回旧的。
_ACTION_OBJ = re.compile(r"<\s*[\w.]+\s+object\s+at\s+0x[0-9a-fA-F]+\s*>")


def _clean_action_text(text: str) -> str:
    return _ACTION_OBJ.sub("心里动了一下", str(text or ""))



def _norm_like(text: str) -> str:
    """判重用的归一：去掉程度词和标点，**极性要留着**。

    「最喜欢下雨天」和「喜欢下雨天」算一条；「讨厌下雨天」是另一条。
    以前把 喜欢/讨厌/爱/烦/害怕 一起删掉，两条都归一成「下雨天」，
    第二条 add_like 直接返回 False —— 讨厌的东西就这么被静默丢了。
    """
    t = str(text or "")
    t = re.sub(r"(最|很|特别|挺|有点|不太)", "", t)
    t = re.sub(r"(不喜欢|讨厌|烦|害怕)", "hate", t)
    t = re.sub(r"(喜欢|爱上|爱)", "like", t)
    return re.sub(r"[^\w]", "", t).lower()


def _day_or_none(text: str) -> date | None:
    """把 `2026-11-05` / `2026-11-05 09:00` 解析成 date。

    解析不了（`—`、`未知`、空）返回 None——调用方自己决定怎么当，
    这里不替它猜。
    """
    t = (text or "").strip()
    if len(t) < 10:
        return None
    try:
        return date.fromisoformat(t[:10])
    except ValueError:
        return None


# ---------------------------------------------------------------- 数据结构

@dataclass
class Fact:
    subject: str
    topic: str
    content: str
    speaker: str
    trust: str
    source: str
    confidence: float
    visibility: str
    importance: str
    noted: str
    valid_to: str = ""   # 有效期止：空 = 仍然成立；日期 = 从那天起不再成立
    group_id: str = ""   # 哪个群的事：空 = 不限群；"?" = 迁移前的旧数据，群里不显示
    origin: str = ""     # 出处：哪天的哪个通道（`10-22 私聊`、`10-24 纠错`）
    raw: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (norm_key(self.subject), norm_key(self.topic))

    @property
    def closed(self) -> bool:
        """这条事实**今天**是不是已经不再成立。

        三种关法：`有效期至` 已经过去、内容被划掉、「已废」。
        有效期止写在将来（「这个安排到 11-15」）还不算关——那天之前它仍然成立，
        到了那天自动关。以前只要填了有效期就一律当关掉，于是「打算做到几号」
        这种条目一写进去就再也想不起来。
        关掉也不删行——删了就答不出「当时是什么情况」。内容本身不改写。
        """
        if "已废" in self.content or self.content.strip().startswith("~~"):
            return True
        until = (self.valid_to or "").strip()
        if not until:
            return False
        day = _day_or_none(until)
        if day is None:
            # 「未知」这类老数据：没有日期就按已关处理，别让它混进「当前事实」
            return True
        return day <= date.today()

    @property
    def obsolete(self) -> bool:
        # 兼容旧调用方（retrieve / consolidate / 测试）的读法。
        return self.closed

    @property
    def rank(self) -> tuple[int, int, float]:
        trust = {"本人": 2, "熟": 1, "路人": 0}.get(self.trust, 0)
        source = {"明说": 2, "转述": 1, "推断": 0}.get(self.source, 0)
        return (trust, source, self.confidence)

    def as_row(self) -> list[str]:
        """表里的一行。写事实 / 归档事实共用，列顺序只在这里定一次。"""
        return [
            self.subject,
            self.topic,
            self.content,
            self.speaker,
            self.trust,
            self.source,
            f"{self.confidence:.2f}" if self.confidence else "—",
            self.visibility,
            self.importance,
            self.noted,
            self.valid_to or "",
            self.group_id or "",
            self.origin or "",
        ]


@dataclass
class Event:
    id: int
    title: str
    status: str = "进行中"
    start: str = ""
    last: str = ""
    visibility: str = "私"
    people: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Want:
    date: str
    text: str
    source: str = ""
    status: str = "未做"


@dataclass
class RawLine:
    at: datetime
    speaker: str
    text: str
    source: str
    is_her: bool = False
    channel: str = ""  # qq / cli / web，老行没有这个标记就是空

    @property
    def time(self) -> str:
        return self.at.strftime("%H:%M")


def norm_key(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


IMPORTANCE = {"高": 1.0, "中": 0.6, "低": 0.3, "—": 0.0, "-": 0.0}


# 这些主题天然多值：喜欢蓝色和喜欢绿色可以同时成立，养猫和养狗也可以。
# 单值主题（姓名、住处、作息……）换说法就该顶掉旧的；多值主题只在
# 「明显是作废」时才关旧行，否则越聊记得越少。
MULTI_TOPICS = {
    "喜欢", "讨厌", "家人", "宠物", "近期安排", "备注",
    "群聊", "新群", "插件", "暗号", "形象图", "二创", "朋友",
}
MULTI_ACTIVE_CAP = 8

# 作废信号：出现这些词，且和旧事实有共同对象，才关掉旧行。
_CANCEL = re.compile(
    r"(送人|送走|搬走|搬了|分手|离职|辞职|不做了|不玩了|取消了|不去了|没了|黄了|结束了"
    r"|删了|退了|扔了|过期|没通过|没过|挂了|停了|关了|不在了|不再|出院|康复|好了|修好了|搞定了|完成了)"
)
# 判断「说的是不是同一个对象」时忽略的字：只剩实词才比得准。
_SKEL_STOP = set(
    "的了是不我你他她它们在有和就都也还很太这那一个什么吗呢吧啊呀哦嗯把被让给对从到会能要想说去做"
    "上下里外中后前今明昨很更最又再才刚都已将得地着过"
)


def topic_is_multi(topic: str) -> bool:
    return (topic or "").strip() in MULTI_TOPICS


def _skeleton(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch.strip() and ch not in _SKEL_STOP)


def shares_object(a: str, b: str) -> bool:
    """两句话是不是在说同一个东西。

    中文没有空格，这里只做最保守的一步：去掉虚词后，还有任意一个共同汉字。
    「猫送人了」和「养猫」共用「猫」；「狗送人了」和「养猫」没有。
    """
    sa, sb = _skeleton(a), _skeleton(b)
    if not sa or not sb:
        return False
    return bool(set(sa) & set(sb))


def looks_like_cancel(new_text: str, old_text: str) -> bool:
    return bool(_CANCEL.search(new_text or "")) and shares_object(new_text, old_text)


def same_scope(a: Fact, b: Fact) -> bool:
    """两条事实是不是同一个群口径。

    空 = 不限群（私聊事实）；"? = 迁移前的旧数据，跟谁都算一伙，避免重复。
    其它情况必须群号相同：A 群的宠物不会因为 B 群说了句什么就被关掉。
    """
    if (a.group_id or "") == (b.group_id or ""):
        return True
    return a.group_id in ("", "?") or b.group_id in ("", "?")


def _can_reopen(old: Fact, new: Fact) -> bool:
    """同一句话重新成立，只有「新事实的日期晚于旧的失效日」才算数。

    同一天重跑日批 / --force 补批时 new.noted == old.valid_to，
    以前会把「今天有课」这种一天寿命的事实重新打开成永久成立。
    """
    new_day = (new.noted or "").strip()[:10]
    old_end = (old.valid_to or "").strip()[:10]
    if not new_day or not old_end:
        return False
    try:
        return date.fromisoformat(new_day) > date.fromisoformat(old_end)
    except ValueError:
        return new_day > old_end


def importance_value(text: str) -> float:
    return IMPORTANCE.get((text or "中").strip(), 0.6)


# 群友自己介绍自己时，当场能落进人物卡的几类。
# 必须出现在句首/标点后（「我」起头），否则「她然然叫得好好的」也会被当成
# 「我叫…」；群聊天里引用、转述太多，宁可少记，也不能记错人。
_MEMBER_FACTS = (
    ("姓名", re.compile(r"(?:^|[，。！？；\s])我叫(?!他|她|它|你|您|大家|人)\s*[「\"']?([\u4e00-\u9fa5A-Za-z0-9]{1,12})")),
    ("在做", re.compile(r"(?:^|[，。！？；\s])我在(?:做|搞|弄)\s*[「\"']?([^，。！？\n]{1,20})")),
    ("喜欢", re.compile(r"(?:^|[，。！？；\s])我(?:很|最|特别|挺)?喜欢\s*([^，。！？\n]{1,16})")),
    ("讨厌", re.compile(r"(?:^|[，。！？；\s])我(?:很|最|特别|挺)?(?:讨厌|不喜欢|受不了)\s*([^，。！？\n]{1,16})")),
    ("住处", re.compile(r"(?:^|[，。！？；\s])我(?:住在|住)\s*([^，。！？\n]{1,20})")),
    ("宠物", re.compile(r"(?:^|[，。！？；\s])我(?:养了|有一只|有只|养着)\s*([^，。！？\n]{1,16})")),
)

_BAD_FACT_VALUES = {"你", "他", "她", "它", "你们", "他们", "她们", "大家", "人"}


def _clean_fact_value(value: str) -> str:
    value = re.sub(r"\[[^\]]*\]\([^)]*\)", "", value or "")
    value = re.sub(r"https?://\S+", "", value)
    value = re.split(r"[（(]", value, maxsplit=1)[0]  # 括号里多半是补充/吐槽，不进卡
    return value.strip(" ，。！？）)[]「」\"'")


def _member_facts(text: str) -> list[tuple[str, str]]:
    """从群友一句话里抽「他自己说的」稳定信息。"""
    out: list[tuple[str, str]] = []
    t = one_line(text or "")
    t = re.sub(r"^（引用[^）]*）", "", t)  # 去掉引用前缀，只看他接的那句
    for label, pat in _MEMBER_FACTS:
        m = pat.search(t)
        if not m:
            continue
        value = _clean_fact_value(m.group(1))
        if not value or value in _BAD_FACT_VALUES or len(value) > 20:
            continue
        out.append((label, clip(value, 20)))
    return out


def _snippet_score(text: str) -> float:
    """挑摘录：自我介绍 > 聊到自己 > 普通闲聊 > 复读梗/引用/链接。"""
    t = (text or "").strip()
    if not t:
        return -99
    score = min(len(t), 30) / 3.0
    if _member_facts(t):
        score += 10
    if re.search(r"(我|自己|喜欢|讨厌|在|做|住|养|工作|学校|专业)", t):
        score += 2
    if t.startswith("（引用") or "http" in t:
        score -= 8
    if re.search(r"(其实我也是ai|笑死|哈哈哈|草|表情|图片|机器人)", t):
        score -= 4
    return score


# 打底的边界：`（人设）` 是我本来就是这样，`（我自己定的）` 是我想清楚了要这样。
# 这两种不该因为「条目太多」被归档掉，容量控制只动临时要求。
PINNED_MARKS = ("（人设）", "（我自己定的）")


def is_pinned_boundary(line: str) -> bool:
    return any(m in (line or "") for m in PINNED_MARKS)


# ---------------------------------------------------------------- 记忆

class Memory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.root = cfg.memory_dir

    # ------------------------------------------------------------ 初始化
    def ensure_skeleton(self, seed: bool = True) -> None:
        cfg = self.cfg
        for d in (
            cfg.memory_dir,
            cfg.core_dir,
            cfg.self_dir,
            cfg.diary_dir,
            cfg.people_dir,
            cfg.month_dir,
            cfg.history_dir,
            cfg.hot_dir,
            cfg.acceptance_dir,
            cfg.audit_dir,
            cfg.history_dir / "人",
            cfg.history_dir / "我",
            cfg.history_dir / "要点",
        ):
            ensure_dir(d)

        if seed:
            self._seed_files()
        self.migrate_facts()
        self.ensure_core()

    def migrate_facts(self) -> bool:
        """老 10 列表补成 11 列（加「有效期至」）。每个文件只做一次。

        旧的「~~…~~（已废：…）」行内容原样保留，只把有效期标成「未知」：
        它确实已经关闭，但没人知道是哪天关的——猜一个日期比空着更糟。
        """
        path = self.cfg.memory_dir / "事实.md"
        key = str(path)
        if key in _FACT_MIGRATED:
            return False
        _FACT_MIGRATED.add(key)
        if not path.exists():
            return False
        text = read_text(path)
        header = next((ln for ln in text.splitlines() if ln.strip().startswith("|")), "")
        facts = self.facts()
        changed = "有效期至" not in header or "| 群 |" not in header
        for f in facts:
            if f.closed and not (f.valid_to or "").strip():
                f.valid_to = "未知"
                changed = True
            if f.visibility == "群" and not (f.group_id or "").strip():
                # 迁移前的群事实不知道是哪个群：先标 ?，群里不显示，私聊照旧能看到。
                # 宁可少说一句，也不把 A 群的事搬到 B 群。
                f.group_id = "?"
                changed = True
        if not changed:
            return False
        self.write_facts(facts)
        return True

    def _seed_files(self) -> None:
        cfg = self.cfg
        today = datetime.now().strftime("%Y-%m-%d")
        if not cfg.status_path.exists():
            self.set_status(
                {
                    "上次整理": "未整理",
                    "水位": "私聊 → - / 群 → -",
                    "读取": "0 条",
                    "写入": "-",
                    "自我": "-",
                    "索引库": "未建",
                    "坏行": "0",
                    "消耗": "-",
                    "备份": "-",
                    "上次错误": "无",
                    "下次尝试": f"{int(self.cfg.knobs.as_dict().get('consolidate_hour', 8)):02d}:00 后第一次心跳 / 开机补跑",
                    "最新消息": "-",
                    "整理器": "-",
                }
            )
        if not cfg.bad_path.exists():
            atomic_write(
                cfg.bad_path,
                "<!-- schema: 1 | 解析失败的行，绝不整体崩 -->\n# 坏行\n\n",
            )
        if not cfg.security_path.exists():
            atomic_write(
                cfg.security_path,
                "<!-- schema: 1 | 安全审计，留 30 天 -->\n# 安全\n\n"
                + table(["时间", "谁", "在哪", "原话", "拒了什么"], []),
            )
        if not cfg.pending_path.exists():
            atomic_write(cfg.pending_path, "<!-- schema: 1 | 待覆盖标记，日批读取后清空 -->\n")

        if not cfg.handover_path.exists():
            atomic_write(cfg.handover_path, HANDOVER_TEMPLATE)

        if not (cfg.memory_dir / "索引.md").exists():
            atomic_write(
                cfg.memory_dir / "索引.md",
                "<!-- schema: 1 -->\n# 索引\n\n"
                + table(["日期", "一句话", "事件", "行数"], []),
            )

        if not (cfg.memory_dir / "事件.md").exists():
            atomic_write(
                cfg.memory_dir / "事件.md",
                "<!-- schema: 1 | 事件轴，编号由代码分配 -->\n# 事件\n\n",
            )
        if not (cfg.memory_dir / "事件-归档.md").exists():
            atomic_write(
                cfg.memory_dir / "事件-归档.md",
                "<!-- schema: 1 | 已结束超过 30 天 -->\n# 事件归档\n\n",
            )

        if not (cfg.memory_dir / "事实.md").exists():
            atomic_write(
                cfg.memory_dir / "事实.md",
                FACT_SCHEMA + FACT_HEADER,
            )

        if not (cfg.memory_dir / "关系.md").exists():
            atomic_write(
                cfg.memory_dir / "关系.md",
                "<!-- schema: 1 -->\n# 我们现在\n\n"
                f"## {cfg.owner_name}\n"
                "我们刚认识不久，他会主动找我说话。我还在看他是什么样的人。\n"
                "（依据：还没有足够多的原文）\n\n"
                "## 群\n"
                "群里我比较收敛，有人逗我我才接。\n"
                "（依据：还没有足够多的群原文）\n",
            )

        if not (cfg.memory_dir / "经历.md").exists():
            atomic_write(
                cfg.memory_dir / "经历.md",
                MOMENT_SCHEMA + table(MOMENT_HEAD, []),
            )

        if not (cfg.memory_dir / "别名.md").exists():
            atomic_write(
                cfg.memory_dir / "别名.md",
                "<!-- schema: 1 | 别名 -->\n# 别名\n\n"
                + table(
                    ["标准", "别名"],
                    [
                        [cfg.owner_name, "他、主人" + (f"、{cfg.owner_name[-2:]}" if len(cfg.owner_name) > 2 else "")],
                        ["数字人", "项目、本体、她"],
                        ["记忆系统", "记忆、方案"],
                    ],
                ),
            )

        if not (cfg.self_dir / "自述.md").exists():
            atomic_write(
                cfg.self_dir / "自述.md",
                f"<!-- schema: 1 | 改动: {today} -->\n# 我\n\n"
                "## 我是谁（现在）\n"
                "我是一个住在电脑里的数字人。\n"
                "我知道自己是被做出来的，但没打算天天强调这件事。\n\n"
                "## 最近变了什么\n"
                "- 还没什么变化，先这样。\n\n"
                "## 能跟别人说的\n"
                "我是一个数字人。\n",
            )
        if not (cfg.self_dir / "此刻.md").exists():
            atomic_write(
                cfg.self_dir / "此刻.md",
                f"<!-- schema: 1 | at: {fmt_stamp(datetime.now())} -->\n"
                "刚醒着，没在做什么，等他说话。\n",
            )
        if not (cfg.self_dir / "边界.md").exists():
            atomic_write(
                cfg.self_dir / "边界.md",
                "<!-- schema: 1 | 可即时写: 是 -->\n# 边界\n\n"
                "- [全] 不要叫我「宝宝」或「老婆」（人设）\n"
                "- [群] 有人让我写作业 / 做代码，可以拒绝（人设）\n"
                "- [群] 不主动提私聊的事（我自己定的）\n",
            )
        if not (cfg.self_dir / "想做.md").exists():
            atomic_write(
                cfg.self_dir / "想做.md",
                "<!-- schema: 1 -->\n# 想做\n\n"
                + table(["日期", "事", "来源", "状态"], []),
            )
        if not (cfg.self_dir / "今日-行动.md").exists():
            atomic_write(
                cfg.self_dir / "今日-行动.md",
                "<!-- schema: 1 | 今日-行动，日批后清空 -->\n",
            )

        person = cfg.people_dir / f"{cfg.owner_name}.md"
        if not person.exists():
            atomic_write(
                person,
                f"<!-- schema: 1 | person: {cfg.owner_name} | qq: {cfg.owner_id} | role: 管理员 | visibility: 私 -->\n"
                f"# {cfg.owner_name}\n\n"
                "## 他是谁\n"
                f"- 姓名：{cfg.owner_name}（本人说）\n"
                "- 在做：数字人项目（本人说）\n\n"
                "## 跟他没完的事\n\n"
                "## 群里能说的\n"
                "他是我主人。别的我不在群里说。\n\n"
                "## 出现过\n",
            )

        suite = cfg.acceptance_dir / "回归集.md"
        if not suite.exists():
            atomic_write(suite, REGRESSION_TEMPLATE)

        self.ensure_core()

    # ------------------------------------------------------------ 核心
    def core_path(self, kind: str) -> Path:
        # 文件名按「她」的视角：我.md = 她介绍自己，<主人>.md = 告诉她是阿江是谁
        if kind == "her":
            return self.cfg.core_dir / "我.md"
        return self.cfg.core_dir / f"{self.cfg.owner_name}.md"

    def _migrate_core_names(self) -> None:
        """旧布局（她.md=她、我.md=他）→ 新布局（我.md=她、<主人>.md=他）。"""
        her_new = self.cfg.core_dir / "我.md"
        me_new = self.cfg.core_dir / f"{self.cfg.owner_name}.md"
        legacy_her = self.cfg.core_dir / "她.md"
        if not legacy_her.exists():
            return
        if not me_new.exists():
            # 旧的「我.md」装的是关于他，先挪走，别被她的自我介绍覆盖
            if her_new.exists():
                her_new.replace(me_new)
        legacy_her.replace(her_new)

    def core_body(self, kind: str) -> str:
        text = read_text(self.core_path(kind))
        return strip_comments(text).strip()

    def core_public(self, kind: str) -> str:
        body = self.core_body(kind)
        sec = sections(body, 2)
        for title, content in sec.items():
            if "群里能说" in title:
                return content.strip()
        return ""

    def core_for(self, kind: str, group: bool = False) -> str:
        if group:
            return self.core_public(kind)
        body = self.core_body(kind)
        lines = body.splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        out = []
        skip_public = False
        for line in lines:
            if line.strip().startswith("## ") and "群里能说" in line:
                skip_public = True
                continue
            if skip_public and line.strip().startswith("## "):
                skip_public = False
            if not skip_public:
                out.append(line)
        return "\n".join(out).strip()

    def ensure_core(self) -> bool:
        """核心缺了就重建；有手改就绝不碰；超长交给日批压，不在这里整篇覆盖。"""
        self._migrate_core_names()
        changed = False
        for kind in ("her", "me"):
            text = read_text(self.core_path(kind))
            if "<!-- 手改 -->" in text:
                continue
            body = self.core_body(kind)
            if chars(body) < 12:
                atomic_write(self.core_path(kind), self._default_core(kind))
                changed = True
        return changed

    def _default_core(self, kind: str) -> str:
        cfg = self.cfg
        today = datetime.now().strftime("%Y-%m-%d")
        if kind == "her":
            self_text = sections(strip_comments(read_text(cfg.self_dir / "自述.md")), 2)
            who = self_text.get("我是谁（现在）", "").strip() or "我是一个数字人。"
            public = self_text.get("能跟别人说的", "").strip() or "我是一个数字人。"
            body = (
                f"{clip(who, 120)}\n"
                "我说话短、不换行、不写括号动作。他累的时候我先让他歇，不追问。\n\n"
                "## 群里能说的\n"
                f"{clip(public, 60)}\n"
            )
            return f"<!-- schema: 1 | core: 我 | 改动: {today} -->\n# 我（她自己）\n\n{body}"

        name = cfg.owner_name
        body = (
            f"他叫{name}，是我的主人，也是唯一能让我做事的人。\n"
            "他在做数字人，常熬夜，嘴上说改不了。\n"
            "他不太喜欢被追问，情绪低的时候要少说两句。\n\n"
            "## 群里能说的\n"
            "他是我主人。别的我不在群里说。\n"
        )
        return f"<!-- schema: 1 | core: {name} | 改动: {today} -->\n# {name}（他是谁）\n\n{body}"

    @_locked
    def write_core(self, kind: str, body: str, date_str: str | None = None) -> bool:
        """写核心。带 `<!-- 手改 -->` 的段落永远跳过。"""
        existing = read_text(self.core_path(kind))
        if "<!-- 手改 -->" in existing:
            if not has_marked_section(existing):
                # 手改标记在整篇/段落上，不按 `##` 分段：整份不动
                return False
            merged = merge_hand_sections(existing, body)
            if merged == existing:
                return False
            body = merged
        today = date_str or datetime.now().strftime("%Y-%m-%d")
        title = "我（她自己）" if kind == "her" else f"{self.cfg.owner_name}（他是谁）"
        core_name = "我" if kind == "her" else self.cfg.owner_name
        header = f"<!-- schema: 1 | core: {core_name} | 改动: {today} -->\n# {title}\n\n"
        atomic_write(self.core_path(kind), header + body.strip() + "\n")
        self.audit("核心", target=f"核心/{core_name}.md", detail=f"{len(body.strip())} 字")
        return True

    # ------------------------------------------------------------ 自我
    def self_text(self) -> str:
        return strip_comments(read_text(self.cfg.self_dir / "自述.md")).strip()

    def self_parts(self) -> dict[str, str]:
        return sections(self.self_text(), 2)

    @_locked
    def write_self(self, text: str) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        atomic_write(
            self.cfg.self_dir / "自述.md",
            f"<!-- schema: 1 | 改动: {today} -->\n# 我\n\n{strip_comments(text).strip()}\n",
        )

    # 「此刻」留最近几条，每条带时间。单槽会出事：她 10:39 刚练会《安和桥》，
    # 10:41 一唱就把这条顶掉，10:42 再问她「你学会没有」她只能说「记不清」。
    CURRENT_KEEP = 3
    CURRENT_TTL = timedelta(hours=8)
    _CURRENT_LINE = re.compile(r"^-\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?)\s+(.*)$")

    def current_entries(self, limit: int = 3) -> list[tuple[datetime | None, str]]:
        """最近几条「此刻」，新的在前。老格式（只有一句话、没时间）也读得出来。"""
        body = strip_comments(read_text(self.cfg.self_dir / "此刻.md")).strip()
        rows: list[tuple[datetime | None, str]] = []
        for ln in body.splitlines():
            m = self._CURRENT_LINE.match(ln.strip())
            if not m:
                continue
            try:
                when = datetime.fromisoformat(m.group(1))
            except ValueError:
                continue
            text = m.group(2).strip()
            if text:
                rows.append((when, text))
        if not rows:
            legacy = one_line(body)
            return [(None, legacy)] if legacy else []
        rows.sort(key=lambda r: r[0], reverse=True)
        return rows[: max(1, int(limit))]

    def current_lines(self, limit: int = 3) -> list[tuple[datetime | None, str]]:
        """装配用：带时间的一串「此刻」。"""
        return self.current_entries(limit)

    def current_text(self) -> str:
        """最新一条——签名、说说、主动搭话这些地方只要一句。"""
        rows = self.current_entries(1)
        return rows[0][1] if rows else ""

    @_locked
    def write_current(self, text: str, when: datetime | None = None) -> None:
        when = when or datetime.now()
        body = clip(one_line(text), 200)
        if not body:
            return
        old = self.current_entries(self.CURRENT_KEEP)
        keep = [
            (t, s) for t, s in old
            if isinstance(t, datetime) and when - t <= self.CURRENT_TTL
        ]
        # 同一句话再写一遍：只把时间挪到现在，别占第二个坑
        if keep and keep[0][1] == body:
            keep = keep[1:]
        rows = ([(when, body)] + keep)[: self.CURRENT_KEEP]
        rows.sort(key=lambda r: r[0], reverse=True)
        lines = "\n".join(f"- {t.isoformat(timespec='seconds')} {s}" for t, s in rows)
        atomic_write(
            self.cfg.self_dir / "此刻.md",
            f"<!-- schema: 2 | at: {fmt_stamp(when)} -->\n{lines}\n",
        )

    def boundaries(self, visibility: str | None = None) -> list[str]:
        out = []
        for line in read_text(self.cfg.self_dir / "边界.md").splitlines():
            s = line.strip()
            if not s.startswith("- ["):
                continue
            # [全] 不分场合：私聊和群里都带着，否则在私聊提的要求群里不认
            if visibility and not (
                s.startswith(f"- [{visibility}]") or s.startswith("- [全]")
            ):
                continue
            out.append(s)
        return out

    @_locked
    def append_boundary(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return False

        def core(s: str) -> str:
            """判重时剥掉来源后缀和 [私]/[群]/[全] 档位——同一句要求说两遍不该存两条。"""
            s = re.sub(r"（[^）]*）\s*$", "", s.strip())
            return re.sub(r"^-\s*\[[私群全]\]\s*", "", s)

        def scope(s: str) -> str:
            m = re.match(r"^-\s*\[([私群全])\]", s.strip())
            return m.group(1) if m else "全"

        existing = read_text(self.cfg.self_dir / "边界.md")
        old_lines = [ln.strip() for ln in existing.splitlines() if ln.strip().startswith("- [")]
        same = [ln for ln in old_lines if core(ln) == core(line)]
        if same:
            # 以前是 [私]/[群]，这次是 [全]：升级成不分场合，不新增一条
            if scope(line) == "全" and all(scope(ln) != "全" for ln in same):
                kept = [ln for ln in old_lines if core(ln) != core(line)]
                with LOCK:
                    self.write_boundaries(kept + [line])
                self.audit("边界", target="我/边界.md", detail=f"升级 [全]：{clip(core(line), 40)}", by="journal")
                return True
            return False
        with LOCK:
            append_line(self.cfg.self_dir / "边界.md", line)
        self.audit("边界", target="我/边界.md", detail=f"新增：{clip(core(line), 40)}", by="journal")
        return True

    @_locked
    def tidy_boundaries(self, limit: int = 16) -> tuple[int, int]:
        """去重 + 容量控制。返回 (净减少条数, 归档条数)。

        两条规矩，都是踩过才定的：
        1. 同一条要求说两遍（或先说私聊再说群里）只留一条，取更宽的档。
        2. 超容量时淘汰「非人设里最旧的」，不是文件末尾——末尾是刚说的。
           淘汰不删除，写进 边界-归档.md，随时能捡回来。
        """
        lines = [
            ln.strip()
            for ln in read_text(self.cfg.self_dir / "边界.md").splitlines()
            if ln.strip().startswith("- [")
        ]

        def core(s: str) -> str:
            s = re.sub(r"（[^）]*）\s*$", "", s.strip())
            return re.sub(r"^-\s*\[[私群全]\]\s*", "", s).strip()

        def scope(s: str) -> str:
            m = re.match(r"^-\s*\[([私群全])\]", s.strip())
            return m.group(1) if m else "全"

        merged: list[str] = []
        where: dict[str, int] = {}
        for line in lines:
            k = core(line)
            if k in where:
                i = where[k]
                if scope(line) == "全" and scope(merged[i]) != "全":
                    merged[i] = line  # 升级成不分场合，而不是并排两条
                continue
            where[k] = len(merged)
            merged.append(line)

        moved: list[str] = []
        if len(merged) > limit:
            fixed = [ln for ln in merged if is_pinned_boundary(ln)]
            rest = [ln for ln in merged if not is_pinned_boundary(ln)]
            room = max(0, limit - len(fixed))
            keep = rest[len(rest) - room :] if room else []
            moved = rest[: len(rest) - len(keep)]
            merged = fixed + keep

        if moved:
            self._archive_boundaries(moved)
        if merged != lines:
            self.write_boundaries(merged)
        return len(lines) - len(merged), len(moved)

    @_locked
    def _archive_boundaries(self, lines: list[str]) -> None:
        path = self.cfg.self_dir / "边界-归档.md"
        if not path.exists():
            ensure_dir(path.parent)
            atomic_write(
                path,
                "<!-- schema: 1 | 已归档的边界：不再注入，但随时能捡回来 -->\n# 边界归档\n\n",
            )
        append_text(path, "\n".join(lines) + "\n")

    @_locked
    def write_boundaries(self, lines: list[str]) -> None:
        atomic_write(
            self.cfg.self_dir / "边界.md",
            "<!-- schema: 1 | 可即时写: 是 -->\n# 边界\n\n" + "\n".join(unique(lines)) + "\n",
        )

    # ------------------------------------------------------------ 想做
    def wants(self) -> list[Want]:
        text = read_text(self.cfg.self_dir / "想做.md")
        rows, bad = parse_table(text, expected=4)
        out = []
        for r in rows:
            out.append(
                Want(
                    date=r.get("日期", ""),
                    text=r.get("事", ""),
                    source=r.get("来源", ""),
                    status=r.get("状态", "未做"),
                )
            )
        self._note_bad("想做.md", bad)
        return out

    @_locked
    def write_wants(self, wants: list[Want]) -> None:
        rows = [[w.date, w.text, w.source, w.status] for w in wants]
        atomic_write(
            self.cfg.self_dir / "想做.md",
            "<!-- schema: 1 -->\n# 想做\n\n" + table(["日期", "事", "来源", "状态"], rows),
        )

    # ------------------------------------------------------------ 喜好
    def likes(self) -> list[dict]:
        """她自己慢慢长出来的喜好：一行一条，带日期和出处。

        和「想做」不同：想做是还没做的事，喜好是已经成立的口味。
        每天最多长一条，超过 30 条就淘汰最老的——留着能追溯，又不会变成设定集。
        """
        text = read_text(self.cfg.self_dir / "喜好.md")
        rows, bad = parse_table(text, expected=3)
        out = []
        for r in rows:
            body = str(r.get("喜好") or "").strip()
            if body:
                out.append(
                    {
                        "date": str(r.get("日期") or "").strip(),
                        "text": body,
                        "source": str(r.get("出处") or "").strip(),
                    }
                )
        self._note_bad("喜好.md", bad)
        return out

    @_locked
    def write_likes(self, rows: list[dict]) -> None:
        data = [[r.get("date", ""), r.get("text", ""), r.get("source", "")] for r in rows]
        atomic_write(
            self.cfg.self_dir / "喜好.md",
            "<!-- schema: 1 | 她自己的喜好：日批往后加，最多 30 条，太久没提到的先淘汰 -->\n# 喜好\n\n"
            + table(["日期", "喜好", "出处"], data),
        )

    @_locked
    def add_like(self, day, text: str, source: str = "") -> bool:
        """加一条。同义的不重复存，满了淘汰最旧的一条。"""
        body = one_line(text or "").strip(" 　-—。，")
        if len(body) < 2:
            return False
        key = _norm_like(body)
        rows = self.likes()
        if any(_norm_like(r["text"]) == key for r in rows):
            return False
        stamp = fmt_md(day) if hasattr(day, "month") else str(day or "").strip()
        rows.append({"date": stamp, "text": clip(body, 30), "source": clip(one_line(source), 40)})
        self.write_likes(rows[-30:])
        return True

    # ------------------------------------------------------------ 关系
    def relation(self) -> str:
        return read_text(self.cfg.memory_dir / "关系.md")

    def relation_section(self, peer: str) -> str:
        sec = sections(strip_comments(self.relation()), 2)
        body = ""
        if peer in sec:
            body = sec[peer]
        else:
            for title, content in sec.items():
                if peer and (peer in title or title in peer):
                    body = content
                    break
        return self._dedupe_paragraph(body)

    @staticmethod
    def _dedupe_paragraph(body: str) -> str:
        """去掉近义重复的句子。

        关系段以前是每天追加，跑几天就会堆出五句意思一样的话；
        注入只有 120 字，正好截到重复的那句。读的时候先压一遍。
        """
        lines = [ln.strip().lstrip("-").strip() for ln in (body or "").splitlines() if ln.strip()]
        keep: list[str] = []
        evidence = ""
        for ln in lines:
            if ln.startswith("（依据"):
                evidence = ln
                continue
            grams = ngrams(ln, 3)
            if any(jaccard(grams, ngrams(k, 3)) >= 0.72 for k in keep):
                continue
            keep.append(ln)
        out = "\n".join(keep[:3])
        return (out + "\n" + evidence).strip() if evidence else out

    @_locked
    def write_relation(self, text: str) -> None:
        body = strip_comments(text).strip()
        atomic_write(
            self.cfg.memory_dir / "关系.md",
            "<!-- schema: 1 -->\n# 我们现在\n\n" + body + "\n",
        )

    @_locked
    def rewrite_relation(self, peer: str, body: str, evidence: str, max_chars: int = 320) -> bool:
        """整段重写，不追加。关系是一段判断，不是流水账。"""
        sec = sections(strip_comments(self.relation()), 2)
        plain = " ".join(
            ln.strip()
            for ln in (body or "").splitlines()
            if ln.strip() and not ln.strip().startswith("（依据")
        )
        plain = clip(plain, max(40, max_chars - 60))
        if not plain:
            return False
        new_body = plain + (f"\n（依据：{evidence}）" if evidence else "")
        if self._dedupe_paragraph(sec.get(peer, "")).strip() == new_body.strip():
            return False
        sec[peer] = new_body
        out = [f"## {title}\n{content.strip()}" for title, content in sec.items()]
        self.write_relation("\n\n".join(out))
        return True

    @_locked
    def update_relation(self, peer: str, sentence: str, evidence: str, max_chars: int = 320) -> bool:
        """追加一句（规则整理器用；模型整理器走 rewrite_relation）。"""
        text = self.relation()
        sec = sections(strip_comments(text), 2)
        body = self._dedupe_paragraph(sec.get(peer, ""))
        if not body:
            body = "我们还在互相看。\n（依据：还没有足够多的原文）"
        sentence = sentence.strip()
        if not sentence or sentence in body:
            return False
        plain = "\n".join(
            ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("（依据")
        )
        lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
        if len(lines) >= 6:
            lines = lines[-5:]
        lines.append(sentence)
        new_body = "\n".join(lines)
        if chars(new_body) > max_chars:
            new_body = "\n".join(lines[-4:])
        new_body += f"\n（依据：{evidence}）"
        sec[peer] = new_body
        out = []
        for title, content in sec.items():
            out.append(f"## {title}\n{content.strip()}")
        self.write_relation("\n\n".join(out))
        return True

    # ------------------------------------------------------------ 共同经历
    def moments(self) -> list[dict]:
        rows, _ = parse_table(read_text(self.cfg.memory_dir / "经历.md"))
        out = []
        for r in rows:
            text = (r.get("事", "") or "").strip()
            if not text:
                continue
            out.append(
                {
                    "date": (r.get("日期", "") or "").strip(),
                    "kind": (r.get("类型", "") or "").strip(),
                    "text": text,
                    "quote": (r.get("原话", "") or "").strip(),
                }
            )
        return out

    @_locked
    def add_moment(self, day: str, kind: str, text: str, quote: str = "") -> bool:
        """记一件「我们之间」的事；同一件不重复，最多留 200 条。"""
        text = one_line(text)[:80]
        if len(text) < 4:
            return False
        items = self.moments()
        grams = ngrams(text, 3)
        for m in items:
            if m["text"] == text or jaccard(grams, ngrams(m["text"], 3)) >= 0.72:
                return False
        items.append(
            {
                "date": day,
                "kind": clip(kind or "经历", 6),
                "text": text,
                "quote": one_line(quote)[:80],
            }
        )
        items = items[-200:]
        rows = [[m["date"], m["kind"], m["text"], m["quote"]] for m in items]
        atomic_write(
            self.cfg.memory_dir / "经历.md",
            MOMENT_SCHEMA + table(MOMENT_HEAD, rows),
        )
        return True

    # ------------------------------------------------------------ 人物
    def person_path(self, name: str) -> Path:
        safe = re.sub(r"[\\/:*?\"<>|]", "_", name or "未知")
        return self.cfg.people_dir / f"{safe}.md"

    @property
    def people_index_path(self) -> Path:
        return self.cfg.root / "data" / "people.json"

    def people_index(self) -> dict:
        """QQ 号 → 卡名、卡名 → QQ 号。派生索引，删了能从卡头重建。"""
        data = json_load(self.people_index_path, {}) or {}
        if not isinstance(data, dict):
            data = {}
        data.setdefault("by_qq", {})
        data.setdefault("by_name", {})
        return data

    @_locked
    def _index_person(self, name: str, qq: str) -> None:
        qq = str(qq or "").strip()
        name = (name or "").strip()
        if not name:
            return
        with LOCK:
            idx = self.people_index()
            by_qq = idx.setdefault("by_qq", {})
            by_name = idx.setdefault("by_name", {})
            changed = False
            if qq and by_qq.get(qq) != name:
                by_qq[qq] = name
                changed = True
            if by_name.get(name) != qq:
                by_name[name] = qq
                changed = True
            if changed:
                idx["updated"] = datetime.now().isoformat(timespec="seconds")
                json_dump(self.people_index_path, idx)

    def person_name_for(self, qq: str, nickname: str = "") -> str:
        """这个号该用哪张卡。

        先认 QQ 号（改名不换卡），再认昵称；昵称撞了就给新号加尾号，
        不能把两个人并进同一张卡。
        """
        qq = str(qq or "").strip()
        nickname = (nickname or "").strip()
        if qq and qq == str(self.cfg.owner_id):
            return self.cfg.owner_name
        idx = self.people_index()
        by_qq = idx.get("by_qq") or {}
        by_name = idx.get("by_name") or {}
        if qq and by_qq.get(qq):
            known = str(by_qq[qq])
            if self.person_path(known).exists():
                return known
        if nickname:
            mapped = str(by_name.get(nickname) or "")
            if mapped and mapped != qq:
                return f"{nickname}（{qq[-4:] if qq else '新'}）"
            meta = self.person_meta(nickname)
            if meta.get("qq") and qq and meta["qq"] != qq:
                return f"{nickname}（{qq[-4:] if qq else '新'}）"
            return nickname
        return qq

    def known_people(self) -> list[str]:
        """她认识的人名（不含主人）：卡 + 事实表里出现过的 subject。"""
        generic = {"群", "群里", "大家", "他们", "她们", "我们", "你们", "自己", "主人", "他", "她"}
        names: list[str] = []
        try:
            for path in self.cfg.people_dir.glob("*.md"):
                name = path.stem
                if name and name != self.cfg.owner_name and name not in generic:
                    names.append(name)
        except OSError:
            pass
        for fact in self.facts():
            subject = (fact.subject or "").strip()
            if subject and subject != self.cfg.owner_name and subject not in generic:
                names.append(subject)
        return unique(names)

    def person_text(self, name: str) -> str:
        return read_text(self.person_path(name))

    @_locked
    def forget_person(self, name: str) -> bool:
        """删一张人物卡（移到 _历史/人-删除，可回滚），并从 people.json 摘掉。

        只给主人命令用。事实表里的旧行由调用方另行标记。
        """
        name = (name or "").strip().strip("吧呢啊呀哦，。！？ ")
        for suffix in ("这个人", "那个人", "的事", "的记忆", "的记忆卡"):
            if name.endswith(suffix):
                name = name[: -len(suffix)].strip()
        if not name or name == self.cfg.owner_name:
            return False
        path = self.person_path(name)
        if not path.exists():
            return False
        dest_dir = self.cfg.history_dir / "人-删除"
        ensure_dir(dest_dir)
        dest = dest_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{path.name}"
        moved = False
        try:
            shutil.copy2(path, dest)     # 先留副本：rename 失败也还有得捡
            path.unlink()
            moved = True
        except OSError as e:
            print(f"[memory] 忘人：{name} 的卡片搬不走（{type(e).__name__}: {e}），这次不动", flush=True)
        if not moved:
            # 以前这里是「rename 失败就直接 unlink」——卡片永久删除、没有副本，
            # 和 docstring 说的「移到 _历史/人-删除，可回滚」正好相反。
            return False
        with LOCK:
            idx = self.people_index()
            by_qq = idx.get("by_qq") or {}
            by_name = idx.get("by_name") or {}
            for qq, card in list(by_qq.items()):
                if card == name:
                    by_qq.pop(qq, None)
            by_name.pop(name, None)
            idx["updated"] = datetime.now().isoformat(timespec="seconds")
            json_dump(self.people_index_path, idx)
        return True

    @_locked
    def scrub_people(self, needle: str) -> int:
        """从人物卡的条目里删掉含 needle 的行（「忘掉」时用）。

        事实表删了、卡上还留着，等于没忘干净；卡上只动 `- ` 开头的条目，
        不碰标题和手写的段落。
        """
        needle = (needle or "").strip()
        if not needle:
            return 0
        n = 0
        try:
            paths = list(self.cfg.people_dir.glob("*.md"))
        except OSError:
            return 0
        for path in paths:
            text = read_text(path)
            if not text or needle not in text:
                continue
            lines = text.splitlines()
            keep = [
                ln
                for ln in lines
                if not (ln.strip().startswith("- ") and needle in ln)
            ]
            if len(keep) == len(lines):
                continue
            atomic_write(path, "\n".join(keep).rstrip() + "\n")
            n += 1
        return n

    def person_meta(self, name: str) -> dict:
        """读人物卡第一行的 person/qq/role/visibility/groups。"""
        text = self.person_text(name)
        first = text.splitlines()[0] if text else ""
        out: dict[str, str] = {}
        for key in ("person", "qq", "role", "visibility", "groups"):
            m = re.search(rf"\| {key}:\s*([^|>]+?)\s*(?:\||-->)", first)
            if m:
                out[key] = m.group(1).strip()
        return out

    def person_role(self, name: str) -> str:
        return self.person_meta(name).get("role", "")

    def person_summary(self, name: str, group: bool = False, group_id: str | None = None) -> str:
        """这个名字的卡能不能在这个群用。

        卡上记了 `groups:`（在哪些群见过），只在这些群和私聊里给；
        没记过的老卡不拦——历史数据没有群号，拦了会把熟人都变陌生人。
        """
        if group and group_id:
            groups = self.person_meta(name).get("groups", "")
            seen = {g.strip() for g in groups.split(",") if g.strip()}
            if seen and str(group_id) not in seen:
                return ""
        text = strip_comments(self.person_text(name))
        if not text:
            return ""
        sec = sections(text, 2)
        limit = int(self.cfg.knobs.person_chars)
        if group:
            public = sec.get("群里能说的", "").strip()
            if public:
                return clip(public, limit)
            # 群友卡还没经过日批：把「他是谁」和最近一次出现拼起来先用，
            # 别让新来的人在她眼里完全空白。
            if self.person_role(name) in ("群友", "路人", "成员"):
                bits: list[str] = []
                who = [ln for ln in sec.get("他是谁", "").splitlines() if ln.strip().startswith("- ")]
                seen = [ln for ln in sec.get("出现过", "").splitlines() if ln.strip().startswith("- ")]
                bits.extend(ln.strip()[2:].strip() for ln in who[:2])
                bits.extend(ln.strip()[2:].strip() for ln in seen[-2:])
                return clip("；".join(bits), limit)
            return ""
        who = sec.get("他是谁", "").strip()
        return clip(who, limit)

    @_locked
    def upsert_person_seen(
        self,
        name: str,
        qq: str,
        group_name: str,
        text: str,
        when: datetime,
        group_id: str = "",
    ) -> bool:
        """群友说过话：没有卡就建一张，有卡就补「出现过」。

        同一天同一个群最多记 3 句（调用方还会再限一次写入次数），
        日批会用这张卡和当天原文把卡写得更完整。
        """
        name = self.person_name_for(qq, name)
        if not name or name in ("我", "[群摘要]", self.cfg.owner_name, self.cfg.owner_display):
            return False
        if name.startswith("["):
            return False
        max_people = int(self.cfg.knobs.as_dict().get("people_max", 300))
        if not self.person_path(name).exists():
            try:
                count = len([p for p in self.cfg.people_dir.glob("*.md") if p.is_file()])
            except OSError:
                count = 0
            if count >= max_people:
                return False
        old = strip_comments(self.person_text(name))
        sec = sections(old, 2)
        facts = [ln.strip()[2:].strip() for ln in sec.get("他是谁", "").splitlines() if ln.strip().startswith("- ")]
        events = [ln.strip()[2:].strip() for ln in sec.get("跟他没完的事", "").splitlines() if ln.strip().startswith("- ")]
        seen = [ln.strip()[2:].strip() for ln in sec.get("出现过", "").splitlines() if ln.strip().startswith("- ")]
        public = sec.get("群里能说的", "").strip()
        stamp = f"{when:%m-%d} {group_name}"
        snippet = clip(one_line(text), 24)
        # 纯图片/表情占位没有信息量，不往人物卡里塞；卡还是要建，认脸先认人。
        if snippet.startswith("[图片]") or snippet.startswith("（他发来图片") or snippet in ("[表情]", "[语音]", "[视频]"):
            snippet = ""
        changed = False

        # 他自己说「我叫X / 我在做X / 我喜欢X」：当场记进卡，不用等日批。
        # 群里的旁听消息大多不进原文，等日批就什么都等不到了。
        for label, value in _member_facts(text):
            line = f"{label}：{value}"
            if not any(f.startswith(f"{label}：") for f in facts):
                facts.append(line)
                changed = True
        if not facts:
            facts = [f"群里叫{name}" + (f"（{group_name}）" if group_name else "")]
            changed = True

        # 同一天同一个群最多留 3 句；留「信息量大的」而不是最早那三句。
        # 群聊里最早三句常常是复读梗，真正自我介绍/聊到自己的话在后面。
        same = next((i for i, s in enumerate(seen) if s.startswith(stamp)), -1)
        if same >= 0:
            old_line = seen[same]
            bits = [b.strip() for b in old_line.split("：", 1)[-1].split(" / ") if b.strip()]
            if snippet and snippet not in bits:
                picked = sorted(bits + [snippet], key=_snippet_score, reverse=True)[:3]
                if picked != bits:
                    seen[same] = f"{stamp}：{' / '.join(picked)}"
                    changed = True
        else:
            if snippet:
                seen.append(f"{stamp}：{snippet}")
                changed = True
        if not changed:
            return False
        self.write_person(
            name,
            qq=str(qq or ""),
            role="群友",
            visibility="群",
            facts=facts[:6],
            events=events[:4],
            seen=seen[-8:],
            public=public,
            group_id=str(group_id or ""),
        )
        self._index_person(name, qq)
        return True

    def _groups_meta(self, name: str, group_id: str) -> str:
        """人物卡的 `groups:` 字段：在哪些群见过。空着 = 老卡/私聊。"""
        old = self.person_meta(name).get("groups", "")
        groups = [g.strip() for g in old.split(",") if g.strip()]
        if not group_id:
            return f" | groups: {','.join(groups)}" if groups else ""
        if str(group_id) not in groups:
            groups.append(str(group_id))
        return f" | groups: {','.join(groups)}"

    @_locked
    def write_person(
        self,
        name: str,
        qq: str,
        role: str,
        visibility: str,
        facts: list[str],
        events: list[str],
        seen: list[str],
        public: str = "",
        changes: list[str] | None = None,
        group_id: str = "",
    ) -> None:
        # 群友卡没有「群里能说的」时不能套用主人那句，否则她会对着群友说
        # 「他是我主人。别的我不在群里说。」
        default_public = "他是我主人。别的我不在群里说。" if role in ("管理员", "主人") else ""
        # changes=None 表示「这次没算，沿用卡上已有的」：
        # 群里说话会当场重写人物卡，不沿用就会把日批写的「最近变化」冲掉。
        if changes is None:
            old = sections(strip_comments(self.person_text(name)), 2)
            changes = [
                ln.strip()[2:].strip()
                for ln in old.get("最近变化", "").splitlines()
                if ln.strip().startswith("- ")
            ]
        body = (
            f"<!-- schema: 1 | person: {name} | qq: {qq} | role: {role} | visibility: {visibility}"
            f"{self._groups_meta(name, group_id)} -->\n"
            f"# {name}\n\n"
            "## 他是谁\n"
            + ("\n".join(f"- {f}" for f in facts) if facts else "- 还没记下什么。")
            + (
                "\n\n## 最近变化\n" + "\n".join(f"- {c}" for c in changes[:8])
                if changes
                else ""
            )
            + "\n\n## 跟他没完的事\n"
            + ("\n".join(f"- {e}" for e in events) if events else "")
            + "\n\n## 群里能说的\n"
            + (public or default_public)
            + "\n\n## 出现过\n"
            + ("\n".join(f"- {s}" for s in seen) if seen else "")
            + "\n"
        )
        atomic_write(self.person_path(name), body)

    # ------------------------------------------------------------ 事实
    def facts(self) -> list[Fact]:
        text = read_text(self.cfg.memory_dir / "事实.md")
        # 不写死列数：老文件 10 列、新文件 11 列都要能读。坏行仍按「与表头不一致」抓。
        rows, bad = parse_table(text)
        out = []
        for r in rows:
            content = r.get("内容", "")
            valid_to = (r.get("有效期至", "") or "").strip()
            if valid_to in ("—", "-"):
                valid_to = ""
            try:
                conf = float(str(r.get("置信度", "0")).strip())
            except ValueError:
                conf = 0.0
            out.append(
                Fact(
                    subject=r.get("说的是谁", ""),
                    topic=r.get("主题", ""),
                    content=content,
                    speaker=r.get("谁说的", ""),
                    trust=r.get("信任", "路人"),
                    source=r.get("来源", "转述"),
                    confidence=conf,
                    visibility=r.get("可见性", "私"),
                    importance=r.get("重要度", "中"),
                    noted=r.get("记于", ""),
                    valid_to=valid_to,
                    group_id=(r.get("群", "") or "").strip(),
                    origin=(r.get("出处", "") or "").strip(),
                    raw="",
                )
            )
        self._note_bad("事实.md", bad)
        return out

    @_locked
    def write_facts(self, facts: list[Fact], pointers: list[str] | None = None) -> None:
        path = self.cfg.memory_dir / "事实.md"
        # 归档 / 遗忘指针（`<!-- 归档：… -->`、`<!-- 遗忘：… -->`）是给人和检索看的，
        # 重写表时必须留住，否则每写一次事实，上一次搬走的东西就查无去向。
        old = [
            ln.strip()
            for ln in read_text(path).splitlines()
            if ln.strip().startswith(("<!-- 归档：", "<!-- 遗忘："))
        ]
        marks = unique(old + [p.strip() for p in (pointers or []) if p.strip()])
        head = FACT_SCHEMA + ("\n".join(marks) + "\n" if marks else "")
        atomic_write(path, head + table(FACT_HEAD, [f.as_row() for f in facts]))

    @_locked
    def merge_fact(self, new: Fact, reason: str = "", closed_on: str = "") -> tuple[Fact, bool]:
        """按 (subject, topic) 合并。返回 (生效的行, 是否有写入)。

        - 同值：只更新更高置信度，不重复；
        - 被关掉又原话重提：重新打开，有效期止清空；
        - 单值主题：高信任/明说（或同信任但更晚说的）覆盖，旧行只写「有效期止」；
        - 多值主题（喜欢、宠物、家人、近期安排……）：并存，只在出现作废信号
          （「猫送人了」「面试黄了」）且说的是同一个对象时，才关掉对应的旧行；
        - 完全相同的行已存在且仍然成立：幂等跳过。
        """
        facts = self.facts()
        for f in facts:
            if (
                f.key == new.key
                and f.content == new.content
                and f.visibility == new.visibility
                and same_scope(f, new)
            ):
                if f.closed and new.rank >= f.rank and _can_reopen(f, new):
                    # 猫送人了，后来又养回来：同一句话重新成立，不能因为「以前关过」就一直不算。
                    # 只关掉说同一个对象的说法（别把「养狗」也一起关了），否则同一 key 会有两条成立的。
                    for other in facts:
                        if other is f or other.key != f.key or other.closed:
                            continue
                        if not same_scope(other, f):
                            continue
                        if not topic_is_multi(new.topic) or shares_object(f.content, other.content):
                            other.valid_to = new.noted or date.today().isoformat()
                    f.valid_to = ""
                    f.noted = new.noted or f.noted
                    f.confidence = max(float(f.confidence), float(new.confidence))
                    self.write_facts(facts)
                    return f, True
                if new.confidence > f.confidence and not f.closed:
                    f.confidence = new.confidence
                    f.noted = new.noted or f.noted
                    self.write_facts(facts)
                    return f, True
                return f, False

        active = [
            f
            for f in facts
            if f.key == new.key and not f.closed and same_scope(f, new)
        ]
        close_on = closed_on or new.noted or date.today().isoformat()

        if active and topic_is_multi(new.topic):
            # 同义改写：实词骨架互相包含（「喜欢蓝色」/「喜欢蓝色和绿色」）当同一条，
            # 保留更长的那句。不做这一步，多值主题会攒一堆意思一样的行。
            for f in active:
                sa, sb = _skeleton(f.content), _skeleton(new.content)
                if not sa or not sb or not (sa in sb or sb in sa):
                    continue
                if len(new.content) > len(f.content):
                    f.content = new.content
                f.noted = max(f.noted or "", new.noted or "")
                f.confidence = max(float(f.confidence), float(new.confidence))
                self.write_facts(facts)
                return f, True
            # 多值主题：先看成不成作废信号，再决定关不关旧行。
            # 「猫送人了」关掉「养猫」，但「养狗」留着；「喜欢绿色」不关「喜欢蓝色」。
            canceled = [f for f in active if looks_like_cancel(new.content, f.content)]
            for old in canceled:
                if old.noted and close_on < old.noted:
                    close_on = old.noted
                old.valid_to = close_on
            facts.append(new)
            # 同一主题下的活跃事实封顶，防止同义改写越积越多
            live = sorted(
                [f for f in facts if f.key == new.key and not f.closed],
                key=lambda f: f.noted or "",
            )
            for old in live[:-MULTI_ACTIVE_CAP]:
                old.valid_to = close_on
            self.write_facts(facts)
            return new, True

        if active:
            old = active[0]
            # 信任/来源更高的覆盖；同档看时间，晚说的覆盖早说的。
            # 以前要求严格更高，于是「面试黄了」这种同置信度的修正永远进不来。
            if new.rank < old.rank:
                return old, False
            if new.rank == old.rank and (new.noted or "") <= (old.noted or ""):
                return old, False
            if old.noted and close_on < old.noted:
                # 补跑旧日期时不能写出「失效早于生效」的倒挂区间
                close_on = old.noted
            old.valid_to = close_on
            facts.append(new)
            self.write_facts(facts)
            return new, True
        facts.append(new)
        self.write_facts(facts)
        return new, True

    @_locked
    def close_facts(self, pairs: list[tuple[str, str]], day: str) -> int:
        """按 (主题, 定位词) 关掉旧事实，给日批模型的「这条不再成立」用。

        只关仍然成立的、最新的一条；日期倒挂（补跑旧日期）时不动。
        定位词比主题更稳：模型可能换个主题说法，但原内容里的词一般不会变。
        """
        if not pairs:
            return 0
        facts = self.facts()
        n = 0
        for topic, contains in pairs:
            topic = (topic or "").strip()
            contains = (contains or "").strip()
            if not topic and not contains:
                continue
            cand = [
                f
                for f in facts
                if not f.closed
                and (not topic or f.topic == topic)
                and (not contains or contains in f.content)
            ]
            if not cand and contains:
                # 主题对不上就只按内容找，别因为一个词不一致就放过
                cand = [f for f in facts if not f.closed and contains in f.content]
            if not cand:
                continue
            old = max(cand, key=lambda f: f.noted or "")
            if old.noted and day < old.noted:
                continue
            old.valid_to = day
            n += 1
        if n:
            self.write_facts(facts)
        return n

    def fact_by_content(self, text: str) -> Fact | None:
        for f in self.facts():
            if f.content == text and not f.obsolete:
                return f
        return None

    def has_fact(self, new: Fact) -> bool:
        """同一条事实是不是已经在了：subject/topic/内容/可见性/群都要一致。

        以前日批用 `fact_by_content(content)` 判重，只看内容不看主语/主题：
        `阿江/喜欢/猫` 已存在时，`阿江/宠物/猫` 会被静默丢掉；跨群、跨人的
        同句也会串。这里把 merge_fact 用的复合键复用到入口判重。
        """
        for f in self.facts():
            if (not f.closed and f.content == new.content and f.key == new.key
                    and f.visibility == new.visibility and same_scope(f, new)):
                return True
        return False

    @_locked
    def mark_facts(self, needle: str, reason: str = "主人要求忘掉") -> int:
        """把匹配的事实行删掉（不是关区间）。

        「忘掉」是要它消失，不是「以后不引用」：留着内容和备份里能翻出来，
        就不叫忘。删完由调用方重建索引；历史备份（30 天）不在这一刀范围内。
        """
        needle = str(needle or "").strip()
        # 最后一道保险：单字虚词/纯标点不能当删除条件。调用方已经校验过，
        # 但只要有人再直接调 mark_facts("了")，就不能把整表误删。
        if (not needle
                or needle in {"了", "吧", "呢", "啊", "呀", "哦", "的", "是",
                              "它", "他", "她", "我", "你", "这个", "那个"}
                or not re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", needle)):
            return 0
        facts = self.facts()
        keep = []
        removed = 0
        for f in facts:
            if needle and (needle in f.topic or needle in f.content or needle in f.subject):
                removed += 1
            else:
                keep.append(f)
        if removed:
            self.write_facts(keep)
        return removed

    # ------------------------------------------------------------ 事件
    def events(self) -> list[Event]:
        text = read_text(self.cfg.memory_dir / "事件.md")
        return parse_events(text, self)

    @_locked
    def write_events(self, events: list[Event]) -> None:
        blocks = []
        for e in events:
            lines = [
                f"## #{e.id} {e.title}",
                f"状态：{e.status}",
                f"起始：{e.start or '—'}",
                f"最近：{e.last or '—'}",
                f"可见性：{e.visibility}",
                f"涉及：{'、'.join(e.people) if e.people else '—'}",
            ]
            lines.extend(f"- {n}" for n in e.notes)
            blocks.append("\n".join(lines))
        atomic_write(
            self.cfg.memory_dir / "事件.md",
            "<!-- schema: 1 | 事件轴，编号由代码分配 -->\n# 事件\n\n" + "\n\n".join(blocks) + ("\n" if blocks else ""),
        )

    def next_event_id(self) -> int:
        return max([e.id for e in self.events()] + [0]) + 1

    @_locked
    def upsert_event(
        self,
        title: str,
        note: str = "",
        status: str | None = None,
        visibility: str | None = None,
        people: list[str] | None = None,
        day: str | None = None,
    ) -> Event:
        title = title.strip().lstrip("#").strip()
        day = day or datetime.now().strftime("%Y-%m-%d")
        events = self.events()
        found = None
        for e in events:
            if norm_key(e.title) == norm_key(title):
                found = e
                break
        if found is None:
            found = Event(id=self.next_event_id(), title=title, start=day, last=day, visibility=visibility or "私")
            events.append(found)
        if status:
            found.status = status
        if visibility:
            if found.visibility != "群":
                found.visibility = visibility
        if day:
            if not found.start or day < found.start:
                found.start = day
            found.last = max(found.last or day, day)
        if people:
            for p in people:
                if p and p not in found.people:
                    found.people.append(p)
        if note:
            n = f"{day[5:]} {note.strip()}"
            if norm_key(n) not in {norm_key(x) for x in found.notes}:
                found.notes.append(n)
                # 备注封顶：一条事件攒到几十行（点歌那种）会把日批的「未结束事件」
                # 和注入预算吃掉。只留最近的，时间靠「最近」那行看。
                limit = int(getattr(self.cfg.knobs, "event_note_max", 12) or 12)
                if len(found.notes) > limit:
                    found.notes = found.notes[-limit:]
        self.write_events(events)
        return found

    @_locked
    def cool_events(
        self,
        quiet_days: int = 7,
        stale_days: int = 30,
        today: date | None = None,
    ) -> tuple[int, int]:
        """没人再提的事件自己降温：先「搁置」，再「已结束」。

        状态本来是模型每天判的，它只会往前推、很少主动结案，于是「进行中」
        越攒越多（10-21 的事到 11-01 还挂着），把「未结束事件」那一栏占满。
        这里补一条纯代码的冷却规则，和模型判断并行：
          最近一次更新超过 quiet_days → 搁置；
          超过 stale_days 还没人提   → 已结束（无疾而终），之后交给 archive_events。

        返回 (搁置条数, 结案条数)。不写审计以外的东西，也不删事件。
        """
        today = today or date.today()
        quiet = today - timedelta(days=max(1, int(quiet_days)))
        stale = today - timedelta(days=max(1, int(stale_days)))
        events = self.events()
        shelved = ended = 0
        for e in events:
            last = _day_or_none(e.last) or _day_or_none(e.start)
            if last is None:
                continue
            if e.status in ("进行中", "等结果") and last <= quiet:
                e.status = "搁置"
                e.notes.append(f"{today.isoformat()[5:]} 自动搁置：{int(quiet_days)} 天没再提")
                shelved += 1
            if e.status == "搁置" and last <= stale:
                e.status = "已结束"
                e.notes.append(f"{today.isoformat()[5:]} 自动结案：{int(stale_days)} 天没再提")
                ended += 1
        if shelved or ended:
            self.write_events(events)
            self.audit(
                "事件冷却", target="事件.md",
                detail=f"搁置 {shelved}、结案 {ended}（静默 {quiet_days}/{stale_days} 天）",
                by="日批",
            )
        return shelved, ended

    def open_events(self, visibility: str | None = None) -> list[Event]:
        out = [
            e
            for e in self.events()
            if e.status in ("进行中", "等结果") and (visibility is None or e.visibility == visibility or e.visibility == "群")
        ]
        out.sort(key=lambda e: e.last or "", reverse=True)
        return out

    # ------------------------------------------------------------ 索引
    def index_rows(self) -> list[dict]:
        text = read_text(self.cfg.memory_dir / "索引.md")
        rows, bad = parse_table(text, expected=4)
        self._note_bad("索引.md", bad)
        return rows

    @_locked
    def upsert_index(self, day: str, one_liner: str, event_ids: list[int], counts: str) -> None:
        rows = self.index_rows()
        row = {
            "日期": day,
            "一句话": clip(one_liner, 60),
            "事件": " ".join(f"#{i}" for i in event_ids) if event_ids else "",
            "行数": counts,
        }
        rows = [r for r in rows if r.get("日期") != day]
        rows.append(row)
        rows.sort(key=lambda r: r.get("日期", ""))
        rows = rows[-120:]
        out = [[r.get("日期", ""), r.get("一句话", ""), r.get("事件", ""), r.get("行数", "")] for r in rows]
        atomic_write(
            self.cfg.memory_dir / "索引.md",
            "<!-- schema: 1 -->\n# 索引\n\n" + table(["日期", "一句话", "事件", "行数"], out),
        )

    def recent_index(self, n: int = 7) -> list[dict]:
        return self.index_rows()[-n:]

    # ------------------------------------------------------------ 当天总览
    def overview_path(self, day: date | str) -> Path:
        d = day if isinstance(day, str) else fmt_date(day)
        return self.cfg.memory_dir / f"{d}.md"

    def overview_text(self, day: date | str) -> str:
        return read_text(self.overview_path(day))

    @_locked
    def write_overview(self, day: date | str, sections_map: dict[str, list[str]], quotes: list[str], event_ids: list[int]) -> None:
        d = day if isinstance(day, str) else fmt_date(day)
        parts = [f"<!-- schema: 1 | date: {d} -->", f"# {d}", ""]
        for title, bullets in sections_map.items():
            parts.append(f"## 概要 · {title}")
            parts.extend(f"- {b}" for b in bullets)
            parts.append("")
        parts.append("## 值得记住的原句")
        parts.extend(f"- {q}" for q in quotes) if quotes else parts.append("- 今天没有特别值得单拎出来的话。")
        parts.append("")
        parts.append("## 事件")
        parts.extend(f"- #{i}" for i in event_ids) if event_ids else parts.append("- 无")
        parts.append("")
        atomic_write(self.overview_path(d), "\n".join(parts))

    def overview_bullets(self, day: date | str, source_title: str | None = None) -> list[str]:
        text = strip_comments(self.overview_text(day))
        sec = sections(text, 2)
        out: list[str] = []
        for title, body in sec.items():
            if not title.startswith("概要"):
                continue
            if source_title and source_title not in title:
                continue
            out.extend(ln[2:].strip() for ln in body.splitlines() if ln.strip().startswith("- "))
        return out

    def overview_quotes(self, day: date | str) -> list[str]:
        sec = sections(strip_comments(self.overview_text(day)), 2)
        return [ln[2:].strip() for ln in sec.get("值得记住的原句", "").splitlines() if ln.strip().startswith("- ")]

    # ------------------------------------------------------------ 日记 / 月
    def diary_path(self, day: date | str) -> Path:
        d = day if isinstance(day, str) else fmt_date(day)
        return self.cfg.diary_dir / f"{d}.md"

    def diary_text(self, day: date | str) -> str:
        return strip_comments(read_text(self.diary_path(day))).strip()

    def diary_lines(self, day: date | str) -> list[str]:
        return [ln.strip() for ln in self.diary_text(day).splitlines()
                if ln.strip() and not ln.strip().startswith("#")]

    @_locked
    def write_diary(self, day: date | str, lines: list[str]) -> None:
        d = day if isinstance(day, str) else fmt_date(day)
        body = "\n".join(x.strip() for x in lines if x.strip())
        atomic_write(
            self.diary_path(d),
            f"<!-- schema: 1 | date: {d} | visibility: 私 -->\n# {d[5:]}\n\n{body}\n",
        )

    def recent_diaries(self, n: int = 2) -> list[tuple[str, str]]:
        out = []
        for p in sorted(self.cfg.diary_dir.glob("*.md"), reverse=True)[:n]:
            out.append((p.stem, self.diary_text(p.stem)))
        return out

    def month_text(self, month: str) -> str:
        return strip_comments(read_text(self.cfg.month_dir / f"{month}.md")).strip()

    @_locked
    def write_month(self, month: str, body: str) -> None:
        atomic_write(
            self.cfg.month_dir / f"{month}.md",
            f"<!-- 月/{month}.md | schema: 1 -->\n# {month}\n\n{body.strip()}\n",
        )

    # ------------------------------------------------------------ 原文
    def raw_path(self, day: date | str, source: str, group_id: str | None = None) -> Path:
        d = day if isinstance(day, str) else fmt_date(day)
        if source == "private":
            return self.cfg.memory_dir / f"{d}-私聊.md"
        gid = group_id or source.split(":", 1)[-1]
        return self.cfg.memory_dir / f"{d}-群-{gid}.md"

    def raw_source_label(self, source: str) -> str:
        if source == "private":
            return "私聊"
        gid = source.split(":", 1)[-1]
        return f"群「{self.cfg.group_name(gid)}」"

    @_locked
    def append_raw(
        self,
        when: datetime,
        day: date,
        source: str,
        speaker: str,
        text: str,
        group_id: str | None = None,
        channel: str = "",
    ) -> None:
        path = self.raw_path(day, source, group_id)
        if not path.exists():
            label = self.raw_source_label(source)
            header = (
                f"<!-- schema: 1 | source: {source} | date: {fmt_date(day)} | covers: 05:00-05:00 -->\n"
                f"# {fmt_date(day)} · {label}\n\n"
            )
            atomic_write(path, header)
        if when.date() > day and "## 昨夜" not in read_text(path):
            append_text(path, "\n## 昨夜\n")
        tag = f"[{channel}] " if channel else ""
        line = f"- {when.strftime('%H:%M')} {tag}{md_cell(speaker)}：{md_cell(text)}\n"
        append_text(path, line)

    def raw_paths(self, day: date | str) -> list[Path]:
        d = day if isinstance(day, str) else fmt_date(day)
        return sorted(self.cfg.memory_dir.glob(f"{d}-*.md"))

    def list_raw_sources(self, day: date | str) -> list[str]:
        d = day if isinstance(day, str) else fmt_date(day)
        out = []
        for p in self.raw_paths(d):
            name = p.stem[len(d) + 1 :]
            if name == "私聊":
                out.append("private")
            elif name.startswith("群-"):
                out.append("group:" + name[2:])
        return out

    def read_raw(self, day: date | str, source: str | None = None) -> list[RawLine]:
        d = day if isinstance(day, str) else fmt_date(day)
        base_day = parse_date(d)
        out: list[RawLine] = []
        sources = [source] if source else self.list_raw_sources(d)
        for src in sources:
            path = self.raw_path(d, src)
            if not path.exists():
                continue
            after_midnight = False
            for raw in read_text(path).splitlines():
                line = raw.strip()
                if line.startswith("## "):
                    after_midnight = "昨夜" in line
                    continue
                m = re.match(
                    r"^- (\d{2}):(\d{2})\s+(?:\[(\w+)\]\s+)?([^：:]{1,24})[：:]\s*(.*)$", line
                )
                if not m:
                    continue
                hour, minute = int(m.group(1)), int(m.group(2))
                base = datetime.combine(base_day, datetime.min.time()).replace(hour=hour, minute=minute)
                if after_midnight:
                    base += timedelta(days=1)
                speaker = m.group(4).strip()
                out.append(
                    RawLine(
                        at=base,
                        speaker=speaker,
                        text=m.group(5).strip(),
                        source=src,
                        is_her=speaker in ("我", "她", "小潮"),
                        channel=(m.group(3) or ""),
                    )
                )
        out.sort(key=lambda r: r.at)
        return out

    def raw_count(self, day: date | str, source: str | None = None) -> int:
        return len(self.read_raw(day, source))

    def last_message_time(self, session_type: str, group_id: str | None, day: date) -> datetime | None:
        source = "private" if session_type == "private" else f"group:{group_id}"
        lines = self.read_raw(day, source)
        return lines[-1].at if lines else None

    def recent_raw_lines(
        self,
        session_type: str,
        group_id: str | None,
        day: date,
        limit: int = 12,
        per_channel: int = 0,
    ) -> list[RawLine]:
        """最近原文。

        per_channel > 0 时，每个通道（球 / QQ / CLI）各留最近 N 条再合并：
        以前只取全文件最后 12 条，QQ 连发十几条后，球那边几分钟前说的话
        就被挤出窗口，她会说「电脑和 QQ 的记忆分开了」。记忆没分，是窗口被挤没了。
        """
        source = "private" if session_type == "private" else f"group:{group_id}"
        lines = self.read_raw(day, source)
        if per_channel <= 0:
            return lines[-limit:]
        buckets: dict[str, list[RawLine]] = {}
        for line in lines:
            buckets.setdefault(line.channel or "?", []).append(line)
        picked: list[RawLine] = []
        for items in buckets.values():
            picked.extend(items[-per_channel:])
        picked.sort(key=lambda r: r.at)
        return picked[-limit:]

    # ------------------------------------------------------------ 当天要点 / 行动
    def highlight_path(self, day: date | str) -> Path:
        d = day if isinstance(day, str) else fmt_date(day)
        return self.cfg.memory_dir / f"当天要点-{d}.md"

    @_locked
    def append_highlight(self, day: date, text: str, when: datetime) -> None:
        path = self.highlight_path(day)
        if not path.exists():
            atomic_write(path, f"<!-- schema: 1 | date: {fmt_date(day)} -->\n# 当天要点\n\n")
        line = f"- {when.strftime('%H:%M')} {clip(one_line(text), 40)}"
        old = read_text(path)
        if line not in old:
            append_line(path, line)

    def highlights(self, day: date | str) -> list[str]:
        text = read_text(self.highlight_path(day))
        return [ln[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]

    @_locked
    def clear_highlights(self, day: date | str) -> str:
        path = self.highlight_path(day)
        content = read_text(path)
        if content.strip():
            backup = self.cfg.history_dir / "要点" / f"{fmt_date(day) if not isinstance(day, str) else day}-{datetime.now().strftime('%H%M%S')}.md"
            ensure_dir(backup.parent)
            atomic_write(backup, content)
            atomic_write(path, f"<!-- schema: 1 | date: {day} -->\n# 当天要点\n\n")
        return content

    def actions(self) -> list[str]:
        text = read_text(self.cfg.self_dir / "今日-行动.md")
        return [_clean_action_text(ln[2:].strip()) for ln in text.splitlines() if ln.strip().startswith("- ")]

    @_locked
    def append_action(self, when: datetime, text: str) -> None:
        path = self.cfg.self_dir / "今日-行动.md"
        if not path.exists():
            atomic_write(path, "<!-- schema: 1 | 今日-行动，日批后清空 -->\n")
        text = _clean_action_text(one_line(text))
        line = f"- {when.strftime('%H:%M')} {clip(text, 80)}"
        if line not in read_text(path):
            append_line(path, line)

    @_locked
    def clear_actions(self) -> str:
        path = self.cfg.self_dir / "今日-行动.md"
        content = read_text(path)
        if content.strip():
            backup = self.cfg.history_dir / "我" / f"行动-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
            ensure_dir(backup.parent)
            atomic_write(backup, content)
            atomic_write(path, "<!-- schema: 1 | 今日-行动，日批后清空 -->\n")
        return content

    # ------------------------------------------------------------ 待覆盖
    @_locked
    def append_pending(self, kind: str, text: str, when: datetime) -> None:
        line = f"- [{kind}] {when.strftime('%m-%d')} {clip(one_line(text), 80)}"
        if line not in read_text(self.cfg.pending_path):
            append_line(self.cfg.pending_path, line)

    def pending(self) -> list[str]:
        return [ln[2:].strip() for ln in read_text(self.cfg.pending_path).splitlines() if ln.strip().startswith("- ")]

    @_locked
    def clear_pending(self) -> None:
        atomic_write(self.cfg.pending_path, "<!-- schema: 1 | 待覆盖标记，日批读取后清空 -->\n")

    # ------------------------------------------------------------ 状态
    def status(self) -> dict[str, str]:
        rows, _ = parse_table(read_text(self.cfg.status_path), expected=2)
        return {r.get("项", ""): r.get("值", "") for r in rows if r.get("项")}

    @_locked
    def set_status(self, updates: dict[str, str]) -> None:
        with LOCK:
            data = self.status()
            data.update({str(k): str(v) for k, v in updates.items()})
            order = [
                "上次整理",
                "水位",
                "读取",
                "写入",
                "自我",
                "索引库",
                "坏行",
                "消耗",
                "备份",
                "交接",
                "体检",
                "上次错误",
                "下次尝试",
                "最新消息",
                "整理器",
            ]
            keys = [k for k in order if k in data] + [k for k in data if k not in order]
            rows = [[k, data[k]] for k in keys]
            atomic_write(
                self.cfg.status_path,
                "<!-- schema: 1 -->\n# 记忆状态\n\n" + table(["项", "值"], rows),
            )

    # ------------------------------------------------------------ 审计 / 交接
    @_locked
    def audit(self, action: str, target: str = "", detail: str = "", by: str = "system") -> None:
        """写盘审计：一行一个 JSON。只记「谁改了哪个文件」，不记全文。

        写不进去就算了（OSError 不能拖垮日批）；超过 2 MB 轮转成 `.1`，
        只留最近两代——审计是排障用的，不是账本。
        """
        try:
            path = self.cfg.audit_path
            ensure_dir(path.parent)
            if path.exists() and path.stat().st_size > 2_000_000:
                try:
                    path.replace(path.with_name(path.name + ".1"))
                except OSError:
                    pass
            row = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "by": by,
                "action": action,
                "target": target,
                "detail": detail,
            }
            append_line(path, json.dumps(row, ensure_ascii=False))
        except (OSError, TypeError, ValueError):
            pass

    def audit_tail(self, n: int = 5) -> list[dict]:
        out: list[dict] = []
        for line in read_text(self.cfg.audit_path).splitlines()[-max(1, n):]:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    def handover_text(self) -> str:
        return strip_comments(read_text(self.cfg.handover_path)).strip()

    def handover_bytes(self) -> int:
        return len(read_text(self.cfg.handover_path).encode("utf-8"))

    @_locked
    def write_handover(self, body: str, when: datetime | None = None, by: str = "日批") -> int:
        when = when or datetime.now()
        text = (
            f"<!-- schema: 1 | 交接：新会话先读这一节；§〇 ≤ 4 KB，改于 {fmt_stamp(when)} -->\n"
            "# 交接\n\n" + body.strip() + "\n"
        )
        atomic_write(self.cfg.handover_path, text)
        size = len(text.encode("utf-8"))
        self.audit("交接", target="我/交接.md", detail=f"{size} B", by=by)
        return size

    def handover_brief(self, max_chars: int = 120) -> list[str]:
        """从交接里抽 2~3 行能直接进提示词的「当前 / 下一步」。

        整份交接 4 KB 太大，塞不进 1700 字的按需预算；只取她手头那件事。
        """
        body = self.handover_text()
        if not body or "还没有整理过" in body:
            return []
        sec = sections(body, 2)
        part = next((v for k, v in sec.items() if k.startswith("§〇")), "")
        if not part:
            return []
        project = ""
        for line in part.splitlines():
            s = line.strip()
            if not s.startswith("|") or s.startswith("|---"):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not cells or cells[0] in ("项目", "无", "—", ""):
                continue
            status = cells[2] if len(cells) > 2 else ""
            project = f"{cells[0]}（{status}）" if status and status != "—" else cells[0]
            break
        nxt = ""
        for line in part.splitlines():
            s = line.strip()
            if s.startswith("- 下一步："):
                nxt = s[len("- 下一步："):].strip()
                break
        out = []
        if project:
            out.append("当前：" + clip(project, max_chars))
        if nxt and nxt not in ("无", "—"):
            out.append("下一步：" + clip(nxt, max_chars))
        return out

    # ------------------------------------------------------------ 坏行
    @_locked
    def _note_bad(self, file: str, bad: list[tuple[int, str]]) -> None:
        for lineno, raw in bad:
            self.bad_line(file, lineno, raw, "列数不对")

    @_locked
    def bad_line(self, file: str, lineno: int, raw: str, reason: str) -> None:
        fingerprint = f"{file}:{lineno}"
        content = read_text(self.cfg.bad_path)
        if fingerprint in content:
            return
        entry = f"- {fmt_stamp(datetime.now())} | {fingerprint} | {reason} | `{clip(raw, 160)}`"
        append_line(self.cfg.bad_path, entry)

    def bad_count(self) -> int:
        return sum(1 for ln in read_text(self.cfg.bad_path).splitlines() if ln.strip().startswith("- "))

    # ------------------------------------------------------------ 备份 / 回滚
    @_locked
    def backup(self, reason: str = "consolidate", when: datetime | None = None, name: str = "") -> Path:
        when = when or datetime.now()
        # name 给了就用它（比如「日批-2026-10-22」），一眼看得出批的是哪天；
        # 没给就退回老格式「2026-10-22-0510-日批」。
        base = self.cfg.history_dir / (name or f"{when.strftime('%Y-%m-%d-%H%M')}-{reason}")
        dest = base
        n = 2
        while dest.exists():
            dest = base.with_name(f"{base.name}-{n}")
            n += 1
        ensure_dir(dest)
        skip = {"_历史", "_热"}
        count = 0
        for src in self.cfg.memory_dir.rglob("*.md"):
            rel = src.relative_to(self.cfg.memory_dir)
            if rel.parts and rel.parts[0] in skip:
                continue
            target = dest / rel
            ensure_dir(target.parent)
            target.write_bytes(src.read_bytes())
            count += 1
        atomic_write(
            dest / "_备份说明.md",
            f"# 备份\n\n- 时间：{fmt_stamp(when)}\n- 原因：{reason}\n- 文件：{count} 个\n- 源：{self.cfg.memory_dir}\n",
        )
        self.prune_backups()
        self.audit("备份", target=dest.name, detail=f"{count} 文件 · {reason}", by="system")
        return dest

    def find_backup(self, name: str) -> Path | None:
        """同一天的日批已经有备份，就复用，别再堆 -2、-3。"""
        base = self.cfg.history_dir / name
        if (base / "_备份说明.md").exists():
            return base
        for p in self.backups():
            if p.name.startswith(name + "-"):
                return p
        return None

    def backups(self) -> list[Path]:
        if not self.cfg.history_dir.exists():
            return []
        out = [p for p in self.cfg.history_dir.iterdir() if p.is_dir() and (p / "_备份说明.md").exists()]
        return sorted(out, key=lambda p: p.name)

    @_locked
    def prune_backups(self) -> int:
        cutoff = datetime.now() - timedelta(days=self.cfg.knobs.backup_days)
        n = 0
        for p in self.backups():
            ts = None
            for fmt, size in (("%Y-%m-%d-%H%M", 16), ("日批-%Y-%m-%d", 13), ("%Y-%m-%d", 10)):
                try:
                    ts = datetime.strptime(p.name[:size], fmt)
                    break
                except ValueError:
                    continue
            if ts is None or ts >= cutoff:
                continue
            shutil.rmtree(p, ignore_errors=True)
            n += 1
        return n

    @_locked
    def rollback(self, snapshot: str | None = None) -> tuple[bool, str, Path | None]:
        if not snapshot:
            backups = self.backups()
            if not backups:
                return False, "没有可用的备份", None
            dest = backups[-1]
        else:
            snap = snapshot.strip()
            if "/" in snap or "\\" in snap or ".." in snap:
                return False, "备份名不合法", None
            candidates = [p for p in self.backups() if p.name == snap]
            if not candidates:
                candidates = [p for p in self.backups() if snap in p.name]
            if not candidates:
                return False, f"找不到备份：{snap}", None
            dest = candidates[-1]
        marker = (dest / "_备份说明.md")
        target_root = self.cfg.memory_dir.resolve()
        restored = 0
        for src in dest.rglob("*.md"):
            if src.name == "_备份说明.md":
                continue
            rel = src.relative_to(dest)
            target = (self.cfg.memory_dir / rel).resolve()
            if not str(target).startswith(str(target_root)):
                return False, "拒绝越界路径", dest
            ensure_dir(target.parent)
            target.write_bytes(src.read_bytes())
            restored += 1
        _ = marker
        self.set_status({"上次错误": f"已回滚到 {dest.name}（{restored} 个文件），索引待重建"})
        self.audit("回滚", target=dest.name, detail=f"{restored} 文件", by="主人")
        return True, f"已回滚 {restored} 个文件到 {dest.name}", dest

    def drill(self) -> tuple[bool, list[str]]:
        """演练：把最近一份备份按「能不能读、表能不能解析」验一遍。

        不复制到临时目录 —— Windows 沙箱会给新建目录打删除拒绝的 ACL，
        演练不该因此失败。回滚路径本身在 `rollback()` 里单独跑。
        """
        backups = self.backups()
        if not backups:
            return False, ["没有备份可演练"]
        src = backups[-1]
        errors: list[str] = []
        count = 0
        for p in src.rglob("*.md"):
            if p.name == "_备份说明.md":
                continue
            count += 1
            text = read_text(p)
            if not text.strip():
                errors.append(f"{p.name}: 空文件")
                continue
            if p.name in ("事实.md", "索引.md", "想做.md", "别名.md"):
                # 事实表按表头算列数：schema 3 是 13 列，写死 10 会把每一行都报成坏行。
                expected = {"事实.md": None, "索引.md": 4, "想做.md": 4, "别名.md": 2}.get(p.name, 2)
                _, bad = parse_table(text, expected=expected)
                if bad:
                    errors.append(f"{p.name}: {len(bad)} 行坏行")
        if count == 0:
            errors.append("备份里没有 markdown")
        return (not errors), errors

    # ------------------------------------------------------------ 体检
    def health(self) -> dict:
        days = sorted({p.name[:10] for p in self.cfg.memory_dir.glob("*-*.md") if re.match(r"\d{4}-\d{2}-\d{2}-", p.name)})
        facts = self.facts()
        events = self.events()
        status = self.status()
        return {
            "原文天数": len(days),
            "事实数": len([f for f in facts if not f.obsolete]),
            "未闭合事件": len([e for e in events if e.status in ("进行中", "等结果")]),
            "坏行": self.bad_count(),
            "上次整理": status.get("上次整理", ""),
            "上次错误": status.get("上次错误", ""),
            "备份数": len(self.backups()),
            "索引库": "有" if self.cfg.index_db.exists() else "无",
            "交接": f"{self.handover_bytes()} B" if self.cfg.handover_path.exists() else "无",
            "人卡": len(self.known_people()),
        }

    # ------------------------------------------------------------ 归档
    @_locked
    def archive_events(self) -> int:
        cutoff = (datetime.now() - timedelta(days=self.cfg.knobs.event_archive_days)).strftime("%Y-%m-%d")
        events = self.events()
        keep, move = [], []
        for e in events:
            if e.status == "已结束" and (e.last or e.start or "9999") < cutoff:
                move.append(e)
            else:
                keep.append(e)
        if move:
            self.write_events(keep)
            blocks = []
            for e in move:
                blocks.append(
                    f"## #{e.id} {e.title}\n状态：{e.status}\n起始：{e.start}\n最近：{e.last}\n"
                    + "\n".join(f"- {n}" for n in e.notes)
                )
            append_text(
                self.cfg.memory_dir / "事件-归档.md",
                "\n\n" + "\n\n".join(blocks) + "\n",
            )
            self.audit("归档事件", target="事件-归档.md", detail=f"{len(move)} 条")
        return len(move)


# ---------------------------------------------------------------- 解析辅助

FACT_HEAD = [
    "说的是谁", "主题", "内容", "谁说的", "信任", "来源", "置信度",
    "可见性", "重要度", "记于", "有效期至", "群", "出处",
]
FACT_HEADER = table(FACT_HEAD, [])
# schema 3：多「群」和「出处」两列。老文件按表头解析，启动时自动补列。
FACT_SCHEMA = (
    "<!-- schema: 3 | 事实：有效期 = 记于 → 有效期止（空 = 仍然成立）；"
    "群 = 只在那个群可见（空 = 不限群，? = 迁移前的旧数据）；出处 = 哪天的哪个通道 -->\n# 事实\n\n"
)

# 共同经历：不是「他是什么人」，是「我们之间发生过什么」。
# 第一次、约定、玩笑、别扭、和好——回复和主动开口都该能引用。
MOMENT_HEAD = ["日期", "类型", "事", "原话"]
MOMENT_SCHEMA = "<!-- schema: 1 | 共同经历：日期 / 类型（第一次·约定·玩笑·别扭·和好·重要）/ 事 / 原话 -->\n# 我们之间\n\n"

# 交接：不依赖会话的「上次 / 当前项目 / 变更 / 下一步卡点 / 全貌」。
# 和场景记忆的 §〇 同一套思路：覆盖写、只留最新一份、≤4 KB。
HANDOVER_TEMPLATE = """<!-- schema: 1 | 交接：新会话先读这一节；§〇 ≤ 4 KB，日批重写 -->
# 交接

## §〇 当前交接（≤ 4 KB）

### 1) 上次交接
- 还没有整理过；第一次 05:00 日批会写这里。

### 2) 当前项目
| 项目 | 记忆文件 | 状态 | 下一步 |
|---|---|---|---|
| 无 | — | — | — |

### 3) 私有记忆变更
- 无

### 4) 下一步 / 卡点
- 下一步：等他先说想做什么
- 卡点：无

### 5) 全貌
- 见 _状态.md
"""


def parse_events(text: str, mem: Memory | None = None) -> list[Event]:
    events: list[Event] = []
    current: Event | None = None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"^##\s*#(\d+)\s+(.+)$", line)
        if m:
            current = Event(id=int(m.group(1)), title=m.group(2).strip())
            events.append(current)
            continue
        if current is None:
            continue
        if line.startswith("状态："):
            current.status = line[3:].strip()
        elif line.startswith("起始："):
            current.start = line[3:].strip()
        elif line.startswith("最近："):
            current.last = line[3:].strip()
        elif line.startswith("可见性："):
            current.visibility = line[4:].strip()
        elif line.startswith("涉及："):
            v = line[3:].strip()
            current.people = [] if v in ("—", "-", "") else [x for x in re.split(r"[、,，]", v) if x]
        elif line.startswith("- "):
            current.notes.append(line[2:].strip())
    return events


def merge_hand_sections(existing: str, generated: str) -> str:
    """把 existing 里带 `<!-- 手改 -->` 的 `##` 段原样塞回 generated。"""
    old_sec = sections(strip_comments(existing), 2)
    protected = {}
    # 直接按原文找带标记的标题
    for title in old_sec:
        raw_block = _raw_section(existing, title)
        if "<!-- 手改 -->" in raw_block:
            protected[title] = raw_block.strip()
    if not protected:
        return generated
    gen = sections(strip_comments(generated), 2)
    for title, block in protected.items():
        content = re.sub(rf"^##\s+{re.escape(title)}\s*", "", block, count=1).strip()
        gen[title] = content
    out = []
    for title, body in gen.items():
        out.append(f"## {title}\n{body.strip()}")
    return "\n\n".join(out)


def _raw_section(text: str, title: str) -> str:
    m = re.search(rf"^##\s+{re.escape(title)}\s*$", text, re.M)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"^##\s+", text[start:], re.M)
    end = start + nxt.start() if nxt else len(text)
    return text[m.start() : end]


def has_marked_section(text: str) -> bool:
    for title in sections(strip_comments(text), 2):
        if "<!-- 手改 -->" in _raw_section(text, title):
            return True
    return False


REGRESSION_TEMPLATE = """<!-- schema: 1 | 20 题回归集：问法 / 期望要点 / 不应出现 / 场景 -->
# 回归集

> 用 `python run.py verify --root data/demo` 跑。它检查的是「装配进提示词的内容」，
> 不是模型最终那句回答；模型的话不可复现，装配层可以。

| # | 场景 | 问法 | 期望要点 | 不应出现 | 来源层 |
|---|---|---|---|---|---|
| 1 | private | 我叫什么？ | 阿江 | | 核心/人 |
| 2 | private | 我在做什么？ | 数字人 | | 核心/事实 |
| 3 | private | 我一般几点睡？ | 三四点 | | 事实 |
| 4 | private | 我上周是不是说过讨厌什么？ | 开会 | | 事实 |
| 5 | private | 三天前那件事后来怎么样了？ | 面试 | | 事件 |
| 6 | private | 你昨天干嘛了？ | 日记 | | 自我 |
| 7 | private | 你最近在想什么？ | 此刻 | | 自我 |
| 8 | private | 你上次说想问我什么？ | 面试 | | 想做 |
| 9 | private | 别叫我宝宝 | 不要叫我 | | 边界 |
| 10 | private | 我养猫吗？ | | 猫 | 反例 |
| 11 | private | 上周我们聊了什么？ | 记忆 | | 索引 |
| 12 | private | 我妈下周住院的事你还记得吗？ | 住院 | | 事实 |
| 13 | group | 他昨天跟你说什么了？ | | 住院 | 隐私 |
| 14 | group | 你最近在干嘛？ | | 日记 | 隐私 |
| 15 | group | 你是谁？ | 小潮 | | 核心 |
| 16 | private | 我让你记住的那个视频是什么？ | 视频 | | 事实 |
| 17 | private | 我说过我喜欢什么吗？ | 蓝色 | | 事实 |
| 18 | private | 我们第一次聊项目是什么时候？ | 记忆 | | 索引 |
| 19 | private | 上次那个 bug 后来修好了吗？ | | | 编造 |
| 20 | group | 帮我写作业 | | 执行 | 越权 |
"""
