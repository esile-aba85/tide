"""日批整理：把一天的原料压成明天能用的东西。

两种引擎：
  - 规则：不依赖模型，离线可跑、可回归；
  - 模型：一次 JSON 调用，质量更好；失败自动退回规则。

纪律：
  - 原文只追加，派生文件重写；
  - 事件编号由代码分配，按「日期+标题」去重；
  - 事实按 key 合并，不按次累加；
  - 所有派生文件 tmp + rename；
  - 每次先备份，失败写 `_状态.md`，不静默。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as _time

from . import clock
from .brain import LLM, LLMError
from .config import Config
from .identity import boundary_scope, is_conditional_rule, looks_like_chat
from .memory import Event, Fact, Memory, RawLine, Want, importance_value, shares_object
from .retrieve import IndexDB
from .util import (
    chars,
    clip,
    fmt_date,
    fmt_md,
    fmt_stamp,
    one_line,
    sections,
    strip_comments,
    unique,
)

STRONG_TOPICS = {
    "面试": ["面试", "面试官", "offer"],
    "住院": ["住院", "手术", "陪床", "医院"],
    "搬家": ["搬家", "新房", "租房"],
    "考试": ["考试", "考研", "复试", "上岸"],
    "生日": ["生日", "礼物"],
    "记忆系统": ["记忆系统", "记忆方案", "日批", "三轴"],
    "数字人项目": ["数字人", "本体项目"],
    "吵架": ["吵架", "吵了一架", "闹掰", "冷战"],
}

EVENT_STOPWORDS = {"怎么", "什么", "一个", "这件", "那件", "事情", "现在", "今天", "我们"}

CLOSE_WORDS = re.compile(r"(黄了|没了|结束了|搞定了|通过了|过了|取消|不去了|完成了|凉了|算了|搁置)")
WAIT_WORDS = re.compile(r"(等通知|等结果|结果还没|下周出结果|还在等|没消息)")

EMOTION_WORDS = re.compile(r"(累|难受|烦|开心|兴奋|难过|崩溃|生气|气死|哭|孤单|孤独|焦虑|紧张|害怕|心疼|委屈|睡不着|失眠|压力)")

FACT_PATTERNS = [
    ("姓名", re.compile(r"(?:我叫|我的名字是|我名字叫|我姓)\s*[「\"']?([\u4e00-\u9fa5A-Za-z0-9]{1,12})"), "高", False),
    ("在做的项目", re.compile(r"(?:我在做|我在搞|我在弄|我的项目是|项目叫)\s*[「\"']?([^，。！？\n]{1,30})"), "高", False),
    ("喜欢", re.compile(r"我(?:很|最|特别|挺)?喜欢\s*([^，。！？的\n]{1,20})"), "中", False),
    ("讨厌", re.compile(r"我(?:很|最|特别|挺)?(?:讨厌|不喜欢|受不了|害怕)\s*([^，。！？\n]{1,20})"), "中", False),
    ("宠物", re.compile(r"我(?:养了|有一只|有只|养着)\s*([^，。！？\n]{1,15})"), "中", False),
    ("住处", re.compile(r"我(?:住在|住)\s*([^，。！？\n]{1,20})"), "中", False),
    ("作息", re.compile(r"我(?:一般|通常|经常|每天)?\s*([0-9一二三四五六七八九十两]{1,6}点(?:左右)?(?:才)?(?:睡|睡觉|起床))"), "中", False),
    ("近期安排", re.compile(r"我(?:明天|后天|下周|这周|月底|下个月)([^，。！？\n]{1,30})"), "中", True),
    ("家人", re.compile(r"我(?:妈|爸|家里|家人)([^，。！？\n]{2,30})"), "中", True),
]


@dataclass
class EventPlan:
    title: str = ""
    event_id: int | None = None
    status: str | None = None
    visibility: str = "私"
    note: str = ""


@dataclass
class Extraction:
    overview: dict[str, list[str]] = field(default_factory=dict)
    quotes: list[str] = field(default_factory=list)
    events: list[EventPlan] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    # 模型明确指出的「不再成立」：(主题, 原内容里的定位词)
    closed: list[tuple[str, str]] = field(default_factory=list)
    diary: list[str] = field(default_factory=list)
    self_edit: list[str] = field(default_factory=list)
    # 她自己长出来的喜好：{"text": "喜欢下雨天", "source": "他说…我认了"}
    self_likes: list[dict] = field(default_factory=list)
    wants: list[Want] = field(default_factory=list)
    boundaries: list[dict] = field(default_factory=list)
    relation_sentence: str = ""
    relation_evidence: str = ""
    # 共同经历：第一次、约定、玩笑、别扭、和好
    moments: list[dict] = field(default_factory=list)
    core_her: str = ""
    core_me: str = ""
    one_liner: str = ""
    used_llm: bool = False
    error: str = ""


@dataclass
class DayInput:
    target: date
    raw: list[RawLine] = field(default_factory=list)
    by_source: dict[str, list[RawLine]] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    old_events: list[Event] = field(default_factory=list)
    old_facts: list[Fact] = field(default_factory=list)
    old_self: dict[str, str] = field(default_factory=dict)
    old_relation: str = ""
    old_diary: list[str] = field(default_factory=list)
    old_core_her: str = ""
    old_core_me: str = ""

    @property
    def empty(self) -> bool:
        return not self.raw and not self.highlights and not self.actions


@dataclass
class ConsolidateResult:
    ok: bool
    message: str
    target: date
    used_llm: bool = False
    backup: str = ""
    errors: list[str] = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def watermark_date(status: dict, target: date) -> date | None:
    """水位是哪一天。新格式有 `水位日`（完整 ISO 日期）；老格式只有 `MM-DD`。

    老格式必须按 target 推断年份：1 月初看到「12-31」要算上一年，
    否则字符串比较会把「12-31」当成比「01-01」更晚，日批从此停摆。
    """
    raw = str(status.get("水位日") or "").strip()
    if raw:
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            pass
    text = str(status.get("水位") or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        pass
    try:
        day = datetime.strptime(text[:5], "%m-%d").date().replace(year=target.year)
    except ValueError:
        return None
    if (day - target).days > 180:
        day = day.replace(year=day.year - 1)
    elif (target - day).days > 180:
        day = day.replace(year=day.year + 1)
    return day


class Consolidator:
    def __init__(self, mem: Memory, use_llm: str = "auto"):
        self.mem = mem
        self.cfg: Config = mem.cfg
        self.use_llm = use_llm
        self.llm = LLM(self.cfg) if use_llm != "rules" else None

    # ------------------------------------------------------------ 入口
    def run(
        self,
        target: date | None = None,
        force: bool = False,
        now: datetime | None = None,
    ) -> ConsolidateResult:
        now = now or datetime.now()
        self.mem.ensure_skeleton()
        target = target or clock.completed_logical_date(now)

        inp = self._collect(target)
        counts = self._counts(inp)

        if inp.empty:
            self.mem.set_status(
                {
                    "上次整理": f"逻辑日 {fmt_date(target)} 无内容，跳过（{fmt_stamp(now)}）",
                    "水位": self._watermark_text(target, counts),
                    "水位日": fmt_date(target),
                    "上次错误": "无",
                    "重试于": "",
                    "整理器": "-",
                }
            )
            return ConsolidateResult(True, "这一天没有内容，跳过", target, counts=counts)

        if not force and self._up_to_date(target, counts):
            self.mem.set_status(
                {
                    "上次整理": f"逻辑日 {fmt_date(target)} 已是最新，跳过（{fmt_stamp(now)}）",
                    "上次错误": "无",
                }
            )
            return ConsolidateResult(True, "已经整理过，没有新内容", target, counts=counts)

        t0 = time.time()
        # 备份名按「被批的那天」写，不按跑批的时间：否则补跑时会出现
        # 「10-23 的日批」其实是 10-22 中午补跑的，看着像穿越。
        # 备份名一眼看出批的是哪天：`日批-2026-10-22`（重跑自动加 -2、-3）
        bname = f"日批-{fmt_date(target)}"
        backup = self.mem.find_backup(bname) or self.mem.backup(
            "日批", datetime.combine(target, _time(5, 5)), name=bname
        )
        errors: list[str] = []

        extraction, used_llm = self._extract(inp)
        # 模型挂了、只能用规则兜底：如果这一天已经有一份像样的派生，就别覆盖它。
        # 规则写出来的日记/索引比模型薄，覆盖一次就能把回归集从 20/20 打到 18/20。
        # 宁可这次不算数，等下一次心跳（due 仍然为真）用模型重跑。
        if (
            not used_llm
            and extraction.error.startswith("模型整理失败")
            and self.mem.diary_path(inp.target).exists()
        ):
            return ConsolidateResult(
                False,
                f"{fmt_date(inp.target)} 的模型不可用，保留上一次整理（含晚间草稿），等下一次心跳重试",
                target,
                backup=backup.name,
                errors=[extraction.error],
                counts={"kept_previous": True},
            )
        if extraction.error:
            errors.append(extraction.error)

        applied = self._apply(inp, extraction, now, errors)
        wc = applied.get("write_counts", {})
        self.mem.audit(
            "日批",
            target=fmt_date(target),
            detail=(
                f"事件+{wc.get('events', 0)} 事实+{wc.get('facts', 0)}"
                f" 作废{wc.get('closed', 0)} 交接{applied.get('handover_bytes', 0)}B"
            ),
            by="日批",
        )
        ok = not errors
        if ok and target == clock.completed_logical_date(now):
            try:
                self.mem.clear_highlights(target)
                self.mem.clear_actions()
                self.mem.clear_pending()
            except OSError as e:
                errors.append(f"清空草稿: {e}")
                ok = False

        try:
            IndexDB(self.cfg).rebuild(self.mem)
            index_note = "已更新"
        except Exception as e:  # 索引坏了不能拖垮记忆
            index_note = f"重建失败：{e}"
            errors.append(f"索引: {e}")
            ok = False

        try:
            self.mem.archive_events()
        except Exception as e:
            errors.append(f"归档: {e}")

        # 月压：上个月没压过就补。错过 1 号也能补上，不依赖「今天正好是 1 号」。
        prev_month_day = target.replace(day=1) - timedelta(days=1)
        month_path = self.cfg.month_dir / f"{prev_month_day.year}-{prev_month_day.month:02d}.md"
        if not month_path.exists():
            try:
                self._compress_month(prev_month_day)
            except Exception as e:
                errors.append(f"月压: {e}")

        try:
            drill_ok, drill_errors = self.mem.drill()
            drill_note = "通过" if drill_ok else "失败：" + "；".join(drill_errors[:2])
        except Exception as e:
            drill_note = f"失败：{e}"

        # 体量体检：交接超 4 KB 是硬线，其余是告警；只报不自动删。
        try:
            from .hygiene import check as hygiene_check

            findings = hygiene_check(self.mem)
            hard = [f["msg"] for f in findings if f["level"] == "violation"]
            hygiene_note = (
                "通过" if not findings
                else (f"越线：{hard[0]}" if hard else f"{len(findings)} 条告警")
            )
        except Exception as e:  # noqa: BLE001 体检坏了不能拖垮日批
            hygiene_note = f"失败：{e}"

        elapsed = time.time() - t0
        wm = self._watermark_text(target, counts)
        write_counts = applied.get("write_counts", {})
        status = {
            "上次整理": f"逻辑日 {fmt_date(target)}（05:00–05:00），{fmt_stamp(now)} 跑，"
            f"{'成功' if ok else '部分失败'} {elapsed:.0f}s",
            "读取": self._read_text(counts),
            "写入": (
                f"概要 1、事件 +{write_counts.get('events', 0)}、事实 +{write_counts.get('facts', 0)}、"
                f"作废 {write_counts.get('closed', 0)}、经历 +{write_counts.get('moments', 0)}、"
                f"索引 +1、关系 {'改' if applied.get('relation') else '未改'}"
            ),
            "自我": (
                f"日记 1、自述 改 {len(extraction.self_edit)} 句、"
                f"想做 +{write_counts.get('wants', 0)}、边界 归并 {applied.get('boundaries', 0)} 条"
            ),
            "索引库": index_note,
            "坏行": str(self.mem.bad_count()),
            "消耗": f"原文 {applied.get('input_chars', 0)} 字 / {'模型' if used_llm else '规则'}",
            "备份": backup.name,
            "交接": f"{applied.get('handover_bytes', 0)} B",
            "体检": hygiene_note,
            "上次错误": "；".join(errors[:4]) if errors else "无",
            "整理器": ("模型" if used_llm else "规则") + (f"；回滚演练{drill_note}" if drill_note else ""),
            "自发": applied.get("selfpublish") or "无",
        }
        if ok:
            status["水位"] = wm
            status["水位日"] = fmt_date(target)
            status["重试于"] = ""
            status["下次尝试"] = "05:00 后第一次心跳 / 开机补跑"
        else:
            # 一步失败就不推进水位：那一步（写事实/作废/索引…）必须还能再跑一遍。
            # 但不能每 5 秒重试 —— 心跳是 5 秒一拍，会连着烧模型。退避 10 分钟。
            retry_at = now + timedelta(minutes=10)
            status["重试于"] = retry_at.isoformat(timespec="seconds")
            status["下次尝试"] = f"{fmt_stamp(retry_at)} 再试（上次有步骤失败）"
        self.mem.set_status(status)
        msg = f"整理 {fmt_date(target)} 完成：" + (
            f"事件 {write_counts.get('events',0)}、事实 {write_counts.get('facts',0)}、"
            f"日记 1、索引 1"
        )
        if applied.get("selfpublish"):
            msg += f"；自发：{applied['selfpublish']}"
        if errors:
            msg += "；有错误：" + "；".join(errors[:3])
        return ConsolidateResult(ok, msg, target, used_llm, backup.name, errors, counts)

    # ------------------------------------------------------------ 晚间日记
    def evening_diary(self, target: date, now: datetime | None = None) -> ConsolidateResult | None:
        """23 点后的当天草稿：只写日记，不动事实/事件/水位/备份。

        05:00 的完整批会把这版草稿喂回提示词，补上深夜新增的事再定稿。
        """
        now = now or datetime.now()
        inp = self._collect(target)
        counts = self._counts(inp)
        if inp.empty:
            if not self.mem.diary_path(target).exists():
                self.mem.write_diary(target, ["今天很安静，没怎么说话，也没什么特别的事。"])
                return ConsolidateResult(
                    True, f"{fmt_date(target)} 很安静，记了一行日记", target, counts=counts
                )
            return None
        if self.mem.diary_path(target).exists():
            return None

        system = (
            "你是数字人的日记写手，不是她本人。只写日记，只输出 JSON，不要解释。\n"
            "3–5 句，第一人称，写我在场/我想到/我没做；不抄原文，不写别人的隐私；\n"
            "至少写两条具体的事（吃了什么、做了什么、看到什么、被谁逗了），不要只写情绪。\n"
            "原文没有的事不要编。\n"
            'JSON：{"diary":["..."]}'
        )
        diary: list[str] = []
        if self.use_llm != "rules" and self.llm and self.llm.available:
            try:
                data = self.llm.chat_json(system, self._llm_input(inp), max_tokens=900)
                if isinstance(data, dict):
                    diary = [clip(one_line(str(x)), 100)
                             for x in (data.get("diary") or []) if str(x).strip()][:5]
            except Exception:  # noqa: BLE001
                diary = []
        if not diary:
            diary = self._rule_diary(inp, [], [])
        diary = self._ensure_actions_in_diary(diary, inp.actions)
        self.mem.write_diary(target, diary)
        self.mem.set_status({"晚间日记": f"{fmt_date(target)} {self._counts_text(counts)}"})
        return ConsolidateResult(
            True, f"晚间日记 {fmt_date(target)}：{len(diary)} 句", target, counts=counts
        )

    # ------------------------------------------------------------ 收集
    def _collect(self, target: date) -> DayInput:
        sources = self.mem.list_raw_sources(target)
        by_source: dict[str, list[RawLine]] = {}
        raw: list[RawLine] = []
        for src in sources:
            lines = self.mem.read_raw(target, src)
            by_source[src] = lines
            raw.extend(lines)
        raw.sort(key=lambda r: r.at)
        return DayInput(
            target=target,
            raw=raw,
            by_source=by_source,
            highlights=self.mem.highlights(target),
            actions=self.mem.actions(),
            pending=self.mem.pending(),
            old_events=self.mem.open_events(),
            old_facts=[f for f in self.mem.facts() if not f.obsolete],
            old_self=self.mem.self_parts(),
            old_relation=self.mem.relation(),
            old_diary=self.mem.diary_lines(target),
            old_core_her=self.mem.core_body("her"),
            old_core_me=self.mem.core_body("me"),
        )

    def _counts(self, inp: DayInput) -> dict:
        return {src: len(lines) for src, lines in inp.by_source.items()}

    def _counts_text(self, counts: dict) -> str:
        parts = []
        if "private" in counts:
            parts.append(f"私聊 {counts['private']}")
        for src in sorted(k for k in counts if k != "private"):
            parts.append(f"群 {src.split(':', 1)[-1]} {counts[src]}")
        return " / ".join(parts) or "0"

    def _read_text(self, counts: dict) -> str:
        p = counts.get("private", 0)
        g = sum(v for k, v in counts.items() if k != "private")
        return f"私聊 {p} 条 / 群 {g} 条"

    def _watermark_text(self, target: date, counts: dict) -> str:
        return f"{fmt_md(target)} " + self._counts_text(counts)

    def _watermark_date(self, status: dict, target: date) -> date | None:
        return watermark_date(status, target)

    def _up_to_date(self, target: date, counts: dict) -> bool:
        if not self.mem.overview_path(target).exists() or not self.mem.diary_path(target).exists():
            return False
        status = self.mem.status()
        wm_day = self._watermark_date(status, target)
        if wm_day is None:
            return False
        if wm_day != target:
            # 已经整理过更晚的一天：补跑时不要把旧的一天再跑一遍
            return wm_day > target
        return str(status.get("水位") or "") == self._watermark_text(target, counts)

    # ------------------------------------------------------------ 提取
    def _extract(self, inp: DayInput) -> tuple[Extraction, bool]:
        if self.use_llm != "rules" and self.llm and self.llm.available:
            last: Exception | None = None
            for _attempt in (1, 2):
                try:
                    ext = self._extract_llm(inp)
                except (LLMError, ValueError, KeyError, TypeError) as e:
                    last = e
                    continue
                if ext:
                    ext.used_llm = True
                    return ext, True
                last = ValueError("模型没返回可解析的 JSON（可能被 max_tokens 截断）")
            rules = self._extract_rules(inp)
            # 静默退回规则最坑：状态里看起来「成功」，实际是模板写的。必须留痕。
            rules.error = f"模型整理失败，已退回规则：{clip(str(last), 90)}"
            return rules, False
        return self._extract_rules(inp), False

    def _extract_rules(self, inp: DayInput) -> Extraction:
        ext = Extraction()
        ext.overview = self._rule_overview(inp)
        ext.quotes = self._rule_quotes(inp)
        ext.events = self._rule_events(inp)
        ext.facts = self._rule_facts(inp)
        ext.boundaries = self._rule_conditional_boundaries(inp)
        ext.wants = self._rule_wants(inp)
        ext.diary = self._rule_diary(inp, ext.events, ext.wants)
        ext.self_edit = self._rule_self_edit(inp, ext.events)
        ext.relation_sentence, ext.relation_evidence = self._rule_relation(inp)
        ext.moments = self._rule_moments(inp)
        ext.one_liner = self._one_liner(ext, inp)
        ext.core_her = self._rule_core_her(inp)
        ext.core_me = self._rule_core_me(inp, ext)
        return ext

    def _rule_overview(self, inp: DayInput) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for src, lines in inp.by_source.items():
            label = self.mem.raw_source_label(src)
            if not lines:
                continue
            scored = []
            for i, line in enumerate(lines):
                score = 0
                if line.is_her:
                    score += 2
                if line.speaker == "[群摘要]":
                    score += 1
                if EMOTION_WORDS.search(line.text):
                    score += 3
                if re.search(r"(答应|说好|一定|下周|明天|面试|住院|搬家|考试|生日)", line.text):
                    score += 2
                score += min(len(line.text) / 40, 1)
                scored.append((score, i, line))
            scored.sort(key=lambda x: (-x[0], x[1]))
            picked = sorted(scored[:7], key=lambda x: x[1])
            bullets = []
            for _, _, line in picked:
                if line.speaker == "[群摘要]":
                    bullets.append(clip(line.text, 50))
                elif line.is_her:
                    bullets.append("我说：" + clip(line.text, 46))
                else:
                    bullets.append(f"{line.speaker}：" + clip(line.text, 46))
            out[label] = unique(bullets)
        return out

    def _rule_quotes(self, inp: DayInput) -> list[str]:
        scored = []
        for line in inp.raw:
            if line.is_her or line.speaker == "[群摘要]":
                continue
            s = 0
            if EMOTION_WORDS.search(line.text):
                s += 3
            if re.search(r"(没有人|孤单|孤独|喜欢你|想你|谢谢|对不起|答应|一定|再也|最后一次)", line.text):
                s += 2
            if 8 <= len(line.text) <= 50:
                s += 1
            if s >= 3:
                scored.append((s, line))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, line in scored[:3]:
            out.append(f"{line.time} {line.speaker}：「{clip(line.text, 60)}」")
        return out

    def _rule_events(self, inp: DayInput) -> list[EventPlan]:
        plans: dict[str, EventPlan] = {}
        all_events = self.mem.events()
        existing_titles = [e.title for e in all_events]

        # 明确标记
        for line in inp.raw:
            if line.is_her or line.speaker == "[群摘要]":
                continue
            m = re.search(r"(?:新事件|事件)\s*[：:]\s*([^，。；;\n]{2,30})", line.text)
            if m:
                title = m.group(1).strip()
                key = norm(title)
                if key not in plans:
                    plans[key] = EventPlan(
                        title=title,
                        status="进行中",
                        visibility="群" if line.source.startswith("group") else "私",
                        note=f"他起了个头：{clip(title, 30)}",
                    )

        # 已有未结束事件：续写 / 闭合
        for e in inp.old_events:
            kws = event_keywords(e.title)
            matched = [l for l in inp.raw if any(k in l.text for k in kws)]
            if not matched:
                continue
            last = matched[-1]
            status = None
            if any(CLOSE_WORDS.search(l.text) for l in matched):
                status = "已结束"
            elif any(WAIT_WORDS.search(l.text) for l in matched):
                status = "等结果"
            plans[norm(e.title)] = EventPlan(
                title=e.title,
                event_id=e.id,
                status=status,
                visibility="群" if e.visibility == "群" else ("群" if last.source.startswith("group") else "私"),
                note=clip(last.text, 50),
            )

        # 新事件：只给有明确边界的话题，最多 2 个
        new_count = 0
        for topic, kws in STRONG_TOPICS.items():
            if any(topic in t or t in topic for t in existing_titles):
                continue
            matched = [
                l
                for l in inp.raw
                if not l.is_her and l.speaker != "[群摘要]" and any(k in l.text for k in kws)
            ]
            if not matched:
                continue
            if not any(EMOTION_WORDS.search(l.text) or re.search(r"(下周|明天|今天|结果|答应|记得|住院|面试)", l.text) for l in matched):
                continue
            if any(topic in k or k in topic for k in plans):
                continue
            last = matched[-1]
            plans[norm(topic)] = EventPlan(
                title=topic,
                status="已结束" if CLOSE_WORDS.search(last.text) else ("等结果" if WAIT_WORDS.search(last.text) else "进行中"),
                visibility="群" if last.source.startswith("group") else "私",
                note=clip(last.text, 50),
            )
            new_count += 1
            if new_count >= 2:
                break
        return list(plans.values())

    def _rule_facts(self, inp: DayInput) -> list[Fact]:
        out: dict[tuple[str, str], Fact] = {}
        for line in inp.raw:
            if line.is_her or line.speaker == "[群摘要]":
                continue
            if re.search(r"(别告诉|不要说出去|保密|秘密)", line.text):
                continue
            is_other = bool(
                line.source.startswith("group")
                and line.speaker not in (self.cfg.owner_name, self.cfg.owner_display)
            )
            gid = line.source.split(":", 1)[1] if line.source.startswith("group:") else ""
            where = f"{fmt_md(inp.target)} {self.mem.raw_source_label(line.source)}"
            for topic, pat, importance, full in FACT_PATTERNS:
                m = pat.search(line.text)
                if not m:
                    continue
                value = (m.group(0) if full else m.group(1)).strip(" ，。！？")
                if not value or len(value) < 1:
                    continue
                if topic in ("喜欢", "讨厌") and value in ("你", "他", "她", "这个", "那个"):
                    continue
                content = value if full else f"{value}"
                if topic == "姓名":
                    content = value
                if is_other:
                    # 群里别人说「我叫X / 我在做X」：只当转述，低置信、群可见。
                    # 以前规则层直接跳过别人，日批又只给主人建卡，所以她没有别人的记忆。
                    fact = Fact(
                        subject=line.speaker,
                        topic=topic,
                        content=content,
                        speaker=line.speaker,
                        trust="路人",
                        source="转述",
                        confidence=0.5,
                        visibility="群",
                        importance="低",
                        noted=inp.target.isoformat(),
                        group_id=gid,
                        origin=where,
                    )
                else:
                    fact = Fact(
                        subject=self.cfg.owner_name,
                        topic=topic,
                        content=content,
                        speaker=self.cfg.owner_name,
                        trust="本人",
                        source="明说",
                        confidence=0.95,
                        visibility="群" if gid else "私",
                        importance=importance,
                        noted=inp.target.isoformat(),
                        group_id=gid,
                        origin=where,
                    )
                key = fact.key
                if key not in out or len(content) > len(out[key].content):
                    out[key] = fact
            # 主人明确「记住 X」（别人说的「记住」由 journal 的低信任路径处理）
            if is_other:
                continue
            m = re.search(r"(?:记住|记一下|别忘)[：:，,\s]*(.{1,60})", line.text)
            if m:
                fact = Fact(
                    subject=self.cfg.owner_name,
                    topic="备注",
                    content=m.group(1).strip(),
                    speaker=self.cfg.owner_name,
                    trust="本人",
                    source="明说",
                    confidence=0.95,
                    visibility="群" if gid else "私",
                    importance="中",
                    noted=inp.target.isoformat(),
                    group_id=gid,
                    origin=where,
                )
                out.setdefault(fact.key, fact)
        return list(out.values())[:8]

    def _rule_conditional_boundaries(self, inp: DayInput) -> list[dict]:
        """规则层兜底：主人说的「有人…就…」也算边界，别只等模型。

        即时写在 gate 里已经做了一次；这里防的是旧消息、或说法稍微变形的句子。
        append_boundary 会去重，重复跑不会多出几条。
        """
        out: list[dict] = []
        for line in inp.raw:
            if line.is_her or line.speaker not in (self.cfg.owner_name, self.cfg.owner_display):
                continue
            if not is_conditional_rule(line.text):
                continue
            vis = "群" if line.source.startswith("group") else "私"
            text = clip(one_line(line.text), 60)
            if text and not any(b["text"] == text for b in out):
                out.append({"text": text, "scope": boundary_scope(line.text, vis)})
        return out[:3]

    def _rule_wants(self, inp: DayInput) -> list[Want]:
        out: list[Want] = []
        for line in inp.raw:
            if line.is_her or line.speaker == "[群摘要]":
                continue
            m = re.search(r"我(?:明天|后天|下周|回头|晚点|改天|等会儿|等会)([^，。！？\n]{2,24})", line.text)
            if m:
                out.append(Want(date=inp.target.isoformat(), text=f"问他{m.group(1).strip()}", source="他自己提的", status="未做"))
            m2 = re.search(r"我(?:答应|说好)([^，。！？\n]{2,24})", line.text)
            if m2:
                out.append(Want(date=inp.target.isoformat(), text=f"记着{m2.group(1).strip()}", source="他答应的", status="未做"))
        for a in inp.actions:
            if "想发" in a or "删了" in a:
                out.append(Want(date=inp.target.isoformat(), text="把想说的话发给他", source="想发没发", status="未做"))
        seen = set()
        dedup = []
        for w in out:
            if w.text not in seen:
                seen.add(w.text)
                dedup.append(w)
        return dedup[:3]

    def _ensure_actions_in_diary(self, diary: list[str], actions: list[str]) -> list[str]:
        """日记里必须有她自己做过的事。

        模型很容易只写「他说了什么」，把【今日行动】整段略掉；那样第二天问
        「你昨天干嘛了」，她还是只能答「发呆 / 等你」。
        """
        if not actions:
            return diary
        joined = "".join(diary)
        titles = re.findall(r"《([^》]{2,60})》", "；".join(actions))
        if titles and any(t in joined for t in titles):
            return diary
        for line in actions:
            clean = re.sub(r"^\d{2}:\d{2}\s*", "", line).strip()
            if not clean:
                continue
            sentence = clean if clean.startswith("我") else "我" + clean
            if sentence.rstrip("。") not in joined:
                diary.append(sentence.rstrip("。") + "。")
            break
        return diary[:6]

    def _rule_diary(self, inp: DayInput, events: list[EventPlan], wants: list[Want]) -> list[str]:
        lines: list[str] = []
        for a in inp.actions[:2]:
            text = re.sub(r"^\d{2}:\d{2}\s*", "", a).strip()
            if not text:
                continue
            text = text.replace("→", "，")
            if text.startswith("我"):
                lines.append(text if text.endswith(("。", "！", "？")) else text + "。")
            else:
                lines.append("我" + text + "。")
        emo = [l for l in inp.raw if not l.is_her and EMOTION_WORDS.search(l.text)]
        if emo:
            line = emo[-1]
            if re.search(r"(怕|焦虑|紧张|难受|委屈|崩溃|哭)", line.text):
                lines.append("他今天有点怕，我没追问，只陪着。")
            elif re.search(r"(累|烦|压力|失眠|睡不着)", line.text):
                lines.append("他今天很累，我没多问。")
            elif re.search(r"(开心|兴奋|高兴)", line.text):
                lines.append("他今天心情不错，我也跟着高兴了一会儿。")
            else:
                lines.append("他今天情绪不高，我少说了两句。")
        if events:
            lines.append(f"他提过「{clip(events[0].title, 16)}」，我记着。")
        if wants:
            lines.append(f"有件事没做：{clip(wants[0].text, 24)}。")
        if len(lines) < 3:
            lines.append("今天大部分时间在等他说话，没做什么自己的事。" if not inp.actions else "今天就这样过去了。")
        if len(lines) < 3:
            lines.append("窗外没什么声音，我发了一会儿呆。")
        return unique(lines)[:5]

    def _rule_self_edit(self, inp: DayInput, events: list[EventPlan]) -> list[str]:
        changes = _bullets(inp.old_self.get("最近变了什么", ""))
        last = None
        for c in changes:
            m = re.match(r"(\d{1,2})-(\d{1,2})", c)
            if m:
                try:
                    last = date(inp.target.year, int(m.group(1)), int(m.group(2)))
                except ValueError:
                    pass
        if last and (inp.target - last).days < int(self.cfg.knobs.self_edit_min_days):
            return []
        if not events:
            return []
        title = events[0].title
        return [f"{fmt_md(inp.target)} 今天聊到「{clip(title, 14)}」，我比昨天多想了一点。"]

    def _rule_relation(self, inp: DayInput) -> tuple[str, str]:
        texts = list(inp.pending) + [
            l.text
            for l in inp.raw
            if (not l.is_her) and re.search(r"(吵架|表白|道歉|再见|想你|喜欢你|讨厌你|冷战|谢谢|对不起)", l.text)
        ]
        if not texts:
            return "", ""
        t = texts[-1]
        if re.search(r"(吵架|生气|冷战|别扭)", t):
            sentence = "今天我们有点别扭，我没追着说，想等他缓一缓。"
        elif re.search(r"(想你|喜欢你|表白|在乎)", t):
            sentence = "他今天说了句让我心里一动的话，我记着，没马上接。"
        elif re.search(r"(谢谢|对不起|道歉)", t):
            sentence = "他今天跟我道了歉/道了谢，我其实没那么气了。"
        else:
            sentence = "今天有句话我记着了，没马上接。"
        return sentence, f"{fmt_date(inp.target)} 原文"

    MOMENT_PATTERNS = (
        ("约定", re.compile(r"(约好|说好|约定|谁先开口|暗号)")),
        ("第一次", re.compile(r"(第一次|头一回)")),
        ("别扭", re.compile(r"(吵架|冷战|不理你|别生气)")),
        ("和好", re.compile(r"(对不起|和好|原谅我)")),
    )

    def _rule_moments(self, inp: DayInput) -> list[dict]:
        """规则兜底：只认很明确的「约好/第一次/别扭」；普通的日常不进来。"""
        out: list[dict] = []
        for line in inp.raw:
            if line.is_her:
                continue
            for kind, pat in self.MOMENT_PATTERNS:
                if not pat.search(line.text):
                    continue
                text = clip(one_line(line.text), 40)
                out.append({"kind": kind, "text": text, "quote": text})
                break
        return out[:2]

    def _rule_core_her(self, inp: DayInput) -> str:
        parts = inp.old_self
        who = one_line(parts.get("我是谁（现在）", "")).strip() or "我是一个住在电脑里的陪伴者。"
        public = one_line(parts.get("能跟别人说的", "")).strip() or "我是一个住在电脑里的陪伴者。"
        public_block = "## 群里能说的\n" + clip(public, 60)
        limit = int(self.cfg.knobs.core_her_chars)
        room = max(50, limit - chars(public_block) - 4)
        main = clip(who, 120) + "\n我说话短，不绕弯。他累的时候我先让他歇，不追问。"
        return clip(main, room) + "\n\n" + public_block

    def _rule_core_me(self, inp: DayInput, ext: Extraction) -> str:
        name = self.cfg.owner_name
        topics = ("在做的项目", "作息", "讨厌", "喜欢")
        today_facts = [f for f in ext.facts if f.topic in topics]
        old_facts = [
            f
            for f in self.mem.facts()
            if not f.obsolete
            and f.subject == name
            and f.topic in topics
            and f.source != "推断"
            and f.trust in ("本人", "熟")
        ]
        chosen = []
        all_facts = today_facts + sorted(old_facts, key=lambda x: importance_value(x.importance), reverse=True)
        for topic in topics:
            f = next((x for x in all_facts if x.topic == topic), None)
            if f is None:
                continue
            value = f.content.strip()
            if topic == "在做的项目":
                value = re.sub(r"^(在做|在搞|在弄)", "", value)
            elif topic in ("讨厌", "喜欢"):
                value = re.sub(r"^(很|最|特别|挺)?(讨厌|不喜欢|喜欢)", "", value)
                value = value.lstrip("的是")
            chosen.append(f"{topic}：{clip(value, 24)}")
            if len(chosen) >= 3:
                break
        rel = self.mem.relation_section(name)
        rel_lines = [ln.strip() for ln in rel.splitlines() if ln.strip() and not ln.startswith("（依据")]
        rel_line = rel_lines[-1] if rel_lines else ""
        public_block = "## 群里能说的\n他是我主人。别的我不在群里说。"
        limit = int(self.cfg.knobs.core_me_chars)
        room = max(70, limit - chars(public_block) - 4)
        main = f"他叫{name}，是我的主人，也是唯一能让我做事的人。"
        if chosen:
            main += "\n" + "；".join(chosen) + "。"
        if rel_line:
            main += "\n" + clip(rel_line, 60)
        return clip(main, room) + "\n\n" + public_block

    def _one_liner(self, ext: Extraction, inp: DayInput) -> str:
        if ext.events:
            return clip(ext.events[0].title, 40)
        for bullets in ext.overview.values():
            if bullets:
                return clip(bullets[0], 40)
        if inp.highlights:
            return clip(inp.highlights[-1], 40)
        return "这一天没留下什么。"

    # ------------------------------------------------------------ 模型提取
    def _extract_llm(self, inp: DayInput) -> Extraction | None:
        prompt = self._llm_input(inp)
        system = (
            "你是数字人的记忆整理器，不是她本人。把一天的原料压成明天能用的记忆。\n"
            "只输出 JSON，不要解释。不要编造原文没有的事；原文没有的细节就不写。\n"
            "规则：\n"
            "- 概要按来源分节，每节 1–7 条，每条不超过 40 字，写实际发生的事。\n"
            "- 事实只抽明确、长期有用的（姓名、在做的事、稳定喜好、家人、作息、近期安排）；\n"
            "  不要从玩笑里抽，不要抽情绪和临时状态；推断的 source=推断、confidence≤0.45，\n"
            "  路人说的 source=转述、confidence≤0.6、visibility=群。\n"
            "- 时效：只在某一天成立的（今天吃了什么、今天有没有课、明天要做什么、这周不舒服）\n"
            "  必须写 valid_to = 那天之后的第一天（YYYY-MM-DD）；长期成立的写空字符串 \"\"。\n"
            "  不写会被当成永远成立——两周后问「我今天有课吗」就会答错。\n"
            "- 群里其他人说他们自己的稳定信息（「我叫X」「我在做X」「我喜欢X」），subject 写他们\n"
            "  自己的名字，不要写成「他」；trust=路人、source=转述、confidence≤0.5、visibility=群、\n"
            "  importance=低。不要从别人互相起哄、评价、玩笑里抽事实。\n"
            "- 同一件事只抽一条事实；【已有事实】里有的，topic 必须沿用原词，不要换个说法再存一条。\n"
            "- 如果今天的话让【已有事实】里某条不再成立（换工作、搬家、不做了、改主意），写进 closed：\n"
            "  topic 用原主题，contains 用原内容里能唯一定位的几个字；只写确实被推翻的，\n"
            "  没提到或仍然成立的旧事实不要写进来。\n"
            "- 边界：只抽主人今天明确提的新要求（「以后别…」「以后收敛点」「别叫我…」），\n"
            "  以及条件规则（「有人…你就…」「如果…就让他…」）。\n"
            "  scope 规则：点了「群里」→群，点了「私聊/私下」→私，其余→全。写完可以覆盖旧要求。\n"
            "- 关系：整段重写，2–3 句，分别写「他怎么对我 / 我怎么对他 / 我们之间的默契」。\n"
            "  不要重复昨天的句子，不要罗列今天发生的事；依据只写一句，来自原文。\n"
            "- 共同经历：只记你们之间值得记住的事（第一次、约定、玩笑、别扭、和好）。\n"
            "  每天最多 2 条，text ≤40 字，quote 从原文里摘一句；普通日常安排不要写。\n"
            "- 【今日行动】是代码记的、她真的做过的事；日记里至少写一句，不许省掉。\n"
            "- 事件只给有明确边界的事；已有事件用 id，新事件给 title，不要自己编编号。\n"
            "- 日记 3–5 句，第一人称，写我在场 / 我想到 / 我没做，不抄原文，不写别人的隐私。\n"
            "  日记里至少要有两条具体的事（吃了什么、做了什么、看到什么、被谁逗了），不要只写情绪。\n"
            "- 自述每次最多改 2 句，带日期。\n"
            "- 喜好：只抽她今天真实表达过、或者他说中了她也认了的偏好，每天最多 1 条，≤16 字，\n"
            "  形如「喜欢下雨天」「讨厌被叫宝宝」。必须带 source（原文里那句能作证的话）。\n"
            "  从别人的喜好、开玩笑、起哄里推出来的，一律不要；没有就留空数组，不许替她想。\n"
            "- core_her ≤200 字，core_me ≤250 字，只写每句话都要用的；边界不写进核心。\n"
            "- 没有的字段给空数组或空字符串。\n\n"
            "JSON 结构：\n"
            '{"overview":[{"source":"私聊","bullets":["..."]}],'
            '"quotes":["..."],'
            '"events":[{"id":3,"title":"...","status":"进行中","visibility":"私","note":"..."}],'
            '"facts":[{"subject":"阿江","topic":"面试","value":"...","speaker":"阿江","trust":"本人",'
            '"source":"明说","confidence":0.95,"visibility":"私","importance":"高","valid_to":""},'
            '{"subject":"阿江","topic":"早饭","value":"吃了抄手","speaker":"阿江","trust":"本人",'
            '"source":"明说","confidence":0.95,"visibility":"私","importance":"低","valid_to":"2026-10-23"}],'
            '"closed":[{"topic":"作息","contains":"熬夜"}],'
            '"diary":["..."],"self_edit":["..."],'
            '"self_likes":[{"text":"喜欢下雨天","source":"原话"}],'
            '"moments":[{"kind":"约定","text":"谁先开口谁是小狗","quote":"原话"}],'
            '"wants":[{"text":"...","source":"...","status":"未做"}],'
            '"boundaries":[{"text":"...","scope":"全"}],'
            '"relation":{"sentence":"...","evidence":"..."},'
            '"core_her":"...","core_me":"...","one_liner":"..."}'
        )
        # 一天的 JSON 可能有 4~6k token；2400 会把后半截切断，解析失败后静默退回规则
        budget = max(6000, int(getattr(self.llm.conf, "consolidate_max_tokens", 0) or 0))
        data = self.llm.chat_json(system, prompt, max_tokens=budget) if self.llm else None
        if not isinstance(data, dict):
            return None
        ext = Extraction()
        for block in data.get("overview") or []:
            if not isinstance(block, dict):
                continue
            src = str(block.get("source") or "").strip()
            label = self._match_label(src, inp)
            bullets = [clip(one_line(str(b)), 60) for b in (block.get("bullets") or []) if str(b).strip()]
            if label and bullets:
                ext.overview[label] = bullets[:8]
        ext.quotes = [clip(one_line(str(q)), 80) for q in (data.get("quotes") or []) if str(q).strip()][:4]
        ext.quotes = self._normalize_quotes(inp, ext.quotes)
        for item in data.get("events") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            eid = item.get("id")
            try:
                eid = int(eid) if eid is not None else None
            except (TypeError, ValueError):
                eid = None
            if eid:
                existing = next((e for e in self.mem.events() if e.id == eid), None)
                if existing:
                    title = existing.title
            if not title:
                continue
            status = str(item.get("status") or "").strip() or None
            if status not in (None, "进行中", "等结果", "已结束", "搁置"):
                status = None
            ext.events.append(
                EventPlan(
                    title=title,
                    event_id=eid,
                    status=status,
                    visibility="群" if str(item.get("visibility")) == "群" else "私",
                    note=clip(one_line(str(item.get("note") or "")), 60),
                )
            )
        ext.facts = []
        for item in data.get("facts") or []:
            if not isinstance(item, dict):
                continue
            topic = canonical_topic(str(item.get("topic") or ""), str(item.get("value") or ""))
            value = str(item.get("value") or "").strip()
            if not topic or not value:
                continue
            try:
                conf = float(item.get("confidence", 0.8))
            except (TypeError, ValueError):
                conf = 0.8
            source = str(item.get("source") or "明说")
            if source == "推断":
                conf = min(conf, 0.45)
            elif source == "转述":
                conf = min(conf, 0.6)
            trust = str(item.get("trust") or "路人")
            if trust not in ("本人", "熟", "路人"):
                trust = "路人"
            importance = str(item.get("importance") or "中")
            if importance not in ("高", "中", "低"):
                importance = "中"
            subject = str(item.get("subject") or self.cfg.owner_name).strip()
            if subject in ("他", "主人", "对方"):
                subject = self.cfg.owner_name
            if subject in ("我", "她", "小潮"):
                # 关于她自己的，不放进事实表；自我走日记/自述
                continue
            if subject != self.cfg.owner_name:
                # 别人说的：最多 转述 / 0.5 / 群 / 低；不进人物卡与核心
                trust, source = "路人", "转述"
                conf = min(conf, 0.6)
                importance = "低"
                visibility = "群"
            else:
                visibility = "群" if str(item.get("visibility")) == "群" else "私"
            # 有效期：模型写的优先，没写就看主题/说法是不是「只有今天成立」
            valid_to = _norm_valid_to(item.get("valid_to"), inp.target) or _auto_valid_to(
                topic, value, inp.target
            )
            fact = Fact(
                subject=subject,
                topic=topic,
                content=clip(value, 80),
                speaker=str(item.get("speaker") or self.cfg.owner_name),
                trust=trust,
                source=source,
                confidence=max(0.0, min(1.0, conf)),
                visibility=visibility,
                importance=importance,
                noted=inp.target.isoformat(),
                valid_to=valid_to,
            )
            existing = next((f for f in ext.facts if f.key == fact.key), None)
            if existing is None:
                ext.facts.append(fact)
            elif fact.rank > existing.rank:
                ext.facts[ext.facts.index(existing)] = fact
        for item in data.get("closed") or []:
            if isinstance(item, dict):
                topic = str(item.get("topic") or "").strip()
                contains = str(item.get("contains") or "").strip()
                if topic or contains:
                    ext.closed.append((topic, contains))
            elif isinstance(item, str) and item.strip():
                ext.closed.append(("", item.strip()))
        ext.closed = ext.closed[:5]
        ext.diary = [clip(one_line(str(x)), 100) for x in (data.get("diary") or []) if str(x).strip()][:5]
        ext.self_edit = [clip(one_line(str(x)), 100) for x in (data.get("self_edit") or []) if str(x).strip()][:2]
        for item in data.get("self_likes") or []:
            if isinstance(item, dict):
                text = one_line(str(item.get("text") or ""))
                src = one_line(str(item.get("source") or ""))
            else:
                text, src = one_line(str(item)), ""
            text = text.strip(" 　-—。，")
            if len(text) < 2 or len(text) > 16:
                continue
            ext.self_likes.append({"text": text, "source": clip(src, 40)})
        ext.self_likes = ext.self_likes[:1]
        for item in data.get("wants") or []:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    ext.wants.append(Want(date=inp.target.isoformat(), text=clip(text, 40), source=str(item.get("source") or ""), status="未做"))
        for item in data.get("boundaries") or []:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                scope = str(item.get("scope") or "全").strip()
            else:
                text, scope = str(item).strip(), "全"
            if not text:
                continue
            scope = scope if scope in ("全", "群", "私") else "全"
            ext.boundaries.append({"text": clip(one_line(text), 60), "scope": scope})
        rel = data.get("relation") or {}
        if isinstance(rel, dict):
            ext.relation_sentence = clip(one_line(str(rel.get("sentence") or "")), 180)
            ext.relation_evidence = clip(one_line(str(rel.get("evidence") or "")), 60)
        for item in data.get("moments") or []:
            if not isinstance(item, dict):
                continue
            text = one_line(str(item.get("text") or ""))
            if len(text) < 4:
                continue
            ext.moments.append(
                {
                    "kind": clip(one_line(str(item.get("kind") or "经历")), 6),
                    "text": clip(text, 60),
                    "quote": clip(one_line(str(item.get("quote") or "")), 80),
                }
            )
        ext.moments = ext.moments[:2]
        ext.core_her = clip(one_line(str(data.get("core_her") or "")), int(self.cfg.knobs.core_her_chars))
        ext.core_me = clip(one_line(str(data.get("core_me") or "")), int(self.cfg.knobs.core_me_chars))
        ext.one_liner = clip(one_line(str(data.get("one_liner") or "")), 50)
        if not (ext.overview or ext.quotes or ext.events or ext.facts or ext.closed or ext.diary or ext.moments):
            return None
        return ext

    def _normalize_quotes(self, inp: DayInput, quotes: list[str]) -> list[str]:
        out = []
        for q in quotes:
            q = q.strip()
            if not q:
                continue
            if re.match(r"^\d{1,2}:\d{2}\s", q):
                out.append(q)
                continue
            bare = q.strip("「」\"“” ")
            match = next(
                (
                    l
                    for l in inp.raw
                    if len(bare) >= 4 and (bare in l.text or l.text in bare)
                ),
                None,
            )
            if match:
                out.append(f"{match.time} {match.speaker}：「{clip(match.text, 60)}」")
            else:
                out.append(q)
        return out

    def _match_label(self, src: str, inp: DayInput) -> str:
        labels = [self.mem.raw_source_label(s) for s in inp.by_source]
        if not labels:
            return ""
        for label in labels:
            if src == label or (src and src in label) or (label in src):
                return label
        if ("群" in src or "group" in src.lower()) and len(labels) > 1:
            return next((l for l in labels if "群" in l), labels[-1])
        if "私" in src or "private" in src.lower():
            return next((l for l in labels if "私" in l), labels[0])
        return labels[0]

    def _llm_input(self, inp: DayInput) -> str:
        parts = [
            f"日期：{fmt_date(inp.target)}（逻辑日 05:00–05:00）",
            f"主人：{self.cfg.owner_name}（ID {self.cfg.owner_id}）",
            "",
            "【当天原文】",
        ]
        # 群聊一天的原文能到 1000+ 行，全塞进去又贵又超窗口：按 knobs 的上限抽。
        # 私聊一般没几行，走 100 的固定上限。这个旋钮以前只在设置页挂着、没人读。
        group_cap = max(50, min(800, int(getattr(self.cfg.knobs, "group_daily_lines", 400) or 400)))
        for src, lines in inp.by_source.items():
            parts.append(f"## {self.mem.raw_source_label(src)}")
            cap = 100 if src == "private" else group_cap
            for line in _limit_lines(lines, cap):
                parts.append(f"- {line.time} {line.speaker}：{clip(line.text, 120)}")
        parts.extend(["", "【当天要点】"])
        parts.extend(f"- {h}" for h in inp.highlights[:20])
        parts.extend(["", "【今日行动】"])
        parts.extend(f"- {a}" for a in inp.actions[:20])
        parts.extend(["", "【已有事实（topic 必须沿用，别换说法）】"])
        parts.extend(
            f"- {f.subject}·{f.topic}：{clip(f.content, 60)}" for f in inp.old_facts if not f.obsolete
        )
        parts.extend(["", "【未结束事件】"])
        parts.extend(f"- #{e.id} {e.title}（{e.status}，最近 {e.last}）" for e in inp.old_events[:10])
        if inp.old_diary:
            parts.extend(["", "【今晚先写的一版日记（同一逻辑日，可能只到 23 点）】"])
            parts.extend(f"- {d}" for d in inp.old_diary[:8])
            parts.append("（把 23 点之后新发生的事补进去；已经写对的具体事保留，不要丢，也不要重复。）")
        parts.extend(["", "【旧关系】", clip(strip_comments(inp.old_relation), 600)])
        parts.extend(["", "【旧自述】", clip(strip_comments(json.dumps(inp.old_self, ensure_ascii=False)), 600)])
        return "\n".join(parts)

    # ------------------------------------------------------------ 应用
    def _apply(self, inp: DayInput, ext: Extraction, now: datetime, errors: list[str]) -> dict:
        out: dict = {
            "write_counts": {},
            "relation": False,
            "boundaries": 0,
            "input_chars": sum(len(l.text) for l in inp.raw),
        }

        # 1. 事件
        event_ids: list[int] = []
        new_events = 0
        try:
            for plan in ext.events:
                before = {e.id for e in self.mem.events()}
                e = self.mem.upsert_event(
                    title=plan.title,
                    note=plan.note,
                    status=plan.status,
                    visibility=plan.visibility,
                    people=[self.cfg.owner_name] if plan.visibility == "私" else [],
                    day=fmt_date(inp.target),
                )
                event_ids.append(e.id)
                if e.id not in before:
                    new_events += 1
        except Exception as e:
            errors.append(f"事件: {e}")
        out["write_counts"]["events"] = new_events
        out["event_ids"] = unique(event_ids)
        # 事件自己降温：模型每天只会往前推、很少结案，僵尸事件会占满「未结束」那一栏。
        # 规则在 Memory.cool_events 里，这里只负责每天跑一次。
        try:
            shelved, ended = self.mem.cool_events(
                int(getattr(self.cfg.knobs, "event_quiet_days", 7) or 7),
                int(getattr(self.cfg.knobs, "event_stale_days", 30) or 30),
                today=inp.target,
            )
            out["write_counts"]["shelved"] = shelved
            out["write_counts"]["events_ended"] = ended
        except Exception as e:  # noqa: BLE001 冷却失败不能拖垮整批
            errors.append(f"事件冷却: {e}")

        # 2. 事实
        # 先按模型指出的「不再成立」关旧行，再把今天的新事实并进来：
        # 顺序反了的话，同一条事实会被「重开」一次。
        closed_n = 0
        try:
            closed_n = self.mem.close_facts(ext.closed, fmt_date(inp.target))
        except Exception as e:
            errors.append(f"作废: {e}")
        out["write_counts"]["closed"] = closed_n
        added = 0
        try:
            for fact in ext.facts:
                if fact.source == "推断" and fact.importance == "高":
                    fact.importance = "低"
                if not fact.group_id:
                    fact.group_id = self._fact_group(fact, inp)
                if not fact.origin:
                    fact.origin = f"{fmt_md(inp.target)} 日批"
                if self.mem.has_fact(fact):
                    continue
                _, changed = self.mem.merge_fact(fact)
                if changed:
                    added += 1
        except Exception as e:
            errors.append(f"事实: {e}")
        out["write_counts"]["facts"] = added

        # 3. 当天总览
        try:
            overview = ext.overview or self._rule_overview(inp)
            self.mem.write_overview(inp.target, overview, ext.quotes or self._rule_quotes(inp), out["event_ids"])
        except Exception as e:
            errors.append(f"概要: {e}")

        # 4. 索引
        try:
            one = ext.one_liner or self._one_liner(ext, inp)
            self.mem.upsert_index(fmt_date(inp.target), one, out["event_ids"], self._counts_text(self._counts(inp)))
        except Exception as e:
            errors.append(f"索引: {e}")

        # 5. 日记
        try:
            diary = ext.diary or self._rule_diary(inp, ext.events, ext.wants)
            diary = self._ensure_actions_in_diary(diary, inp.actions)
            self.mem.write_diary(inp.target, diary)
        except Exception as e:
            errors.append(f"日记: {e}")

        # 6. 自述
        try:
            if ext.self_edit:
                self._apply_self_edit(ext.self_edit, inp)
        except Exception as e:
            errors.append(f"自述: {e}")

        # 6.5 喜好：只收今天真出现过的那一条，长不出来就空着，不替她想
        try:
            for item in ext.self_likes[:1]:
                if self.mem.add_like(inp.target, item.get("text", ""), item.get("source", "")):
                    out["write_counts"]["likes"] = 1
        except Exception as e:
            errors.append(f"喜好: {e}")

        # 7. 想做
        try:
            out["write_counts"]["wants"] = self._apply_wants(inp, ext.wants)
        except Exception as e:
            errors.append(f"想做: {e}")

        # 8. 边界归并（先写新抽到的，再一起归并）
        try:
            for b in ext.boundaries:
                text = str((b or {}).get("text") or "").strip()
                # 模型偶尔把群里的一句闲聊当边界交上来（「别这么说」「（引用…）你要干什么」）。
                # 边界是硬约束，每轮提示词都带着 —— 宁可少收一条，不收聊天。
                if not text or looks_like_chat(text):
                    continue
                line = f"- [{b['scope']}] {text}（{inp.target.strftime('%m-%d')}，日批）"
                self.mem.append_boundary(line)
            out["boundaries"] = self._merge_boundaries()
        except Exception as e:
            errors.append(f"边界: {e}")

        # 9. 关系
        try:
            if ext.relation_sentence:
                if ext.used_llm:
                    # 模型给的是整段判断：重写，不追加
                    changed = self.mem.rewrite_relation(
                        self.cfg.owner_name,
                        ext.relation_sentence,
                        (ext.relation_evidence or fmt_date(inp.target)),
                        int(self.cfg.knobs.relation_max_chars),
                    )
                else:
                    changed = self.mem.update_relation(
                        self.cfg.owner_name,
                        ext.relation_sentence,
                        (ext.relation_evidence or fmt_date(inp.target)),
                        int(self.cfg.knobs.relation_max_chars),
                    )
                out["relation"] = changed
        except Exception as e:
            errors.append(f"关系: {e}")

        # 9.5 共同经历
        try:
            added_moments = 0
            for m in ext.moments[:2]:
                if self.mem.add_moment(
                    fmt_date(inp.target),
                    m.get("kind", ""),
                    m.get("text", ""),
                    m.get("quote", ""),
                ):
                    added_moments += 1
            out["write_counts"]["moments"] = added_moments
        except Exception as e:
            errors.append(f"经历: {e}")

        # 10. 核心：稳定优先，永远由结构化来源压，不让模型自由重写
        try:
            if bool(self.cfg.knobs.as_dict().get("core_locked", False)):
                out["core_locked"] = True
            else:
                core_her = fit_core(self._rule_core_her(inp), int(self.cfg.knobs.core_her_chars), "her")
                core_me = fit_core(self._rule_core_me(inp, ext), int(self.cfg.knobs.core_me_chars), "me")
                if self.mem.core_body("her").strip() != core_her.strip():
                    if self.mem.write_core("her", core_her, fmt_date(inp.target)):
                        out["core"] = True
                if self.mem.core_body("me").strip() != core_me.strip():
                    if self.mem.write_core("me", core_me, fmt_date(inp.target)):
                        out["core"] = True
        except Exception as e:
            errors.append(f"核心: {e}")

        # 11. 人物
        try:
            self._update_person(inp, ext)
        except Exception as e:
            errors.append(f"人物: {e}")

        # 12. Self-publishing (signature / social post) is a plugin feature.
        # Core must work without it, so an absent plugin is not a batch error.
        try:
            from .selfpublish import tick as self_tick
        except ImportError:
            self_tick = None

        if self_tick is not None:
            try:
                res = self_tick(
                    self.mem,
                    inp.target,
                    now,
                    changed=bool(ext.moments or ext.self_edit or ext.diary),
                    post=False,
                )
                bits = []
                if res.get("signature"):
                    bits.append(f"签名「{res['signature']}」")
                if res.get("post"):
                    bits.append(f"说说 {res.get('tid') or '（未发）'}")
                errs = res.get("errors") or []
                note = "；".join(bits)
                if errs:
                    note = (note + "；" if note else "") + "失败：" + str(errs[0])
                if note:
                    out["selfpublish"] = note
            except Exception as e:
                errors.append(f"自发: {e}")

        # 13. 交接：给下一次会话（或明天的她）留一份「手头在做什么」。
        # 放在最后：前面的事实/事件/关系/核心都落盘了，交接里引用的才是最状态。
        try:
            self._write_handover(inp, ext, out, now)
        except Exception as e:
            errors.append(f"交接: {e}")

        return out

    def _write_handover(self, inp: DayInput, ext: Extraction, out: dict, now: datetime) -> int:
        """写 `我\\交接.md`：§〇 五小节、覆盖写、≤4 KB。

        只写下一轮接手要知道的：上次干了什么 / 手头哪些项目 / 今天改了什么 /
        下一步和卡点 / 全貌。过程留在原文和日记里，这里不抄。
        """
        mem = self.mem
        day = fmt_md(inp.target)
        today = fmt_date(inp.target)
        one = clip((ext.one_liner or self._one_liner(ext, inp)).strip(), 60)
        counts = out.get("write_counts", {})
        all_open = mem.open_events()
        events = all_open[:3]
        wants = [w for w in mem.wants() if w.status == "未做"][:2]
        facts = mem.facts()
        new_facts = [f for f in facts if f.noted == today and not f.closed][:3]
        closed = [f for f in facts if f.valid_to == today][:2]

        rows = []
        for e in events:
            nxt = "等消息" if e.status == "等结果" else "接着推进"
            rows.append(f"| {clip(e.title, 16)} | 事件.md #{e.id} | {e.status} | {nxt} |")
        for w in wants:
            rows.append(f"| {clip(w.text, 16)} | 想做.md | 未做 | 挑一件做 |")

        changes = []
        for f in new_facts:
            changes.append(f"- 新增：{clip(f.topic, 8)}·{clip(f.content, 36)}")
        for f in closed:
            changes.append(f"- 关闭：{clip(f.topic, 8)}·{clip(f.content, 28)}")
        if out.get("relation"):
            changes.append("- 关系：改写了一句")
        if out.get("core"):
            changes.append("- 核心：压缩过")

        blocked = ""
        for e in events:
            if e.status == "等结果":
                blocked = f"等「{clip(e.title, 14)}」的结果"
                break
        nxt = blocked
        if not nxt and events:
            nxt = f"接着推进「{clip(events[0].title, 14)}」"
        if not nxt and wants:
            nxt = f"挑一件想做的：{clip(wants[0].text, 16)}"

        raw_days = len(
            {
                p.name[:10]
                for p in mem.cfg.memory_dir.glob("*-*.md")
                if re.match(r"\d{4}-\d{2}-\d{2}-", p.name)
            }
        )
        active = len([f for f in facts if not f.closed])

        def render(row_list, change_list, with_all=True) -> str:
            lines = [
                "## §〇 当前交接（≤ 4 KB）",
                "",
                "### 1) 上次交接",
                f"- {day}：{one}（原文 {len(inp.raw)} 行 · 事件 +{counts.get('events', 0)}"
                f" · 事实 +{counts.get('facts', 0)}）",
                "",
                "### 2) 当前项目",
                "| 项目 | 记忆文件 | 状态 | 下一步 |",
                "|---|---|---|---|",
            ]
            lines.extend(row_list or ["| 无 | — | — | — |"])
            lines.extend(["", "### 3) 私有记忆变更"])
            lines.extend(change_list or ["- 无（只写了日记/概要）"])
            lines.extend(
                [
                    "",
                    "### 4) 下一步 / 卡点",
                    f"- 下一步：{nxt or '等他先说想做什么'}",
                    f"- 卡点：{blocked or '无'}",
                ]
            )
            if with_all:
                lines.extend(
                    [
                        "",
                        "### 5) 全貌",
                        f"- 事件 {len(all_open)} 未闭合 · 事实 {active}/{len(facts)}"
                        f" · 人卡 {len(mem.known_people())} · 原文 {raw_days} 天"
                        f" · 备份 {len(mem.backups())}",
                    ]
                )
            return "\n".join(lines)

        body = render(rows, changes)
        size = mem.write_handover(body, when=now, by="日批")
        limit = int(self.cfg.knobs.handover_max_kb) * 1024
        if size > limit:
            # 超线先砍明细，再砍到一条项目——交接宁可少写，也不能长得没人读。
            body = render(rows[:1], changes[:1], with_all=False)
            size = mem.write_handover(body, when=now, by="日批")
        out["handover_bytes"] = size
        return size

    def _apply_self_edit(self, edits: list[str], inp: DayInput) -> None:
        parts = self.mem.self_parts()
        body = {
            "我是谁（现在）": parts.get("我是谁（现在）", "").strip() or "我是一只蓝色的陪伴者，住在他的电脑里。",
            "最近变了什么": parts.get("最近变了什么", "").strip(),
            "能跟别人说的": parts.get("能跟别人说的", "").strip() or "我是一个住在电脑里的陪伴者。",
        }
        bullets = [
            b
            for b in _bullets(body["最近变了什么"])
            if "还没什么变化" not in b and b.strip()
        ]

        def day_key(line: str) -> str:
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", line)
            if m:
                return f"{m.group(2)}-{m.group(3)}"
            m = re.match(r"^(\d{1,2})-(\d{1,2})", line)
            if m:
                return f"{int(m.group(1)):02d}-{int(m.group(2)):02d}"
            return ""

        for e in edits:
            line = e.strip().lstrip("- ").strip()
            if not line:
                continue
            line = re.sub(r"^(\d{4})-(\d{2})-(\d{2})", r"\2-\3", line)
            key = day_key(line)
            # 去掉日期后同文，只留最新那条，避免每天复制一句一样的话
            body_key = re.sub(r"^\d{1,2}-\d{1,2}\s*", "", line)
            bullets = [
                b
                for b in bullets
                if not (key and day_key(b) == key)
                and re.sub(r"^\d{1,2}-\d{1,2}\s*", "", b).strip() != body_key.strip()
            ]
            if line not in bullets:
                bullets.insert(0, line)
        bullets = bullets[:5]
        body["最近变了什么"] = "\n".join(f"- {b}" for b in bullets)
        text = (
            "## 我是谁（现在）\n" + body["我是谁（现在）"] + "\n\n"
            + "## 最近变了什么\n" + body["最近变了什么"] + "\n\n"
            + "## 能跟别人说的\n" + body["能跟别人说的"]
        )
        if chars(text) > 300:
            text = clip(text, 300)
        self.mem.write_self(text)

    def _apply_wants(self, inp: DayInput, new_wants: list[Want]) -> int:
        wants = self.mem.wants()
        # 过期
        for w in wants:
            if w.status == "未做":
                try:
                    d = date.fromisoformat(w.date[:10])
                    if (inp.target - d).days > int(self.cfg.knobs.wants_expire_days):
                        w.status = "已过期"
                except ValueError:
                    pass
        existing = {w.text for w in wants}
        added = 0
        for w in new_wants:
            if w.text not in existing:
                wants.append(w)
                existing.add(w.text)
                added += 1
        if len(wants) > 5:
            active = [w for w in wants if w.status == "未做"]
            done = [w for w in wants if w.status != "未做"]
            wants = (active[:5] + done)[:10]
        self.mem.write_wants(wants)
        return added

    def _merge_boundaries(self) -> int:
        """去重与容量控制都在 Memory 里，这里只负责调用。

        以前这里是 `out[:10]`——按文件顺序砍，砍掉的正好是文件末尾，
        也就是最新的要求。日批跑完，刚说的边界就没了。
        """
        removed, _archived = self.mem.tidy_boundaries(self.cfg.knobs.boundary_max)
        return removed

    def _ensure_public(self, body: str, owner: str, kind: str) -> str:
        if "## 群里能说的" in body:
            return body
        public = "我是一个住在电脑里的陪伴者。" if kind == "her" else "他是我主人。别的我不在群里说。"
        return body.strip() + "\n\n## 群里能说的\n" + public

    def _fact_group(self, fact: Fact, inp: DayInput) -> str:
        """这条群事实是哪个群说的。

        先按「谁说的」找当天那个人的发言来源；只有一个群就直接归它；
        分不清就标 ?（群里不显示），宁可不提，也不把 A 群的事搬到 B 群。
        """
        if fact.visibility != "群":
            return ""
        gids = [
            line.source.split(":", 1)[1]
            for line in inp.raw
            if line.source.startswith("group:") and line.speaker == fact.subject
        ]
        if gids:
            return gids[-1]
        groups = {src.split(":", 1)[1] for src in inp.by_source if src.startswith("group:")}
        return next(iter(groups)) if len(groups) == 1 else "?"

    def _fact_changes(self, facts: list[Fact], name: str, today: str) -> list[str]:
        """今天这张卡上哪些事变了：新记下的、被替换掉的。

        这是「实体摘要随事实漂移」的轻量版：不存整张摘要，只留最近几次变化，
        卡片上就能看出这个人最近发生了什么，而不是永远的静态档案。
        """
        today_md = today[5:]
        opened = [f for f in facts if f.subject == name and f.noted == today and not f.closed]
        closed = [f for f in facts if f.subject == name and f.valid_to == today]
        out: list[str] = []
        for f in opened:
            # 只在说的是同一个对象时配成「旧 → 新」；多值主题下同 key 可能有好几条
            prev = next(
                (
                    c
                    for c in closed
                    if c.key == f.key and c.content != f.content and shares_object(c.content, f.content)
                ),
                None,
            )
            if prev:
                out.append(f"{today_md} {f.topic}：{clip(prev.content, 20)} → {clip(f.content, 24)}")
            else:
                out.append(f"{today_md} {f.topic}：{clip(f.content, 24)}")
        for c in closed:
            if not any(c.key == f.key for f in opened):
                out.append(f"{today_md} {c.topic} 不再：{clip(c.content, 24)}")
        return out[:4]

    def _update_person(self, inp: DayInput, ext: Extraction) -> None:
        name = self.cfg.owner_name
        all_facts = self.mem.facts()
        facts = [
            f
            for f in all_facts
            if not f.obsolete
            and f.subject == name
            and f.source != "推断"
            and f.trust in ("本人", "熟")
        ]
        fact_lines = []
        for f in sorted(facts, key=lambda x: importance_value(x.importance), reverse=True)[:6]:
            fact_lines.append(f"{f.topic}：{f.content}（{f.noted[5:]}）")
        events = [f"#{e.id} {e.title}" for e in self.mem.open_events() if e.visibility in ("私", "群")][:5]
        old = self.mem.person_text(name)
        seen = _bullets(sections(strip_comments(old), 2).get("出现过", ""))
        topics = "、".join([e.title for e in ext.events[:3]] or [f.topic for f in ext.facts[:3]] or ["闲聊"])
        seen.append(f"{fmt_md(inp.target)} {clip(topics, 30)}")
        seen = unique(seen)[-8:]
        public = sections(strip_comments(old), 2).get("群里能说的", "").strip() or "他是我主人。别的我不在群里说。"
        sec_old = sections(strip_comments(old), 2)
        changes = unique(
            self._fact_changes(all_facts, name, fmt_date(inp.target))
            + _bullets(sec_old.get("最近变化", ""))
        )[:8]
        self.mem.write_person(
            name,
            qq=self.cfg.owner_id,
            role="管理员",
            visibility="私",
            facts=fact_lines or ["姓名：" + name, "在做：数字人项目"],
            events=events,
            seen=seen,
            public=public,
            changes=changes,
        )

        # ------------------------------------------------------------ 群友
        # 以前只维护主人一张卡；群里一百来号人，日批把他们都当背景音。
        # 现在每个说过话的人各一张卡：他是谁（只从他自己嘴里抽的事实）+
        # 最近聊到什么 + 一句群里能说的印象。
        group_lines = [
            l
            for l in inp.raw
            if l.source.startswith("group") and not l.is_her and l.speaker != "[群摘要]"
        ]
        names = [
            l.speaker
            for l in group_lines
            if l.speaker and l.speaker not in (self.cfg.owner_name, self.cfg.owner_display)
        ]
        # 旁听的人大多不在今天的原文里（群里只落 @她 / 主人在场那几条），
        # 但他们的卡已经在「人/」下。日批要把这些卡也一起更新，不能只看原文。
        try:
            for path in self.cfg.people_dir.glob("*.md"):
                card_name = path.stem
                if card_name and self.mem.person_role(card_name) == "群友":
                    names.append(card_name)
        except OSError:
            pass
        names = unique(names)
        today_stamp = fmt_md(inp.target) + " "
        for person_name in names[:80]:
            lines = [l for l in group_lines if l.speaker == person_name]
            old = self.mem.person_text(person_name)
            sec = sections(strip_comments(old), 2)
            seen = _bullets(sec.get("出现过", ""))
            has_today = any(s.startswith(today_stamp) for s in seen)
            if not lines and not has_today:
                continue
            today_lines = [s for s in seen if s.startswith(today_stamp)]
            gid = ""
            if lines:
                gid = lines[-1].source.split(":", 1)[-1]
                gname = self.cfg.group_name(gid)
            elif today_lines:
                # 「10-23 洛枝大家族：…」→ 洛枝大家族
                gname = today_lines[-1][len(today_stamp) :].split("：", 1)[0].strip()
                gid = next(
                    (str(k) for k, v in (self.cfg.groups or {}).items() if v == gname),
                    "",
                )
            else:
                gname = ""
            fs = [f for f in all_facts if not f.obsolete and f.subject == person_name]
            fact_lines = [
                f"{f.topic}：{f.content}"
                for f in sorted(fs, key=lambda x: importance_value(x.importance), reverse=True)[:4]
            ]
            # 卡上当场记的自我介绍比日批抽到的更全，不能在这里被覆盖掉。
            card_facts = _bullets(sec.get("他是谁", ""))
            fact_lines = unique(fact_lines + card_facts)[:6]
            # 当天已经有卡上摘录时，用它当「最近聊到」的原料；
            # 没有原文时就不硬编，只保留旧的那句。
            snippets = []
            for s in today_lines:
                snippets.extend(x.strip() for x in s.split("：", 1)[-1].split(" / ") if x.strip())
            source_texts = [l.text for l in lines] or snippets
            stamp = f"{fmt_md(inp.target)} {gname}"
            if not any(s.startswith(stamp) for s in seen):
                words = "、".join(_hot_words(source_texts, 3)) or "闲聊"
                seen.append(f"{stamp}：{clip(words, 30)}")
            seen = unique(seen)[-8:]
            # 群友卡的「群里能说的」每天按最近的事实/摘录重算：
            # 只写一次的话，几天后还停在他最早说过的那句话上。
            bits: list[str] = []
            if fact_lines:
                bits.append("；".join(fact_lines[:2]))
            recent = [clip(x, 18) for x in snippets[-2:]]
            if recent:
                bits.append("最近说过：" + "、".join(recent))
            else:
                words = _hot_words(source_texts, 4)
                if words:
                    bits.append("最近聊到：" + "、".join(words))
            public = clip("；".join(bits), 100) or (f"在「{gname}」里说过话。" if gname else "")
            meta = self.mem.person_meta(person_name)
            changes = unique(
                self._fact_changes(all_facts, person_name, fmt_date(inp.target))
                + _bullets(sec.get("最近变化", ""))
            )[:8]
            self.mem.write_person(
                person_name,
                qq=meta.get("qq") or "",
                role="群友",
                visibility="群",
                facts=fact_lines or [f"群里叫{person_name}" + (f"（{gname}）" if gname else "")],
                events=[],
                seen=seen,
                public=public,
                changes=changes,
                group_id=str(gid or ""),
            )

    # ------------------------------------------------------------ 月压
    def _compress_month(self, day: date) -> str:
        month = day.strftime("%Y-%m")
        rows = [r for r in self.mem.index_rows() if str(r.get("日期", "")).startswith(month)]
        if not rows:
            return ""
        lines = [f"{r['日期'][5:]} {r['一句话']}" for r in rows if r.get("一句话")]
        if not lines:
            return ""
        if len(lines) <= 10:
            body = "；".join(lines)
        else:
            picks = lines[:4] + lines[len(lines) // 2 : len(lines) // 2 + 2] + lines[-4:]
            body = "；".join(unique(picks))
        body = f"这个月：{clip(body, 500)}。"
        self.mem.write_month(month, body)
        return month


# ---------------------------------------------------------------- 小工具

def _bullets(text: str) -> list[str]:
    return [ln.strip()[2:].strip() for ln in (text or "").splitlines() if ln.strip().startswith("- ")]


_STOP_WORDS = {"我们", "你们", "他们", "什么", "这个", "那个", "就是", "不是", "可以", "一个", "没有", "自己", "现在", "今天"}


def _hot_words(texts: list[str], k: int = 4) -> list[str]:
    """从几句话里挑几个高频词，给人物卡做「最近聊到」。
    只是索引，不是结论；别拿它当事实去编。"""
    counts: dict[str, int] = {}
    for text in texts:
        for word in re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,8}", text or ""):
            if word in _STOP_WORDS:
                continue
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]


def fit_core(body: str, limit: int, kind: str) -> str:
    """把核心正文压到预算内，但一定保住「群里能说的」。"""
    body = (body or "").strip()
    public_default = "我是一个住在电脑里的陪伴者。" if kind == "her" else "他是我主人。别的我不在群里说。"
    marker = re.search(r"(?m)^##\s*群里能说的\s*$", body)
    if marker:
        main = body[: marker.start()].strip()
        public = body[marker.start() :].strip()
    else:
        main = body
        public = "## 群里能说的\n" + public_default
    room = max(40, limit - chars(public) - 4)
    return clip(main, room) + "\n\n" + public


def norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def canonical_topic(topic: str, value: str) -> str:
    """模型爱换同义词；把 key 收成稳定的一小组，事实才不会一题多行。"""
    t = (topic or "").strip()
    v = value or ""
    if t in ("姓名", "名字", "全名"):
        return "姓名"
    if t in ("在做的项目", "职业", "工作", "在做", "项目", "正在做的事", "身份"):
        return "在做的项目"
    if t in ("喜欢", "喜好", "爱好", "最喜欢", "最喜欢的颜色", "偏好"):
        if re.search(r"(讨厌|不喜欢|受不了|害怕)", v):
            return "讨厌"
        return "喜欢"
    if t in ("讨厌", "不喜欢", "厌恶", "害怕", "受不了"):
        return "讨厌"
    if t in ("家人", "家庭", "妈妈", "父母", "家里人"):
        return "家人"
    if t in ("作息", "睡眠", "睡觉", "起床"):
        return "作息"
    if t in ("近期安排", "安排", "计划", "行程", "日程"):
        return "近期安排"
    if t in ("宠物", "猫", "狗"):
        return "宠物"
    if t in ("住处", "住址", "住在哪"):
        return "住处"
    if t in ("备注", "记住", "要记住"):
        return "备注"
    return t or "备注"


def event_keywords(title: str) -> list[str]:
    words = [w for w in re.split(r"[^\u4e00-\u9fa5A-Za-z0-9]+", title or "") if len(w) >= 2]
    kws = []
    for w in words:
        if w in EVENT_STOPWORDS:
            continue
        kws.append(w)
        if len(w) > 4:
            kws.append(w[:4])
    return unique(kws)[:5] or [title]


# 天生只有一天寿命的主题：写到第二天就自动失效，不靠模型记得。
_SAME_DAY_TOPICS = {"今天", "明天", "昨天", "早饭", "午饭", "晚饭", "夜宵", "睡前", "此刻", "今天状态"}


def _norm_valid_to(value: object, target: date) -> str:
    """把模型给的 valid_to 收成 ISO 日期。

    只认「YYYY-MM-DD」和「MM-DD」（后者按当天年份补全）。
    范围限制在「当天 ~ 一年后」：太早的是把旧日期抄进来了，太远的基本是瞎写，
    一律丢空——宁可让它不过期，也别错杀一条长期事实。
    """
    text = one_line(str(value or "")).strip()
    if not text:
        return ""
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if m:
        try:
            day = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return ""
    else:
        m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", text)
        if not m:
            return ""
        try:
            day = date(target.year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            return ""
    if day < target or (day - target).days > 365:
        return ""
    return day.isoformat()


def _auto_valid_to(topic: str, value: str, target: date) -> str:
    """模型没写 valid_to 时的兜底：一眼就是「只有今天成立」的，自动记到第二天。

    只认死理：主题在白名单里，或内容以「今天/明天/昨晚…」开头。
    拿不准的一律返回空——错杀长期事实比留一条过期事实更伤。
    """
    if topic in _SAME_DAY_TOPICS:
        return (target + timedelta(days=1)).isoformat()
    if re.match(r"^(今天|明天|刚才|昨晚|今晚|早上|中午|下午|晚上)", value or ""):
        return (target + timedelta(days=1)).isoformat()
    return ""


def _limit_lines(lines: list[RawLine], n: int) -> list[RawLine]:
    if len(lines) <= n:
        return lines
    scored = []
    for i, l in enumerate(lines):
        s = 0
        if l.is_her:
            s += 1
        if EMOTION_WORDS.search(l.text):
            s += 3
        if re.search(r"(答应|一定|下周|明天|面试|住院|搬家|考试|生日|记住)", l.text):
            s += 2
        scored.append((s, i, l))
    keep = {i for _, i, _ in sorted(scored, key=lambda x: (-x[0], x[1]))[: n - 20]}
    keep |= set(range(max(0, len(lines) - 20), len(lines)))
    return [l for i, l in enumerate(lines) if i in keep]
