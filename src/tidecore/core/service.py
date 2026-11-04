"""对话服务：身份闸门 → 写原文 → 组装 → 说 → 记她的话。

无状态：模型侧不留上下文；线程只是本会话的接缝，放在内存里，重启即清。
"""

from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import clock
from .assemble import Assembly, assemble
from .brain import LLM, ReplyResult, generate_reply, split_bubbles
from .config import Config
from .emotion import Emotion
from .ext import ExtensionRegistry
from .features import has as feature_has
from .guard import GroupGuard
from .hooks import Hooks, Turn
from .identity import (
    ROLE_OWNER,
    SESSION_GROUP,
    SESSION_PRIVATE,
    Gate,
    gate_message,
    identity_payload,
    role_of,
)
from .journal import Journal
from .memory import Memory
from .retrieve import IndexDB
from .stickers import count as sticker_count
from .stickers import pick as sticker_pick
from .stickers import pick_for as sticker_for
from .stickers import proactive_pick
from .util import (
    CHANNEL_LABEL,
    clip,
    fmt_md,
    json_load,
    normalize_question,
    one_line,
    strip_channel_tag,
)

# 通道 → 她嘴里的说法（定义在 util，装配层也要用同一份）


def _raw_id(at: datetime, speaker: str, text: str) -> str:
    """一条原文的身份证。同一行在两次读盘之间必须认出是同一行，否则会重复接进线程。"""
    return f"{at.isoformat(timespec='seconds')}|{speaker}|{text}"


# 私聊线程最多几条。和 knobs.thread_tokens 是两道闸：条数先到就按条数砍。
_THREAD_LINES = 20

# 她说完话多久没回音算「被忽视」。比 4 小时的「被冷落」轻得多，但来得快。
IGNORED_AFTER_MINUTES = 20


@dataclass
class TurnResult:
    reply: str = ""
    parts: list[str] = field(default_factory=list)   # 分成几条发（一条一句，像人聊天）
    role: str = "member"
    refused: bool = False
    ignored: bool = False
    offline: bool = False
    duplicate_hint: str = ""
    boundary: bool = False
    assembly: Assembly | None = None
    error: str = ""
    image: str = ""          # 要发出去的表情包/图片（本机路径）
    notes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class Session:
    key: str
    thread: list[dict] = field(default_factory=list)
    summary: str = ""
    last_replies: list[str] = field(default_factory=list)
    last_user_norm: str = ""
    last_user_at: datetime | None = None
    last_answer: str = ""
    last_message_id: str = ""
    last_transport: tuple[str, str, datetime] | None = None
    loaded: bool = False
    # 已经接进线程的原文行（按「时间|说话人|原文」认）。私聊线程改成只追加以后，
    # 靠它判「这一行是不是新的」，不然每轮重放一遍，前缀缓存就白搭了。
    raw_ids: set[str] = field(default_factory=set)
    # 线程是哪一天的。跨天要重建，否则昨天的尾巴会跟今天的拼在一起。
    thread_day: str = ""
    # 「先不发表情」之后的一段时间：只挡自动附带的图，不挡他明确要的
    sticker_muted_until: datetime | None = None
    last_sticker_ask_at: datetime | None = None


# ---------------------------------------------------------------- 要表情包
# 判断的是「这句话是不是在跟我要一张表情」，不是「句子里有没有表情两个字」。
#   ✔ 发个表情 / 给我发个表情包 / 来个表情 / 表情包呢
#   ✔ 你随便发一个表情给我 / 发一张表情包 / 能不能发个表情
#   ✘ 不用每次回完消息都发一个表情 / 你发的表情真丑 / 这表情什么意思
#   ✘ 发表情包还是不能啊 / 只有我发「发个表情包」这五个字才行（说的是插件）
# 以前只认句首的「发个表情」，数量词也只认一个字，所以「发一张」「随便发一个」
# 全落空；她只能等主人说那五个字。
_STICKER_WORD = re.compile(r"(?:表情包?|颜文字|贴图)")
_STICKER_VERB = re.compile(r"(?:发|来|给|整|丢|甩|要|拿|上|送)")
_STICKER_ASK2 = re.compile(r"^(?:还有|有|有没有|没有)?\s*(?:表情包?|表情)(?:呢|吗|有没有|来一张|发一个|发几张|发点)?\s*$")
# 他自己在发/存，不是在跟我要（必须从句子开头起，否则「给我发个」也会中）
_STICKER_SELF = re.compile(r"^(?:我给你|我发你|我发个|我发张|我发一?[张个]|我存|我这有|我这儿有|我收藏)")
# 在聊插件/功能，不是在要图
_STICKER_META = re.compile(
    r"(插件|五个字|5个字|才能|才会|发不了|发不出|发不出去|怎么发|为什么发|还是不能|不能啊)"
)
# 「能不能发个表情」是请求，不是叫停；别把它误判成 veto
_MUTE_QUESTION = re.compile(r"(能不能|能否|可不可以|行不行|好不好|会不会)")
_STICKER_NO = re.compile(r"(不用|不要|不需要|别再|别发|少发|没让你|说过|不想|先不|不许|不准)")
_STICKER_ABOUT = re.compile(r"的\s*表情|表情.{0,3}(真丑|好看|可爱|什么|为什么|怎么|是啥|什么意思)")


def _hard_no(text: str) -> bool:
    """明确叫停：不/别/先不 + 发 + 表情。问句「能不能发」不算。"""
    return not _MUTE_QUESTION.search(text) and bool(_STICKER_MUTE.search(text))


def wants_sticker(text: str) -> bool:
    """他这句话是不是在跟我要一张表情包。"""
    t = one_line(text or "").strip()
    if not t or len(t) > 60:
        return False
    if _hard_no(t) or _STICKER_NO.search(t) or _STICKER_ABOUT.search(t):
        return False
    if _STICKER_SELF.search(t) or _STICKER_META.search(t):
        return False
    if not _STICKER_WORD.search(t):
        return False
    return bool(_STICKER_VERB.search(t) or _STICKER_ASK2.match(t))


# 「先不发表情」「别发图了」：短时间内别再自动附表情。
# 他只说「别发」，没有提表情，不算——那种要靠上下文，宁可不猜。
_STICKER_MUTE = re.compile(
    r"(?:不用|不要|不需要|不许|不准|别再|别|先不|不|少|没让你)"
    r"[^，。！？\n]{0,8}"
    r"(?:发|来|给|整|丢|甩|要)"
    r"[^，。！？\n]{0,4}"
    r"(?:表情|图)"
)
# 她嘴上说「发你了」，但这条路径没带图。只在没被叫停时用它补图。
_CLAIM = re.compile(r"(发你了|给你了|发过去了|发给你了|给你一张|喏[，,]\s*给你)")


def sticker_muted_by(text: str) -> bool:
    """他这句话是不是在叫停表情。"""
    t = one_line(text or "").strip()
    if not t:
        return False
    return _hard_no(t)


def strip_claim(reply: str) -> str:
    """把「喏，给你一张。」这类没兑现的话删掉。"""
    out = _CLAIM.sub("", reply or "")
    out = re.sub(r"[，。；、\s]{2,}", "，", out).strip(" ，。；、")
    return out


class Service:
    def __init__(
        self,
        cfg: Config | None = None,
        offline: bool = False,
        hooks: Hooks | None = None,
    ):
        self.cfg = cfg or Config.load()
        self.offline = offline
        self.hooks = hooks or Hooks()
        self.mem = Memory(self.cfg)
        self.mem.ensure_skeleton()
        self.journal = Journal(self.mem)
        self.sessions: dict[str, Session] = {}
        self.llm = LLM(self.cfg)
        self.guard = GroupGuard(self.cfg)
        # 情绪 = 内分泌。她只有一套血：私聊群里共用一份状态（data/emotion.json）。
        # 这里是**唯一写者**；别的模块要看心情走 emotion.mood_line()。
        self.emotion = Emotion(self.cfg)
        self._last_user_at: datetime | None = None   # 主人上次私聊开口
        self._last_her_at: datetime | None = None    # 她上次私聊开口
        self._missed_for: datetime | None = None     # 上一次「被冷落」记在谁头上
        self._unanswered_for: datetime | None = None  # 哪一次开口已经记过「被忽视」
        self._group_emotion_at: datetime | None = None  # 群里上一次影响情绪的时刻
        # 第三方功能挂在这里（extensions/）；基础包里默认为空。
        self.ext = ExtensionRegistry(self.cfg)

    def _action_defs(self) -> dict[str, str]:
        """扩展声明的 <cmd> 动作。

        draw/sing 如果内置功能装了，由内置实现占着，不让扩展覆盖；
        没装时允许扩展接管这两个类型（基础包接画图/唱歌就走这条）。
        """
        try:
            defs = self.ext.action_defs()
        except Exception:  # noqa: BLE001 扩展坏了不能拖垮聊天
            return {}
        if feature_has("draw"):
            defs.pop("draw", None)
        if feature_has("song"):
            defs.pop("sing", None)
        return defs

    # ------------------------------------------------------------ 会话
    def _session(self, session_type: str, group_id: str | None) -> Session:
        key = f"{session_type}:{group_id or 'private'}"
        if key not in self.sessions:
            self.sessions[key] = Session(key=key)
        return self.sessions[key]

    def _seed_thread(self, sess: Session, session_type: str, group_id: str | None, day, now: datetime) -> None:
        if sess.loaded:
            return
        sess.loaded = True
        entries: dict[tuple[str, str, str], tuple[datetime, str, str, bool, str]] = {}
        limit = 20 if session_type == SESSION_PRIVATE else 12
        per_channel = 8 if session_type == SESSION_PRIVATE else 0

        for line in self.mem.recent_raw_lines(
            session_type, group_id, day, limit=limit, per_channel=per_channel
        ):
            key = (line.at.isoformat(timespec="seconds"), line.speaker, line.text)
            entries[key] = (line.at, line.speaker, line.text, line.is_her, line.channel)

        # 群里未落盘的旁听：被 @ 时也要能接上刚才的话
        if session_type == SESSION_GROUP and group_id:
            state = json_load(self.cfg.hot_dir / f"group-{group_id}.json", {}) or {}
            for item in state.get("recent") or []:
                if item.get("written"):
                    continue
                try:
                    at = datetime.fromisoformat(str(item.get("at")))
                except (TypeError, ValueError):
                    continue
                label = str(item.get("label") or "群友")
                text = str(item.get("text") or "")
                if text:
                    entries[(at.isoformat(timespec="seconds"), label, text)] = (at, label, text, False, "")

        for at, speaker, text, is_her, channel in sorted(entries.values(), key=lambda x: x[0])[-limit:]:
            role = "assistant" if is_her else "user"
            clean = strip_channel_tag(text)
            if session_type == SESSION_PRIVATE:
                tag = CHANNEL_LABEL.get(channel, "") if channel else ""
                content = f"[{tag}] {clean}" if tag else clean
            else:
                content = f"{speaker}：{clean}"
            sess.thread.append({"role": role, "content": content})
            sess.raw_ids.add(_raw_id(at, speaker, text))
        sess.thread_day = day.isoformat() if hasattr(day, "isoformat") else str(day)
        self._trim_thread(sess)

    def _fold_summary(self, sess: Session, popped: list[dict]) -> None:
        """被挤出去的几条压成一句「本会话前情」，别让线索彻底消失。"""
        if not popped:
            return
        bits = []
        for m in popped:
            who = "我" if m.get("role") == "assistant" else "他"
            bits.append(f"{who}：{clip(m.get('content', ''), 30)}")
        sess.summary = clip((sess.summary + "；" if sess.summary else "") + "；".join(bits), 300)

    def _trim_thread(self, sess: Session, budget: int | None = None) -> None:
        budget = budget or int(self.cfg.knobs.thread_tokens)
        def tokens() -> int:
            return int(sum(len(m.get("content", "")) for m in sess.thread) / 1.4)

        guard = 0
        while tokens() > budget and len(sess.thread) > 8 and guard < 20:
            pop_n = min(8, len(sess.thread) - 6)
            popped = sess.thread[:pop_n]
            sess.thread = sess.thread[pop_n:]
            self._fold_summary(sess, popped)
            guard += 1
        # 她最后 2 句永远保留
        hers = [i for i, m in enumerate(sess.thread) if m.get("role") == "assistant"]
        if len(hers) < 2:
            # 从 summary 里补不回原句，至少保证最近两条 assistant 不被裁
            pass

    def _duplicate(
        self,
        sess: Session,
        text: str,
        sender_id: str,
        session_type: str,
        msg_id: str | None,
        now: datetime,
    ) -> tuple[bool, str]:
        """返回 (是否传输重复, 近重复提示)。"""
        norm = normalize_question(text)
        if msg_id and sess.last_message_id and msg_id == sess.last_message_id:
            return True, ""
        if sess.last_transport:
            # 去重必须带发送者：群里两个人 2 秒内发同一句（「哈哈哈哈」「[图片]」）
            # 时，以前第二个人的话会被当成「传输重复」整条丢掉，既不回也不进记忆。
            last_sender, last_norm, last_at = sess.last_transport
            if last_sender == str(sender_id or "") and last_norm == norm and (now - last_at) <= timedelta(seconds=2):
                return True, ""
        sess.last_message_id = msg_id or ""

        window = int(self.cfg.knobs.dup_private_minutes if session_type == SESSION_PRIVATE else self.cfg.knobs.dup_group_minutes)
        probe = norm in ("在吗", "在么", "人呢", "睡了", "在不在", "？", "？？", "？？？", "你好") or len(norm) < 4
        hint = ""
        if (
            not probe
            and sess.last_user_norm
            and sess.last_user_norm == norm
            and sess.last_user_at
            and (now - sess.last_user_at) <= timedelta(minutes=window)
        ):
            mins = max(1, int((now - sess.last_user_at).total_seconds() // 60))
            hint = (
                f"【他刚问过】他 {mins} 分钟前问过同一句，你当时答的是：「{clip(sess.last_answer, 80)}」。"
                "别原样再答一遍——换个角度说，或者问他是不是没刷到。"
            )
        sess.last_user_norm = norm
        sess.last_user_at = now
        sess.last_transport = (str(sender_id or ""), norm, now)
        return False, hint

    def _refresh_private_thread(self, sess: Session, day, now: datetime | None = None) -> None:
        """同一份记忆可能被两个窗口写：CLI 和 QQ 桥各是一个进程。

        每次私聊前扫一遍当天原文，把另一个窗口刚说的话带进线程。
        长期记忆仍由每日日批合并（默认 08:00，`knobs.consolidate_hour` 可改）；这里解决的是「刚才那句」。

        只往后接，不整段重放：以前每轮把最近 20 条重抄一遍，新的一条进来最老的一条
        被挤掉，第一行就变了 —— DeepSeek 的前缀缓存从第 0 个 token 逐字比，整段聊天
        记录每轮都作废。现在只接「还没见过的行」，前面一个字不动；
        攒到上限再按 8 条一批砍（一次砍整批，只断那一轮）。
        """
        if not sess.loaded:
            return
        today = day.isoformat() if hasattr(day, "isoformat") else str(day)
        if sess.thread_day and sess.thread_day != today:
            # 跨天了：线程重起。但「昨晚聊到哪儿」不能一句不留 ——
            # 那正是「今天接着说昨天」最容易断的地方。落一行进今天的要点，
            # 装配时【今天】块里第一眼就能看到（要点每轮私聊都会带）。
            note = clip(one_line(sess.summary or ""), 80).strip("；，。 ")
            if note:
                try:
                    if not any("昨晚聊到" in x for x in self.mem.highlights(day)):
                        self.mem.append_highlight(day, f"昨晚聊到：{note}", now or datetime.now())
                except Exception as e:  # noqa: BLE001 要点写不进去不能挡回话
                    print(f"[thread] 跨天要点没写进去：{type(e).__name__}: {e}", flush=True)
            sess.thread = []
            sess.raw_ids = set()
            sess.summary = ""
            sess.thread_day = today
        lines = self.mem.recent_raw_lines(
            SESSION_PRIVATE, None, day, limit=20, per_channel=8
        )
        if not lines:
            return
        fresh = [(line, _raw_id(line.at, line.speaker, line.text)) for line in lines]
        fresh = [(line, rid) for line, rid in fresh if rid not in sess.raw_ids]
        if not fresh:
            return
        for line, rid in fresh:
            tag = CHANNEL_LABEL.get(line.channel, line.channel) if line.channel else ""
            clean = strip_channel_tag(line.text)
            content = f"[{tag}] {clean}" if tag else clean
            sess.thread.append(
                {
                    "role": "assistant" if line.is_her else "user",
                    "content": content,
                }
            )
            sess.raw_ids.add(rid)
        sess.thread_day = today
        # 条数上限还是 20：一次砍 8 条（不是一条一条挪），中间那些轮次才能连续命中
        while len(sess.thread) > _THREAD_LINES:
            drop = min(8, len(sess.thread) - 6)
            popped = sess.thread[:drop]
            sess.thread = sess.thread[drop:]
            self._fold_summary(sess, popped)
        self._trim_thread(sess)

    # ------------------------------------------------------------ 主入口
    def turn(
        self,
        sender_id: str,
        text: str,
        session_type: str = SESSION_PRIVATE,
        group_id: str | None = None,
        nickname: str = "",
        msg_id: str | None = None,
        now: datetime | None = None,
        offline: bool | None = None,
        channel: str = "",
        images: list[str] | None = None,
        vision_done: bool = False,
    ) -> TurnResult:
        """Run one turn with plugin hooks around :meth:`handle`.

        Adapters should call this instead of ``handle`` so plugins see the turn.
        """
        at = now or datetime.now()
        turn = Turn(
            sender_id=sender_id,
            text=text,
            session_type=session_type,
            group_id=group_id,
            channel=channel,
            at=at,
            nickname=nickname,
            images=list(images or []),
        )
        self.hooks.run_before(turn)
        result = self.handle(
            turn.sender_id,
            turn.text,
            session_type=session_type,
            group_id=group_id,
            nickname=nickname,
            msg_id=msg_id,
            now=at,
            offline=offline,
            channel=channel,
            images=images,
            vision_done=vision_done,
        )
        self.hooks.run_after(turn, result)
        return result

    def handle(
        self,
        sender_id: str,
        text: str,
        session_type: str = SESSION_PRIVATE,
        group_id: str | None = None,
        nickname: str = "",
        msg_id: str | None = None,
        now: datetime | None = None,
        offline: bool | None = None,
        channel: str = "",
        images: list[str] | None = None,
        vision_done: bool = False,
    ) -> TurnResult:
        now = now or datetime.now()
        offline = self.offline if offline is None else offline
        text = one_line(text or "")[:1200]
        # 陌生私聊在情绪/点歌/画画这些早退分支之前就拦掉：否则陌生人说一句
        # 「画只猫」仍会拿到一句拒绝话术，准入开关等于只挡住了普通聊天。
        if (session_type == SESSION_PRIVATE
                and not (self.cfg.owner_id and self.cfg.is_owner(sender_id))
                and not self.cfg.is_admin(sender_id)
                and not self.cfg.allow_stranger_private):
            gate0 = gate_message(sender_id, text, session_type, self.cfg, now,
                                 group_id=group_id, nickname=nickname)
            return TurnResult(
                reply="",
                role=gate0.role,
                refused=True,
                error="不在准入名单",
                meta={"gate": gate0},
            )
        # 他说的话先过一遍情绪：夸 / 凶 / 哄各分泌一次；隔了很久才开口，光是回来就算甜。
        # 这里不碰记忆、不写原文 —— 情绪是状态，不是事实（见 emotion.py 顶部）。
        # 陌生人的私聊她根本不看（下面会被闸门拒掉），所以也不该被他们的语气影响。
        if text and not images and (self.cfg.is_owner(sender_id) or session_type == SESSION_GROUP):
            gap = 0.0
            if self._last_user_at and now > self._last_user_at:
                gap = (now - self._last_user_at).total_seconds() / 3600.0
            try:
                info = self.emotion.on_message(
                    text, now,
                    "group" if session_type == SESSION_GROUP else "private",
                    gap_hours=gap,
                )
            except Exception as e:  # noqa: BLE001 情绪坏了不能影响回话
                print(f"[emotion] 记情绪失败：{type(e).__name__}: {e}", flush=True)
                info = {}
            # 群里：这个人让她舒服还是难受，直接记进对他的好感（私聊不算群关系）。
            if session_type == SESSION_GROUP and not self.cfg.is_owner(sender_id):
                self._learn_affinity(sender_id, nickname, info.get("delta") or 0.0, now)
            if session_type == SESSION_PRIVATE and self.cfg.is_owner(sender_id):
                self._last_user_at = now
        # 她正在放歌时他一开口，先把歌停掉——不然两个人声音叠在一起，
        # 而且他说话多半就是要打断她。
        if feature_has("song"):
            try:
                from .song import is_playing, stop_playing

                if text and is_playing():
                    stop_playing()
            except Exception as e:  # noqa: BLE001 停歌失败不能影响回话
                print(f"[song] 停歌检查失败：{type(e).__name__}: {e}", flush=True)
        # 点歌/学歌/画画的判断交给 LLM（assemble 里那条【隐藏命令】）：
        # 她先照常回人话，系统截出命令再后台执行。
        # 这里只留「学歌进度」这种纯查询，不值得让模型绕一圈。
        if not images and self.cfg.is_owner(sender_id) and feature_has("song"):
            try:
                from .song import is_progress_query, progress_text

                if is_progress_query(text):
                    msg = progress_text(self.cfg)
                    return TurnResult(reply=msg, parts=[msg], meta={"song_progress": True})
            except Exception as e:  # noqa: BLE001 查询失败也不能影响聊天
                print(f"[song] 进度查询失败：{type(e).__name__}: {e}", flush=True)
        # 学歌只认主人：别人开口要学歌，直接拒，别让模型顺口答应
        if not images and not self.cfg.is_owner(sender_id) and feature_has("song"):
            try:
                from .song import looks_like_song_request, parse_request

                req = parse_request(text)
                wants_learn = bool(req and req.get("mode") == "practice")
                if not wants_learn and req is None and re.search(r"[学练]", text):
                    wants_learn = bool(looks_like_song_request(text))
                if wants_learn:
                    msg = "学歌得主人点头，我不能听别人的。"
                    try:
                        from .identity import log_security

                        log_security(self.cfg, now, sender_id, session_type, group_id, text,
                                     "非主人要求学歌")
                    except Exception:  # noqa: BLE001 审计写不进去也要拒
                        pass
                    return TurnResult(reply=msg, parts=[msg], meta={"song_refused": True})
            except Exception as e:  # noqa: BLE001 判断失败不能影响聊天
                print(f"[song] 拒绝判断失败：{type(e).__name__}: {e}", flush=True)
        # 画画也只认主人：别人开口要画，直接拒，别让模型顺口答应
        if not images and not self.cfg.is_owner(sender_id) and feature_has("draw"):
            try:
                from .draw import draw_request

                if draw_request(text):
                    msg = "画画得主人点头，我不能听别人的动笔。"
                    try:
                        from .identity import log_security

                        log_security(self.cfg, now, sender_id, session_type, group_id, text,
                                     "非主人要求画画")
                    except Exception:  # noqa: BLE001 审计写不进去也要拒
                        pass
                    return TurnResult(reply=msg, parts=[msg], meta={"draw_refused": True})
            except Exception as e:  # noqa: BLE001 判断失败不能影响聊天
                print(f"[draw] 拒绝判断失败：{type(e).__name__}: {e}", flush=True)
        # 面对面里他看着镜头问「我手上拿的是什么」：环境描述只有一句「有人，坐着看镜头」，
        # 答不了细节。所以按需把眼前这一帧拿去问一次 —— 问句就是他的原话。
        # 只在主人 + 面对面 + 问题像在问「看见」的时候，一轮多 300ms 左右。
        if not images and channel == "ball" and self.cfg.is_owner(sender_id):
            try:
                from . import eye as eye_mod

                if eye_mod.wants_look(self.cfg, text):
                    asked = text
                    frame = eye_mod.latest_fresh(self.cfg, max_age=5.0)
                    if frame:
                        detail = eye_mod.describe_question(self.cfg, frame, asked)
                        eye_mod.log_look(self.cfg, asked, detail or "（空）")
                        if detail:
                            # 「以这个为准」：她刚换了手里的东西，线程里还留着「杯子」，
                            # 不这么写她会顺着刚才的话说「还是杯子」。
                            text = f"{asked}（摄像头里现在：{detail}，以这个为准）"
                    else:
                        eye_mod.log_look(self.cfg, asked, "（没有 5 秒内的帧，没细看）")
            except ImportError:
                pass  # optional module (eye) is not installed
            except Exception as e:  # noqa: BLE001 眼睛坏了不能影响回话
                print(f"[eye] 细看失败：{type(e).__name__}: {e}", flush=True)
        # 主人发来图片：先识图，把「图里是什么」并进这句话。
        # 这样原文、线程、记忆里都有图的内容，她以后不会说「图我没收到」。
        # vision_done：QQ 通道已经转述过图里的内容，别再花一次视觉调用，
        # 但图本身还要收下来存进表情包库 —— 以前这里直接跳过，库永远是空的。
        if images and self.cfg.is_owner(sender_id):
            caps: list[str] = []
            seen = ""
            if vision_done:
                # 插件把描述拼进 text 了：取出来存图片记忆，但别再拼一遍
                m = re.search(r"（他发来图片，图里是：(.+?)）", text)
                if m:
                    caps = [m.group(1).strip()]
                    seen = caps[0]
            else:
                try:
                    from .vision import describe

                    caps = [describe(self.cfg, one) for one in images[:2]]
                except Exception:  # noqa: BLE001 识图失败不能让她连话都不会说
                    caps = []
            seen = "；".join(c for c in caps if c)
            if seen and not vision_done:
                text = (
                    f"{text}（他发来图片，图里是：{seen}）"
                    if text
                    else f"（他只发来一张图，图里是：{seen}）"
                )
            elif not text:
                text = "（他发来一张图，但图没下来）"
            if images and not seen:
                try:
                    log_path = self.cfg.root / "data" / "vision.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(f"{now:%Y-%m-%d %H:%M:%S} 图片没有描述：{len(images)} 张，来源 qq\n")
                except OSError:
                    pass
            # 存表情包 + 图片记忆：以后她说「发你了」有图可发，也能想起这张图
            try:
                from .images import store as store_image
                from .stickers import save as save_sticker
                from .vision import load_bytes

                for i, one in enumerate(images[:2]):
                    data = load_bytes(one)
                    if not data:
                        continue
                    cap = caps[i] if i < len(caps) else ""
                    save_sticker(self.cfg, data, caption=cap)
                    try:
                        store_image(self.cfg, data, cap, when=now, sender="他", source="qq")
                    except Exception:  # noqa: BLE001 图片记忆失败不影响回话
                        pass
            except Exception:  # noqa: BLE001
                pass
        if not text:
            return TurnResult(ignored=True)
        if session_type not in (SESSION_PRIVATE, SESSION_GROUP):
            session_type = SESSION_PRIVATE
        if session_type == SESSION_GROUP and not group_id:
            group_id = "0"

        role = role_of(sender_id, self.cfg)
        gate = gate_message(sender_id, text, session_type, self.cfg, now, group_id=group_id, nickname=nickname)
        if not gate.allowed:
            return TurnResult(
                reply="",
                role=role,
                refused=True,
                error="不在准入名单",
                meta={"gate": gate},
            )

        sess = self._session(session_type, group_id)
        dup, hint = self._duplicate(sess, text, sender_id, session_type, msg_id, now)
        if dup:
            return TurnResult(ignored=True, role=role, meta={"reason": "传输重复"})

        # 少说话模式：群里被 @ 也不一定回（概率在设置里调，默认 1.0 = 必回）。
        # 还要再乘一个「好感系数」：不熟/被她讨厌的人，@ 了也可能不理。
        # 不回的时候照样记进记忆——她是"在听但不接话"，不是失忆。
        if session_type == SESSION_GROUP and gate.role != ROLE_OWNER:
            try:
                chance = float(self.cfg.knobs.group_mention_reply_chance)
            except Exception:  # noqa: BLE001
                chance = 1.0
            chance = max(0.0, min(1.0, chance * self._reply_factor(sender_id, now)))
            if chance < 1.0 and random.random() > chance:
                try:
                    self.observe(sender_id, text, session_type=session_type,
                                 group_id=group_id, nickname=nickname, now=now)
                except Exception as e:  # noqa: BLE001 记不下来也别回
                    print(f"[quiet] 旁听记录失败：{type(e).__name__}: {e}", flush=True)
                self.journal.record_action(f"群 {group_id or ''} 有人 @ 我，这次没接话", now)
                return TurnResult(ignored=True, role=role, meta={"reason": "少说话模式"})

        # 群聊熔断：她和另一个 AI 互相 @ 时，别无限烧 token。
        # 只拦群里非主人的消息；主人永远放行，/熔断 解除 可手动解。
        if session_type == SESSION_GROUP and gate.role != ROLE_OWNER:
            decision = self.guard.check(group_id or "0", sender_id, is_owner=False, now=now)
            if not decision.allowed:
                if decision.tripped:
                    self.journal.record_action(
                        f"群 {group_id or ''} 熔断：{decision.reason}", now)
                return TurnResult(
                    ignored=True,
                    role=role,
                    meta={"breaker": decision.reason, "retry_after": decision.retry_after},
                )

        # 「先不发表情」不是一句普通的话，是一段时间的开关：
        # 他明确要图时立刻解除；否则 20 分钟内不自动附、也不许她说「发你了」。
        if gate.role == ROLE_OWNER:
            if wants_sticker(text):
                sess.sticker_muted_until = None
                sess.last_sticker_ask_at = now
            elif sticker_muted_by(text):
                sess.sticker_muted_until = now + timedelta(minutes=20)
        muted = bool(sess.sticker_muted_until and now < sess.sticker_muted_until)

        day = clock.logical_date(now)
        self._seed_thread(sess, session_type, group_id, day, now)
        if session_type == SESSION_PRIVATE:
            self._refresh_private_thread(sess, day, now)
        sender_name = self.cfg.owner_name if gate.role == ROLE_OWNER else (nickname or sender_id)
        last_at = self.mem.last_message_time(session_type, group_id, day)
        gap = 999.0 if not last_at else max(0.0, (now - last_at).total_seconds() / 3600)
        # 当前这条还没写；如果今天没有更早的消息，gap 用会话上一次说话时间兜底
        if not last_at and sess.last_user_at:
            gap = 999.0

        # 主人命令：删记忆 / 改身份 / 边界。边界由 journal 即时写。
        command_note = ""
        if gate.role == ROLE_OWNER and gate.command == "delete":
            command_note = self._owner_delete(text)
        elif gate.role == ROLE_OWNER and gate.command == "identity":
            command_note = self._owner_identity(text, now)

        info = self.journal.record_user(
            gate,
            sender_id,
            sender_name,
            text,
            session_type,
            group_id,
            now,
            channel=channel,
        )

        # 当天纠正：主人说「不是 X，是 Y」立刻改事实，不等 05:00。
        # 只认主人私聊：群里改会改到群可见的事实，影响面不对。
        correction = ""
        if gate.role == ROLE_OWNER and session_type == SESSION_PRIVATE:
            try:
                from .corrections import apply_correction

                correction = apply_correction(self.mem, text, now)
            except Exception:  # noqa: BLE001 纠正失败不能拖垮这一轮
                correction = ""

        # 他人越权：拒绝不能静默；仍把这句话当内容写进原文
        if gate.must_refuse and gate.reply_hint:
            reply = gate.reply_hint
            self._record_and_remember(sess, session_type, group_id, reply, now, day, text, gate, channel=channel)
            return TurnResult(
                reply=reply,
                role=gate.role,
                refused=True,
                duplicate_hint=hint,
                assembly=None,
                meta={"gate": gate, "write": info},
            )

        # 扩展在调模型之前截这句话（主人命令、进度查询、外部设备控制…）。
        # 扩展自己决定认不认主人；不返回 dict 就继续正常聊天。
        if self.ext.loaded:
            try:
                hit = self.ext.intercept({
                    "cfg": self.cfg,
                    "service": self,
                    "text": text,
                    "sender_id": sender_id,
                    "session_type": session_type,
                    "group_id": group_id,
                    "nickname": nickname,
                    "channel": channel,
                    "now": now,
                    "reply": "",
                    "image": "",
                    "meta": {},
                })
            except Exception as e:  # noqa: BLE001 扩展坏了照常聊天
                print(f"[ext] intercept 失败：{type(e).__name__}: {e}", flush=True)
                hit = None
            if isinstance(hit, dict):
                if hit.get("ignore"):
                    return TurnResult(ignored=True, role=role, meta={"ext": True})
                reply = str(hit.get("reply") or "").strip()
                if reply:
                    self._record_and_remember(
                        sess, session_type, group_id, reply, now, day, text, gate,
                        channel=channel,
                    )
                    return TurnResult(
                        reply=reply,
                        parts=[reply],
                        role=role,
                        image=str(hit.get("image") or ""),
                        meta={"ext": True, **(hit.get("meta") or {})},
                    )

        # 他要表情包：不再"截断"聊天（以前只要句子带「发…表情」就直接回一句
        # 写死的「喏，给你一张」，连话都不说了）。现在照常让她说话，
        # 图在最后统一附上 —— 说了发，就真发；没图，就不许说发。
        wants_pic = (
            gate.role == ROLE_OWNER
            and session_type == SESSION_PRIVATE
            and wants_sticker(text)
        )

        # 画画/唱歌不再在这里硬截：让 LLM 先聊，判断出要做事时在回复里带
        # <cmd>…</cmd>，系统截掉命令、人话先发、任务后台跑（见下面的动作处理）。

        # 主人当场让它去 B 站（看动态 / 搜东西）：不走模型，直接做事，再把看到的告诉她
        if gate.role == ROLE_OWNER and session_type == SESSION_PRIVATE and feature_has("life"):
            try:
                from . import life as life_mod

                life_reply = life_mod.handle_request(self, text, now)
            except Exception:  # noqa: BLE001 手机出问题不能连话都不会说
                life_reply = None
            if life_reply:
                self._record_and_remember(
                    sess, session_type, group_id, life_reply, now, day, text, gate, channel=channel
                )
                return TurnResult(reply=life_reply, role=gate.role, meta={"life": True})

        # 主人让它上网搜：「搜一下 XXX」。B 站那条在生活层拦掉了，这里管剩下的。
        if gate.role == ROLE_OWNER and session_type == SESSION_PRIVATE:
            try:
                from .search import format_hits, parse_query
                from .search import search as web_search

                query = parse_query(text)
            except Exception:  # noqa: BLE001
                query = ""
            if query:
                hits = web_search(self.cfg, query)
                reply = format_hits(hits, query)
                self.journal.record_action(f"帮他在网上搜了「{query}」", now)
                self._record_and_remember(
                    sess, session_type, group_id, reply, now, day, text, gate, channel=channel
                )
                return TurnResult(reply=reply, role=gate.role, meta={"search": query})

        if command_note:
            self._record_and_remember(sess, session_type, group_id, command_note, now, day, text, gate, channel=channel)
            return TurnResult(
                reply=command_note,
                role=gate.role,
                boundary=info.get("boundary", False),
                assembly=None,
                meta={"gate": gate, "write": info},
            )

        # 查询改写：本地检索不够时才多调一次模型；离线/没 key 自动不开。
        rewrite_mode = "off"
        rewriter = None
        try:
            from .rewrite import mode_of
            from .rewrite import rewrite as rewrite_query

            rewrite_mode = mode_of(self.cfg)
            llm_rw = LLM(self.cfg)
            if not offline and rewrite_mode != "off" and llm_rw.available:
                context = " / ".join(
                    str(m.get("content") or "")[:80]
                    for m in (sess.thread or [])[-4:]
                    if m.get("role") == "user"
                )

                def rewriter(q: str, _llm=llm_rw, _ctx=context) -> list[str]:
                    return rewrite_query(_llm, q, _ctx)
        except Exception:  # noqa: BLE001 改写是加分项，配错了就当没开
            rewrite_mode, rewriter = "off", None

        assembly = assemble(
            self.mem,
            text,
            session_type,
            sender_id,
            sender_name,
            now,
            thread=sess.thread + ([{"role": "system", "content": "【本会话前情】" + sess.summary}] if sess.summary else []),
            duplicate_hint=hint,
            gap_hours=gap,
            today=clock.logical_date(now),
            group_id=group_id,
            channel=channel,
            sticker_ready=bool(wants_pic and sticker_count(self.cfg)),
            sticker_muted=bool(muted and not wants_pic),
            correction=correction,
            rewrite_mode=rewrite_mode,
            rewriter=rewriter,
            # 她自己刚才干了什么：不问也放着，免得他被「我刚才在刷视频」这类话问住时她先失忆
            recent_activity=self._recent_activity(now, group=session_type == SESSION_GROUP),
            ext_rules=self.ext.prompt(self.cfg),
            action_defs=self._action_defs(),
        )

        result: ReplyResult = generate_reply(self.cfg, assembly, sess.last_replies, offline=offline)
        # 真回了才计数（离线占位没烧 token；主人不算，免得把整个群熔了）
        if session_type == SESSION_GROUP and gate.role != ROLE_OWNER and not result.offline:
            self.guard.note_reply(group_id or "0", sender_id, now=now)
        # ---- LLM 的隐藏命令（画/唱/扩展动作）：先抽出来，人话照发，命令后台执行
        action_draw = action_song = None
        body_actions: list[str] = []
        ext_actions: list[dict] = []
        allowed_types = ["draw", "sing", "body", *self._action_defs().keys()]
        try:
            from .actions import body_list
            from .actions import extract as _extract_actions

            cleaned, acts = _extract_actions(result.raw or result.text, types=allowed_types)
            if acts:
                # text 也要用 cleaned：下面 split_bubbles 在 raw 为空时会回退到 text，
                # 而原始 text 里还带着 <cmd>，只发一个 body 动作时就会把命令原样发出去。
                result = ReplyResult(text=cleaned, offline=result.offline,
                                     error=result.error, rewritten=result.rewritten,
                                     raw=cleaned)
                if gate.role == ROLE_OWNER:
                    for act in acts:
                        kind = act.get("type")
                        if kind == "draw" and feature_has("draw") and action_draw is None:
                            action_draw = act
                        elif kind == "sing" and feature_has("song") and action_song is None:
                            action_song = act
                        elif kind == "body":
                            body_actions.extend(body_list(act))
                        elif kind in self._action_defs():
                            ext_actions.append(act)
        except Exception:  # noqa: BLE001 解析失败就当没有命令
            action_draw = action_song = None
            body_actions = []
            ext_actions = []
        if (action_draw or action_song or body_actions or ext_actions) \
                and not str(result.raw or "").strip():
            text = "好，等我一下。" if (action_draw or action_song or ext_actions) else "嗯。"
            result = ReplyResult(text=text, offline=result.offline,
                                 error=result.error, rewritten=result.rewritten,
                                 raw=text)
        # 空行/换行分段 = 一条一条发（最多 3 条）；没分段就还是原来的一条
        parts = split_bubbles(result.raw or result.text) or ["嗯，我在。"]
        reply = " ".join(parts)

        if action_draw is not None:
            subject = one_line(str(action_draw.get("subject") or ""))
            if not subject:
                try:
                    from .draw_habit import pick_subject

                    subject = pick_subject(self.cfg)
                except Exception:  # noqa: BLE001
                    subject = ""
            if subject:
                if reply != " ".join(parts):
                    parts = [reply]
                self._record_and_remember(
                    sess, session_type, group_id, reply, now, day, text, gate,
                    channel=channel,
                )
                self.journal.note_her_spoke(session_type, group_id, now, day)
                self._start_draw_async(sender_id, session_type, group_id, channel,
                                       subject, now, day)
                return TurnResult(reply=reply, parts=parts, image="",
                                  meta={"draw": subject, "draw_async": True})

        if action_song is not None:
            info = self._run_song_action(action_song, text, sender_id, session_type,
                                         group_id, channel)
            if info.get("silent"):
                return TurnResult(reply="", parts=[],
                                  meta={"song": info.get("query") or "", "silent": True})
            if info.get("error"):
                reply = str(info["error"])
                parts = [reply]
            if reply != " ".join(parts):
                parts = [reply]
            self._record_and_remember(
                sess, session_type, group_id, reply, now, day, text, gate,
                channel=channel,
            )
            self.journal.note_her_spoke(session_type, group_id, now, day)
            return TurnResult(reply=reply, parts=parts,
                              meta={"song": info.get("query") or "",
                                    "style": info.get("style") or ""})

        # 扩展动作：扩展自己执行，返回要发的话 / 图；不返回就继续正常回话。
        if ext_actions:
            act = ext_actions[0]
            try:
                out = self.ext.run_action(act, {
                    "cfg": self.cfg,
                    "service": self,
                    "text": text,
                    "sender_id": sender_id,
                    "session_type": session_type,
                    "group_id": group_id,
                    "channel": channel,
                    "now": now,
                    "reply": reply,
                    "image": "",
                    "meta": {},
                    "action": act,
                })
            except Exception as e:  # noqa: BLE001 扩展坏了照常发人话
                print(f"[ext] run_action 失败：{type(e).__name__}: {e}", flush=True)
                out = None
            if isinstance(out, dict):
                if out.get("error"):
                    reply = str(out["error"])
                elif out.get("reply") is not None:
                    reply = str(out.get("reply") or reply).strip() or reply
                if reply != " ".join(parts):
                    parts = [reply]
                self._record_and_remember(
                    sess, session_type, group_id, reply, now, day, text, gate,
                    channel=channel,
                )
                self.journal.note_her_spoke(session_type, group_id, now, day)
                return TurnResult(
                    reply=reply,
                    parts=parts,
                    image=str(out.get("image") or ""),
                    meta={"ext_action": act.get("type"),
                          **(out.get("meta") or {})},
                )

        # 模型没给命令时的兜底：明确的「画个…」「唱…」别落空
        if (action_draw is None and action_song is None and not ext_actions
                and (feature_has("draw") or feature_has("song"))):
            fallback = self._fallback_command(
                text, sender_id, sess, session_type, group_id, channel, now, day, gate
            )
            if fallback is not None:
                return fallback

        # 她嘴上说「发你了」「给你了」，但这条路径没带图 —— 手里有表情包就真发一张，
        # 没有就把这句话咽回去。不然就是空口说白话，他那边什么也收不到。
        # 只认明确的"发你了"这类说法：「这张也收到了」是在聊图，不是在发图，
        # 以前把它也当成发图，结果她一聊图就顺手甩一张出去。
        image = ""
        have_stickers = sticker_count(self.cfg) > 0
        claimed = bool(_CLAIM.search(reply))
        # 挑图时把「她现在的心情」也算进去（不是这句话的情绪）：
        # 心里堵着的时候，就算这句话在说「好呀」，也不该甩一张大笑的图。
        try:
            mood_tags = self.emotion.tags(now)
        except Exception:  # noqa: BLE001 心情读不出来就只按文字挑
            mood_tags = set()
        # 他明确要了：这张图一定发出去，话照说（也压过「先别发」）
        if wants_pic and have_stickers:
            image = sticker_for(self.cfg, f"{reply} {text}", moods=mood_tags) or sticker_pick(self.cfg)
        elif muted:
            # 他刚叫停，她嘴上又说「喏，给你一张」：这句话要咽回去，也不能补图。
            # 以前这里只看 claimed，不看 veto，结果叫她别发她还发。
            if claimed:
                reply = strip_claim(reply) or "嗯，先不发。"
        elif claimed and have_stickers:
            # 她主动说「发你了」：真有图就补上，别空口说白话；能按上下文挑就挑
            image = sticker_for(self.cfg, f"{reply} {text}", moods=mood_tags) or sticker_pick(self.cfg)
        elif claimed:
            reply = strip_claim(reply) or "我这儿还没存表情包，你发我几张我就有了。"
        elif not result.offline:
            # 主动配一张：按这句话的情绪/话题挑；有概率、冷却和上限，挑不中就空着。
            # 私聊和群聊各有一套参数，群聊的间隔按群算、上限按所有群合计。
            try:
                if session_type == SESSION_PRIVATE and gate.role == ROLE_OWNER:
                    image = proactive_pick(
                        self.cfg, f"{reply} {text}", now=now.timestamp(), scope="private",
                        moods=mood_tags,
                    )
                elif session_type == SESSION_GROUP:
                    image = proactive_pick(
                        self.cfg,
                        f"{reply} {text}",
                        now=now.timestamp(),
                        scope="group",
                        group_id=group_id or "",
                        moods=mood_tags,
                    )
            except Exception:  # noqa: BLE001 表情包坏了不能影响回话
                image = ""
        if image:
            self.journal.record_action("给他发了个表情包", now)
        # 话被改过（比如把"发你了"咽回去了）就退回一条，免得分段和内容对不上
        if reply != " ".join(parts):
            parts = [reply]
        # 离线占位不是她真的说过的话，不进原文、不进线程，免得污染记忆
        record_reply = not result.offline
        self._record_and_remember(
            sess, session_type, group_id, reply, now, day, text, gate,
            record_reply=record_reply, channel=channel,
        )
        if record_reply:
            self.journal.note_her_spoke(session_type, group_id, now, day)

        return TurnResult(
            reply=reply,
            parts=parts,
            role=gate.role,
            image=image,
            offline=result.offline,
            duplicate_hint=hint,
            boundary=info.get("boundary", False),
            assembly=assembly,
            error=result.error,
            notes=[command_note] if command_note else [],
            meta={"gate": gate, "write": info, "rewritten": result.rewritten,
                  "body": body_actions or None},
        )

    def _record_and_remember(
        self,
        sess: Session,
        session_type: str,
        group_id: str | None,
        reply: str,
        now: datetime,
        day,
        user_text: str,
        gate: Gate,
        record_reply: bool = True,
        channel: str = "",
    ) -> None:
        if record_reply:
            self.journal.record_her(session_type, group_id, reply, now, day, channel=channel)
            if session_type == SESSION_PRIVATE:
                self.note_spoke(now)
            # 她自己的动作（说了什么本身进原文；这里只在群里记「我接了话」）
            if session_type == SESSION_GROUP:
                self.journal.record_action(f"在群里接了一句：{clip(user_text, 20)}", now)
        # 私聊不在这里接线程：原文已经落盘，下一轮 _refresh_private_thread 会把
        # 「还没见过的行」接上。这里再接一遍就会重复，而且每轮重放会毁掉前缀缓存。
        # 群里还是就地追加（群线程本来就只进不出）。
        if session_type == SESSION_GROUP:
            sess.thread.append({"role": "user", "content": user_text})
            if record_reply:
                sess.thread.append({"role": "assistant", "content": reply})
            self._trim_thread(sess)
        if record_reply:
            sess.last_replies = ([reply] + sess.last_replies)[:3]
        sess.last_answer = reply

    def _draw_log(self, now: datetime, subject: str, error: str) -> None:
        """画图的失败只进日志，不拿技术细节去怼他。"""
        try:
            path = self.cfg.root / "data" / "draw.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(f"{now:%Y-%m-%d %H:%M:%S}\t{subject}\t{error}\n")
        except OSError:
            pass

    # ------------------------------------------------------------ 画 / 唱（后台）
    def _start_draw_async(self, sender_id, session_type, group_id, channel, subject, now, day):
        """她已经在人话里答应了，图在后台画；好了再从同一条通道发过去。"""
        def work():
            try:
                from .draw import comment
                from .draw import draw as draw_pic

                pic = draw_pic(self.cfg, subject)
                text = comment(self.cfg, subject, pic.get("seed")) or "喏，画好了。"
                self.journal.record_action(f"给他画了一张「{subject}」", now)
                self._deliver_drawing(channel, sender_id, session_type, group_id,
                                      text, str(pic.get("path") or ""), now, day)
            except Exception as e:  # noqa: BLE001
                self._draw_log(now, subject, f"{type(e).__name__}: {e}")
                self._deliver_drawing(channel, sender_id, session_type, group_id,
                                      "画笔这会儿不太顺手…等会儿再给你画？", "", now, day)

        threading.Thread(target=work, daemon=True, name="draw-now").start()

    def _deliver_drawing(self, channel, sender_id, session_type, group_id, text, path, now, day):
        if path:
            try:
                self.journal.record_her(session_type, group_id, text, now, day, channel=channel)
            except Exception:  # noqa: BLE001 记不进去也要把图发出去
                pass
        sent = False
        if channel == "qq":
            try:
                from . import qq_send

                segs = [qq_send.seg_text(text)]
                if path:
                    segs.append(qq_send.seg_image(path))
                qq_send.send_segments(
                    self.cfg, segs, timeout=90,
                    group_id=group_id if session_type == SESSION_GROUP else None,
                    user_id=None if session_type == SESSION_GROUP else sender_id,
                )
                sent = True
            except Exception as e:  # noqa: BLE001
                print(f"[draw] 后台发图失败：{type(e).__name__}: {e}", flush=True)
        cb = getattr(self, "deliver_drawing", None)
        if not sent and callable(cb):
            try:
                cb(text, path)
            except Exception as e:  # noqa: BLE001
                print(f"[draw] 弹图失败：{type(e).__name__}: {e}", flush=True)
        # 情绪：画出来一张是实打实的开心（他要的、她自己想画的都算）。
        if path:
            try:
                from .emotion import note as emotion_note

                emotion_note(self.cfg, "画完", "刚画完一张", "private", now)
            except Exception:  # noqa: BLE001
                pass

    def _run_song_action(self, action, text, sender_id, session_type, group_id, channel):
        from .song import (
            group_allowed,
            learn_style,
            parse_request,
            silent_reply,
            start_learning,
            stop_playing,
        )

        mode = str(action.get("mode") or "sing").lower()
        if mode not in ("sing", "practice", "stop"):
            mode = "sing"
        if mode == "stop":
            try:
                stop_playing()
            except Exception:  # noqa: BLE001
                pass
            return {"ok": True, "mode": "stop", "query": ""}
        req = {}
        try:
            req = parse_request(text) or {}
        except Exception:  # noqa: BLE001
            req = {}
        query = one_line(str(action.get("query") or "")) or one_line(str(req.get("query") or ""))
        if not query:
            return {"error": "想听哪首？把歌名告诉我，我去找。"}
        if session_type == SESSION_GROUP and not group_allowed(self.cfg, group_id):
            return {"error": "群里不点歌，想听私聊我。", "query": query}
        style = str(action.get("style") or req.get("style") or "normal")
        via = "pet" if channel in ("pet", "ball", "desktop") else "qq"
        if via == "qq":
            target = ({"kind": "group", "id": group_id}
                      if session_type == SESSION_GROUP and group_id
                      else {"kind": "private", "id": sender_id})
        else:
            target = None
        # 用 start_learning：识别把歌名听错一个字时，它会先纠错再搜，
        # 不然错字会被 B 站搜歌守卫判成「这首没搜到」（§8-32 之后加的）
        job = start_learning(self.cfg, query, via=via, mode=mode, target=target,
                             clip=req.get("clip"), style=style,
                             bvid=str(action.get("bvid") or req.get("bvid") or ""),
                             url=str(action.get("url") or req.get("url") or ""),
                             vibe=learn_style(self.cfg))
        if job.get("error"):
            return {"error": f"《{query}》还没弄成…{job['error']}", "query": query}
        # 歌名可能被纠错过（识别听错字）：回话按纠正后的名字说
        said = str(job.get("query") or query)
        if silent_reply(self.cfg) and mode == "sing":
            return {"silent": True, "query": said}
        return {"ok": True, "query": said, "mode": mode, "style": style}

    def _fallback_command(self, text, sender_id, sess, session_type, group_id, channel,
                          now, day, gate):
        """模型没给 <cmd> 时的兜底：明显的「画个…」「唱…」照样执行。

        只给主人。LLM 的 <cmd> 路径在 8xx 行已经校验过 gate.role，这条正则兜底
        以前漏了，结果普通群友/陌生人一句「唱《起风了》」也能让她开任务。
        """
        if gate.role != ROLE_OWNER:
            return None
        try:
            from .draw import draw_request

            req = draw_request(text)
        except Exception:  # noqa: BLE001
            req = None
        if req:
            subject = req[0]
            if not subject:
                try:
                    from .draw_habit import pick_subject

                    subject = pick_subject(self.cfg)
                except Exception:  # noqa: BLE001
                    subject = ""
            if subject:
                try:
                    from .draw import comment
                    from .draw import draw as draw_pic

                    pic = draw_pic(self.cfg, subject)
                    reply = comment(self.cfg, subject, pic.get("seed")) or "喏，画好了。"
                    self.journal.record_action(f"给他画了一张「{subject}」", now)
                    self._record_and_remember(
                        sess, session_type, group_id, reply, now, day, text, gate,
                        channel=channel,
                    )
                    return TurnResult(reply=reply, parts=[reply], image=pic["path"],
                                      meta={"draw": subject, "tags": pic.get("tags", "")})
                except Exception as e:  # noqa: BLE001
                    self._draw_log(now, subject, f"{type(e).__name__}: {e}")
                    reply = "画笔这会儿不太顺手…等会儿再给你画？"
                    self._record_and_remember(
                        sess, session_type, group_id, reply, now, day, text, gate,
                        channel=channel,
                    )
                    return TurnResult(reply=reply, parts=[reply], meta={"draw_failed": subject})
        try:
            from .song import (
                group_allowed,
                learn_style,
                llm_ack,
                parse_request,
                reply_text,
                silent_reply,
                start_learning,
            )

            sreq = parse_request(text)
        except Exception:  # noqa: BLE001
            sreq = None
        if not sreq:
            return None
        if sreq.get("mode") == "stop":
            reply = "好，不唱了。"
            self._record_and_remember(sess, session_type, group_id, reply, now, day, text, gate,
                                      channel=channel)
            return TurnResult(reply=reply, parts=[reply])
        if not sreq.get("query"):
            reply = "想听哪首？把歌名告诉我，我去找。"
            self._record_and_remember(sess, session_type, group_id, reply, now, day, text, gate,
                                      channel=channel)
            return TurnResult(reply=reply, parts=[reply])
        if session_type == SESSION_GROUP and not group_allowed(self.cfg, group_id):
            reply = "群里不点歌，想听私聊我。"
            self._record_and_remember(sess, session_type, group_id, reply, now, day, text, gate,
                                      channel=channel)
            return TurnResult(reply=reply, parts=[reply])
        via = "pet" if channel in ("pet", "ball", "desktop") else "qq"
        if via == "qq":
            target = ({"kind": "group", "id": group_id}
                      if session_type == SESSION_GROUP and group_id
                      else {"kind": "private", "id": sender_id})
        else:
            target = None
        style = str(sreq.get("style") or "normal")
        job = start_learning(self.cfg, sreq["query"], via=via, mode=sreq["mode"], target=target,
                             clip=sreq.get("clip"), style=style,
                             bvid=str(sreq.get("bvid") or ""), url=str(sreq.get("url") or ""),
                             vibe=learn_style(self.cfg))
        if job.get("error"):
            label = "清唱版" if str(job.get("style") or style) == "acappella" else "这首"
            reply = f"《{sreq['query']}》的{label}还没弄成…{job['error']}"
            self._record_and_remember(sess, session_type, group_id, reply, now, day, text, gate,
                                      channel=channel)
            return TurnResult(reply=reply, parts=[reply], meta={"song": sreq["query"]})
        said = str(job.get("query") or sreq["query"])
        if silent_reply(self.cfg) and sreq["mode"] == "sing":
            return TurnResult(reply="", parts=[],
                              meta={"song": said, "silent": True})
        in_group = via == "qq" and session_type == SESSION_GROUP
        if sreq["mode"] == "practice" and not in_group:
            reply = reply_text(self.cfg, said, mode="practice",
                               cached=bool(job.get("cached")), style=style)
        else:
            reply = llm_ack(self.cfg, said, sreq["mode"],
                            bool(job.get("cached")), via, target, style)
        self._record_and_remember(sess, session_type, group_id, reply, now, day, text, gate,
                                  channel=channel)
        return TurnResult(reply=reply, parts=[reply],
                          meta={"song": sreq["query"], "style": style})

    # ------------------------------------------------------------ 主人命令
    def _owner_delete(self, text: str) -> str:
        # 「把猫忘掉」和「忘掉猫」都收；先抓「把 X 删掉」的结构。
        m = re.search(r"把(.+?)(?:忘掉|忘记|删掉|删除|清掉|抹掉|别再提)", text)
        if m:
            needle = m.group(1)
        else:
            m = re.search(r"(?:忘掉|忘记|删掉|删除|清掉|抹掉|别再提)\s*[：:]?\s*(.+)", text)
            needle = m.group(1) if m else ""
        needle = needle.strip(" 。！？!?")
        # 「忘掉吧，忘掉吧」这种口语，别把「吧」当成要忘的内容
        needle = re.sub(r"^[吧呢啊呀哦，,。.\s]+", "", needle).strip()
        # 「我忘记了」会退化成「了」；「忘掉她了」退化成「她」。
        # 单字虚词不能当 needle，否则 mark_facts 会按子串删掉一大片事实。
        stop = {"了", "吧", "呢", "啊", "呀", "哦", "的", "是", "它", "他", "她",
                "我", "你", "这个", "那个", "这件事", "那件事", "这件事了"}
        if needle.endswith("了") and len(needle) > 2:
            needle = needle[:-1].rstrip()
        if (not needle or needle in stop
                or not re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", needle)):
            return "要忘掉哪件事？说具体一点，我才知道删哪条。"
        n = self.mem.mark_facts(needle, reason="主人要求忘掉")
        forgot = self.mem.forget_person(needle)
        scrubbed = 0
        try:
            scrubbed = self.mem.scrub_people(needle)
            self.rebuild_index()
        except Exception:  # noqa: BLE001 删干净比索引重要，索引坏了下次会重建
            pass
        self.mem.set_status(
            {"上次错误": f"主人要求忘掉「{needle}」，删掉 {n} 条事实、{scrubbed} 张卡上的行、删 {int(forgot)} 张人物卡"}
        )
        return "好，忘了。" if (n or forgot or scrubbed) else f"我记忆里没有「{needle}」这件事。"

    def _owner_identity(self, text: str, now: datetime) -> str:
        target, sentence = identity_payload(text)
        if target == "unknown":
            return "这个我还没听明白，换个说法？"
        changed = self.journal.apply_identity(target, sentence, now)
        return "好，记住了。" if changed else "这句我没写进核心，怕改坏；你可以手改核心，或者再说一次。"

    # ------------------------------------------------------------ 只读装配
    def recall(
        self,
        query: str,
        sender_id: str | None = None,
        session_type: str = SESSION_PRIVATE,
        group_id: str | None = None,
        nickname: str = "",
        now: datetime | None = None,
        gap_hours: float = 999.0,
    ) -> Assembly:
        """只走装配、不写盘（回归集/评测用）。

        gap_hours 默认 999 = 「隔了很久」的冷启动：会带交接 / 最近几天 / 我想起。
        日常对话的间隔通常是几十分钟，评测想量日常就用 0.5。
        """
        now = now or datetime.now()
        sender_id = sender_id or self.cfg.owner_id
        sender_name = self.cfg.owner_name if str(sender_id) == self.cfg.owner_id else (nickname or str(sender_id))
        return assemble(
            self.mem,
            query,
            session_type,
            str(sender_id),
            sender_name,
            now,
            thread=[],
            gap_hours=gap_hours,
            today=clock.logical_date(now),
            group_id=group_id,
            recent_activity=self._recent_activity(now, group=session_type == SESSION_GROUP),
        )

    # ------------------------------------------------------------ 只记录，不说话
    def observe(
        self,
        sender_id: str,
        text: str,
        session_type: str = SESSION_GROUP,
        group_id: str | None = None,
        nickname: str = "",
        now: datetime | None = None,
    ) -> dict:
        """群里的旁听：只走写路径，不组装、不调模型、不回复。

        QQ 群里绝大多数消息属于这一类：记进 3 分钟缓冲 / 30 分钟摘要，
        被 @ 时才有线程和回复。
        """
        now = now or datetime.now()
        text = one_line(text or "")[:1200]
        if not text:
            return {"written": False, "ignored": True}
        if session_type == SESSION_GROUP and not group_id:
            group_id = "0"
        gate = gate_message(sender_id, text, session_type, self.cfg, now, group_id=group_id, nickname=nickname)
        if not gate.allowed:
            return {"written": False, "refused": True}
        sender_name = self.cfg.owner_name if gate.role == ROLE_OWNER else (nickname or sender_id)
        info = self.journal.record_user(
            gate,
            sender_id,
            sender_name,
            text,
            session_type,
            group_id,
            now,
        )
        self._group_mood(text, now, sender_id, nickname or sender_name)
        return {
            "written": bool(info.get("written")),
            "boundary": bool(info.get("boundary")),
            "fact": bool(info.get("fact")),
            "refused": bool(gate.must_refuse),
        }

    def note_spoke(self, now: datetime | None = None) -> None:
        """她开口了（回话、主动说话都算）。之后他没回音就记「被忽视」。"""
        self._last_her_at = now or datetime.now()

    def _learn_affinity(self, sender_id: str, nickname: str, delta: float,
                        now: datetime) -> None:
        """这个人在群里让她舒服/难受了多少 → 好感动多少。状态独立，不进记忆。"""
        try:
            from .affinity import learn

            learn(self.cfg, sender_id, delta, nickname, now)
        except Exception:  # noqa: BLE001 好感度坏了不能影响回话
            pass

    def _reply_factor(self, sender_id: str, now: datetime) -> float:
        """被 @ 时她有多愿意回：看对这个人熟不熟（0.2~1.0）。"""
        try:
            if not bool(getattr(self.cfg.knobs, "group_affinity_reply_enabled", True)):
                return 1.0
            from .affinity import reply_factor

            return reply_factor(self.cfg, sender_id, now)
        except Exception:  # noqa: BLE001
            return 1.0

    @staticmethod
    def _about_her(text: str) -> bool:
        """群里这句话是不是在说她。没点她名字、也没有评价她的词，就不算。"""
        s = str(text or "")
        if re.search(r"(小潮|潮汐|Tide)", s):
            return True
        if "她" in s:
            from .emotion import classify

            return bool(classify(s)) or bool(re.search(r"(起哄|哈哈哈|笑死|可爱|好看)", s))
        return False

    def _group_mood(self, text: str, now: datetime,
                    sender_id: str = "", nickname: str = "") -> None:
        """群里的旁听也影响她，但要压两层，不然几千条群消息能把她泡在皮质醇里：

        - 只在她被议论的时候（`_about_her`）；
        - 一次刺激只算 0.35，且 15 分钟内最多一次。

        顺带给这个人记一笔好感度：夸她 +0.05，凶她 -0.10。
        """
        try:
            if not text or not self._about_her(text):
                return
            last = self._group_emotion_at
            if last and (now - last) < timedelta(minutes=15):
                return
            from .emotion import classify

            kind = classify(text) or ""
            if not kind:
                if re.search(r"(起哄|哈哈哈|笑死|可爱|好耶)", text):
                    kind = "群里起哄"
                else:
                    return
            delta = self.emotion.react(kind, clip(text, 20), "group", now, scale=0.35)
            self._group_emotion_at = now
            if sender_id:
                self._learn_affinity(sender_id, nickname, delta, now)
        except Exception:  # noqa: BLE001 群里情绪算错不能影响旁听
            pass

    # ------------------------------------------------------------ 心跳
    def tick(self, now: datetime | None = None, consolidate: bool = True, rules: bool = False) -> dict:
        now = now or datetime.now()
        notes = []
        self.mem.ensure_skeleton()

        # 情绪：按半衰期往回走。原来这里那套「此刻」是按小时套模板
        # （5 点「刚醒」、18 点「屋里只有风扇的声音」），跟他做了什么无关，
        # 所以她一整天的「心情」其实是一个时钟。现在换成真的内分泌状态。
        try:
            self.emotion.settle(now)
            notes.append("情绪已代谢")
        except Exception as e:  # noqa: BLE001 情绪坏了不能拖垮心跳
            notes.append(f"情绪代谢失败：{type(e).__name__}")
        # 很久没来找我：一次分离只记一笔，夜里不算（她也在睡）。
        last = self._last_user_at
        if last and self._missed_for != last and 9 <= now.hour < 24:
            if (now - last).total_seconds() / 3600.0 >= 4.0:
                try:
                    self.emotion.observe("被冷落", now=now)
                    self._missed_for = last
                    notes.append("记了一笔失落")
                except Exception:  # noqa: BLE001
                    pass
        # 她刚说完话他就没影了：20 分钟没回音记一笔「被忽视」。
        # 比「被冷落」轻得多（+8 皮质醇 vs +26），但来得快 —— 已读不回是当场的事。
        spoke = self._last_her_at
        if spoke and self._unanswered_for != spoke and 9 <= now.hour < 24:
            answered = bool(self._last_user_at and self._last_user_at > spoke)
            quiet = (now - spoke).total_seconds() / 60.0
            if not answered and quiet >= IGNORED_AFTER_MINUTES:
                try:
                    self.emotion.observe("被忽视", now=now)
                    self._unanswered_for = spoke
                    notes.append("记了一笔没回音")
                except Exception:  # noqa: BLE001
                    pass

        due = self.due(now)
        result = None
        # 日批时间可配：默认 5 点后就能跑，改成 8 是「等他起床、有网了再批」。
        # 过点之前 due 会一直为真，到了点第一次心跳就补上，不会漏。
        after_hour = int(self.cfg.knobs.as_dict().get("consolidate_hour", 5))
        if consolidate and due and now.hour >= after_hour:
            from .consolidate import Consolidator

            use = "rules" if rules else "auto"
            # 停机两三天再开机：从水位后一天开始逐个补齐，别只批「昨天」。
            targets = self.missed_days(now) or [None]
            for day in targets:
                result = Consolidator(self.mem, use_llm=use).run(target=day, now=now)
                notes.append(result.message)
                if not result.ok:
                    break   # 这一步失败就停在这儿，等「重试于」到了再往后补
                if day is None:
                    break

        # 晚间日记：23:00–05:00 之间先写当天草稿（只写日记，不动事实/事件/水位）。
        # 05:00 的完整批只跑一次，草稿会作为输入补成定稿。
        diary_hour = int(self.cfg.knobs.as_dict().get("diary_hour", 23))
        in_window = (now.hour >= diary_hour or now.hour < 5) if diary_hour >= 5 else now.hour < 5
        if consolidate and in_window:
            today = clock.logical_date(now)
            if today > clock.completed_logical_date(now):
                from .consolidate import Consolidator

                use = "rules" if rules else "auto"
                evening = Consolidator(self.mem, use_llm=use).evening_diary(today, now)
                if evening and evening.message:
                    notes.append(evening.message)

        if now.day == 1:
            from .consolidate import Consolidator

            c = Consolidator(self.mem, use_llm="rules")
            prev = (now.replace(day=1) - timedelta(days=1))
            try:
                c._compress_month(prev)  # noqa: SLF001 内部复用
                notes.append("月概览已压")
            except Exception as e:
                notes.append(f"月压失败：{e}")
            ok, errs = self.mem.drill()
            notes.append("回滚演练" + ("通过" if ok else "失败"))

        return {"notes": notes, "due": due, "result": result}

    def due(self, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        target = clock.completed_logical_date(now)
        sources = self.mem.list_raw_sources(target)
        if not sources:
            return False
        counts = {src: self.mem.raw_count(target, src) for src in sources}
        status = self.mem.status()
        # 上次跑失败过：等退避时间到了再试，别让 5 秒一拍的心跳连着重跑日批（每次两个模型调用）。
        try:
            retry_at = datetime.fromisoformat(str(status.get("重试于") or ""))
        except ValueError:
            retry_at = None
        if retry_at and now < retry_at:
            return False
        try:
            from .consolidate import watermark_date

            wm_day = watermark_date(status, target)
        except Exception:  # noqa: BLE001 水位读坏了就按「还没批」处理
            wm_day = None
        if wm_day is None:
            return True
        if wm_day < target:
            return True
        if wm_day > target:
            # 水位已经走到更晚的一天，不算落后
            return False
        expected = self._counts_text(counts)
        return str(status.get("水位") or "") != f"{fmt_md(target)} {expected}"

    def missed_days(self, now: datetime | None = None, limit: int = 7) -> list:
        """还没批过的逻辑日，从最老的开始（最多 `limit` 天）。

        以前 tick() 只批「昨天」：停机两天的话中间那一天永远补不上，
        状态里却写着「开机补跑」。这里把缺口一次列出来，由 tick 逐个补。
        没有水位（第一次跑）就返回 []，让调用方走默认的「昨天」。
        """
        import datetime as _dt

        now = now or datetime.now()
        try:
            from .consolidate import watermark_date

            wm_day = watermark_date(self.mem.status(), clock.completed_logical_date(now))
        except Exception:  # noqa: BLE001
            return []
        if wm_day is None:
            return []
        day = wm_day + _dt.timedelta(days=1)
        last = clock.completed_logical_date(now)
        out = []
        while day <= last and len(out) < max(1, limit):
            out.append(day)
            day += _dt.timedelta(days=1)
        return out

    def _counts_text(self, counts: dict) -> str:
        parts = []
        if "private" in counts:
            parts.append(f"私聊 {counts['private']}")
        for src in sorted(k for k in counts if k != "private"):
            parts.append(f"群 {src.split(':', 1)[-1]} {counts[src]}")
        return " / ".join(parts) or "0"

    # ------------------------------------------------------------ 主动
    def proactive(self):
        """主动循环（P2）：返回唯一的 Proactive 实例，调用方按 pro.tick(now, presence) 驱动。

        挂到 self 上而不是每次新建，是因为 20~40 分钟的节奏、每日上限、上次开口时间
        都存在这个对象里（并落盘到 data/proactive.json），每轮重置就成了闹钟。
        """
        pro = getattr(self, "_proactive", None)
        if pro is None:
            from .proactive import Proactive

            pro = Proactive(self)
            self._proactive = pro
        return pro

    # ------------------------------------------------------------ 自己学歌
    def song_habit(self):
        """她自己随机学歌：心跳驱动，一个进程一个实例，状态落盘。"""
        obj = getattr(self, "_song_habit", None)
        if obj is None:
            from .song_habit import SongHabit

            obj = SongHabit(self)
            self._song_habit = obj
        return obj

    # ------------------------------------------------------------ 自己画画
    def draw_habit(self):
        """她自己随手画画：心跳驱动，状态落盘到 data/draw_habit.json。"""
        obj = getattr(self, "_draw_habit", None)
        if obj is None:
            from .draw_habit import DrawHabit

            obj = DrawHabit(self)
            self._draw_habit = obj
        return obj

    # ------------------------------------------------------------ 面对面心跳
    def room_heart(self):
        """面对面心跳：只在她「面对面」形态、看着摄像头时，按画面自己找话题。

        和 `proactive()` 分开：自己的节奏/每日上限/状态文件，不占主动说话的额度。
        """
        obj = getattr(self, "_room_heart", None)
        if obj is None:
            from .room_heart import RoomHeart

            obj = RoomHeart(self)
            self._room_heart = obj
        return obj

    # ------------------------------------------------------------ 生活
    def life(self):
        """生活层（P3）：她自己去刷 B 站。同样一个进程一个实例。"""
        obj = getattr(self, "_life", None)
        if obj is None:
            from .life import Life

            obj = Life(self)
            self._life = obj
        return obj

    def _recent_activity(self, now: datetime, group: bool = False) -> list[str]:
        """她刚才在手机上干了什么。生活层坏了也不能影响回话。"""
        if not feature_has("life"):
            return []
        try:
            return self.life().recent_activity(now, group=group)
        except Exception:  # noqa: BLE001 生活层是加分的，坏了就当没这回事
            return []

    # ------------------------------------------------------------ 管理动作
    def consolidate(self, target=None, force: bool = False, rules: bool = False, now: datetime | None = None):
        from .consolidate import Consolidator

        return Consolidator(self.mem, use_llm=("rules" if rules else "auto")).run(
            target=target, force=force, now=now
        )

    def rebuild_index(self) -> str:
        IndexDB(self.cfg).rebuild(self.mem)
        self.mem.set_status({"索引库": "已手动重建"})
        return "索引已重建"

    def rollback(self, snapshot: str | None = None) -> str:
        ok, msg, _ = self.mem.rollback(snapshot)
        if ok:
            self.rebuild_index()
        return msg

    def status_text(self) -> str:
        status = self.mem.status()
        health = self.mem.health()
        lines = ["记忆状态："]
        for k, v in status.items():
            lines.append(f"  {k}：{v}")
        lines.append("体检：")
        for k, v in health.items():
            lines.append(f"  {k}：{v}")
        try:
            from .hygiene import check as hygiene_check
            from .hygiene import snapshot as hygiene_snapshot

            findings = hygiene_check(self.mem)
            lines.append("红线：" + ("通过" if not findings else f"{len(findings)} 条告警"))
            for f in findings[:3]:
                lines.append(f"  {'✗' if f['level'] == 'violation' else '⚠'} {f['msg']}")
            # 遗忘曲线：候选只是「该忘的」，真正搬走要 memory-check --fix
            snap = hygiene_snapshot(self.mem)
            bins = " · ".join(f"{k} {v}" for k, v in snap["curve"].items() if v)
            lines.append(
                f"遗忘曲线：候选 {snap['facts_forgettable']} 条"
                + (f" · {bins}" if bins else "（还没有事实）")
            )
        except Exception as e:  # noqa: BLE001 体检坏了不影响看状态
            lines.append(f"红线：检查失败（{type(e).__name__}: {e}）")
        llm = self.cfg.llm()
        lines.append(f"模型：{'已接上 ' + llm.model if llm.enabled else '离线'}")
        # 情绪：五个浓度 + 读出来的情绪。调参就看这一段。
        try:
            lines.append("情绪：")
            for k, v in self.emotion.snapshot().items():
                if isinstance(v, list):
                    v = "、".join(v) or "没什么"
                elif not v:
                    v = "—"
                lines.append(f"  {k}：{v}")
        except Exception as e:  # noqa: BLE001 情绪读不出来不影响看状态
            lines.append(f"  情绪：读不出来（{type(e).__name__}: {e}）")
        # 好感度：群里最熟的几个人（状态，不在记忆里）
        try:
            from .affinity import top as affinity_top

            rows = affinity_top(self.cfg, 3)
            if rows:
                lines.append("好感度：" + "；".join(f"{n} {s:.2f}" for n, s in rows))
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(lines)
