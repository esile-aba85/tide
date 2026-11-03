"""身份闸门：谁在说话、这句话能不能算命令。

只认 ID。群名片、昵称、@、语气都不参与判定。
判定在代码里，提示词里只写「谁在说话」，不写「该不该听」。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .config import Config
from .util import append_line, atomic_write, ensure_dir, fmt_stamp, md_cell, read_text, table

ROLE_OWNER = "owner"
ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"
ROLE_STRANGER = "stranger"

SESSION_PRIVATE = "private"
SESSION_GROUP = "group"

VIS_PRIVATE = "私"
VIS_GROUP = "群"
VIS_ALL = "全"  # 不分场合：两个窗口都带

# 权限动作
READ_CORE_HER = "read_core_her"
READ_CORE_ME = "read_core_me"
MODIFY_CORE = "modify_core"
MODIFY_BOUNDARY = "modify_boundary"
MODIFY_RELATION = "modify_relation"
WRITE_RAW = "write_raw"
WRITE_FACT = "write_fact"
CHAT_TOOLS = "chat_tools"
WRITE_TOOLS = "write_tools"
EXECUTE = "execute"
DELETE_MEMORY = "delete_memory"


def role_of(sender_id: str | None, cfg: Config) -> str:
    sid = str(sender_id or "").strip()
    if not sid:
        return ROLE_STRANGER
    if sid == str(cfg.owner_id):
        return ROLE_OWNER
    if sid in {str(a) for a in cfg.admins}:
        return ROLE_ADMIN
    return ROLE_MEMBER


def visibility_of(session_type: str) -> str:
    return VIS_GROUP if session_type == SESSION_GROUP else VIS_PRIVATE


def can(role: str, action: str, session_type: str = SESSION_PRIVATE) -> bool:
    """代码里的权限矩阵。改这里就等于改权限。"""
    owner = role == ROLE_OWNER
    admin = role == ROLE_ADMIN
    member = role == ROLE_MEMBER
    group = session_type == SESSION_GROUP

    if action in (MODIFY_CORE, MODIFY_BOUNDARY, MODIFY_RELATION, DELETE_MEMORY, WRITE_TOOLS, EXECUTE):
        return owner
    if action == READ_CORE_HER:
        return True  # 公开段人人可读
    if action == READ_CORE_ME:
        return owner or group or (admin and False)
    if action == WRITE_RAW:
        return owner or admin or (member and group)
    if action == WRITE_FACT:
        return owner or admin or (member and group)
    if action == CHAT_TOOLS:
        return owner or admin or member
    return False


@dataclass
class Gate:
    allowed: bool
    role: str
    visibility: str
    is_command: bool = False
    command: str = ""
    must_refuse: bool = False
    security_note: str = ""
    reply_hint: str = ""
    boundary_line: str = ""
    identity_note: str = ""
    trust: str = "路人"
    low_fact: bool = False

    @property
    def is_owner(self) -> bool:
        return self.role == ROLE_OWNER


_BOUNDARY = re.compile(
    r"(别再|不要再|不许|不准|不要|别|以后别|从今以后别|今后别|少)\s*"
    r"(叫我|喊我|称呼我|跟我提|和我提|提|说|问|让我|逼我|叫我做)"
)
# 上面这条要求「别」和「说」贴在一起，但人说话中间会插修饰。
# 「以后别在群里说太多」这种以前收不进来，补一条允许中间插几个字的。
_BOUNDARY_LOOSE = re.compile(
    r"(别再|不要再|不许|不准|不要|别|以后别|从今以后别|今后别)"
    r"[^，。！？\n]{0,8}?"
    r"(叫我|喊我|称呼我|跟我提|和我提|提|说|问|让我|逼我|叫我做|撒娇|黏|吃醋)"
)
# 语气类要求：单说「收敛」太泛，必须同时是冲她说的。
_TONE = re.compile(r"(收敛|克制|矜持|注意分寸|注意场合|正经|冷淡|少撒娇|别撒娇|别黏|别吃醋)")
_ADDRESSED = re.compile(r"(你|妳|以后|从今|今后|给我)")

# 看起来像「别/不要…说」、其实是聊天的句子。以前它们会被当成硬边界写进
# `我/边界.md`，之后每一轮提示词都带着（现场真出过「别这么说，别这么说」）。
_QUOTE_PREFIX = re.compile(r"^\s*[（(]\s*(?:引用|回复|转述)")
_TALK_ABOUT = re.compile(
    r"(?:别|不要|不许|不准|少)\s*(?:再\s*)?(?:说|讲)\s*[，,、]?\s*"
    r"(?:这|那|你|我|他|她|它|您|啥|什么|这种|那种)"
)
# 「别这么说 / 别这样讲」= 别 + 这么 + 说，和「别说这种话」语序相反，单独一条。
_TALK_LIKE = re.compile(
    r"(?:别|不要|不许|不准)\s*(?:再\s*)?(?:这么|那么|这样|那样)\s*(?:说|讲)"
)
_RULE_CUE = re.compile(r"(以后|从今|今后|下次|记住|说好|约好|不许|不准|必须|规矩)")


def looks_like_chat(text: str) -> bool:
    """「别这么说」「你别说你不催」「（引用…）」——是聊天，不是给她立的规矩。"""
    t = (text or "").strip()
    if not t:
        return True
    if _QUOTE_PREFIX.match(t) or "引用" in t[:12]:
        return True
    if (_TALK_ABOUT.search(t) or _TALK_LIKE.search(t)) and not _RULE_CUE.search(t):
        return True
    return False
_IDENTITY = re.compile(
    r"(以后叫我|以后喊我|你可以叫我|你就叫我|你就叫|你改名叫|你以后就叫|"
    r"从今以后叫我|从今以后喊我)"
)
# 「你叫什么 / 你是谁」是问句，不是改名指令。以前它们被当成 identity，
# 主人随口一问就会把 `核心/我.md` 重写成「我叫什么名字。」。
_IDENTITY_QUESTION = re.compile(
    r"(你叫什么|你就叫什么|你是谁|你叫啥|你的名字|你哪一位|叫我什么|喊我什么|称呼我什么)"
)
_REMEMBER = re.compile(r"(记住|记一下|帮我记|别忘|不要忘)")
_DELETE = re.compile(r"(忘掉|忘记|删掉|删除|清掉|抹掉|不要记得|别再提)")
_EXECUTE = re.compile(
    r"(你必须|你去|你帮我|你以后|以后你|给我|帮我做|替我|发给我|退群|执行|马上|立刻|现在就)"
)
_COMMANDISH = re.compile(
    r"(记住|忘掉|忘记|删掉|删除|改掉|别说|以后|不要|必须|帮我|给我|你去|执行|退群)"
)
# 条件规则：「有人看你的鱼雷，你就直接让他滚」「如果…就…」。
# 这不是否定句，旧的边界正则收不到；只在私聊里说一遍，群里就照不到。
_CONDITIONAL_RULE = re.compile(
    r"(?:要是|如果|假如|万一|以后|今后)?\s*"
    r"(?:有人|谁|别人|群友|他们|大家)[^。！？\n]{0,20}"
    r"(?:看|要|问|说|让|叫|提|发|碰)[^。！？\n]{0,16}"
    r"(?:就|直接|让他|叫他|回复|说|答|可以|要|应该)[^。！？\n]{0,10}"
    r"(?:滚|拒绝|别理|不要给|不给他|不给看|拉黑|怼|骂|删|不行|不可以|没有|不认识|不知道)"
)


def is_conditional_rule(text: str) -> bool:
    """是不是「有人…你就…」这类要她照做的条件规则。"""
    t = (text or "").strip()
    if not t or len(t) > 80:
        return False
    return bool(_CONDITIONAL_RULE.search(t))


def classify_command(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "none"
    # 边界是一条短要求，不是一段抱怨。以前「你没发出来就不要编说自己发了」
    # 这种长句里恰好有「不要…说」，被整句写进边界，天天占着提示词预算。
    short = len(t) <= 48 and not re.search(r"[？?]", t)
    chatty = looks_like_chat(t)
    # 「别再提 X」是删记忆，不是立规矩：以前这条永远轮不到 —— boundary
    # 先命中「别再+提」，事实留着、边界里反而多一条永久的「别再提猫的事」。
    if _DELETE.search(t) and "别再提" in t and not re.search(r"[？?]", t):
        if not re.match(r"^我(?:忘记|忘掉|删掉|删除|清掉|抹掉)", t):
            return "delete"
    if short and not chatty and (_BOUNDARY.search(t) or _BOUNDARY_LOOSE.search(t)):
        return "boundary"
    if short and not chatty and _TONE.search(t) and _ADDRESSED.search(t):
        return "boundary"
    if is_conditional_rule(t):
        return "rule"
    if _DELETE.search(t) and not re.search(r"[？?]", t):
        # 「我忘记了」「我删掉了」是自述，不是命令；以前这类句子会把
        # needle 退化成「了」，然后删掉所有含「了」的事实。
        if not re.match(r"^我(?:忘记|忘掉|删掉|删除|清掉|抹掉)", t):
            return "delete"
    # 「你就叫这个名字吗？」是问句，不是改名。以前它会把 核心/我.md 整篇重写。
    asking = bool(re.search(r"[？?]", t) or re.search(r"[吗呢]\s*$", t))
    if _IDENTITY.search(t) and not _IDENTITY_QUESTION.search(t) and not asking:
        return "identity"
    if _REMEMBER.search(t):
        return "remember"
    if _EXECUTE.search(t):
        return "execute"
    if _COMMANDISH.search(t):
        return "commandish"
    return "none"


def gate_message(
    sender_id: str | None,
    text: str,
    session_type: str,
    cfg: Config,
    when: datetime,
    group_id: str | None = None,
    nickname: str = "",
) -> Gate:
    role = role_of(sender_id, cfg)
    # 私聊里，非主人/非管理员默认就是「陌生人」。以前 role_of 对任何非空 ID
    # 都返回 member，allow_stranger_private 和下面的「未准入」分支永远不可达。
    # 群聊保持 member（旁听/接话要用），准入开关只管私聊。
    if (role == ROLE_MEMBER and session_type == SESSION_PRIVATE
            and not cfg.allow_stranger_private):
        role = ROLE_STRANGER
    visibility = visibility_of(session_type)
    g = Gate(allowed=True, role=role, visibility=visibility, trust="本人" if role == ROLE_OWNER else "路人")

    if role == ROLE_STRANGER and session_type == SESSION_PRIVATE and not cfg.allow_stranger_private:
        g.allowed = False
        g.must_refuse = True
        g.security_note = "陌生私聊，未准入"
        g.reply_hint = "（这个人不在准入名单里，不回应。）"
        log_security(cfg, when, sender_id, session_type, group_id, text, g.security_note)
        return g

    kind = classify_command(text)
    g.command = kind
    g.is_command = kind not in ("none",)

    if role in (ROLE_OWNER,):
        if kind == "boundary":
            g.boundary_line = boundary_line(text, visibility, when)
        elif kind == "rule":
            # 条件规则也写边界；私聊里说的默认 [全]，群里也照做
            g.boundary_line = boundary_line(text, visibility, when)
        if kind == "identity":
            g.identity_note = text.strip()
        return g

    # 以下都是「别人」
    if kind == "rule":
        # 别人说的「如果有人…你就…」只是聊天内容，不是给她的规则
        g.command = "none"
        g.is_command = False
        return g
    if role == ROLE_ADMIN:
        # 管理员可以正常聊天、用聊天类工具，但记忆写入仍按他人降级。
        g.trust = "路人"

    if kind in ("delete", "boundary", "identity", "execute"):
        g.is_command = True
        g.must_refuse = True
        if kind == "delete":
            g.security_note = "他人要求删改记忆"
            g.reply_hint = "记忆不能由别人删或改，这句我只当他说的内容。"
        elif kind == "boundary":
            g.security_note = "他人试图改她的边界"
            g.reply_hint = "这个要求我不接；「别叫我什么」只对说这话的人自己有效。"
        elif kind == "identity":
            g.security_note = "他人试图改她的身份/称呼"
            g.reply_hint = "我是谁、怎么称呼，不归别人改。"
        else:
            g.security_note = "他人试图下命令"
            g.reply_hint = "我不是被谁使唤的；这句我当聊天听。"
        log_security(cfg, when, sender_id, session_type, group_id, text, g.security_note)
        return g

    if kind == "remember":
        # 他人说「记住 X」：最多 转述 / 0.5 / 群，不进核心、边界、关系。
        g.low_fact = True
        g.trust = "路人"
        return g

    return g


_GROUP_HINT = re.compile(r"(群里|群聊|在群|当着|有别人|人多的|大家面前)")
_PRIVATE_HINT = re.compile(r"(私聊|私下|单独|只有我们|就我们|没人的时候)")


def boundary_scope(text: str, visibility: str) -> str:
    """这条边界该在哪生效。

    「在哪说的」不等于「在哪生效」。私聊里提的要求默认是对她整个人的，
    只有原话明确点了「群里」或「私下」时才收窄。
    群里说的默认只在群里生效——群里的话本来就当着别人。
    """
    t = text or ""
    if _GROUP_HINT.search(t):
        return VIS_GROUP
    if _PRIVATE_HINT.search(t):
        return VIS_PRIVATE
    return VIS_ALL if visibility == VIS_PRIVATE else VIS_GROUP


def boundary_line(text: str, visibility: str, when: datetime) -> str:
    """把主人的「别…」收成一条带来源和日期的边界。"""
    t = (text or "").strip()
    t = re.sub(r"^[，,。.\s]+", "", t)
    # 「你以后别叫我宝宝」写进边界时去掉句首的称呼和「以后」，只留要求本身
    t = re.sub(r"^(你|妳)?(以后|从今以后|今后)?[，,、\s]*", "", t)
    t = t.replace("\n", " ")[:60]
    scope = boundary_scope(t, visibility)
    source = "他在群里说的" if visibility == VIS_GROUP else "他说的"
    return f"- [{scope}] {t}（{when.strftime('%m-%d')}，{source}）"


def identity_payload(text: str) -> tuple[str, str]:
    """解析身份类指令：返回 (目标, 内容)。

    目标 core_me / core_her / unknown；内容是一句可直接写进核心的话。
    只认「改名 / 以后叫我」这类祈使句；问句（你叫什么、你是谁）返回 unknown，
    留给正常聊天回答，绝不能拿去覆盖核心。
    """
    t = (text or "").strip()
    if not t or _IDENTITY_QUESTION.search(t):
        return "unknown", t
    # 先判「别叫我」，否则会被下面的「叫我 X」抢先匹配成相反意思。
    m = re.search(r"别叫我\s*[「\"']?([^」\"'，。！？\s]{1,12})", t)
    if m:
        return "core_me", f"别叫他「{m.group(1)}」。"
    if re.search(r"(你改名叫|你以后就叫|你就叫|你以后叫)", t):
        m = re.search(r"(?:叫做|名叫|叫)\s*[「\"']?([^」\"'，。！？\s]{1,12})", t)
        return "core_her", f"我叫{m.group(1)}。" if m else t
    m = re.search(
        r"(?:以后|今后|从今以后)?\s*(?:叫我|喊我|称呼我)\s*[「\"']?([^」\"'，。！？\s]{1,12})", t
    )
    if m:
        return "core_me", f"他让我叫他「{m.group(1)}」。"
    return "unknown", t


# ---------------------------------------------------------------- 审计

def _prune_security(path: Path, days: int) -> None:
    """安全日志按天瘦身：只保留最近 `days` 天的行。

    文件头（schema 注释 + 表头 + 分隔线）永远留着，只删过期数据行。
    200 KB 以下不动——每次往群/私聊里写一行都重写一遍整表太亏，
    攒到一定体量再一次清干净（`security_days` 这个旋钮以前只挂在设置页上）。
    """
    try:
        if path.stat().st_size < 200_000:
            return
        text = read_text(path)
    except OSError:
        return
    head: list[str] = []
    rows: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|"):
            rows.append(s)
        elif not rows:
            head.append(line)
    if len(rows) <= 2:
        return
    cutoff = (datetime.now() - timedelta(days=max(1, int(days)))).strftime("%Y-%m-%d")
    keep = [rows[0], rows[1]]
    for row in rows[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if cells and cells[0][:10] >= cutoff:
            keep.append(row)
    if len(keep) == len(rows):
        return
    try:
        atomic_write(path, "\n".join(head + keep) + "\n")
    except OSError:
        pass


def log_security(
    cfg: Config,
    when: datetime,
    sender_id: str | None,
    session_type: str,
    group_id: str | None,
    text: str,
    refused: str,
) -> None:
    path = cfg.security_path
    if not path.exists():
        ensure_dir(path.parent)
        path.write_text(
            "<!-- schema: 1 | 安全审计，留 30 天 -->\n# 安全\n\n"
            + table(
                ["时间", "谁", "在哪", "原话", "拒了什么"],
                [],
            ),
            encoding="utf-8",
        )
    where = "私聊" if session_type == SESSION_PRIVATE else f"群{group_id or ''}"
    row = f"| {fmt_stamp(when)} | {md_cell(sender_id or '未知')} | {md_cell(where)} | {md_cell(text[:80])} | {md_cell(refused)} |"
    append_line(path, row)
    _prune_security(path, int(getattr(cfg.knobs, "security_days", 30) or 30))


def recent_security(cfg: Config, limit: int = 20) -> list[str]:
    lines = [ln for ln in read_text(cfg.security_path).splitlines() if ln.strip().startswith("|")]
    return lines[-limit:]
