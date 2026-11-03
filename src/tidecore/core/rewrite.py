"""查询改写：把口语问法换成记忆里更可能出现的词。

只用于检索，不改写用户原话，也不进提示词。触发条件刻意收窄：
本地检索命中不足 2 条、或者问法里几乎没有实词时，才多花一次模型调用。
失败、超时、返回空，一律退回原句——改写不该让整轮变慢或变错。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 只为类型；运行时不能 import brain（brain 反过来 import assemble）
    from .brain import LLM

MODES = ("off", "weak", "always")

# 纯寒暄不值得改写：改出来的词也不会有检索价值
_TRIVIAL = re.compile(
    r"^(在吗|在不在|你在吗|你好|您好|嗨|hi|hello|嗯+|哦+|好|好的|行|晚安|早安|早上好|晚上好|"
    r"哈+|呵+|嘿+|睡了?吗|吃了吗|人呢)[？?！!。.~\s]*$",
    re.I,
)

SYSTEM = (
    "你在帮一个聊天机器人查记忆。把用户这句话改写成检索用的关键词，不要回答问题。\n"
    "规则：\n"
    "- 只输出 JSON：{\"keywords\":[\"词1\",\"词2\",\"词3\"]}，3–5 个，不要解释。\n"
    "- 保留人名、项目名、时间词；不要编造上文没有的实体。\n"
    "- 上文只用来解开「那件事 / 上次那个 / 他说的」；跟这句话无关的上文不要写进关键词。\n"
    "- 同义词换成聊天原文里更常见的说法（对象→女朋友，加班→工作忙，破插件→插件 报错）。\n"
)


_NOISE = {"还记得吗", "还记得", "上次", "那个", "这个", "怎么样", "什么", "事情", "那事", "那个事", "那件事"}

# 只有带指代的问法才给上文：不然「我那个破插件」也会被上文里的「我妈」带偏。
_REFER = re.compile(r"(那件事|那个事|那事|这件事|这个事|这事|上次|之前说|前面说|他说的|你说的|刚才说|那家)")


def use_context(query: str) -> bool:
    return bool(_REFER.search(query or ""))


def mode_of(cfg) -> str:
    value = str(getattr(cfg.knobs, "rewrite_query_mode", "weak") or "weak").strip().lower()
    return value if value in MODES else "weak"


def should_rewrite(mode: str, query: str, matched: int, terms: int, top_score: float = 1.0) -> bool:
    if mode not in ("weak", "always"):
        return False
    text = (query or "").strip()
    if not text or _TRIVIAL.match(text):
        return False
    if len(re.sub(r"\W", "", text)) < 2:
        return False
    if mode == "always":
        return True
    # 一条都没命中、或者命中的那条很弱（分数低）、或者问法几乎没有实词：
    # 这三种才是「本地检索基本没戏」，值得多花一次调用。
    return matched == 0 or terms < 2 or top_score < 0.5


def rewrite(llm: LLM | None, query: str, context: str = "", timeout: int = 12) -> list[str]:
    """返回 3–5 个检索词；任何异常都返回空列表。"""
    if not llm or not llm.available or not (query or "").strip():
        return []
    if context and not use_context(query):
        context = ""
    user = (f"上文：{context[:300]}\n" if context else "") + f"这句话：{query.strip()[:200]}"
    try:
        data = llm.chat_json(
            SYSTEM,
            user,
            temperature=0.0,
            max_tokens=140,
            timeout=timeout,
            thinking="disabled",
            tag="rewrite",
        )
    except Exception:  # noqa: BLE001 改写是加分项，任何失败都退回原句
        return []
    if isinstance(data, dict):
        raw = data.get("keywords") or data.get("terms") or []
    elif isinstance(data, list):
        raw = data
    else:
        raw = []
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        word = re.sub(r"[\s,，。；;、]+", " ", str(item)).strip()
        if word and word not in out and word not in _NOISE:
            out.append(word)
    return out[:5]
