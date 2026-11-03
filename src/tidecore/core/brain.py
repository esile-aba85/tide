"""模型接入与出口清洗。

- 真模型：OpenAI 兼容 /chat/completions，标准库实现，不加依赖。
- 离线：模型没接上时，只用装配好的记忆说一句，明确标「离线」，不假装是她。
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .assemble import Assembly
from .config import Config, LLMConfig
from .util import strip_channel_tag
from . import usage


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.conf: LLMConfig = cfg.llm()

    @property
    def available(self) -> bool:
        return bool(self.conf.enabled and self.conf.api_key)

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        thinking: str | None = None,
        tag: str = "chat",
    ) -> str:
        if not self.available:
            raise LLMError("没有配置模型 key")
        if not self.conf.api_base.startswith(("http://", "https://")):
            raise LLMError(f"api_base 不是合法地址：{self.conf.api_base[:40]}")
        payload = {
            "model": self.conf.model,
            "messages": messages,
            "temperature": self.conf.reply_temperature if temperature is None else temperature,
            "max_tokens": self.conf.reply_max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        # deepseek-flash 默认开思考模式：短任务（改写）要关掉，不然 token 全花在 CoT 上
        if thinking in ("enabled", "disabled"):
            payload["thinking"] = {"type": thinking}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.conf.api_base + "/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.conf.api_key}",
            },
            method="POST",
        )
        last_err: Exception | None = None
        for attempt in range(2):
            try:
                began = time.time()
                with urllib.request.urlopen(req, timeout=timeout or self.conf.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                msg = (data.get("choices") or [{}])[0].get("message") or {}
                content = (msg.get("content") or "").strip()
                # 记账：命中多少、未命中多少，按 usage 记一行。写不进去就算了，不影响回话。
                try:
                    usage.record(
                        self.cfg.root,
                        self.conf.model,
                        tag,
                        data.get("usage") or {},
                        ms=(time.time() - began) * 1000,
                    )
                except Exception:  # noqa: BLE001 电表坏了不能拖垮聊天
                    pass
                if content:
                    return content
                # 有些推理模型把话全花在 reasoning 上；不拿 CoT 当回复
                raise LLMError("模型返回了空内容（可能 token 用在了推理上）")
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "ignore")[:300]
                last_err = LLMError(f"HTTP {e.code}: {detail}")
                if e.code in (429, 500, 502, 503, 504) and attempt == 0:
                    time.sleep(1.2)
                    continue
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = LLMError(f"连不上模型：{e}")
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break
            except ValueError as e:
                last_err = LLMError(f"模型返回不是 JSON：{e}")
                break
        raise last_err or LLMError("模型调用失败")

    def chat_json(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        thinking: str | None = None,
        tag: str = "json",
    ) -> dict | list | None:
        raw = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self.conf.consolidate_temperature if temperature is None else temperature,
            max_tokens=self.conf.consolidate_max_tokens if max_tokens is None else max_tokens,
            timeout=timeout,
            thinking=thinking,
            tag=tag,
        )
        return parse_json_loose(raw)


def parse_json_loose(raw: str):
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


_STAGE = re.compile(r"[（(][^（）()\n]{1,10}[）)]")
_MD = re.compile(r"^[#>*\-\s]+", re.M)


def clean_reply(text: str) -> str:
    text = (text or "").replace("\r", "")
    offline = text.lstrip().startswith("（离线）")
    if offline:
        text = text.lstrip()[len("（离线）") :]
    text = re.sub(r"^```.*?\n|```$", "", text, flags=re.S)
    text = strip_channel_tag(text)
    text = _STAGE.sub("", text)
    text = _MD.sub("", text)
    text = re.sub(r"[ \t]*\n+[ \t]*", " ", text)
    text = re.sub(r"^(小潮|我)[：:]\s*", "", text)
    text = text.strip().strip("\"“”")
    text = re.sub(r"\s{2,}", " ", text)
    text = text[:400].strip()
    return (("（离线）" if offline else "") + text).strip()


def split_bubbles(text: str, limit: int = 3, max_chars: int = 80) -> list[str]:
    """她连发几条：一行一条消息。

    聊天不是写作文，一句一句发才像人。模型本来就会换行，以前被 clean_reply
    压成一行了 —— 这里先按换行拆，再逐条清洗。没有换行就还是原来的一条。
    最多 limit 条；超了把剩下的并进最后一条，免得刷屏。
    """
    raw = (text or "").replace("\r", "")
    blocks = [b for b in re.split(r"\n+", raw) if b.strip()]
    if len(blocks) <= 1:
        one = clean_reply(raw)
        return [one] if one else []
    parts = [clean_reply(b) for b in blocks]
    parts = [p for p in parts if p]
    if not parts:
        return []
    if len(parts) > limit:
        parts = parts[: limit - 1] + [" ".join(parts[limit - 1 :])]
    return [p[:max_chars] for p in parts]


def offline_reply(assembly: Assembly, reason: str = "") -> str:
    meta = assembly.meta
    facts = meta.get("facts") or []
    if facts:
        line = re.sub(r"^你之前说过——|^我记得好像是——|^（我记得不太清）", "", facts[0])
        return f"（离线）我记得你说过：{line}。\n模型没接上，我先只照记忆说。"
    if meta.get("diary"):
        return "（离线）我今天没什么特别的，就在这儿待着。模型没接上，先这样。"
    return "（离线）我这会儿没接上模型，先只能陪你说这一句。"


@dataclass
class ReplyResult:
    text: str
    offline: bool = False
    error: str = ""
    rewritten: bool = False
    raw: str = ""          # 模型原话（保留换行，用来拆成一条一条发）


def generate_reply(
    cfg: Config,
    assembly: Assembly,
    last_replies: list[str] | None = None,
    offline: bool = False,
) -> ReplyResult:
    llm = LLM(cfg)
    if offline or not llm.available:
        return ReplyResult(offline_reply(assembly), offline=True)

    # assemble 已经把请求排好了（人设+死规矩在最前，记忆和这一轮的规矩贴着这句话），
    # 这里不要再自己拼 system —— 顺序一变，前缀缓存就废了。
    messages = list(assembly.messages)
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": assembly.system}] + messages
    try:
        raw = llm.chat(messages)
    except LLMError as e:
        return ReplyResult(offline_reply(assembly), offline=True, error=str(e))

    text = clean_reply(raw)
    if not text:
        return ReplyResult(offline_reply(assembly), offline=True, error="空回复")

    # 自我重复：和她最近 2 条 3-gram 相似度 > 0.75，重写一次，只一次
    last_replies = last_replies or []
    from .util import jaccard, ngrams

    if any(jaccard(ngrams(text, 3), ngrams(old, 3)) > 0.75 for old in last_replies if old):
        try:
            fix = (
                "你刚才的回答和前面太像了。换一个角度、换一种说法，只回一句到三句，"
                "不要解释，不要重复原句。\n\n你刚才想说的是：" + text
            )
            raw2 = llm.chat(messages + [{"role": "assistant", "content": text}, {"role": "user", "content": fix}])
            text2 = clean_reply(raw2)
            if text2 and not any(jaccard(ngrams(text2, 3), ngrams(old, 3)) > 0.75 for old in last_replies if old):
                return ReplyResult(text2, rewritten=True, raw=raw2)
        except LLMError:
            pass
    return ReplyResult(text, raw=raw)
