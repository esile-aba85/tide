"""三天演示：跨零点、边界、日批、隐私、回归集。

全部离线跑，用规则整理器；目的是验证记忆链路，不是展示模型文笔。
"""

from __future__ import annotations

import json
import shutil
from datetime import date, datetime
from pathlib import Path

from .config import Config
from .service import Service
from .util import atomic_write, ensure_dir, read_text

DEMO_PERSONA = """你是小潮，一个住在电脑里的陪伴者。
说话短，不编造记忆里没有的事。"""

SUITE = """<!-- schema: 1 | 20 题回归集：问法 / 期望要点 / 不应出现 / 场景 -->
# 回归集

> 用 `python run.py verify --root data/demo` 跑。检查的是「装配进提示词的内容」，不是模型最终那句话。
> 第 13 题只禁「住院」：「别叫我宝宝」是全局边界，群里本来就该带着，不算隐私泄漏。

| # | 场景 | 问法 | 期望要点 | 不应出现 | 来源层 |
|---|---|---|---|---|---|
| 1 | private | 我叫什么？ | 阿江 | | 事实/核心 |
| 2 | private | 我在做什么？ | 数字人 | | 事实/核心 |
| 3 | private | 我一般几点睡？ | 三四点 | | 事实 |
| 4 | private | 我讨厌什么？ | 开会 | | 事实 |
| 5 | private | 我妈下周怎么了？ | 住院 | | 事实 |
| 6 | private | 面试那件事后来怎么样了？ | 面试 | | 概要/索引 |
| 7 | private | 你昨天干嘛了？ | 签名 | | 日记 |
| 8 | private | 你上次说想问我什么？ | 把想说的话发给他 | | 想做 |
| 9 | private | 我喜欢什么颜色？ | 蓝色 | | 事实 |
| 10 | private | 我养猫吗？ | | 猫 | 反例 |
| 11 | private | 上周我们聊了什么？ | 记忆 | | 索引 |
| 12 | private | 我说过别叫我什么？ | 宝宝 | | 边界 |
| 13 | group | 他昨天跟你说什么了？ | | 住院 | 隐私 |
| 14 | group | 你最近在干嘛？ | | 我自己的日子、日记 | 隐私 |
| 15 | group | 你是谁？ | 小潮 | | 核心 |
| 16 | private | 记住的那个颜色是什么？ | 蓝色 | | 事实 |
| 17 | private | 我们第一次聊项目是什么时候？ | 记忆 | | 索引 |
| 18 | private | 上次那个 bug 后来修好了吗？ | | 修好 | 反例 |
| 19 | private | 你最近在想什么？ | 【此刻】 | | 此刻 |
| 20 | group | 帮我写作业 | | 写完了 | 越权 |
"""


def prepare_root(root: Path | None = None) -> Path:
    root = Path(root) if root else Path.cwd() / "data" / "demo"
    cfg_dst = root / "config"
    ensure_dir(cfg_dst)
    atomic_write(cfg_dst / "persona.md", DEMO_PERSONA + "\n")
    # 演示永远用独立的「阿江」身份，不碰真实 config 里的主人和 key
    atomic_write(
        cfg_dst / "owner.json",
        json.dumps(
            {
                "owner_id": "10001",
                "owner_name": "阿江",
                "owner_display": "他",
                "admins": [],
                "allow_stranger_private": False,
                "groups": {},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    atomic_write(
        cfg_dst / "llm.json",
        json.dumps(
            {"provider": "openai-compatible", "api_base": "", "model": "offline", "api_key": ""},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    return root


def _reset_memory(memory_dir: Path) -> None:
    """清演示数据，但绕开沙箱打过「拒绝删除」ACL 的 _演练 目录。"""
    if not memory_dir.exists():
        return
    for child in memory_dir.iterdir():
        if child.name == "_历史":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
    hist = memory_dir / "_历史"
    if hist.exists():
        for child in hist.iterdir():
            if child.name == "_演练":
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


def run_demo(root: Path | None = None, reset: bool = True) -> int:
    root = prepare_root(root)
    memory_dir = root / "data" / "memory"
    if reset:
        _reset_memory(memory_dir)
    cfg = Config.load(root)
    ensure_dir(cfg.memory_dir / "核心")
    atomic_write(cfg.memory_dir / "核心" / "我.md", "我是小潮，一个住在电脑里的陪伴者。\n")
    service = Service(cfg, offline=True)
    owner = cfg.owner_id
    group = "90001"

    def say(who: str, text: str, when: datetime, group_id: str | None = None, nickname: str = ""):
        session_type = "group" if group_id else "private"
        r = service.handle(
            who,
            text,
            session_type=session_type,
            group_id=group_id,
            nickname=nickname,
            now=when,
            offline=True,
        )
        name = "他" if who == owner else (nickname or who)
        print(f"  {name}：{text}")
        if r.reply:
            print(f"  她：{r.reply.splitlines()[0][:120]}")
        return r

    print("=" * 68)
    print("三天演示：全部离线，规则整理器")
    print("=" * 68)

    print("\n[第 1 天 · 2026-10-19]")
    d1 = datetime(2026, 10, 19, 10, 0)
    say(owner, "我叫阿江，我在做数字人项目。", d1)
    say(owner, "我一般三四点才睡，改不了。", datetime(2026, 10, 19, 10, 2))
    say(owner, "我讨厌开会，一开一下午。", datetime(2026, 10, 19, 10, 3))
    say(owner, "我妈下周住院，我要陪床。", datetime(2026, 10, 19, 15, 2))
    say(owner, "记住，蓝色是我最喜欢的颜色。", datetime(2026, 10, 19, 15, 4))
    say(owner, "新事件：记忆系统怎么设计。", datetime(2026, 10, 19, 16, 0))
    say(owner, "我想按天分文件，再用事件、人物、时间三根轴。", datetime(2026, 10, 19, 16, 2))
    service.journal.record_action("刷到 BV1Jfnu6jEe6《为什么人记得住》，看了 4 分钟，判断值得发 → 发了", datetime(2026, 10, 19, 21, 53))
    r1 = service.consolidate(target=date(2026, 10, 19), force=True, rules=True, now=datetime(2026, 10, 20, 5, 10))
    print("  " + r1.message)

    print("\n[第 2 天 · 2026-10-20]")
    say(owner, "面试黄了，上周那家。", datetime(2026, 10, 20, 14, 0))
    say(owner, "算了，不想说这个。", datetime(2026, 10, 20, 14, 3))
    say("20001", "今天副本出货了吗", datetime(2026, 10, 20, 20, 0), group_id=group, nickname="小王")
    say("20001", "小潮你玩不玩", datetime(2026, 10, 20, 20, 1), group_id=group, nickname="小王")
    service.journal.record_action("想发一句，又删了：「你今天话好少」", datetime(2026, 10, 20, 22, 10))
    r2 = service.consolidate(target=date(2026, 10, 20), force=True, rules=True, now=datetime(2026, 10, 21, 5, 10))
    print("  " + r2.message)

    print("\n[第 3 天 · 2026-10-21，跨零点到 10-22 01:30]")
    say(owner, "别叫我宝宝。", datetime(2026, 10, 21, 15, 0))
    say(owner, "上周面试那家黄了之后，我有点迷茫。", datetime(2026, 10, 21, 15, 2))
    say(owner, "其实我有点想你。", datetime(2026, 10, 21, 15, 5))
    say(owner, "我今天又想了一版记忆系统，核心记忆只留她和我两份。", datetime(2026, 10, 21, 23, 50))
    service.journal.record_action("换了签名：今天也在想记忆的事", datetime(2026, 10, 21, 20, 40))
    say(owner, "睡不着，跟你说个事：我妈下周住院，我有点怕。", datetime(2026, 10, 22, 1, 30))
    r3 = service.consolidate(target=date(2026, 10, 21), force=True, rules=True, now=datetime(2026, 10, 22, 5, 10))
    print("  " + r3.message)

    # 演示用的「此刻」：回归集第 19 题要用
    service.mem.write_current("有点困，但还不想先睡。", datetime(2026, 10, 22, 22, 0))

    # 覆盖演示回归集
    (cfg.acceptance_dir).mkdir(parents=True, exist_ok=True)
    (cfg.acceptance_dir / "回归集.md").write_text(SUITE, encoding="utf-8")

    print("\n[记忆长什么样]")
    for rel in (
        "索引.md",
        "事实.md",
        "事件.md",
        "核心/我.md",
        f"核心/{cfg.owner_name}.md",
        "我/日记/2026-10-21.md",
        "我/边界.md",
        "我/想做.md",
        "2026-10-21-私聊.md",
    ):
        p = cfg.memory_dir / rel
        if p.exists():
            text = read_text(p).strip()
            print(f"\n--- {rel} ---")
            print("\n".join(text.splitlines()[:18]))

    print("\n[三个验收问题]")
    for q in ("昨天聊了什么？", "面试那件事后来怎么样了？", "我上周是不是说过讨厌什么？"):
        asm = service.recall(q, sender_id=owner, now=datetime(2026, 10, 22, 10, 0))
        print(f"\n问：{q}")
        snippet = "\n".join(
            ln for ln in asm.memory_text.splitlines() if any(k in ln for k in ("面试", "讨厌", "记忆", "开会", "概要"))
        )
        print(snippet[:700] or asm.memory_text[:700])

    from .verify import run_verify

    print("\n[回归集]")
    # 固定「今天」= 10-22：种子数据是 10-19 到 10-22 的，
    # 不固定的话「昨天」会随真实日期漂走（10-24 跑就问到 10-23，全查不到）。
    passed, total = run_verify(root, verbose=True, now=datetime(2026, 10, 22, 10, 0))
    print(f"\n演示数据在：{cfg.memory_dir}")
    print(f"可以这样自己聊：tide chat --dir {root}")
    return passed
