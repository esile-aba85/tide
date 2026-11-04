"""配置：主人 ID、人格、旋钮、模型凭据。

原则：
  - 身份只认 ID，不认昵称；
  - 主人 ID 只从 config/owner.json 来，聊天里改不了；
  - 模型 key 不落进项目文件，依次读环境变量 / llm.json；
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .util import PROJECT_ROOT, atomic_write, ensure_dir, json_load, json_load_strict, read_text

DEFAULT_OWNER = {
    "owner_id": "",
    "owner_name": "主人",
    "owner_display": "他",
    "admins": [],
    "allow_stranger_private": False,
    "groups": {},
}

FALLBACK_PERSONA = """你是一个住在电脑里的数字人。你说话短，像聊天不像写文章。
你有自己的情绪和日子，但不编造没发生的事。你不是助手，不是客服。
只使用【核心】【你的记忆】【你自己】里真实存在的内容；没有的就说没有。"""

KNOB_DEFAULTS = {
    # 心跳 / 她自己学歌
    "body_heartbeat_seconds": 5,       # 主循环多久看一次（设置页可改）
    "memory_heartbeat_minutes": 10,    # 多久跑一次 tick()：补日批 / 重写此刻
    "proactive_min_minutes": 20,       # 她多久主动醒一次（下限）
    "proactive_max_minutes": 40,       # 她多久主动醒一次（上限）
    # 主动说话：开关 / 每日上限 / 两次最短间隔 / 概率倍率 / 安静时段
    "proactive_enabled": True,
    "proactive_speak_cap": 5,
    "proactive_min_gap_minutes": 100,
    "proactive_probability_scale": 1.0,
    "proactive_quiet_from": 2,
    "proactive_quiet_to": 8,
    # 面对面心跳：只在面对面形态、摄像头有新鲜画面时自己找话题（不占上面的额度）
    "room_heart_enabled": True,
    "room_heart_warmup_seconds": 45,        # 刚进面对面先看两眼
    "room_heart_min_gap_seconds": 150,      # 两次开口至少隔 2.5 分钟
    "room_heart_min_silence_seconds": 45,   # 你/她刚说完话，先等 45 秒
    "room_heart_same_scene_seconds": 600,   # 画面没变时，10 分钟内不重复
    "room_heart_daily_cap": 24,             # 每天最多开口几次（0 = 关）
    "room_heart_fresh_seconds": 25,         # 画面多少秒内算「现在」
    "room_heart_retry_seconds": 20,         # 条件不满足时多久再看一眼
    "song_habit_enabled": True,
    "song_habit_min_minutes": 180,     # 两次随机学歌之间至少 3 小时
    "song_habit_max_minutes": 480,     # 最多 8 小时
    "song_habit_daily_cap": 1,         # 每天最多学 1 首
    # 她自己画画：心跳闲着时画一张，存进画册；画完可以弹给他看 / 发 QQ
    "draw_habit_enabled": True,
    "draw_habit_min_minutes": 60,      # 两次之间至少 1 小时
    "draw_habit_max_minutes": 240,     # 最多 4 小时
    "draw_habit_daily_cap": 2,         # 每天最多画 2 张
    "draw_habit_idle_minutes": 5,      # 键鼠至少闲 5 分钟才动笔
    "draw_habit_show_pet": True,       # 在电脑前：气泡 + 弹图
    "draw_habit_qq": True,             # 画完发 QQ 私聊（文字 + 图）；关掉就只在电脑上弹
    # 她拉起来的 ComfyUI 闲置多久自动关（分钟，0 = 不关；手动开的不会被自动关）
    "draw_server_idle_stop_minutes": 20,
    "consolidate_hour": 5,
    "diary_hour": 23,
    "core_locked": False,
    "core_her_chars": 200,
    "core_me_chars": 250,
    "resident_budget": 1160,
    "ondemand_budget": 1700,
    "relation_chars": 120,
    "moments_limit": 2,
    "moments_chars": 110,
    # 主动配表情包：概率、间隔、每日上限
    "sticker_proactive": True,
    "sticker_chance": 0.18,
    "sticker_min_gap_min": 45,
    "sticker_daily_cap": 8,
    "sticker_group_enabled": True,
    "sticker_group_chance": 0.06,
    "sticker_group_min_gap_min": 120,
    "sticker_group_daily_cap": 3,
    # 她自己发：签名 / 说说
    "self_signature_enabled": True,
    "self_signature_min_days": 7,
    "self_qzone_enabled": True,
    "self_qzone_chance": 0.08,
    "self_qzone_min_days": 3,
    "self_qzone_month_cap": 8,
    "self_qzone_right": 1,
    # 她自己去刷 B 站（需要模拟器 + B站已登录）
    "life_enabled": True,
    "life_start_hour": 9,
    "life_end_hour": 24,
    "life_gap_min_minutes": 150,
    "life_gap_max_minutes": 270,
    "life_session_min_minutes": 10,
    "life_session_max_minutes": 20,
    "life_video_minutes": 5,
    "life_max_videos": 2,
    "life_sample_minutes": 2,
    "life_look_enabled": True,     # 刷B站时截画面给识图（不只是读标题）
    "life_look_every_sec": 30,     # 每 30 秒看一屏
    "life_look_max": 8,            # 一个视频最多看 8 屏
    "person_chars": 100,
    "self_current_chars": 80,
    "boundary_chars": 170,
    "boundary_max": 10,
    "facts_limit": 6,
    "facts_min": 3,
    "events_limit": 6,
    "overview_days": 3,
    "diary_days": 2,
    "index_rows": 7,
    "wants_limit": 2,
    "thread_tokens": 1200,
    "dup_private_minutes": 10,
    "dup_group_minutes": 3,
    "group_mention_reply_chance": 1.0,   # 群里被 @ 时回不回：1.0 必回，0.2 = 两成概率
    # 熟人在群里说话、没 @ 她，她会不会主动接一句（好感度 + 情绪都在里面算）
    "group_jump_in_enabled": True,
    "group_jump_in_chance": 0.06,        # 基准概率：真熟 + 心情好时还会再乘
    "group_jump_in_min_score": 0.50,     # 好感度够这个分才算「熟人」
    # 60 次/天 ≈ 一个 1000 条/天的群里每 17 分钟插一句，像个活跃成员而不是水军。
    # 钱上无所谓：单次回话实测 ¥0.0034，60 次两毛钱。
    "group_jump_in_daily_cap": 60,       # 一个群一天最多插几次话（天花板，不是目标）
    "group_jump_in_min_gap_min": 8,      # 两次插话至少隔多久（真正的节奏闸）
    "group_jump_in_try_min_minutes": 3,  # 多久才掷一次骰子（群消息量大，别每句都算）
    # 被 @ 时回不回，再按对他的好感打折：不熟/讨厌的人 @ 了也可能不理
    "group_affinity_reply_enabled": True,
    "group_breaker_enabled": True,
    "group_breaker_window_sec": 3,
    "group_breaker_max_replies": 10,
    "group_breaker_cooldown_sec": 2,
    "group_breaker_repeat_window_min": 30,
    "group_breaker_repeat_trips": 3,
    "group_breaker_repeat_cooldown_sec": 600,
    "group_breaker_streak": 3,
    "group_breaker_streak_gap_sec": 30,
    "group_keep_around_min": 3,
    "group_daily_lines": 400,
    "group_summary_minutes": 30,
    "highlight_batch_private": 20,
    "highlight_batch_group": 50,
    "confidence_floor": 0.5,
    "fact_stale_days": 90,
    # 遗忘曲线（艾宾浩斯）：R = e^(-Δt/S)。S 是稳定度，被想起一次就涨一截。
    # 45 天没碰掉到 1/e；重要的事起点更高；上限一年，再牢也会开始淡。
    "forget_enabled": True,
    "forget_stability_days": 45,
    "forget_growth": 2.0,
    "forget_stability_max": 365,
    "forget_floor": 0.05,          # 低于这个可回忆度：没有正经命中就不再注入
    "forget_archive_days": 270,    # 从没被想起、又淡到几乎没有，才进 _历史\事实-遗忘.md
    "forget_archive_below": 0.02,
    "backup_days": 30,
    "security_days": 30,
    "wants_expire_days": 14,
    "event_quiet_days": 7,         # 事件多久没更新算「搁置」（代码冷却，不靠模型）
    "event_stale_days": 30,        # 搁置/进行中再拖这么久就自动结案
    "event_note_max": 12,          # 一条事件最多留几条备注（点歌那种会攒几十条）
    "event_archive_days": 30,
    "person_archive_days": 30,
    "people_max": 300,
    "self_edit_min_days": 2,
    "relation_max_chars": 320,
    # 记忆体检：红线。交接是硬上限（超了 check 不过），其余超了告警。
    "handover_max_kb": 4,
    "memory_file_max_kb": 25,
    "facts_max_rows": 200,
    "events_open_max": 30,
    "raw_total_max_mb": 200,
    "fact_archive_days": 180,
    # 查询改写：off / weak（本地检索不够时才改写）/ always（每轮）
    "rewrite_query_mode": "weak",
}


class Knobs:
    def __init__(self, data: dict | None = None):
        self._d = dict(KNOB_DEFAULTS)
        self._d.update({k: v for k, v in (data or {}).items() if k in KNOB_DEFAULTS})
        for k, v in (data or {}).items():
            self._d.setdefault(k, v)

    def __getattr__(self, name):
        try:
            return self._d[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def as_dict(self) -> dict:
        return dict(self._d)


@dataclass
class LLMConfig:
    enabled: bool
    api_key: str
    api_base: str
    model: str
    timeout: int = 90
    reply_temperature: float = 0.85
    reply_max_tokens: int = 800
    consolidate_temperature: float = 0.2
    consolidate_max_tokens: int = 2400
    rewrite_query: bool = True
    source: str = ""

    def public(self) -> dict:
        return {
            "enabled": self.enabled,
            "api_base": self.api_base,
            "model": self.model,
            "key_source": self.source,
        }


@dataclass
class VisionConfig:
    """识图用的模型：默认走本地 llama.cpp，连不上再回退主模型。

    `llm.json` 里加一段 `vision` 就生效；不加则一切照旧（主模型识图）。
    """

    enabled: bool
    api_base: str
    model: str
    api_key: str
    max_tokens: int = 500
    timeout: int = 60
    fallback: bool = True
    think: bool = False
    use_for_images: bool = False   # 单图默认仍走云端：2B 会把发色、小物件看错
    use_for_video: bool = True     # 视频抽帧用本地：多帧一次问，云端又慢又贵
    source: str = ""

    def public(self) -> dict:
        return {
            "enabled": self.enabled,
            "api_base": self.api_base,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "fallback": self.fallback,
            "think": self.think,
            "use_for_images": self.use_for_images,
            "use_for_video": self.use_for_video,
            "key_source": self.source,
        }


@dataclass
class Config:
    root: Path
    memory_dir: Path
    config_dir: Path
    owner_id: str
    owner_name: str
    owner_display: str
    admins: set[str]
    allow_stranger_private: bool
    groups: dict[str, str]
    knobs: Knobs
    persona: str
    llm_raw: dict
    _llm: LLMConfig | None = field(default=None, repr=False)
    _vision: VisionConfig | None = field(default=None, repr=False)

    # ------------------------------------------------------------ 路径
    @property
    def core_dir(self) -> Path:
        return self.memory_dir / "核心"

    @property
    def self_dir(self) -> Path:
        return self.memory_dir / "我"

    @property
    def diary_dir(self) -> Path:
        return self.self_dir / "日记"

    @property
    def people_dir(self) -> Path:
        return self.memory_dir / "人"

    @property
    def month_dir(self) -> Path:
        return self.memory_dir / "月"

    @property
    def history_dir(self) -> Path:
        return self.memory_dir / "_历史"

    @property
    def hot_dir(self) -> Path:
        return self.memory_dir / "_热"

    @property
    def acceptance_dir(self) -> Path:
        return self.memory_dir / "_验收"

    @property
    def index_db(self) -> Path:
        return self.memory_dir / "_索引.db"

    @property
    def status_path(self) -> Path:
        return self.memory_dir / "_状态.md"

    @property
    def bad_path(self) -> Path:
        return self.memory_dir / "_坏行.md"

    @property
    def security_path(self) -> Path:
        return self.memory_dir / "_安全.md"

    @property
    def pending_path(self) -> Path:
        return self.memory_dir / "_待覆盖.md"

    @property
    def handover_path(self) -> Path:
        """`我\交接.md`：≤4 KB 的「上次 / 当前项目 / 变更 / 下一步卡点 / 全貌」。"""
        return self.self_dir / "交接.md"

    @property
    def audit_dir(self) -> Path:
        return self.memory_dir / "_审计"

    @property
    def audit_path(self) -> Path:
        """写盘审计：一行一个 JSON，谁在什么时候改了哪个文件。"""
        return self.audit_dir / "写入.jsonl"

    def group_name(self, group_id: str | None) -> str:
        if not group_id:
            return ""
        return self.groups.get(str(group_id), f"群{group_id}")

    def is_owner(self, sender_id: str | None) -> bool:
        return str(sender_id or "") == str(self.owner_id)

    def is_admin(self, sender_id: str | None) -> bool:
        return str(sender_id or "") in {str(a) for a in self.admins}

    def save_knobs(self, patch: dict) -> dict:
        """只认 KNOB_DEFAULTS 里有的键；写回 knobs.json 并热更新内存。

        底稿读不出来（BOM/写坏/被占用）时**这次不保存**：以前拿 `{}` 当底稿，
        在设置页随便改一项就把另外 100 多个键整份冲掉，而且没有备份。
        """
        clean = {k: v for k, v in (patch or {}).items() if k in KNOB_DEFAULTS}
        if not clean:
            return {}
        path = self.config_dir / "knobs.json"
        raw = json_load_strict(path)
        if raw is None:
            print(f"[config] {path.name} 读不出来，这次改动没保存（原件已备份成 .bad-*）", flush=True)
            return {}
        raw.update(clean)
        atomic_write(path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        self.knobs = Knobs(raw)
        return {k: getattr(self.knobs, k) for k in clean}

    # ------------------------------------------------------------ 模型
    def llm(self, offline: bool = False) -> LLMConfig:
        if self._llm is None:
            self._llm = resolve_llm(self.llm_raw, self.config_dir, offline=offline)
        if offline:
            return LLMConfig(
                enabled=False,
                api_key="",
                api_base=self._llm.api_base,
                model=self._llm.model,
                timeout=self._llm.timeout,
                source="offline",
            )
        return self._llm

    def vision(self) -> VisionConfig:
        """本地识图模型。没配 `vision` 段就返回 disabled（调用方回退主模型）。"""
        if self._vision is None:
            self._vision = resolve_vision(self.llm_raw, self._llm or self.llm())
        return self._vision

    # ------------------------------------------------------------ 构造
    @classmethod
    def load(cls, root: str | Path | None = None, offline: bool = False) -> Config:
        root = Path(root).resolve() if root else PROJECT_ROOT
        config_dir = root / "config"
        memory_dir = root / "data" / "memory"

        owner = dict(DEFAULT_OWNER)
        owner.update(json_load(config_dir / "owner.json", {}) or {})
        knobs = Knobs(json_load(config_dir / "knobs.json", {}) or {})
        llm_raw = json_load(config_dir / "llm.json", {}) or {}

        persona = read_text(config_dir / "persona.md", "").strip() or FALLBACK_PERSONA

        admins = owner.get("admins") or []
        if isinstance(admins, str):
            admins = [admins]
        groups = owner.get("groups") or {}
        if not isinstance(groups, dict):
            groups = {}

        cfg = cls(
            root=root,
            memory_dir=memory_dir,
            config_dir=config_dir,
            owner_id=str(owner.get("owner_id", "")),
            owner_name=str(owner.get("owner_name", "主人")),
            owner_display=str(owner.get("owner_display", "他")),
            admins={str(a) for a in admins},
            allow_stranger_private=bool(owner.get("allow_stranger_private", False)),
            groups={str(k): str(v) for k, v in groups.items()},
            knobs=knobs,
            persona=persona,
            llm_raw=llm_raw,
        )
        cfg._llm = resolve_llm(llm_raw, config_dir, offline=offline)
        cfg._vision = resolve_vision(llm_raw, cfg._llm)
        return cfg

    @classmethod
    def for_test(cls, root: str | Path, owner_id: str = "10001", owner_name: str = "Owner") -> Config:
        root = Path(root).resolve()
        cfg = cls(
            root=root,
            memory_dir=root / "memory",
            config_dir=root / "config",
            owner_id=str(owner_id),
            owner_name=owner_name,
            owner_display="他",
            admins=set(),
            allow_stranger_private=False,
            groups={"90001": "测试群"},
            knobs=Knobs({}),
            persona="你是一个住在电脑里的陪伴者。说话短，不编造记忆里没有的事。",
            llm_raw={"model": "offline", "api_base": ""},
        )
        cfg._llm = LLMConfig(enabled=False, api_key="", api_base="", model="offline", source="test")
        cfg._vision = VisionConfig(False, "", "", "", source="test")
        ensure_dir(cfg.memory_dir)
        ensure_dir(cfg.config_dir)
        return cfg


def resolve_llm(raw: dict, config_dir: Path, offline: bool = False) -> LLMConfig:
    if offline:
        return LLMConfig(False, "", "", "offline", source="offline")

    api_key = str(raw.get("api_key") or "").strip()
    source = "config/llm.json" if api_key else ""

    if not api_key:
        env_names = [raw.get("api_key_env") or "TIDE_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
        for name in env_names:
            val = os.environ.get(name, "").strip()
            if val:
                api_key = val
                source = f"env:{name}"
                break

    api_base = str(raw.get("api_base") or "https://api.deepseek.com/v1").rstrip("/")
    model = str(raw.get("model") or "deepseek-chat").strip()

    if api_base and not api_base.startswith(("http://", "https://")):
        # 常见手误：把 key 填进了 api_base
        return LLMConfig(
            enabled=False,
            api_key="",
            api_base=api_base,
            model=model,
            source="api_base 非法（可能填成了 key）",
        )

    return LLMConfig(
        enabled=bool(api_key and api_base),
        api_key=api_key,
        api_base=api_base,
        model=model,
        timeout=int(raw.get("timeout", 90)),
        reply_temperature=float(raw.get("reply_temperature", 0.85)),
        reply_max_tokens=int(raw.get("reply_max_tokens", 800)),
        consolidate_temperature=float(raw.get("consolidate_temperature", 0.2)),
        consolidate_max_tokens=int(raw.get("consolidate_max_tokens", 2400)),
        rewrite_query=bool(raw.get("rewrite_query", True)),
        source=source,
    )


def resolve_vision(raw: dict, llm: LLMConfig | None) -> VisionConfig:
    """`llm.json` 的 `vision` 段。

    - 没写：enabled=False，vision.py 照旧用主模型；
    - 写了 model：默认连本机 llama.cpp（127.0.0.1:8790），key 随意；
    - offline（测试）：一律关掉，别让单测去打网络。
    """
    if llm is not None and llm.source == "offline":
        return VisionConfig(False, "", "", "", source="offline")
    v = raw.get("vision") or {}
    if not isinstance(v, dict) or not v:
        return VisionConfig(False, "", "", "", source="no vision block")
    model = str(v.get("model") or "").strip()
    api_base = str(v.get("api_base") or "http://127.0.0.1:8790/v1").rstrip("/")
    if not model:
        return VisionConfig(False, api_base, "", "", source="vision 缺 model")
    if api_base and not api_base.startswith(("http://", "https://")):
        return VisionConfig(False, api_base, "", "", source="vision.api_base 非法")
    return VisionConfig(
        enabled=True,
        api_base=api_base,
        model=model,
        api_key=str(v.get("api_key") or "EMPTY").strip() or "EMPTY",
        max_tokens=int(v.get("max_tokens", 500)),
        timeout=int(v.get("timeout", 60)),
        fallback=bool(v.get("fallback", True)),
        think=bool(v.get("think", False)),
        use_for_images=bool(v.get("use_for_images", False)),
        use_for_video=bool(v.get("use_for_video", True)),
        source="config/llm.json#vision",
    )
