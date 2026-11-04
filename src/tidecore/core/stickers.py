"""表情包：存下来，聊天时按语境发一张。

存原图（不压缩），文件名是内容 sha1，天然去重。
每张图配一句描述（存 `index.json`），靠它做「这句话配哪张图」的匹配；
描述由识图模型生成，没标过的老图可以 `插件 stickers-tag` 补。
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

from .util import atomic_write, json_load, ngrams

SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
MOODS = (
    ("开心", r"(开心|高兴|哈哈|笑|太好|喜欢|可爱|乐|兴奋)"),
    ("生气", r"(生气|气死|烦|讨厌|哼|过分|滚|暴怒|吃醋|火大)"),
    ("难过", r"(难过|委屈|想哭|难受|心疼|孤单|失落|疲惫|无奈|心累)"),
    ("困", r"(困|睡|晚安|累|熬夜|疲惫)"),
    ("无语", r"(无语|服了|离谱|麻了|算了|呆滞)"),
)


def folder(cfg) -> Path:
    """cfg 可以是 Config，也可以直接是项目根 Path（设置面板里后者更顺手）。

    注意：Path 自己也有 .root 属性（盘符根），不能用 getattr 兜底，必须先判类型。
    """
    root = Path(cfg) if isinstance(cfg, (str, Path)) else Path(cfg.root)
    path = root / "data" / "stickers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_path(cfg) -> Path:
    return folder(cfg) / "index.json"


def _load(cfg) -> dict:
    raw = json_load(_index_path(cfg), {}) or {}
    if not isinstance(raw, dict):
        raw = {}
    raw.setdefault("items", {})
    raw.setdefault("sent", [])
    return raw


def _save(cfg, data: dict) -> None:
    atomic_write(_index_path(cfg), json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def _ext(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"GIF8"):
        return ".gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def save(cfg, data: bytes, caption: str = "") -> str:
    """存一张，返回路径（已存在就返回原路径）。caption 有就顺手记下。

    超过 8MB 直接拒收：以前是 `write_bytes(data[:8MB])` 截断后落盘，
    文件是残的、digest/索引还是原图的，QQ 发出去就是坏图。
    """
    if not data or len(data) < 200:
        return ""
    if len(data) > 8 * 1024 * 1024:
        print(f"[stickers] 图太大（{len(data) // 1024} KB > 8 MB），不存", flush=True)
        return ""
    digest = hashlib.sha1(data).hexdigest()[:16]
    path = folder(cfg) / (digest + _ext(data))
    if not path.exists():
        path.write_bytes(data)
    if caption and path.exists():
        data_index = _load(cfg)
        item = data_index["items"].setdefault(path.name, {})
        if not item.get("caption"):
            item["caption"] = caption[:160]
            item["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _save(cfg, data_index)
    return str(path)


def tag(cfg, path: str, caption: str = "") -> str:
    """给一张图补描述；不传 caption 就用识图模型看。"""
    p = Path(path)
    if not p.is_file():
        return ""
    text = caption.strip()
    if not text:
        from .vision import describe

        text = describe(cfg, str(p))
    if not text:
        return ""
    data = _load(cfg)
    item = data["items"].setdefault(p.name, {})
    item["caption"] = text[:160]
    item["at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save(cfg, data)
    return text


def tag_pending(cfg, limit: int = 20) -> tuple[int, int]:
    """给还没描述的老图补描述。返回 (成功, 待补总数)。"""
    files = sorted(p for p in folder(cfg).iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    data = _load(cfg)
    pending = [p for p in files if not (data["items"].get(p.name) or {}).get("caption")]
    done = 0
    for p in pending[:limit]:
        if tag(cfg, str(p)):
            done += 1
    return done, max(0, len(pending) - done)


def pick(cfg, rng: random.Random | None = None) -> str:
    files = sorted(p for p in folder(cfg).iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not files:
        return ""
    return str((rng or random).choice(files))


def count(cfg) -> int:
    return len([p for p in folder(cfg).iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES])


# ---------------------------------------------------------------- 设置面板用

def list_items(cfg) -> list[dict]:
    """按时间倒序列出所有表情包，给设置面板的「表情包」页用。"""
    data = _load(cfg)
    out: list[dict] = []
    for p in folder(cfg).iterdir():
        if not p.is_file() or p.suffix.lower() not in SUFFIXES:
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        item = data["items"].get(p.name) or {}
        out.append({
            "name": p.name,
            "path": str(p),
            "caption": str(item.get("caption") or ""),
            "at": str(item.get("at") or ""),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        })
    out.sort(key=lambda x: x["mtime"], reverse=True)
    return out


def delete(cfg, name: str) -> bool:
    """删一张（文件 + 索引条目）。name 只取文件名，挡住 ../ 之类的路径。"""
    p = folder(cfg) / Path(str(name)).name
    removed = False
    try:
        if p.is_file():
            p.unlink()
            removed = True
    except OSError:
        return False
    data = _load(cfg)
    data["items"].pop(p.name, None)
    _save(cfg, data)
    return removed


def clear(cfg) -> int:
    """清空表情包库：只删图片和索引，不动 data/memory 里的图片记忆。"""
    removed = 0
    for p in list(folder(cfg).iterdir()):
        if p.is_file() and p.suffix.lower() in SUFFIXES:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    data = _load(cfg)
    data["items"] = {}
    data["sent"] = []
    _save(cfg, data)
    return removed


def import_file(cfg, src: str | Path) -> str:
    """从任意位置导入一张（按内容 sha1 存，重复导入不会多一份）。"""
    p = Path(src)
    if not p.is_file() or p.suffix.lower() not in SUFFIXES:
        return ""
    try:
        return save(cfg, p.read_bytes())
    except OSError:
        return ""


def _moods(text: str) -> set[str]:
    import re

    return {name for name, pattern in MOODS if re.search(pattern, text or "")}


def _mood(text: str) -> str:
    got = _moods(text)
    return next(iter(got), "")


def _recent(data: dict, now: float, gap: float) -> set[str]:
    out = set()
    for item in data.get("sent") or []:
        try:
            if now - float(item.get("ts") or 0) < gap:
                out.add(str(item.get("name") or ""))
        except (TypeError, ValueError):
            continue
    return out


def pick_for(cfg, text: str, now: float | None = None, rng: random.Random | None = None,
             moods: set[str] | None = None) -> str:
    """按这句话挑一张最贴的；没有够贴的就返回空，不硬发。

    ``moods`` 是她**此刻的心情**（`emotion.mood_tags()`），不是这句话的情绪。
    两样都算分，心情那一样权重更高：她心里堵着的时候，就算这句话在说「好呀」，
    也不该甩一张大笑的图出去。
    """
    now = time.time() if now is None else now
    data = _load(cfg)
    items = {
        name: meta
        for name, meta in (data.get("items") or {}).items()
        if (meta or {}).get("caption")
    }
    if not items:
        return ""
    terms = ngrams(text, 2)
    text_moods = _moods(text)
    mood_hint = {str(m) for m in (moods or ())}
    recent = _recent(data, now, 6 * 3600)
    scored: list[tuple[float, str]] = []
    for name, meta in items.items():
        cap = str(meta.get("caption") or "")
        cap_moods = _moods(cap)
        bonus = 0.25 if (text_moods & cap_moods) else 0.0
        if mood_hint & cap_moods:
            bonus += 0.30
        overlap = len(terms & ngrams(cap, 2)) / max(1, len(terms))
        score = overlap + bonus
        if name in recent:
            score -= 0.6
        scored.append((score, name))
    scored.sort(reverse=True)
    best_score, best_name = scored[0]
    if best_score < 0.18:
        return ""
    path = folder(cfg) / best_name
    return str(path) if path.exists() else ""


def proactive_pick(
    cfg,
    text: str,
    now: float | None = None,
    rng: random.Random | None = None,
    scope: str = "private",
    group_id: str = "",
    moods: set[str] | None = None,
) -> str:
    """她主动配一张图：看概率、间隔、每日上限，再按语境挑。

    scope="private" 用私聊那组参数；scope="group" 用群聊那组，间隔按群算、
    上限按“所有群合计”算，免得几个群同时刷。
    """
    knobs = cfg.knobs
    if not bool(getattr(knobs, "sticker_proactive", True)):
        return ""
    if scope == "group":
        if not bool(getattr(knobs, "sticker_group_enabled", True)):
            return ""
        chance = float(getattr(knobs, "sticker_group_chance", 0.06))
        gap = float(getattr(knobs, "sticker_group_min_gap_min", 120)) * 60
        cap = int(getattr(knobs, "sticker_group_daily_cap", 3))
    else:
        chance = float(getattr(knobs, "sticker_chance", 0.18))
        gap = float(getattr(knobs, "sticker_min_gap_min", 45)) * 60
        cap = int(getattr(knobs, "sticker_daily_cap", 8))
    if count(cfg) <= 0:
        return ""
    now = time.time() if now is None else now
    data = _load(cfg)
    times: list[float] = []        # 当前群（或私聊）：算「两次之间至少隔多久」
    all_times: list[float] = []    # 所有群合计：算「每天最多几张」
    for item in data.get("sent") or []:
        kind = str(item.get("scope") or "private")
        if kind != scope:
            continue
        try:
            ts = float(item.get("ts") or 0)
        except (TypeError, ValueError):
            continue
        all_times.append(ts)
        if scope == "group" and group_id and str(item.get("group") or "") != str(group_id):
            continue
        times.append(ts)
    if any(now - ts < gap for ts in times):
        return ""
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    # 群聊的上限是「所有群合计」（docstring 和设置页都这么写），间隔才是按群算的。
    cap_times = all_times if scope == "group" else times
    if sum(1 for ts in cap_times if time.strftime("%Y-%m-%d", time.localtime(ts)) == today) >= cap:
        return ""
    if (rng or random).random() > chance:
        return ""
    path = pick_for(cfg, text, now=now, rng=rng, moods=moods)
    if not path:
        return ""
    entry = {"name": Path(path).name, "ts": now, "scope": scope}
    if group_id:
        entry["group"] = str(group_id)
    data.setdefault("sent", []).append(entry)
    data["sent"] = data["sent"][-200:]
    _save(cfg, data)
    return path
