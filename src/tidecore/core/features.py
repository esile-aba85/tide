"""可选功能探测：说话 / 唱歌 / 画图 / 生活层。

核心发行版不带这些功能。插件把自己的实现放到同名模块即可；核心只问
`has(name)`，不直接 import 可选模块。
"""

from __future__ import annotations

import importlib
from functools import lru_cache

# 功能名 -> 承载它的模块。缺模块 = 没装，探测返回 False。
FEATURE_MODULES = {
    "voice": "tidecore.core.voice",   # 说话：TTS
    "song": "tidecore.core.song",     # 唱歌 / 练歌 / 清唱
    "draw": "tidecore.core.draw",     # 画图
    "life": "tidecore.core.life",     # 生活层
}

FEATURE_LABELS = {
    "voice": "说话（TTS）",
    "song": "唱歌",
    "draw": "画图",
    "life": "刷 B 站",
}


@lru_cache(maxsize=None)
def has(name: str) -> bool:
    """这个功能装没装。桩模块里 `AVAILABLE = False` 就是没装。"""
    module = FEATURE_MODULES.get(str(name or "").strip().lower())
    if not module:
        return False
    try:
        mod = importlib.import_module(module)
    except Exception:  # noqa: BLE001 模块坏了也当没装，不能拖垮启动
        return False
    return bool(getattr(mod, "AVAILABLE", True))


def installed() -> list[str]:
    return [name for name in FEATURE_MODULES if has(name)]


def missing() -> list[str]:
    return [name for name in FEATURE_MODULES if not has(name)]


def label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)
