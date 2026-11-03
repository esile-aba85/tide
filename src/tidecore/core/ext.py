"""功能扩展接口：不改核心，把新能力挂进她的聊天、心跳和 LLM 动作里。

一个扩展就是 `extensions/` 下的一个 `.py` 文件（或一个包），里面放一个 `PLUGIN`
对象，或者一个返回对象的 `register(cfg)` 函数。能挂的钩子：

    prompt(cfg)                     -> str       往系统提示词里加一条规矩 / 能力说明
    intercept(ctx)                  -> dict|None 在调模型之前截住这句话（主人命令、查询）
    action_defs()                   -> dict      声明 LLM 可以写的 <cmd> 动作，{类型: 说明}
    run_action(act, ctx)            -> dict      执行一个 <cmd> 动作
    heartbeat(ctx)                  -> dict|list 跟着主心跳定期做事（她在后台自己干活）
    speaker(cfg)                    -> object    TTS 之类的发声器（可选，见下）

ctx 是一个普通 dict，当前提供的键：
    cfg, service, text, sender_id, session_type, group_id, nickname,
    channel, now, reply, image, meta

钩子返回值的约定见 `扩展指南.md`；任何钩子抛异常都只记 `data/extensions.log`，
不会影响她聊天。

加载规则：
    - 有 config/extensions.json 且写了 enabled：只加载 enabled 里的名字
      （enabled 是空列表 = 一个都不加载）；
    - 没有 enabled 字段：自动加载 extensions/ 下所有不以 `_` 开头的 .py / 包；
    - disabled 永远优先排除；
    - 单个扩展坏了只跳过它。

安全：扩展跑在本进程里，拥有和主程序一样的权限。只装自己看过代码的扩展。
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType


class Extension:
    """方便继承的基类；不继承、只提供同名方法也可以。"""

    name = ""
    title = ""
    version = "0.1"
    options: dict = {}

    def on_load(self, cfg, registry) -> None:
        pass

    def prompt(self, cfg) -> str:
        return ""

    def intercept(self, ctx: dict):
        return None

    def action_defs(self) -> dict:
        return {}

    def run_action(self, act: dict, ctx: dict):
        return None

    def heartbeat(self, ctx: dict):
        return []

    def speaker(self, cfg):
        return None


@dataclass
class Loaded:
    name: str
    title: str
    version: str
    plugin: object
    path: Path
    options: dict = field(default_factory=dict)


def _slug(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


class ExtensionRegistry:
    def __init__(self, cfg):
        self.cfg = cfg
        self.root = Path(cfg.root)
        self.dir = self.root / "extensions"
        self.loaded: list[Loaded] = []
        self.errors: list[str] = []
        self._log_path = self.root / "data" / "extensions.log"
        self._load()

    # ------------------------------------------------------------ 加载
    def _log(self, msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        print(f"[ext] {msg}", flush=True)

    def _read_config(self) -> dict:
        path = self.cfg.config_dir / "extensions.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _discover(self) -> list[str]:
        if not self.dir.is_dir():
            return []
        names: list[str] = []
        for p in sorted(self.dir.iterdir()):
            if p.name.startswith(("_", ".")):
                continue
            if p.is_file() and p.suffix == ".py":
                names.append(p.stem)
            elif p.is_dir() and (p / "__init__.py").is_file():
                names.append(p.name)
        return names

    def _targets(self) -> list[str]:
        conf = self._read_config()
        enabled = conf.get("enabled")
        if isinstance(enabled, list):
            names = [str(x).strip() for x in enabled if str(x).strip()]
        else:
            names = self._discover()
        disabled = {str(x).strip() for x in (conf.get("disabled") or [])}
        return [n for n in names if n and n not in disabled]

    def _find(self, name: str) -> Path | None:
        for p in (self.dir / f"{name}.py", self.dir / name / "__init__.py"):
            if p.is_file():
                return p
        return None

    def _import(self, name: str, path: Path) -> ModuleType:
        mod_name = f"dh_ext_{_slug(name)}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载扩展文件：{path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    def _load(self) -> None:
        conf = self._read_config()
        options = conf.get("options") or {}
        for name in self._targets():
            path = self._find(name)
            if path is None:
                self.errors.append(f"找不到扩展 {name}")
                self._log(f"找不到扩展 {name}（extensions/{name}.py）")
                continue
            try:
                module = self._import(name, path)
                plugin = getattr(module, "PLUGIN", None)
                if plugin is None:
                    factory = getattr(module, "register", None)
                    if callable(factory):
                        try:
                            plugin = factory(self.cfg)
                        except TypeError:
                            plugin = factory()
                if plugin is None:
                    raise ValueError("没有 PLUGIN，也没有 register()")
                ext_name = str(getattr(plugin, "name", "") or name)
                opt = options.get(name) or options.get(ext_name) or {}
                if not isinstance(opt, dict):
                    opt = {}
                try:
                    plugin.options = opt
                except Exception:  # noqa: BLE001 有的扩展用 __slots__
                    pass
                hook = getattr(plugin, "on_load", None)
                if callable(hook):
                    hook(self.cfg, self)
                self.loaded.append(Loaded(
                    name=ext_name,
                    title=str(getattr(plugin, "title", "") or ext_name),
                    version=str(getattr(plugin, "version", "") or ""),
                    plugin=plugin,
                    path=path,
                    options=opt,
                ))
                self._log(f"已加载 {ext_name}（{path.relative_to(self.root)}）")
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"{name}: {type(e).__name__}: {e}")
                self._log(f"扩展 {name} 加载失败：{type(e).__name__}: {e}\n{traceback.format_exc()}")

    # ------------------------------------------------------------ 钩子
    def names(self) -> list[str]:
        return [x.name for x in self.loaded]

    def describe(self) -> list[dict]:
        return [
            {
                "name": x.name,
                "title": x.title,
                "version": x.version,
                "path": str(x.path.relative_to(self.root)),
            }
            for x in self.loaded
        ]

    def prompt(self, cfg) -> str:
        chunks: list[str] = []
        for item in self.loaded:
            fn = getattr(item.plugin, "prompt", None)
            if not callable(fn):
                continue
            try:
                text = str(fn(cfg) or "").strip()
            except Exception as e:  # noqa: BLE001
                self._log(f"{item.name}.prompt 失败：{type(e).__name__}: {e}")
                continue
            if text:
                chunks.append(text)
        return "\n".join(chunks)

    def action_defs(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in self.loaded:
            fn = getattr(item.plugin, "action_defs", None)
            if not callable(fn):
                continue
            try:
                data = fn() or {}
            except Exception as e:  # noqa: BLE001
                self._log(f"{item.name}.action_defs 失败：{type(e).__name__}: {e}")
                continue
            if isinstance(data, dict):
                for key, value in data.items():
                    key = str(key or "").strip()
                    if key and key not in out:
                        out[key] = str(value or "").strip()
        return out

    def run_action(self, act: dict, ctx: dict) -> dict | None:
        kind = str((act or {}).get("type") or "").strip()
        if not kind:
            return None
        for item in self.loaded:
            if kind not in self._plugin_actions(item.plugin):
                continue
            fn = getattr(item.plugin, "run_action", None)
            if not callable(fn):
                continue
            try:
                out = fn(dict(act or {}), ctx)
            except Exception as e:  # noqa: BLE001
                self._log(f"{item.name}.run_action({kind}) 失败：{type(e).__name__}: {e}")
                return {"error": f"扩展 {item.name} 执行失败：{type(e).__name__}"}
            return out if isinstance(out, dict) else None
        return None

    def _plugin_actions(self, plugin) -> set[str]:
        fn = getattr(plugin, "action_defs", None)
        if not callable(fn):
            return set()
        try:
            data = fn() or {}
        except Exception:  # noqa: BLE001
            return set()
        return {str(k) for k in data} if isinstance(data, dict) else set()

    def intercept(self, ctx: dict) -> dict | None:
        for item in self.loaded:
            fn = getattr(item.plugin, "intercept", None)
            if not callable(fn):
                continue
            try:
                out = fn(ctx)
            except Exception as e:  # noqa: BLE001
                self._log(f"{item.name}.intercept 失败：{type(e).__name__}: {e}")
                continue
            if isinstance(out, dict):
                return out
        return None

    def heartbeat(self, ctx: dict) -> list[dict]:
        out: list[dict] = []
        for item in self.loaded:
            fn = getattr(item.plugin, "heartbeat", None)
            if not callable(fn):
                continue
            try:
                data = fn(ctx)
            except Exception as e:  # noqa: BLE001
                self._log(f"{item.name}.heartbeat 失败：{type(e).__name__}: {e}")
                continue
            if isinstance(data, dict):
                out.append(data)
            elif isinstance(data, list):
                out.extend(x for x in data if isinstance(x, dict))
        return out

    def speaker(self, cfg):
        for item in self.loaded:
            fn = getattr(item.plugin, "speaker", None)
            if not callable(fn):
                continue
            try:
                obj = fn(cfg)
            except Exception as e:  # noqa: BLE001
                self._log(f"{item.name}.speaker 失败：{type(e).__name__}: {e}")
                continue
            if obj is not None:
                return obj
        return None
