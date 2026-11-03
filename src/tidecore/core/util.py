"""小工具：原子写、追加、markdown 表、清洗。全部只用标准库。

这里的所有函数都不认识「记忆」，只负责把文件写稳。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
from datetime import date, datetime
from pathlib import Path

# Default data root: explicit `TIDE_ROOT`, else the current working directory.
# Callers that know their root (Config.load, CLI --dir) should pass it explicitly.
PROJECT_ROOT = Path(os.environ.get("TIDE_ROOT", Path.cwd())).resolve()

# 读坏了 / 写坏了要留证据：同一个文件只备份一次，别每轮都复制一份。
_BAD_ONCE: set[str] = set()


def _warn(msg: str) -> None:
    """配置 / 记忆文件出问题时留一行话。别静默 —— 静默丢配置最难查。"""
    try:
        print(f"[util] {msg}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 打印失败也不能影响主流程
        pass


def _backup_bad(path: Path) -> Path | None:
    """把读坏的文件原样留一份 `<名字>.bad-时间戳`，好让人能捡回来。"""
    key = str(path)
    if key in _BAD_ONCE:
        return None
    _BAD_ONCE.add(key)
    try:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = path.with_name(f"{path.name}.bad-{stamp}")
        dest.write_bytes(path.read_bytes())
        _warn(f"{path.name} 解析失败，原件已存为 {dest.name}（这次运行不再重复备份）")
        return dest
    except OSError as e:
        _warn(f"{path.name} 解析失败，备份也失败：{type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------- 文件

def ensure_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def read_text(path: Path, default: str = "") -> str:
    """读文本。

    - `utf-8-sig`：手改配置时 PowerShell 的 `Set-Content -Encoding UTF8` 会加 BOM，
      带 BOM 的 json 会让 `json.loads` 直接抛错、带 BOM 的 md 会把 U+FEFF 带进提示词。
    - Windows 上并发读改写时 `os.replace` 会短暂占住文件，读这边重试几次再放弃。
    """
    p = Path(path)
    for attempt in range(4):
        try:
            return p.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return default
        except UnicodeDecodeError:
            try:
                return p.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                return default
        except PermissionError:
            if attempt == 3:
                _warn(f"读不了 {p.name}（被占用），这次按空值处理")
                return default
            time.sleep(0.05 * (attempt + 1))
        except OSError as e:
            _warn(f"读不了 {p.name}：{type(e).__name__}: {e}")
            return default
    return default


def atomic_write(path: Path, text: str) -> None:
    """同目录 tmp + os.replace。崩在写一半也不会留下半份文件。

    Win32 上目标文件只要还被别的进程/线程以读方式打开，`os.replace` 就会
    `WinError 5`（拒绝访问）。读窗口通常只有几毫秒，退避重试几次基本都能过去；
    实在过不去就把异常抛给调用方（它才知道这一笔该不该算失败）。
    """
    path = Path(path)
    ensure_dir(path.parent)
    last: Exception | None = None
    for attempt in range(5):
        fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last = e
            time.sleep(0.04 * (attempt + 1))
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
    assert last is not None  # noqa: S101 只为让类型检查器安静，下面立刻抛
    raise last


def append_text(path: Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def append_line(path: Path, line: str) -> None:
    append_text(path, line.rstrip("\n") + "\n")


def exists_nonempty(path: Path) -> bool:
    try:
        return Path(path).is_file() and Path(path).stat().st_size > 0
    except OSError:
        return False


def copy_file(src: Path, dst: Path) -> None:
    ensure_dir(Path(dst).parent)
    Path(dst).write_bytes(Path(src).read_bytes())


def mtime(path: Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


# ---------------------------------------------------------------- 文本

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def one_line(text: str) -> str:
    return _WS.sub(" ", (text or "").replace("\r", " ").replace("\n", " ")).strip()


def sanitize(text: str, limit: int = 2000) -> str:
    """进原文前的一道清洗：去控制符、压成一行、抹掉明显的凭据。"""
    text = _CTRL.sub("", text or "")
    text = one_line(text)
    return redact(text)[:limit].strip()


_SECRET_PATTERNS = [
    # 裸 key：整段替掉（以前 split(":") 切不出前缀，整段原样留下，等于没脱敏）
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"), lambda m: "[已隐去]"),
    (
        re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret)\b\s*[:=：]\s*\S+"),
        lambda m: m.group(1) + "：[已隐去]",
    ),
    (
        re.compile(r"(?i)(密码|口令|密钥|令牌)\s*[:=：]\s*\S+"),
        lambda m: m.group(1) + "：[已隐去]",
    ),
]


def redact(text: str) -> str:
    """抹掉明显的凭据。`key=value` 留键名，裸 key 整段消掉。"""
    out = text or ""
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


def clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[: max(0, n - 1)].rstrip() + "…"


def chars(text: str) -> int:
    """预算按可见字符算，不把空白算进去。"""
    return len(re.sub(r"\s", "", text or ""))


def normalize_question(text: str) -> str:
    """重复检测用：去空白/标点/emoji、全角转半角、英文小写。"""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
    return text


def ngrams(text: str, n: int = 3) -> set[str]:
    text = normalize_question(text)
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def md_cell(text: str) -> str:
    """表格单元格里不能有换行和裸竖线。"""
    return one_line(str(text)).replace("|", "｜").strip()


def split_md_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


_TABLE_SEP = re.compile(r"^:?-{2,}:?$")


def parse_table(text: str, expected: int | None = None) -> tuple[list[dict], list[tuple[int, str]]]:
    """解析 markdown 表，返回 (行, 坏行[(行号, 原文)])。

    第一行当表头；分隔行跳过；列数不对的进坏行，绝不整体崩。
    """
    rows: list[dict] = []
    bad: list[tuple[int, str]] = []
    header: list[str] | None = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = split_md_row(line)
        if all(_TABLE_SEP.match(c) for c in cells if c):
            continue
        if header is None:
            header = cells
            continue
        if expected is not None and len(cells) != expected:
            bad.append((lineno, raw))
            continue
        if len(cells) != len(header):
            bad.append((lineno, raw))
            continue
        rows.append(dict(zip(header, cells)))
    return rows, bad


def table(headers: list[str], rows: list[list[str]]) -> str:
    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(md_cell(c) for c in row) + " |"

    out = [fmt(headers), "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend(fmt(r) for r in rows)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- markdown 小结构

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def sections(text: str, level: int = 2) -> dict[str, str]:
    """按 `## 标题` 切段，返回 {标题: 正文}。重复标题后者覆盖。"""
    out: dict[str, str] = {}
    current: str | None = None
    buf: list[str] = []
    for line in (text or "").splitlines():
        m = _HEADING.match(line)
        if m and len(m.group(1)) == level:
            if current is not None:
                out[current] = "\n".join(buf).strip()
            current = m.group(2).strip()
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        out[current] = "\n".join(buf).strip()
    return out


def get_section(text: str, title: str, level: int = 2) -> str:
    return sections(text, level).get(title, "")


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text or "", flags=re.S)


def first_sentences(text: str, n: int = 2) -> str:
    parts = re.split(r"(?<=[。！？!?\.])\s*", strip_comments(text).strip())
    parts = [p.strip() for p in parts if p.strip()]
    return "".join(parts[:n])


# ---------------------------------------------------------------- 时间/JSON

def now_local() -> datetime:
    return datetime.now()


def fmt_date(d: date | datetime) -> str:
    return d.strftime("%Y-%m-%d")


def fmt_md(d: date | datetime) -> str:
    return d.strftime("%m-%d")


def fmt_hm(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def fmt_stamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


def parse_date(text: str) -> date:
    return date.fromisoformat((text or "").strip()[:10])


def parse_dt(text: str) -> datetime | None:
    for f in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime((text or "").strip()[: len("2026-10-22 05:00")], f)
        except ValueError:
            continue
    return None


def json_load(path: Path, default):
    """读 JSON。吃 BOM（utf-8-sig）；读坏了留一份 `.bad-时间戳` 再从默认值起。

    以前这里 `except (OSError, ValueError): return default`：文件只是多了个
    BOM 就当成空配置，owner.json 一坏她就谁都不认，而且没人知道发生过。
    """
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return default
    except ValueError:
        _backup_bad(p)
        return default
    except OSError as e:
        _warn(f"读不了 {p.name}：{type(e).__name__}: {e}")
        return default


def json_load_strict(path: Path) -> dict | None:
    """给「读-改-写」用的读：区分「文件不存在」和「读坏了」。

    - 文件不存在 → `{}`（第一次写，正常）
    - 读坏了 → `None`（调用方必须**拒绝回写**，否则拿空底稿覆盖 = 静默清配置）
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except ValueError:
        _backup_bad(p)
        return None
    except OSError as e:
        _warn(f"读不了 {p.name}：{type(e).__name__}: {e}")
        return None
    return data if isinstance(data, dict) else None


def json_dump(path: Path, obj) -> None:
    atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def run_id(dt: datetime) -> str:
    return dt.strftime("%Y%m%d-%H%M%S")


def unique(seq):
    seen = set()
    out = []
    for item in seq:
        key = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# 通道 → 她嘴里的说法。原文里存英文标识，喂给模型时换成人话。
# 记原文（service）和装配（assemble）都用这一份，免得两处写法不一样。
CHANNEL_LABEL = {
    "qq": "QQ",
    "cli": "电脑",
    "web": "网页",
    "ball": "球",
    "phone": "手机",
}

_CHANNEL_TAG = re.compile(
    r"^\s*(?:[\[【]\s*(?:qq|球|ball|web|网页|cli|电脑|phone|手机)\s*[\]】]\s*)+",
    re.IGNORECASE,
)


def strip_channel_tag(text: str) -> str:
    """去掉她学着写的 [QQ] / 【qq】 / [球] 前缀。"""
    return _CHANNEL_TAG.sub("", text or "").lstrip()
