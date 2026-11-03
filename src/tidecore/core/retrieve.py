"""读路径：别名展开、bigram 索引、算分、置信度措辞。

不引向量。中文两字一组，够用；向量库只会多一个「检索不准时不知道怪谁」的变量。
`_索引.db` 只是加速层，删了能从 markdown 重建。
"""

from __future__ import annotations

import math
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from . import forget
from .config import Config
from .memory import Fact, Memory, importance_value
from .util import clip, parse_table, read_text, sections, strip_comments

STOP_PHRASES = [
    "你还记得",
    "你记得",
    "还记得",
    "记不记得",
    "是不是",
    "有没有",
    "说过",
    "那个",
    "那件",
    "那事",
    "后来",
    "怎么样",
    "怎么了",
    "什么",
    "上次",
    "之前",
    "以前",
    "我们",
    "你们",
    "我",
    "你",
    "吗",
    "呢",
    "啊",
    "吧",
    "的",
]
STOP_GRAMS = {"什么", "怎么", "这个", "那个", "我们", "你们", "他们", "是不是", "有没有"}
STOP_CHARS = set("的了是我你他她它们吗呢啊吧呀哦嗯这那有和就都也还很太在个一不")

VIS_ORDER = {"私": 2, "群": 1}
LAYER_WEIGHT = {
    "core": 1.0,
    "relation": 0.9,
    "fact": 0.9,
    "image": 0.85,
    "event": 0.8,
    "diary": 0.65,
    "current": 0.55,
    "wants": 0.6,
    "overview": 0.55,
    "index": 0.5,
    "month": 0.5,
    "raw": 0.35,
    "person": 0.7,
    "boundary": 1.0,
    "moment": 0.8,
}

# 两路检索的融合参数。
# - BM25：tf 和文档长度归一化，偏爱「短而准」的原话；
# - RRF：只看名次不看分数，两路排序打架时不会一边全赢。
# 覆盖率那一路（ov_idf）偏爱「把问法的词都覆盖到」的长文本，正好和 BM25 互补。
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60


def rrf_fuse(rows: list[dict], key_a: str, key_b: str) -> None:
    """两路分数 → RRF 名次分（0~1），写进 row["rrf"]。

    只看名次、不看量级：一路给的分是覆盖率、另一路是 BM25，量级根本不可比。
    两路都排第一 = 1.0；只在其中一路靠前 = 0.5 上下。
    """
    if not rows:
        return
    rank_a = {
        r["id"]: i + 1
        for i, r in enumerate(sorted(rows, key=lambda r: r.get(key_a, 0.0), reverse=True))
    }
    rank_b = {
        r["id"]: i + 1
        for i, r in enumerate(sorted(rows, key=lambda r: r.get(key_b, 0.0), reverse=True))
    }
    best = 2.0 / (RRF_K + 1)
    for r in rows:
        r["rrf"] = (
            1.0 / (RRF_K + rank_a[r["id"]]) + 1.0 / (RRF_K + rank_b[r["id"]])
        ) / best


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", "", text)


def tokenize(text: str) -> set[str]:
    text = normalize(text)
    out: set[str] = set()
    for word in re.findall(r"[a-z0-9_]{2,}", text):
        out.add(word)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) <= 2:
            out.add(run)
        for i in range(len(run) - 1):
            gram = run[i : i + 2]
            if gram not in STOP_GRAMS:
                out.add(gram)
        for ch in run:
            if ch not in STOP_CHARS:
                out.add(ch)
        if 2 < len(run) <= 6:
            out.add(run)
    return {g for g in out if g}


def tokenize_counts(text: str) -> dict[str, int]:
    """跟 tokenize 同一套切法，但带次数——BM25 的 tf 要用。

    两字以内的小段只算一次：tokenize 里那个 set 会把它去重，
    这里跟着保持一致，不然「养猫」会被算两次。
    """
    text = normalize(text)
    out: dict[str, int] = {}

    def add(g: str) -> None:
        if g:
            out[g] = out.get(g, 0) + 1

    for word in re.findall(r"[a-z0-9_]{2,}", text):
        add(word)
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(run) <= 2:
            add(run)
        else:
            for i in range(len(run) - 1):
                gram = run[i : i + 2]
                if gram not in STOP_GRAMS:
                    add(gram)
        for ch in run:
            if ch not in STOP_CHARS:
                add(ch)
        if 2 < len(run) <= 6:
            add(run)
    return out


def clean_query(text: str) -> str:
    q = text or ""
    for phrase in STOP_PHRASES:
        q = q.replace(phrase, " ")
    q = re.sub(r"[，。！？、,.!?;；:：「」\"'（）()\[\]【】…]+", " ", q)
    return q.strip()


def query_terms(text: str, mem: Memory) -> set[str]:
    cleaned = clean_query(text)
    terms = tokenize(cleaned)
    if not terms:
        terms = tokenize(text)
    for group in alias_groups(mem):
        words = [w for w in group if w]
        joined = "".join(words)
        if any(w and w in (text or "") for w in words) or (joined and joined in (text or "")):
            for w in words:
                terms |= tokenize(w)
    return terms


def alias_groups(mem: Memory) -> list[list[str]]:
    text = read_text(mem.cfg.memory_dir / "别名.md")
    rows, _ = parse_table(text, expected=2)
    groups = []
    for r in rows:
        std = (r.get("标准") or "").strip()
        alts = [x.strip() for x in re.split(r"[、,，/]", r.get("别名") or "") if x.strip()]
        if std or alts:
            groups.append([std] + alts)
    return groups


def overlap_ratio(terms: set[str], text: str) -> float:
    if not terms:
        return 0.0
    grams = tokenize(text)
    if not grams:
        return 0.0
    return len(terms & grams) / len(terms)


def recency(day: str, today: date | None = None) -> float:
    today = today or date.today()
    d = parse_loose_date(day, today)
    if not d:
        return 0.35
    days = max(0, (today - d).days)
    return math.exp(-days / 45.0)


def parse_loose_date(text: str, today: date | None = None) -> date | None:
    today = today or date.today()
    t = (text or "").strip()
    if not t or t in ("—", "-"):
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.match(r"^(\d{1,2})-(\d{1,2})$", t)
    if m:
        try:
            d = date(today.year, int(m.group(1)), int(m.group(2)))
            if d > today + timedelta(days=31):
                d = date(today.year - 1, d.month, d.day)
            return d
        except ValueError:
            return None
    return None


def confidence_phrase(conf: float) -> str:
    if conf >= 0.8:
        return "你之前说过"
    if conf >= 0.5:
        return "我记得好像是"
    return ""


# ---------------------------------------------------------------- 日期区间

_RANGE_MD = re.compile(
    r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?\s*(?:到|至|~|～|—|－|-)\s*"
    r"(?:(\d{1,2})\s*月\s*)?(\d{1,2})\s*[日号]?"
)
_RANGE_DASH = re.compile(
    r"(?<!\d)(\d{1,2})\s*[-/]\s*(\d{1,2})\s*(?:到|至|~|～|—|－)\s*"
    r"(\d{1,2})\s*[-/]\s*(\d{1,2})(?!\d)"
)
_SINGLE_MD = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_SINGLE_DASH = re.compile(r"(?<!\d)(\d{1,2})\s*[-/]\s*(\d{1,2})(?![点时\d])")

_CN_NUM = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_PAST_MARK = re.compile(r"(以前|之前|曾经|当时|那时候|那阵子|原来|过去|之前说|改口|变(?:了|卦)|后来)")
_PAST_NEG = re.compile(r"(以前没|之前没|从没|没有过|第一次)")


def wants_closed(query: str) -> bool:
    """问法是不是在问「以前是什么样」。

    「后来」也算：老事实不进来，就答不出「那事儿后来怎么变的」。
    但「以前没去过」这种否定句不算，避免把历史事实硬拉进来说它存在过。
    """
    q = query or ""
    if _PAST_NEG.search(q):
        return False
    return bool(_PAST_MARK.search(q))


def _safe_day(year: int, month: int, day: int, today: date) -> date | None:
    try:
        d = date(year, month, day)
    except ValueError:
        return None
    # 一月里说「12月30号」：指的是去年；跨年问法不能落到未来。
    if d > today + timedelta(days=1):
        try:
            d = date(year - 1, month, day)
        except ValueError:
            return None
    return d


def _cn_int(text: str) -> int | None:
    text = (text or "").strip()
    if text.isdigit():
        return int(text)
    if text in _CN_NUM:
        return _CN_NUM[text]
    if text.startswith("十") and len(text) == 2 and text[1] in _CN_NUM:
        return 10 + _CN_NUM[text[1]]
    if text.endswith("十") and len(text) == 2 and text[0] in _CN_NUM:
        return _CN_NUM[text[0]] * 10
    return None


def parse_day_range(query: str, today: date | None = None) -> tuple[date, date] | None:
    """从问法里取一个日期区间：那几天 / 上周 / 10月20号到23号。

    只认明确的时间词，不做猜测。认不出来返回 None，检索就照旧按相关性来。
    """
    today = today or date.today()
    q = query or ""
    m = _RANGE_MD.search(q)
    if m:
        m1, d1 = int(m.group(1)), int(m.group(2))
        m2 = int(m.group(3)) if m.group(3) else m1
        d2 = int(m.group(4))
        start = _safe_day(today.year, m1, d1, today)
        end = _safe_day(today.year, m2, d2, today)
        if start and end:
            return (start, end) if start <= end else (end, start)
    m = _RANGE_DASH.search(q)
    if m:
        start = _safe_day(today.year, int(m.group(1)), int(m.group(2)), today)
        end = _safe_day(today.year, int(m.group(3)), int(m.group(4)), today)
        if start and end:
            return (start, end) if start <= end else (end, start)

    if re.search(r"上上个?星期|上上周", q):
        monday = today - timedelta(days=today.weekday() + 14)
        return monday, monday + timedelta(days=6)
    if re.search(r"上周|上个?星期|上星期", q):
        monday = today - timedelta(days=today.weekday() + 7)
        return monday, monday + timedelta(days=6)
    if re.search(r"这周|本周|这个?星期|这星期", q):
        return today - timedelta(days=today.weekday()), today
    if re.search(r"上个?月|上个月", q):
        first = today.replace(day=1)
        end = first - timedelta(days=1)
        return end.replace(day=1), end
    if re.search(r"这个?月|本月", q):
        return today.replace(day=1), today

    m = re.search(r"(?:最近|近|前|这)\s*(\d+|[一二两三四五六七八九十])\s*天", q)
    if m:
        n = _cn_int(m.group(1))
        if n and 1 <= n <= 31:
            return today - timedelta(days=n - 1), today
    m = re.search(r"(\d+|[一二两三四五六七八九十])\s*天前", q)
    if m:
        n = _cn_int(m.group(1))
        if n and 1 <= n <= 31:
            return today - timedelta(days=n), today - timedelta(days=n)
    if re.search(r"这两天", q):
        return today - timedelta(days=1), today
    if re.search(r"这几天|这几日", q):
        return today - timedelta(days=2), today

    m = _SINGLE_MD.search(q)
    if m:
        d = _safe_day(today.year, int(m.group(1)), int(m.group(2)), today)
        if d:
            return d, d
    m = _SINGLE_DASH.search(q)
    if m:
        d = _safe_day(today.year, int(m.group(1)), int(m.group(2)), today)
        if d:
            return d, d
    return None


# ---------------------------------------------------------------- 索引库

# 重建是「写一个新库 → 换掉旧库」：同进程里只能一个人做。
# 以前心跳（日批后重建）和请求线程（ensure/bump_hit）会同时进 rebuild，
# Windows 上一个人还开着连接，另一个人 unlink 旧库就 WinError 32。
_REBUILD_LOCK = threading.Lock()


class IndexDB:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.path = cfg.index_db

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA journal_mode=WAL")
        self._migrate(conn)
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """老库补一列 stability（遗忘曲线的稳定度）。

        不重建：hit 表里是「这条被想起过几次」，重建索引时都要留着，
        顺手删掉就等于把她记得最牢的东西全忘了。
        """
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(hit)")}
            if cols and "stability" not in cols:
                conn.execute("ALTER TABLE hit ADD COLUMN stability REAL DEFAULT 0")
                conn.commit()
        except sqlite3.Error:
            pass

    def available(self) -> bool:
        return self.path.exists()

    def ensure(self, mem: Memory, max_age: int = 300) -> None:
        if not self.path.exists():
            self.rebuild(mem)
            return
        try:
            age = time.time() - self.path.stat().st_mtime
            newest = max(
                [p.stat().st_mtime for p in self._indexable(mem)]
                or [0]
            )
            if newest > self.path.stat().st_mtime and age > max_age:
                self.rebuild(mem)
        except OSError:
            self.rebuild(mem)

    def stale(self, mem: Memory) -> bool:
        if not self.path.exists():
            return True
        try:
            newest = max([p.stat().st_mtime for p in self._indexable(mem)] or [0])
            return newest > self.path.stat().st_mtime
        except OSError:
            return True

    def _indexable(self, mem: Memory) -> list[Path]:
        out = []
        skip_prefix = ("_历史", "_热", "_验收")
        for p in self.cfg.memory_dir.rglob("*.md"):
            rel = p.relative_to(self.cfg.memory_dir).as_posix()
            if rel.startswith(skip_prefix):
                continue
            if p.name in ("_状态.md", "_坏行.md", "_安全.md", "_待覆盖.md", "交接.md"):
                continue
            out.append(p)
        return out

    def rebuild(self, mem: Memory) -> None:
        with _REBUILD_LOCK:
            self._rebuild(mem)

    def _rebuild(self, mem: Memory) -> None:
        tmp = self.path.with_suffix(".db.tmp")
        if tmp.exists():
            tmp.unlink()
        conn = sqlite3.connect(str(tmp))
        try:
            conn.executescript(
                """
                CREATE TABLE doc (
                    id INTEGER PRIMARY KEY,
                    file TEXT, line INTEGER, layer TEXT, day TEXT,
                    visibility TEXT, speaker TEXT, weight REAL,
                    confidence REAL, text TEXT
                );
                CREATE TABLE bigram (gram TEXT, doc_id INTEGER);
                CREATE INDEX idx_gram ON bigram(gram);
                CREATE TABLE df (gram TEXT PRIMARY KEY, n INTEGER);
                CREATE TABLE hit (
                    key TEXT PRIMARY KEY, count INTEGER, last TEXT,
                    stability REAL DEFAULT 0
                );
                CREATE TABLE alias (standard TEXT, alias TEXT);
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
                """
            )
            docs: list[tuple] = []
            grams: list[tuple] = []
            # df = 这个词出现在几篇里。idf 要用：一个词只在 3 篇里出现，
            # 命中它比命中「百度」这种几千篇都有的词值钱得多。
            dfs: dict[str, int] = {}
            total_len = 0
            for path in self._indexable(mem):
                for doc in self._docs_for(path, mem):
                    docs.append(doc)
                    did = len(docs)
                    grams_of_doc = tokenize(doc[8])
                    total_len += len(grams_of_doc)
                    for g in grams_of_doc:
                        grams.append((g, did))
                        dfs[g] = dfs.get(g, 0) + 1
            conn.executemany(
                "INSERT INTO doc(file,line,layer,day,visibility,speaker,weight,confidence,text) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                docs,
            )
            conn.executemany("INSERT INTO bigram(gram,doc_id) VALUES (?,?)", grams)
            conn.executemany("INSERT INTO df(gram,n) VALUES (?,?)", dfs.items())
            avg_len = (total_len / len(docs)) if docs else 0.0
            conn.execute("INSERT INTO meta(key,value) VALUES ('avg_len',?)", (f"{avg_len:.2f}",))
            for group in alias_groups(mem):
                std = group[0] if group else ""
                for alt in group:
                    conn.execute("INSERT INTO alias(standard,alias) VALUES (?,?)", (std, alt))
            conn.execute("INSERT INTO meta(key,value) VALUES ('built_at',?)", (datetime.now().isoformat(timespec="seconds"),))
            conn.execute("INSERT INTO meta(key,value) VALUES ('docs',?)", (str(len(docs)),))
            conn.commit()
        finally:
            conn.close()
        # 保留旧 hit 表（含遗忘曲线的稳定度；老库没这列就补 0）
        old_hits = []
        if self.path.exists():
            try:
                old = sqlite3.connect(str(self.path))
                try:
                    old_hits = old.execute(
                        "SELECT key,count,last,COALESCE(stability,0) FROM hit"
                    ).fetchall()
                except sqlite3.Error:
                    old_hits = [tuple(r) + (0.0,) for r in old.execute(
                        "SELECT key,count,last FROM hit"
                    ).fetchall()]
                old.close()
            except sqlite3.Error:
                old_hits = []
        if old_hits:
            conn = sqlite3.connect(str(tmp))
            conn.executemany(
                "INSERT OR REPLACE INTO hit(key,count,last,stability) VALUES (?,?,?,?)",
                old_hits,
            )
            conn.commit()
            conn.close()
        # 换文件：别的线程/进程可能还开着旧库（WAL 也有 -wal/-shm），
        # Windows 上这时 unlink 会 WinError 32。退避重试几轮再放弃。
        last: Exception | None = None
        for attempt in range(6):
            try:
                self.path.unlink(missing_ok=True)
                tmp.replace(self.path)
                return
            except PermissionError as e:
                last = e
                time.sleep(0.08 * (attempt + 1))
        if last is not None:
            raise last
        raise OSError(f"换不了索引文件：{self.path}")

    def _docs_for(self, path: Path, mem: Memory) -> list[tuple]:
        rel = path.relative_to(self.cfg.memory_dir).as_posix()
        name = path.name
        day = _day_from_name(name)
        lines = read_text(path).splitlines()
        out: list[tuple] = []

        def add(layer: str, lineno: int, vis: str, speaker: str, weight: float, conf: float, text: str):
            text = text.strip()
            if not text:
                return
            out.append((rel, lineno, layer, day, vis, speaker, weight, conf, text))

        if name == "事实.md":
            # 已关闭的事实不进全文索引：历史由 recall_facts 的「以前」分支负责，
            # 混进普通检索只会让「他现在怎么样」被旧状态带偏。
            rows, _ = parse_table(path.read_text(encoding="utf-8"))
            for r in rows:
                content = r.get("内容", "")
                if "已废" in content or (r.get("有效期至", "") or "").strip():
                    continue
                vis = "群" if r.get("可见性") == "群" else "私"
                try:
                    conf = float(r.get("置信度", "0.6"))
                except ValueError:
                    conf = 0.6
                w = 0.7 + 0.3 * importance_value(r.get("重要度", "中"))
                add("fact", 0, vis, r.get("谁说的", ""), w, conf,
                    f"{r.get('主题','')} {content}")
            return out

        if name == "事件.md":
            cur_vis = "私"
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if s.startswith("## #"):
                    cur_vis = "私"
                    add("event", i, cur_vis, "", LAYER_WEIGHT["event"], 0.8, s.lstrip("# ").strip())
                elif s.startswith("可见性："):
                    cur_vis = s[4:].strip()
                elif s.startswith("- "):
                    add("event", i, cur_vis, "", LAYER_WEIGHT["event"], 0.8, s[2:])
            return out

        if name == "经历.md":
            rows, _ = parse_table(path.read_text(encoding="utf-8"))
            for i, r in enumerate(rows, 1):
                text = f"{r.get('类型', '')}：{r.get('事', '')}"
                if r.get("原话"):
                    text += f"（{r.get('原话')}）"
                if (r.get("事") or "").strip():
                    add("moment", i, "私", "", LAYER_WEIGHT["moment"], 0.85, text)
            return out

        if name == "图片.md":
            rows, _ = parse_table(path.read_text(encoding="utf-8"))
            for i, r in enumerate(rows, 1):
                cap = (r.get("图里是什么") or "").strip()
                if not cap:
                    continue
                text = f"{r.get('日期', '')} {r.get('时间', '')} 他发的图：{cap}"
                add("image", i, "私", "", LAYER_WEIGHT["image"], 0.85, text)
            return out

        if name == "关系.md":
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("<!--"):
                    continue
                add("relation", i, "私", "", LAYER_WEIGHT["relation"], 0.85, s)
            return out

        if rel.startswith("核心/"):
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("<!--") or s.startswith("##"):
                    continue
                add("core", i, "私", "", 1.0, 0.95, s)
            return out

        if rel.startswith("我/"):
            layer = "diary" if "日记" in rel else ("current" if name == "此刻.md" else ("boundary" if name == "边界.md" else ("wants" if name == "想做.md" else "self")))
            vis = "私"
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("<!--"):
                    continue
                if layer == "boundary" and s.startswith("- ["):
                    # 「群」在索引里代表两边可见；[全] 也归这一档，[私] 只有私聊看得到
                    vis = "私" if s.startswith("- [私]") else "群"
                add(layer, i, vis, "", LAYER_WEIGHT.get(layer, 0.5), 0.8, s)
            return out

        if name == "索引.md" or rel.startswith("月/"):
            for i, line in enumerate(lines, 1):
                if line.strip().startswith("|") and not re.match(r"^\|[\s:\-|]+\|$", line.strip()):
                    add("index" if name == "索引.md" else "month", i, "私", "", LAYER_WEIGHT["index"], 0.7, line)
            return out

        if re.match(r"^\d{4}-\d{2}-\d{2}\.md$", name):
            cur_vis = "私"
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if s.startswith("## 概要"):
                    cur_vis = "群" if "群" in s else "私"
                elif s.startswith("## "):
                    cur_vis = "私"
                if s.startswith("- "):
                    add("overview", i, cur_vis, "", LAYER_WEIGHT["overview"], 0.7, s[2:])
            return out

        if re.match(r"^\d{4}-\d{2}-\d{2}-(私聊|群-.+)\.md$", name):
            vis = "群" if "-群-" in name else "私"
            for i, line in enumerate(lines, 1):
                m = re.match(r"^- \d{2}:\d{2}\s+([^：:]{1,24})[：:]\s*(.*)$", line.strip())
                if m:
                    add("raw", i, vis, m.group(1), LAYER_WEIGHT["raw"], 0.8, m.group(2))
            return out

        if rel.startswith("人/"):
            # 群友卡是群可见的；主人卡是私的。以前一律写成「私」，
            # 群窗口检索永远搜不到群友这张卡。
            first = lines[0] if lines else ""
            vis = "群" if "visibility: 群" in first else "私"
            for i, line in enumerate(lines, 1):
                s = line.strip()
                if not s or s.startswith("#") or s.startswith("<!--") or s.startswith("##"):
                    continue
                add("person", i, vis, name[:-3] if name.endswith(".md") else name, LAYER_WEIGHT["person"], 0.8, s)
            return out

        return out

    def search(
        self,
        terms: set[str],
        visibility: str,
        limit: int = 12,
        day_range: tuple[date, date] | None = None,
    ) -> list[dict]:
        if not terms or not self.path.exists():
            return []
        try:
            conn = self._connect()
        except sqlite3.Error:
            return []
        try:
            qmarks = ",".join("?" for _ in terms)
            # 按「命中几个查询词」排，再截 800。
            # 以前是 DISTINCT 之后直接 LIMIT 800：长问法里像「10」「22」这种
            # 高频 gram 会把候选塞满，真正相关的行（只命中「局限」那两句）
            # 反而在 800 之外，等于按 doc_id 抽样，和相关性无关。
            rows = conn.execute(
                f"SELECT doc_id, COUNT(*) AS n FROM bigram WHERE gram IN ({qmarks}) "
                f"GROUP BY doc_id ORDER BY n DESC, doc_id LIMIT 800",
                tuple(terms),
            ).fetchall()
            ids = [r[0] for r in rows]
            if not ids:
                return []
            idmarks = ",".join("?" for _ in ids)
            sql = (
                f"SELECT id,file,line,layer,day,visibility,speaker,weight,confidence,text "
                f"FROM doc WHERE id IN ({idmarks})"
            )
            params: tuple = tuple(ids)
            if day_range:
                # 没日期的文档（事实/关系/核心）不受区间限制：它们不是「某天说的话」。
                sql += " AND (day = '' OR (day >= ? AND day <= ?))"
                params += (day_range[0].isoformat(), day_range[1].isoformat())
            docs = conn.execute(sql, params).fetchall()
        except sqlite3.Error:
            return []
        finally:
            conn.close()

        out = []
        strong_terms = {t for t in terms if len(t) >= 2 and not t.isdigit()}
        # idf 加权覆盖率：命中一个罕见词（「局限」「馄饨」）比命中一个
        # 几千篇都有的词（「图片」「今天」）值钱。老库没有 df 表时退回纯覆盖率。
        idf = self.idf_map(terms)
        idf_total = sum(idf.values())
        avg_len = self.avg_doc_len()
        for d in docs:
            if visibility == "群" and d[5] != "群":
                continue
            if visibility == "私" and d[5] not in ("私", "群"):
                continue
            counts = tokenize_counts(d[9])
            hit = terms & counts.keys()
            if not hit:
                continue
            ov = len(hit) / len(terms) if terms else 0.0
            if idf_total > 0:
                ov_idf = sum(idf.get(t, 0.0) for t in hit) / idf_total
            else:
                ov_idf = ov
            # 只靠单字命中的，算「擦边」：问「我养猫吗」会捞到「一只橘猫趴键盘上」
            # 这种图里的猫，翻出来就成了「他养猫」。词（≥2 字）命中才算数。
            strong = (
                len({t for t in hit if len(t) >= 2 and not t.isdigit()}) / len(strong_terms)
                if strong_terms
                else ov
            )
            bm25 = 0.0
            if avg_len > 0 and idf_total > 0:
                dl = max(1, len(counts))
                norm = BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_len)
                for t in hit:
                    tf = counts[t]
                    bm25 += idf.get(t, 0.0) * tf * (BM25_K1 + 1) / (tf + norm)
            out.append(
                {
                    "id": d[0],
                    "file": d[1],
                    "line": d[2],
                    "layer": d[3],
                    "day": d[4],
                    "visibility": d[5],
                    "speaker": d[6],
                    "text": d[9],
                    "weight": float(d[7]),
                    "confidence": float(d[8]),
                    "overlap": ov,
                    "overlap_idf": ov_idf,
                    "overlap_strong": strong,
                    "bm25": bm25,
                }
            )

        # 两路排名 → RRF 融合。
        # 覆盖率那一路看「问法的词覆盖了多少」，BM25 看「命中得多密、文档多短」，
        # 两路打架时 RRF 各给一半名次分，不会一边全赢。最后再叠层先验。
        if out and idf_total > 0 and avg_len > 0:
            rrf_fuse(out, "overlap_idf", "bm25")
            for r in out:
                rel = 0.5 * r["rrf"] + 0.5 * r["overlap_idf"]
                r["score"] = (
                    0.5 * rel
                    + 0.2 * r["weight"]
                    + 0.15 * recency(r["day"] or "")
                    + 0.15 * r["confidence"]
                )
        else:
            # 老库没有 df 表 / avg_len：退回纯覆盖率的旧算法，别让排序失去意义。
            for r in out:
                r["rrf"] = 0.0
                r["score"] = (
                    0.5 * r["overlap"]
                    + 0.2 * r["weight"]
                    + 0.15 * recency(r["day"] or "")
                    + 0.15 * r["confidence"]
                )
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:limit]

    def avg_doc_len(self) -> float:
        """平均长度（按词数）。老库没有这项就返回 0，调用方退回旧算法。"""
        if not self.path.exists():
            return 0.0
        try:
            conn = self._connect()
            try:
                row = conn.execute("SELECT value FROM meta WHERE key='avg_len'").fetchone()
            finally:
                conn.close()
        except sqlite3.Error:
            return 0.0
        try:
            return float(row[0]) if row else 0.0
        except (TypeError, ValueError):
            return 0.0

    def idf_map(self, terms: set[str]) -> dict[str, float]:
        """查询词的 idf。缺 df 表（老库）时返回空，调用方退回纯覆盖率。

        分母只算「索引里真的存在」的词：别名展开出来的词若全库没出现过，
        它永远不可能命中，算进分母只会把所有文档的分数一起压低。
        """
        if not terms or not self.path.exists():
            return {}
        try:
            conn = self._connect()
            try:
                qmarks = ",".join("?" for _ in terms)
                rows = conn.execute(
                    f"SELECT gram,n FROM df WHERE gram IN ({qmarks})", tuple(terms)
                ).fetchall()
                total = int(conn.execute("SELECT COUNT(*) FROM doc").fetchone()[0] or 0)
            finally:
                conn.close()
        except sqlite3.Error:
            return {}
        if total <= 0:
            return {}
        out = {}
        for gram, n in rows:
            n = max(1, int(n or 1))
            out[str(gram)] = math.log(1 + (total - n + 0.5) / (n + 0.5))
        return out

    def bump_hit(
        self,
        key: str,
        base: float = 0.0,
        seen: date | None = None,
        strengthen: bool = True,
        today: date | None = None,
    ) -> None:
        """记一次「想起了这条」，顺便按间隔重复把稳定度往上抬。

        `strengthen=False` 是「只是被列进了提示词」：保底凑数、被关掉的旧行。
        那种不算真的想起来 —— 只加计数，不刷新 last、也不涨稳定度，
        否则一条从没被问起过的事，光靠当背景板就能永不遗忘。

        `seen` 是这条记忆上次被想起（没有记录时用「记于」）的日期，
        新库第一次想起时拿它算间隔。
        """
        if not self.path.exists():
            return
        knobs = self.cfg.knobs
        today = today or date.today()
        try:
            conn = self._connect()
            row = conn.execute(
                "SELECT count,last,COALESCE(stability,0) FROM hit WHERE key=?", (key,)
            ).fetchone()
            if not strengthen and row is None:
                # 从没被想起过、这次也只是凑数：不留痕，否则一条从没被问过的事
                # 光靠当背景板就能躲过遗忘。
                conn.close()
                return
            count = int(row[0] or 0) + 1 if row else 1
            if not strengthen:
                last = (row[1] if row and row[1] else "")
                stab = float(row[2] or 0.0) if row else 0.0
            else:
                prev = _parse_day(row[1]) if row else None
                start = float(base) if base and float(base) > 0 else forget.DEFAULT_STABILITY
                stab = forget.review(
                    float(row[2] or 0.0) if row else start,
                    prev or seen,
                    today,
                    base=start,
                    growth=float(getattr(knobs, "forget_growth", forget.DEFAULT_GROWTH)),
                    cap=float(getattr(knobs, "forget_stability_max", forget.MAX_STABILITY)),
                )
                last = today.isoformat()
            conn.execute(
                "INSERT INTO hit(key,count,last,stability) VALUES (?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "count=excluded.count,last=excluded.last,stability=excluded.stability",
                (key, count, last, stab),
            )
            conn.commit()
            conn.close()
        except sqlite3.Error:
            pass

    def hit_stats(self) -> dict[str, dict] | None:
        """一次读出所有 key 的 {count, last, stability}，别一条一条查。

        读不出来返回 **None**（和「读出来是空表」区分）：遗忘归档靠它判断
        「有没有被想起过」，把坏库当成空表就等于把她记得最牢的事当没想起过。
        """
        if not self.path.exists():
            return None
        conn = None
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT key,count,last,COALESCE(stability,0) FROM hit"
            ).fetchall()
        except (sqlite3.Error, OSError) as e:
            print(f"[retrieve] hit 表读不出来：{type(e).__name__}: {e}", flush=True)
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
        return {
            r[0]: {
                "count": int(r[1] or 0),
                "last": _parse_day(r[2]),
                "stability": float(r[3] or 0.0),
            }
            for r in rows
        }

    def last_hit(self, key: str) -> date | None:
        if not self.path.exists():
            return None
        try:
            conn = self._connect()
            row = conn.execute("SELECT last FROM hit WHERE key=?", (key,)).fetchone()
            conn.close()
            return _parse_day(row[0]) if row else None
        except sqlite3.Error:
            return None


def _parse_day(text: str | None) -> date | None:
    try:
        return date.fromisoformat((text or "")[:10])
    except (ValueError, TypeError):
        return None


def _day_from_name(name: str) -> str:
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- 检索

@dataclass
class RecalledFact:
    fact: Fact
    score: float
    phrase: str
    line: str


class Retriever:
    def __init__(self, mem: Memory):
        self.mem = mem
        self.cfg = mem.cfg
        self.db = IndexDB(self.cfg)

    def ensure(self) -> None:
        self.db.ensure(self.mem)

    def recall_facts(
        self,
        query: str,
        visibility: str,
        subject: str = "",
        limit: int | None = None,
        today: date | None = None,
        with_subject: bool = True,
        include_closed: bool = False,
        window: tuple[date, date] | None = None,
        group_id: str | None = None,
    ) -> list[RecalledFact]:
        """按问题召回事实。

        subject 是「这一轮该引用谁的事」：群里默认是当前说话人，
        问法里明确点了别人就换成那个人。空着才全表召回（调试/回归用）。
        以前 speaker 参数收进来却没用，事实表按关键词全库捞，
        群里问「我喜欢什么」可能把别人的事实当成他的。

        include_closed 只在问法带「以前/当时/原来」时为真：
        关掉的事实带「（后来变了）」进来，答得出「那会儿是什么情况」。

        window 是问法里点明的日期区间：按「那个时间成立」取事实。
        10-20 号问「我是不是在面试」，取的是 10-20 时有效的那条，
        不是今天的。没有这条，区间就白记了。
        """
        limit = limit or self.cfg.knobs.facts_limit
        knobs = self.cfg.knobs
        today = today or date.today()
        terms = query_terms(query, self.mem)
        subject = (subject or "").strip()
        facts = [
            f
            for f in self.mem.facts()
            if (not subject or f.subject == subject)
        ]
        floor = float(knobs.confidence_floor)
        # 遗忘曲线要用到的东西一次读齐：每条被想起过几次、上次什么时候、稳定度。
        # 读不出来（坏库/被锁）就当谁都没被想起过 —— 只是排序退化，不影响回话。
        stats = self.db.hit_stats() or {}
        forget_floor = float(getattr(knobs, "forget_floor", 0.05))
        base_days = float(getattr(knobs, "forget_stability_days", forget.DEFAULT_STABILITY))

        def curve_of(f: Fact) -> tuple[float, float]:
            """(初始稳定度, 今天的可回忆度)。没被想起过就从「记于」那天开始衰减。"""
            base = forget.stability0(base_days, f.importance)
            st = stats.get(f"{f.subject}|{f.topic}") or {}
            touched = st.get("last") or parse_loose_date(f.noted, today)
            stab = float(st.get("stability") or 0.0) or base
            return base, forget.retrievability(stab, touched, today)

        # 事实也走和原文同一套：idf 来自整份记忆的 df（事实表只有几十条，
        # 自己算 df 会抖得厉害）。没索引时退回纯覆盖率。
        idf = self.db.idf_map(terms)
        idf_total = sum(idf.values())
        rows: list[dict] = []
        for f in facts:
            if window:
                start, end = window
                noted = parse_loose_date(f.noted, today)
                if noted and noted > end:
                    continue  # 那时还没有这条
                if f.closed:
                    until = parse_loose_date(f.valid_to, today)
                    # 「未知」这类没有日期的旧数据不能当成「窗口开始前就失效」：
                    # 否则 10-21 问「我那天说熬夜，后来改口了吗」会把旧说法整条丢掉。
                    if until and until <= start:
                        continue
            else:
                if f.closed and not include_closed:
                    continue
            # 老数据里「已废」会把置信度清零；那是关闭动作留下的，不代表不可信。
            if float(f.confidence) < floor and not (f.closed and float(f.confidence) == 0):
                continue
            if visibility == "群":
                if f.visibility != "群":
                    continue
                # 只在「这个群」或「不限群」的事里找：A 群的宠物、B 群的暗号，
                # 不能因为都属于「群可见」就互相串。
                if (f.group_id or "") not in ((group_id or "").strip(), ""):
                    continue
            # 私聊可以看到所有群的事实（他是同一个人），但看不到别人的私聊事实。
            text = f"{f.subject} {f.topic} {f.content}"
            counts = tokenize_counts(text)
            hit = terms & counts.keys()
            ov = (len(hit) / len(terms)) if terms else 0.0
            if idf_total > 0:
                ov_idf = sum(idf.get(t, 0.0) for t in hit) / idf_total
            else:
                ov_idf = ov
            imp = importance_value(f.importance)
            _, recall = curve_of(f)
            if ov == 0 and (imp < 0.9 or f.closed):
                # 没命中又不是高权重，不硬塞；关掉的事实更不该靠权重挤进来
                continue
            # 遗忘曲线：淡到快想不起来了，问法又只是擦边（一半的词都没命中），
            # 就先不拿它当「当前事实」。问「以前/当时」或按日期问不受影响——
            # 那是主动回忆，不是她没事提。
            if (
                recall < forget_floor
                and not window
                and not include_closed
                and ov < 0.5
            ):
                continue
            key = f"{f.subject}|{f.topic}"
            if float(f.confidence) < 0.6 and not f.closed:
                last = (stats.get(key) or {}).get("last")
                noted = parse_loose_date(f.noted, today)
                if last is None and noted and (today - noted).days > int(knobs.fact_stale_days):
                    continue
            rows.append(
                {
                    "id": key,
                    "fact": f,
                    "ov": ov,
                    "ov_idf": ov_idf,
                    "imp": imp,
                    "recall": recall,
                    "conf": float(f.confidence),
                    "counts": counts,
                    "hit": hit,
                    "dl": max(1, len(counts)),
                }
            )

        # 覆盖率 + BM25 两路 → RRF，再叠重要度 / 可回忆度 / 置信度。
        avg_dl = (sum(r["dl"] for r in rows) / len(rows)) if rows else 0.0
        use_rank = bool(rows) and idf_total > 0 and avg_dl > 0
        for r in rows:
            bm25 = 0.0
            if use_rank:
                norm = BM25_K1 * (1 - BM25_B + BM25_B * r["dl"] / avg_dl)
                for t in r["hit"]:
                    tf = r["counts"][t]
                    bm25 += idf.get(t, 0.0) * tf * (BM25_K1 + 1) / (tf + norm)
            r["bm25"] = bm25
        if use_rank:
            rrf_fuse(rows, "ov_idf", "bm25")

        scored: list[tuple[float, Fact]] = []
        for r in rows:
            rel = (0.5 * r["rrf"] + 0.5 * r["ov_idf"]) if use_rank else r["ov"]
            # 相关性占大头：先验（重要度/可回忆度/置信度）只用来打平手。
            # 以前四项各 0.2/0.15 很平均，结果是「刚被别的问题想起过」的那条
            # 只要有 0.03 的先验优势，就能把真正对题的事实挤出前 6——
            # 同一句问法，先问「我叫什么」再问「群有多大」，答案会变。
            score = 0.65 * rel + 0.15 * r["imp"] + 0.1 * r["recall"] + 0.1 * r["conf"]
            if r["fact"].closed:
                # 按日期问：旧事实在那个时点是正确的，不该像「后来变了」那样重罚
                score *= 0.9 if window else 0.65
            scored.append((score, r["fact"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        label = ""
        if window:
            start, end = window
            label = f"{start:%m-%d}" if start == end else f"{start:%m-%d}到{end:%m-%d}"
        out = [
            RecalledFact(
                f, s, confidence_phrase(float(f.confidence)), self._format_fact(f, with_subject, label)
            )
            for s, f in scored[:limit]
        ]
        if len(out) < int(knobs.facts_min):
            used = {x.fact.key for x in out}
            extras = sorted(
                [
                    f
                    for f in facts
                    if f.key not in used
                    and not f.closed
                    and float(f.confidence) >= floor
                    and curve_of(f)[1] >= forget_floor
                    and (
                        visibility != "群"
                        or (
                            f.visibility == "群"
                            and (f.group_id or "") in ((group_id or "").strip(), "")
                        )
                    )
                    and importance_value(f.importance) >= 0.9
                ],
                key=lambda f: (importance_value(f.importance), curve_of(f)[1]),
                reverse=True,
            )
            for f in extras[: max(0, int(knobs.facts_min) - len(out))]:
                out.append(
                    RecalledFact(
                        f,
                        0.0,
                        confidence_phrase(float(f.confidence)),
                        self._format_fact(f, with_subject),
                    )
                )
        for item in out:
            f = item.fact
            base, _ = curve_of(f)
            # 只有真的被问到的才算「想起来」：保底凑出来的、关掉的旧行不涨稳定度。
            self.db.bump_hit(
                f"{f.subject}|{f.topic}",
                base=base,
                seen=parse_loose_date(f.noted, today),
                strengthen=item.score > 0 and not f.closed,
                today=today,
            )
        return out

    def _format_fact(self, f: Fact, with_subject: bool = True, as_of: str = "") -> str:
        # 老数据的已废行带 markdown 删除线：喂进提示词前先剥掉，只留原话。
        content = re.sub(r"^~~(.*?)~~\s*（已废[^）]*）$", r"\1", (f.content or "").strip())
        # 群里/问到别人时必须带「说的是谁」；私聊只讲主人自己的事时可以省，
        # 省下的字正好留给概要，不然回归集里「百人群」那条会被挤掉。
        body = f"{f.subject}·{f.topic}：{content}" if with_subject else f"{f.topic}：{content}"
        noted = f.noted[5:] if len(f.noted or "") >= 10 else ""
        until = f.valid_to[5:] if len(f.valid_to or "") >= 10 else ""
        if f.closed:
            if as_of:
                # 按日期问：这条在窗口里成立，但必须把「记于哪天 / 有效到哪天」
                # 一起带上。否则 10-22 的「早饭吃了抄手」会被当成 10-27 的早饭。
                if noted:
                    span = f"（{noted} 记的" + (f"，{until} 前有效）" if until else "）")
                else:
                    span = f"（{as_of} 那会儿）"
                return f"{span}{body}"
            return f"（后来变了）{body}" + (f"（{until} 前）" if until else "")
        phrase = confidence_phrase(float(f.confidence))
        # 有明确日期时，事实自己的「记于」日期跟着走：模型才分得清
        # 「10-22 的早饭」和「10-27 的早饭」。
        stamp = f"（{noted}）" if (as_of and noted) else ""
        if phrase:
            return f"{stamp}{phrase}——{body}"
        return f"{stamp}（我记得不太清）{body}"

    def recall_docs(
        self,
        query: str,
        visibility: str,
        layers: tuple[str, ...] | None = None,
        limit: int = 8,
        today: date | None = None,
        day_range: tuple[date, date] | None = None,
        include_group: bool = True,
    ) -> list[dict]:
        """翻原文 / 日记 / 概要。

        include_group=False 时把「群」可见的原文和群摘要挡在外面：
        私聊问「我最近在修什么」，不该翻出群友的「最近猛猛搓视频」；
        群里的事只有问法真的沾群时才由调用方显式放开。
        """
        terms = query_terms(query, self.mem)
        if not terms:
            return []
        # 先多取一些候选再过滤：当天原文里群聊往往比私聊多，取少了会误伤。
        docs = self.db.search(
            terms, visibility, limit=limit * (4 if include_group else 8), day_range=day_range
        )
        if layers:
            docs = [d for d in docs if d["layer"] in layers]
        if not include_group:
            docs = [d for d in docs if d["visibility"] != "群"]
        return docs[:limit]
