# Memory format

Tide keeps a companion's memory as plain files. This document is the stability
contract; changes require a `schema:` bump and a migration.

## Directory

```
data/memory/
├─ 2026-11-03-私聊.md          raw private conversation (append-only)
├─ 2026-11-03-群-<id>.md       raw group conversation (append-only)
├─ 2026-11-03.md               daily overview (derived)
├─ 事实.md                     facts (derived, schema 3)
├─ 事件.md                     event axis (derived)
├─ 关系.md                     slow-changing relationship notes
├─ 经历.md                     shared moments
├─ 别名.md                     aliases
├─ 索引.md                     human-readable index
├─ 核心/
│  ├─ 我.md                    who she is (≤ 200 chars)
│  └─ <owner>.md               who the owner is (≤ 250 chars)
├─ 我/
│  ├─ 自述.md  此刻.md  边界.md  想做.md  今日-行动.md
│  ├─ 没说的话.md  喜好.md
│  ├─ 日记/<date>.md
│  └─ 交接.md                  ≤ 4 KB handover
├─ 人/<nickname>.md            person cards
├─ 图片/                        image memory
├─ 我/...                       self state
├─ _状态.md                     batch status and watermark
├─ _审计/写入.jsonl             write audit
├─ _历史/                       pre-batch snapshots
└─ _索引.db                     rebuildable SQLite index
```

Derived files are always rebuildable from raw files. Deleting `_索引.db` is safe;
deleting a raw day file is not.

## Raw line

```
- 15:02 [qq] 他：上周面试那家黄了之后，我有点迷茫。
```

- time is local wall clock inside the logical day (05:00–05:00);
- `[qq]` / `[ball]` is the channel tag;
- `他` / `我` is the speaker;
- the rest is verbatim text (single line, trimmed).

## Fact schema (schema 3)

13 columns:

| # | column | meaning |
|---|---|---|
| 1 | subject | who the fact is about |
| 2 | topic | short topic key |
| 3 | content | the fact itself |
| 4 | speaker | who said it |
| 5 | source | 本人 / 转述 / 推断 |
| 6 | evidence | 明说 / 推断 |
| 7 | confidence | 0.0–1.0 |
| 8 | visibility | 私 / 群 |
| 9 | importance | 高 / 中 / 低 |
| 10 | recorded_at | date it was recorded |
| 11 | valid_to | last day it holds; empty = long-lived |
| 12 | group_id | group scope, empty = private |
| 13 | origin | which batch produced it |

`valid_to` is how one-off facts (a meal, a class) stop being "current" without
being deleted. Closed facts stay in the file for audit; retrieval ignores them
unless the question explicitly asks about the past.

## Events

An event is an open thread: `## #<id> <title>`, status line, start/last dates,
visibility, participants, and notes. Events cool down and archive on their own
schedule; they are not facts.

## Index

`_索引.db` stores documents (raw lines, facts, events, diary sentences, overview
bullets) with layer, day, visibility and a bigram postings table. It can be
rebuilt at any time from the text files.

## Privacy

Visibility is a property of the document, applied at assembly time. A group
channel never receives documents whose visibility is private, regardless of what
the prompt says.
