# %% [markdown]
# # Ingest — Chuang Tzu
# Source: data/eastern/chuang_tzu.txt (Project Gutenberg, Giles translation, 1889)
#
# Passage granularity: one passage per prose paragraph within each chapter.
# Editorial commentary (indented paragraphs) and _Argument_ blocks are skipped.
#
# book        = chapter label, e.g. "I — Transcendental Bliss"
# section     = None
# unit_number = paragraph index within chapter (1-based)
# unit_label  = "Chuang Tzu I:3"

import re

import duckdb

from db.ingest import _get_or_create_corpus, insert_passages
from db.models import CorpusRecord, PassageRecord
from db.schema import get_conn

TXT_PATH = "data/eastern/chuang_tzu.txt"

CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+)\.$")


def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total, prev = 0, 0
    for ch in reversed(s):
        v = vals.get(ch, 0)
        total += v if v >= prev else -v
        prev = v
    return total


def _is_editorial(paragraph: str) -> bool:
    """True if every non-empty line in the paragraph starts with whitespace
    (i.e. it's an indented commentary/note block, not main prose)."""
    lines = [l for l in paragraph.splitlines() if l.strip()]
    if not lines:
        return True
    return all(l[0] == " " or l[0] == "\t" for l in lines)


def _iter_chuang_tzu_passages(txt_path: str):
    with open(txt_path, encoding="utf-8-sig") as f:
        raw = f.read()

    # Trim Gutenberg boilerplate
    start = raw.find("*** START OF THE PROJECT GUTENBERG")
    end   = raw.find("*** END OF THE PROJECT GUTENBERG")
    if start != -1:
        raw = raw[raw.index("\n", start) + 1:]
    if end != -1:
        raw = raw[:end]

    # Normalise line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Split into paragraphs on blank lines
    paragraphs = re.split(r"\n{2,}", raw)

    current_chapter_num  = None
    current_chapter_rom  = None
    current_chapter_title = None
    expect_title = False   # next non-empty, non-editorial para is the chapter title
    para_idx = 0

    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue

        # Detect CHAPTER heading
        m = CHAPTER_RE.match(stripped)
        if m:
            current_chapter_rom   = m.group(1)
            current_chapter_num   = _roman_to_int(current_chapter_rom)
            current_chapter_title = None
            expect_title = True
            para_idx = 0
            continue

        # Next clean line after chapter heading is the title
        if expect_title and not _is_editorial(para):
            current_chapter_title = stripped.rstrip(".")
            expect_title = False
            continue

        # Skip until we're inside a chapter
        if current_chapter_num is None:
            continue

        # Skip editorial/commentary (indented) paragraphs
        if _is_editorial(para):
            continue

        text = stripped
        if len(text) < 30:
            continue

        para_idx += 1
        book = f"{current_chapter_rom} — {current_chapter_title}" if current_chapter_title else current_chapter_rom
        yield {
            "book":        book,
            "chapter_num": current_chapter_num,
            "unit_number": para_idx,
            "unit_label":  f"Chuang Tzu {current_chapter_rom}:{para_idx}",
            "text":        text,
        }


def ingest_chuang_tzu(
    txt_path: str = TXT_PATH,
    corpus_db_conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[int, list[int]]:
    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Taoist",
        name="Chuang Tzu (Giles)",
        type="scripture",
        language="en",
        era="ancient",
        metadata={
            "translator": "Herbert A. Giles",
            "year": 1889,
            "source": "Project Gutenberg #59709",
        },
    )

    existing = conn.execute(
        "SELECT id FROM corpus WHERE name = ?", [corpus_record.name]
    ).fetchone()
    if existing:
        corpus_id = existing[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
        ).fetchone()[0]
        if count > 0:
            print(f"[Chuang Tzu] already ingested ({count:,} passages) — skipping")
            return corpus_id, [
                r[0] for r in conn.execute(
                    "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
                ).fetchall()
            ]
        print("[Chuang Tzu] corpus row exists but 0 passages — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)
    print(f"[Chuang Tzu] corpus_id={corpus_id}, parsing {txt_path}...")

    passages = [
        PassageRecord(
            corpus_id=corpus_id,
            book=row["book"],
            section=None,
            unit_number=row["unit_number"],
            unit_label=row["unit_label"],
            text=row["text"],
            metadata={"chapter_num": row["chapter_num"]},
        )
        for row in _iter_chuang_tzu_passages(txt_path)
    ]

    print(f"[Chuang Tzu] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Chuang Tzu] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


if __name__ == "__main__":
    conn = get_conn()
    corpus_id, passage_ids = ingest_chuang_tzu(corpus_db_conn=conn)

    total = conn.execute(
        "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
    ).fetchone()[0]
    print(f"\nChuang Tzu in DB: {total:,} passages")

    sample = conn.execute(
        "SELECT book, unit_label, text FROM passage WHERE corpus_id = ? ORDER BY id LIMIT 6",
        [corpus_id],
    ).fetchall()
    print("\nSample passages:")
    for book, label, text in sample:
        print(f"  [{label}] {text[:90]}...")

    conn.close()
