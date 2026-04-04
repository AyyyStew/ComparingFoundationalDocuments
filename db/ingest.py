"""
db/ingest.py

Loaders that read source data and insert into the DuckDB corpus.

Currently implemented:
  - ingest_scrollmapper_bible()  — handles KJV, ACV, YLT, BBE

Adding a new corpus: implement a function that yields PassageRecord instances
and calls insert_passages(), following the same pattern.
"""

import json
import sqlite3
from typing import Generator

import duckdb

from db.models import CorpusRecord, PassageRecord
from db.schema import get_conn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_create_tradition(conn: duckdb.DuckDBPyConnection, name: str) -> int:
    row = conn.execute(
        "SELECT id FROM corpus_tradition WHERE name = ?", [name]
    ).fetchone()
    if row:
        return row[0]
    conn.execute("INSERT INTO corpus_tradition (name) VALUES (?)", [name])
    return conn.execute(
        "SELECT id FROM corpus_tradition WHERE name = ?", [name]
    ).fetchone()[0]


def _get_or_create_corpus(conn: duckdb.DuckDBPyConnection, record: CorpusRecord) -> int:
    row = conn.execute(
        "SELECT id FROM corpus WHERE name = ?", [record.name]
    ).fetchone()
    if row:
        return row[0]

    tradition_id = _get_or_create_tradition(conn, record.tradition_name)
    conn.execute(
        """
        INSERT INTO corpus (tradition_id, name, type, language, era, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            tradition_id,
            record.name,
            record.type,
            record.language,
            record.era,
            json.dumps(record.metadata),
        ],
    )
    return conn.execute(
        "SELECT id FROM corpus WHERE name = ?", [record.name]
    ).fetchone()[0]


def insert_passages(
    conn: duckdb.DuckDBPyConnection,
    passages: list[PassageRecord],
    batch_size: int = 2000,
) -> list[int]:
    """Insert passages in batches. Returns list of inserted passage IDs in order."""
    ids = []
    for i in range(0, len(passages), batch_size):
        batch = passages[i : i + batch_size]
        rows = [
            (
                p.corpus_id,
                p.book,
                p.section,
                p.unit_number,
                p.unit_label,
                p.text,
                json.dumps(p.metadata),
            )
            for p in batch
        ]
        conn.executemany(
            """
            INSERT INTO passage (corpus_id, book, section, unit_number, unit_label, text, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        # fetch back the IDs we just inserted (DuckDB lacks RETURNING in executemany)
        batch_ids = conn.execute(
            f"SELECT id FROM passage ORDER BY id DESC LIMIT {len(batch)}"
        ).fetchall()
        ids = [r[0] for r in reversed(batch_ids)] + ids

    return ids


# ── Scrollmapper Bible loader ─────────────────────────────────────────────────

BIBLE_TRANSLATION_META = {
    "KJV": {"full_name": "King James Version",       "era": "modern", "year": 1611},
    "ACV": {"full_name": "A Conservative Version",   "era": "modern", "year": 2005},
    "YLT": {"full_name": "Young's Literal Translation", "era": "modern", "year": 1898},
    "BBE": {"full_name": "Bible in Basic English",   "era": "modern", "year": 1949},
}


def _iter_bible_verses(
    db_path: str, translation: str
) -> Generator[dict, None, None]:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        f"""
        SELECT b.id, b.name, v.chapter, v.verse, v.text
        FROM {translation}_verses v
        JOIN {translation}_books b ON v.book_id = b.id
        ORDER BY b.id, v.chapter, v.verse
        """
    )
    for book_id, book_name, chapter, verse, text in cursor:
        yield {
            "book_id": book_id,
            "book":    book_name,
            "chapter": chapter,
            "verse":   verse,
            "text":    text,
        }
    conn.close()


def ingest_scrollmapper_bible(
    translation: str,
    bible_db_path: str,
    corpus_db_conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[int, list[int]]:
    """
    Load one bible translation into the corpus DB.

    Returns (corpus_id, [passage_ids]).
    Safe to call again — skips if corpus already exists.
    """
    conn = corpus_db_conn or get_conn()

    meta = BIBLE_TRANSLATION_META.get(translation, {})
    corpus_record = CorpusRecord(
        tradition_name="Abrahamic",
        name=f"Bible — {translation}" + (f" ({meta.get('full_name')})" if meta else ""),
        type="scripture",
        language="en",
        era=meta.get("era"),
        metadata={k: v for k, v in meta.items() if k != "era"},
    )

    # Skip if already ingested (must have passages — a corpus row with 0 passages
    # means a previous run crashed mid-ingest and needs to be resumed)
    existing = conn.execute(
        "SELECT id FROM corpus WHERE name = ?", [corpus_record.name]
    ).fetchone()
    if existing:
        corpus_id = existing[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
        ).fetchone()[0]
        if count > 0:
            print(f"[{translation}] already ingested ({count:,} passages) — skipping")
            passage_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
                ).fetchall()
            ]
            return corpus_id, passage_ids
        print(f"[{translation}] corpus exists but has 0 passages (previous crash?) — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)
    print(f"[{translation}] corpus_id={corpus_id}, loading verses...")

    passages = [
        PassageRecord(
            corpus_id=corpus_id,
            book=row["book"],
            section=str(row["chapter"]),
            unit_number=row["verse"],
            unit_label=f"{row['chapter']}:{row['verse']}",
            text=row["text"],
        )
        for row in _iter_bible_verses(bible_db_path, translation)
        if row["text"] and row["text"].strip()
    ]

    print(f"[{translation}] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[{translation}] done — {len(passage_ids):,} passages inserted")

    return corpus_id, passage_ids


# ── Bhagavad Gita loader ─────────────────────────────────────────────────────

def ingest_bhagavad_gita(
    csv_path: str,
    corpus_db_conn=None,
) -> tuple[int, list[int]]:
    """
    Load the Bhagavad Gita CSV into the corpus DB.
    Expects columns: chapter_number, chapter_title, chapter_verse, translation
    Returns (corpus_id, [passage_ids]).
    """
    import pandas as pd

    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Dharmic",
        name="Bhagavad Gita",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"source": csv_path},
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
            print(f"[Bhagavad Gita] already ingested ({count:,} passages) — skipping")
            passage_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
                ).fetchall()
            ]
            return corpus_id, passage_ids
        print("[Bhagavad Gita] corpus exists but 0 passages (previous crash?) — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)

    df = pd.read_csv(csv_path)
    df = df[df["translation"].notna() & (df["translation"].str.strip() != "")]

    passages = []
    for _, row in df.iterrows():
        # chapter_number is "Chapter 1" etc — extract the int
        chapter_str = str(row["chapter_number"]).strip()
        chapter_num = int("".join(filter(str.isdigit, chapter_str))) if any(
            c.isdigit() for c in chapter_str
        ) else None

        # chapter_verse is "1.1" or "1.4 – 1.6" — use first number as unit_number
        verse_label = str(row["chapter_verse"]).strip()
        first_part  = verse_label.split("–")[0].split("-")[0].strip()
        try:
            unit_number = int(first_part.split(".")[-1])
        except ValueError:
            unit_number = None

        passages.append(PassageRecord(
            corpus_id=corpus_id,
            book=str(row["chapter_title"]).strip(),
            section=str(chapter_num) if chapter_num is not None else None,
            unit_number=unit_number,
            unit_label=verse_label,
            text=str(row["translation"]).strip(),
            metadata={"chapter_number": chapter_str},
        ))

    print(f"[Bhagavad Gita] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Bhagavad Gita] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Dhammapada loader ────────────────────────────────────────────────────────

def ingest_dhammapada(
    txt_path: str,
    corpus_db_conn=None,
) -> tuple[int, list[int]]:
    """
    Parse the Project Gutenberg plain-text Dhammapada (F. Max Müller translation).
    Verses are numbered lines like '1. All that we are...' and may wrap across
    multiple lines. Chapters are 'Chapter I. The Twin-Verses' headings.
    """
    import re

    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Buddhist",
        name="Dhammapada (Müller)",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"translator": "F. Max Müller", "source": "Project Gutenberg #2017"},
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
            print(f"[Dhammapada] already ingested ({count:,} passages) — skipping")
            passage_ids = [r[0] for r in conn.execute(
                "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
            ).fetchall()]
            return corpus_id, passage_ids
        print("[Dhammapada] corpus exists but 0 passages — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Strip Gutenberg header/footer
    start = raw.find("*** START OF THE PROJECT GUTENBERG")
    end   = raw.find("*** END OF THE PROJECT GUTENBERG")
    if start != -1:
        raw = raw[raw.find("\n", start) + 1:]
    if end != -1:
        raw = raw[:end]

    # Chapter heading pattern: "Chapter I. The Twin-Verses"
    chapter_pattern = re.compile(
        r"^Chapter\s+([IVXLC]+)\.\s+(.+)$", re.MULTILINE
    )
    # Verse start pattern: "1. " or "123. " at the start of a line
    verse_start_pattern = re.compile(r"^\d+\.", re.MULTILINE)

    # Split into chapter blocks
    chapter_splits = list(chapter_pattern.finditer(raw))
    passages = []

    for i, ch_match in enumerate(chapter_splits):
        chapter_roman = ch_match.group(1).strip()
        chapter_title = ch_match.group(2).strip()
        # convert Roman numeral to int
        roman_map = {"I":1,"V":5,"X":10,"L":50,"C":100}
        chapter_num = 0
        prev = 0
        for ch in reversed(chapter_roman):
            val = roman_map.get(ch, 0)
            chapter_num += val if val >= prev else -val
            prev = val

        # Text block for this chapter
        block_start = ch_match.end()
        block_end   = chapter_splits[i + 1].start() if i + 1 < len(chapter_splits) else len(raw)
        block       = raw[block_start:block_end]

        # Split block into individual verse chunks
        verse_starts = [m.start() for m in verse_start_pattern.finditer(block)]
        for j, vs in enumerate(verse_starts):
            ve      = verse_starts[j + 1] if j + 1 < len(verse_starts) else len(block)
            chunk   = block[vs:ve].strip()
            # Extract verse number
            dot_idx = chunk.index(".")
            try:
                verse_num = int(chunk[:dot_idx].strip())
            except ValueError:
                continue
            text = chunk[dot_idx + 1:].strip()
            # Collapse wrapped lines
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            if not text:
                continue

            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=chapter_title,
                section=str(chapter_num),
                unit_number=verse_num,
                unit_label=f"{chapter_num}.{verse_num}",
                text=text,
            ))

    print(f"[Dhammapada] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Dhammapada] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Dao De Jing loader ───────────────────────────────────────────────────────

def ingest_dao_de_jing(
    html_path: str,
    corpus_db_conn=None,
) -> tuple[int, list[int]]:
    """
    Parse the Project Gutenberg HTML Dao De Jing (Linnell minimalist translation).
    Each chapter is a single passage — the DDJ has no named sub-verses,
    so chapter is the atomic unit. Chinese text is stripped; only English kept.
    """
    import re
    from bs4 import BeautifulSoup

    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Taoist",
        name="Dao De Jing (Linnell)",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"translator": "Bruce R. Linnell, PhD", "source": "Project Gutenberg #49965"},
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
            print(f"[Dao De Jing] already ingested ({count:,} passages) — skipping")
            passage_ids = [r[0] for r in conn.execute(
                "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
            ).fetchall()]
            return corpus_id, passage_ids
        print("[Dao De Jing] corpus exists but 0 passages — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)

    word_to_num = {
        "One":1,"Two":2,"Three":3,"Four":4,"Five":5,"Six":6,"Seven":7,"Eight":8,
        "Nine":9,"Ten":10,"Eleven":11,"Twelve":12,"Thirteen":13,"Fourteen":14,
        "Fifteen":15,"Sixteen":16,"Seventeen":17,"Eighteen":18,"Nineteen":19,
        "Twenty":20,"Twenty-one":21,"Twenty-two":22,"Twenty-three":23,"Twenty-four":24,
        "Twenty-five":25,"Twenty-six":26,"Twenty-seven":27,"Twenty-eight":28,
        "Twenty-nine":29,"Thirty":30,"Thirty-one":31,"Thirty-two":32,"Thirty-three":33,
        "Thirty-four":34,"Thirty-five":35,"Thirty-six":36,"Thirty-seven":37,
        "Thirty-eight":38,"Thirty-nine":39,"Forty":40,"Forty-one":41,"Forty-two":42,
        "Forty-three":43,"Forty-four":44,"Forty-five":45,"Forty-six":46,"Forty-seven":47,
        "Forty-eight":48,"Forty-nine":49,"Fifty":50,"Fifty-one":51,"Fifty-two":52,
        "Fifty-three":53,"Fifty-four":54,"Fifty-five":55,"Fifty-six":56,"Fifty-seven":57,
        "Fifty-eight":58,"Fifty-nine":59,"Sixty":60,"Sixty-one":61,"Sixty-two":62,
        "Sixty-three":63,"Sixty-four":64,"Sixty-five":65,"Sixty-six":66,"Sixty-seven":67,
        "Sixty-eight":68,"Sixty-nine":69,"Seventy":70,"Seventy-one":71,"Seventy-two":72,
        "Seventy-three":73,"Seventy-four":74,"Seventy-five":75,"Seventy-six":76,
        "Seventy-seven":77,"Seventy-eight":78,"Seventy-nine":79,"Eighty":80,"Eighty-one":81,
    }

    with open(html_path, encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Collect (chapter_num, content_tag) pairs
    # Each chapter heading <p> is immediately followed by a sibling <p> with the content
    chapter_tags = []
    for tag in soup.find_all("p"):
        b = tag.find("b")
        if not b:
            continue
        m = re.match(r"Chapter\s+(\S+(?:-\w+)?)", b.get_text(strip=True))
        if not m:
            continue
        word = m.group(1)
        num  = word_to_num.get(word)
        if num is None:
            continue
        # Content is in the next <div> sibling, not a <p>
        sib = tag.find_next_sibling("div")
        if sib:
            chapter_tags.append((num, sib))

    # Strip Chinese characters (CJK Unified Ideographs block) and notation symbols
    whitespace_re = re.compile(r"\s+")
    noise_re      = re.compile(r"[\(\)\[\]•♦_]+")

    passages = []
    for chapter_num, tag in chapter_tags:
        # Table row 0, cell 1 = clean English translation (no Chinese, no notes)
        table = tag.find("table")
        if not table:
            continue
        rows = table.find_all("tr")
        if not rows:
            continue
        cells = rows[0].find_all("td")
        if len(cells) < 2:
            continue
        raw_text = cells[1].get_text(separator=" ")
        # Strip leftover notation symbols and collapse whitespace
        text = noise_re.sub(" ", raw_text)
        text = whitespace_re.sub(" ", text).strip()
        if not text:
            continue
        passages.append(PassageRecord(
            corpus_id=corpus_id,
            book="Dao De Jing",
            section=str(chapter_num),
            unit_number=chapter_num,
            unit_label=str(chapter_num),
            text=text,
        ))

    passages.sort(key=lambda p: p.unit_number)

    print(f"[Dao De Jing] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Dao De Jing] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Yoga Sutras loader ───────────────────────────────────────────────────────

def ingest_yoga_sutras(
    txt_path: str,
    corpus_db_conn=None,
) -> tuple[int, list[int]]:
    """
    Parse the Project Gutenberg Yoga Sutras of Patanjali (Johnston translation).
    Sutras are numbered lines like '1. OM: Here follows...' within 4 books.
    Commentary paragraphs are stored in metadata, not in passage text.
    """
    import re

    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Dharmic",
        name="Yoga Sutras of Patanjali (Johnston)",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"translator": "Charles Johnston", "source": "Project Gutenberg #2526"},
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
            print(f"[Yoga Sutras] already ingested ({count:,} passages) — skipping")
            passage_ids = [r[0] for r in conn.execute(
                "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
            ).fetchall()]
            return corpus_id, passage_ids
        print("[Yoga Sutras] corpus exists but 0 passages — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Strip Gutenberg header/footer
    start = raw.find("*** START OF THE PROJECT GUTENBERG")
    end   = raw.find("*** END OF THE PROJECT GUTENBERG")
    if start != -1:
        raw = raw[raw.find("\n", start) + 1:]
    if end != -1:
        raw = raw[:end]

    book_pattern  = re.compile(r"^BOOK\s+(I{1,3}V?|IV|V?I{0,3})$", re.MULTILINE)
    sutra_pattern = re.compile(r"^(\d+)\.\s+(.+?)(?=\n\n|\n\d+\.)", re.MULTILINE | re.DOTALL)

    book_splits = list(book_pattern.finditer(raw))
    roman_map   = {"I": 1, "II": 2, "III": 3, "IV": 4}

    passages = []
    for i, book_match in enumerate(book_splits):
        book_roman  = book_match.group(1).strip()
        book_num    = roman_map.get(book_roman, i + 1)
        book_label  = f"Book {book_roman}"

        block_start = book_match.end()
        block_end   = book_splits[i + 1].start() if i + 1 < len(book_splits) else len(raw)
        block       = raw[block_start:block_end]

        for sutra_match in sutra_pattern.finditer(block):
            sutra_num  = int(sutra_match.group(1))
            full_chunk = sutra_match.group(2).strip()

            # First paragraph = sutra text, remaining = commentary
            paragraphs    = re.split(r"\n\n+", full_chunk)
            sutra_text    = re.sub(r"\s+", " ", paragraphs[0]).strip()
            commentary    = " ".join(
                re.sub(r"\s+", " ", p).strip()
                for p in paragraphs[1:] if p.strip()
            )

            if not sutra_text:
                continue

            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=book_label,
                section=str(book_num),
                unit_number=sutra_num,
                unit_label=f"{book_num}.{sutra_num}",
                text=sutra_text,
                metadata={"commentary": commentary} if commentary else {},
            ))

    print(f"[Yoga Sutras] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Yoga Sutras] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Shared text chunker ───────────────────────────────────────────────────────

def chunk_text(text: str, sentences_per_chunk: int = 4) -> list[str]:
    """
    Split text into chunks of ~sentences_per_chunk sentences.
    Handles common abbreviations to avoid false splits.
    Returns a list of non-empty chunk strings.
    """
    import re

    # Temporarily mask common abbreviations so their periods don't split
    abbrevs = r"\b(Mr|Mrs|Ms|Dr|St|vs|etc|cf|Jr|Sr|Rev|Gen|Lt|Col|Sgt|Prof|Vol|pp|No)\."
    text = re.sub(abbrevs, r"\1⟨DOT⟩", text)

    # Split on ". " or "! " or "? " followed by a capital letter or quote
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", text.strip())
    # Restore masked dots
    sentences = [s.replace("⟨DOT⟩", ".").strip() for s in sentences if s.strip()]

    chunks = []
    for i in range(0, max(1, len(sentences)), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


# ── Literature loaders ────────────────────────────────────────────────────────

def _strip_gutenberg(raw: str) -> str:
    start = raw.find("*** START OF THE PROJECT GUTENBERG")
    end   = raw.find("*** END OF THE PROJECT GUTENBERG")
    if start != -1:
        raw = raw[raw.find("\n", start) + 1:]
    if end != -1:
        raw = raw[:end]
    return raw.strip()


def _ingest_novel(
    txt_path: str,
    corpus_record,
    chapter_pattern,
    chapter_name_fn,  # (match) -> (chapter_num: int, chapter_title: str)
    conn,
    sentences_per_chunk: int = 4,
) -> tuple[int, list[int]]:
    """Shared logic for chapter-based novels."""
    import re

    existing = conn.execute(
        "SELECT id FROM corpus WHERE name = ?", [corpus_record.name]
    ).fetchone()
    if existing:
        corpus_id = existing[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
        ).fetchone()[0]
        if count > 0:
            print(f"[{corpus_record.name}] already ingested ({count:,} passages) — skipping")
            return corpus_id, [r[0] for r in conn.execute(
                "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
            ).fetchall()]
        print(f"[{corpus_record.name}] corpus exists but 0 passages — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    splits  = list(chapter_pattern.finditer(raw))
    passages = []

    for i, match in enumerate(splits):
        chapter_num, chapter_title = chapter_name_fn(match)
        block_start = match.end()
        block_end   = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
        block       = raw[block_start:block_end].strip()
        # Remove any sub-headings (all-caps lines) and stage directions
        block = re.sub(r"\n[A-Z][A-Z\s]{4,}\n", "\n", block)

        chunks = chunk_text(block, sentences_per_chunk)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=chapter_title or f"Chapter {chapter_num}",
                section=str(chapter_num),
                unit_number=idx + 1,
                unit_label=f"{chapter_num}.{idx + 1}",
                text=chunk,
            ))

    print(f"[{corpus_record.name}] inserting {len(passages):,} passages ({len(splits)} chapters)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[{corpus_record.name}] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


def ingest_frankenstein(txt_path: str, conn=None) -> tuple[int, list[int]]:
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Literature",
            name="Frankenstein (Shelley)",
            type="literature",
            language="en",
            era="modern",
            metadata={"author": "Mary Wollstonecraft Shelley", "year": 1818,
                      "source": "Project Gutenberg #84"},
        ),
        chapter_pattern=re.compile(r"^Chapter\s+\d+", re.MULTILINE | re.IGNORECASE),
        chapter_name_fn=lambda m: (
            int(re.search(r"\d+", m.group()).group()),
            m.group().strip(),
        ),
        conn=conn,
    )


def ingest_pride_and_prejudice(txt_path: str, conn=None) -> tuple[int, list[int]]:
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Literature",
            name="Pride and Prejudice (Austen)",
            type="literature",
            language="en",
            era="modern",
            metadata={"author": "Jane Austen", "year": 1813,
                      "source": "Project Gutenberg #1342"},
        ),
        chapter_pattern=re.compile(r"^CHAPTER\s+[IVXLC\d]+\.?", re.MULTILINE),
        chapter_name_fn=lambda m: (
            _roman_to_int(re.search(r"[IVXLC\d]+", m.group()).group()),
            m.group().strip(),
        ),
        conn=conn,
    )


def ingest_don_quixote(txt_path: str, conn=None) -> tuple[int, list[int]]:
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Literature",
            name="Don Quixote (Cervantes)",
            type="literature",
            language="en",
            era="modern",
            metadata={"author": "Miguel de Cervantes", "year": 1605,
                      "source": "Project Gutenberg #996"},
        ),
        chapter_pattern=re.compile(r"^CHAPTER\s+[IVXLC]+", re.MULTILINE),
        chapter_name_fn=lambda m: (
            _roman_to_int(re.search(r"[IVXLC]+", m.group().replace("CHAPTER", "")).group()),
            m.group().strip(),
        ),
        conn=conn,
    )


def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    result, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        result += v if v >= prev else -v
        prev = v
    return result or 0


def ingest_romeo_and_juliet(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Romeo & Juliet: each scene is one passage.
    Stage directions and character names are stripped; only dialogue kept.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Literature",
        name="Romeo and Juliet (Shakespeare)",
        type="literature",
        language="en",
        era="modern",
        metadata={"author": "William Shakespeare", "year": 1597,
                  "source": "Project Gutenberg #1513"},
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
            print(f"[Romeo and Juliet] already ingested ({count:,} passages) — skipping")
            return corpus_id, [r[0] for r in conn.execute(
                "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
            ).fetchall()]

    corpus_id = _get_or_create_corpus(conn, corpus_record)

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    act_pattern   = re.compile(r"^ACT\s+([IVX]+)\s*$", re.MULTILINE)
    scene_pattern = re.compile(r"^SCENE\s+([IVX]+)\.\s+(.+)$", re.MULTILINE)

    # Find all acts and scenes with positions
    acts = {m.group(1): m.start() for m in act_pattern.finditer(raw)}

    # Build list of (act_num, scene_num, scene_title, start_pos)
    scene_list = []
    current_act = None
    for m in re.finditer(r"^(ACT\s+([IVX]+)|SCENE\s+([IVX]+)\.\s+(.+))$", raw, re.MULTILINE):
        if m.group(2):
            current_act = _roman_to_int(m.group(2))
        elif m.group(3) and current_act:
            scene_list.append((current_act, _roman_to_int(m.group(3)), m.group(4).strip(), m.end()))

    passages = []
    for i, (act_num, scene_num, scene_title, start) in enumerate(scene_list):
        end   = scene_list[i + 1][3] - len(scene_list[i + 1][2]) - 10 if i + 1 < len(scene_list) else len(raw)
        block = raw[start:end]

        # Strip stage directions (lines starting with "Enter"/"Exit"/"Exeunt" or all-whitespace-indented)
        block = re.sub(r"^\s*(Enter|Exit|Exeunt)[^\n]*\n", "", block, flags=re.MULTILINE)
        # Strip character names (ALL CAPS lines)
        block = re.sub(r"^[A-Z][A-Z\s]+\.\s*$", "", block, flags=re.MULTILINE)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", block).strip()

        if not text:
            continue

        passages.append(PassageRecord(
            corpus_id=corpus_id,
            book=f"Act {act_num}",
            section=str(act_num),
            unit_number=scene_num,
            unit_label=f"{act_num}.{scene_num}",
            text=text,
            metadata={"scene_title": scene_title},
        ))

    print(f"[Romeo and Juliet] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Romeo and Juliet] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Historical document loaders ───────────────────────────────────────────────

def _get_or_skip(conn, corpus_record):
    """Return (corpus_id, passage_ids_or_None). passage_ids is None if ingest needed."""
    existing = conn.execute(
        "SELECT id FROM corpus WHERE name = ?", [corpus_record.name]
    ).fetchone()
    if existing:
        corpus_id = existing[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
        ).fetchone()[0]
        if count > 0:
            print(f"[{corpus_record.name}] already ingested ({count:,} passages) — skipping")
            passage_ids = [r[0] for r in conn.execute(
                "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
            ).fetchall()]
            return corpus_id, passage_ids
        print(f"[{corpus_record.name}] corpus exists but 0 passages — re-ingesting")
    corpus_id = _get_or_create_corpus(conn, corpus_record)
    return corpus_id, None


def ingest_code_of_hammurabi(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse the Project Gutenberg Code of Hammurabi (C. H. W. Johns translation).
    Each 'section N.' is one passage.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Historical",
        name="Code of Hammurabi",
        type="legal",
        language="en",
        era="ancient",
        metadata={"author": "Hammurabi", "translator": "C. H. W. Johns",
                  "source": "Project Gutenberg #17150"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # Start from "THE TEXT OF THE CODE"
    code_start = raw.upper().find("THE TEXT OF THE CODE")
    if code_start != -1:
        raw = raw[raw.find("\n", code_start) + 1:]

    section_re = re.compile(r"^section\s+(\d+)\.\s+", re.MULTILINE | re.IGNORECASE)
    matches = list(section_re.finditer(raw))

    passages = []
    for i, m in enumerate(matches):
        law_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text = re.sub(r"\s+", " ", raw[start:end]).strip()
        if not text:
            continue
        passages.append(PassageRecord(
            corpus_id=corpus_id,
            book="The Code",
            section="1",
            unit_number=law_num,
            unit_label=f"§{law_num}",
            text=text,
        ))

    print(f"[Code of Hammurabi] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Code of Hammurabi] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


def ingest_luther_theses(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse Luther's 95 Theses (Project Gutenberg plain text).
    Each numbered thesis is one passage.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Historical",
        name="Luther's 95 Theses",
        type="theological",
        language="en",
        era="modern",
        metadata={"author": "Martin Luther", "year": 1517,
                  "source": "Project Gutenberg #274"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # Normalize mid-sentence thesis breaks like "priest.  8. The" → "priest.\n8. The"
    raw = re.sub(r"\.\s{2,}(\d{1,2})\.\s+", r".\n\1. ", raw)

    thesis_re = re.compile(r"^(\d{1,2})\.\s+", re.MULTILINE)
    matches = list(thesis_re.finditer(raw))

    passages = []
    for i, m in enumerate(matches):
        thesis_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text = re.sub(r"\s+", " ", raw[start:end]).strip()
        if not text:
            continue
        passages.append(PassageRecord(
            corpus_id=corpus_id,
            book="95 Theses",
            section="1",
            unit_number=thesis_num,
            unit_label=f"Thesis {thesis_num}",
            text=text,
        ))

    print(f"[Luther's 95 Theses] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Luther's 95 Theses] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


def ingest_magna_carta(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse the Magna Carta (Project Gutenberg plain text).
    Uses the first numbered-clause version only. Each (N) clause is one passage.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Historical",
        name="Magna Carta",
        type="legal",
        language="en",
        era="medieval",
        metadata={"year": 1215, "source": "Project Gutenberg #10000"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Strip Gutenberg header
    start = raw.find("*** START OF THE PROJECT GUTENBERG")
    if start != -1:
        raw = raw[raw.find("\n", start) + 1:]

    # Use only the first version — stop before second version or strange-chars marker
    for marker in ["***Strange characters", "Magna Carta 1215", "The Magna Carta (The Great Charter)"]:
        pos = raw.find(marker)
        if pos != -1:
            raw = raw[:pos]
            break

    clause_re = re.compile(r"^\s*\((\d+)\)\s+", re.MULTILINE)
    matches = list(clause_re.finditer(raw))

    passages = []
    for i, m in enumerate(matches):
        clause_num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text = re.sub(r"\s+", " ", raw[start:end]).strip()
        if not text:
            continue
        passages.append(PassageRecord(
            corpus_id=corpus_id,
            book="Magna Carta",
            section="1",
            unit_number=clause_num,
            unit_label=f"Clause {clause_num}",
            text=text,
        ))

    print(f"[Magna Carta] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Magna Carta] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


def ingest_us_constitution(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse the US Constitution (Project Gutenberg plain text).
    Preamble + each Article/Section is one passage.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Historical",
        name="US Constitution",
        type="legal",
        language="en",
        era="modern",
        metadata={"year": 1787, "source": "Project Gutenberg #5"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # Stop before signatory list (ARTICLE SEVEN ends the text)
    pg_footer = raw.find("*** END OF THE PROJECT GUTENBERG")
    if pg_footer != -1:
        raw = raw[:pg_footer]

    word_to_num = {
        "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4,
        "FIVE": 5, "SIX": 6, "SEVEN": 7,
    }
    article_re = re.compile(r"^(?:Article|ARTICLE)\s+(\w+)", re.MULTILINE)
    section_re = re.compile(r"^Section\s+(\d+)\.\s+", re.MULTILINE)

    article_matches = list(article_re.finditer(raw))
    passages = []

    # Preamble — everything before the first Article
    if article_matches:
        preamble_block = raw[:article_matches[0].start()]
        # Strip boilerplate notes at the top (ALL-CAPS lines, short lines)
        preamble_text = re.sub(r"\s+", " ", preamble_block).strip()
        # Find "We the people" as real start
        we_pos = preamble_text.lower().find("we the people")
        if we_pos != -1:
            preamble_text = preamble_text[we_pos:].strip()
        if preamble_text:
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book="Preamble",
                section="0",
                unit_number=1,
                unit_label="Preamble",
                text=preamble_text,
            ))

    for ai, art_match in enumerate(article_matches):
        raw_num = art_match.group(1).upper()
        article_num = word_to_num.get(raw_num) or (int(raw_num) if raw_num.isdigit() else None)
        if article_num is None:
            continue
        art_start = art_match.end()
        art_end = article_matches[ai + 1].start() if ai + 1 < len(article_matches) else len(raw)
        art_block = raw[art_start:art_end]

        section_matches = list(section_re.finditer(art_block))
        if not section_matches:
            text = re.sub(r"\s+", " ", art_block).strip()
            if text:
                passages.append(PassageRecord(
                    corpus_id=corpus_id,
                    book=f"Article {article_num}",
                    section=str(article_num),
                    unit_number=1,
                    unit_label=f"Article {article_num}",
                    text=text,
                ))
        else:
            for si, sec_match in enumerate(section_matches):
                section_num = int(sec_match.group(1))
                sec_start = sec_match.end()
                sec_end = section_matches[si + 1].start() if si + 1 < len(section_matches) else len(art_block)
                text = re.sub(r"\s+", " ", art_block[sec_start:sec_end]).strip()
                if text:
                    passages.append(PassageRecord(
                        corpus_id=corpus_id,
                        book=f"Article {article_num}",
                        section=str(article_num),
                        unit_number=section_num,
                        unit_label=f"Article {article_num}, Section {section_num}",
                        text=text,
                    ))

    print(f"[US Constitution] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[US Constitution] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


def ingest_communist_manifesto(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse the Communist Manifesto (Project Gutenberg plain text).
    Each paragraph within a section is one passage.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Historical",
        name="Communist Manifesto",
        type="philosophy",
        language="en",
        era="modern",
        metadata={"authors": ["Karl Marx", "Friedrich Engels"], "year": 1848,
                  "source": "Project Gutenberg #61"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # Section headers appear as "I.\nBOURGEOIS AND PROLETARIANS" (roman numeral + title)
    section_re = re.compile(
        r"^([IVX]+)\.\n([A-Z][A-Z ,\-]+)\n", re.MULTILINE
    )
    section_matches = list(section_re.finditer(raw))

    passages = []

    # Preface/preamble before first section
    intro_end = section_matches[0].start() if section_matches else len(raw)
    intro_block = raw[:intro_end].strip()
    if intro_block:
        for idx, para in enumerate(re.split(r"\n\n+", intro_block)):
            text = re.sub(r"\s+", " ", para).strip()
            if len(text) > 40:
                passages.append(PassageRecord(
                    corpus_id=corpus_id,
                    book="Preface",
                    section="0",
                    unit_number=idx + 1,
                    unit_label=f"Preface.{idx + 1}",
                    text=text,
                ))

    for si, sec_match in enumerate(section_matches):
        section_num = _roman_to_int(sec_match.group(1))
        section_title = sec_match.group(2).strip().title()
        sec_start = sec_match.end()
        sec_end = section_matches[si + 1].start() if si + 1 < len(section_matches) else len(raw)
        sec_block = raw[sec_start:sec_end]

        para_idx = 0
        for para in re.split(r"\n\n+", sec_block):
            text = re.sub(r"\s+", " ", para).strip()
            # Skip short lines (sub-headings, blank separators)
            if len(text) < 40:
                continue
            para_idx += 1
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=section_title,
                section=str(section_num),
                unit_number=para_idx,
                unit_label=f"{section_num}.{para_idx}",
                text=text,
            ))

    print(f"[Communist Manifesto] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Communist Manifesto] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


def ingest_federalist_papers(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse The Federalist Papers (Project Gutenberg plain text).
    Each paper is chunked into 4-sentence passages using chunk_text().
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Historical",
        name="Federalist Papers",
        type="political",
        language="en",
        era="modern",
        metadata={"authors": ["Alexander Hamilton", "John Jay", "James Madison"],
                  "year": 1788, "source": "Project Gutenberg #1404"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    paper_re = re.compile(r"^FEDERALIST No\.\s+(\d+)", re.MULTILINE)
    matches = list(paper_re.finditer(raw))

    passages = []
    for i, m in enumerate(matches):
        paper_num = int(m.group(1))
        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        block = raw[block_start:block_end].strip()

        # Strip short header lines (title, date, author name) before the body
        # Body paragraphs are substantially longer than headers
        paras = re.split(r"\n\n+", block)
        body_paras = []
        found_body = False
        for para in paras:
            para = para.strip()
            if not found_body and len(para) < 80:
                continue
            found_body = True
            body_paras.append(para)

        body = " ".join(body_paras)
        chunks = chunk_text(body, sentences_per_chunk=4)

        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=f"Federalist No. {paper_num}",
                section=str(paper_num),
                unit_number=idx + 1,
                unit_label=f"No. {paper_num}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Federalist Papers] inserting {len(passages):,} passages ({len(matches)} papers)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Federalist Papers] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── News articles loader ──────────────────────────────────────────────────────

def ingest_news_articles(csv_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Load news articles CSV (columns: Article, Date, Heading, NewsType).
    Each article is chunked into 4-sentence passages via chunk_text().
    book    = NewsType (e.g. "sports", "business")
    section = article index (str)
    unit    = chunk index within article
    metadata stores Heading and Date.
    """
    import re
    import pandas as pd

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="News",
        name="News Articles",
        type="news",
        language="en",
        era="modern",
        metadata={"source": csv_path, "categories": ["sports", "business"]},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df[df["Article"].notna() & (df["Article"].str.strip() != "")].reset_index(drop=True)

    passages = []
    for article_idx, row in df.iterrows():
        news_type = str(row.get("NewsType", "unknown")).strip().lower()
        heading   = str(row.get("Heading", "")).strip()
        date      = str(row.get("Date", "")).strip()
        text      = str(row["Article"]).strip()

        chunks = chunk_text(text, sentences_per_chunk=4)
        for chunk_idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=news_type,
                section=str(article_idx),
                unit_number=chunk_idx + 1,
                unit_label=f"{article_idx}.{chunk_idx + 1}",
                text=chunk,
                metadata={"heading": heading, "date": date},
            ))

    print(f"[News Articles] inserting {len(passages):,} passages ({len(df):,} articles)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[News Articles] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Quran loader ─────────────────────────────────────────────────────────────

def ingest_quran(csv_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Load the Quran CSV (Clear Quran translation) verse-level into the corpus DB.
    Verse granularity mirrors the Bible (one ayah = one passage).
    Columns used: surah_no, surah_name_en, ayah_no_surah, ayah_en, juz_no
    book = surah name, section = surah number, unit = ayah number within surah.
    """
    import pandas as pd

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Abrahamic",
        name="Quran (Clear Quran Translation)",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"translator": "Dr. Mustafa Khattab", "source": csv_path},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    df = pd.read_csv(csv_path)
    df = df[df["ayah_en"].notna() & (df["ayah_en"].str.strip() != "")].reset_index(drop=True)

    passages = []
    for _, row in df.iterrows():
        surah_no  = int(row["surah_no"])
        surah_name = str(row["surah_name_en"]).strip()
        ayah_no   = int(row["ayah_no_surah"])
        text      = str(row["ayah_en"]).strip()
        if not text:
            continue
        passages.append(PassageRecord(
            corpus_id=corpus_id,
            book=surah_name,
            section=str(surah_no),
            unit_number=ayah_no,
            unit_label=f"{surah_no}:{ayah_no}",
            text=text,
            metadata={"surah_no": surah_no, "juz_no": int(row.get("juz_no", 0) or 0)},
        ))

    print(f"[Quran] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Quran] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Upanishads loader ────────────────────────────────────────────────────────

def ingest_upanishads(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse the Project Gutenberg Upanishads (Swami Paramananda translation).
    Contains 3 Upanishads: Isa, Katha, Kena — split by heavily-indented title-case
    headers that mark the start of each body section. The file uses CRLF line endings.
    Only the first occurrence of each title is used (second = commentary notes).
    Text is chunked with chunk_text(). book = upanishad name, section = 1/2/3.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Dharmic",
        name="Upanishads (Paramananda)",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"translator": "Swami Paramananda", "source": "Project Gutenberg #3283"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Heavily-indented (≥5 spaces) title-case headers mark start of each Upanishad body.
    # Multiple occurrences exist; take only the first of each.
    content_pattern = re.compile(
        r"^\s{5,}(Isa-Upanishad|Katha-Upanishad|Kena-Upanishad)\s*$",
        re.MULTILINE,
    )
    all_matches = list(content_pattern.finditer(raw))
    seen, splits = set(), []
    for m in all_matches:
        key = m.group(1)
        if key not in seen:
            seen.add(key)
            splits.append(m)

    display_names = {
        "Isa-Upanishad":   "Isa Upanishad",
        "Katha-Upanishad": "Katha Upanishad",
        "Kena-Upanishad":  "Kena Upanishad",
    }

    passages = []
    for i, match in enumerate(splits):
        name_key       = match.group(1)
        upanishad_name = display_names[name_key]
        section_num    = i + 1

        block_start = match.end()
        block_end   = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
        block       = raw[block_start:block_end]

        # Collect substantial paragraphs (skip short headers / peace chants)
        paras = re.split(r"\n\n+", block)
        body  = " ".join(
            re.sub(r"\s+", " ", p).strip()
            for p in paras if len(p.strip()) > 40
        )
        chunks = chunk_text(body, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=upanishad_name,
                section=str(section_num),
                unit_number=idx + 1,
                unit_label=f"{section_num}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Upanishads] inserting {len(passages):,} passages ({len(splits)} upanishads)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Upanishads] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Srimad Bhagavatam loader ──────────────────────────────────────────────────

def ingest_srimad_bhagavatam(csv_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Load the Srimad Bhagavatam CSV verse-level into the corpus DB.
    13k verses across 12 cantos; each translation row is one passage.
    Columns: canto_number, canto_title, chapter_number, chapter_title, text, translation
    book = chapter_title, section = canto num (1–12),
    unit_number = text index, unit_label = "C.ch.text".
    """
    import re
    import pandas as pd

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Dharmic",
        name="Srimad Bhagavatam",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"source": csv_path},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    df = pd.read_csv(csv_path)
    df = df[df["translation"].notna() & (df["translation"].str.strip() != "")].reset_index(drop=True)

    def _parse_num(s):
        m = re.search(r"\d+", str(s))
        return int(m.group()) if m else 0

    passages = []
    for _, row in df.iterrows():
        canto_num   = _parse_num(row["canto_number"])
        chapter_num = _parse_num(row["chapter_number"])
        text_num    = _parse_num(row["text"])
        translation = str(row["translation"]).strip()
        chapter_title = str(row["chapter_title"]).strip()
        if not translation:
            continue
        passages.append(PassageRecord(
            corpus_id=corpus_id,
            book=chapter_title,
            section=str(canto_num),
            unit_number=text_num,
            unit_label=f"{canto_num}.{chapter_num}.{text_num}",
            text=translation,
            metadata={"canto": canto_num, "chapter": chapter_num},
        ))

    print(f"[Srimad Bhagavatam] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Srimad Bhagavatam] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Diamond Sutra loader ──────────────────────────────────────────────────────

def ingest_diamond_sutra(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse the Project Gutenberg Diamond Sutra (Gemmell/Kumarajiva translation).
    Chapter markers: '[Chapter N]' on their own line. Footnotes (indented [N] blocks)
    are stripped; remaining text is chunked with chunk_text().
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Buddhist",
        name="Diamond Sutra (Gemmell)",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"translators": ["William Gemmell", "Kumarajiva"],
                  "source": "Project Gutenberg #64623"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    chapter_pattern = re.compile(r"^\[Chapter (\d+)\]\s*$", re.MULTILINE)
    splits = list(chapter_pattern.finditer(raw))

    passages = []
    for i, match in enumerate(splits):
        chapter_num = int(match.group(1))
        block_start = match.end()
        block_end   = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
        block       = raw[block_start:block_end]

        # Strip indented footnotes (lines starting with 2+ spaces + "[N]")
        block = re.sub(r"\n  \[\d+\][^\n]*(?:\n(?!  \[|\n)[^\n]*)*", "", block)
        text  = re.sub(r"\s+", " ", block).strip()
        if not text:
            continue

        chunks = chunk_text(text, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book="Diamond Sutra",
                section=str(chapter_num),
                unit_number=idx + 1,
                unit_label=f"{chapter_num}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Diamond Sutra] inserting {len(passages):,} passages ({len(splits)} chapters)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Diamond Sutra] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Poetic Edda loader ────────────────────────────────────────────────────────

_EDDIC_POEM_TITLES = [
    "THE WISE-WOMAN\u2019S PROPHECY",  # curly apostrophe in Gutenberg file
    "HOVAMOL",
    "THE BALLAD OF VAFTHRUTHNIR",
    "THE BALLAD OF GRIMNIR",
    "THE BALLAD OF SKIRNIR",
    "THE POEM OF HARBARTH",
    "THE LAY OF HYMIR",
    "LOKASENNA",
    "THE LAY OF THRYM",
    "THE BALLAD OF ALVIS",
    "THE SONG OF RIG",
    "THE POEM OF HYNDLA",
    "THE BALLAD OF SVIPDAG",
    "THE LAY OF FJOLSVITH",
    "THE LAY OF VÖLUND",
    "THE LAY OF HELGI THE SON OF HJORVARTH",
    "THE FIRST LAY OF HELGI HUNDINGSBANE",
    "THE SECOND LAY OF HELGI HUNDINGSBANE",
    "THE BALLAD OF REGIN",
    "THE BALLAD OF FAFNIR",
    "THE BALLAD OF THE VICTORY-BRINGER",
    "THE FIRST LAY OF GUTHRUN",
    "THE SHORT LAY OF SIGURTH",
    "THE SLAYING OF THE NIFLUNGS",
    "THE SECOND, OR OLD, LAY OF GUTHRUN",
    "THE THIRD LAY OF GUTHRUN",
    "THE LAMENT OF ODDRUN",
    "THE GREENLAND LAY OF ATLI",
    "THE GREENLAND BALLAD OF ATLI",
    "THE BALLAD OF HAMTHER",
]


def ingest_poetic_edda(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse the Project Gutenberg Poetic Edda (Bellows translation).
    30 named poems, matched from a known title list to avoid false positives.
    Each poem is chunked with chunk_text(); stanza numbers are stripped.
    book = poem title, section = poem index.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Norse",
        name="Poetic Edda (Bellows)",
        type="scripture",
        language="en",
        era="medieval",
        metadata={"translator": "Henry Adams Bellows", "source": "Project Gutenberg #73533"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    escaped = [re.escape(t) for t in _EDDIC_POEM_TITLES]
    poem_pattern = re.compile(
        r"^(" + "|".join(escaped) + r")\s*$",
        re.MULTILINE,
    )
    splits = list(poem_pattern.finditer(raw))

    passages = []
    for i, match in enumerate(splits):
        poem_title = match.group(1).strip()
        # Title-case but preserve short prepositions
        poem_display = " ".join(
            w if w.lower() in {"or", "of", "the", "and"} else w.title()
            for w in poem_title.split()
        )
        poem_display = poem_display[0].upper() + poem_display[1:]

        block_start = match.end()
        block_end   = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
        block       = raw[block_start:block_end]

        # Strip stanza numbers (lines that are only digits + period)
        block = re.sub(r"^\s*\d+\.\s*$", "", block, flags=re.MULTILINE)
        # Collect paragraphs ≥30 chars
        paras = re.split(r"\n\n+", block)
        body  = " ".join(
            re.sub(r"\s+", " ", p).strip()
            for p in paras if len(p.strip()) >= 30
        )
        if not body:
            continue

        chunks = chunk_text(body, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=poem_display,
                section=str(i + 1),
                unit_number=idx + 1,
                unit_label=f"{i + 1}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Poetic Edda] inserting {len(passages):,} passages ({len(splits)} poems)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Poetic Edda] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Analects of Confucius loader ──────────────────────────────────────────────

def ingest_analects(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Parse the Project Gutenberg Analects of Confucius (Legge translation).
    20 Books; each book is collected and chunked with chunk_text().
    book = "Book I", section = book number, chunks within.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Confucian",
        name="Analects of Confucius (Legge)",
        type="scripture",
        language="en",
        era="ancient",
        metadata={"author": "Confucius", "translator": "James Legge",
                  "source": "Project Gutenberg #3330"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # "BOOK I.  HSIO R." — roman numeral + book title
    book_pattern = re.compile(r"^BOOK ([IVXLC]+)\.\s+\S", re.MULTILINE)
    splits = list(book_pattern.finditer(raw))

    passages = []
    for i, match in enumerate(splits):
        book_roman = re.search(r"[IVXLC]+", match.group()).group()
        book_num   = _roman_to_int(book_roman)

        block_start = match.start()
        block_end   = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
        block       = raw[block_start:block_end]

        # Strip chapter headings (CHAP. or CHAPTER lines)
        block = re.sub(r"^\s*CHAP(?:TER)?\.\s+[IVXLC]+\.", "", block, flags=re.MULTILINE)
        body  = re.sub(r"\s+", " ", block).strip()

        chunks = chunk_text(body, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=f"Book {book_roman}",
                section=str(book_num),
                unit_number=idx + 1,
                unit_label=f"{book_num}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Analects] inserting {len(passages):,} passages ({len(splits)} books)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Analects] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Philosophy loaders ────────────────────────────────────────────────────────

def ingest_the_republic(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """Plato's Republic (Jowett) — BOOK I–X, chunked passages per book."""
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Philosophy",
            name="The Republic (Plato)",
            type="philosophy",
            language="en",
            era="ancient",
            metadata={"author": "Plato", "translator": "Benjamin Jowett",
                      "year": -380, "source": "Project Gutenberg #1497"},
        ),
        chapter_pattern=re.compile(r"^BOOK [IVXLC]+\.", re.MULTILINE),
        chapter_name_fn=lambda m: (
            _roman_to_int(re.search(r"[IVXLC]+", m.group()).group()),
            m.group().strip(),
        ),
        conn=conn,
    )


def ingest_ethics_aristotle(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """Aristotle's Nicomachean Ethics — BOOK I–X, chunked per book."""
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Philosophy",
            name="Nicomachean Ethics (Aristotle)",
            type="philosophy",
            language="en",
            era="ancient",
            metadata={"author": "Aristotle", "source": "Project Gutenberg"},
        ),
        chapter_pattern=re.compile(r"^BOOK ([IVXLC]+)\s*$", re.MULTILINE),
        chapter_name_fn=lambda m: (
            _roman_to_int(m.group(1).strip()),
            f"Book {m.group(1).strip()}",
        ),
        conn=conn,
    )


def ingest_beyond_good_and_evil(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """Nietzsche's Beyond Good and Evil — CHAPTER I–IX, chunked."""
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Philosophy",
            name="Beyond Good and Evil (Nietzsche)",
            type="philosophy",
            language="en",
            era="modern",
            metadata={"author": "Friedrich Nietzsche", "year": 1886,
                      "source": "Project Gutenberg"},
        ),
        chapter_pattern=re.compile(r"^CHAPTER ([IVXLC]+)\.", re.MULTILINE),
        chapter_name_fn=lambda m: (
            _roman_to_int(m.group(1)),
            m.group().strip(),
        ),
        conn=conn,
    )


def ingest_discourse_on_method(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """Descartes' Discourse on the Method — PART I–VI, chunked."""
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Philosophy",
            name="Discourse on the Method (Descartes)",
            type="philosophy",
            language="en",
            era="modern",
            metadata={"author": "René Descartes", "year": 1637,
                      "source": "Project Gutenberg"},
        ),
        chapter_pattern=re.compile(r"^PART ([IVXLC]+)\s*$", re.MULTILINE),
        chapter_name_fn=lambda m: (
            _roman_to_int(m.group(1).strip()),
            f"Part {m.group(1).strip()}",
        ),
        conn=conn,
    )


def ingest_thus_spake_zarathustra(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Nietzsche's Thus Spake Zarathustra — 4 parts.
    The translator's commentary section (which also uses PART I/II/III/IV headings)
    is excluded by cutting off at the first 'PART I. THE PROLOGUE.' commentary header.
    Actual content uses unique part markers identified by surrounding text.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Philosophy",
        name="Thus Spake Zarathustra (Nietzsche)",
        type="philosophy",
        language="en",
        era="modern",
        metadata={"author": "Friedrich Nietzsche", "year": 1883,
                  "source": "Project Gutenberg"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # Strip translator's commentary (starts with "PART I. THE PROLOGUE." standalone)
    commentary = re.search(r"^PART I\. THE PROLOGUE\.\s*$", raw, re.MULTILINE)
    if commentary:
        raw = raw[:commentary.start()]

    # The 4 actual part headers in the Nietzsche content
    # The Gutenberg file uses curly apostrophes and the parts are unindented.
    # Content markers (no leading whitespace, unlike the indented ToC):
    # FIRST PART line: "FIRST PART. ZARATHUSTRA\u2019S DISCOURSES." (trailing content)
    # Others are standalone. All must be at column 0 (unindented, unlike the ToC).
    part_pattern = re.compile(
        r"^(FIRST PART\.[^\n]+"
        r"|THUS SPAKE ZARATHUSTRA\. SECOND PART\."
        r"|THIRD PART\."
        r"|FOURTH AND LAST PART\.)\s*$",
        re.MULTILINE,
    )
    splits     = list(part_pattern.finditer(raw))
    part_names = ["Part I", "Part II", "Part III", "Part IV"]

    passages = []
    for i, match in enumerate(splits):
        block_start = match.end()
        block_end   = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
        block       = raw[block_start:block_end]
        body        = re.sub(r"\s+", " ", block).strip()

        chunks = chunk_text(body, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=part_names[i] if i < len(part_names) else f"Part {i + 1}",
                section=str(i + 1),
                unit_number=idx + 1,
                unit_label=f"{i + 1}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Thus Spake Zarathustra] inserting {len(passages):,} passages ({len(splits)} parts)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Thus Spake Zarathustra] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


def ingest_critique_pure_reason(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Kant's Critique of Pure Reason — split by major named sections
    (Introduction, Aesthetic, Analytic, Dialectic, Methodology) plus BOOK I/II
    subdivisions. Each section block is chunked with chunk_text().
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Philosophy",
        name="Critique of Pure Reason (Kant)",
        type="philosophy",
        language="en",
        era="modern",
        metadata={"author": "Immanuel Kant", "year": 1781,
                  "source": "Project Gutenberg"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # This Gutenberg file has clean headers for BOOK I and BOOK II of the
    # Transcendental Analytic but not for other major divisions.
    # Any text before BOOK I is treated as "Introduction & Aesthetic".
    section_pattern = re.compile(
        r"^(BOOK [IVXLC]+\. .+)\s*$",
        re.MULTILINE,
    )
    splits = list(section_pattern.finditer(raw))

    # Build (name, block) pairs; prepend preamble as section 0 if text exists before BOOK I
    splits_data = []
    if splits and splits[0].start() > 100:
        preamble = raw[:splits[0].start()]
        if len(preamble.strip()) > 200:
            splits_data.append(("Introduction and Transcendental Aesthetic", preamble))

    if not splits:
        splits_data.append(("Critique of Pure Reason", raw))
    else:
        for i, match in enumerate(splits):
            block_start = match.end()
            block_end   = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
            splits_data.append((match.group(1).strip(), raw[block_start:block_end]))

    passages = []
    for sec_idx, (section_name, block) in enumerate(splits_data):
        body   = re.sub(r"\s+", " ", block).strip()
        chunks = chunk_text(body, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=section_name,
                section=str(sec_idx + 1),
                unit_number=idx + 1,
                unit_label=f"{sec_idx + 1}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Critique of Pure Reason] inserting {len(passages):,} passages ({len(splits_data)} sections)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Critique of Pure Reason] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Scientific loaders ────────────────────────────────────────────────────────

def ingest_origin_of_species(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """Darwin's On the Origin of Species — CHAPTER I–XIV, chunked."""
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Scientific",
            name="On the Origin of Species (Darwin)",
            type="scientific",
            language="en",
            era="modern",
            metadata={"author": "Charles Darwin", "year": 1859,
                      "source": "Project Gutenberg #1228"},
        ),
        chapter_pattern=re.compile(r"^CHAPTER \d+\.", re.MULTILINE),
        chapter_name_fn=lambda m: (
            int(re.search(r"\d+", m.group()).group()),
            m.group().strip(),
        ),
        conn=conn,
    )


def ingest_varieties_religious_experience(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """William James' Varieties of Religious Experience — LECTURE I–XX, chunked."""
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Scientific",
            name="Varieties of Religious Experience (James)",
            type="scientific",
            language="en",
            era="modern",
            metadata={"author": "William James", "year": 1902,
                      "source": "Project Gutenberg"},
        ),
        chapter_pattern=re.compile(
            # Matches both singular and plural: "LECTURE I." / "LECTURES IV AND V."
            # / "LECTURES XI, XII, AND XIII."
            r"^LECTURES?\s+[IVXLC][IVXLC,\s]*(?:AND\s+[IVXLC]+\s*)?\. ",
            re.MULTILINE,
        ),
        chapter_name_fn=lambda m: (
            _roman_to_int(re.search(r"[IVXLC]+", m.group()).group()),
            m.group().strip(),
        ),
        conn=conn,
    )


def ingest_civilization_discontents(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """Freud's Civilization and Its Discontents — sections I–VIII."""
    import re
    conn = conn or get_conn()
    return _ingest_novel(
        txt_path,
        CorpusRecord(
            tradition_name="Scientific",
            name="Civilization and Its Discontents (Freud)",
            type="scientific",
            language="en",
            era="modern",
            metadata={"author": "Sigmund Freud", "year": 1930,
                      "source": "Project Gutenberg"},
        ),
        # Standalone Roman numerals I–VIII on their own line
        chapter_pattern=re.compile(
            r"^(VIII|VII|VI|IV|V|III|II|I)\s*$", re.MULTILINE
        ),
        chapter_name_fn=lambda m: (
            _roman_to_int(m.group(1).strip()),
            f"Section {m.group(1).strip()}",
        ),
        conn=conn,
    )


def ingest_opticks(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Newton's Opticks — split by Book I, II, III (each covers optics experiments).
    Headers use 'FIRST/SECOND/THIRD BOOK OF OPTICKS' — deduplicated so title-page
    occurrences don't double-count the content.
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Scientific",
        name="Opticks (Newton)",
        type="scientific",
        language="en",
        era="modern",
        metadata={"author": "Isaac Newton", "year": 1704,
                  "source": "Project Gutenberg"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # Matches both single-line and multi-line book headers
    book_pattern = re.compile(
        r"(?:THE\s+)?(?P<ord>FIRST|SECOND|THIRD)\s+BOOK\s+OF\s+OPTICKS",
        re.MULTILINE | re.DOTALL,
    )
    all_matches = list(book_pattern.finditer(raw))

    # Keep only the first occurrence of each book ordinal
    seen, unique_splits = set(), []
    for m in all_matches:
        key = m.group("ord")
        if key not in seen:
            seen.add(key)
            unique_splits.append(m)

    book_nums = {"FIRST": 1, "SECOND": 2, "THIRD": 3}

    passages = []
    for i, match in enumerate(unique_splits):
        book_num    = book_nums[match.group("ord")]
        block_start = match.end()
        block_end   = unique_splits[i + 1].start() if i + 1 < len(unique_splits) else len(raw)
        block       = raw[block_start:block_end]
        body        = re.sub(r"\s+", " ", block).strip()

        chunks = chunk_text(body, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=f"Book {book_num}",
                section=str(book_num),
                unit_number=idx + 1,
                unit_label=f"{book_num}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Opticks] inserting {len(passages):,} passages ({len(unique_splits)} books)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Opticks] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


def ingest_psychology_unconscious(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Jung's Psychology of the Unconscious — PART I and PART II.
    Part headers are heavily indented in the Gutenberg text.
    Each part is chunked with chunk_text().
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Scientific",
        name="Psychology of the Unconscious (Jung)",
        type="scientific",
        language="en",
        era="modern",
        metadata={"author": "C. G. Jung", "year": 1912,
                  "source": "Project Gutenberg"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # Part headers appear with heavy indentation: "                PART I"
    part_pattern = re.compile(r"^\s{5,}PART (I{1,2})\s*$", re.MULTILINE)
    splits = list(part_pattern.finditer(raw))

    part_names = {1: "Part I", 2: "Part II"}

    passages = []
    if not splits:
        # Fallback: chunk the whole text as one section
        body   = re.sub(r"\s+", " ", raw).strip()
        chunks = chunk_text(body, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book="Psychology of the Unconscious",
                section="1",
                unit_number=idx + 1,
                unit_label=f"1.{idx + 1}",
                text=chunk,
            ))
    else:
        for i, match in enumerate(splits):
            part_num    = _roman_to_int(match.group(1).strip())
            block_start = match.end()
            block_end   = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
            block       = raw[block_start:block_end]
            body        = re.sub(r"\s+", " ", block).strip()

            chunks = chunk_text(body, sentences_per_chunk=4)
            for idx, chunk in enumerate(chunks):
                passages.append(PassageRecord(
                    corpus_id=corpus_id,
                    book=part_names.get(part_num, f"Part {part_num}"),
                    section=str(part_num),
                    unit_number=idx + 1,
                    unit_label=f"{part_num}.{idx + 1}",
                    text=chunk,
                ))

    print(f"[Psychology of the Unconscious] inserting {len(passages):,} passages ({len(splits) or 1} parts)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Psychology of the Unconscious] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Siddhartha loader ─────────────────────────────────────────────────────────

def ingest_siddhartha(txt_path: str, conn=None) -> tuple[int, list[int]]:
    """
    Hesse's Siddhartha — chapters are ALL-CAPS titles on their own line
    (e.g. "THE SON OF THE BRAHMAN", "THE FERRYMAN", "FIRST PART").
    Each chapter is chunked with chunk_text().
    """
    import re

    conn = conn or get_conn()
    corpus_record = CorpusRecord(
        tradition_name="Literature",
        name="Siddhartha (Hesse)",
        type="literature",
        language="en",
        era="modern",
        metadata={"author": "Hermann Hesse", "year": 1922,
                  "source": "Project Gutenberg #2500"},
    )
    corpus_id, existing_ids = _get_or_skip(conn, corpus_record)
    if existing_ids is not None:
        return corpus_id, existing_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        raw = _strip_gutenberg(f.read())

    # All-caps standalone lines of ≥1 word and ≥4 chars (chapter titles, part headers).
    # Includes single-word titles like GOTAMA, AWAKENING, SANSARA and
    # two-word titles like THE SON, FIRST PART.
    chapter_pattern = re.compile(
        r"^([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\s*$", re.MULTILINE
    )
    splits = list(chapter_pattern.finditer(raw))

    passages = []
    for i, match in enumerate(splits):
        chapter_title = match.group(1).strip().title()
        block_start   = match.end()
        block_end     = splits[i + 1].start() if i + 1 < len(splits) else len(raw)
        block         = raw[block_start:block_end].strip()
        body          = re.sub(r"\s+", " ", block).strip()
        if not body:
            continue

        chunks = chunk_text(body, sentences_per_chunk=4)
        for idx, chunk in enumerate(chunks):
            passages.append(PassageRecord(
                corpus_id=corpus_id,
                book=chapter_title,
                section=str(i + 1),
                unit_number=idx + 1,
                unit_label=f"{i + 1}.{idx + 1}",
                text=chunk,
            ))

    print(f"[Siddhartha] inserting {len(passages):,} passages ({len(splits)} chapters)...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Siddhartha] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    TRANSLATIONS = ["KJV", "ACV", "YLT", "BBE"]
    BIBLE_DB_DIR = "data/bibles"

    conn = get_conn()
    for t in TRANSLATIONS:
        db_path = os.path.join(BIBLE_DB_DIR, f"{t}.db")
        ingest_scrollmapper_bible(t, db_path, corpus_db_conn=conn)

    total = conn.execute("SELECT COUNT(*) FROM passage").fetchone()[0]
    print(f"\nTotal passages in DB: {total:,}")
    conn.close()
