# %% [markdown]
# # Ingest — Sri Guru Granth Sahib
# Reads from data/sikh/gurbanidb.sqlite (converted from MySQL dump),
# loads English translations line-by-line into corpus.duckdb.
#
# Passage granularity: one passage per scripture line (verse-level).
# book    = raag / named chapter (e.g. "Jap Ji Sahib", "Siri Raag")
# section = sub-section where present (e.g. "Sukhmani Sahib")
# unit_number = line index within the hymn (1-based)
# unit_label  = unique human-readable key, e.g. "Siri Raag — Sukhmani Sahib — 100:3"
#
# Embeddings are handled in a separate script.

# %% Imports
import json
import sqlite3

import duckdb

from db.schema import get_conn
from db.models import CorpusRecord, PassageRecord
from db.ingest import _get_or_create_corpus, insert_passages

SIKH_DB   = "data/sikh/gurbanidb.sqlite"
CORPUS_NAME = "Sri Guru Granth Sahib"
EN_LANG_ID  = 13   # English (en-US), accuracy=99

# %% Load source data

def _load_passages(sikh_db: str) -> list[dict]:
    """
    Pull every scripture line that has an English translation.
    Returns rows sorted by scripture id (i.e. canonical text order).
    Derives per-hymn line index for unit_number.
    """
    con = sqlite3.connect(sikh_db)
    con.row_factory = sqlite3.Row

    rows = con.execute(
        """
        SELECT
            s.id            AS scripture_id,
            s.hymn          AS hymn,
            s.page          AS page,
            s.line          AS page_line,
            s.section       AS raw_section,
            s.text          AS gurmukhi,
            a.author        AS author,
            m.melody        AS melody,
            t.text          AS en_text
        FROM scriptures s
        JOIN translations t
            ON t.scripture_id = s.id AND t.language_id = ?
        LEFT JOIN authors  a ON a.id = s.author_id
        LEFT JOIN melodies m ON m.id = s.melody_id
        ORDER BY s.id
        """,
        (EN_LANG_ID,),
    ).fetchall()

    con.close()

    # Derive line index within each hymn (1-based, in text order)
    hymn_counter: dict[int, int] = {}
    result = []
    for row in rows:
        hymn = row["hymn"]
        hymn_counter[hymn] = hymn_counter.get(hymn, 0) + 1
        result.append({**dict(row), "hymn_line": hymn_counter[hymn]})

    return result


def _split_section(raw: str | None) -> tuple[str | None, str | None]:
    """
    Split "Siri Raag - Sukhmani Sahib (Peace Of Mind)" into
    book="Siri Raag", section="Sukhmani Sahib (Peace Of Mind)".
    If no " - ", section is None.
    """
    if not raw:
        return None, None
    parts = raw.split(" - ", maxsplit=1)
    book = parts[0].strip() or None
    section = parts[1].strip() if len(parts) > 1 else None
    return book, section


def _make_unit_label(book: str | None, section: str | None, hymn: int, line: int) -> str:
    parts = [p for p in [book, section] if p]
    parts.append(f"{hymn}:{line}")
    return " — ".join(parts)


# %% Ingest function

def ingest_sikh(
    sikh_db: str = SIKH_DB,
    corpus_db_conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[int, list[int]]:
    """
    Load the Sri Guru Granth Sahib into the corpus DB.
    Returns (corpus_id, [passage_ids]).
    Safe to call again — skips if already ingested.
    """
    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Sikh",
        name=CORPUS_NAME,
        type="scripture",
        language="en",
        era="medieval",
        metadata={
            "source": "GurbaniDB v2",
            "original_language": "Punjabi (Gurmukhi)",
            "compiled": "1604 CE, finalised 1708 CE",
            "translation_language_id": EN_LANG_ID,
        },
    )

    # Idempotency check
    existing = conn.execute(
        "SELECT id FROM corpus WHERE name = ?", [corpus_record.name]
    ).fetchone()
    if existing:
        corpus_id = existing[0]
        count = conn.execute(
            "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
        ).fetchone()[0]
        if count > 0:
            print(f"[SGGS] already ingested ({count:,} passages) — skipping")
            passage_ids = [
                r[0] for r in conn.execute(
                    "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
                ).fetchall()
            ]
            return corpus_id, passage_ids
        print("[SGGS] corpus row exists but 0 passages (previous crash?) — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)
    print(f"[SGGS] corpus_id={corpus_id}, loading passages from {sikh_db}...")

    rows = _load_passages(sikh_db)
    print(f"[SGGS] {len(rows):,} lines with English translations found")

    passages = []
    for row in rows:
        en_text = row["en_text"]
        if not en_text or not en_text.strip():
            continue

        book, section = _split_section(row["raw_section"])
        hymn      = row["hymn"]
        hymn_line = row["hymn_line"]

        passages.append(
            PassageRecord(
                corpus_id=corpus_id,
                book=book,
                section=section,
                unit_number=hymn_line,
                unit_label=_make_unit_label(book, section, hymn, hymn_line),
                text=en_text,
                metadata={
                    "hymn":        hymn,
                    "page":        row["page"],
                    "page_line":   row["page_line"],
                    "author":      row["author"],
                    "melody":      row["melody"],
                    "gurmukhi":    row["gurmukhi"],
                    "scripture_id": row["scripture_id"],
                },
            )
        )

    print(f"[SGGS] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[SGGS] done — {len(passage_ids):,} passages inserted")

    return corpus_id, passage_ids


# %% Run

if __name__ == "__main__":
    conn = get_conn()
    corpus_id, passage_ids = ingest_sikh(corpus_db_conn=conn)

    # Sanity check
    total = conn.execute(
        "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
    ).fetchone()[0]
    print(f"\nSri Guru Granth Sahib in DB:")
    print(f"  Passages: {total:,}")

    sample = conn.execute(
        """
        SELECT book, section, unit_label, text
        FROM passage WHERE corpus_id = ?
        ORDER BY id LIMIT 5
        """,
        [corpus_id],
    ).fetchall()
    print("\nSample passages:")
    for book, section, label, text in sample:
        print(f"  [{label}] {text[:80]}...")

    conn.close()
