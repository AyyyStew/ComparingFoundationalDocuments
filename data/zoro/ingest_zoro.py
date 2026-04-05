# %% [markdown]
# # Ingest — Zoroastrian Texts (Yasna + Vendidad)
#
# Sources:
#   data/zoro/AVESTA_ YASNA_ (English).html  — Yasna, tr. Mills (1898)
#   data/zoro/vd_ebook.epub                  — Vendidad, tr. Darmesteter (1898)
#
# Passage granularity: one passage per numbered paragraph within each chapter.
# book    = chapter heading  (e.g. "28.", "Fargard 2")
# section = sub-section label where present (H4 in Yasna; None in Vendidad)
# unit_number = paragraph index within chapter (1-based)
# unit_label  = unique key, e.g. "Yasna 28:5" / "Vendidad Fargard 2:7"
#
# Gatha chapters (Yasna 28–34, 43–51, 53) are flagged in metadata.

import json
import re
import warnings
import zipfile

import duckdb
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from db.ingest import _get_or_create_corpus, insert_passages
from db.models import CorpusRecord, PassageRecord
from db.schema import get_conn

YASNA_HTML  = "data/zoro/AVESTA_ YASNA_ (English).html"
VENDIDAD_EPUB = "data/zoro/vd_ebook.epub"

GATHA_CHAPTERS = {28, 29, 30, 31, 32, 33, 34, 43, 44, 45, 46, 47, 48, 49, 50, 51, 53}


def _strip_verse_numbers(text: str) -> str:
    """Remove leading verse numbers from text lines, e.g. '1. text' → 'text'.
    Handles multi-stanza paragraphs where numbers appear mid-text on their own line."""
    # Remove numbers at the start of any line (covers "1.\ntext\n2.\ntext" patterns)
    cleaned = re.sub(r"(?m)^\d+\.\s*", "", text)
    # Collapse resulting blank lines and tidy whitespace
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    return cleaned


# ── Yasna parser ──────────────────────────────────────────────────────────────

def _iter_yasna_passages(html_path: str):
    """
    Yield dicts with keys: chapter, chapter_title, subsection,
    unit_number, unit_label, text, is_gatha.
    Skips the intro, contents, and abbreviation sections.
    """
    with open(html_path, encoding="latin-1") as f:
        soup = BeautifulSoup(f, "html.parser")

    current_chapter_raw = None
    current_chapter_num = None
    current_title = None
    current_subsection = None
    para_idx = 0
    started = False   # flip True on first numbered chapter

    for tag in soup.find_all(["h2", "h3", "h4", "p"]):
        if tag.name in ("h2", "h3") and tag.name == "h2":
            text = tag.get_text().strip()
            # Extract leading number  e.g. "28." or "12. The Zoroastrian Creed"
            m = re.match(r"^(\d+)\.?\s*(.*)", text)
            if m:
                started = True
                current_chapter_num = int(m.group(1))
                current_chapter_raw = str(current_chapter_num)  # clean "28" not "28."
                current_title = m.group(2).strip() or None
            elif started:
                # Named section without a number (e.g. "THE HOM YASHT")
                current_chapter_num = None
                current_chapter_raw = text
                current_title = None
            current_subsection = None
            para_idx = 0
            continue

        if tag.name == "h4":
            current_subsection = tag.get_text().strip() or None
            continue

        if tag.name == "p" and started:
            text = _strip_verse_numbers(tag.get_text().strip())
            if len(text) < 20:
                continue
            para_idx += 1
            chap_label = current_chapter_raw or "intro"
            unit_label = f"Yasna {chap_label}:{para_idx}"
            yield {
                "chapter":      chap_label,
                "chapter_num":  current_chapter_num,
                "chapter_title": current_title,
                "subsection":   current_subsection,
                "unit_number":  para_idx,
                "unit_label":   unit_label,
                "text":         text,
                "is_gatha":     current_chapter_num in GATHA_CHAPTERS
                                if current_chapter_num else False,
            }


def ingest_yasna(
    html_path: str = YASNA_HTML,
    corpus_db_conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[int, list[int]]:
    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Zoroastrian",
        name="Avesta: Yasna",
        type="scripture",
        language="en",
        era="ancient",
        metadata={
            "full_title": "Avesta: Yasna — Sacred Liturgy and Gathas/Hymns of Zarathushtra",
            "translator": "L. H. Mills (Sacred Books of the East, 1898); Gathas by C. Bartholomae",
            "editor": "Joseph H. Peterson",
            "gatha_chapters": sorted(GATHA_CHAPTERS),
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
            print(f"[Yasna] already ingested ({count:,} passages) — skipping")
            return corpus_id, [
                r[0] for r in conn.execute(
                    "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
                ).fetchall()
            ]
        print("[Yasna] corpus row exists but 0 passages — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)
    print(f"[Yasna] corpus_id={corpus_id}, parsing {html_path}...")

    passages = [
        PassageRecord(
            corpus_id=corpus_id,
            book=row["chapter"],
            section=row["subsection"],
            unit_number=row["unit_number"],
            unit_label=row["unit_label"],
            text=row["text"],
            metadata={
                "chapter_num":   row["chapter_num"],
                "chapter_title": row["chapter_title"],
                "is_gatha":      row["is_gatha"],
            },
        )
        for row in _iter_yasna_passages(html_path)
    ]

    print(f"[Yasna] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Yasna] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Vendidad parser ───────────────────────────────────────────────────────────

def _spine_content_files(epub_path: str) -> list[str]:
    """Return content0XXX.xhtml files in spine order (skipping front matter)."""
    with zipfile.ZipFile(epub_path) as z:
        opf = z.read("OEBPS/content.opf").decode("utf-8")
    soup = BeautifulSoup(opf, "html.parser")
    order = [item.get("idref", "") for item in soup.find_all("itemref")]
    return [f for f in order if re.match(r"content\d+\.xhtml", f)
            and f not in ("content0001.xhtml", "content0002.xhtml", "content0003.xhtml")]


def _iter_vendidad_passages(epub_path: str):
    """
    Yield dicts with keys: fargard, fargard_num, fargard_title,
    unit_number, unit_label, text.
    Only yields paragraphs from the Translation section of each Fargard.
    """
    content_files = _spine_content_files(epub_path)

    with zipfile.ZipFile(epub_path) as z:
        for fname in content_files:
            raw = z.read(f"OEBPS/text/{fname}").decode("utf-8")
            soup = BeautifulSoup(raw, "html.parser")

            # Find fargard heading
            fargard_raw = None
            fargard_num = None
            fargard_title = None
            for h in soup.find_all(["h2", "h3", "h4"]):
                m = re.match(r"FARGARD\s+(\d+)\.\s*(.*)", h.get_text().strip(), re.I)
                if m:
                    fargard_num = int(m.group(1))
                    fargard_title = m.group(2).strip() or None
                    fargard_raw = f"Fargard {fargard_num}"
                    break

            if not fargard_raw:
                continue  # skip non-fargard content files

            # Collect only paragraphs that fall after a "Translation" heading
            # and before a "Notes" heading
            in_translation = False
            para_idx = 0
            for tag in soup.find_all(["h2", "h3", "h4", "p"]):
                if tag.name in ("h2", "h3", "h4"):
                    heading = tag.get_text().strip()
                    if re.search(r"\bTranslation\b", heading, re.I):
                        in_translation = True
                    elif re.search(r"\bNotes?\b", heading, re.I) and in_translation:
                        break
                    continue

                if not in_translation:
                    continue

                # Remove footnote superscripts before extracting text
                for sup in tag.find_all("sup"):
                    sup.decompose()
                text = _strip_verse_numbers(tag.get_text().strip())
                if len(text) < 20:
                    continue

                para_idx += 1
                yield {
                    "fargard":       fargard_raw,
                    "fargard_num":   fargard_num,
                    "fargard_title": fargard_title,
                    "unit_number":   para_idx,
                    "unit_label":    f"Vendidad {fargard_raw}:{para_idx}",
                    "text":          text,
                }


def ingest_vendidad(
    epub_path: str = VENDIDAD_EPUB,
    corpus_db_conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[int, list[int]]:
    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Zoroastrian",
        name="Avesta: Vendidad",
        type="scripture",
        language="en",
        era="ancient",
        metadata={
            "full_title": "Vendidad (Vidēvdād) or Laws against the Demons",
            "translator": "James Darmesteter (Sacred Books of the East, 1898)",
            "editor": "Joseph H. Peterson",
            "note": "Translation sections only; introductions and notes excluded",
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
            print(f"[Vendidad] already ingested ({count:,} passages) — skipping")
            return corpus_id, [
                r[0] for r in conn.execute(
                    "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
                ).fetchall()
            ]
        print("[Vendidad] corpus row exists but 0 passages — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)
    print(f"[Vendidad] corpus_id={corpus_id}, parsing {epub_path}...")

    passages = [
        PassageRecord(
            corpus_id=corpus_id,
            book=row["fargard"],
            section=None,
            unit_number=row["unit_number"],
            unit_label=row["unit_label"],
            text=row["text"],
            metadata={
                "fargard_num":   row["fargard_num"],
                "fargard_title": row["fargard_title"],
            },
        )
        for row in _iter_vendidad_passages(epub_path)
    ]

    print(f"[Vendidad] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Vendidad] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


# ── Run both ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    conn = get_conn()

    y_id, y_pids = ingest_yasna(corpus_db_conn=conn)
    v_id, v_pids = ingest_vendidad(corpus_db_conn=conn)

    for name, cid in [("Avesta: Yasna", y_id), ("Avesta: Vendidad", v_id)]:
        count = conn.execute(
            "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [cid]
        ).fetchone()[0]
        sample = conn.execute(
            "SELECT unit_label, text FROM passage WHERE corpus_id = ? ORDER BY id LIMIT 3",
            [cid],
        ).fetchall()
        print(f"\n{name} — {count:,} passages")
        for label, text in sample:
            print(f"  [{label}] {text[:90]}...")

    conn.close()
