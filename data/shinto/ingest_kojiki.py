# %% [markdown]
# # Ingest — Kojiki
# Source: data/shinto/kojiki/ (scraped from sacred-texts.com)
# Translation: Basil Hall Chamberlain, 1919
#
# Passage granularity: one passage per prose paragraph within each section.
# Footnotes and page-number markers ("p. N") are skipped.
#
# book        = volume, e.g. "Volume I — Age of the Gods"
# section     = section heading, e.g. "Sect. I — The Beginning of Heaven and Earth"
# unit_number = paragraph index within section (1-based)
# unit_label  = "Kojiki I:3:2"  (volume_num:section_num:para_num)

import re
from pathlib import Path

import duckdb
from bs4 import BeautifulSoup

from db.ingest import _get_or_create_corpus, insert_passages
from db.models import CorpusRecord, PassageRecord
from db.schema import get_conn

KOJIKI_DIR = Path("data/shinto/kojiki")

# Human-readable volume names
VOLUME_NAMES = {
    "I":   "Volume I — Age of the Gods",
    "II":  "Volume II — Age of the Early Emperors",
    "III": "Volume III — Age of the Later Emperors",
}

# Pages to skip: intro, preface, appendices
SKIP_PAGES = {f"kj{n:03d}.htm" for n in range(0, 8)}   # kj000–kj007

PAGE_NUM_RE  = re.compile(r"^p\.\s*\d+$", re.I)
SECTION_RE   = re.compile(r"SECT(?:ION)?\.?\s+([IVXLC\d]+)", re.I)
APPENDIX_RE  = re.compile(r"APPENDIX", re.I)


def _extract_volume(title_text: str) -> str | None:
    m = re.search(r"Volume\s+(I{1,3}V?|IV|VI{0,3})", title_text, re.I)
    return m.group(1).upper() if m else None


def _clean_section_heading(raw: str) -> str:
    """'[SECT. I.—THE BEGINNING OF HEAVEN AND EARTH.]' → 'Sect. I — The Beginning of Heaven and Earth'"""
    s = raw.strip().strip("[]")
    s = re.sub(r"^SECT(?:ION)?\.?\s*", "Sect. ", s, flags=re.I)
    s = s.replace("—", " — ").replace(".—", " — ")
    s = re.sub(r"\s{2,}", " ", s)
    s = s.rstrip(".]").strip()
    # Title-case the description part
    parts = s.split(" — ", maxsplit=1)
    if len(parts) == 2:
        s = parts[0] + " — " + parts[1].title()
    return s


def _section_num(heading_raw: str) -> int | None:
    """Extract integer section number from raw heading text."""
    m = SECTION_RE.search(heading_raw)
    if not m:
        return None
    roman = m.group(1)
    # Convert Roman numeral
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total, prev = 0, 0
    for ch in reversed(roman.upper()):
        v = vals.get(ch, 0)
        total += v if v >= prev else -v
        prev = v
    return total or None


def _iter_kojiki_passages(kojiki_dir: Path):
    pages = sorted(kojiki_dir.glob("kj*.htm"))

    for page_path in pages:
        fname = page_path.name
        if fname in SKIP_PAGES:
            continue

        html = page_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        # Volume from <title>
        title_tag = soup.find("title")
        volume_rom = _extract_volume(title_tag.text if title_tag else "")
        if not volume_rom:
            continue  # appendix or non-content page

        volume_label = VOLUME_NAMES.get(volume_rom, f"Volume {volume_rom}")

        # Section heading: first H2 or H3 containing SECT or APPENDIX
        sect_tag = None
        for tag in soup.find_all(["h2", "h3", "h4"]):
            t = tag.get_text()
            if SECTION_RE.search(t) or APPENDIX_RE.search(t):
                sect_tag = tag
                break

        if not sect_tag:
            continue

        section_raw   = sect_tag.get_text().strip()
        section_label = _clean_section_heading(section_raw)
        section_num   = _section_num(section_raw)

        # Collect paragraphs after section heading, stop at Footnotes
        in_content = False
        para_idx   = 0

        for tag in soup.find_all(["h2", "h3", "h4", "p"]):
            if tag is sect_tag:
                in_content = True
                continue

            if not in_content:
                continue

            if tag.name in ("h2", "h3", "h4"):
                if "footnote" in tag.get_text().lower():
                    break
                continue

            # Remove footnote anchors and reference links
            for a in tag.find_all("a", href=lambda h: h and h.startswith("#fn_")):
                a.decompose()
            for a in tag.find_all("a", attrs={"name": re.compile(r"^fr_")}):
                a.decompose()

            text = re.sub(r"[\xa0\s]+", " ", tag.get_text()).strip()

            # Skip page-number markers like "p. 15"
            if PAGE_NUM_RE.match(text):
                continue
            if len(text) < 20:
                continue

            para_idx += 1
            # Descriptive label: "Kojiki I — The Land of Hades — 9:2"
            sect_parts = section_label.split(" — ", maxsplit=1)
            sect_desc = sect_parts[1] if len(sect_parts) > 1 else section_label
            unit_label = f"Kojiki {volume_rom} — {sect_desc} — {section_num or '?'}:{para_idx}"
            yield {
                "volume":       volume_label,
                "volume_rom":   volume_rom,
                "section":      section_label,
                "section_num":  section_num,
                "unit_number":  para_idx,
                "unit_label":   unit_label,
                "text":         text,
            }


def ingest_kojiki(
    kojiki_dir: Path = KOJIKI_DIR,
    corpus_db_conn: duckdb.DuckDBPyConnection | None = None,
) -> tuple[int, list[int]]:
    conn = corpus_db_conn or get_conn()

    corpus_record = CorpusRecord(
        tradition_name="Shinto",
        name="Kojiki",
        type="scripture",
        language="en",
        era="ancient",
        metadata={
            "full_title": "Kojiki (Records of Ancient Matters)",
            "translator": "Basil Hall Chamberlain",
            "year": 1919,
            "source": "sacred-texts.com",
            "volumes": list(VOLUME_NAMES.values()),
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
            print(f"[Kojiki] already ingested ({count:,} passages) — skipping")
            return corpus_id, [
                r[0] for r in conn.execute(
                    "SELECT id FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
                ).fetchall()
            ]
        print("[Kojiki] corpus row exists but 0 passages — re-ingesting")

    corpus_id = _get_or_create_corpus(conn, corpus_record)
    print(f"[Kojiki] corpus_id={corpus_id}, parsing {kojiki_dir}...")

    passages = [
        PassageRecord(
            corpus_id=corpus_id,
            book=row["volume"],
            section=row["section"],
            unit_number=row["unit_number"],
            unit_label=row["unit_label"],
            text=row["text"],
            metadata={
                "volume_rom":  row["volume_rom"],
                "section_num": row["section_num"],
            },
        )
        for row in _iter_kojiki_passages(kojiki_dir)
    ]

    print(f"[Kojiki] inserting {len(passages):,} passages...")
    passage_ids = insert_passages(conn, passages)
    print(f"[Kojiki] done — {len(passage_ids):,} passages inserted")
    return corpus_id, passage_ids


if __name__ == "__main__":
    conn = get_conn()
    corpus_id, passage_ids = ingest_kojiki(corpus_db_conn=conn)

    total = conn.execute(
        "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
    ).fetchone()[0]

    by_vol = conn.execute(
        "SELECT book, COUNT(*) FROM passage WHERE corpus_id = ? GROUP BY book ORDER BY book",
        [corpus_id],
    ).fetchall()

    print(f"\nKojiki in DB: {total:,} passages")
    for vol, cnt in by_vol:
        print(f"  {vol}: {cnt:,}")

    print("\nSample passages:")
    sample = conn.execute(
        "SELECT book, section, unit_label, text FROM passage WHERE corpus_id = ? ORDER BY id LIMIT 5",
        [corpus_id],
    ).fetchall()
    for vol, sect, label, text in sample:
        print(f"  [{label}] {text[:90]}...")

    conn.close()
