# %% [markdown]
# # Clean Diamond Sutra — Remove Junk Passages
#
# The Diamond Sutra ingest left some artefact passages:
#   - Translator name fragments ("Beal.", "Max Müller.", "T. W. Rhys Davids.")
#   - Footnote/editorial text that slipped through the chapter parser
#
# This script deletes those passages and their embeddings from the DB.
# It is idempotent — re-running it after a clean DB is a no-op.
#
# A passage is considered junk if ANY of the following are true:
#   1. Text length < MIN_CHARS after stripping whitespace
#   2. Text matches a known translator/editorial pattern
#   3. Text starts with "Footnotes" or contains only a name-like fragment

# %% Imports
import re
from db.schema import get_conn

CORPUS_NAME = "Diamond Sutra (Gemmell)"
MIN_CHARS   = 40   # shorter than this is almost certainly a fragment

# Patterns that indicate editorial/translator noise rather than scripture
JUNK_PATTERNS = [
    re.compile(r"^\s*[\w\.\s\-]{1,40}\.\s*$"),            # lone "Name." fragments
    re.compile(r"^\s*Footnotes", re.IGNORECASE),
    re.compile(r"^\s*(Beal|M[üu]ller|Rhys Davids|Kern|Thomson|Gemmell)\b", re.IGNORECASE),
    re.compile(r"Word spellings have been standardized", re.IGNORECASE),
    re.compile(r"Text contained within underscores is italicised", re.IGNORECASE),
    re.compile(r"Footnotes have been moved", re.IGNORECASE),
    re.compile(r"^\s*\d+\.\s*$"),                          # bare number "3."
]

def is_junk(text: str) -> bool:
    t = text.strip()
    if len(t) < MIN_CHARS:
        return True
    return any(p.search(t) for p in JUNK_PATTERNS)

# %% Find and delete
conn = get_conn()

corpus_row = conn.execute(
    "SELECT id FROM corpus WHERE name = ?", [CORPUS_NAME]
).fetchone()

if corpus_row is None:
    print(f"Corpus '{CORPUS_NAME}' not found in DB — nothing to do.")
    conn.close()
else:
    corpus_id = corpus_row[0]

    rows = conn.execute(
        "SELECT id, unit_label, text FROM passage WHERE corpus_id = ?",
        [corpus_id]
    ).fetchall()

    junk_ids = [r[0] for r in rows if is_junk(r[2])]

    if not junk_ids:
        print(f"No junk passages found in '{CORPUS_NAME}' — DB is already clean.")
    else:
        print(f"Found {len(junk_ids)} junk passage(s) in '{CORPUS_NAME}':")
        for r in rows:
            if r[0] in set(junk_ids):
                print(f"  [{r[1]}] {r[2][:80]!r}")

        # Delete embeddings first (FK constraint), then passages
        placeholders = ", ".join("?" * len(junk_ids))
        conn.execute(
            f"DELETE FROM embedding WHERE passage_id IN ({placeholders})", junk_ids
        )
        conn.execute(
            f"DELETE FROM passage WHERE id IN ({placeholders})", junk_ids
        )
        conn.commit()
        print(f"\nDeleted {len(junk_ids)} passage(s) and their embeddings.")

    # Verify
    remaining = conn.execute(
        "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
    ).fetchone()[0]
    print(f"'{CORPUS_NAME}' now has {remaining:,} passages.")

    conn.close()
