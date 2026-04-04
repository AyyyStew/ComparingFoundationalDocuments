# %% [markdown]
# # Filter Gutenberg Boilerplate from DB (Script 00)
#
# Some ingest scripts let Gutenberg license/footer text slip through as passages.
# This script identifies and deletes those passages (+ their embeddings) from the DB.
#
# Safe to re-run at any time — idempotent.
# Run this before any analysis script if new corpora have been ingested.

# %% Imports
from db.schema import get_conn

# %% Boilerplate fingerprints — any passage matching ANY of these is deleted
GUTENBERG_PATTERNS = [
    "%Project Gutenberg%",
    "%gutenberg.org%",
    "%General Terms of Use%",
    "%END OF THE PROJECT GUTENBERG%",
    "%START OF THE PROJECT GUTENBERG%",
    "%GUTENBERG LITERARY ARCHIVE%",
    # License body text (distinct from header/footer)
    "%If you do not charge anything for copies of this eBook%",
    "%complying with the trademark license%",
    "%derivative works, reports, performances%",
    "%electronic works in formats readable%",
    "%gutenberg Literary Archive Foundation%",
    "%www.gutenberg%",
    "%Gutenberg-tm%",
]

# %% Find polluted passages
conn = get_conn()

where_clause = " OR ".join(f"p.text ILIKE ?" for _ in GUTENBERG_PATTERNS)
params = GUTENBERG_PATTERNS

preview = conn.execute(f"""
    SELECT c.name, p.id, p.unit_label, p.text
    FROM passage p
    JOIN corpus c ON p.corpus_id = c.id
    WHERE {where_clause}
    ORDER BY c.name, p.id
""", params).fetchall()

if not preview:
    print("No Gutenberg boilerplate found — DB is clean.")
    conn.close()
else:
    print(f"Found {len(preview)} polluted passages:\n")
    for corpus, pid, label, text in preview:
        print(f"  [{corpus}] {label} (id={pid})")
        print(f"    {text[:120]}")
        print()

    # %% Delete embeddings first (FK constraint), then passages
    polluted_ids = [row[1] for row in preview]

    conn.execute(f"""
        DELETE FROM embedding
        WHERE passage_id IN (
            SELECT p.id FROM passage p
            JOIN corpus c ON p.corpus_id = c.id
            WHERE {where_clause}
        )
    """, params)

    conn.execute(f"""
        DELETE FROM passage p
        WHERE {where_clause.replace('p.text', 'text')}
    """, params)

    # Verify
    remaining = conn.execute(f"""
        SELECT COUNT(*) FROM passage p WHERE {where_clause}
    """, params).fetchone()[0]

    print(f"Deleted {len(polluted_ids)} passages and their embeddings.")
    print(f"Remaining polluted passages: {remaining}")

    # %% Summary
    print("\n=== DB passage counts after cleaning ===")
    rows = conn.execute("""
        SELECT t.name, c.name, COUNT(p.id)
        FROM corpus c
        JOIN corpus_tradition t ON c.tradition_id = t.id
        LEFT JOIN passage p ON p.corpus_id = c.id
        GROUP BY t.name, c.name
        ORDER BY t.name, c.name
    """).fetchall()
    for trad, corpus, n in rows:
        print(f"  [{trad:12s}] {corpus:45s}  {n:,}")

conn.close()
print("\nDone.")
