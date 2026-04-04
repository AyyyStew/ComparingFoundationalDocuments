# %% [markdown]
# # Ingest & Embed — Philosophy Texts (Script 31)
#
# New 'Philosophy' tradition added:
#   - The Republic (Plato, Jowett) — BOOK I–X
#   - Nicomachean Ethics (Aristotle) — BOOK I–X
#   - Beyond Good and Evil (Nietzsche) — CHAPTER I–IX
#   - Thus Spake Zarathustra (Nietzsche) — PART I–IV
#   - Critique of Pure Reason (Kant) — major sections
#   - Discourse on the Method (Descartes) — PART I–VI
#
# All chunked to ~4-sentence passages; embedded with 'all-mpnet-base-v2'.

# %% Imports
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import (
    ingest_the_republic,
    ingest_ethics_aristotle,
    ingest_beyond_good_and_evil,
    ingest_thus_spake_zarathustra,
    ingest_critique_pure_reason,
    ingest_discourse_on_method,
)

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

CORPORA = [
    ("The Republic",           lambda conn: ingest_the_republic(
        "data/philosophy/the_republic.txt", conn=conn)),
    ("Ethics (Aristotle)",     lambda conn: ingest_ethics_aristotle(
        "data/philosophy/the_ethics_of_aristole.txt", conn=conn)),
    ("Beyond Good and Evil",   lambda conn: ingest_beyond_good_and_evil(
        "data/philosophy/beyond_good_and_evil.txt", conn=conn)),
    ("Thus Spake Zarathustra", lambda conn: ingest_thus_spake_zarathustra(
        "data/philosophy/thus_spake_zara.txt", conn=conn)),
    ("Critique of Pure Reason", lambda conn: ingest_critique_pure_reason(
        "data/philosophy/critique_of_pure_reason.txt", conn=conn)),
    ("Discourse on Method",    lambda conn: ingest_discourse_on_method(
        "data/philosophy/discourse_on_the_method.txt", conn=conn)),
]

# %% Ingest + Embed
conn  = get_conn()
model = None

for label, ingest_fn in CORPORA:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    corpus_id, passage_ids = ingest_fn(conn)

    existing = conn.execute(
        """
        SELECT COUNT(*) FROM embedding e
        JOIN passage p ON e.passage_id = p.id
        WHERE p.corpus_id = ? AND e.model_name = ?
        """,
        [corpus_id, MODEL_NAME],
    ).fetchone()[0]

    if existing and not FORCE:
        print(f"  Embeddings already present ({existing:,}) — skipping")
        continue

    if existing and FORCE:
        print(f"  FORCE=True — deleting {existing:,} existing embeddings...")
        conn.execute(
            "DELETE FROM embedding WHERE model_name = ? AND passage_id IN "
            "(SELECT id FROM passage WHERE corpus_id = ?)",
            [MODEL_NAME, corpus_id],
        )

    rows  = conn.execute(
        "SELECT id, text FROM passage WHERE corpus_id = ? ORDER BY id", [corpus_id]
    ).fetchall()
    ids   = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    if model is None:
        print(f"\nLoading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, device="cuda")

    print(f"  Encoding {len(texts):,} passages...")
    vectors = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    print("  Writing embeddings to DB...")
    conn.executemany(
        "INSERT INTO embedding (passage_id, model_name, vector) VALUES (?, ?, ?)",
        [(int(pid), MODEL_NAME, vec.tolist()) for pid, vec in zip(ids, vectors)],
    )
    print(f"  Stored {len(ids):,} embeddings")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% Summary
print("\n" + "="*60)
print("SUMMARY — Philosophy tradition")
print("="*60)
rows = conn.execute(
    """
    SELECT c.name, COUNT(p.id), COUNT(e.passage_id)
    FROM corpus c
    JOIN corpus_tradition t ON c.tradition_id = t.id
    LEFT JOIN passage p ON p.corpus_id = c.id
    LEFT JOIN embedding e ON e.passage_id = p.id AND e.model_name = ?
    WHERE t.name = 'Philosophy'
    GROUP BY c.name
    ORDER BY c.name
    """,
    [MODEL_NAME],
).fetchall()
for name, np_, ne in rows:
    print(f"  {name:45s}  passages={np_:,}  embeddings={ne:,}")

conn.close()
print("\nDone.")
