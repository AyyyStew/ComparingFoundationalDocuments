# %% [markdown]
# # Ingest & Embed — Scientific Texts + Siddhartha (Script 32)
#
# New 'Scientific' tradition:
#   - On the Origin of Species (Darwin) — CHAPTER I–XIV
#   - Varieties of Religious Experience (James) — LECTURE I–XX
#   - Civilization and Its Discontents (Freud) — sections I–VIII
#   - Opticks (Newton) — BOOK I–III
#   - Psychology of the Unconscious (Jung) — PART I–II
#
# Literature addition:
#   - Siddhartha (Hesse) — ALL-CAPS chapter titles
#
# All chunked to ~4-sentence passages; embedded with 'all-mpnet-base-v2'.

# %% Imports
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import (
    ingest_origin_of_species,
    ingest_varieties_religious_experience,
    ingest_civilization_discontents,
    ingest_opticks,
    ingest_psychology_unconscious,
    ingest_siddhartha,
)

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

CORPORA = [
    ("Origin of Species",              lambda conn: ingest_origin_of_species(
        "data/scientific/on_the_origin_of_the_species.txt", conn=conn)),
    ("Varieties of Religious Exp.",    lambda conn: ingest_varieties_religious_experience(
        "data/scientific/varieties_of_religous_experience.txt", conn=conn)),
    ("Civilization & Its Discontents", lambda conn: ingest_civilization_discontents(
        "data/scientific/civilization_and_its_discontents.txt", conn=conn)),
    ("Opticks",                        lambda conn: ingest_opticks(
        "data/scientific/optiks.txt", conn=conn)),
    ("Psychology of the Unconscious",  lambda conn: ingest_psychology_unconscious(
        "data/scientific/psychology_of_the_unconscious.txt", conn=conn)),
    ("Siddhartha",                     lambda conn: ingest_siddhartha(
        "data/literature/siddartha.txt", conn=conn)),
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

# %% Summary — all corpora in DB
print("\n" + "="*60)
print("FULL DB SUMMARY")
print("="*60)
rows = conn.execute(
    """
    SELECT t.name, c.name, COUNT(p.id), COUNT(e.passage_id)
    FROM corpus c
    JOIN corpus_tradition t ON c.tradition_id = t.id
    LEFT JOIN passage p ON p.corpus_id = c.id
    LEFT JOIN embedding e ON e.passage_id = p.id AND e.model_name = ?
    GROUP BY t.name, c.name
    ORDER BY t.name, c.name
    """,
    [MODEL_NAME],
).fetchall()
for trad, name, np_, ne in rows:
    print(f"  [{trad:12s}] {name:45s}  p={np_:,}  e={ne:,}")

conn.close()
print("\nDone.")
