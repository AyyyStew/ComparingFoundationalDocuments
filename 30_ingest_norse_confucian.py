# %% [markdown]
# # Ingest & Embed — Norse + Confucian Texts (Script 30)
#
# New traditions added:
#   - Poetic Edda (Bellows translation) — Norse, 30 named poems chunked
#   - Analects of Confucius (Legge translation) — Confucian, 20 books chunked
#
# All use the existing 'all-mpnet-base-v2' model.

# %% Imports
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import ingest_poetic_edda, ingest_analects

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

CORPORA = [
    ("Poetic Edda",  lambda conn: ingest_poetic_edda(
        "data/norse/poetic_eda.txt", conn=conn)),
    ("Analects",     lambda conn: ingest_analects(
        "data/philosophy/analects_confucian.txt", conn=conn)),
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
print("SUMMARY")
print("="*60)
rows = conn.execute(
    """
    SELECT c.name, t.name, COUNT(p.id), COUNT(e.passage_id)
    FROM corpus c
    JOIN corpus_tradition t ON c.tradition_id = t.id
    LEFT JOIN passage p ON p.corpus_id = c.id
    LEFT JOIN embedding e ON e.passage_id = p.id AND e.model_name = ?
    WHERE t.name IN ('Norse', 'Confucian')
    GROUP BY c.name, t.name
    """,
    [MODEL_NAME],
).fetchall()
for name, trad, np_, ne in rows:
    print(f"  [{trad:12s}] {name:40s}  passages={np_:,}  embeddings={ne:,}")

conn.close()
print("\nDone.")
