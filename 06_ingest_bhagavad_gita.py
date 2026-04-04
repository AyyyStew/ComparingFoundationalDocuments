# %% [markdown]
# # Ingest & Embed — Bhagavad Gita
# Loads data/dharmic/bhagavad_gita_verses.csv into corpus.duckdb,
# encodes with all-mpnet-base-v2 on GPU, and stores embeddings.

# %% Imports
import numpy as np
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import ingest_bhagavad_gita

CSV_PATH   = "data/dharmic/bhagavad_gita_verses.csv"
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 512
FORCE      = False  # set True to re-encode even if embeddings exist

# %% Ingest passages
conn = get_conn()
corpus_id, passage_ids = ingest_bhagavad_gita(CSV_PATH, corpus_db_conn=conn)

# %% Check if embeddings already exist
existing = conn.execute(
    """
    SELECT COUNT(*) FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    WHERE p.corpus_id  = ?
      AND e.model_name = ?
    """,
    [corpus_id, MODEL_NAME],
).fetchone()[0]

if existing and not FORCE:
    print(f"Embeddings already in DB ({existing:,} rows) — skipping")
else:
    if existing and FORCE:
        print(f"FORCE=True — deleting {existing:,} existing embeddings...")
        conn.execute(
            "DELETE FROM embedding WHERE model_name = ? AND passage_id IN "
            "(SELECT id FROM passage WHERE corpus_id = ?)",
            [MODEL_NAME, corpus_id],
        )

    # %% Fetch texts
    rows = conn.execute(
        "SELECT id, text FROM passage WHERE corpus_id = ? ORDER BY id",
        [corpus_id],
    ).fetchall()
    ids   = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    # %% Encode on GPU
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    print(f"Encoding {len(texts):,} passages...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    # %% Write to DB
    print("Writing embeddings to DB...")
    conn.executemany(
        "INSERT INTO embedding (passage_id, model_name, vector) VALUES (?, ?, ?)",
        [(int(pid), MODEL_NAME, vec.tolist()) for pid, vec in zip(ids, vectors)],
    )
    print(f"Stored {len(ids):,} embeddings")

# %% Sanity check
total_passages  = conn.execute(
    "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
).fetchone()[0]
total_embeddings = conn.execute(
    "SELECT COUNT(*) FROM embedding e JOIN passage p ON e.passage_id = p.id "
    "WHERE p.corpus_id = ? AND e.model_name = ?", [corpus_id, MODEL_NAME]
).fetchone()[0]

print(f"\nBhagavad Gita in DB:")
print(f"  Passages:   {total_passages:,}")
print(f"  Embeddings: {total_embeddings:,}")

# %% Preview — sample a few passages
sample = conn.execute(
    """
    SELECT p.book, p.unit_label, p.text
    FROM passage p
    WHERE p.corpus_id = ?
    ORDER BY p.id
    LIMIT 5
    """,
    [corpus_id],
).fetchall()

print("\nSample passages:")
for book, label, text in sample:
    print(f"  [{book} {label}] {text[:80]}...")

conn.close()
