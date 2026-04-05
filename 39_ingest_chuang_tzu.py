# %% [markdown]
# # Ingest & Embed — Chuang Tzu
# Calls data/eastern/ingest_chuang_tzu.py loader to ensure passages are in DB,
# then encodes with all-mpnet-base-v2 on GPU.

# %% Imports
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from data.eastern.ingest_chuang_tzu import ingest_chuang_tzu

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

# %% Ingest passages
conn = get_conn()
corpus_id, passage_ids = ingest_chuang_tzu(corpus_db_conn=conn)

# %% Embed
existing = conn.execute(
    """
    SELECT COUNT(*) FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    WHERE p.corpus_id = ? AND e.model_name = ?
    """,
    [corpus_id, MODEL_NAME],
).fetchone()[0]

if existing and not FORCE:
    print(f"Embeddings already in DB ({existing:,} rows) — skipping")
else:
    import torch

    if existing and FORCE:
        print(f"FORCE=True — deleting {existing:,} existing embeddings...")
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

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    print(f"Encoding {len(texts):,} passages...")
    vectors = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    torch.cuda.empty_cache()

    print("Writing embeddings to DB...")
    conn.executemany(
        "INSERT INTO embedding (passage_id, model_name, vector) VALUES (?, ?, ?)",
        [(int(pid), MODEL_NAME, vec.tolist()) for pid, vec in zip(ids, vectors)],
    )
    print(f"Stored {len(ids):,} embeddings")

# %% Sanity check
total_p = conn.execute(
    "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
).fetchone()[0]
total_e = conn.execute(
    "SELECT COUNT(*) FROM embedding e JOIN passage p ON e.passage_id = p.id "
    "WHERE p.corpus_id = ? AND e.model_name = ?", [corpus_id, MODEL_NAME]
).fetchone()[0]

print(f"\nChuang Tzu in DB:")
print(f"  Passages:   {total_p:,}")
print(f"  Embeddings: {total_e:,}")

# %% Preview
sample = conn.execute(
    """
    SELECT p.book, p.unit_label, p.text
    FROM passage p WHERE p.corpus_id = ?
    ORDER BY p.id LIMIT 5
    """,
    [corpus_id],
).fetchall()

print("\nSample passages:")
for book, label, text in sample:
    print(f"  [{label}] {text[:90]}...")

conn.close()
