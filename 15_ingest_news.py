# %% [markdown]
# # Ingest & Embed — News Articles
# Loads data/news/Articles.csv (2,692 articles; sports + business).
# Each article is split into 4-sentence chunks and stored in corpus.duckdb.
# Then encoded with all-mpnet-base-v2 on GPU.

# %% Imports
import torch
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import ingest_news_articles

CSV_PATH   = "data/news/Articles.csv"
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

# %% Ingest
conn = get_conn()
corpus_id, passage_ids = ingest_news_articles(CSV_PATH, conn=conn)

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

    torch.cuda.empty_cache()
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    print(f"Encoding {len(texts):,} passages...")
    vectors = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    print("Writing embeddings to DB...")
    conn.executemany(
        "INSERT INTO embedding (passage_id, model_name, vector) VALUES (?, ?, ?)",
        [(int(pid), MODEL_NAME, vec.tolist()) for pid, vec in zip(ids, vectors)],
    )
    print(f"Stored {len(ids):,} embeddings")

# %% Summary
total_p = conn.execute(
    "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
).fetchone()[0]
total_e = conn.execute(
    "SELECT COUNT(*) FROM embedding e JOIN passage p ON e.passage_id = p.id "
    "WHERE p.corpus_id = ? AND e.model_name = ?", [corpus_id, MODEL_NAME]
).fetchone()[0]

print(f"\nNews Articles in DB:")
print(f"  Passages:   {total_p:,}")
print(f"  Embeddings: {total_e:,}")

by_type = conn.execute(
    "SELECT book, COUNT(*) FROM passage WHERE corpus_id = ? GROUP BY book ORDER BY book",
    [corpus_id]
).fetchall()
for news_type, count in by_type:
    print(f"  {news_type:<12} {count:>6,} passages")

conn.close()
