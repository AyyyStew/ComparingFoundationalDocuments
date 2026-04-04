# %% [markdown]
# # Ingest & Embed — Literature
# Ingests Frankenstein, Pride and Prejudice, Don Quixote, Romeo and Juliet
# into corpus.duckdb, then encodes with all-mpnet-base-v2 on GPU.
#
# Granularity:
#   Novels     — 4-sentence chunks per chapter (comparable density to a verse)
#   Shakespeare — one passage per scene (dialogue only, stage directions stripped)

# %% Imports
import numpy as np
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import (
    ingest_frankenstein,
    ingest_pride_and_prejudice,
    ingest_don_quixote,
    ingest_romeo_and_juliet,
)

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

TEXTS = [
    ("Frankenstein",        "data/literature/frankenstein.txt",        ingest_frankenstein),
    ("Pride and Prejudice", "data/literature/pride_and_prejudice.txt",  ingest_pride_and_prejudice),
    ("Don Quixote",         "data/literature/don_quixote.txt",          ingest_don_quixote),
    ("Romeo and Juliet",    "data/literature/romeo_and_juliet.txt",     ingest_romeo_and_juliet),
]

# %% Ingest all texts
conn = get_conn()
corpus_ids = {}

for label, path, fn in TEXTS:
    print(f"\n{'='*50}\n{label}")
    corpus_id, _ = fn(path, conn=conn)
    corpus_ids[label] = corpus_id

# %% Embed all
import torch
torch.cuda.empty_cache()
print(f"\nLoading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device="cuda")

for label, corpus_id in corpus_ids.items():
    existing = conn.execute(
        """
        SELECT COUNT(*) FROM embedding e
        JOIN passage p ON e.passage_id = p.id
        WHERE p.corpus_id = ? AND e.model_name = ?
        """,
        [corpus_id, MODEL_NAME],
    ).fetchone()[0]

    if existing and not FORCE:
        print(f"[{label}] embeddings already in DB ({existing:,}) — skipping")
        continue

    if existing and FORCE:
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

    print(f"[{label}] encoding {len(texts):,} passages...")
    vectors = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    conn.executemany(
        "INSERT INTO embedding (passage_id, model_name, vector) VALUES (?, ?, ?)",
        [(int(pid), MODEL_NAME, vec.tolist()) for pid, vec in zip(ids, vectors)],
    )
    print(f"[{label}] stored {len(ids):,} embeddings")

# %% Summary
print("\n=== Corpus DB summary ===")
rows = conn.execute("""
    SELECT c.name, COUNT(p.id) AS passages,
           SUM(CASE WHEN e.passage_id IS NOT NULL THEN 1 ELSE 0 END) AS embedded
    FROM corpus c
    LEFT JOIN passage p  ON p.corpus_id = c.id
    LEFT JOIN embedding e ON e.passage_id = p.id AND e.model_name = ?
    GROUP BY c.name
    ORDER BY c.name
""", [MODEL_NAME]).fetchall()

for name, passages, embedded in rows:
    print(f"  {name:<45} {passages:>7,} passages  {embedded:>7,} embedded")

conn.close()
