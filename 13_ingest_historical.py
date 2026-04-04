# %% [markdown]
# # Ingest & Embed — Historical Documents
# Ingests six historical/political texts into corpus.duckdb,
# then encodes with all-mpnet-base-v2 on GPU.
#
# Granularity:
#   Code of Hammurabi  — one law per section (~1–5 sentences)
#   Luther's 95 Theses — one passage per thesis
#   Magna Carta        — one passage per numbered clause
#   US Constitution    — one passage per Article/Section
#   Communist Manifesto — one paragraph per passage
#   Federalist Papers  — 4-sentence chunks per paper (85 papers)

# %% Imports
import torch
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import (
    ingest_code_of_hammurabi,
    ingest_luther_theses,
    ingest_magna_carta,
    ingest_us_constitution,
    ingest_communist_manifesto,
    ingest_federalist_papers,
)

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

TEXTS = [
    ("Code of Hammurabi",   "data/historical/code_of_hammurabi.txt",  ingest_code_of_hammurabi),
    ("Luther's 95 Theses",  "data/historical/luther_theses.txt",      ingest_luther_theses),
    ("Magna Carta",         "data/historical/magna_carta.txt",         ingest_magna_carta),
    ("US Constitution",     "data/historical/us_constitution.txt",     ingest_us_constitution),
    ("Communist Manifesto", "data/historical/communist_manifesto.txt", ingest_communist_manifesto),
    ("Federalist Papers",   "data/historical/federalist_papers.txt",   ingest_federalist_papers),
]

# %% Ingest all texts
conn = get_conn()
corpus_ids = {}

for label, path, fn in TEXTS:
    print(f"\n{'='*50}\n{label}")
    corpus_id, _ = fn(path, conn=conn)
    corpus_ids[label] = corpus_id

# %% Embed all
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
print("\n=== Corpus DB summary (Historical) ===")
rows = conn.execute("""
    SELECT c.name, COUNT(p.id) AS passages,
           SUM(CASE WHEN e.passage_id IS NOT NULL THEN 1 ELSE 0 END) AS embedded
    FROM corpus c
    JOIN corpus_tradition t ON c.tradition_id = t.id
    LEFT JOIN passage p  ON p.corpus_id = c.id
    LEFT JOIN embedding e ON e.passage_id = p.id AND e.model_name = ?
    WHERE t.name = 'Historical'
    GROUP BY c.name
    ORDER BY c.name
""", [MODEL_NAME]).fetchall()

for name, passages, embedded in rows:
    print(f"  {name:<45} {passages:>7,} passages  {embedded:>7,} embedded")

conn.close()
