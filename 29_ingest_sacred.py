# %% [markdown]
# # Ingest & Embed — New Sacred Texts (Script 29)
#
# Corpora added:
#   - Quran (Clear Quran Translation) — Abrahamic, verse-level (~6,236 ayahs)
#   - Upanishads (Paramananda) — Dharmic, 3 texts chunked (~4-sentence passages)
#   - Srimad Bhagavatam — Dharmic, verse-level (~13,000 verses)
#   - Diamond Sutra (Gemmell) — Buddhist, chunked by chapter
#
# All use the existing 'all-mpnet-base-v2' model.

# %% Imports
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import (
    ingest_quran,
    ingest_upanishads,
    ingest_srimad_bhagavatam,
    ingest_diamond_sutra,
)

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

CORPORA = [
    ("Quran",            lambda conn: ingest_quran(
        "data/islamic/The Quran Dataset.csv", conn=conn)),
    ("Upanishads",       lambda conn: ingest_upanishads(
        "data/dharmic/upanishads.txt", conn=conn)),
    ("Srimad Bhagavatam", lambda conn: ingest_srimad_bhagavatam(
        "data/dharmic/Srimad_Bhagavatam_Data.csv", conn=conn)),
    ("Diamond Sutra",    lambda conn: ingest_diamond_sutra(
        "data/buhhdist/diamond_sutra.txt", conn=conn)),
]

# %% Ingest + Embed
conn  = get_conn()
model = None   # load lazily to avoid holding VRAM during ingest

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
for label, ingest_fn in CORPORA:
    row = conn.execute(
        """
        SELECT COUNT(p.id), COUNT(e.passage_id)
        FROM passage p
        JOIN corpus c ON p.corpus_id = c.id
        LEFT JOIN embedding e ON e.passage_id = p.id AND e.model_name = ?
        WHERE c.name LIKE ?
        """,
        [MODEL_NAME, f"%{label.split()[0]}%"],
    ).fetchone()
    print(f"  {label:30s}  passages={row[0]:,}  embeddings={row[1]:,}")

conn.close()
print("\nDone.")
