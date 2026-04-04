"""
compute_embeddings.py

For each bible translation:
  1. Ingest passages into DuckDB (skips if already present)
  2. Encode verses with all-mpnet-base-v2 on GPU
  3. Write embeddings into the embedding table (skips if already present)

Usage:
    python compute_embeddings.py
    python compute_embeddings.py --force   # re-encode even if embeddings exist
"""

import argparse
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from db.ingest import ingest_scrollmapper_bible

TRANSLATIONS = ["KJV", "ACV", "YLT", "BBE"]
BIBLE_DB_DIR = "data/bibles"
MODEL_NAME   = "all-mpnet-base-v2"
BATCH_SIZE   = 512


def encode_and_store(
    translation: str,
    conn,
    model: SentenceTransformer,
    force: bool = False,
) -> None:
    db_path = os.path.join(BIBLE_DB_DIR, f"{translation}.db")

    # 1. Ingest passages (no-op if already done)
    corpus_id, passage_ids = ingest_scrollmapper_bible(translation, db_path, conn)

    # 2. Check if embeddings already exist
    existing = conn.execute(
        """
        SELECT COUNT(*) FROM embedding e
        JOIN passage p ON e.passage_id = p.id
        WHERE p.corpus_id = ? AND e.model_name = ?
        """,
        [corpus_id, MODEL_NAME],
    ).fetchone()[0]

    if existing and not force:
        print(f"[{translation}] embeddings already in DB ({existing:,} rows) — skipping")
        return

    if existing and force:
        print(f"[{translation}] force=True, deleting {existing:,} existing embeddings...")
        conn.execute(
            """
            DELETE FROM embedding WHERE model_name = ?
              AND passage_id IN (SELECT id FROM passage WHERE corpus_id = ?)
            """,
            [MODEL_NAME, corpus_id],
        )

    # 3. Fetch texts in passage_id order
    rows = conn.execute(
        "SELECT id, text FROM passage WHERE corpus_id = ? ORDER BY id",
        [corpus_id],
    ).fetchall()
    ids   = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    print(f"[{translation}] encoding {len(texts):,} passages on GPU...")
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    # 4. Bulk insert into embedding table
    rows_to_insert = [
        (int(pid), MODEL_NAME, vec.tolist())
        for pid, vec in zip(ids, vectors)
    ]
    if not rows_to_insert:
        print(f"[{translation}] no embeddings to write — skipping\n")
        return
    print(f"[{translation}] writing embeddings to DB...")
    conn.executemany(
        "INSERT INTO embedding (passage_id, model_name, vector) VALUES (?, ?, ?)",
        rows_to_insert,
    )
    print(f"[{translation}] done — {len(rows_to_insert):,} embeddings stored\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-encode even if embeddings exist")
    args = parser.parse_args()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    conn = get_conn()
    for t in TRANSLATIONS:
        encode_and_store(t, conn, model, force=args.force)

    total_emb = conn.execute("SELECT COUNT(*) FROM embedding").fetchone()[0]
    print(f"Total embeddings in DB: {total_emb:,}")
    conn.close()
