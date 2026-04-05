# %% [markdown]
# # Ingest & Embed — Zoroastrian Texts (Yasna + Vendidad)
# Calls data/zoro/ingest_zoro.py loaders to ensure passages are in DB,
# then encodes both corpora with all-mpnet-base-v2 on GPU.

# %% Imports
import numpy as np
from sentence_transformers import SentenceTransformer

from db.schema import get_conn
from data.zoro.ingest_zoro import ingest_yasna, ingest_vendidad

MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 128
FORCE      = False

# %% Ingest passages
conn = get_conn()
y_corpus_id, _ = ingest_yasna(corpus_db_conn=conn)
v_corpus_id, _ = ingest_vendidad(corpus_db_conn=conn)

corpus_ids = [y_corpus_id, v_corpus_id]

# %% Embed each corpus
import torch

for corpus_id in corpus_ids:
    name = conn.execute(
        "SELECT name FROM corpus WHERE id = ?", [corpus_id]
    ).fetchone()[0]

    existing = conn.execute(
        """
        SELECT COUNT(*) FROM embedding e
        JOIN passage p ON e.passage_id = p.id
        WHERE p.corpus_id = ? AND e.model_name = ?
        """,
        [corpus_id, MODEL_NAME],
    ).fetchone()[0]

    if existing and not FORCE:
        print(f"[{name}] embeddings already in DB ({existing:,}) — skipping")
        continue

    if existing and FORCE:
        print(f"[{name}] FORCE=True — deleting {existing:,} existing embeddings...")
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

    print(f"[{name}] loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    print(f"[{name}] encoding {len(texts):,} passages...")
    vectors = model.encode(
        texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_numpy=True
    ).astype("float32")

    torch.cuda.empty_cache()

    print(f"[{name}] writing embeddings to DB...")
    conn.executemany(
        "INSERT INTO embedding (passage_id, model_name, vector) VALUES (?, ?, ?)",
        [(int(pid), MODEL_NAME, vec.tolist()) for pid, vec in zip(ids, vectors)],
    )
    print(f"[{name}] stored {len(ids):,} embeddings")

# %% Sanity check
print()
for corpus_id in corpus_ids:
    name = conn.execute("SELECT name FROM corpus WHERE id = ?", [corpus_id]).fetchone()[0]
    total_p = conn.execute(
        "SELECT COUNT(*) FROM passage WHERE corpus_id = ?", [corpus_id]
    ).fetchone()[0]
    total_e = conn.execute(
        "SELECT COUNT(*) FROM embedding e JOIN passage p ON e.passage_id = p.id "
        "WHERE p.corpus_id = ? AND e.model_name = ?", [corpus_id, MODEL_NAME]
    ).fetchone()[0]
    print(f"{name}")
    print(f"  Passages:   {total_p:,}")
    print(f"  Embeddings: {total_e:,}")

conn.close()
