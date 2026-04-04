# %% [markdown]
# # Export DuckDB → Parquet
#
# Exports all four tables to `data/parquet/`.
# Vectors are stored as a float32 array column (compatible with most Parquet readers).
# Run this before uploading to Kaggle.

import os
import numpy as np
import pandas as pd
from db.schema import get_conn

OUT_DIR = "data/parquet"
os.makedirs(OUT_DIR, exist_ok=True)

conn = get_conn()

# ── corpus_tradition ─────────────────────────────────────────────────────────
df = conn.execute("SELECT * FROM corpus_tradition ORDER BY id").df()
path = f"{OUT_DIR}/corpus_tradition.parquet"
df.to_parquet(path, index=False)
print(f"Exported {len(df):,} rows → {path}")

# ── corpus ───────────────────────────────────────────────────────────────────
df = conn.execute("SELECT * FROM corpus ORDER BY id").df()
path = f"{OUT_DIR}/corpus.parquet"
df.to_parquet(path, index=False)
print(f"Exported {len(df):,} rows → {path}")

# ── passage ──────────────────────────────────────────────────────────────────
df = conn.execute("SELECT * FROM passage ORDER BY id").df()
path = f"{OUT_DIR}/passage.parquet"
df.to_parquet(path, index=False)
print(f"Exported {len(df):,} rows → {path}")

# ── embedding ────────────────────────────────────────────────────────────────
# DuckDB returns FLOAT[] as numpy arrays; convert to a single float32 2D array
# stored as object column so pyarrow can serialise it as a list<float32>.
rows = conn.execute(
    "SELECT passage_id, model_name, vector FROM embedding ORDER BY passage_id"
).fetchall()

emb_df = pd.DataFrame(rows, columns=["passage_id", "model_name", "vector"])
emb_df["vector"] = emb_df["vector"].apply(lambda v: np.array(v, dtype="float32"))

path = f"{OUT_DIR}/embedding.parquet"
emb_df.to_parquet(path, index=False)
print(f"Exported {len(emb_df):,} rows → {path}")

conn.close()

# Report sizes
total = 0
for fname in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, fname)
    size = os.path.getsize(fpath)
    total += size
    print(f"  {fname}: {size / 1e6:.1f} MB")
print(f"Total: {total / 1e6:.1f} MB")
