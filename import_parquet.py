# %% [markdown]
# # Import Parquet → DuckDB
#
# Rebuilds `data/corpus.duckdb` from the four Parquet files in `data/parquet/`.
# Idempotent — skips rows that already exist (matched by natural key).
# Use this after downloading the Kaggle dataset.

import os
import numpy as np
import pandas as pd
from db.schema import get_conn

IN_DIR = "data/parquet"

for fname in ["corpus_tradition.parquet", "corpus.parquet", "passage.parquet", "embedding.parquet"]:
    if not os.path.exists(f"{IN_DIR}/{fname}"):
        raise FileNotFoundError(f"Missing: {IN_DIR}/{fname} — download the Kaggle dataset first.")

conn = get_conn()

# ── corpus_tradition ─────────────────────────────────────────────────────────
df = pd.read_parquet(f"{IN_DIR}/corpus_tradition.parquet")
existing = {r[0] for r in conn.execute("SELECT name FROM corpus_tradition").fetchall()}
new_rows = df[~df["name"].isin(existing)]
for _, row in new_rows.iterrows():
    conn.execute(
        "INSERT INTO corpus_tradition (id, name) VALUES (?, ?)",
        [int(row["id"]), row["name"]],
    )
conn.commit()
print(f"corpus_tradition: {len(new_rows)} inserted, {len(existing)} already present")

# ── corpus ───────────────────────────────────────────────────────────────────
df = pd.read_parquet(f"{IN_DIR}/corpus.parquet")
existing = {r[0] for r in conn.execute("SELECT name FROM corpus").fetchall()}
new_rows = df[~df["name"].isin(existing)]
for _, row in new_rows.iterrows():
    conn.execute(
        """INSERT INTO corpus (id, tradition_id, name, type, language, era, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [int(row["id"]), int(row["tradition_id"]), row["name"],
         row.get("type"), row.get("language"), row.get("era"), row.get("metadata")],
    )
conn.commit()
print(f"corpus: {len(new_rows)} inserted, {len(existing)} already present")

# ── passage ──────────────────────────────────────────────────────────────────
df = pd.read_parquet(f"{IN_DIR}/passage.parquet")
existing_ids = {r[0] for r in conn.execute("SELECT id FROM passage").fetchall()}
new_rows = df[~df["id"].isin(existing_ids)]

BATCH = 5000
for start in range(0, len(new_rows), BATCH):
    batch = new_rows.iloc[start : start + BATCH]
    conn.executemany(
        """INSERT INTO passage (id, corpus_id, book, section, unit_number, unit_label, text, metadata)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            [int(r["id"]), int(r["corpus_id"]), r.get("book"), r.get("section"),
             int(r["unit_number"]) if pd.notna(r.get("unit_number")) else None,
             r.get("unit_label"), r["text"], r.get("metadata")]
            for _, r in batch.iterrows()
        ],
    )
    conn.commit()
    print(f"  passage: inserted up to {start + len(batch):,} / {len(new_rows):,}", end="\r")

print(f"\npassage: {len(new_rows)} inserted, {len(existing_ids)} already present")

# ── embedding ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(f"{IN_DIR}/embedding.parquet")
existing_ids = {r[0] for r in conn.execute("SELECT passage_id FROM embedding").fetchall()}
new_rows = df[~df["passage_id"].isin(existing_ids)]

for start in range(0, len(new_rows), BATCH):
    batch = new_rows.iloc[start : start + BATCH]
    conn.executemany(
        "INSERT INTO embedding (passage_id, model_name, vector) VALUES (?, ?, ?)",
        [
            [int(r["passage_id"]), r["model_name"],
             r["vector"].astype("float32").tolist()]
            for _, r in batch.iterrows()
        ],
    )
    conn.commit()
    print(f"  embedding: inserted up to {start + len(batch):,} / {len(new_rows):,}", end="\r")

print(f"\nembedding: {len(new_rows)} inserted, {len(existing_ids)} already present")

conn.close()
print("\nDone. DB is ready.")
