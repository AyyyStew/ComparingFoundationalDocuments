# %% [markdown]
# # Universal & Distinct Passages
#
# Two analyses using cross-tradition nearest-neighbor similarity:
#
# **Universal passages** — For each passage, find its closest match from a
# *different* tradition. Passages with the highest cross-tradition similarity
# are the "same thought, different culture" moments.
# Report the top pairs (A → B) ranked by cosine similarity.
#
# **Distinct passages** — Passages whose closest cross-tradition neighbor is
# still very *far* away. These have no semantic analog elsewhere in the corpus —
# genuinely unique ideas or styles. Ranked by lowest max cross-tradition similarity.
#
# Scope: all traditions except News (too noisy) and duplicate Bible translations.
# Computed in chunks to stay memory-friendly.
#
# Output: 35_universal_passages.txt, 35_distinct_passages.txt

# %% Imports
import numpy as np
import pandas as pd
from tqdm import tqdm

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA

MODEL_NAME  = "all-mpnet-base-v2"
TOP_N       = 20    # top pairs to report per section
CHUNK_SIZE  = 2000  # passages per chunk for similarity computation
SKIP_TRAD   = {"News"}  # exclude from both source and target

# %% Load
conn = get_conn()
rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus,
           p.id AS passage_id, p.unit_label, p.text, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY t.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "passage_id", "unit_label", "text", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df = df[~df["tradition"].isin(SKIP_TRAD)].reset_index(drop=True)
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} passages across {df['tradition'].nunique()} traditions")
print(df.groupby("tradition").size().to_string())

# %% L2-normalise
vecs = np.stack(df["vector"].values).astype("float32")
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs = vecs / np.clip(norms, 1e-9, None)

traditions    = df["tradition"].values
passage_ids   = df["passage_id"].values

# %% Cross-tradition nearest neighbor
# For each passage i, find the passage j with the highest cosine similarity
# where traditions[j] != traditions[i].
# We store:  best_cross_sim[i], best_cross_idx[i]

print("\nComputing cross-tradition nearest neighbors (chunked, vectorized) ...")

# Encode traditions as integers for fast broadcast masking
trad_labels, trad_codes = np.unique(traditions, return_inverse=True)
trad_codes = trad_codes.astype("int32")

n = len(df)
best_cross_sim = np.full(n, -np.inf, dtype="float32")
best_cross_idx = np.full(n, -1,      dtype="int32")

with tqdm(total=n, unit="passages", desc="NN search") as pbar:
    for start in range(0, n, CHUNK_SIZE):
        end   = min(start + CHUNK_SIZE, n)
        chunk = vecs[start:end]       # (chunk_size, 768)
        sims  = chunk @ vecs.T        # (chunk_size, n)  — BLAS, fast

        # Mask same-tradition: broadcast (chunk_size, 1) == (1, n)
        chunk_trads = trad_codes[start:end]          # (chunk_size,)
        same_mask   = chunk_trads[:, None] == trad_codes[None, :]  # (chunk_size, n)
        sims[same_mask] = -np.inf

        # Mask self-similarity
        local_idx = np.arange(end - start)
        sims[local_idx, np.arange(start, end)] = -np.inf

        # Vectorised argmax — no Python for loop
        best_j   = sims.argmax(axis=1)              # (chunk_size,)
        best_sim = sims[local_idx, best_j]          # (chunk_size,)

        best_cross_sim[start:end] = best_sim
        best_cross_idx[start:end] = best_j

        pbar.update(end - start)

print("Done.")

df["cross_sim"] = best_cross_sim
df["cross_idx"] = best_cross_idx

# %% Build output helpers
report_lines = []

def emit(line=""):
    print(line)
    report_lines.append(line)

def format_pair(i, j, sim):
    ri = df.iloc[i]
    rj = df.iloc[j]
    return (
        f"  sim={sim:.4f}\n"
        f"  A: [{ri['tradition']}] {ri['corpus']} | {ri['unit_label']}\n"
        f"     {ri['text'][:220]}\n"
        f"  B: [{rj['tradition']}] {rj['corpus']} | {rj['unit_label']}\n"
        f"     {rj['text'][:220]}"
    )

# %% Universal passages — highest cross-tradition similarity
emit("=" * 70)
emit(f"UNIVERSAL PASSAGES — Top {TOP_N} cross-tradition nearest-neighbor pairs")
emit("=" * 70)
emit("These passages say the same thing across different traditions.")
emit()

# Deduplicate: keep only rows where index < cross_idx so each (i,j) pair
# appears exactly once — both directions have the same sim value.
df_dedup = df[df.index < df["cross_idx"]].copy()
top_universal = df_dedup.nlargest(TOP_N, "cross_sim")
for rank, (_, row) in enumerate(top_universal.iterrows(), 1):
    i   = int(row.name)
    j   = int(row["cross_idx"])
    sim = float(row["cross_sim"])
    emit(f"\n[{rank}]")
    emit(format_pair(i, j, sim))

# %% Universal by tradition pair — top 3 per unique (trad_A, trad_B) combo
emit()
emit("=" * 70)
emit("TOP UNIVERSAL PAIRS — by tradition combination")
emit("=" * 70)

df["cross_trad"] = [df.iloc[int(j)]["tradition"] for j in df["cross_idx"]]
df["trad_pair"]  = df.apply(
    lambda r: tuple(sorted([r["tradition"], r["cross_trad"]])), axis=1
)

# Deduplicate per-pair section same way — only keep i < j rows
df_dedup2 = df[df.index < df["cross_idx"]].copy()
for pair, group in df_dedup2.groupby("trad_pair"):
    if pair[0] == pair[1]:
        continue
    top = group.nlargest(3, "cross_sim")
    emit(f"\n{'─'*60}")
    emit(f"  {pair[0]} ↔ {pair[1]}")
    for _, row in top.iterrows():
        i   = int(row.name)
        j   = int(row["cross_idx"])
        sim = float(row["cross_sim"])
        emit(format_pair(i, j, sim))
        emit()

# %% Distinct passages — lowest cross-tradition similarity
emit("=" * 70)
emit(f"DISTINCT PASSAGES — Top {TOP_N} most semantically isolated")
emit("=" * 70)
emit("These passages have no close analog anywhere else in the corpus.")
emit()

top_distinct = df.nsmallest(TOP_N, "cross_sim")
for rank, (_, row) in enumerate(top_distinct.iterrows(), 1):
    i   = df.index.get_loc(row.name)
    j   = int(row["cross_idx"])
    sim = float(row["cross_sim"])
    emit(f"\n[{rank}] Best cross-tradition match is only sim={sim:.4f}")
    ri = df.iloc[i]
    rj = df.iloc[j]
    emit(f"  SOURCE: [{ri['tradition']}] {ri['corpus']} | {ri['unit_label']}")
    emit(f"  {ri['text'][:300]}")
    emit(f"  Closest other: [{rj['tradition']}] {rj['corpus']} | {rj['unit_label']}")
    emit(f"  {rj['text'][:150]}")

# %% Distinct by tradition — which tradition has the most isolated passages on average?
emit()
emit("=" * 70)
emit("AVERAGE CROSS-TRADITION SIMILARITY BY TRADITION")
emit("(lower = more isolated / more unique)")
emit("=" * 70)
trad_stats = (
    df.groupby("tradition")["cross_sim"]
    .agg(mean="mean", median="median", min="min", max="max")
    .round(4)
    .sort_values("mean")
)
emit(trad_stats.to_string())

# Also per-corpus
emit()
emit("=== Per-corpus mean cross-tradition similarity ===")
corpus_stats = (
    df.groupby(["tradition", "corpus"])["cross_sim"]
    .mean()
    .round(4)
    .sort_values()
    .rename("mean_cross_sim")
)
emit(corpus_stats.to_string())

# %% Save reports
with open("34_universal_passages.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print("\nSaved: 34_universal_passages.txt")
