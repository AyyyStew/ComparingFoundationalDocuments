# %% [markdown]
# # Cross-Tradition Nearest Neighbours
#
# For every passage, find its most similar passages from *other* tradition groups.
# Then run BERTopic on the pool of "bridge" passages — ones that were repeatedly
# pulled as a cross-tradition match. These are the passages living between
# traditions and are where genuinely shared themes surface.
#
# ## How the similarity search works
#
# Cosine similarity measures the angle between two embedding vectors — range is
# -1 to 1. 1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite meaning.
#
# Computing every passage against every other passage (41k × 41k) would require
# ~6.7 GB of memory and is wasteful — we only care about *cross-tradition* pairs.
# So instead we compute one tradition-pair at a time (e.g. Sacred vs Literature),
# which keeps peak memory under ~500 MB per chunk.
#
# Within each chunk:
#   1. L2-normalise both sets of vectors (divide each by its own length).
#      After this step, dot-product == cosine similarity — no extra math needed.
#   2. Matrix-multiply the two sets: result[i, j] = similarity(passage_i, passage_j)
#   3. Take only the top-k columns per row, discard the rest immediately.
#
# The output is: for every passage, its top-k nearest neighbours from each
# other tradition — without ever holding the full 41k × 41k matrix in memory.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

from db.schema import get_conn
from analysis_utils import (
    make_vectorizer, SKIP_CORPORA, TRADITION_GROUP, GROUP_COLORS, SACRED_TRADITIONS
)

MODEL_NAME = "all-mpnet-base-v2"
TOP_K      = 5    # neighbours per passage per opposing group
MIN_HITS   = 3    # a passage needs this many cross-tradition matches to be a "bridge"

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
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "passage_id", "unit_label", "text", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df["group"] = df["tradition"].map(TRADITION_GROUP).fillna("Other")

# Stack all vectors into a float32 matrix and L2-normalise
vecs = np.stack(df["vector"].apply(np.array).values).astype("float32")
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_normed = vecs / np.clip(norms, 1e-9, None)

print(f"Loaded {len(df):,} passages across {df['group'].nunique()} groups")
print(df.groupby("group").size().to_string())

# %% Cross-tradition nearest-neighbour search
# For each group-pair (A, B) find top-k passages in B for every passage in A,
# then also top-k in A for every passage in B. Accumulate as a dataframe of
# (passage_id_query, passage_id_match, group_query, group_match, similarity).

groups = sorted(df["group"].unique())
group_idx = {g: df.index[df["group"] == g].tolist() for g in groups}

records = []

for i, g_a in enumerate(groups):
    for g_b in groups[i+1:]:
        idx_a = group_idx[g_a]
        idx_b = group_idx[g_b]

        V_a = vecs_normed[idx_a]   # shape: (n_a, 768)
        V_b = vecs_normed[idx_b]   # shape: (n_b, 768)

        # sim[i, j] = cosine similarity between passage idx_a[i] and idx_b[j]
        sim = V_a @ V_b.T          # shape: (n_a, n_b)

        k_ab = min(TOP_K, sim.shape[1])
        k_ba = min(TOP_K, sim.shape[0])

        # Top-k in B for each passage in A
        top_cols = np.argpartition(sim, -k_ab, axis=1)[:, -k_ab:]
        for row_i, cols in enumerate(top_cols):
            for col_j in cols:
                records.append((
                    df.loc[idx_a[row_i], "passage_id"],
                    df.loc[idx_b[col_j], "passage_id"],
                    g_a, g_b,
                    float(sim[row_i, col_j]),
                ))

        # Top-k in A for each passage in B
        top_rows = np.argpartition(sim, -k_ba, axis=0)[-k_ba:, :]
        for col_j, rows_ in enumerate(top_rows.T):
            for row_i in rows_:
                records.append((
                    df.loc[idx_b[col_j], "passage_id"],
                    df.loc[idx_a[row_i], "passage_id"],
                    g_b, g_a,
                    float(sim[row_i, col_j]),
                ))

        print(f"  {g_a} × {g_b}: {len(idx_a):,} × {len(idx_b):,} → done")

pairs_df = pd.DataFrame(records, columns=[
    "query_id", "match_id", "query_group", "match_group", "similarity"
]).drop_duplicates()

print(f"\nTotal cross-tradition pairs: {len(pairs_df):,}")
print(pairs_df.groupby(["query_group", "match_group"])["similarity"].mean().to_string())

# %% Identify bridge passages
# A bridge passage is one that appears frequently as a match for other traditions.
# Count how many distinct cross-tradition matches each passage has.

match_counts = (
    pairs_df.groupby("match_id")["query_group"]
    .nunique()
    .rename("n_traditions_matched")
    .reset_index()
    .rename(columns={"match_id": "passage_id"})
)

df = df.merge(match_counts, on="passage_id", how="left")
df["n_traditions_matched"] = df["n_traditions_matched"].fillna(0).astype(int)

# Build bridge pool: passages matched by MIN_HITS or more distinct tradition groups
bridge_ids = set(df[df["n_traditions_matched"] >= MIN_HITS]["passage_id"])
bridge_df  = df[df["passage_id"].isin(bridge_ids)].copy()

print(f"\nBridge passages (matched by {MIN_HITS}+ groups): {len(bridge_df):,}")
print(bridge_df.groupby("group").size().to_string())

# %% ── Plot 1: Top cross-tradition pairs by similarity ────────────────────────
# Show the 20 highest-similarity cross-tradition pairs as a readable table

id_to_row = df.set_index("passage_id")

top_pairs = (
    pairs_df[pairs_df["query_group"] != pairs_df["match_group"]]
    .sort_values("similarity", ascending=False)
    .drop_duplicates(subset=["query_id", "match_id"])
    .head(20)
)

print("\n=== Top 20 cross-tradition pairs by similarity ===")
for _, row in top_pairs.iterrows():
    q = id_to_row.loc[row["query_id"]]
    m = id_to_row.loc[row["match_id"]]
    print(f"\n  [{row['similarity']:.3f}] {q['corpus']} | {q['unit_label']}")
    print(f"    {q['text'][:120]}")
    print(f"  ↔ {m['corpus']} | {m['unit_label']}")
    print(f"    {m['text'][:120]}")

# %% ── Plot 2: Bridge passage leaderboard ─────────────────────────────────────
# Top 15 most universally resonant passages — highest cross-tradition match count

top_bridges = (
    df[df["n_traditions_matched"] > 0]
    .sort_values("n_traditions_matched", ascending=False)
    .head(15)[["corpus", "unit_label", "text", "group", "n_traditions_matched"]]
)

print("\n=== Top 15 most universally resonant passages ===")
for _, row in top_bridges.iterrows():
    print(f"\n  [{row['n_traditions_matched']} traditions] {row['corpus']} | {row['unit_label']}")
    print(f"    {row['text'][:150]}")

# %% ── Plot 3: Cross-tradition similarity heatmap ─────────────────────────────
# Mean similarity between each pair of groups

pivot = (
    pairs_df.groupby(["query_group", "match_group"])["similarity"]
    .mean()
    .reset_index()
)
# Make it symmetric
pivot_rev = pivot.rename(columns={"query_group": "match_group", "match_group": "query_group"})
pivot_sym = pd.concat([pivot, pivot_rev]).groupby(["query_group", "match_group"])["similarity"].mean().unstack()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(pivot_sym.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(pivot_sym.columns)))
ax.set_xticklabels(pivot_sym.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(pivot_sym.index)))
ax.set_yticklabels(pivot_sym.index, fontsize=9)
for i in range(len(pivot_sym.index)):
    for j in range(len(pivot_sym.columns)):
        val = pivot_sym.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
plt.colorbar(im, ax=ax, label="Mean cosine similarity")
ax.set_title("Mean Cross-Tradition Similarity (top-k neighbours)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("18_cross_tradition_similarity.png", dpi=150)
plt.show()
print("Saved: 18_cross_tradition_similarity.png")

# %% ── BERTopic on bridge passages only ───────────────────────────────────────
# These passages are by construction the ones living between traditions.
# Topics found here are genuinely cross-tradition.

if len(bridge_df) < 100:
    print(f"\nOnly {len(bridge_df)} bridge passages — skipping BERTopic (need 100+)")
else:
    bridge_vecs = np.stack(bridge_df["vector"].apply(np.array).values).astype("float32")
    bridge_docs = bridge_df["text"].tolist()

    topic_model = BERTopic(
        umap_model=UMAP(n_components=5, n_neighbors=min(15, len(bridge_df) - 1),
                        min_dist=0.0, metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=max(10, len(bridge_df) // 50),
                              min_samples=5, metric="euclidean", prediction_data=True),
        vectorizer_model=make_vectorizer(),
        nr_topics="auto",
        verbose=True,
    )

    bridge_topics, _ = topic_model.fit_transform(bridge_docs, embeddings=bridge_vecs)
    bridge_df = bridge_df.copy()
    bridge_df["topic"] = bridge_topics

    n_topics = topic_model.get_topic_info().shape[0] - 1
    print(f"\nBridge BERTopic — {n_topics} topics found across {len(bridge_df):,} passages")

    topic_info = topic_model.get_topic_info()
    topic_info = topic_info[topic_info["Topic"] != -1]
    topic_labels = {
        row["Topic"]: " / ".join([w for w, _ in topic_model.get_topic(row["Topic"])[:4]])
        for _, row in topic_info.iterrows()
    }

    # Stacked bar — tradition breakdown per bridge topic
    topic_grp = (
        bridge_df[bridge_df["topic"] != -1]
        .groupby(["topic", "group"]).size()
        .unstack(fill_value=0)
    )
    topic_grp["_total"] = topic_grp.sum(axis=1)
    topic_grp = topic_grp.sort_values("_total", ascending=False).head(25).drop(columns="_total")
    index_labels = [f"T{t}: {topic_labels.get(t, '')}" for t in topic_grp.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(topic_grp))
    for group, color in GROUP_COLORS.items():
        if group not in topic_grp.columns:
            continue
        vals = topic_grp[group].values
        ax.bar(range(len(topic_grp)), vals, bottom=bottom, color=color, label=group)
        bottom += vals

    ax.set_xticks(range(len(topic_grp)))
    ax.set_xticklabels(index_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Passages")
    ax.set_title("Bridge Passage Topics — Cross-Tradition Only (top 25)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("18_bridge_topics.png", dpi=150)
    plt.show()
    print("Saved: 18_bridge_topics.png")

    # Print top topics with sample passages from each tradition
    print("\n=== Bridge topics with sample passages ===")
    top_bridge_topics = (
        bridge_df[bridge_df["topic"] != -1]
        .groupby("topic").size()
        .sort_values(ascending=False)
        .head(10).index
    )
    for topic in top_bridge_topics:
        words = " / ".join([w for w, _ in topic_model.get_topic(topic)[:5]])
        topic_passages = bridge_df[bridge_df["topic"] == topic]
        groups_present = topic_passages["group"].value_counts()
        print(f"\n  Topic {topic} [{words}] — {len(topic_passages)} passages")
        for g, cnt in groups_present.items():
            sample = topic_passages[topic_passages["group"] == g].iloc[0]
            print(f"    {g} ({cnt}): [{sample['corpus']} | {sample['unit_label']}]")
            print(f"      {sample['text'][:120]}")
