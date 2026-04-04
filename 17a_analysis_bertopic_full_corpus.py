# %% [markdown]
# # BERTopic — Full Corpus (Option D)
# Run BERTopic across all traditions simultaneously using pre-computed embeddings.
# The goal is to find which topics are tradition-pure vs cross-tradition mixed.
# Mixed topics are the interesting ones — they show shared semantic ground.

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
    make_vectorizer, SKIP_CORPORA, TRADITION_GROUP, GROUP_COLORS
)

MODEL_NAME = "all-mpnet-base-v2"

# %% Load passages + embeddings
conn = get_conn()

rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus,
           p.text, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()

conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "text", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df["group"] = df["tradition"].map(TRADITION_GROUP).fillna("Other")

embeddings = np.stack(df["vector"].apply(np.array).values).astype("float32")
docs = df["text"].tolist()

print(f"Loaded {len(df):,} passages across {df['corpus'].nunique()} corpora")
print(df.groupby("group").size().to_string())

# %% Fit BERTopic
# Use custom UMAP/HDBSCAN so we control the clustering behaviour.
# n_components=5 for clustering (BERTopic default pattern) keeps more signal
# than going straight to 2D.

umap_model = UMAP(
    n_components=5, n_neighbors=15, min_dist=0.0,
    metric="cosine", random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=60, min_samples=10,
    metric="euclidean", prediction_data=True
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=make_vectorizer(),
    nr_topics="auto",
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
df["topic"] = topics

print(f"\nTopics found: {topic_model.get_topic_info().shape[0] - 1} (excl. outliers)")
print(topic_model.get_topic_info().head(20).to_string())

# %% ── Plot 1: Topic size + tradition breakdown (stacked bar) ─────────────────
# For each topic (excl. -1 outliers), show how many passages come from each group.

topic_info = topic_model.get_topic_info()
topic_info = topic_info[topic_info["Topic"] != -1].copy()

# Build per-topic group counts
topic_group = (
    df[df["topic"] != -1]
    .groupby(["topic", "group"])
    .size()
    .unstack(fill_value=0)
)
# Sort by total size descending, keep top 30
topic_group["_total"] = topic_group.sum(axis=1)
topic_group = topic_group.sort_values("_total", ascending=False).head(30)
topic_group = topic_group.drop(columns="_total")

# Topic labels: top 3 words
topic_labels = {
    row["Topic"]: " / ".join([w for w, _ in topic_model.get_topic(row["Topic"])[:3]])
    for _, row in topic_info.iterrows()
}
index_labels = [f"T{t}: {topic_labels.get(t, '')}" for t in topic_group.index]

fig, ax = plt.subplots(figsize=(14, 8))
bottom = np.zeros(len(topic_group))
for group in GROUP_COLORS:
    if group not in topic_group.columns:
        continue
    vals = topic_group[group].values
    ax.bar(range(len(topic_group)), vals, bottom=bottom,
           color=GROUP_COLORS[group], label=group)
    bottom += vals

ax.set_xticks(range(len(topic_group)))
ax.set_xticklabels(index_labels, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("Passages")
ax.set_title("Topic Composition by Tradition Group (top 30 topics)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("17a_topic_composition.png", dpi=150)
plt.show()
print("Saved: 17a_topic_composition.png")

# %% ── Plot 2: Tradition × Topic heatmap ──────────────────────────────────────
# Normalised within each tradition — shows what fraction of each tradition's
# passages fall into each topic. Reveals tradition-specific vs shared topics.

# Use top 25 topics by total size
top_topics = (
    df[df["topic"] != -1]
    .groupby("topic").size()
    .sort_values(ascending=False)
    .head(25).index
)

heatmap_df = (
    df[df["topic"].isin(top_topics)]
    .groupby(["group", "topic"])
    .size()
    .unstack(fill_value=0)
)
# Normalise each row (tradition) to fractions
heatmap_norm = heatmap_df.div(heatmap_df.sum(axis=1), axis=0)
heatmap_norm.columns = [f"T{t}: {topic_labels.get(t, '')}" for t in heatmap_norm.columns]

fig, ax = plt.subplots(figsize=(16, 5))
im = ax.imshow(heatmap_norm.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(heatmap_norm.columns)))
ax.set_xticklabels(heatmap_norm.columns, rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(len(heatmap_norm.index)))
ax.set_yticklabels(heatmap_norm.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Fraction of group's passages")
ax.set_title("Topic Distribution by Tradition Group (normalised)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("17a_topic_heatmap.png", dpi=150)
plt.show()
print("Saved: 17a_topic_heatmap.png")

# %% ── Print: most cross-tradition topics ────────────────────────────────────
# Topics where 3+ groups are present with >10% share each = genuinely mixed

print("\n=== Most cross-tradition topics ===")
for topic in top_topics:
    grp_shares = (
        df[df["topic"] == topic]
        .groupby("group").size()
    )
    grp_shares = grp_shares / grp_shares.sum()
    n_groups_above_10 = (grp_shares > 0.10).sum()
    if n_groups_above_10 >= 3:
        words = " / ".join([w for w, _ in topic_model.get_topic(topic)[:5]])
        print(f"\n  Topic {topic} [{words}]")
        for g, s in grp_shares.sort_values(ascending=False).items():
            print(f"    {g:<15} {s:.1%}")

# %% ── Print: most tradition-pure topics ─────────────────────────────────────

print("\n=== Most tradition-pure topics (>80% from one group) ===")
for topic in top_topics:
    grp_shares = (
        df[df["topic"] == topic]
        .groupby("group").size()
    )
    grp_shares = grp_shares / grp_shares.sum()
    dominant = grp_shares.idxmax()
    if grp_shares[dominant] > 0.80:
        words = " / ".join([w for w, _ in topic_model.get_topic(topic)[:5]])
        total = df[df["topic"] == topic].shape[0]
        print(f"  Topic {topic} [{words}] → {dominant} ({grp_shares[dominant]:.1%}, {total} passages)")
