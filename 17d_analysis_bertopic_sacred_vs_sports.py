# %% [markdown]
# # BERTopic — Sacred Texts vs Sports News (Option D variant)
# The UMAP showed sports news clustering with sacred texts — not historical.
# BERTopic should reveal whether they actually share topics (narrative, struggle,
# redemption, heroism) or just occupy adjacent but distinct semantic space.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from db.schema import get_conn
from analysis_utils import make_vectorizer, SKIP_CORPORA, SACRED_TRADITIONS

MODEL_NAME = "all-mpnet-base-v2"

GROUP_COLORS = {
    "Sacred Texts": "#e05c5c",
    "Sports News":  "#44bb99",
}

# %% Load
conn = get_conn()
rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus,
           p.book, p.text, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "book", "text", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)]

# Keep sacred traditions + sports news only
is_sacred = df["tradition"].isin(SACRED_TRADITIONS)
is_sports  = (df["tradition"] == "News") & (df["book"] == "sports")
df = df[is_sacred | is_sports].reset_index(drop=True)
df["group"] = df.apply(
    lambda r: "Sports News" if r["tradition"] == "News" else "Sacred Texts", axis=1
)

embeddings = np.stack(df["vector"].apply(np.array).values).astype("float32")
docs = df["text"].tolist()

print(f"Loaded {len(df):,} passages")
print(df.groupby("group").size().to_string())

# %% Fit BERTopic
topic_model = BERTopic(
    umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                    metric="cosine", random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=50, min_samples=10,
                          metric="euclidean", prediction_data=True),
    vectorizer_model=make_vectorizer(),
    nr_topics="auto",
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
df["topic"] = topics

n_topics = topic_model.get_topic_info().shape[0] - 1
print(f"\nTopics found: {n_topics} (excl. outliers)")
print(topic_model.get_topic_info().head(20).to_string())

# %% Topic labels
topic_info = topic_model.get_topic_info()
topic_info = topic_info[topic_info["Topic"] != -1]
topic_labels = {
    row["Topic"]: " / ".join([w for w, _ in topic_model.get_topic(row["Topic"])[:3]])
    for _, row in topic_info.iterrows()
}

# %% ── Plot 1: Stacked bar ─────────────────────────────────────────────────────
topic_grp = (
    df[df["topic"] != -1]
    .groupby(["topic", "group"]).size()
    .unstack(fill_value=0)
)
topic_grp["_total"] = topic_grp.sum(axis=1)
topic_grp = topic_grp.sort_values("_total", ascending=False).head(30).drop(columns="_total")
index_labels = [f"T{t}: {topic_labels.get(t, '')}" for t in topic_grp.index]

fig, ax = plt.subplots(figsize=(14, 8))
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
ax.set_title("Sacred Texts vs Sports News — Topic Composition (top 30)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("17d_topic_composition.png", dpi=150)
plt.show()
print("Saved: 17d_topic_composition.png")

# %% ── Plot 2: Heatmap ─────────────────────────────────────────────────────────
top_topics = (
    df[df["topic"] != -1].groupby("topic").size()
    .sort_values(ascending=False).head(25).index
)

heatmap_df = (
    df[df["topic"].isin(top_topics)]
    .groupby(["group", "topic"]).size()
    .unstack(fill_value=0)
)
heatmap_norm = heatmap_df.div(heatmap_df.sum(axis=1), axis=0)
heatmap_norm.columns = [f"T{t}: {topic_labels.get(t, '')}" for t in heatmap_norm.columns]

fig, ax = plt.subplots(figsize=(16, 3))
im = ax.imshow(heatmap_norm.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(heatmap_norm.columns)))
ax.set_xticklabels(heatmap_norm.columns, rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(len(heatmap_norm.index)))
ax.set_yticklabels(heatmap_norm.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Fraction of group's passages")
ax.set_title("Sacred Texts vs Sports News — Topic Distribution (normalised)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("17d_topic_heatmap.png", dpi=150)
plt.show()
print("Saved: 17d_topic_heatmap.png")

# %% ── Print: shared topics (the interesting ones) ───────────────────────────
print("\n=== Shared topics (both groups >20%) ===")
for topic in top_topics:
    shares = df[df["topic"] == topic].groupby("group").size()
    shares = shares / shares.sum()
    if len(shares) == 2 and shares.min() > 0.20:
        words = " / ".join([w for w, _ in topic_model.get_topic(topic)[:5]])
        total = df[df["topic"] == topic].shape[0]
        print(f"\n  T{topic} [{words}] ({total} passages)")
        for g, s in shares.sort_values(ascending=False).items():
            print(f"    {g:<15} {s:.1%}")
        # Show a sample passage from each group
        for grp in ["Sacred Texts", "Sports News"]:
            sample = df[(df["topic"] == topic) & (df["group"] == grp)]["text"].iloc[0]
            print(f"    Sample ({grp}): {sample[:120]}...")

print("\n=== Group-pure topics (>90% one group) ===")
for topic in top_topics:
    shares = df[df["topic"] == topic].groupby("group").size()
    shares = shares / shares.sum()
    dominant = shares.idxmax()
    if shares[dominant] > 0.90:
        words = " / ".join([w for w, _ in topic_model.get_topic(topic)[:5]])
        total = df[df["topic"] == topic].shape[0]
        print(f"  T{topic} [{words}] → {dominant} ({shares[dominant]:.1%}, {total} passages)")
