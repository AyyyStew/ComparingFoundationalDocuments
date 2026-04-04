# %% [markdown]
# # BERTopic — Sacred Texts Only (Option B)
# Run BERTopic on sacred texts only. Since all passages are "Sacred Texts",
# color by individual tradition to see whether traditions share topics
# or each occupies its own semantic territory.

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

TRADITION_COLORS = {
    "Abrahamic": "#e05c5c",
    "Dharmic":   "#f5a623",
    "Buddhist":  "#f5d623",
    "Taoist":    "#7ed321",
}

# %% Load
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
df = df[~df["corpus"].isin(SKIP_CORPORA)]
df = df[df["tradition"].isin(SACRED_TRADITIONS)].reset_index(drop=True)

embeddings = np.stack(df["vector"].apply(np.array).values).astype("float32")
docs = df["text"].tolist()

print(f"Loaded {len(df):,} passages")
print(df.groupby(["tradition", "corpus"]).size().to_string())

# %% Fit BERTopic
topic_model = BERTopic(
    umap_model=UMAP(n_components=5, n_neighbors=15, min_dist=0.0,
                    metric="cosine", random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=60, min_samples=10,
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

# %% ── Plot 1: Stacked bar — tradition breakdown per topic ────────────────────
topic_trad = (
    df[df["topic"] != -1]
    .groupby(["topic", "tradition"]).size()
    .unstack(fill_value=0)
)
topic_trad["_total"] = topic_trad.sum(axis=1)
topic_trad = topic_trad.sort_values("_total", ascending=False).head(30).drop(columns="_total")
index_labels = [f"T{t}: {topic_labels.get(t, '')}" for t in topic_trad.index]

fig, ax = plt.subplots(figsize=(14, 8))
bottom = np.zeros(len(topic_trad))
for trad, color in TRADITION_COLORS.items():
    if trad not in topic_trad.columns:
        continue
    vals = topic_trad[trad].values
    ax.bar(range(len(topic_trad)), vals, bottom=bottom, color=color, label=trad)
    bottom += vals

ax.set_xticks(range(len(topic_trad)))
ax.set_xticklabels(index_labels, rotation=45, ha="right", fontsize=7.5)
ax.set_ylabel("Passages")
ax.set_title("Sacred Texts — Topic Composition by Tradition (top 30)", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.savefig("17b_topic_composition.png", dpi=150)
plt.show()
print("Saved: 17b_topic_composition.png")

# %% ── Plot 2: Heatmap — tradition × topic (normalised) ───────────────────────
top_topics = (
    df[df["topic"] != -1].groupby("topic").size()
    .sort_values(ascending=False).head(25).index
)

heatmap_df = (
    df[df["topic"].isin(top_topics)]
    .groupby(["tradition", "topic"]).size()
    .unstack(fill_value=0)
)
heatmap_norm = heatmap_df.div(heatmap_df.sum(axis=1), axis=0)
heatmap_norm.columns = [f"T{t}: {topic_labels.get(t, '')}" for t in heatmap_norm.columns]

fig, ax = plt.subplots(figsize=(16, 4))
im = ax.imshow(heatmap_norm.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(heatmap_norm.columns)))
ax.set_xticklabels(heatmap_norm.columns, rotation=45, ha="right", fontsize=7.5)
ax.set_yticks(range(len(heatmap_norm.index)))
ax.set_yticklabels(heatmap_norm.index, fontsize=9)
plt.colorbar(im, ax=ax, label="Fraction of tradition's passages")
ax.set_title("Sacred Texts — Topic Distribution by Tradition (normalised)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("17b_topic_heatmap.png", dpi=150)
plt.show()
print("Saved: 17b_topic_heatmap.png")

# %% ── Print: shared vs pure topics ──────────────────────────────────────────
print("\n=== Cross-tradition topics (3+ traditions with >10%) ===")
for topic in top_topics:
    shares = df[df["topic"] == topic].groupby("tradition").size()
    shares = shares / shares.sum()
    if (shares > 0.10).sum() >= 3:
        words = " / ".join([w for w, _ in topic_model.get_topic(topic)[:5]])
        print(f"\n  T{topic} [{words}]")
        for t, s in shares.sort_values(ascending=False).items():
            print(f"    {t:<12} {s:.1%}")

print("\n=== Tradition-pure topics (>80% one tradition) ===")
for topic in top_topics:
    shares = df[df["topic"] == topic].groupby("tradition").size()
    shares = shares / shares.sum()
    dominant = shares.idxmax()
    if shares[dominant] > 0.80:
        words = " / ".join([w for w, _ in topic_model.get_topic(topic)[:5]])
        total = df[df["topic"] == topic].shape[0]
        print(f"  T{topic} [{words}] → {dominant} ({shares[dominant]:.1%}, {total} passages)")
