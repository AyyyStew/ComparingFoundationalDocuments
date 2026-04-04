# %% [markdown]
# # Sacred Cross-Tradition Nearest Neighbours
#
# Script 18 ran across all groups (sacred, literature, historical, news) so the
# bridge passages were diluted by noise. This script restricts to sacred traditions
# only (Abrahamic, Dharmic, Buddhist, Taoist) and asks:
#
#   Which passages are pulled toward by 2+ *other* sacred traditions?
#
# Those are the genuine shared-theme passages — the semantic overlap between
# the world's religious texts.
#
# Pipeline:
#   1. Load sacred embeddings only
#   2. For each tradition-pair, compute top-k cross-tradition nearest neighbours
#   3. Bridge passages = those matched by 2+ distinct other traditions
#   4. Print top pairs by similarity + bridge leaderboard
#   5. BERTopic on bridge pool — topics here are genuinely cross-sacred

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

from db.schema import get_conn
from analysis_utils import make_vectorizer, SKIP_CORPORA

MODEL_NAME = "all-mpnet-base-v2"
TOP_K      = 5   # neighbours per passage per opposing tradition
MIN_HITS   = 2   # bridge threshold: pulled by this many distinct other traditions
                 # (only 4 traditions total so 2 is already meaningful)

TRADITION_COLORS = {
    "Abrahamic": "#e05c5c",
    "Dharmic":   "#f0a500",
    "Buddhist":  "#44bb99",
    "Taoist":    "#9b59b6",
}

# %% Load — sacred only
conn = get_conn()
rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus,
           p.id AS passage_id, p.unit_label, p.text, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    AND t.name IN ('Abrahamic', 'Dharmic', 'Buddhist', 'Taoist')
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "passage_id", "unit_label", "text", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)

vecs  = np.stack(df["vector"].apply(np.array).values).astype("float32")
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_normed = vecs / np.clip(norms, 1e-9, None)

print(f"Loaded {len(df):,} sacred passages")
print(df.groupby(["tradition", "corpus"]).size().to_string())

# %% Cross-tradition nearest-neighbour search
traditions  = list(TRADITION_COLORS.keys())
trad_idx    = {t: df.index[df["tradition"] == t].tolist() for t in traditions}

records = []

for i, t_a in enumerate(traditions):
    for t_b in traditions[i+1:]:
        idx_a = trad_idx[t_a]
        idx_b = trad_idx[t_b]

        V_a = vecs_normed[idx_a]
        V_b = vecs_normed[idx_b]
        sim = V_a @ V_b.T   # cosine similarity (n_a, n_b)

        k_ab = min(TOP_K, sim.shape[1])
        k_ba = min(TOP_K, sim.shape[0])

        # Top-k in B for each passage in A
        top_cols = np.argpartition(sim, -k_ab, axis=1)[:, -k_ab:]
        for row_i, cols in enumerate(top_cols):
            for col_j in cols:
                records.append((
                    df.loc[idx_a[row_i], "passage_id"],
                    df.loc[idx_b[col_j], "passage_id"],
                    t_a, t_b,
                    float(sim[row_i, col_j]),
                ))

        # Top-k in A for each passage in B
        top_rows = np.argpartition(sim, -k_ba, axis=0)[-k_ba:, :]
        for col_j, rows_ in enumerate(top_rows.T):
            for row_i in rows_:
                records.append((
                    df.loc[idx_b[col_j], "passage_id"],
                    df.loc[idx_a[row_i], "passage_id"],
                    t_b, t_a,
                    float(sim[row_i, col_j]),
                ))

        print(f"  {t_a} × {t_b}: {len(idx_a):,} × {len(idx_b):,} → done")

pairs_df = pd.DataFrame(records, columns=[
    "query_id", "match_id", "query_trad", "match_trad", "similarity"
]).drop_duplicates()

print(f"\nTotal cross-sacred pairs: {len(pairs_df):,}")
print(pairs_df.groupby(["query_trad", "match_trad"])["similarity"].mean().round(3).unstack().to_string())

# %% Bridge passages
match_counts = (
    pairs_df.groupby("match_id")["query_trad"]
    .nunique()
    .rename("n_traditions_matched")
    .reset_index()
    .rename(columns={"match_id": "passage_id"})
)

df = df.merge(match_counts, on="passage_id", how="left")
df["n_traditions_matched"] = df["n_traditions_matched"].fillna(0).astype(int)

bridge_df = df[df["n_traditions_matched"] >= MIN_HITS].copy()

print(f"\nBridge passages (pulled by {MIN_HITS}+ other traditions): {len(bridge_df):,}")
print(bridge_df.groupby("tradition").size().to_string())

# %% ── Plot 1: Cross-tradition similarity heatmap ─────────────────────────────
pivot = (
    pairs_df.groupby(["query_trad", "match_trad"])["similarity"]
    .mean().reset_index()
)
pivot_rev = pivot.rename(columns={"query_trad": "match_trad", "match_trad": "query_trad"})
pivot_sym = (
    pd.concat([pivot, pivot_rev])
    .groupby(["query_trad", "match_trad"])["similarity"]
    .mean().unstack()
)
# Fill diagonal with within-tradition top-k similarity — same method as cross-tradition
# so the diagonal is directly comparable to the off-diagonal values.
for t in traditions:
    idx_t = trad_idx[t]
    V_t   = vecs_normed[idx_t]
    sim_t = V_t @ V_t.T                         # (n_t, n_t)
    np.fill_diagonal(sim_t, -np.inf)             # exclude self-similarity
    k_t   = min(TOP_K, sim_t.shape[1] - 1)
    top_k_vals = np.partition(sim_t, -k_t, axis=1)[:, -k_t:]
    pivot_sym.loc[t, t] = float(top_k_vals.mean())

pivot_sym = pivot_sym.loc[traditions, traditions]

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(pivot_sym.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(traditions)))
ax.set_xticklabels(traditions, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(traditions)))
ax.set_yticklabels(traditions, fontsize=10)
for i in range(len(traditions)):
    for j in range(len(traditions)):
        val = pivot_sym.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if i == j else "normal")
plt.colorbar(im, ax=ax, label="Mean cosine similarity")
ax.set_title("Sacred Tradition Similarity\n(diagonal = within-tradition)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("23_tradition_similarity_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 23_tradition_similarity_heatmap.png")

# %% ── Print: top 20 cross-sacred pairs ──────────────────────────────────────
id_to_row = df.set_index("passage_id")

top_pairs = (
    pairs_df.sort_values("similarity", ascending=False)
    .drop_duplicates(subset=["query_id", "match_id"])
    .head(20)
)

print("\n=== Top 20 cross-sacred pairs by similarity ===")
for _, row in top_pairs.iterrows():
    q = id_to_row.loc[row["query_id"]]
    m = id_to_row.loc[row["match_id"]]
    print(f"\n  [{row['similarity']:.3f}]  {q['tradition']} | {q['corpus']} | {q['unit_label']}")
    print(f"    {q['text'][:150]}")
    print(f"  ↔ {m['tradition']} | {m['corpus']} | {m['unit_label']}")
    print(f"    {m['text'][:150]}")

# %% ── Print: bridge leaderboard ─────────────────────────────────────────────
print(f"\n=== Top 20 most universally resonant sacred passages ===")
top_bridges = (
    df[df["n_traditions_matched"] > 0]
    .sort_values("n_traditions_matched", ascending=False)
    .head(20)
)
for _, row in top_bridges.iterrows():
    print(f"\n  [{row['n_traditions_matched']} traditions] {row['tradition']} | {row['corpus']} | {row['unit_label']}")
    print(f"    {row['text'][:180]}")

# %% ── BERTopic on bridge passages ────────────────────────────────────────────
if len(bridge_df) < 50:
    print(f"\nOnly {len(bridge_df)} bridge passages — skipping BERTopic (need 50+)")
else:
    bridge_vecs = np.stack(bridge_df["vector"].apply(np.array).values).astype("float32")
    bridge_docs = bridge_df["text"].tolist()

    topic_model = BERTopic(
        umap_model=UMAP(n_components=5, n_neighbors=min(15, len(bridge_df) - 1),
                        min_dist=0.0, metric="cosine", random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=max(5, len(bridge_df) // 40),
                              min_samples=3, metric="euclidean", prediction_data=True),
        vectorizer_model=make_vectorizer(),
        nr_topics="auto",
        verbose=True,
    )

    bridge_topics, _ = topic_model.fit_transform(bridge_docs, embeddings=bridge_vecs)
    bridge_df = bridge_df.copy()
    bridge_df["topic"] = bridge_topics

    n_topics = topic_model.get_topic_info().shape[0] - 1
    print(f"\nBridge BERTopic — {n_topics} topics across {len(bridge_df):,} passages")

    topic_info   = topic_model.get_topic_info()
    topic_labels = {
        row["Topic"]: " / ".join([w for w, _ in topic_model.get_topic(row["Topic"])[:4]])
        for _, row in topic_info[topic_info["Topic"] != -1].iterrows()
    }

    # Stacked bar — tradition breakdown per bridge topic
    topic_grp = (
        bridge_df[bridge_df["topic"] != -1]
        .groupby(["topic", "tradition"]).size()
        .unstack(fill_value=0)
    )
    topic_grp["_total"] = topic_grp.sum(axis=1)
    topic_grp = topic_grp.sort_values("_total", ascending=False).head(25).drop(columns="_total")
    index_labels = [f"T{t}: {topic_labels.get(t, '')}" for t in topic_grp.index]

    fig, ax = plt.subplots(figsize=(14, 7))
    bottom = np.zeros(len(topic_grp))
    for trad, color in TRADITION_COLORS.items():
        if trad not in topic_grp.columns:
            continue
        vals = topic_grp[trad].values
        ax.bar(range(len(topic_grp)), vals, bottom=bottom, color=color, label=trad)
        bottom += vals

    ax.set_xticks(range(len(topic_grp)))
    ax.set_xticklabels(index_labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel("Passages")
    ax.set_title(f"Bridge Topic Composition — Sacred Traditions Only (top 25)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("23_bridge_topics.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: 23_bridge_topics.png")

    # Print bridge topics with sample passages per tradition
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
        trad_counts    = topic_passages["tradition"].value_counts()
        print(f"\n  Topic {topic} [{words}] — {len(topic_passages)} passages")
        for trad, cnt in trad_counts.items():
            sample = topic_passages[topic_passages["tradition"] == trad].iloc[0]
            print(f"    {trad} ({cnt}): [{sample['corpus']} | {sample['unit_label']}]")
            print(f"      {sample['text'][:150]}")
