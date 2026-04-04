# %% [markdown]
# # GMM Soft Topic Analysis (K=20)
#
# Fit a 20-component Gaussian Mixture Model on UMAP-reduced embeddings.
# Unlike BERTopic (hard clustering), every passage gets a soft probability
# distribution over all 20 topics — revealing cross-tradition themes that
# hard clustering misses.
#
# Pipeline:
#   1. Load embeddings, UMAP → 15D (same as 21a)
#   2. Fit GMM(K=20)
#   3. For each topic:
#        - Tradition/group composition (weighted by P(topic|passage))
#        - Top exemplar passages (highest P(topic|passage))
#   4. Plots:
#        a. Stacked bar — tradition composition per topic, sorted by entropy
#           (most cross-tradition topics on the left)
#        b. 2D UMAP scatter coloured by dominant topic
#        c. Printed report — cross-tradition topics with exemplars from each tradition

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from umap import UMAP

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA, TRADITION_GROUP, GROUP_COLORS

MODEL_NAME = "all-mpnet-base-v2"
UMAP_DIMS  = 15
K          = 20
N_INIT     = 5

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
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} passages")
print(df.groupby("group").size().to_string())

# %% UMAP 15D — fit on non-news, transform all
vecs    = np.stack(df["vector"].values).astype("float32")
is_news = df["group"] == "News"

print(f"\nFitting UMAP ({UMAP_DIMS}D) on {(~is_news).sum():,} non-news passages ...")
reducer = UMAP(
    n_components=UMAP_DIMS,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
reducer.fit(vecs[~is_news])

print("Transforming all passages ...")
X = reducer.transform(vecs).astype("float32")

# %% Fit GMM
print(f"\nFitting GMM (K={K}, n_init={N_INIT}) ...")
gmm = GaussianMixture(
    n_components=K,
    covariance_type="full",
    n_init=N_INIT,
    random_state=42,
    verbose=1,
)
gmm.fit(X)

probs        = gmm.predict_proba(X)          # (n_passages, K)
hard_topics  = probs.argmax(axis=1)          # dominant topic per passage
df["topic"]  = hard_topics
df["topic_prob"] = probs.max(axis=1)         # confidence in dominant topic

print(f"\nGMM converged: {gmm.converged_}")
print(f"BIC: {gmm.bic(X):,.0f}")

# %% Tradition composition per topic (soft — weighted by P(topic|passage))
groups = df["group"].unique()

# For each topic k: sum of P(k | passage) across passages in each group
# Shape: (K, n_groups)
group_weights = {}
for g in groups:
    mask = (df["group"] == g).values
    group_weights[g] = probs[mask].sum(axis=0)   # sum over passages in group → (K,)

comp_df = pd.DataFrame(group_weights, index=range(K))   # rows=topics, cols=groups

# Normalise each topic row to sum to 1 (fraction of that topic's weight per group)
comp_frac = comp_df.div(comp_df.sum(axis=1), axis=0)

# Shannon entropy of tradition distribution per topic — higher = more cross-tradition
def entropy(row):
    p = row[row > 0]
    return float(-np.sum(p * np.log(p)))

comp_frac["entropy"] = comp_frac[list(groups)].apply(entropy, axis=1)
comp_frac = comp_frac.sort_values("entropy", ascending=False)

print("\nTopic entropy (higher = more cross-tradition):")
print(comp_frac["entropy"].round(3).to_string())

# %% ── Plot 1: Stacked bar — tradition composition, sorted by entropy ─────────

plot_groups = [g for g in ["Sacred Texts", "Literature", "Historical", "News"] if g in groups]
topic_order = comp_frac.index.tolist()   # sorted by entropy (descending)

fig, ax = plt.subplots(figsize=(16, 5))
bottom = np.zeros(K)

for g in plot_groups:
    vals = comp_frac.loc[topic_order, g].values
    color = GROUP_COLORS.get(g, "#aaaaaa")
    ax.bar(range(K), vals, bottom=bottom, color=color, label=g, width=0.8)
    bottom += vals

ax.set_xticks(range(K))
ax.set_xticklabels([f"T{t}" for t in topic_order], fontsize=8)
ax.set_ylabel("Fraction of topic weight")
ax.set_xlabel("← more cross-tradition          Topics (sorted by entropy)          more single-tradition →")
ax.set_title(f"GMM Topic Composition by Tradition (K={K}, sorted by entropy)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 1)

# Annotate entropy values
for i, t in enumerate(topic_order):
    e = comp_frac.loc[t, "entropy"]
    ax.text(i, 1.01, f"{e:.2f}", ha="center", va="bottom", fontsize=6, color="gray")

plt.tight_layout()
plt.savefig("21b_topic_composition.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 21b_topic_composition.png")

# %% ── Plot 2: 2D UMAP scatter coloured by dominant topic ─────────────────────

print("\nFitting 2D UMAP for visualisation ...")
reducer_2d = UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
reducer_2d.fit(vecs[~is_news])
X2 = reducer_2d.transform(vecs).astype("float32")

cmap   = plt.get_cmap("tab20", K)
colors = [cmap(t) for t in hard_topics]

fig, ax = plt.subplots(figsize=(12, 9))
ax.scatter(X2[:, 0], X2[:, 1], c=hard_topics, cmap="tab20",
           s=2, alpha=0.4, linewidths=0, vmin=0, vmax=K-1)

# Legend patches — topic number only
patches = [mpatches.Patch(color=cmap(t), label=f"T{t}") for t in range(K)]
ax.legend(handles=patches, fontsize=7, ncol=4,
          loc="lower right", markerscale=1.5)

ax.set_title(f"2D UMAP — passages coloured by dominant GMM topic (K={K})",
             fontsize=12, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("21b_umap_topics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 21b_umap_topics.png")

# %% ── Plot 3: 2D UMAP scatter coloured by group (for reference) ──────────────

group_color_arr = df["group"].map(GROUP_COLORS).fillna("#aaaaaa").values

fig, ax = plt.subplots(figsize=(12, 9))
for g in plot_groups:
    mask = df["group"] == g
    ax.scatter(X2[mask, 0], X2[mask, 1],
               c=GROUP_COLORS.get(g, "#aaaaaa"),
               s=2, alpha=0.4, linewidths=0, label=g)

ax.legend(fontsize=9, markerscale=3)
ax.set_title("2D UMAP — passages coloured by tradition group (reference)",
             fontsize=12, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig("21b_umap_groups.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 21b_umap_groups.png")

# %% ── Printed report: top topics by entropy with exemplars ───────────────────

print("\n" + "="*70)
print(f"TOP 10 MOST CROSS-TRADITION TOPICS (by entropy)")
print("="*70)

for rank, topic in enumerate(topic_order[:10], 1):
    frac_row = comp_frac.loc[topic]
    ent      = frac_row["entropy"]

    # Top 3 exemplar passages per group present in this topic
    # Use raw P(topic | passage) weight
    topic_probs = probs[:, topic]
    df_tmp = df.copy()
    df_tmp["_p"] = topic_probs

    print(f"\n{'─'*60}")
    print(f"  Rank {rank} | Topic {topic} | Entropy {ent:.3f}")
    # Composition line
    parts = [f"{g}: {frac_row[g]:.1%}" for g in plot_groups if frac_row.get(g, 0) > 0.01]
    print(f"  Composition: {' | '.join(parts)}")

    # One exemplar passage per group, highest P(topic|passage)
    for g in plot_groups:
        sub = df_tmp[df_tmp["group"] == g].nlargest(1, "_p")
        if sub.empty:
            continue
        row = sub.iloc[0]
        print(f"\n  [{g}] {row['corpus']} | {row['unit_label']}  (p={row['_p']:.3f})")
        print(f"    {row['text'][:200]}")

print("\n" + "="*70)
print(f"TOP 5 MOST SINGLE-TRADITION TOPICS (lowest entropy)")
print("="*70)

for topic in topic_order[-5:]:
    frac_row = comp_frac.loc[topic]
    ent      = frac_row["entropy"]
    dominant = frac_row[plot_groups].idxmax()
    dom_pct  = frac_row[dominant]
    print(f"\n  Topic {topic} | Entropy {ent:.3f} | Dominant: {dominant} ({dom_pct:.1%})")

    topic_probs = probs[:, topic]
    df_tmp = df.copy()
    df_tmp["_p"] = topic_probs
    top3 = df_tmp.nlargest(3, "_p")[["corpus", "unit_label", "text", "_p"]]
    for _, row in top3.iterrows():
        print(f"  [{row['_p']:.3f}] {row['corpus']} | {row['unit_label']}")
        print(f"    {row['text'][:150]}")
