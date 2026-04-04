# %% [markdown]
# # GMM Soft Topic Analysis — Sacred Texts Only (K=20)
#
# Same pipeline as 21b but restricted to sacred traditions.
# Colored by individual tradition (Abrahamic / Dharmic / Buddhist / Taoist)
# so cross-tradition mixing within the sacred space is visible.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
from umap import UMAP

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA

MODEL_NAME = "all-mpnet-base-v2"
UMAP_DIMS  = 15
K          = 20
N_INIT     = 5

TRADITION_COLORS = {
    "Abrahamic": "#e05c5c",
    "Dharmic":   "#f0a500",
    "Buddhist":  "#44bb99",
    "Taoist":    "#9b59b6",
}

# %% Load — sacred traditions only
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
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} sacred passages")
print(df.groupby(["tradition", "corpus"]).size().to_string())

traditions = list(TRADITION_COLORS.keys())

# %% UMAP 15D
vecs = np.stack(df["vector"].values).astype("float32")

print(f"\nFitting UMAP ({UMAP_DIMS}D) ...")
reducer = UMAP(
    n_components=UMAP_DIMS,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
X = reducer.fit_transform(vecs).astype("float32")

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

probs       = gmm.predict_proba(X)       # (n_passages, K)
hard_topics = probs.argmax(axis=1)
df["topic"]      = hard_topics
df["topic_prob"] = probs.max(axis=1)

print(f"\nGMM converged: {gmm.converged_}  |  BIC: {gmm.bic(X):,.0f}")

# %% Tradition composition per topic (soft weights)
trad_weights = {}
for t in traditions:
    mask = (df["tradition"] == t).values
    trad_weights[t] = probs[mask].sum(axis=0)   # (K,)

comp_df   = pd.DataFrame(trad_weights, index=range(K))
comp_frac = comp_df.div(comp_df.sum(axis=1), axis=0)

def entropy(row):
    p = row[row > 0]
    return float(-np.sum(p * np.log(p)))

comp_frac["entropy"] = comp_frac[traditions].apply(entropy, axis=1)
comp_frac = comp_frac.sort_values("entropy", ascending=False)

print("\nTopic entropy (higher = more cross-tradition):")
print(comp_frac["entropy"].round(3).to_string())

# %% ── Plot 1: Stacked bar sorted by entropy ──────────────────────────────────
topic_order = comp_frac.index.tolist()

fig, ax = plt.subplots(figsize=(16, 5))
bottom = np.zeros(K)

for t in traditions:
    vals  = comp_frac.loc[topic_order, t].values
    color = TRADITION_COLORS[t]
    ax.bar(range(K), vals, bottom=bottom, color=color, label=t, width=0.8)
    bottom += vals

ax.set_xticks(range(K))
ax.set_xticklabels([f"T{t}" for t in topic_order], fontsize=8)
ax.set_ylabel("Fraction of topic weight")
ax.set_xlabel("← more cross-tradition          Topics (sorted by entropy)          more single-tradition →")
ax.set_title(f"GMM Sacred Topic Composition by Tradition (K={K}, sorted by entropy)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 1)

for i, t in enumerate(topic_order):
    e = comp_frac.loc[t, "entropy"]
    ax.text(i, 1.01, f"{e:.2f}", ha="center", va="bottom", fontsize=6.5, color="gray")

plt.tight_layout()
plt.savefig("22b_topic_composition.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 22b_topic_composition.png")

# %% ── Plot 2: 2D UMAP coloured by dominant GMM topic ────────────────────────
print("\nFitting 2D UMAP for visualisation ...")
reducer_2d = UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
X2 = reducer_2d.fit_transform(vecs).astype("float32")

cmap = plt.get_cmap("tab20", K)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Left — coloured by GMM topic
axes[0].scatter(X2[:, 0], X2[:, 1], c=hard_topics, cmap="tab20",
                s=2, alpha=0.4, linewidths=0, vmin=0, vmax=K-1)
patches = [mpatches.Patch(color=cmap(t), label=f"T{t}") for t in range(K)]
axes[0].legend(handles=patches, fontsize=7, ncol=4, loc="lower right", markerscale=1.5)
axes[0].set_title(f"Coloured by GMM topic (K={K})", fontsize=11, fontweight="bold")
axes[0].set_xticks([]); axes[0].set_yticks([])

# Right — coloured by tradition (reference)
for t in traditions:
    mask = df["tradition"] == t
    axes[1].scatter(X2[mask, 0], X2[mask, 1],
                    c=TRADITION_COLORS[t], s=2, alpha=0.4,
                    linewidths=0, label=t)
axes[1].legend(fontsize=9, markerscale=3)
axes[1].set_title("Coloured by tradition (reference)", fontsize=11, fontweight="bold")
axes[1].set_xticks([]); axes[1].set_yticks([])

plt.suptitle("Sacred Texts — GMM Topics vs Tradition Structure",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("22b_umap_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 22b_umap_comparison.png")

# %% ── Printed report ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TOP 10 MOST CROSS-TRADITION TOPICS (by entropy)")
print("="*70)

for rank, topic in enumerate(topic_order[:10], 1):
    frac_row = comp_frac.loc[topic]
    ent      = frac_row["entropy"]

    topic_probs  = probs[:, topic]
    df_tmp       = df.copy()
    df_tmp["_p"] = topic_probs

    print(f"\n{'─'*60}")
    print(f"  Rank {rank} | Topic {topic} | Entropy {ent:.3f}")
    parts = [f"{t}: {frac_row[t]:.1%}" for t in traditions if frac_row[t] > 0.01]
    print(f"  Composition: {' | '.join(parts)}")

    # Best exemplar per tradition
    for t in traditions:
        sub = df_tmp[df_tmp["tradition"] == t].nlargest(1, "_p")
        if sub.empty:
            continue
        row = sub.iloc[0]
        print(f"\n  [{t}] {row['corpus']} | {row['unit_label']}  (p={row['_p']:.3f})")
        print(f"    {row['text'][:200]}")

print("\n" + "="*70)
print("TOP 5 MOST SINGLE-TRADITION TOPICS (lowest entropy)")
print("="*70)

for topic in topic_order[-5:]:
    frac_row = comp_frac.loc[topic]
    ent      = frac_row["entropy"]
    dominant = frac_row[traditions].idxmax()
    dom_pct  = frac_row[dominant]

    topic_probs  = probs[:, topic]
    df_tmp       = df.copy()
    df_tmp["_p"] = topic_probs

    print(f"\n  Topic {topic} | Entropy {ent:.3f} | Dominant: {dominant} ({dom_pct:.1%})")
    top3 = df_tmp.nlargest(3, "_p")[["corpus", "unit_label", "text", "_p"]]
    for _, row in top3.iterrows():
        print(f"  [{row['_p']:.3f}] {row['corpus']} | {row['unit_label']}")
        print(f"    {row['text'][:150]}")
