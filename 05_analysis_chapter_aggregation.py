# %% [markdown]
# # Chapter-level Aggregation Comparison
# Aggregates verse embeddings to chapter level using 6 methods,
# runs UMAP on each, and plots them side by side for comparison.
#
# Methods: mean, max, weighted mean (by verse length),
#          first+last, TF-IDF weighted, medoid

# %% Imports
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer

from db.schema import get_conn

CORPUS_NAME = "Bible — KJV (King James Version)"
MODEL_NAME  = "all-mpnet-base-v2"
RANDOM_STATE = 42

UMAP_PARAMS = dict(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=RANDOM_STATE,
)

# %% Load from DuckDB
conn = get_conn()
rows = conn.execute(
    """
    SELECT p.book, p.section::INTEGER AS chapter, p.unit_number AS verse,
           p.text, e.vector
    FROM passage p
    JOIN embedding e ON e.passage_id = p.id
    JOIN corpus c    ON c.id = p.corpus_id
    WHERE c.name       = ?
      AND e.model_name = ?
    ORDER BY p.id
    """,
    [CORPUS_NAME, MODEL_NAME],
).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["book", "chapter", "verse", "text", "vector"])
df["chapter_key"] = df["book"] + "|" + df["chapter"].astype(str)

# Canonical chapter order
chapter_meta = df[["book", "chapter", "chapter_key"]].drop_duplicates("chapter_key")
book_order   = df["book"].drop_duplicates().tolist()
chapter_meta["book_idx"] = chapter_meta["book"].map({b: i for i, b in enumerate(book_order)})
chapter_meta = chapter_meta.sort_values(["book_idx", "chapter"]).reset_index(drop=True)

print(f"Loaded {len(df):,} verses → {len(chapter_meta):,} chapters")

# %% Aggregation functions
# Each takes a group DataFrame + its embedding rows and returns a 1D vector.

def agg_mean(embs: np.ndarray, texts: list[str]) -> np.ndarray:
    return embs.mean(axis=0)


def agg_max(embs: np.ndarray, texts: list[str]) -> np.ndarray:
    return embs.max(axis=0)


def agg_weighted_mean(embs: np.ndarray, texts: list[str]) -> np.ndarray:
    weights = np.array([len(t.split()) for t in texts], dtype=np.float32)
    weights /= weights.sum()
    return (embs * weights[:, None]).sum(axis=0)


def agg_first_last(embs: np.ndarray, texts: list[str]) -> np.ndarray:
    if len(embs) == 1:
        return embs[0]
    return (embs[0] + embs[-1]) / 2.0


def agg_medoid(embs: np.ndarray, texts: list[str]) -> np.ndarray:
    if len(embs) == 1:
        return embs[0]
    dists = pairwise_distances(embs, metric="cosine")
    return embs[dists.sum(axis=1).argmin()]


# TF-IDF weights built once over the full corpus
print("Building TF-IDF weights...")
tfidf = TfidfVectorizer(max_features=50_000)
tfidf.fit(df["text"].tolist())

def agg_tfidf(embs: np.ndarray, texts: list[str]) -> np.ndarray:
    tfidf_matrix = tfidf.transform(texts).toarray()  # (n_verses, vocab)
    weights = tfidf_matrix.sum(axis=1).astype(np.float32)
    if weights.sum() == 0:
        return embs.mean(axis=0)
    weights /= weights.sum()
    return (embs * weights[:, None]).sum(axis=0)


METHODS = {
    "Mean":          agg_mean,
    "Max":           agg_max,
    "Weighted Mean": agg_weighted_mean,
    "First + Last":  agg_first_last,
    "TF-IDF":        agg_tfidf,
    "Medoid":        agg_medoid,
}

# %% Compute chapter embeddings for each method
print("Aggregating chapters...")

chapter_embs = {}   # method -> (n_chapters, 768)

for method_name, fn in METHODS.items():
    vecs = []
    for ck in chapter_meta["chapter_key"]:
        mask  = df["chapter_key"].values == ck
        embs  = np.vstack(df.loc[mask, "vector"].values).astype(np.float32)
        texts = df.loc[mask, "text"].tolist()
        vecs.append(fn(embs, texts))
    chapter_embs[method_name] = np.vstack(vecs).astype(np.float32)
    print(f"  {method_name}: {chapter_embs[method_name].shape}")

# %% UMAP — fit on Mean, project all others into the same space
REFERENCE_METHOD = "Mean"

print(f"Fitting UMAP on '{REFERENCE_METHOD}'...")
reducer = umap.UMAP(**UMAP_PARAMS)
chapter_2d = {}
chapter_2d[REFERENCE_METHOD] = reducer.fit_transform(chapter_embs[REFERENCE_METHOD])

print("Projecting all other methods into the same space...")
for method_name, X in chapter_embs.items():
    if method_name == REFERENCE_METHOD:
        continue
    print(f"  transform: {method_name}...")
    chapter_2d[method_name] = reducer.transform(X)

# %% Plot — 2×3 grid, one panel per method, colored by book
n_books     = len(book_order)
book_colors = cm.nipy_spectral(np.linspace(0, 1, n_books))
book_cmap   = {b: book_colors[i] for i, b in enumerate(book_order)}
point_colors = chapter_meta["book"].map(book_cmap).tolist()

fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes = axes.flatten()

for ax, method_name in zip(axes, METHODS.keys()):
    coords = chapter_2d[method_name]
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=point_colors,
        s=4,
        alpha=0.6,
        linewidths=0,
    )
    title = f"{method_name} (reference space)" if method_name == REFERENCE_METHOD else method_name
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.15)

# Shared legend (books) — one entry per book, tiny markers
handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=book_colors[i],
               markersize=4, label=b)
    for i, b in enumerate(book_order)
]
fig.legend(
    handles=handles,
    loc="lower center",
    ncol=11,
    fontsize=5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
)

fig.suptitle(
    f"Chapter-level UMAP — 6 aggregation methods\n{CORPUS_NAME}",
    fontsize=14, y=1.01,
)
plt.tight_layout()
plt.savefig("05_chapter_aggregation_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved 05_chapter_aggregation_comparison.png")
