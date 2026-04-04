# %% [markdown]
# # KJV Bible — Book-level Embedding Comparison
# Loads passages + embeddings from DuckDB, aggregates to book level,
# then projects with UMAP and clusters with HDBSCAN.

# %% Imports
import numpy as np
import pandas as pd
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from db.schema import get_conn

CORPUS_NAME      = "Bible — KJV (King James Version)"
MODEL_NAME       = "all-mpnet-base-v2"
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST    = 0.1
RANDOM_STATE     = 42

# %% Load passages + embeddings from DuckDB
conn = get_conn()

rows = conn.execute(
    """
    SELECT p.book, p.section, p.unit_number, p.text,
           e.vector
    FROM passage p
    JOIN embedding e ON e.passage_id = p.id
    JOIN corpus c    ON c.id = p.corpus_id
    WHERE c.name      = ?
      AND e.model_name = ?
    ORDER BY p.id
    """,
    [CORPUS_NAME, MODEL_NAME],
).fetchall()

conn.close()

df = pd.DataFrame(rows, columns=["book", "chapter", "verse", "text", "vector"])
verse_embeddings = np.vstack(df["vector"].values).astype(np.float32)

print(f"Loaded {len(df):,} verses across {df['book'].nunique()} books")
print(f"Embedding matrix: {verse_embeddings.shape}")

# %% Aggregate to book level (mean pooling)
book_order = df["book"].drop_duplicates().reset_index(drop=True)
books = (
    df.groupby("book", sort=False)
    .apply(lambda g: verse_embeddings[g.index].mean(axis=0), include_groups=False)
    .reset_index()
    .rename(columns={0: "embedding"})
)
books = book_order.to_frame().merge(books, on="book")
books["book_index"] = np.arange(len(books))  # 0 = Genesis, 65 = Revelation

book_embeddings = np.vstack(books["embedding"].values)
print(f"Book embedding matrix: {book_embeddings.shape}")

# %% UMAP on book-level embeddings
reducer = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=2,
    metric="cosine",
    random_state=RANDOM_STATE,
)
book_2d = reducer.fit_transform(book_embeddings)

# %% HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=3, metric="euclidean")
cluster_labels = clusterer.fit_predict(book_2d)

books["x"]       = book_2d[:, 0]
books["y"]       = book_2d[:, 1]
books["cluster"] = cluster_labels

print(books[["book", "book_index", "cluster"]].to_string(index=False))

# %% Plot 1 — book level with canonical order shown as color gradient + arrows
fig, ax = plt.subplots(figsize=(15, 10))

# Color each point by its canonical position (Genesis=dark, Revelation=bright)
cmap      = cm.plasma
norm      = Normalize(vmin=0, vmax=len(books) - 1)
sm        = cm.ScalarMappable(cmap=cmap, norm=norm)

# Draw arrows connecting consecutive books to show the journey through the canon
xs = books["x"].values
ys = books["y"].values
for i in range(len(books) - 1):
    ax.annotate(
        "",
        xy=(xs[i + 1], ys[i + 1]),
        xytext=(xs[i], ys[i]),
        arrowprops=dict(
            arrowstyle="-|>",
            color="grey",
            lw=0.6,
            alpha=0.4,
            mutation_scale=8,
        ),
    )

# Draw points colored by canonical index
for _, row in books.iterrows():
    color = cmap(norm(row["book_index"]))
    ax.scatter(row["x"], row["y"], color=color, s=90, zorder=4, edgecolors="white", linewidths=0.5)
    ax.annotate(
        f"{int(row['book_index']) + 1}. {row['book']}",
        (row["x"], row["y"]),
        fontsize=6.5,
        ha="center",
        va="bottom",
        xytext=(0, 5),
        textcoords="offset points",
        zorder=5,
    )

cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
cbar.set_label("Canonical order (Genesis → Revelation)", fontsize=9)
cbar.set_ticks([0, 38, 65])
cbar.set_ticklabels(["Genesis (1)", "Malachi / Matthew (39)", "Revelation (66)"])

ax.set_title("KJV Books — UMAP with Canonical Order (all-mpnet-base-v2)", fontsize=13)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig("02_kjv_book_embeddings_umap.png", dpi=150)
plt.show()
print("Saved 02_kjv_book_embeddings_umap.png")

# %% UMAP on all verses
print("Running verse-level UMAP...")
verse_reducer = umap.UMAP(
    n_neighbors=30,
    min_dist=0.05,
    n_components=2,
    metric="cosine",
    random_state=RANDOM_STATE,
)
verse_2d = verse_reducer.fit_transform(verse_embeddings)

# Attach 2D coords back to df for reuse
df["ux"] = verse_2d[:, 0]
df["uy"] = verse_2d[:, 1]

# %% Plot 2a — verse cloud + book centroids overlaid as large labeled markers
unique_books   = df["book"].drop_duplicates().tolist()
n_books        = len(unique_books)
book_colors    = cm.nipy_spectral(np.linspace(0, 1, n_books))
book_color_map = {b: book_colors[i] for i, b in enumerate(unique_books)}

# Per-book centroids in 2D UMAP space
centroids = df.groupby("book", sort=False)[["ux", "uy"]].mean().reindex(unique_books)

fig, ax = plt.subplots(figsize=(16, 12))

# Verse scatter (dim)
for book, grp in df.groupby("book", sort=False):
    ax.scatter(
        grp["ux"], grp["uy"],
        color=book_color_map[book],
        s=1, alpha=0.15,
    )

# Centroid markers + labels (bright, on top)
for book in unique_books:
    cx, cy = centroids.loc[book, ["ux", "uy"]]
    ax.scatter(cx, cy, color=book_color_map[book], s=60, zorder=5,
               edgecolors="white", linewidths=0.8)
    ax.annotate(
        book, (cx, cy),
        fontsize=6, fontweight="bold",
        ha="center", va="bottom",
        xytext=(0, 5), textcoords="offset points",
        zorder=6,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
    )

ax.set_title("KJV Verses — UMAP with Book Centroids (all-mpnet-base-v2)", fontsize=13)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig("02_kjv_verse_embeddings_umap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved 02_kjv_verse_embeddings_umap.png")

# %% Plot 2b — density heatmap of all verses (testament split)
# Shows where the semantic mass actually sits, regardless of book boundaries.
# OT vs NT split lets you see how much the two traditions overlap.
OT_BOOKS = set(unique_books[:39])

fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
x_all, y_all = df["ux"].values, df["uy"].values

# Shared grid
margin = 0.5
x_range = (x_all.min() - margin, x_all.max() + margin)
y_range = (y_all.min() - margin, y_all.max() + margin)
grid_size = 300

for ax, (label, mask) in zip(
    axes,
    [("Old Testament", df["book"].isin(OT_BOOKS)), ("New Testament", ~df["book"].isin(OT_BOOKS))],
):
    sub = df[mask]
    heatmap, xedges, yedges = np.histogram2d(
        sub["ux"], sub["uy"],
        bins=grid_size,
        range=[x_range, y_range],
    )
    heatmap = gaussian_filter(heatmap.T, sigma=3)
    ax.imshow(
        heatmap,
        origin="lower",
        extent=[*x_range, *y_range],
        cmap="inferno",
        aspect="auto",
    )
    ax.set_title(f"{label} — Verse Density", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

plt.suptitle("KJV Semantic Density — Old vs New Testament", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("02_kjv_testament_density.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved 02_kjv_testament_density.png")
