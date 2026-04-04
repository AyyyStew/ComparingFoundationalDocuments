# %% [markdown]
# # KJV — PCA Exploration
# What are the main axes of semantic variation across the Bible?
# We run PCA on the book-level mean embeddings and inspect what each
# principal component captures by looking at which books load high/low
# and what words/themes dominate those books.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from db.schema import get_conn

CORPUS_NAME = "Bible — KJV (King James Version)"
MODEL_NAME  = "all-mpnet-base-v2"
N_COMPONENTS = 10  # inspect top 10 PCs

# %% Load embeddings from DuckDB
conn = get_conn()
rows = conn.execute(
    """
    SELECT p.book, p.unit_label, p.text, e.vector
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

df = pd.DataFrame(rows, columns=["book", "unit_label", "text", "vector"])
verse_embeddings = np.vstack(df["vector"].values).astype(np.float32)

# Canonical book order
book_order = df["book"].drop_duplicates().tolist()

# Book-level mean embeddings
book_embeddings = np.vstack([
    verse_embeddings[df["book"].values == b].mean(axis=0)
    for b in book_order
])
print(f"Book matrix: {book_embeddings.shape}")

# %% Fit PCA
# Standardize first so no embedding dimension dominates purely by scale
scaler = StandardScaler()
X = scaler.fit_transform(book_embeddings)

pca = PCA(n_components=N_COMPONENTS, random_state=42)
book_pcs = pca.fit_transform(X)  # (66, N_COMPONENTS)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("\nVariance explained per component:")
for i, (e, c) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1:02d}: {e*100:.1f}%  (cumulative: {c*100:.1f}%)")

# %% Plot 1 — Scree plot
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(1, N_COMPONENTS + 1), explained * 100, color="steelblue", alpha=0.8, label="Per component")
ax.plot(range(1, N_COMPONENTS + 1), cumulative * 100, "o-", color="tomato", label="Cumulative")
ax.axhline(50, color="grey", lw=0.8, ls="--", alpha=0.6)
ax.axhline(80, color="grey", lw=0.8, ls="--", alpha=0.6)
ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Explained (%)")
ax.set_title("PCA Scree Plot — KJV Book Embeddings")
ax.legend()
ax.set_xticks(range(1, N_COMPONENTS + 1))
plt.tight_layout()
plt.savefig("03_kjv_pca_scree.png", dpi=150)
plt.show()

# %% Plot 2 — PC1 vs PC2, books labeled and colored by canonical position
cmap = cm.plasma
colors = cmap(np.linspace(0, 1, len(book_order)))

fig, ax = plt.subplots(figsize=(13, 9))
for i, book in enumerate(book_order):
    ax.scatter(book_pcs[i, 0], book_pcs[i, 1], color=colors[i], s=80,
               edgecolors="white", linewidths=0.5, zorder=3)
    ax.annotate(book, (book_pcs[i, 0], book_pcs[i, 1]),
                fontsize=6.5, ha="center", va="bottom",
                xytext=(0, 4), textcoords="offset points")

ax.axhline(0, color="grey", lw=0.5, alpha=0.5)
ax.axvline(0, color="grey", lw=0.5, alpha=0.5)
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", fontsize=10)
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)", fontsize=10)
ax.set_title("KJV Books — PC1 vs PC2", fontsize=13)

sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, len(book_order)-1))
cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.02)
cbar.set_label("Canonical order (Genesis → Revelation)", fontsize=8)
cbar.set_ticks([0, 38, 65])
cbar.set_ticklabels(["Genesis", "Malachi / Matthew", "Revelation"])

plt.tight_layout()
plt.savefig("03_kjv_pca_pc1_pc2.png", dpi=150)
plt.show()

# %% Plot 3 — PC loadings across the canon (what does each PC "look like"?)
# Each row = a book's score on that PC, plotted along the canon axis.
# High score = this PC is "active" for that book.
fig, axes = plt.subplots(5, 2, figsize=(16, 14), sharex=True)
axes = axes.flatten()

for pc_idx in range(N_COMPONENTS):
    ax = axes[pc_idx]
    scores = book_pcs[:, pc_idx]
    colors_bar = ["tomato" if s > 0 else "steelblue" for s in scores]
    ax.bar(range(len(book_order)), scores, color=colors_bar, alpha=0.85)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"PC{pc_idx+1} ({explained[pc_idx]*100:.1f}%)", fontsize=9)
    ax.set_xticks(range(len(book_order)))
    ax.set_xticklabels(book_order, rotation=90, fontsize=5)
    ax.set_ylabel("Score")

plt.suptitle("PC Scores per Book — KJV (Genesis → Revelation)", fontsize=12, y=1.01)
plt.tight_layout()
plt.savefig("03_kjv_pca_loadings.png", dpi=150, bbox_inches="tight")
plt.show()

# %% Summary table — top & bottom 5 books per PC
print("\n=== Books with strongest loadings per PC ===")
for pc_idx in range(N_COMPONENTS):
    scores = pd.Series(book_pcs[:, pc_idx], index=book_order)
    top    = scores.nlargest(5).index.tolist()
    bottom = scores.nsmallest(5).index.tolist()
    print(f"\nPC{pc_idx+1} ({explained[pc_idx]*100:.1f}%)")
    print(f"  HIGH: {', '.join(top)}")
    print(f"  LOW:  {', '.join(bottom)}")

# %% Project individual verses onto PCA axes for interpretation
# Transform every verse embedding through the same scaler+PCA pipeline
verse_pcs = pca.transform(scaler.transform(verse_embeddings))  # (N_verses, N_COMPONENTS)

N_EXAMPLES = 8  # verses to show at each extreme per PC

lines = []
lines.append(f"PCA Dimensions — {CORPUS_NAME}")
lines.append(f"Model: {MODEL_NAME}  |  Components: {N_COMPONENTS}")
lines.append("=" * 80)

lines.append("\nVARIANCE EXPLAINED")
lines.append("-" * 40)
for i, (e, c) in enumerate(zip(explained, cumulative)):
    lines.append(f"  PC{i+1:02d}: {e*100:.1f}%  (cumulative: {c*100:.1f}%)")

for pc_idx in range(N_COMPONENTS):
    book_scores  = pd.Series(book_pcs[:, pc_idx], index=book_order)
    verse_scores = pd.Series(verse_pcs[:, pc_idx], index=range(len(df)))

    top_books    = book_scores.nlargest(5).index.tolist()
    bottom_books = book_scores.nsmallest(5).index.tolist()

    top_verse_idx    = verse_scores.nlargest(N_EXAMPLES).index.tolist()
    bottom_verse_idx = verse_scores.nsmallest(N_EXAMPLES).index.tolist()

    lines.append(f"\n{'=' * 80}")
    lines.append(f"PC{pc_idx+1}  —  {explained[pc_idx]*100:.1f}% of variance")
    lines.append(f"{'=' * 80}")

    lines.append(f"\nBOOKS — HIGH end: {', '.join(top_books)}")
    lines.append(f"BOOKS — LOW  end: {', '.join(bottom_books)}")

    lines.append(f"\nVERSES — HIGH end (PC{pc_idx+1} most positive):")
    for idx in top_verse_idx:
        row = df.iloc[idx]
        snippet = row["text"][:120].replace("\n", " ")
        lines.append(f"  [{row['book']} {row['unit_label']}]  {snippet}")

    lines.append(f"\nVERSES — LOW  end (PC{pc_idx+1} most negative):")
    for idx in bottom_verse_idx:
        row = df.iloc[idx]
        snippet = row["text"][:120].replace("\n", " ")
        lines.append(f"  [{row['book']} {row['unit_label']}]  {snippet}")

output = "\n".join(lines)
print(output)

with open("03_pca_dimensions.txt", "w", encoding="utf-8") as f:
    f.write(output + "\n")

print("\nSaved → 03_pca_dimensions.txt")
