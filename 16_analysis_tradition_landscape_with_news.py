# %% [markdown]
# # Tradition Landscape (with News Baseline)
# Same two views as 14 but includes News Articles as a secular baseline.
#
#   1. Corpus-level  — each text collapsed to its mean embedding, one point per corpus
#   2. Book-level    — each book/category collapsed to its mean, labelled by group

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap

from db.schema import get_conn

MODEL_NAME = "all-mpnet-base-v2"

SKIP_CORPORA = {
    "Bible — ACV (A Conservative Version)",
    "Bible — BBE (Bible in Basic English)",
    "Bible — YLT (Young's Literal Translation)",
}

TRADITION_GROUP = {
    "Abrahamic":  "Sacred Texts",
    "Dharmic":    "Sacred Texts",
    "Buddhist":   "Sacred Texts",
    "Taoist":     "Sacred Texts",
    "Literature": "Literature",
    "Historical": "Historical",
    "News":       "News",
}

GROUP_COLORS = {
    "Sacred Texts": "#e05c5c",
    "Literature":   "#4a90d9",
    "Historical":   "#9b59b6",
}

NEWS_CATEGORY_COLORS = {
    "sports":   "#44bb99",
    "business": "#bbaa44",
}

# %% Load all embeddings
conn = get_conn()

rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus,
           p.book, p.section,
           e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()

conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "book", "section", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].copy()
df["vector"] = df["vector"].apply(np.array)
df["group"] = df["tradition"].map(TRADITION_GROUP).fillna("Other")

print(f"Loaded {len(df):,} passages across {df['corpus'].nunique()} corpora")
print(df.groupby("group")["corpus"].nunique().to_string())

# %% ── Plot 1: Corpus-level ──────────────────────────────────────────────────

corpus_means = (
    df.groupby(["group", "tradition", "corpus"])["vector"]
    .apply(lambda vecs: np.mean(np.stack(vecs), axis=0))
    .reset_index()
)
corpus_means.columns = ["group", "tradition", "corpus", "mean_vec"]

non_news_corpus = corpus_means[corpus_means["group"] != "News"]
news_corpus     = corpus_means[corpus_means["group"] == "News"]

X_non_news = np.stack(non_news_corpus["mean_vec"].values)
reducer_corpus = umap.UMAP(
    n_neighbors=min(10, len(X_non_news) - 1),
    min_dist=0.3, metric="cosine", random_state=42
)
xy_non_news = reducer_corpus.fit_transform(X_non_news)
corpus_means.loc[non_news_corpus.index, "x"] = xy_non_news[:, 0]
corpus_means.loc[non_news_corpus.index, "y"] = xy_non_news[:, 1]

if len(news_corpus):
    xy_news = reducer_corpus.transform(np.stack(news_corpus["mean_vec"].values))
    corpus_means.loc[news_corpus.index, "x"] = xy_news[:, 0]
    corpus_means.loc[news_corpus.index, "y"] = xy_news[:, 1]

fig, ax = plt.subplots(figsize=(13, 10))

for _, row in corpus_means.iterrows():
    color = GROUP_COLORS.get(row["group"], "#888888")
    ax.scatter(row["x"], row["y"], color=color, s=180, zorder=3,
               edgecolors="white", linewidths=0.8)

    label = row["corpus"]
    for s in ["Bible — ", " (Shelley)", " (Austen)", " (Cervantes)",
              " (Shakespeare)", " (Müller)", " (Linnell)", " (Johnston)"]:
        label = label.replace(s, "")
    label = label.replace("of Patanjali", "").strip()

    ax.annotate(label, (row["x"], row["y"]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8.5, color="#222222")

legend_patches = [
    mpatches.Patch(color=c, label=g)
    for g, c in GROUP_COLORS.items()
    if g in corpus_means["group"].values
]
ax.legend(handles=legend_patches, loc="best", fontsize=9, framealpha=0.85)
ax.set_title("Corpus Landscape — Mean Embedding per Text (News as Baseline)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_aspect("equal", "datalim")
plt.tight_layout()
plt.savefig("16_corpus_landscape_with_news.png", dpi=150)
plt.show()
print("Saved: 16_corpus_landscape_with_news.png")

# %% ── Plot 2: Book-level ────────────────────────────────────────────────────
# Non-news: aggregate by book (as in 14)
# News: aggregate by article (section) so each article is one point

non_news = df[df["group"] != "News"]
news     = df[df["group"] == "News"]

book_means = (
    non_news.groupby(["group", "tradition", "corpus", "book"])["vector"]
    .apply(lambda vecs: np.mean(np.stack(vecs), axis=0))
    .reset_index()
)
book_means.columns = ["group", "tradition", "corpus", "book", "mean_vec"]

article_means = (
    news.groupby(["group", "tradition", "corpus", "book", "section"])["vector"]
    .apply(lambda vecs: np.mean(np.stack(vecs), axis=0))
    .reset_index()
)
article_means.columns = ["group", "tradition", "corpus", "news_category", "section", "mean_vec"]

print(f"\nBook-level points: {len(book_means):,}  |  Article points: {len(article_means):,}")

X_book    = np.stack(book_means["mean_vec"].values)
X_article = np.stack(article_means["mean_vec"].values)

# Fit on non-news only so news doesn't skew the axes
reducer_book = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
xy_book = reducer_book.fit_transform(X_book)
book_means["x"] = xy_book[:, 0]
book_means["y"] = xy_book[:, 1]

# Project news articles into that same space
xy_article = reducer_book.transform(X_article)
article_means["x"] = xy_article[:, 0]
article_means["y"] = xy_article[:, 1]

all_means = pd.concat([book_means, article_means], ignore_index=True, sort=False)

fig, ax = plt.subplots(figsize=(16, 12))

# News articles — split by category, very transparent, small
for category, grp in all_means[all_means["group"] == "News"].groupby("news_category"):
    color = NEWS_CATEGORY_COLORS.get(category, "#aaaaaa")
    ax.scatter(grp["x"], grp["y"], color=color, s=12, alpha=0.25,
               edgecolors="none", zorder=1)

# Everything else
for group, grp in all_means[all_means["group"] != "News"].groupby("group"):
    color = GROUP_COLORS.get(group, "#888888")
    ax.scatter(grp["x"], grp["y"], color=color, s=35, alpha=0.75,
               edgecolors="none", label=group, zorder=2)

# Corpus centroid labels (non-news only)
for corpus, grp in all_means[all_means["group"] != "News"].groupby("corpus"):
    cx, cy = grp["x"].mean(), grp["y"].mean()
    label = corpus
    for s in ["Bible — ", " (Shelley)", " (Austen)", " (Cervantes)",
              " (Shakespeare)", " (Müller)", " (Linnell)", " (Johnston)"]:
        label = label.replace(s, "")
    label = label.replace("of Patanjali", "").strip()
    ax.annotate(label, (cx, cy), fontsize=7.5, color="#222222",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

legend_patches = [
    mpatches.Patch(color=c, label=g)
    for g, c in GROUP_COLORS.items()
    if g in all_means["group"].values
] + [
    mpatches.Patch(color=c, label=f"News — {cat}")
    for cat, c in NEWS_CATEGORY_COLORS.items()
]
ax.legend(handles=legend_patches, loc="best", fontsize=9, framealpha=0.85)
ax.set_title("Book / Category Landscape — Mean Embedding per Book (News as Baseline)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
ax.set_aspect("equal", "datalim")
plt.tight_layout()
plt.savefig("16_book_landscape_with_news.png", dpi=150)
plt.show()
print("Saved: 16_book_landscape_with_news.png")
