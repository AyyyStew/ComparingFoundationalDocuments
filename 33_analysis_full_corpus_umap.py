# %% [markdown]
# # Full Corpus UMAP — All Traditions
#
# Every embedded passage projected into 2D with UMAP.
# Color = tradition. Excludes the 3 non-KJV Bible translations.
#
# Two plots:
#   1. All passages — one color per tradition, small dots
#   2. Same layout but faceted: one subplot per tradition,
#      that tradition highlighted in color, everything else grey.
#      Lets you see each tradition's footprint in the shared space.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from umap import UMAP

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA

MODEL_NAME = "all-mpnet-base-v2"

TRADITION_COLORS = {
    "Abrahamic":  "#e63946",  # crimson red
    "Dharmic":    "#ffd166",  # golden yellow
    "Buddhist":   "#06d6a0",  # emerald green
    "Taoist":     "#7209b7",  # deep violet
    "Norse":      "#118ab2",  # cobalt blue
    "Confucian":  "#f4845f",  # coral orange
    "Philosophy": "#f72585",  # hot magenta
    "Scientific": "#b5e853",  # lime green
    "Literature": "#48cae4",  # bright cyan
    "Historical": "#cb997e",  # tan
    "News":       "#4a4e69",  # slate grey
}

# Order for legend / facets (most semantically interesting first)
TRADITION_ORDER = [
    "Abrahamic", "Dharmic", "Buddhist", "Taoist",
    "Norse", "Confucian",
    "Philosophy", "Scientific",
    "Literature", "Historical", "News",
]

# %% Load
conn = get_conn()
rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus,
           p.id AS passage_id, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY t.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "passage_id", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} passages")
print(df.groupby("tradition").size().rename("passages").to_string())

# %% UMAP 2D
vecs = np.stack(df["vector"].values).astype("float32")

print("\nFitting UMAP (2D) on full corpus ...")
reducer = UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
X2 = reducer.fit_transform(vecs).astype("float32")
df["ux"] = X2[:, 0]
df["uy"] = X2[:, 1]
print("UMAP done.")

# %% Corpus centroids — mean UMAP position per corpus (for labelling)
# Short display names to avoid clutter on the plot
CORPUS_SHORT = {
    "Bible — KJV (King James Version)":          "Bible (KJV)",
    "Quran (Clear Quran Translation)":            "Quran",
    "Bhagavad Gita":                              "Bhagavad Gita",
    "Srimad Bhagavatam":                          "Srimad Bhagavatam",
    "Upanishads (Paramananda)":                   "Upanishads",
    "Yoga Sutras of Patanjali (Johnston)":        "Yoga Sutras",
    "Dhammapada (Müller)":                        "Dhammapada",
    "Diamond Sutra (Gemmell)":                    "Diamond Sutra",
    "Dao De Jing (Linnell)":                      "Dao De Jing",
    "Poetic Edda (Bellows)":                      "Poetic Edda",
    "Analects of Confucius (Legge)":              "Analects",
    "The Republic (Plato)":                       "The Republic",
    "Nicomachean Ethics (Aristotle)":             "Ethics (Aristotle)",
    "Beyond Good and Evil (Nietzsche)":           "Beyond Good & Evil",
    "Thus Spake Zarathustra (Nietzsche)":         "Zarathustra",
    "Critique of Pure Reason (Kant)":             "Critique (Kant)",
    "Discourse on the Method (Descartes)":        "Descartes",
    "On the Origin of Species (Darwin)":          "Origin of Species",
    "Varieties of Religious Experience (James)":  "Varieties (James)",
    "Civilization and Its Discontents (Freud)":   "Freud",
    "Opticks (Newton)":                           "Opticks",
    "Psychology of the Unconscious (Jung)":       "Jung",
    "Don Quixote (Cervantes)":                    "Don Quixote",
    "Frankenstein (Shelley)":                     "Frankenstein",
    "Pride and Prejudice (Austen)":               "Pride & Prejudice",
    "Romeo and Juliet (Shakespeare)":             "Romeo & Juliet",
    "Siddhartha (Hesse)":                         "Siddhartha",
    "Federalist Papers":                          "Federalist Papers",
    "Communist Manifesto":                        "Communist Manifesto",
    "Code of Hammurabi":                          "Hammurabi",
    "Magna Carta":                                "Magna Carta",
    "US Constitution":                            "US Constitution",
    "Luther's 95 Theses":                         "Luther's Theses",
    "News Articles":                              "News",
}

corpus_centroids = (
    df.groupby(["tradition", "corpus"])[["ux", "uy"]]
    .mean()
    .reset_index()
)
corpus_centroids["label"] = corpus_centroids["corpus"].map(
    lambda c: CORPUS_SHORT.get(c, c)
)


def _draw_corpus_labels(ax, traditions_to_label=None, fontsize=7):
    """Overlay corpus centroid dots + labels on ax.
    traditions_to_label: if set, only label those traditions.
    """
    for _, row in corpus_centroids.iterrows():
        if traditions_to_label and row["tradition"] not in traditions_to_label:
            continue
        color = TRADITION_COLORS.get(row["tradition"], "#aaaaaa")
        # Centroid marker
        ax.scatter(row["ux"], row["uy"],
                   s=40, color=color, edgecolors="white",
                   linewidths=0.6, zorder=10, marker="D")
        # Label with a slight offset
        ax.annotate(
            row["label"],
            xy=(row["ux"], row["uy"]),
            fontsize=fontsize,
            color="white",
            fontweight="bold",
            xytext=(4, 4),
            textcoords="offset points",
            zorder=11,
            bbox=dict(boxstyle="round,pad=0.2", fc="#1a1a2e",
                      ec="none", alpha=0.6),
        )


# %% Plot 1 — Full corpus, all traditions
fig, ax = plt.subplots(figsize=(14, 10))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

# Draw in TRADITION_ORDER so more interesting traditions render on top
for trad in TRADITION_ORDER:
    mask = df["tradition"] == trad
    if not mask.any():
        continue
    color = TRADITION_COLORS.get(trad, "#aaaaaa")
    # News and Historical get smaller/more transparent — less semantically interesting
    alpha = 0.12 if trad in ("News",) else 0.30
    size  = 1   if trad in ("News",) else 2
    ax.scatter(
        df.loc[mask, "ux"], df.loc[mask, "uy"],
        s=size, alpha=alpha, color=color, linewidths=0,
    )

# Legend
patches = [
    mpatches.Patch(color=TRADITION_COLORS.get(t, "#aaaaaa"), label=t)
    for t in TRADITION_ORDER
    if (df["tradition"] == t).any()
]
ax.legend(
    handles=patches, fontsize=9, loc="lower right",
    facecolor="#1a1a2e", labelcolor="white", framealpha=0.6,
    ncol=2,
)
_draw_corpus_labels(ax, fontsize=7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Full Corpus — All Traditions in Semantic Space",
             fontsize=14, fontweight="bold", color="white", pad=12)

plt.tight_layout()
plt.savefig("33_full_corpus_umap.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 33_full_corpus_umap.png")

# %% Plot 2 — Faceted: each tradition highlighted
traditions_present = [t for t in TRADITION_ORDER if (df["tradition"] == t).any()]
n_trads = len(traditions_present)
ncols   = 4
nrows   = (n_trads + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
fig.patch.set_facecolor("#1a1a2e")
axes_flat = axes.flatten()

# Precompute grey background (all passages)
bg_x = df["ux"].values
bg_y = df["uy"].values

for idx, trad in enumerate(traditions_present):
    ax    = axes_flat[idx]
    mask  = (df["tradition"] == trad).values
    color = TRADITION_COLORS.get(trad, "#aaaaaa")

    ax.set_facecolor("#1a1a2e")
    # Grey background — all other passages
    ax.scatter(bg_x[~mask], bg_y[~mask],
               s=0.5, alpha=0.06, color="#888888", linewidths=0)
    # Highlighted tradition
    ax.scatter(bg_x[mask], bg_y[mask],
               s=2, alpha=0.5, color=color, linewidths=0)

    # Label only this tradition's corpora — keeps each subplot readable
    _draw_corpus_labels(ax, traditions_to_label={trad}, fontsize=6)

    n = mask.sum()
    ax.set_title(f"{trad}\n({n:,})", fontsize=9, fontweight="bold",
                 color=color, pad=4)
    ax.set_xticks([]); ax.set_yticks([])

# Hide unused subplots
for idx in range(n_trads, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("Full Corpus — Per-Tradition Footprint in Semantic Space",
             fontsize=13, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
plt.savefig("33_full_corpus_umap_faceted.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: 33_full_corpus_umap_faceted.png")

# %% Quick stats — centroid distances between traditions
print("\n=== Tradition centroids (mean UMAP coords) ===")
centroids = df.groupby("tradition")[["ux", "uy"]].mean().round(2)
print(centroids.to_string())
