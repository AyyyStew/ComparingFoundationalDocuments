# %% [markdown]
# # Narrative Flow — Sacred Texts
#
# Does each sacred text "travel" through semantic space chapter by chapter,
# or does it circle the same conceptual neighborhood?
#
# Pipeline:
#   1. Fit UMAP 2D on all sacred passage embeddings (the full semantic landscape)
#   2. Compute mean embedding per chapter for each corpus
#   3. Project chapter means into the 2D space
#   4. Plot each corpus as a directed path — dots numbered by chapter,
#      arrows showing the sequence, color gradient blue→red (start→end)
#
# One subplot per sacred corpus. Background shows the full passage cloud faintly
# so you can see where each text sits relative to the broader sacred space.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from umap import UMAP

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA

MODEL_NAME = "all-mpnet-base-v2"

SACRED_CORPORA = [
    "Bible — KJV (King James Version)",
    "Bhagavad Gita",
    "Dhammapada (Müller)",
    "Yoga Sutras of Patanjali (Johnston)",
    "Dao De Jing (Linnell)",
]

SHORT_NAMES = {
    "Bible — KJV (King James Version)":   "Bible (KJV)",
    "Bhagavad Gita":                       "Bhagavad Gita",
    "Dhammapada (Müller)":                 "Dhammapada",
    "Yoga Sutras of Patanjali (Johnston)": "Yoga Sutras",
    "Dao De Jing (Linnell)":               "Dao De Jing",
}

# Abbreviated Bible book names in canonical order
BIBLE_ABBREV = [
    "Gen","Exo","Lev","Num","Deu","Jos","Jdg","Rut","1Sa","2Sa",
    "1Ki","2Ki","1Ch","2Ch","Ezr","Neh","Est","Job","Psa","Pro",
    "Ecc","Son","Isa","Jer","Lam","Eze","Dan","Hos","Joe","Amo",
    "Oba","Jon","Mic","Nah","Hab","Zep","Hag","Zec","Mal",
    "Mat","Mar","Luk","Joh","Act","Rom","1Co","2Co","Gal","Eph",
    "Php","Col","1Th","2Th","1Ti","2Ti","Tit","Phm","Heb","Jam",
    "1Pe","2Pe","1Jo","2Jo","3Jo","Jud","Rev",
]

# %% Load
conn = get_conn()
rows = conn.execute("""
    SELECT c.name AS corpus, p.id AS passage_id,
           p.book, p.unit_number, p.unit_label, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    AND t.name IN ('Abrahamic', 'Dharmic', 'Buddhist', 'Taoist')
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["corpus", "passage_id", "book", "unit_number", "unit_label", "vector"])
df = df[df["corpus"].isin(SACRED_CORPORA)].reset_index(drop=True)
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} passages")
print(df.groupby("corpus").size().to_string())

# %% Chapter key + canonical order
def make_chapter_key(row):
    if row["corpus"] == "Bible — KJV (King James Version)":
        chap = row["unit_label"].split(":")[0]
        return f"{row['book']}||{chap}"          # book||chap_num for sorting
    elif row["corpus"] == "Dao De Jing (Linnell)":
        return f"DDJ||{int(row['unit_number'])}"
    else:
        return row["book"]

df["chapter_key"] = df.apply(make_chapter_key, axis=1)

# %% UMAP 2D — fit on all passage-level embeddings
vecs = np.stack(df["vector"].values).astype("float32")

print("\nFitting UMAP (2D) on all sacred passages ...")
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

# %% Chapter means — projected into 2D space via passage mean
def mean_normed(vecs_list):
    m = np.mean(np.stack(vecs_list), axis=0)
    return m / (np.linalg.norm(m) + 1e-9)

# For each corpus, compute 2D mean per chapter (mean of passage UMAP coords)
def get_chapter_sequence(corpus_name):
    sub = df[df["corpus"] == corpus_name].copy()

    if corpus_name == "Bible — KJV (King James Version)":
        # Aggregate to book level — one point per book (66 total)
        sub["_book_order"] = sub["book"].apply(
            lambda b: BIBLE_ABBREV.index(b[:3]) if b[:3] in BIBLE_ABBREV else 9999
        )
        book_means = (
            sub.groupby(["_book_order", "book"])
            .agg(ux=("ux", "mean"), uy=("uy", "mean"))
            .reset_index()
            .sort_values("_book_order")
        )
        labels = [BIBLE_ABBREV[r["_book_order"]] if r["_book_order"] < len(BIBLE_ABBREV) else r["book"][:3]
                  for _, r in book_means.iterrows()]
        coords = book_means[["ux", "uy"]].values

    elif corpus_name == "Dao De Jing (Linnell)":
        sub["_chap"] = sub["unit_number"].astype(int)
        chap_means = (
            sub.groupby("_chap")
            .agg(ux=("ux", "mean"), uy=("uy", "mean"))
            .reset_index()
            .sort_values("_chap")
        )
        labels = chap_means["_chap"].astype(str).tolist()
        coords = chap_means[["ux", "uy"]].values

    else:
        # book = chapter; canonical order by min passage_id
        book_order = sub.groupby("book")["passage_id"].min().sort_values().index.tolist()
        chap_means = (
            sub.groupby("book")
            .agg(ux=("ux", "mean"), uy=("uy", "mean"))
            .reindex(book_order)
            .reset_index()
        )
        # Short label: chapter number extracted from book name
        import re
        def short_label(name):
            m = re.search(r"(\d+)", name)
            return m.group(1) if m else name[:6]
        labels = [short_label(b) for b in chap_means["book"]]
        coords = chap_means[["ux", "uy"]].values

    return coords, labels

# %% Plot
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
axes_flat = axes.flatten()

# Pre-compute chapter sequences so we can zoom before drawing background
corpus_sequences = {c: get_chapter_sequence(c) for c in SACRED_CORPORA}

for idx, corpus in enumerate(SACRED_CORPORA):
    ax     = axes_flat[idx]
    coords, labels = corpus_sequences[corpus]
    n      = len(coords)

    # Zoom to bounding box of this corpus's chapter walk + padding
    pad = max(1.0, (coords[:, 0].max() - coords[:, 0].min() + coords[:, 1].max() - coords[:, 1].min()) * 0.15)
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)

    # Background — only passages within the zoomed window
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    in_view = (
        (X2[:, 0] >= xlim[0]) & (X2[:, 0] <= xlim[1]) &
        (X2[:, 1] >= ylim[0]) & (X2[:, 1] <= ylim[1])
    )
    ax.scatter(X2[in_view, 0], X2[in_view, 1],
               s=1, alpha=0.15, color="#aaaaaa", linewidths=0)
    cmap   = cm.get_cmap("coolwarm")
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # Draw arrows between consecutive chapters
    for i in range(n - 1):
        ax.annotate(
            "",
            xy=coords[i + 1],
            xytext=coords[i],
            arrowprops=dict(
                arrowstyle="->",
                color=colors[i],
                lw=1.0,
                alpha=0.7,
            ),
        )

    # Plot chapter dots
    for i, (coord, color) in enumerate(zip(coords, colors)):
        ax.scatter(*coord, color=color, s=35, zorder=5, linewidths=0)

    # Label all points; font size scales with count
    fontsize = 5.5 if n > 50 else 7 if n > 20 else 8
    for i, (coord, label) in enumerate(zip(coords, labels)):
        ax.annotate(
            label,
            xy=coord,
            fontsize=fontsize,
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            color="#333333",
            zorder=6,
        )

    # Start / end markers
    ax.scatter(*coords[0],  marker="^", s=60, color="steelblue", zorder=7, label="Start")
    ax.scatter(*coords[-1], marker="s", s=60, color="tomato",    zorder=7, label="End")

    ax.set_title(SHORT_NAMES[corpus], fontsize=11, fontweight="bold")
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    if idx == 0:
        ax.legend(fontsize=8, loc="lower right", framealpha=0.7)

# Colorbar — start→end gradient
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes_flat[5], fraction=0.5, pad=0.05)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(["Start", "End"])
axes_flat[5].set_visible(False)

fig.patch.set_facecolor("white")
fig.suptitle("Narrative Flow — Sacred Texts in Semantic Space",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("27_narrative_flow.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("Saved: 27_narrative_flow.png")
