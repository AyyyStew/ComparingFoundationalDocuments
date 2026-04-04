# %% [markdown]
# # Sacred Texts vs Literature — Per-pair Book/Chapter Heatmaps
# One heatmap per (sacred corpus × literature corpus) pair.
#   rows = books of the sacred text  (in canonical / ingestion order)
#   cols = chapters of the literature corpus  (in chapter order)
# Cell value = mean cosine similarity between all passages in that
# sacred book and all passages in that literature chapter.

# %% Imports
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA, SACRED_TRADITIONS

MODEL_NAME = "all-mpnet-base-v2"

LITERATURE_CORPORA = [
    "Don Quixote (Cervantes)",
    "Frankenstein (Shelley)",
    "Pride and Prejudice (Austen)",
    "Romeo and Juliet (Shakespeare)",
]

SACRED_CORPUS_ORDER = [
    "Bible — KJV (King James Version)",
    "Bhagavad Gita",
    "Dhammapada (Müller)",
    "Yoga Sutras of Patanjali (Johnston)",
    "Dao De Jing (Linnell)",
]

# %% Helpers

def _roman_to_int(s: str) -> int:
    vals = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    result, prev = 0, 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        result += v if v >= prev else -v
        prev = v
    return result or 0

def sort_key_chapter(name: str) -> int:
    """Return a sort integer from a chapter/act name."""
    # "Act 1" / "Act 2"
    m = re.search(r"Act\s+(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "Chapter 1", "Chapter 10"  (arabic)
    m = re.search(r"Chapter\s+(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # "CHAPTER\nIX" / "CHAPTER IX." / "CHAPTER L."  (roman)
    m = re.search(r"Chapter[\s\n]+([IVXLCDM]+)\.?", name, re.IGNORECASE)
    if m:
        return _roman_to_int(m.group(1))
    return 0

def short_lit_label(name: str) -> str:
    """Short column label for a literature chapter."""
    m = re.search(r"Act\s+(\d+)", name, re.IGNORECASE)
    if m:
        return f"Act {m.group(1)}"
    m = re.search(r"Chapter\s+(\d+)", name, re.IGNORECASE)
    if m:
        return f"Ch {m.group(1)}"
    m = re.search(r"Chapter[\s\n]+([IVXLCDM]+)\.?", name, re.IGNORECASE)
    if m:
        return f"Ch {_roman_to_int(m.group(1))}"
    return name[:12]

# %% Load all embeddings
conn = get_conn()
rows = conn.execute("""
    SELECT c.name AS corpus, p.book, p.id AS passage_id, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["corpus", "book", "passage_id", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].copy()
df["vector"] = df["vector"].apply(np.array)

sacred_df = df[df["corpus"].isin(SACRED_CORPUS_ORDER)].copy()
lit_df    = df[df["corpus"].isin(LITERATURE_CORPORA)].copy()

print(f"Sacred passages:     {len(sacred_df):,}")
print(f"Literature passages: {len(lit_df):,}")

# %% Canonical book order per sacred corpus (by first passage_id = ingestion order)
def get_book_order(corpus_df):
    return (
        corpus_df.groupby("book")["passage_id"].min()
        .sort_values().index.tolist()
    )

# %% Mean embedding per (corpus, book) — L2-normalised
def mean_normed(vecs):
    m = np.mean(np.stack(vecs.values), axis=0)
    return m / (np.linalg.norm(m) + 1e-9)

sacred_means = (
    sacred_df.groupby(["corpus", "book"])["vector"]
    .apply(mean_normed).reset_index()
)
sacred_means.columns = ["corpus", "book", "vec"]

lit_means = (
    lit_df.groupby(["corpus", "book"])["vector"]
    .apply(mean_normed).reset_index()
)
lit_means.columns = ["corpus", "book", "vec"]

# %% ── Generate one heatmap per (sacred, literature) pair ─────────────────────

# Compute global vmin/vmax across all pairs for a shared colour scale
all_sims = []
for sc in SACRED_CORPUS_ORDER:
    s_sub = sacred_means[sacred_means["corpus"] == sc]
    for lc in LITERATURE_CORPORA:
        l_sub = lit_means[lit_means["corpus"] == lc]
        if s_sub.empty or l_sub.empty:
            continue
        S = np.stack(s_sub["vec"].values)
        L = np.stack(l_sub["vec"].values)
        all_sims.append((S @ L.T).ravel())

all_sims_flat = np.concatenate(all_sims)
vmin, vmax = float(all_sims_flat.min()), float(all_sims_flat.max())
print(f"\nGlobal similarity range: {vmin:.3f} – {vmax:.3f}")

for sc in SACRED_CORPUS_ORDER:
    s_sub = sacred_means[sacred_means["corpus"] == sc].copy()
    if s_sub.empty:
        continue

    # Canonical row order
    sacred_book_order = get_book_order(sacred_df[sacred_df["corpus"] == sc])
    s_sub = s_sub.set_index("book").loc[sacred_book_order].reset_index()

    sc_short = (sc
        .replace("Bible — KJV (King James Version)", "Bible (KJV)")
        .replace(" (Müller)", "").replace(" (Johnston)", "").replace(" (Linnell)", "")
    )

    for lc in LITERATURE_CORPORA:
        l_sub = lit_means[lit_means["corpus"] == lc].copy()
        if l_sub.empty:
            continue

        # Sort literature chapters numerically
        l_sub["_sort"] = l_sub["book"].apply(sort_key_chapter)
        l_sub = l_sub.sort_values("_sort").reset_index(drop=True)
        col_labels = l_sub["book"].apply(short_lit_label).tolist()

        S = np.stack(s_sub["vec"].values)  # (n_sacred_books, 768)
        L = np.stack(l_sub["vec"].values)  # (n_lit_chapters, 768)
        mat = S @ L.T                       # (n_sacred_books, n_lit_chapters)

        n_rows, n_cols = mat.shape
        col_width  = max(6, n_cols * 0.18)
        row_height = max(4, n_rows * 0.25)

        fig, ax = plt.subplots(figsize=(col_width, row_height))

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)

        # Column labels (literature chapters) — rotate 90° if many
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels,
                           rotation=90 if n_cols > 15 else 45,
                           ha="center" if n_cols > 15 else "right",
                           fontsize=max(5, min(8, 120 // n_cols)))
        ax.xaxis.set_ticks_position("bottom")

        # Row labels (sacred books)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(s_sub["book"].tolist(),
                           fontsize=max(5, min(8, 120 // n_rows)))

        lc_short = lc.split("(")[0].strip()
        ax.set_xlabel(lc_short, fontsize=9)
        ax.set_ylabel(sc_short, fontsize=9)

        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Cosine similarity")
        ax.set_title(f"{sc_short}  ×  {lc_short}",
                     fontsize=11, fontweight="bold")

        plt.tight_layout()
        safe_sc = sc_short.lower().replace(" ", "_").replace("(","").replace(")","")
        safe_lc = lc_short.lower().replace(" ", "_")
        fname = f"19_{safe_sc}_vs_{safe_lc}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {fname}")

        # Print top 5 most similar pairs for this combination
        flat = [(mat[i, j], s_sub.iloc[i]["book"], l_sub.iloc[j]["book"])
                for i in range(n_rows) for j in range(n_cols)]
        flat.sort(reverse=True)
        print(f"  Top 5: " + " | ".join(f"{s:.3f} {rb} ↔ {lb}" for s, rb, lb in flat[:5]))
