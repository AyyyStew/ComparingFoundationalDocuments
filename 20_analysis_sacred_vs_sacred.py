# %% [markdown]
# # Sacred Texts vs Sacred Texts — Book-level Heatmaps
#
# For each sacred corpus (source), one figure with 4 subplots — one per
# other sacred corpus (target). Source books on y-axis (canonical order),
# target books on x-axis (canonical order).
#
# Dao De Jing has only 1 "book" so its chapters (81 passages) are used
# as the x/y axis instead to give meaningful granularity.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA, SACRED_TRADITIONS

MODEL_NAME = "all-mpnet-base-v2"

SACRED_CORPUS_ORDER = [
    "Bible — KJV (King James Version)",
    "Bhagavad Gita",
    "Dhammapada (Müller)",
    "Yoga Sutras of Patanjali (Johnston)",
    "Dao De Jing (Linnell)",
]

SHORT_NAMES = {
    "Bible — KJV (King James Version)":      "Bible (KJV)",
    "Bhagavad Gita":                          "Bhagavad Gita",
    "Dhammapada (Müller)":                    "Dhammapada",
    "Yoga Sutras of Patanjali (Johnston)":    "Yoga Sutras",
    "Dao De Jing (Linnell)":                  "Dao De Jing",
}

# %% Load embeddings
conn = get_conn()
rows = conn.execute("""
    SELECT c.name AS corpus, p.book, p.unit_number, p.unit_label, p.id AS pid, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    AND t.name IN (?, ?, ?, ?)
    ORDER BY c.name, p.id
""", [MODEL_NAME, "Abrahamic", "Dharmic", "Buddhist", "Taoist"]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["corpus", "book", "unit_number", "unit_label", "pid", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].copy()
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} sacred passages")
print(df.groupby("corpus").size().to_string())

# %% Compute mean embedding per (corpus, axis_label) — L2-normalised
# For Dao De Jing: use unit_label (chapter number) as the axis since there's only 1 book.
# For all others: use book.

def mean_normed(vecs):
    m = np.mean(np.stack(vecs.values), axis=0)
    return m / (np.linalg.norm(m) + 1e-9)

def get_axis_means(corpus_name):
    """Return (labels_in_order, mean_vectors) for a corpus."""
    sub = df[df["corpus"] == corpus_name].copy()
    if corpus_name == "Dao De Jing (Linnell)":
        # Use chapter number as axis
        sub["axis"] = sub["unit_number"].astype(str).str.zfill(3)  # zero-pad for sort
        labels_ordered = sub.groupby("axis")["pid"].min().sort_values().index.tolist()
        display_labels = [str(int(l)) for l in labels_ordered]  # "001" -> "1"
    else:
        sub["axis"] = sub["book"]
        labels_ordered = sub.groupby("axis")["pid"].min().sort_values().index.tolist()
        display_labels = labels_ordered

    means = (
        sub.groupby("axis")["vector"]
        .apply(mean_normed)
    )
    vecs = np.stack([means.loc[l] for l in labels_ordered])
    return display_labels, vecs

# Pre-compute means for all corpora
corpus_data = {}
for c in SACRED_CORPUS_ORDER:
    labels, vecs = get_axis_means(c)
    corpus_data[c] = {"labels": labels, "vecs": vecs}
    print(f"  {SHORT_NAMES[c]}: {len(labels)} axis labels")

# %% Global colour scale
all_sims = []
for i, ca in enumerate(SACRED_CORPUS_ORDER):
    for cb in SACRED_CORPUS_ORDER[i+1:]:
        Va = corpus_data[ca]["vecs"]
        Vb = corpus_data[cb]["vecs"]
        all_sims.append((Va @ Vb.T).ravel())

flat = np.concatenate(all_sims)
vmin, vmax = float(flat.min()), float(flat.max())
print(f"\nGlobal similarity range: {vmin:.3f} – {vmax:.3f}")

# %% ── One figure per source corpus ──────────────────────────────────────────
# 4 subplots side by side — one per other sacred corpus.
# Column widths proportional to number of target axis labels.

for source in SACRED_CORPUS_ORDER:
    targets = [c for c in SACRED_CORPUS_ORDER if c != source]

    src_labels = corpus_data[source]["labels"]
    src_vecs   = corpus_data[source]["vecs"]
    n_rows     = len(src_labels)
    src_short  = SHORT_NAMES[source]

    # Column widths proportional to target label counts
    tgt_sizes  = [len(corpus_data[t]["labels"]) for t in targets]
    col_widths = [max(1.5, n * 0.18) for n in tgt_sizes]
    fig_width  = sum(col_widths) + 1.5   # +1.5 for colorbar
    fig_height = max(5, n_rows * 0.22)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = GridSpec(1, len(targets), figure=fig,
                   width_ratios=col_widths, wspace=0.4)

    axes = [fig.add_subplot(gs[0, j]) for j in range(len(targets))]

    ims = []
    for j, (ax, target) in enumerate(zip(axes, targets)):
        tgt_labels = corpus_data[target]["labels"]
        tgt_vecs   = corpus_data[target]["vecs"]
        tgt_short  = SHORT_NAMES[target]

        mat = src_vecs @ tgt_vecs.T  # (n_src, n_tgt)
        ims.append(ax.imshow(mat, aspect="auto", cmap="YlOrRd",
                             vmin=vmin, vmax=vmax))

        # X axis — target chapters/books
        n_cols = len(tgt_labels)
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(
            tgt_labels,
            rotation=90,
            fontsize=max(4, min(7, 80 // max(n_cols, 1))),
        )
        ax.set_xlabel(tgt_short, fontsize=8, labelpad=4)

        # Y axis — source books, only on the leftmost subplot
        ax.set_yticks(range(n_rows))
        if j == 0:
            ax.set_yticklabels(
                src_labels,
                fontsize=max(4, min(7, 80 // max(n_rows, 1))),
            )
            ax.set_ylabel(src_short, fontsize=9)
        else:
            ax.set_yticklabels([])

        ax.set_title(f"vs {tgt_short}", fontsize=8, pad=4)

    # Shared colorbar on the right
    fig.colorbar(ims[-1], ax=axes[-1], fraction=0.08, pad=0.04,
                 label="Cosine similarity")

    fig.suptitle(f"{src_short} vs all Sacred Texts",
                 fontsize=11, fontweight="bold", y=1.01)

    plt.tight_layout()
    safe = src_short.lower().replace(" ", "_").replace("(","").replace(")","")
    fname = f"20_{safe}_vs_sacred.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

    # Top 5 most similar pairs across all targets
    print(f"\n  Top 5 cross-sacred pairs for {src_short}:")
    top = []
    for target in targets:
        tgt_labels = corpus_data[target]["labels"]
        tgt_vecs   = corpus_data[target]["vecs"]
        mat = src_vecs @ tgt_vecs.T
        for i in range(mat.shape[0]):
            for j2 in range(mat.shape[1]):
                top.append((mat[i, j2], src_labels[i],
                             SHORT_NAMES[target], tgt_labels[j2]))
    top.sort(reverse=True)
    for sim, sl, tc, tl in top[:5]:
        print(f"    {sim:.3f}  {sl}  ↔  {tc} | {tl}")
    print()
