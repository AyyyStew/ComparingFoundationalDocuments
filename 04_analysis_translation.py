# %% [markdown]
# # KJV Bible — Translation Comparison
# Pick a reference translation, compute cosine similarity of each other
# translation's verses against it, then visualize divergence.
#
# Analyses:
#   1. UMAP — all 4 translations together, colored by translation
#   2. Book × translation-pair heatmap of mean cosine similarity
#   3. Canon trajectory — reference mean embedding per book as baseline,
#      each translation's deviation from it plotted above + below
#   4. Most divergent verses table

# %% Imports
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.metrics.pairwise import cosine_similarity

from db.schema import get_conn

MODEL_NAME = "all-mpnet-base-v2"

TRANSLATIONS = {
    "KJV": "Bible — KJV (King James Version)",
    "ACV": "Bible — ACV (A Conservative Version)",
    "YLT": "Bible — YLT (Young's Literal Translation)",
    "BBE": "Bible — BBE (Bible in Basic English)",
}

# Reference translation — cosine distances are computed relative to this
REFERENCE = "KJV"

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST    = 0.05
RANDOM_STATE     = 42

# %% Load all translations from DuckDB
conn = get_conn()

def load_translation(conn, corpus_name: str, model_name: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT p.book, p.section::INTEGER AS chapter, p.unit_number AS verse, e.vector
        FROM passage p
        JOIN embedding e ON e.passage_id = p.id
        JOIN corpus c    ON c.id = p.corpus_id
        WHERE c.name       = ?
          AND e.model_name = ?
        ORDER BY p.id
        """,
        [corpus_name, model_name],
    ).fetchall()
    df = pd.DataFrame(rows, columns=["book", "chapter", "verse", "vector"])
    df["ref"] = df["book"] + "|" + df["chapter"].astype(str) + "|" + df["verse"].astype(str)
    return df

all_dfs = {t: load_translation(conn, name, MODEL_NAME) for t, name in TRANSLATIONS.items()}
conn.close()

# Canonical book order from reference translation
book_order = all_dfs[REFERENCE]["book"].drop_duplicates().tolist()

print("Loaded verses per translation:")
for t, df in all_dfs.items():
    print(f"  {t}: {len(df):,}")

# %% Align verses across translations on shared ref keys
# Inner join on book|chapter|verse — missing verses (e.g. BBE's 16) are dropped
shared_keys = set(all_dfs[REFERENCE]["ref"])
for t, df in all_dfs.items():
    shared_keys &= set(df["ref"])

print(f"\nShared verses across all 4 translations: {len(shared_keys):,}")

aligned = {}
for t, df in all_dfs.items():
    aligned[t] = df[df["ref"].isin(shared_keys)].set_index("ref").sort_index()

# Aligned embedding matrices (same row order for all)
index_order = sorted(shared_keys)
emb = {t: np.vstack(aligned[t].loc[index_order, "vector"].values).astype(np.float32)
       for t in TRANSLATIONS}

# Meta df (book/chapter/verse) from reference
meta = aligned[REFERENCE].loc[index_order, ["book", "chapter", "verse"]].copy()
meta["canon_idx"] = meta["book"].map({b: i for i, b in enumerate(book_order)}) * 10000 \
                  + meta["chapter"] * 100 + meta["verse"]
meta = meta.sort_values("canon_idx").reset_index(drop=True)
# Re-sort embeddings to match
sorted_keys = meta.index  # already reset, use positional after re-sort
index_order_sorted = meta.index  # reset_index gives 0..N

# Re-align after sort
meta2 = aligned[REFERENCE].loc[sorted(shared_keys), ["book", "chapter", "verse"]].reset_index()
meta2 = meta2.rename(columns={"ref": "key"})
meta2["canon_idx"] = meta2["book"].map({b: i for i, b in enumerate(book_order)}) * 10000 \
                   + meta2["chapter"] * 100 + meta2["verse"]
meta2 = meta2.sort_values("canon_idx").reset_index(drop=True)

emb_sorted = {t: np.vstack(
    aligned[t].loc[meta2["key"].values, "vector"].values
).astype(np.float32) for t in TRANSLATIONS}

meta = meta2

# %% Cosine similarity of each translation vs reference, per verse
ref_emb = emb_sorted[REFERENCE]
cos_sims = {}
for t in TRANSLATIONS:
    if t == REFERENCE:
        continue
    # diagonal of cosine_similarity gives per-verse sim to reference
    sims = (ref_emb * emb_sorted[t]).sum(axis=1) / (
        np.linalg.norm(ref_emb, axis=1) * np.linalg.norm(emb_sorted[t], axis=1) + 1e-9
    )
    cos_sims[t] = sims  # shape (N,)
    print(f"Mean cosine sim {REFERENCE}↔{t}: {sims.mean():.4f}  min: {sims.min():.4f}")

# %% ── PLOT 1: UMAP of all translations together ─────────────────────────────
print("\nRunning joint UMAP (all translations)...")

all_embs   = np.vstack([emb_sorted[t] for t in TRANSLATIONS])
all_labels = np.repeat(list(TRANSLATIONS.keys()), [len(emb_sorted[t]) for t in TRANSLATIONS])

reducer = umap.UMAP(
    n_neighbors=UMAP_N_NEIGHBORS,
    min_dist=UMAP_MIN_DIST,
    n_components=2,
    metric="cosine",
    random_state=RANDOM_STATE,
)
all_2d = reducer.fit_transform(all_embs)

t_colors = {"KJV": "#e63946", "ACV": "#457b9d", "YLT": "#2a9d8f", "BBE": "#e9c46a"}

fig, ax = plt.subplots(figsize=(15, 11))
for t in TRANSLATIONS:
    mask = all_labels == t
    ax.scatter(all_2d[mask, 0], all_2d[mask, 1],
               color=t_colors[t], s=1, alpha=0.2, label=t)

# Centroids per translation
for t in TRANSLATIONS:
    mask = all_labels == t
    cx, cy = all_2d[mask].mean(axis=0)
    ax.scatter(cx, cy, color=t_colors[t], s=120, zorder=5, edgecolors="white", linewidths=1)
    ax.annotate(t, (cx, cy), fontsize=10, fontweight="bold", ha="center", va="bottom",
                xytext=(0, 7), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

ax.set_title("All 4 Translations — Joint UMAP (colored by translation)", fontsize=13)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(markerscale=6, fontsize=9, frameon=False)
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig("04_translations_umap_all.png", dpi=150)
plt.show()

# %% ── PLOT 2: Book × translation-pair heatmap ───────────────────────────────
pairs = [(t, REFERENCE) for t in TRANSLATIONS if t != REFERENCE]

heatmap_data = pd.DataFrame(index=book_order, columns=[f"{t}↔{REFERENCE}" for t, _ in pairs])

for t, ref in pairs:
    col = f"{t}↔{ref}"
    for book in book_order:
        mask = meta["book"].values == book
        if mask.sum() == 0:
            heatmap_data.loc[book, col] = np.nan
            continue
        r = ref_emb[mask]
        o = emb_sorted[t][mask]
        sims = (r * o).sum(axis=1) / (
            np.linalg.norm(r, axis=1) * np.linalg.norm(o, axis=1) + 1e-9
        )
        heatmap_data.loc[book, col] = sims.mean()

heatmap_data = heatmap_data.astype(float)
vals = heatmap_data.values
data_min, data_max = np.nanmin(vals), np.nanmax(vals)
data_mid = (data_min + data_max) / 2

def _draw_heatmap(ax, vmin, vmax, title):
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{t}↔{REFERENCE}" for t, _ in pairs], fontsize=10)
    ax.set_yticks(range(len(book_order)))
    ax.set_yticklabels(book_order, fontsize=7)
    ax.set_title(title, fontsize=11)
    return im

fig, (ax_rel, ax_abs) = plt.subplots(1, 2, figsize=(14, 18))

# Relative — stretched to observed range
im_rel = _draw_heatmap(ax_rel, data_min, data_max, f"Relative scale\n(vs {REFERENCE})")
cbar_rel = fig.colorbar(im_rel, ax=ax_rel, fraction=0.03, pad=0.04, label="Mean cosine similarity")
cbar_rel.set_ticks([data_min, data_mid, data_max])
cbar_rel.set_ticklabels([f"{data_min:.3f}", f"{data_mid:.3f} (mid)", f"{data_max:.3f}"])

# Absolute — fixed -1 to 1
im_abs = _draw_heatmap(ax_abs, -1.0, 1.0, f"Absolute scale\n(vs {REFERENCE})")
fig.colorbar(im_abs, ax=ax_abs, fraction=0.03, pad=0.04, label="Mean cosine similarity")

plt.suptitle(f"Per-book cosine similarity vs {REFERENCE}", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("04_translations_book_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

# %% ── PLOT 3: Canon trajectory ───────────────────────────────────────────────
# Top panel: mean cosine distance from reference per book, per translation
# Bottom panel: deviation from cross-translation mean (convergence/divergence)

book_means = {}  # t -> array of length n_books
for t in TRANSLATIONS:
    if t == REFERENCE:
        continue
    means = []
    for book in book_order:
        mask = meta["book"].values == book
        if mask.sum() == 0:
            means.append(np.nan)
            continue
        r = ref_emb[mask]
        o = emb_sorted[t][mask]
        sims = (r * o).sum(axis=1) / (
            np.linalg.norm(r, axis=1) * np.linalg.norm(o, axis=1) + 1e-9
        )
        means.append(1 - sims.mean())   # cosine distance
    book_means[t] = np.array(means)

x       = np.arange(len(book_order))
overall = np.nanmean(np.vstack(list(book_means.values())), axis=0)  # cross-translation mean

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

# Top panel — raw cosine distance from reference
for t, vals in book_means.items():
    ax1.plot(x, vals, color=t_colors[t], lw=1.8, label=t, marker="o", ms=3)
ax1.plot(x, overall, color="black", lw=1.2, ls="--", label="Cross-translation mean", alpha=0.6)
ax1.set_ylabel(f"Cosine distance from {REFERENCE}")
ax1.set_title(f"Translation divergence from {REFERENCE} across the canon", fontsize=12)
ax1.legend(fontsize=9, frameon=False)
ax1.grid(True, alpha=0.2)

# Bottom panel — each translation's deviation from the cross-translation mean
ax2.axhline(0, color="black", lw=0.8, alpha=0.5)
for t, vals in book_means.items():
    deviation = vals - overall
    ax2.fill_between(x, deviation, 0,
                     color=t_colors[t], alpha=0.35, label=t)
    ax2.plot(x, deviation, color=t_colors[t], lw=1.2)
ax2.set_ylabel("Deviation from mean")
ax2.set_xlabel("Book (Genesis → Revelation)")
ax2.legend(fontsize=8, frameon=False)
ax2.grid(True, alpha=0.2)

ax2.set_xticks(x)
ax2.set_xticklabels(book_order, rotation=90, fontsize=6)

# OT/NT divider
for a in [ax1, ax2]:
    a.axvline(38.5, color="grey", lw=1, ls=":", alpha=0.7)
    a.text(19, a.get_ylim()[0], "Old Testament", ha="center", fontsize=7,
           color="grey", va="bottom")
    a.text(52, a.get_ylim()[0], "New Testament", ha="center", fontsize=7,
           color="grey", va="bottom")

plt.tight_layout()
plt.savefig("04_translations_canon_trajectory.png", dpi=150)
plt.show()

# %% ── PLOT 4: Most divergent verses table ───────────────────────────────────
# Rank by variance of cosine similarity across all translation pairs
all_sims = np.vstack([cos_sims[t] for t in cos_sims]).T  # (N_verses, N_pairs)
variance = np.var(all_sims, axis=1)

top_n  = 20
top_idx = np.argsort(variance)[::-1][:top_n]

print(f"\n{'='*80}")
print(f"Top {top_n} most divergent verses (by cross-translation similarity variance)")
print(f"Reference: {REFERENCE}")
print(f"{'='*80}")

conn2 = get_conn()
for rank, idx in enumerate(top_idx, 1):
    row   = meta.iloc[idx]
    book, ch, vs = row["book"], int(row["chapter"]), int(row["verse"])
    print(f"\n#{rank}  {book} {ch}:{vs}  (variance={variance[idx]:.5f})")
    for t, corpus_name in TRANSLATIONS.items():
        result = conn2.execute(
            """
            SELECT p.text FROM passage p
            JOIN corpus c ON c.id = p.corpus_id
            WHERE c.name = ? AND p.book = ? AND p.section = ? AND p.unit_number = ?
            """,
            [corpus_name, book, str(ch), vs],
        ).fetchone()
        text = result[0] if result else "(missing)"
        print(f"  [{t}] {text}")

conn2.close()
