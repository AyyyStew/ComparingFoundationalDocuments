# %% [markdown]
# # Sacred Tradition Similarity — Lift Analysis
#
# Computes both the top-k and random-pair similarity matrices (same methods as
# scripts 23 and 23b), then derives:
#
#   Difference = top-k minus random  → how much the best matches exceed the baseline
#   Ratio      = top-k / random      → relative lift, normalised for absolute level
#
# High diagonal lift = tradition has densely similar internal clusters (hot pockets).
# High off-diagonal lift = cross-tradition connection concentrated in a few good matches
#                          rather than spread broadly across the texts.
#
# Uses only the corpora available at script 23's point in the project.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from db.schema import get_conn

MODEL_NAME  = "all-mpnet-base-v2"
TOP_K       = 5
N_SAMPLE    = 5000
RANDOM_SEED = 42

CORPORA_AT_23 = {
    "Abrahamic": ["Bible — KJV (King James Version)"],
    "Dharmic":   ["Bhagavad Gita", "Yoga Sutras of Patanjali (Johnston)"],
    "Buddhist":  ["Dhammapada (Müller)"],
    "Taoist":    ["Dao De Jing (Linnell)"],
}

traditions = list(CORPORA_AT_23.keys())

# %% Load embeddings
conn = get_conn()
all_corpora  = [c for names in CORPORA_AT_23.values() for c in names]
placeholders = ", ".join("?" * len(all_corpora))

rows = conn.execute(f"""
    SELECT t.name AS tradition, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
      AND c.name IN ({placeholders})
    ORDER BY c.name, p.id
""", [MODEL_NAME] + all_corpora).fetchall()
conn.close()

df     = pd.DataFrame(rows, columns=["tradition", "vector"])
vecs   = np.stack(df["vector"].apply(np.array).values).astype("float32")
norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_n = vecs / np.clip(norms, 1e-9, None)

trad_idx = {t: np.where(df["tradition"].values == t)[0] for t in traditions}

print("Passages per tradition:")
print(df.groupby("tradition").size().to_string())

# %% Build top-k similarity matrix (method from script 23)
topk_mat = pd.DataFrame(index=traditions, columns=traditions, dtype=float)

for t_a in traditions:
    for t_b in traditions:
        idx_a = trad_idx[t_a]
        idx_b = trad_idx[t_b]
        V_a   = vecs_n[idx_a]
        V_b   = vecs_n[idx_b]
        sim   = V_a @ V_b.T

        if t_a == t_b:
            np.fill_diagonal(sim, -np.inf)
            k = min(TOP_K, sim.shape[1] - 1)
        else:
            k = min(TOP_K, sim.shape[1])

        top_k_vals = np.partition(sim, -k, axis=1)[:, -k:]
        topk_mat.loc[t_a, t_b] = float(top_k_vals.mean())

print("\nTop-k similarity matrix:")
print(topk_mat.round(3).to_string())

# %% Build random-pair similarity matrix (method from script 23b)
rng      = np.random.default_rng(RANDOM_SEED)
rand_mat = pd.DataFrame(index=traditions, columns=traditions, dtype=float)

for t_a in traditions:
    for t_b in traditions:
        idx_a = trad_idx[t_a]
        idx_b = trad_idx[t_b]

        si = rng.choice(idx_a, size=N_SAMPLE, replace=True)
        sj = rng.choice(idx_b, size=N_SAMPLE, replace=True)

        if t_a == t_b:
            same = si == sj
            while same.any():
                sj[same] = rng.choice(idx_b, size=same.sum(), replace=True)
                same = si == sj

        rand_mat.loc[t_a, t_b] = float((vecs_n[si] * vecs_n[sj]).sum(axis=1).mean())

print("\nRandom-pair similarity matrix:")
print(rand_mat.round(3).to_string())

# %% Derive difference and ratio
topk_vals = topk_mat.values.astype(float)
rand_vals = rand_mat.values.astype(float)

diff_vals  = topk_vals - rand_vals
ratio_vals = topk_vals / np.clip(rand_vals, 1e-9, None)

print("\nDifference (top-k minus random):")
print(pd.DataFrame(diff_vals, index=traditions, columns=traditions).round(3).to_string())

print("\nRatio (top-k / random):")
print(pd.DataFrame(ratio_vals, index=traditions, columns=traditions).round(3).to_string())

# %% Plot — 2 panels side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

panels = [
    (diff_vals,  "Difference  (top-k − random)",  "YlOrRd", ".3f"),
    (ratio_vals, "Ratio  (top-k ÷ random)",        "YlOrRd", ".2f"),
]

for ax, (vals, title, cmap, fmt) in zip(axes, panels):
    im = ax.imshow(vals, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(traditions)))
    ax.set_xticklabels(traditions, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(traditions)))
    ax.set_yticklabels(traditions, fontsize=10)
    for i in range(len(traditions)):
        for j in range(len(traditions)):
            ax.text(j, i, format(vals[i, j], fmt),
                    ha="center", va="center", fontsize=9,
                    fontweight="bold" if i == j else "normal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=11, fontweight="bold")

plt.suptitle("Similarity Lift — Top-k vs Random Pair (Sacred Traditions)",
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("23c_similarity_lift.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 23c_similarity_lift.png")
