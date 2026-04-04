# %% [markdown]
# # Sacred Tradition Similarity — Random-Pair Baseline
#
# Script 23 computed the similarity heatmap using top-k nearest neighbours,
# which inflates the numbers (you're always comparing best matches).
# This script uses random-pair sampling instead — the same method as the
# distribution plot in script 07 — so you can see what the traditions look
# like when they're *not* trying their best to agree.
#
# Uses only the corpora available at script 23's point in the project:
#   Abrahamic : Bible — KJV (King James Version)
#   Dharmic   : Bhagavad Gita, Yoga Sutras of Patanjali (Johnston)
#   Buddhist  : Dhammapada (Müller)
#   Taoist    : Dao De Jing (Linnell)

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from db.schema import get_conn

MODEL_NAME  = "all-mpnet-base-v2"
N_SAMPLE    = 5000   # random pairs per tradition combination
RANDOM_SEED = 42

CORPORA_AT_23 = {
    "Abrahamic": ["Bible — KJV (King James Version)"],
    "Dharmic":   ["Bhagavad Gita", "Yoga Sutras of Patanjali (Johnston)"],
    "Buddhist":  ["Dhammapada (Müller)"],
    "Taoist":    ["Dao De Jing (Linnell)"],
}

TRADITION_COLORS = {
    "Abrahamic": "#e05c5c",
    "Dharmic":   "#f0a500",
    "Buddhist":  "#44bb99",
    "Taoist":    "#9b59b6",
}

# %% Load embeddings — restricted to original corpora
conn = get_conn()

all_corpora = [c for names in CORPORA_AT_23.values() for c in names]
placeholders = ", ".join("?" * len(all_corpora))

rows = conn.execute(f"""
    SELECT t.name AS tradition, c.name AS corpus, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
      AND c.name IN ({placeholders})
    ORDER BY c.name, p.id
""", [MODEL_NAME] + all_corpora).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "vector"])

vecs   = np.stack(df["vector"].apply(np.array).values).astype("float32")
norms  = np.linalg.norm(vecs, axis=1, keepdims=True)
vecs_n = vecs / np.clip(norms, 1e-9, None)

print("Passages per tradition:")
print(df.groupby("tradition").size().to_string())

# %% Compute random-pair mean cosine similarity for every tradition combination
rng        = np.random.default_rng(RANDOM_SEED)
traditions = list(CORPORA_AT_23.keys())
trad_idx   = {t: np.where(df["tradition"].values == t)[0] for t in traditions}

def random_pair_sim(idx_a: np.ndarray, idx_b: np.ndarray, n: int, rng) -> float:
    """Mean cosine similarity of n random (i, j) pairs drawn from idx_a × idx_b."""
    n = min(n, len(idx_a) * len(idx_b))
    si = rng.choice(idx_a, size=n, replace=True)
    sj = rng.choice(idx_b, size=n, replace=True)
    return float((vecs_n[si] * vecs_n[sj]).sum(axis=1).mean())

sim_matrix = pd.DataFrame(index=traditions, columns=traditions, dtype=float)

for t_a in traditions:
    for t_b in traditions:
        if t_a == t_b:
            idx = trad_idx[t_a]
            # Within-tradition: draw pairs where i != j
            i_vals = rng.choice(idx, size=N_SAMPLE, replace=True)
            j_vals = rng.choice(idx, size=N_SAMPLE, replace=True)
            # re-draw any self-pairs
            same = i_vals == j_vals
            while same.any():
                j_vals[same] = rng.choice(idx, size=same.sum(), replace=True)
                same = i_vals == j_vals
            sim_matrix.loc[t_a, t_b] = float((vecs_n[i_vals] * vecs_n[j_vals]).sum(axis=1).mean())
        else:
            sim_matrix.loc[t_a, t_b] = random_pair_sim(
                trad_idx[t_a], trad_idx[t_b], N_SAMPLE, rng
            )

print("\nRandom-pair mean cosine similarity:")
print(sim_matrix.round(3).to_string())

# %% Plot — same layout as 23 for direct comparison
fig, ax = plt.subplots(figsize=(7, 6))
vals = sim_matrix.values.astype(float)
im   = ax.imshow(vals, cmap="YlOrRd", aspect="auto")

ax.set_xticks(range(len(traditions)))
ax.set_xticklabels(traditions, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(traditions)))
ax.set_yticklabels(traditions, fontsize=10)

for i in range(len(traditions)):
    for j in range(len(traditions)):
        ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold" if i == j else "normal")

plt.colorbar(im, ax=ax, label="Mean cosine similarity")
ax.set_title("Sacred Tradition Similarity — Random-Pair Baseline\n"
             f"(diagonal = within-tradition, {N_SAMPLE:,} random pairs each)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("23b_tradition_similarity_random.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 23b_tradition_similarity_random.png")
