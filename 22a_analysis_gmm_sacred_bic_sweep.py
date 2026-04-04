# %% [markdown]
# # GMM K Selection (Sacred Texts Only) — BIC/AIC Sweep
#
# Same pipeline as 21a but restricted to sacred traditions only.
# Coloring will be per-tradition (Abrahamic / Dharmic / Buddhist / Taoist)
# rather than per-group, since all passages are sacred.

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from umap import UMAP

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA, SACRED_TRADITIONS

MODEL_NAME = "all-mpnet-base-v2"
UMAP_DIMS  = 15
K_MIN, K_MAX, K_STEP = 2, 50, 2
N_INIT     = 3

# %% Load — sacred traditions only
conn = get_conn()
rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus, p.id AS passage_id, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    AND t.name IN ('Abrahamic', 'Dharmic', 'Buddhist', 'Taoist')
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "passage_id", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} sacred passages")
print(df.groupby(["tradition", "corpus"]).size().to_string())

# %% UMAP 15D
vecs = np.stack(df["vector"].values).astype("float32")

print(f"\nFitting UMAP ({UMAP_DIMS}D) on {len(vecs):,} passages ...")
reducer = UMAP(
    n_components=UMAP_DIMS,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
X = reducer.fit_transform(vecs).astype("float32")
print(f"Reduced shape: {X.shape}")

# %% BIC / AIC sweep
ks   = list(range(K_MIN, K_MAX + 1, K_STEP))
bics = []
aics = []

print(f"\nSweeping K = {ks[0]} … {ks[-1]} (n_init={N_INIT} each) ...")
for k in ks:
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",
        n_init=N_INIT,
        random_state=42,
    )
    gmm.fit(X)
    b, a = gmm.bic(X), gmm.aic(X)
    bics.append(b)
    aics.append(a)
    print(f"  K={k:3d}  BIC={b:,.0f}  AIC={a:,.0f}")

# %% Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, scores, label, color in [
    (axes[0], bics, "BIC", "#e05c5c"),
    (axes[1], aics, "AIC", "#4a90d9"),
]:
    ax.plot(ks, scores, marker="o", markersize=4, color=color, linewidth=1.5)
    best_k = ks[int(np.argmin(scores))]
    ax.axvline(best_k, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("K (number of components)")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs K  (min at K={best_k})", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)

plt.suptitle("GMM Component Selection — Sacred Texts Only", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("22a_gmm_sacred_bic_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 22a_gmm_sacred_bic_sweep.png")

# %% Marginal gains table — helps pick elbow manually
print("\nMarginal BIC improvement per step:")
for i in range(1, len(ks)):
    delta = bics[i] - bics[i-1]   # negative = still improving
    print(f"  K={ks[i-1]:2d}→{ks[i]:2d}  Δ={delta:>10,.0f}")

best_bic_k = ks[int(np.argmin(bics))]
best_aic_k = ks[int(np.argmin(aics))]
print(f"\nBIC minimum at K = {best_bic_k}")
print(f"AIC minimum at K = {best_aic_k}")
