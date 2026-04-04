# %% [markdown]
# # GMM K Selection — BIC/AIC Sweep
#
# Fit a Gaussian Mixture Model for K = 2, 4, 6, ... 50 on UMAP-reduced embeddings.
# Plot BIC and AIC to find the elbow / minimum before running the full GMM analysis.
#
# Pipeline:
#   1. Load all embeddings (excluding SKIP_CORPORA)
#   2. UMAP reduce to 15 dims (fit on sacred+lit+historical, transform news in)
#   3. Sweep K, record BIC + AIC
#   4. Plot and print recommended K

# %% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from umap import UMAP

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA, TRADITION_GROUP

MODEL_NAME  = "all-mpnet-base-v2"
UMAP_DIMS   = 15
K_MIN, K_MAX, K_STEP = 2, 50, 2
N_INIT      = 3   # GMM restarts per K — reduces sensitivity to random init

# %% Load embeddings
conn = get_conn()
rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus, p.id AS passage_id, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["tradition", "corpus", "passage_id", "vector"])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df["group"] = df["tradition"].map(TRADITION_GROUP).fillna("Other")
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} passages")
print(df.groupby("group").size().to_string())

# %% UMAP reduction
# Fit on non-news to keep news from dominating the manifold, then transform all
vecs = np.stack(df["vector"].values).astype("float32")

is_news  = df["group"] == "News"
vecs_ref = vecs[~is_news]

print(f"\nFitting UMAP on {vecs_ref.shape[0]:,} non-news passages → {UMAP_DIMS} dims ...")
reducer = UMAP(
    n_components=UMAP_DIMS,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
reducer.fit(vecs_ref)

print("Transforming all passages ...")
X = reducer.transform(vecs).astype("float32")
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
    bics.append(gmm.bic(X))
    aics.append(gmm.aic(X))
    print(f"  K={k:3d}  BIC={gmm.bic(X):,.0f}  AIC={gmm.aic(X):,.0f}")

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

plt.suptitle("GMM Component Selection — BIC & AIC Sweep", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("21a_gmm_bic_sweep.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 21a_gmm_bic_sweep.png")

# %% Summary
best_bic_k = ks[int(np.argmin(bics))]
best_aic_k = ks[int(np.argmin(aics))]

print(f"\nBIC minimum at K = {best_bic_k}  (more conservative — prefer this)")
print(f"AIC minimum at K = {best_aic_k}  (more liberal)")
print("\nNote: if the curve has no clear elbow and keeps falling, try extending K_MAX.")
print("For topic interpretability, values near the BIC elbow are usually best.")
