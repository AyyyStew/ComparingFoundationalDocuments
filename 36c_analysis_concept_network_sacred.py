# %% [markdown]
# # Concept Network — Sacred Texts, Centroid Similarity (Script 36c)
#
# Same pipeline as script 36 (load → balance → UMAP → HDBSCAN → KeyBERT labels)
# but restricted to sacred traditions only.
#
# Edges are centroid similarity only (as in 36b) — no co-occurrence edges.
# Only multi-tradition clusters get edges.
#
# Color mode: set COLOR_BY = "tradition" or "corpus"

# %% Imports
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import torch
from collections import defaultdict
from hdbscan import HDBSCAN
from keybert import KeyBERT
from pyvis.network import Network
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from umap import UMAP

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA, STOP_WORDS

# %% Parameters
MODEL_NAME              = "all-mpnet-base-v2"
TARGET_K                = 25      # fewer clusters than full-corpus run
CAP                     = 1500    # max passages per corpus before balancing
MIN_CLUSTER_SIZE_SWEEP  = [20, 30, 50, 75, 100, 150]
CENTROID_SIM_THRESHOLD  = 0.7
CENTROID_MIN_TRADITIONS = 2

COLOR_BY = "tradition"   # "tradition" | "corpus"

SACRED_TRADITIONS = {"Abrahamic", "Dharmic", "Buddhist", "Taoist", "Norse", "Confucian"}

TRADITION_COLORS = {
    "Abrahamic":  "#e63946",
    "Dharmic":    "#ffd166",
    "Buddhist":   "#06d6a0",
    "Taoist":     "#7209b7",
    "Norse":      "#118ab2",
    "Confucian":  "#f4845f",
}

# %% Load
conn = get_conn()
rows = conn.execute("""
    SELECT t.name AS tradition, c.name AS corpus,
           p.id AS passage_id, p.book, p.unit_number, p.unit_label, p.text, e.vector
    FROM embedding e
    JOIN passage p ON e.passage_id = p.id
    JOIN corpus c  ON p.corpus_id = c.id
    JOIN corpus_tradition t ON c.tradition_id = t.id
    WHERE e.model_name = ?
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df_all = pd.DataFrame(rows, columns=[
    "tradition", "corpus", "passage_id", "book", "unit_number", "unit_label", "text", "vector"
])
df_all = df_all[~df_all["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df_all = df_all[df_all["tradition"].isin(SACRED_TRADITIONS)].reset_index(drop=True)
df_all["vector"] = df_all["vector"].apply(np.array)

print(f"Loaded {len(df_all):,} passages")
print(df_all.groupby(["tradition", "corpus"]).size().to_string())

# %% L2-normalise
vecs_all  = np.stack(df_all["vector"].values).astype("float32")
norms     = np.linalg.norm(vecs_all, axis=1, keepdims=True)
vecs_norm = vecs_all / np.clip(norms, 1e-9, None)

# %% Balance oversized corpora
corpus_sizes = df_all.groupby("corpus").size()
oversized    = corpus_sizes[corpus_sizes > CAP].index.tolist()
print(f"\nOversized corpora (>{CAP}): {oversized}")

kept_ids = set(df_all["passage_id"])
for corp in oversized:
    corp_mask = df_all["corpus"] == corp
    corp_idx  = df_all.index[corp_mask].tolist()
    other_idx = df_all.index[~corp_mask].tolist()
    V_corp    = vecs_norm[corp_idx]
    V_other   = vecs_norm[other_idx]

    chunk_size = 2000
    max_sim    = np.full(len(corp_idx), -np.inf, dtype="float32")
    for start in range(0, len(corp_idx), chunk_size):
        end = min(start + chunk_size, len(corp_idx))
        sim = V_corp[start:end] @ V_other.T
        max_sim[start:end] = sim.max(axis=1)

    top_local_idx = np.argpartition(max_sim, -CAP)[-CAP:]
    keep_pids     = {df_all.loc[corp_idx[i], "passage_id"] for i in top_local_idx}
    drop_pids     = {df_all.loc[corp_idx[i], "passage_id"] for i in range(len(corp_idx))
                     if i not in set(top_local_idx)}
    kept_ids -= drop_pids
    print(f"  {corp}: {len(corp_idx):,} → {len(keep_pids):,} kept")

df = df_all[df_all["passage_id"].isin(kept_ids)].reset_index(drop=True)
print(f"\nBalanced: {len(df):,} passages")
print(df.groupby(["tradition", "corpus"]).size().to_string())

# %% UMAP 10D
vecs = np.stack(df["vector"].values).astype("float32")
print("\nFitting UMAP (10D) ...")
reducer = UMAP(
    n_components=10,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
    low_memory=False,
)
X = reducer.fit_transform(vecs).astype("float32")

# %% HDBSCAN sweep
print(f"\nSweeping min_cluster_size to find ~{TARGET_K} clusters ...")
best_mcs, best_k, best_labels = None, None, None
for mcs in MIN_CLUSTER_SIZE_SWEEP:
    labels    = HDBSCAN(min_cluster_size=mcs, min_samples=5,
                        metric="euclidean", prediction_data=False).fit_predict(X)
    k         = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    print(f"  min_cluster_size={mcs:4d}  →  {k:3d} clusters  ({noise_pct:.1f}% noise)")
    if best_k is None or abs(k - TARGET_K) < abs(best_k - TARGET_K):
        best_mcs, best_k, best_labels = mcs, k, labels

print(f"\nUsing min_cluster_size={best_mcs} → {best_k} clusters")
df["cluster"] = best_labels

# %% KeyBERT labels
kw_model = KeyBERT(model=SentenceTransformer(MODEL_NAME, device="cuda"))
cluster_ids    = sorted(c for c in df["cluster"].unique() if c != -1)
cluster_labels = {}

for c in tqdm(cluster_ids, desc="KeyBERT labeling"):
    texts = df[df["cluster"] == c]["text"].tolist()
    if len(texts) > 300:
        texts = pd.Series(texts).sample(300, random_state=42).tolist()
    doc = " ".join(texts)
    kws = kw_model.extract_keywords(
        doc,
        keyphrase_ngram_range=(1, 2),
        stop_words=list(STOP_WORDS),
        top_n=4,
        use_mmr=True,
        diversity=0.5,
    )
    cluster_labels[c] = " / ".join(kw for kw, _ in kws)

del kw_model
torch.cuda.empty_cache()

print("\nCluster labels:")
for c, label in cluster_labels.items():
    n    = (df["cluster"] == c).sum()
    trad = df[df["cluster"] == c]["tradition"].value_counts().idxmax()
    print(f"  C{c:2d} ({n:4d} passages, dom={trad[:4]}): {label}")

# %% Corpus color map (for COLOR_BY = "corpus")
all_corpora   = sorted(df["corpus"].unique())
corpus_colors = {}
import colorsys
for i, corp in enumerate(all_corpora):
    h = i / max(len(all_corpora), 1)
    r, g, b = colorsys.hsv_to_rgb(h, 0.7, 0.9)
    corpus_colors[corp] = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))

# %% Compute centroids and build NetworkX graph
centroids = {}
for c in cluster_ids:
    sub      = df[df["cluster"] == c]
    vecs_c   = np.stack(sub["vector"].values).astype("float32")
    centroid = vecs_c.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    centroids[c] = centroid

G = nx.Graph()
for c in cluster_ids:
    sub         = df[df["cluster"] == c]
    trad_counts = sub["tradition"].value_counts()
    dom_trad    = trad_counts.idxmax()
    dom_pct     = trad_counts.iloc[0] / trad_counts.sum()
    trad_str    = " | ".join(f"{t}: {v}" for t, v in trad_counts.items())

    # Dominant corpus
    corp_counts = sub["corpus"].value_counts()
    dom_corpus  = corp_counts.idxmax()

    centroid = centroids[c]
    c_vecs_n = np.stack(sub["vector"].values).astype("float32")
    c_vecs_n = c_vecs_n / (np.linalg.norm(c_vecs_n, axis=1, keepdims=True) + 1e-9)

    trad_exemplar_lines = []
    for t in trad_counts.index:
        t_sub    = sub[sub["tradition"] == t]
        t_vecs_n = np.stack(t_sub["vector"].values).astype("float32")
        t_vecs_n = t_vecs_n / (np.linalg.norm(t_vecs_n, axis=1, keepdims=True) + 1e-9)
        best     = t_sub.iloc[(t_vecs_n @ centroid).argmax()]
        trad_exemplar_lines.append(
            f"[{t}] {best['corpus']} | {best['unit_label']}\n  {best['text'][:150]}"
        )

    trad_counts_dict = {t: int(trad_counts.get(t, 0)) for t in TRADITION_COLORS}

    G.add_node(
        int(c),
        label=cluster_labels[c],
        n_passages=len(sub),
        dom_tradition=dom_trad,
        dom_corpus=dom_corpus,
        dom_pct=float(dom_pct),
        trad_breakdown=trad_str,
        exemplar="\n\n".join(trad_exemplar_lines),
        **{f"n_{t.lower()}": trad_counts_dict[t] for t in TRADITION_COLORS},
    )

# %% Centroid similarity edges (multi-tradition clusters only)
multi_trad = [
    c for c in cluster_ids
    if df[df["cluster"] == c]["tradition"].nunique() >= CENTROID_MIN_TRADITIONS
]
print(f"\nMulti-tradition clusters: {len(multi_trad)} / {len(cluster_ids)}")

centroid_mat = np.stack([centroids[c] for c in multi_trad])
sim_matrix   = centroid_mat @ centroid_mat.T

edges = []
for i in range(len(multi_trad)):
    for j in range(i + 1, len(multi_trad)):
        sim = float(sim_matrix[i, j])
        if sim >= CENTROID_SIM_THRESHOLD:
            edges.append((multi_trad[i], multi_trad[j], sim))

edges.sort(key=lambda x: -x[2])
print(f"Centroid similarity edges (sim >= {CENTROID_SIM_THRESHOLD}): {len(edges)}")
print("\nTop 20 edges:")
for a, b, sim in edges[:20]:
    print(f"  {sim:.3f}  C{a} [{cluster_labels[a]}]  ↔  C{b} [{cluster_labels[b]}]")

for a, b, sim in edges:
    G.add_edge(a, b, weight=sim)

# %% Save
with open("36c_concept_graph.pkl", "wb") as f:
    pickle.dump(G, f)
with open("36c_centroids.pkl", "wb") as f:
    pickle.dump(centroids, f)
print("\nSaved: 36c_concept_graph.pkl, 36c_centroids.pkl")

# %% Pyvis
def scale_size(n, lo=15, hi=60, min_n=None, max_n=None):
    if max_n == min_n:
        return (lo + hi) / 2
    return lo + (hi - lo) * ((n - min_n) / (max_n - min_n)) ** 0.5

all_n = [G.nodes[c]["n_passages"] for c in G.nodes]
min_n, max_n = min(all_n), max(all_n)

net = Network(height="900px", width="100%", bgcolor="#1a1a2e", font_color="white", notebook=False)

for c in G.nodes:
    nd = G.nodes[c]
    if COLOR_BY == "corpus":
        color = corpus_colors.get(nd["dom_corpus"], "#aaaaaa")
    else:
        color = TRADITION_COLORS.get(nd["dom_tradition"], "#aaaaaa")

    tooltip = (
        f"Cluster {c}: {nd['label']}\n"
        f"Passages: {nd['n_passages']}\n"
        f"Traditions: {nd['trad_breakdown']}\n"
        f"Dominant: {nd['dom_tradition']} ({nd['dom_pct']:.0%})\n\n"
        f"Exemplar:\n{nd['exemplar']}"
    )
    net.add_node(
        c,
        label=nd["label"],
        title=tooltip,
        color=color,
        size=scale_size(nd["n_passages"], min_n=min_n, max_n=max_n),
        font={"size": 11, "color": "white"},
    )

for a, b, data in G.edges(data=True):
    sim = data["weight"]
    net.add_edge(
        int(a), int(b),
        value=sim,
        title=f"Centroid similarity: {sim:.3f}",
        color={"color": "#ffd70066", "highlight": "#ffd700cc"},
    )

net.set_options("""
{
  "physics": {
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -150,
      "centralGravity": 0.005,
      "springLength": 220,
      "springConstant": 0.03,
      "damping": 0.9
    },
    "maxVelocity": 15,
    "stabilization": {"iterations": 300, "updateInterval": 50}
  },
  "edges": {
    "smooth": {"type": "continuous"}
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 100
  }
}
""")

out_file = "36c_concept_network_sacred.html"
net.save_graph(out_file)
print(f"Saved: {out_file}")
