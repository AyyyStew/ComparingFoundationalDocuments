# %% [markdown]
# # Concept Network — Full Corpus (Script 36)
#
# Script 25 built a balanced concept network for sacred texts only.
# This script extends it to the full corpus (all traditions except News),
# using the same pipeline:
#   - HDBSCAN clustering on UMAP-reduced embeddings
#   - Chapter-level co-occurrence edges
#   - Pyvis interactive network
#
# ## Balancing strategy
#
# Three corpora dominate by size: Bible KJV (31k), Srimad Bhagavatam (13k),
# Quran (6k). Everything else is under 2k passages.
# For each oversized corpus, we keep only the top-CAP passages by their
# max cosine similarity to any passage from any *other* corpus — i.e. the
# passages most relevant to the rest of the corpus.
# This preserves cross-corpus signal while preventing any one text from
# drowning out the others.

# %% Imports
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from pyvis.network import Network

from db.schema import get_conn
from analysis_utils import SKIP_CORPORA, STOP_WORDS

MODEL_NAME      = "all-mpnet-base-v2"
TARGET_K        = 35       # aim for ~35 clusters across the broader corpus
MIN_EDGE_WEIGHT = 2
CAP             = 1500     # max passages per corpus before balancing kicks in

TRADITION_COLORS = {
    "Abrahamic":  "#e63946",
    "Dharmic":    "#ffd166",
    "Buddhist":   "#06d6a0",
    "Taoist":     "#7209b7",
    "Norse":      "#118ab2",
    "Confucian":  "#f4845f",
    "Philosophy": "#f72585",
    "Scientific": "#b5e853",
    "Literature": "#48cae4",
    "Historical": "#cb997e",
}

SKIP_TRADITIONS = {"News"}

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
df_all = df_all[~df_all["tradition"].isin(SKIP_TRADITIONS)].reset_index(drop=True)
df_all["vector"] = df_all["vector"].apply(np.array)

print(f"Loaded {len(df_all):,} passages")
print(df_all.groupby(["tradition", "corpus"]).size().to_string())

# %% L2-normalise
vecs_all  = np.stack(df_all["vector"].values).astype("float32")
norms     = np.linalg.norm(vecs_all, axis=1, keepdims=True)
vecs_norm = vecs_all / np.clip(norms, 1e-9, None)

# %% Balance oversized corpora
# For each corpus with more than CAP passages:
#   compute each passage's max cosine similarity to any passage from any other corpus,
#   keep the top-CAP by that score.
corpus_sizes = df_all.groupby("corpus").size()
oversized    = corpus_sizes[corpus_sizes > CAP].index.tolist()
print(f"\nOversized corpora (>{CAP} passages): {oversized}")

kept_ids = set(df_all["passage_id"])  # start with all

for corp in oversized:
    corp_mask    = df_all["corpus"] == corp
    corp_idx     = df_all.index[corp_mask].tolist()
    other_idx    = df_all.index[~corp_mask].tolist()
    V_corp       = vecs_norm[corp_idx]     # (n_corp, 768)
    V_other      = vecs_norm[other_idx]    # (n_other, 768)

    # Max similarity of each corpus passage to any other-corpus passage
    chunk_size = 2000
    max_sim    = np.full(len(corp_idx), -np.inf, dtype="float32")
    for start in range(0, len(corp_idx), chunk_size):
        end = min(start + chunk_size, len(corp_idx))
        sim = V_corp[start:end] @ V_other.T   # (chunk, n_other)
        max_sim[start:end] = sim.max(axis=1)

    # Keep top-CAP
    top_local_idx = np.argpartition(max_sim, -CAP)[-CAP:]
    keep_pids     = {df_all.loc[corp_idx[i], "passage_id"] for i in top_local_idx}
    drop_pids     = {df_all.loc[corp_idx[i], "passage_id"] for i in range(len(corp_idx))
                     if i not in set(top_local_idx)}
    kept_ids -= drop_pids
    print(f"  {corp}: {len(corp_idx):,} → {len(keep_pids):,} kept")

df = df_all[df_all["passage_id"].isin(kept_ids)].reset_index(drop=True)
print(f"\nBalanced corpus: {len(df):,} passages")
print(df.groupby(["tradition", "corpus"]).size().to_string())

# %% Chapter key
def make_chapter_key(row):
    corp = row["corpus"]
    if corp == "Bible — KJV (King James Version)":
        chap = row["unit_label"].split(":")[0]
        return f"{row['book']} {chap}"
    elif corp == "Quran (Clear Quran Translation)":
        return f"Quran {row['book']}"
    elif corp == "Dao De Jing (Linnell)":
        return f"DDJ {int(row['unit_number'])}"
    elif corp == "Srimad Bhagavatam":
        # section = canto number
        return f"SB {row['book']}"
    else:
        return row["book"]

df["chapter_key"] = df.apply(make_chapter_key, axis=1)
print(f"\nUnique chapters: {df['chapter_key'].nunique():,}")

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

# %% Auto-select min_cluster_size to hit ~TARGET_K clusters
print(f"\nSweeping min_cluster_size to find ~{TARGET_K} clusters ...")

sweep_sizes = [20, 30, 50, 75, 100, 150, 200, 300]
best_mcs, best_k, best_labels = None, None, None

for mcs in sweep_sizes:
    labels    = HDBSCAN(min_cluster_size=mcs, min_samples=5,
                        metric="euclidean", prediction_data=False).fit_predict(X)
    k         = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    print(f"  min_cluster_size={mcs:4d}  →  {k:3d} clusters  ({noise_pct:.1f}% noise)")
    if best_k is None or abs(k - TARGET_K) < abs(best_k - TARGET_K):
        best_mcs, best_k, best_labels = mcs, k, labels

print(f"\nUsing min_cluster_size={best_mcs} → {best_k} clusters")
df["cluster"] = best_labels

# %% Label clusters with top distinctive words (c-TF-IDF)
tfidf = TfidfVectorizer(stop_words=STOP_WORDS, max_features=10_000,
                        ngram_range=(1, 2), min_df=2)
tfidf.fit(df["text"])
vocab = np.array(tfidf.get_feature_names_out())

cluster_ids    = sorted(c for c in df["cluster"].unique() if c != -1)
cluster_labels = {}

for c in cluster_ids:
    texts   = df[df["cluster"] == c]["text"].tolist()
    vec     = tfidf.transform([" ".join(texts)]).toarray()[0]
    top_idx = vec.argsort()[::-1][:4]
    cluster_labels[c] = " / ".join(vocab[top_idx])

print("\nCluster labels:")
for c, label in cluster_labels.items():
    n    = (df["cluster"] == c).sum()
    trad = df[df["cluster"] == c]["tradition"].value_counts().idxmax()
    print(f"  C{c:2d} ({n:4d} passages, dom={trad[:4]}): {label}")

# %% Chapter-level co-occurrence edges
chapter_clusters = (
    df[df["cluster"] != -1]
    .groupby("chapter_key")["cluster"]
    .apply(set)
)

edge_weights = defaultdict(int)
for clusters_in_chap in chapter_clusters:
    cluster_list = sorted(clusters_in_chap)
    for i in range(len(cluster_list)):
        for j in range(i + 1, len(cluster_list)):
            edge_weights[(cluster_list[i], cluster_list[j])] += 1

print(f"\nRaw edges: {len(edge_weights):,}")
filtered_edges = {k: v for k, v in edge_weights.items() if v >= MIN_EDGE_WEIGHT}
print(f"Edges after min_weight={MIN_EDGE_WEIGHT} filter: {len(filtered_edges):,}")

# %% Build NetworkX graph
G = nx.Graph()

for c in cluster_ids:
    sub          = df[df["cluster"] == c]
    trad_counts  = sub["tradition"].value_counts()
    dom_trad     = trad_counts.idxmax()
    dom_pct      = trad_counts.iloc[0] / trad_counts.sum()
    trad_str     = " | ".join(f"{t}: {v}" for t, v in trad_counts.items())

    c_vecs   = np.stack(sub["vector"].values).astype("float32")
    centroid = c_vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    c_vecs_n = c_vecs / (np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-9)

    # One exemplar per tradition — passage closest to cluster centroid
    trad_exemplar_lines = []
    for t in trad_counts.index:
        t_sub    = sub[sub["tradition"] == t]
        t_vecs_n = np.stack(t_sub["vector"].values).astype("float32")
        t_vecs_n = t_vecs_n / (np.linalg.norm(t_vecs_n, axis=1, keepdims=True) + 1e-9)
        best     = t_sub.iloc[(t_vecs_n @ centroid).argmax()]
        trad_exemplar_lines.append(
            f"[{t}] {best['corpus']} | {best['unit_label']}\n  {best['text'][:150]}"
        )
    exemplars_str = "\n\n".join(trad_exemplar_lines)

    trad_counts_dict = {t: int(trad_counts.get(t, 0)) for t in TRADITION_COLORS}

    G.add_node(
        int(c),
        label=cluster_labels[c],
        n_passages=len(sub),
        dom_tradition=dom_trad,
        dom_pct=float(dom_pct),
        trad_breakdown=trad_str,
        exemplar=exemplars_str,
        **{f"n_{t.lower()}": trad_counts_dict[t] for t in TRADITION_COLORS},
    )

for (c_a, c_b), weight in filtered_edges.items():
    if G.has_node(int(c_a)) and G.has_node(int(c_b)):
        G.add_edge(int(c_a), int(c_b), weight=weight)

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

between_cent = nx.betweenness_centrality(G, weight="weight")
print("\nTop 10 nodes by betweenness centrality:")
for c, bc in sorted(between_cent.items(), key=lambda x: -x[1])[:10]:
    print(f"  C{c} [{cluster_labels[c]}]  betweenness={bc:.3f}  degree={G.degree(c)}")

# %% Pyvis network
net = Network(
    height="900px",
    width="100%",
    bgcolor="#1a1a2e",
    font_color="white",
    notebook=False,
)

max_n = max(G.nodes[c]["n_passages"] for c in G.nodes)
min_n = min(G.nodes[c]["n_passages"] for c in G.nodes)

def scale_size(n, lo=15, hi=60):
    if max_n == min_n:
        return (lo + hi) / 2
    return lo + (hi - lo) * ((n - min_n) / (max_n - min_n)) ** 0.5

for c in G.nodes:
    nd      = G.nodes[c]
    color   = TRADITION_COLORS.get(nd["dom_tradition"], "#aaaaaa")
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
        size=scale_size(nd["n_passages"]),
        font={"size": 11, "color": "white"},
    )

for c_a, c_b, data in G.edges(data=True):
    net.add_edge(
        int(c_a), int(c_b),
        value=data["weight"],
        title=f"Co-occurs in {data['weight']} chapter(s)",
        color={"color": "#ffffff22", "highlight": "#ffffff99"},
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

out_file = "36_concept_network_full.html"
net.save_graph(out_file)
print(f"\nSaved: {out_file}")

# Save graph for potential graph-theory follow-up
with open("36_concept_graph.pkl", "wb") as f:
    pickle.dump(G, f)
print("Saved: 36_concept_graph.pkl")

# %% Community detection
print("\n=== Network communities (Louvain) ===")
try:
    import community as community_louvain
    partition   = community_louvain.best_partition(G)
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    for comm_id, nodes in sorted(communities.items()):
        print(f"\n  Community {comm_id} ({len(nodes)} nodes):")
        for n in nodes:
            print(f"    C{n} [{cluster_labels[n]}] — {G.nodes[n]['dom_tradition']}")
except ImportError:
    print("  (install python-louvain for community detection: pip install python-louvain)")
