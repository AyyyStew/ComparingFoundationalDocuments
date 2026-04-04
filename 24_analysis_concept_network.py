# %% [markdown]
# # Concept Network — Sacred Texts
#
# Each passage is assigned to a concept cluster (HDBSCAN on UMAP-reduced embeddings).
# Two concept nodes are connected if they co-occur in the same chapter.
# Edge weight = number of chapters across the corpus containing both concepts.
#
# Output: interactive pyvis HTML network

# %% Imports
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

MODEL_NAME     = "all-mpnet-base-v2"
TARGET_K       = 25   # aim for approximately this many clusters
MIN_EDGE_WEIGHT = 2   # minimum chapter co-occurrences to draw an edge

TRADITION_COLORS = {
    "Abrahamic": "#e05c5c",
    "Dharmic":   "#f0a500",
    "Buddhist":  "#44bb99",
    "Taoist":    "#9b59b6",
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
    AND t.name IN ('Abrahamic', 'Dharmic', 'Buddhist', 'Taoist')
    ORDER BY c.name, p.id
""", [MODEL_NAME]).fetchall()
conn.close()

df = pd.DataFrame(rows, columns=[
    "tradition", "corpus", "passage_id", "book", "unit_number", "unit_label", "text", "vector"
])
df = df[~df["corpus"].isin(SKIP_CORPORA)].reset_index(drop=True)
df["vector"] = df["vector"].apply(np.array)

print(f"Loaded {len(df):,} sacred passages")
print(df.groupby(["tradition", "corpus"]).size().to_string())

# %% Chapter key — the unit of co-occurrence
# Bible:      book + chapter part of unit_label  (e.g. "Genesis 1")
# Dao De Jing: unit_number (81 chapters, single book)
# Others:     book IS the chapter

def make_chapter_key(row):
    if row["corpus"] == "Bible — KJV (King James Version)":
        chap = row["unit_label"].split(":")[0]   # "3:16" → "3"
        return f"{row['book']} {chap}"
    elif row["corpus"] == "Dao De Jing (Linnell)":
        return f"DDJ {int(row['unit_number'])}"
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

# %% Auto-select min_cluster_size to land near TARGET_K clusters
print(f"\nSweeping min_cluster_size to find ~{TARGET_K} clusters ...")

sweep_sizes = [20, 30, 50, 75, 100, 150, 200, 300]
best_mcs, best_k, best_labels = None, None, None

for mcs in sweep_sizes:
    labels = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=5,
        metric="euclidean",
        prediction_data=False,
    ).fit_predict(X)
    k = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    print(f"  min_cluster_size={mcs:4d}  →  {k:3d} clusters  ({noise_pct:.1f}% noise)")
    if best_k is None or abs(k - TARGET_K) < abs(best_k - TARGET_K):
        best_mcs, best_k, best_labels = mcs, k, labels

print(f"\nUsing min_cluster_size={best_mcs} → {best_k} clusters")
df["cluster"] = best_labels

# %% Label each cluster with top 3 distinctive words (c-TF-IDF)
tfidf = TfidfVectorizer(stop_words=STOP_WORDS, max_features=10_000,
                        ngram_range=(1, 2), min_df=2)
tfidf.fit(df["text"])
vocab = np.array(tfidf.get_feature_names_out())

cluster_ids = sorted(c for c in df["cluster"].unique() if c != -1)
cluster_labels = {}

for c in cluster_ids:
    texts = df[df["cluster"] == c]["text"].tolist()
    joined = " ".join(texts)
    vec = tfidf.transform([joined]).toarray()[0]
    top_idx = vec.argsort()[::-1][:4]
    cluster_labels[c] = " / ".join(vocab[top_idx])

print("\nCluster labels:")
for c, label in cluster_labels.items():
    n = (df["cluster"] == c).sum()
    trad = df[df["cluster"] == c]["tradition"].value_counts().idxmax()
    print(f"  C{c:2d} ({n:4d} passages, dom={trad[:4]}): {label}")

# %% Chapter-level co-occurrence edges
# For each chapter, collect the set of clusters present.
# For each pair of clusters in that chapter, add 1 to their edge weight.

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
    sub      = df[df["cluster"] == c]
    n_pass   = len(sub)
    trad_counts = sub["tradition"].value_counts()
    dom_trad    = trad_counts.idxmax()
    dom_pct     = trad_counts.iloc[0] / trad_counts.sum()

    # Tradition breakdown for tooltip
    trad_str = " | ".join(f"{t}: {v}" for t, v in trad_counts.items())

    # Exemplar passage (closest to cluster centroid)
    c_vecs   = np.stack(sub["vector"].values)
    centroid = c_vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    c_vecs_n = c_vecs / (np.linalg.norm(c_vecs, axis=1, keepdims=True) + 1e-9)
    exemplar_idx = (c_vecs_n @ centroid).argmax()
    exemplar = sub.iloc[exemplar_idx]

    G.add_node(
        c,
        label=cluster_labels[c],
        n_passages=n_pass,
        dom_tradition=dom_trad,
        dom_pct=float(dom_pct),
        trad_breakdown=trad_str,
        exemplar=f"{exemplar['corpus']} | {exemplar['unit_label']}\n{exemplar['text'][:200]}",
    )

for (c_a, c_b), weight in filtered_edges.items():
    if G.has_node(c_a) and G.has_node(c_b):
        G.add_edge(c_a, c_b, weight=weight)

print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Centrality
degree_cent = nx.degree_centrality(G)
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

# Scale node sizes: sqrt of passage count, mapped to 15–60
max_n = max(G.nodes[c]["n_passages"] for c in G.nodes)
min_n = min(G.nodes[c]["n_passages"] for c in G.nodes)

def scale_size(n, lo=15, hi=60):
    if max_n == min_n:
        return (lo + hi) / 2
    return lo + (hi - lo) * ((n - min_n) / (max_n - min_n)) ** 0.5

# Scale edge widths: 1–10
max_w = max(d["weight"] for _, _, d in G.edges(data=True)) if G.edges else 1

for c in G.nodes:
    nd       = G.nodes[c]
    color    = TRADITION_COLORS.get(nd["dom_tradition"], "#aaaaaa")
    size     = scale_size(nd["n_passages"])
    tooltip  = (
        f"Cluster {c}: {nd['label']}\n"
        f"Passages: {nd['n_passages']}\n"
        f"Traditions: {nd['trad_breakdown']}\n"
        f"Dominant: {nd['dom_tradition']} ({nd['dom_pct']:.0%})\n\n"
        f"Exemplar:\n{nd['exemplar']}"
    )
    net.add_node(
        int(c),
        label=nd["label"],
        title=tooltip,
        color=color,
        size=size,
        font={"size": 11, "color": "white"},
    )

for c_a, c_b, data in G.edges(data=True):
    w = data["weight"]
    net.add_edge(
        int(c_a), int(c_b),
        value=w,
        title=f"Co-occurs in {w} chapter(s)",
        color={"color": "#ffffff22", "highlight": "#ffffff99"},
    )

net.set_options("""
{
  "physics": {
    "solver": "forceAtlas2Based",
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "centralGravity": 0.01,
      "springLength": 120,
      "springConstant": 0.05,
      "damping": 0.4
    },
    "maxVelocity": 50,
    "stabilization": {"iterations": 200}
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

out_file = "24_concept_network.html"
net.save_graph(out_file)
print(f"\nSaved: {out_file}")

# %% Print community structure
print("\n=== Network communities (Louvain) ===")
try:
    import community as community_louvain
    partition = community_louvain.best_partition(G)
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    for comm_id, nodes in sorted(communities.items()):
        labels = [cluster_labels[n] for n in nodes]
        trads  = [G.nodes[n]["dom_tradition"] for n in nodes]
        print(f"\n  Community {comm_id} ({len(nodes)} nodes):")
        for n, l, t in zip(nodes, labels, trads):
            print(f"    C{n} [{l}] — {t}")
except ImportError:
    print("  (install python-louvain for community detection)")
